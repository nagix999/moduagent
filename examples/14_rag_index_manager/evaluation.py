"""Content-free retrieval and active-index quality checks.

The evaluator keeps private queries at the embedding boundary.  Reports contain
only opaque case/source identifiers and aggregate metrics, making them suitable
for CI artifacts and production telemetry.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import Any

from .backends import TextEmbedder
from .models import RAGIndexError
from .stores import SearchHit, VectorStore


MAX_EVALUATION_CASES = 1_000
MAX_EVALUATION_QUERY_CHARS = 4_096
MAX_EVALUATION_TAGS = 16
MAX_SEARCH_CONCURRENCY = 32
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SOURCE_ID = re.compile(r"^src_[A-Za-z0-9][A-Za-z0-9_.-]{0,123}$")


class RetrievalEvaluationError(RAGIndexError):
    """An evaluation input or backend result violated the safe contract."""


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One private query with relevant and explicitly forbidden source IDs."""

    case_id: str
    query: str
    expected_source_ids: tuple[str, ...]
    forbidden_source_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or _CASE_ID.fullmatch(self.case_id) is None
        ):
            raise ValueError("case_id must be a safe bounded identifier")
        if (
            not isinstance(self.query, str)
            or not self.query.strip()
            or len(self.query) > MAX_EVALUATION_QUERY_CHARS
            or any(
                ord(character) < 32 and character not in "\t\n\r"
                for character in self.query
            )
        ):
            raise ValueError("evaluation query must be non-empty and bounded")
        expected = _source_ids(self.expected_source_ids, "expected_source_ids")
        if not expected:
            raise ValueError("expected_source_ids must not be empty")
        forbidden = _source_ids(self.forbidden_source_ids, "forbidden_source_ids")
        if set(expected) & set(forbidden):
            raise ValueError("expected and forbidden source IDs must be disjoint")
        tags = tuple(self.tags)
        if (
            len(tags) > MAX_EVALUATION_TAGS
            or len(set(tags)) != len(tags)
            or any(
                not isinstance(tag, str) or _CASE_ID.fullmatch(tag) is None
                for tag in tags
            )
        ):
            raise ValueError("tags must be unique safe bounded identifiers")
        object.__setattr__(self, "query", self.query.strip())
        object.__setattr__(self, "expected_source_ids", expected)
        object.__setattr__(self, "forbidden_source_ids", forbidden)
        object.__setattr__(self, "tags", tags)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationThresholds:
    """Minimum release gate for one fixed evaluation set."""

    min_hit_rate_at_1: float = 0.0
    min_hit_rate_at_k: float = 0.0
    min_mean_reciprocal_rank: float = 0.0
    min_mean_recall_at_k: float = 0.0
    min_mean_average_precision: float = 0.0
    max_forbidden_case_rate: float = 1.0
    max_forbidden_at_1_case_rate: float = 1.0
    min_slice_hit_rate_at_1: float = 0.0
    min_slice_hit_rate_at_k: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "min_hit_rate_at_1",
            "min_hit_rate_at_k",
            "min_mean_reciprocal_rank",
            "min_mean_recall_at_k",
            "min_mean_average_precision",
            "max_forbidden_case_rate",
            "max_forbidden_at_1_case_rate",
            "min_slice_hit_rate_at_1",
            "min_slice_hit_rate_at_k",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a finite number between zero and one")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    case_id: str
    expected_source_ids: tuple[str, ...]
    retrieved_source_ids: tuple[str, ...]
    hit: bool
    reciprocal_rank: float
    hit_at_1: bool = False
    recall_at_k: float = 0.0
    average_precision: float = 0.0
    expected_ranks: tuple[int, ...] = ()
    forbidden_hit_count: int = 0
    forbidden_at_1: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalSliceMetrics:
    tag: str
    case_count: int
    hit_rate_at_1: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    mean_average_precision: float
    forbidden_case_rate: float
    forbidden_at_1_case_rate: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    generation_id: str
    top_k: int
    case_count: int
    hit_count: int
    hit_rate: float
    mean_reciprocal_rank: float
    results: tuple[RetrievalCaseResult, ...]
    candidate_limit: int = 0
    hit_at_1_count: int = 0
    hit_rate_at_1: float = 0.0
    mean_recall_at_k: float = 0.0
    mean_average_precision: float = 0.0
    forbidden_hit_count: int = 0
    forbidden_case_rate: float = 0.0
    forbidden_at_1_count: int = 0
    forbidden_at_1_case_rate: float = 0.0
    slices: tuple[RetrievalSliceMetrics, ...] = ()
    passed: bool = True
    violations: tuple[str, ...] = ()
    retrieval_mode: str = "dense"
    duration_seconds: float = 0.0

    def as_dict(self, *, include_results: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generation_id": self.generation_id,
            "retrieval_mode": self.retrieval_mode,
            "case_count": self.case_count,
            "top_k": self.top_k,
            "candidate_limit": self.candidate_limit,
            "duration_seconds": self.duration_seconds,
            "queries_per_second": (
                self.case_count / self.duration_seconds
                if self.duration_seconds > 0
                else 0.0
            ),
            "hit_rate_at_1": self.hit_rate_at_1,
            "hit_rate_at_k": self.hit_rate,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_average_precision": self.mean_average_precision,
            "forbidden_case_rate": self.forbidden_case_rate,
            "forbidden_at_1_case_rate": self.forbidden_at_1_case_rate,
            "passed": self.passed,
            "violations": list(self.violations),
            "slices": [
                {
                    "tag": item.tag,
                    "case_count": item.case_count,
                    "hit_rate_at_1": item.hit_rate_at_1,
                    "hit_rate_at_k": item.hit_rate_at_k,
                    "mean_reciprocal_rank": item.mean_reciprocal_rank,
                    "mean_recall_at_k": item.mean_recall_at_k,
                    "mean_average_precision": item.mean_average_precision,
                    "forbidden_case_rate": item.forbidden_case_rate,
                    "forbidden_at_1_case_rate": item.forbidden_at_1_case_rate,
                }
                for item in self.slices
            ],
        }
        if include_results:
            payload["results"] = [
                {
                    "case_id": item.case_id,
                    "expected_source_ids": list(item.expected_source_ids),
                    "retrieved_source_ids": list(item.retrieved_source_ids),
                    "hit_at_1": item.hit_at_1,
                    "hit_at_k": item.hit,
                    "reciprocal_rank": item.reciprocal_rank,
                    "recall_at_k": item.recall_at_k,
                    "average_precision": item.average_precision,
                    "expected_ranks": list(item.expected_ranks),
                    "forbidden_hit_count": item.forbidden_hit_count,
                    "forbidden_at_1": item.forbidden_at_1,
                }
                for item in self.results
            ]
        return payload


async def evaluate_retrieval(
    embedder: TextEmbedder,
    vector_store: VectorStore,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    top_k: int = 5,
    candidate_limit: int | None = None,
    search_concurrency: int = 8,
    thresholds: RetrievalEvaluationThresholds | None = None,
    use_hybrid: bool = True,
) -> RetrievalEvaluationReport:
    """Evaluate source-level ranking while keeping query and content private.

    Vector stores return chunks, but release decisions are normally made at the
    source-document level.  A larger candidate pool is therefore retrieved and
    duplicate source IDs are collapsed before computing metrics.
    """

    started_at = time.monotonic()
    if not isinstance(cases, tuple) or not cases or len(cases) > MAX_EVALUATION_CASES:
        raise ValueError("cases must be a non-empty bounded tuple")
    if any(not isinstance(case, RetrievalEvaluationCase) for case in cases):
        raise TypeError("cases must contain RetrievalEvaluationCase values")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("evaluation case IDs must be unique")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between one and 100")
    resolved_candidate_limit = (
        min(100, max(top_k, top_k * 10)) if candidate_limit is None else candidate_limit
    )
    if (
        type(resolved_candidate_limit) is not int
        or not top_k <= resolved_candidate_limit <= 100
    ):
        raise ValueError("candidate_limit must be between top_k and 100")
    if (
        type(search_concurrency) is not int
        or not 1 <= search_concurrency <= MAX_SEARCH_CONCURRENCY
    ):
        raise ValueError(
            f"search_concurrency must be between one and {MAX_SEARCH_CONCURRENCY}"
        )
    if type(use_hybrid) is not bool:
        raise TypeError("use_hybrid must be a bool")
    resolved_thresholds = thresholds or RetrievalEvaluationThresholds()
    if not isinstance(resolved_thresholds, RetrievalEvaluationThresholds):
        raise TypeError("thresholds must be RetrievalEvaluationThresholds or None")

    generation_id = await vector_store.active_generation_id()
    if generation_id is None:
        raise RetrievalEvaluationError("no published vector generation is available")
    batch = await embedder.embed(tuple(case.query for case in cases))
    if len(batch.vectors) != len(cases):
        raise RetrievalEvaluationError("embedding response count does not match cases")

    semaphore = asyncio.Semaphore(search_concurrency)
    hybrid_search = getattr(vector_store, "hybrid_search", None)
    use_hybrid_backend = use_hybrid and callable(hybrid_search)

    async def one(
        case: RetrievalEvaluationCase,
        vector: tuple[float, ...],
    ) -> RetrievalCaseResult:
        async with semaphore:
            if use_hybrid_backend:
                hits = await hybrid_search(
                    case.query,
                    vector,
                    limit=resolved_candidate_limit,
                )
            else:
                hits = await vector_store.search(vector, limit=resolved_candidate_limit)
        return _case_result(case, hits, top_k=top_k)

    results = tuple(
        await asyncio.gather(
            *(one(case, vector) for case, vector in zip(cases, batch.vectors))
        )
    )
    final_generation = await vector_store.active_generation_id()
    if final_generation != generation_id:
        raise RetrievalEvaluationError("active generation changed during evaluation")

    metrics = _aggregate(results)
    slices = tuple(
        _slice(
            tag,
            tuple(result for case, result in zip(cases, results) if tag in case.tags),
        )
        for tag in sorted({tag for case in cases for tag in case.tags})
    )
    violations = _threshold_violations(metrics, slices, resolved_thresholds)
    return RetrievalEvaluationReport(
        generation_id=generation_id,
        top_k=top_k,
        candidate_limit=resolved_candidate_limit,
        case_count=len(results),
        hit_count=metrics.hit_count,
        hit_rate=metrics.hit_rate_at_k,
        mean_reciprocal_rank=metrics.mean_reciprocal_rank,
        results=results,
        hit_at_1_count=metrics.hit_at_1_count,
        hit_rate_at_1=metrics.hit_rate_at_1,
        mean_recall_at_k=metrics.mean_recall_at_k,
        mean_average_precision=metrics.mean_average_precision,
        forbidden_hit_count=metrics.forbidden_hit_count,
        forbidden_case_rate=metrics.forbidden_case_rate,
        forbidden_at_1_count=metrics.forbidden_at_1_count,
        forbidden_at_1_case_rate=metrics.forbidden_at_1_case_rate,
        slices=slices,
        passed=not violations,
        violations=violations,
        retrieval_mode="hybrid" if use_hybrid_backend else "dense",
        duration_seconds=time.monotonic() - started_at,
    )


@dataclass(frozen=True, slots=True)
class _Aggregate:
    hit_count: int
    hit_at_1_count: int
    hit_rate_at_k: float
    hit_rate_at_1: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    mean_average_precision: float
    forbidden_hit_count: int
    forbidden_case_rate: float
    forbidden_at_1_count: int
    forbidden_at_1_case_rate: float


def _case_result(
    case: RetrievalEvaluationCase,
    hits: tuple[SearchHit, ...],
    *,
    top_k: int,
) -> RetrievalCaseResult:
    if not isinstance(hits, tuple) or any(
        not isinstance(hit, SearchHit) for hit in hits
    ):
        raise RetrievalEvaluationError("vector store returned invalid search hits")
    chunk_ids = tuple(hit.chunk_id for hit in hits)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise RetrievalEvaluationError("vector store returned duplicate chunk hits")
    if any(left.score < right.score for left, right in zip(hits, hits[1:])):
        raise RetrievalEvaluationError("vector store returned unsorted search hits")

    ranked_sources = tuple(dict.fromkeys(hit.source_id for hit in hits))[:top_k]
    rank_by_source = {
        source_id: index for index, source_id in enumerate(ranked_sources, start=1)
    }
    expected_ranks = tuple(
        rank_by_source.get(source_id, 0) for source_id in case.expected_source_ids
    )
    found_ranks = sorted(rank for rank in expected_ranks if rank)
    reciprocal_rank = 0.0 if not found_ranks else 1.0 / found_ranks[0]
    recall_at_k = len(found_ranks) / len(case.expected_source_ids)
    relevant = set(case.expected_source_ids)
    relevant_seen = 0
    precision_sum = 0.0
    for rank, source_id in enumerate(ranked_sources, start=1):
        if source_id in relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    average_precision = precision_sum / len(relevant)
    forbidden_hit_count = len(set(ranked_sources) & set(case.forbidden_source_ids))
    forbidden_at_1 = bool(
        ranked_sources and ranked_sources[0] in set(case.forbidden_source_ids)
    )
    return RetrievalCaseResult(
        case_id=case.case_id,
        expected_source_ids=case.expected_source_ids,
        retrieved_source_ids=ranked_sources,
        hit=bool(found_ranks),
        reciprocal_rank=reciprocal_rank,
        hit_at_1=bool(ranked_sources and ranked_sources[0] in relevant),
        recall_at_k=recall_at_k,
        average_precision=average_precision,
        expected_ranks=expected_ranks,
        forbidden_hit_count=forbidden_hit_count,
        forbidden_at_1=forbidden_at_1,
    )


def _aggregate(results: tuple[RetrievalCaseResult, ...]) -> _Aggregate:
    count = len(results)
    hit_count = sum(result.hit for result in results)
    hit_at_1_count = sum(result.hit_at_1 for result in results)
    forbidden_hit_count = sum(result.forbidden_hit_count for result in results)
    forbidden_at_1_count = sum(result.forbidden_at_1 for result in results)
    values = (
        hit_count / count,
        hit_at_1_count / count,
        sum(result.reciprocal_rank for result in results) / count,
        sum(result.recall_at_k for result in results) / count,
        sum(result.average_precision for result in results) / count,
        sum(result.forbidden_hit_count > 0 for result in results) / count,
        forbidden_at_1_count / count,
    )
    if any(not math.isfinite(value) for value in values):
        raise RetrievalEvaluationError("evaluation metrics are not finite")
    return _Aggregate(
        hit_count=hit_count,
        hit_at_1_count=hit_at_1_count,
        hit_rate_at_k=values[0],
        hit_rate_at_1=values[1],
        mean_reciprocal_rank=values[2],
        mean_recall_at_k=values[3],
        mean_average_precision=values[4],
        forbidden_hit_count=forbidden_hit_count,
        forbidden_case_rate=values[5],
        forbidden_at_1_count=forbidden_at_1_count,
        forbidden_at_1_case_rate=values[6],
    )


def _slice(tag: str, results: tuple[RetrievalCaseResult, ...]) -> RetrievalSliceMetrics:
    metrics = _aggregate(results)
    return RetrievalSliceMetrics(
        tag=tag,
        case_count=len(results),
        hit_rate_at_1=metrics.hit_rate_at_1,
        hit_rate_at_k=metrics.hit_rate_at_k,
        mean_reciprocal_rank=metrics.mean_reciprocal_rank,
        mean_recall_at_k=metrics.mean_recall_at_k,
        mean_average_precision=metrics.mean_average_precision,
        forbidden_case_rate=metrics.forbidden_case_rate,
        forbidden_at_1_case_rate=metrics.forbidden_at_1_case_rate,
    )


def _threshold_violations(
    metrics: _Aggregate,
    slices: tuple[RetrievalSliceMetrics, ...],
    thresholds: RetrievalEvaluationThresholds,
) -> tuple[str, ...]:
    checks = (
        (
            metrics.hit_rate_at_1 < thresholds.min_hit_rate_at_1,
            "hit_rate_at_1_below_minimum",
        ),
        (
            metrics.hit_rate_at_k < thresholds.min_hit_rate_at_k,
            "hit_rate_at_k_below_minimum",
        ),
        (
            metrics.mean_reciprocal_rank < thresholds.min_mean_reciprocal_rank,
            "mean_reciprocal_rank_below_minimum",
        ),
        (
            metrics.mean_recall_at_k < thresholds.min_mean_recall_at_k,
            "mean_recall_at_k_below_minimum",
        ),
        (
            metrics.mean_average_precision < thresholds.min_mean_average_precision,
            "mean_average_precision_below_minimum",
        ),
        (
            metrics.forbidden_case_rate > thresholds.max_forbidden_case_rate,
            "forbidden_case_rate_above_maximum",
        ),
        (
            metrics.forbidden_at_1_case_rate > thresholds.max_forbidden_at_1_case_rate,
            "forbidden_at_1_case_rate_above_maximum",
        ),
    )
    violations = [code for failed, code in checks if failed]
    for item in slices:
        if item.hit_rate_at_1 < thresholds.min_slice_hit_rate_at_1:
            violations.append(f"slice_{item.tag}_hit_rate_at_1_below_minimum")
        if item.hit_rate_at_k < thresholds.min_slice_hit_rate_at_k:
            violations.append(f"slice_{item.tag}_hit_rate_at_k_below_minimum")
    return tuple(violations)


def _source_ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    resolved = tuple(values)
    if (
        len(resolved) > 100
        or len(set(resolved)) != len(resolved)
        or any(
            not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None
            for source_id in resolved
        )
    ):
        raise ValueError(f"{name} must contain unique safe source IDs")
    return resolved


__all__ = [
    "MAX_EVALUATION_CASES",
    "MAX_EVALUATION_QUERY_CHARS",
    "MAX_EVALUATION_TAGS",
    "MAX_SEARCH_CONCURRENCY",
    "RetrievalCaseResult",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationError",
    "RetrievalEvaluationReport",
    "RetrievalEvaluationThresholds",
    "RetrievalSliceMetrics",
    "evaluate_retrieval",
]
