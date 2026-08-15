"""Deterministic, single-domain validation corpus and lifecycle harness.

This module deliberately uses synthetic policy documents.  It can therefore
exercise create/modify/delete/rollback paths without touching a real corporate
corpus, while still calling the configured Docling, Gemma, BGE-M3, and Milvus
services.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
    RetrievalEvaluationThresholds,
    evaluate_retrieval,
)
from .models import RAGIndexError, stable_digest
from .pipeline import RAGIndexManager, SyncReport


VALIDATION_SCHEMA_VERSION = 1
MAX_VALIDATION_DOCUMENTS = 200
_MARKER_NAME = ".moduagent-validation-corpus.state"
_KB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FORMATS = ("txt", "md", "html", "csv")
_CATEGORIES = (
    "leave",
    "incident",
    "procurement",
    "retention",
    "backup",
    "access",
    "vendor",
    "change",
    "travel",
    "training",
)
_SOURCE_ID = re.compile(r"^src_[a-f0-9]{32}$")
_POLICY_REF = re.compile(r"^CORP-[A-Z]+-[0-9]{3}$")
_TEAM = re.compile(r"^Orion-[0-9]{3}$")


class ValidationHarnessError(RAGIndexError):
    """The generated corpus or lifecycle violated its deterministic contract."""


@dataclass(frozen=True, slots=True)
class ValidationDocument:
    index: int
    relative_path: str
    source_id: str
    category: str
    policy_ref: str
    team: str
    revision: int
    format: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or not 0 <= self.index <= 999:
            raise ValueError("validation document index is invalid")
        path = PurePosixPath(self.relative_path)
        if (
            not isinstance(self.relative_path, str)
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in path.parts)
        ):
            raise ValueError("validation document path is unsafe")
        if _SOURCE_ID.fullmatch(self.source_id) is None:
            raise ValueError("validation document source ID is invalid")
        if self.category not in _CATEGORIES:
            raise ValueError("validation document category is invalid")
        if _POLICY_REF.fullmatch(self.policy_ref) is None:
            raise ValueError("validation document policy reference is invalid")
        if _TEAM.fullmatch(self.team) is None:
            raise ValueError("validation document team is invalid")
        if type(self.revision) is not int or not 1 <= self.revision <= 100:
            raise ValueError("validation document revision is invalid")
        if self.format not in _FORMATS or path.suffix != f".{self.format}":
            raise ValueError("validation document format is invalid")


@dataclass(frozen=True, slots=True)
class GeneratedValidationCorpus:
    root: Path
    kb_id: str
    documents: tuple[ValidationDocument, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("validation corpus root must be an absolute Path")
        if not isinstance(self.kb_id, str) or _KB_ID.fullmatch(self.kb_id) is None:
            raise ValueError("validation corpus KB ID is invalid")
        documents = tuple(self.documents)
        if any(not isinstance(item, ValidationDocument) for item in documents):
            raise TypeError("validation corpus contains an invalid document")
        if len({item.index for item in documents}) != len(documents) or len(
            {item.relative_path for item in documents}
        ) != len(documents):
            raise ValueError("validation corpus document identity is not unique")
        if any(
            item.source_id
            != "src_" + stable_digest(self.kb_id, item.relative_path)[:32]
            for item in documents
        ):
            raise ValueError("validation corpus source ID does not match its path")
        object.__setattr__(self, "documents", documents)

    @property
    def cases(self) -> tuple[RetrievalEvaluationCase, ...]:
        by_category: dict[str, list[ValidationDocument]] = {}
        by_fact: dict[tuple[str, str], list[ValidationDocument]] = {}
        for document in self.documents:
            by_category.setdefault(document.category, []).append(document)
            by_fact.setdefault((document.category, _fact(document)[2]), []).append(
                document
            )
        cases: list[RetrievalEvaluationCase] = []
        for document in self.documents:
            peers = by_category[document.category]
            expected_reverse = tuple(
                item.source_id
                for item in by_fact[(document.category, _fact(document)[2])]
            )
            forbidden = tuple(
                peer.source_id
                for peer in peers
                if peer.source_id not in expected_reverse
            )[:2]
            exact, semantic, korean, reverse, reverse_korean = _queries(document)
            common_tags = (document.category, f"format-{document.format}")
            cases.extend(
                (
                    RetrievalEvaluationCase(
                        f"doc-{document.index:03d}-exact",
                        exact,
                        (document.source_id,),
                        forbidden,
                        (*common_tags, "exact"),
                    ),
                    RetrievalEvaluationCase(
                        f"doc-{document.index:03d}-semantic",
                        semantic,
                        (document.source_id,),
                        forbidden,
                        (*common_tags, "semantic"),
                    ),
                    RetrievalEvaluationCase(
                        f"doc-{document.index:03d}-ko",
                        korean,
                        (document.source_id,),
                        forbidden,
                        (*common_tags, "multilingual-ko"),
                    ),
                    RetrievalEvaluationCase(
                        f"doc-{document.index:03d}-reverse",
                        reverse,
                        expected_reverse,
                        forbidden,
                        (*common_tags, "anchor-free", "reverse-lookup"),
                    ),
                    RetrievalEvaluationCase(
                        f"doc-{document.index:03d}-reverse-ko",
                        reverse_korean,
                        expected_reverse,
                        forbidden,
                        (
                            *common_tags,
                            "anchor-free",
                            "multilingual-ko",
                            "reverse-lookup-ko",
                        ),
                    ),
                )
            )
        return tuple(cases)


@dataclass(frozen=True, slots=True)
class ValidationCorpusMutation:
    before: GeneratedValidationCorpus
    after: GeneratedValidationCorpus
    modified_source_ids: tuple[str, ...]
    deleted_source_ids: tuple[str, ...]
    added_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidationPhase:
    name: str
    generation_id: str | None
    status: str
    document_count: int
    chunk_count: int
    new_count: int
    modified_count: int
    deleted_count: int
    unchanged_count: int
    duration_seconds: float = 0.0

    @classmethod
    def from_sync(
        cls,
        name: str,
        report: SyncReport,
        *,
        duration_seconds: float = 0.0,
    ) -> ValidationPhase:
        return cls(
            name=name,
            generation_id=report.generation_id,
            status=report.status,
            document_count=report.document_count,
            chunk_count=report.chunk_count,
            new_count=report.new_count,
            modified_count=report.modified_count,
            deleted_count=report.deleted_count,
            unchanged_count=report.unchanged_count,
            duration_seconds=duration_seconds,
        )


@dataclass(frozen=True, slots=True)
class ValidationLifecycleReport:
    kb_id: str
    document_count: int
    query_count: int
    phases: tuple[ValidationPhase, ...]
    baseline_quality: RetrievalEvaluationReport
    mutated_quality: RetrievalEvaluationReport
    rollback_quality: RetrievalEvaluationReport
    final_quality: RetrievalEvaluationReport
    passed: bool
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "document_count": self.document_count,
            "query_count": self.query_count,
            "phases": [
                {
                    "name": phase.name,
                    "generation_id": phase.generation_id,
                    "status": phase.status,
                    "document_count": phase.document_count,
                    "chunk_count": phase.chunk_count,
                    "new_count": phase.new_count,
                    "modified_count": phase.modified_count,
                    "deleted_count": phase.deleted_count,
                    "unchanged_count": phase.unchanged_count,
                    "duration_seconds": phase.duration_seconds,
                }
                for phase in self.phases
            ],
            "baseline_quality": self.baseline_quality.as_dict(),
            "mutated_quality": self.mutated_quality.as_dict(),
            "rollback_quality": self.rollback_quality.as_dict(),
            "final_quality": self.final_quality.as_dict(),
            "passed": self.passed,
            "violations": list(self.violations),
        }


def generate_validation_corpus(
    root: str | os.PathLike[str],
    *,
    kb_id: str = "rag-validation",
    document_count: int = 100,
) -> GeneratedValidationCorpus:
    """Create or verify a deterministic policy corpus in a dedicated directory."""

    if not isinstance(kb_id, str) or _KB_ID.fullmatch(kb_id) is None:
        raise ValueError("kb_id must be a safe bounded identifier")
    if (
        type(document_count) is not int
        or not 2 <= document_count <= MAX_VALIDATION_DOCUMENTS
    ):
        raise ValueError(
            f"document_count must be between two and {MAX_VALIDATION_DOCUMENTS}"
        )
    target = Path(root).expanduser().absolute()
    _require_safe_directory(target, create=True)
    marker = target / _MARKER_NAME
    entries = tuple(target.iterdir())
    if entries:
        if not marker.is_file() or marker.is_symlink():
            raise ValidationHarnessError(
                "validation root is not an owned generated corpus"
            )
        existing = load_validation_corpus(target)
        if existing.kb_id != kb_id or len(existing.documents) != document_count:
            raise ValidationHarnessError(
                "existing generated corpus does not match requested configuration"
            )
        _verify_corpus_files(existing)
        return existing

    documents = tuple(
        _document(kb_id, index, revision=1) for index in range(document_count)
    )
    corpus = GeneratedValidationCorpus(target, kb_id, documents)
    _write_corpus(corpus)
    return corpus


def load_validation_corpus(
    root: str | os.PathLike[str],
) -> GeneratedValidationCorpus:
    target = Path(root).expanduser().absolute()
    _require_safe_directory(target, create=False)
    marker = target / _MARKER_NAME
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationHarnessError("validation corpus marker is invalid") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != VALIDATION_SCHEMA_VERSION
        or not isinstance(raw.get("kb_id"), str)
        or not isinstance(raw.get("documents"), list)
    ):
        raise ValidationHarnessError("validation corpus marker schema is invalid")
    try:
        documents = tuple(ValidationDocument(**item) for item in raw["documents"])
    except (TypeError, ValueError) as exc:
        raise ValidationHarnessError(
            "validation corpus document metadata is invalid"
        ) from exc
    corpus = GeneratedValidationCorpus(target, raw["kb_id"], documents)
    if len(documents) < 2 or len(documents) > MAX_VALIDATION_DOCUMENTS:
        raise ValidationHarnessError("validation corpus size is invalid")
    if len({item.relative_path for item in documents}) != len(documents):
        raise ValidationHarnessError("validation corpus paths are not unique")
    return corpus


def apply_validation_mutations(
    corpus: GeneratedValidationCorpus,
    *,
    modify_count: int | None = None,
    delete_count: int | None = None,
    add_count: int | None = None,
) -> ValidationCorpusMutation:
    """Apply a bounded deterministic mutation set to an owned corpus."""

    if not isinstance(corpus, GeneratedValidationCorpus):
        raise TypeError("corpus must be a GeneratedValidationCorpus")
    _verify_corpus_files(corpus)
    count = len(corpus.documents)
    modified_total = max(1, count // 10) if modify_count is None else modify_count
    deleted_total = max(1, count // 20) if delete_count is None else delete_count
    added_total = deleted_total if add_count is None else add_count
    for value, name in (
        (modified_total, "modify_count"),
        (deleted_total, "delete_count"),
        (added_total, "add_count"),
    ):
        if type(value) is not int or value < 0 or value > MAX_VALIDATION_DOCUMENTS:
            raise ValueError(f"{name} is outside its supported range")
    if modified_total + deleted_total >= count:
        raise ValueError("mutations must retain at least one original document")
    if count - deleted_total + added_total > MAX_VALIDATION_DOCUMENTS:
        raise ValueError("mutated corpus exceeds its document limit")

    ordered = tuple(sorted(corpus.documents, key=lambda item: item.index))
    modified = ordered[:modified_total]
    deleted = ordered[-deleted_total:] if deleted_total else ()
    retained_middle = ordered[modified_total : count - deleted_total]
    revised = tuple(
        _document(
            corpus.kb_id,
            item.index,
            revision=item.revision + 1,
            relative_path=item.relative_path,
        )
        for item in modified
    )
    next_index = max(item.index for item in ordered) + 1
    added = tuple(
        _document(corpus.kb_id, next_index + offset, revision=1)
        for offset in range(added_total)
    )
    for item in deleted:
        path = corpus.root / item.relative_path
        _require_owned_regular_file(path)
        path.unlink()
    after = GeneratedValidationCorpus(
        corpus.root,
        corpus.kb_id,
        tuple(
            sorted((*revised, *retained_middle, *added), key=lambda item: item.index)
        ),
    )
    _write_corpus(after)
    return ValidationCorpusMutation(
        before=corpus,
        after=after,
        modified_source_ids=tuple(item.source_id for item in revised),
        deleted_source_ids=tuple(item.source_id for item in deleted),
        added_source_ids=tuple(item.source_id for item in added),
    )


def default_validation_thresholds() -> RetrievalEvaluationThresholds:
    """Return the fixed quality gate used by generated validation."""

    return RetrievalEvaluationThresholds(
        min_hit_rate_at_1=0.70,
        min_hit_rate_at_k=0.90,
        min_mean_reciprocal_rank=0.80,
        min_mean_recall_at_k=0.90,
        min_mean_average_precision=0.80,
        # A same-topic peer in Top-K is a useful diagnostic but is not an
        # error. Substitution at rank one is the hard-negative gate.
        max_forbidden_case_rate=1.0,
        max_forbidden_at_1_case_rate=0.20,
        min_slice_hit_rate_at_1=0.60,
        min_slice_hit_rate_at_k=0.80,
    )


async def evaluate_validation_corpus(
    manager: RAGIndexManager,
    corpus: GeneratedValidationCorpus,
    *,
    top_k: int = 5,
    thresholds: RetrievalEvaluationThresholds | None = None,
) -> RetrievalEvaluationReport:
    """Evaluate the published generation without rebuilding or mutating it."""

    if not isinstance(manager, RAGIndexManager):
        raise TypeError("manager must be a RAGIndexManager")
    if not isinstance(corpus, GeneratedValidationCorpus):
        raise TypeError("corpus must be a GeneratedValidationCorpus")
    if manager.config.kb_id != corpus.kb_id:
        raise ValidationHarnessError("manager and validation corpus KB IDs differ")
    if manager.config.document_root.absolute() != corpus.root:
        raise ValidationHarnessError("manager and validation corpus roots differ")
    status = await manager.status()
    if not status.consistent or status.document_count != len(corpus.documents):
        raise ValidationHarnessError(
            "published generation does not match the generated corpus"
        )
    return await evaluate_retrieval(
        manager.embedder,
        manager.vector_store,
        corpus.cases,
        top_k=top_k,
        thresholds=thresholds or default_validation_thresholds(),
    )


async def run_validation_lifecycle(
    manager: RAGIndexManager,
    corpus: GeneratedValidationCorpus,
    *,
    top_k: int = 5,
    thresholds: RetrievalEvaluationThresholds | None = None,
    modify_count: int | None = None,
    delete_count: int | None = None,
    add_count: int | None = None,
) -> ValidationLifecycleReport:
    """Exercise publish, no-op, mutation, rollback, and recovery end to end."""

    if not isinstance(manager, RAGIndexManager):
        raise TypeError("manager must be a RAGIndexManager")
    if not isinstance(corpus, GeneratedValidationCorpus):
        raise TypeError("corpus must be a GeneratedValidationCorpus")
    if manager.config.kb_id != corpus.kb_id:
        raise ValidationHarnessError("manager and validation corpus KB IDs differ")
    if manager.config.document_root.absolute() != corpus.root:
        raise ValidationHarnessError("manager and validation corpus roots differ")
    resolved_thresholds = thresholds or default_validation_thresholds()

    phases: list[ValidationPhase] = []
    violations: list[str] = []
    phase_started = time.monotonic()
    baseline = await manager.sync(force_rebuild=True)
    phases.append(
        ValidationPhase.from_sync(
            "baseline_rebuild",
            baseline,
            duration_seconds=time.monotonic() - phase_started,
        )
    )
    _expect(baseline.status == "published", "baseline_not_published", violations)
    _expect(
        baseline.document_count == len(corpus.documents),
        "baseline_document_count_mismatch",
        violations,
    )
    baseline_quality = await evaluate_retrieval(
        manager.embedder,
        manager.vector_store,
        corpus.cases,
        top_k=top_k,
        thresholds=resolved_thresholds,
    )

    phase_started = time.monotonic()
    noop = await manager.sync()
    phases.append(
        ValidationPhase.from_sync(
            "baseline_noop",
            noop,
            duration_seconds=time.monotonic() - phase_started,
        )
    )
    _expect(noop.status == "noop", "unchanged_sync_not_noop", violations)
    _expect(
        noop.unchanged_count == len(corpus.documents),
        "unchanged_count_mismatch",
        violations,
    )

    mutation = apply_validation_mutations(
        corpus,
        modify_count=modify_count,
        delete_count=delete_count,
        add_count=add_count,
    )
    phase_started = time.monotonic()
    changed = await manager.sync()
    phases.append(
        ValidationPhase.from_sync(
            "mutation",
            changed,
            duration_seconds=time.monotonic() - phase_started,
        )
    )
    _expect(changed.status == "published", "mutation_not_published", violations)
    _expect(
        changed.modified_count == len(mutation.modified_source_ids),
        "modified_count_mismatch",
        violations,
    )
    _expect(
        changed.deleted_count == len(mutation.deleted_source_ids),
        "deleted_count_mismatch",
        violations,
    )
    _expect(
        changed.new_count == len(mutation.added_source_ids),
        "new_count_mismatch",
        violations,
    )
    mutated_quality = await evaluate_retrieval(
        manager.embedder,
        manager.vector_store,
        mutation.after.cases,
        top_k=top_k,
        thresholds=resolved_thresholds,
    )

    phase_started = time.monotonic()
    rolled_back = await manager.rollback()
    phases.append(
        ValidationPhase.from_sync(
            "rollback",
            rolled_back,
            duration_seconds=time.monotonic() - phase_started,
        )
    )
    _expect(rolled_back.status == "rolled_back", "rollback_failed", violations)
    _expect(
        rolled_back.generation_id == baseline.generation_id,
        "rollback_generation_mismatch",
        violations,
    )
    rollback_quality = await evaluate_retrieval(
        manager.embedder,
        manager.vector_store,
        mutation.before.cases,
        top_k=top_k,
        thresholds=resolved_thresholds,
    )

    phase_started = time.monotonic()
    recovered = await manager.sync()
    phases.append(
        ValidationPhase.from_sync(
            "recover_mutation",
            recovered,
            duration_seconds=time.monotonic() - phase_started,
        )
    )
    _expect(recovered.status == "published", "recovery_not_published", violations)
    final_quality = await evaluate_retrieval(
        manager.embedder,
        manager.vector_store,
        mutation.after.cases,
        top_k=top_k,
        thresholds=resolved_thresholds,
    )
    phase_started = time.monotonic()
    final_noop = await manager.sync()
    phases.append(
        ValidationPhase.from_sync(
            "final_noop",
            final_noop,
            duration_seconds=time.monotonic() - phase_started,
        )
    )
    _expect(final_noop.status == "noop", "final_sync_not_noop", violations)
    status = await manager.status()
    _expect(status.consistent, "final_manifest_vector_mismatch", violations)
    _expect(
        status.document_count == len(mutation.after.documents),
        "final_document_count_mismatch",
        violations,
    )
    for name, quality in (
        ("baseline", baseline_quality),
        ("mutation", mutated_quality),
        ("rollback", rollback_quality),
        ("final", final_quality),
    ):
        if not quality.passed:
            violations.extend(f"{name}_{code}" for code in quality.violations)
    return ValidationLifecycleReport(
        kb_id=corpus.kb_id,
        document_count=len(corpus.documents),
        query_count=len(corpus.cases),
        phases=tuple(phases),
        baseline_quality=baseline_quality,
        mutated_quality=mutated_quality,
        rollback_quality=rollback_quality,
        final_quality=final_quality,
        passed=not violations,
        violations=tuple(violations),
    )


def _document(
    kb_id: str,
    index: int,
    *,
    revision: int,
    relative_path: str | None = None,
) -> ValidationDocument:
    category = _category(index)
    extension = _FORMATS[index % len(_FORMATS)]
    path = relative_path or f"{category}/policy-{index:03d}.{extension}"
    return ValidationDocument(
        index=index,
        relative_path=path,
        source_id="src_" + stable_digest(kb_id, path)[:32],
        category=category,
        policy_ref=f"CORP-{category.upper()}-{index:03d}",
        team=f"Orion-{index:03d}",
        revision=revision,
        format=Path(path).suffix.lstrip("."),
    )


def _category(index: int) -> str:
    return _CATEGORIES[index % len(_CATEGORIES)]


def _fact(document: ValidationDocument) -> tuple[str, str, str, str]:
    value = 7 + (document.index * 7 + document.revision * 3) % 83
    amount = 500 + value * 25
    values = {
        "leave": (
            f"Members of {document.team} receive {value} paid annual leave days.",
            "paid annual leave days",
            str(value),
            "유급 연차 일수",
        ),
        "incident": (
            f"{document.team} must acknowledge a severity-one incident within {value} minutes.",
            "severity-one acknowledgement deadline in minutes",
            str(value),
            "심각도 1 사고 승인 제한 시간(분)",
        ),
        "procurement": (
            f"{document.team} requires finance and security approval above {amount} dollars.",
            "dual-approval purchase threshold in dollars",
            str(amount),
            "재무·보안 이중 승인 구매 기준 금액",
        ),
        "retention": (
            f"{document.team} retains audit evidence for {value} months.",
            "audit evidence retention period in months",
            str(value),
            "감사 증적 보존 개월 수",
        ),
        "backup": (
            f"{document.team} uses a recovery point objective of {value} hours.",
            "recovery point objective in hours",
            str(value),
            "복구 시점 목표 시간",
        ),
        "access": (
            f"{document.team} reviews privileged access every {value} days.",
            "privileged-access review interval in days",
            str(value),
            "특권 접근 검토 주기(일)",
        ),
        "vendor": (
            f"{document.team} completes critical vendor assessments within {value} days.",
            "critical-vendor assessment deadline in days",
            str(value),
            "중요 공급업체 평가 기한(일)",
        ),
        "change": (
            f"{document.team} freezes production changes for {value} hours before quarter close.",
            "pre-close production change freeze in hours",
            str(value),
            "분기 마감 전 운영 변경 동결 시간",
        ),
        "travel": (
            f"{document.team} has a nightly lodging cap of {amount} dollars.",
            "nightly lodging cap in dollars",
            str(amount),
            "1박 숙박비 한도",
        ),
        "training": (
            f"{document.team} completes {value} hours of annual security training.",
            "annual security training hours",
            str(value),
            "연간 보안 교육 시간",
        ),
    }
    return values[document.category]


def _queries(document: ValidationDocument) -> tuple[str, str, str, str, str]:
    _statement, subject, answer, korean_subject = _fact(document)
    return (
        f"Under {document.policy_ref}, what is the {subject} for {document.team}?",
        f"For the {document.team} business unit, identify its required {subject}.",
        f"{document.policy_ref}에 따르면 {document.team}의 {korean_subject}는 얼마인가?",
        f"Which policy has a {subject} of {answer}, and which business unit owns it?",
        f"{korean_subject}가 {answer}인 정책과 담당 업무 단위를 찾아라.",
    )


def _render(document: ValidationDocument) -> str:
    statement, subject, answer, _korean_subject = _fact(document)
    fields = {
        "title": f"Corporate {document.category.title()} Policy",
        "policy_reference": document.policy_ref,
        "business_unit": document.team,
        "revision": str(document.revision),
        "rule": statement,
        "control_subject": subject,
        "control_value": answer,
        "review_note": "The policy owner reviews this control every quarter.",
    }
    if document.format == "md":
        return "\n".join(
            (
                f"# {fields['title']}",
                f"\n**Policy reference:** {fields['policy_reference']}",
                f"\n**Business unit:** {fields['business_unit']}",
                f"\n## Mandatory control\n\n{fields['rule']}",
                f"\n## Review\n\n{fields['review_note']}",
            )
        )
    if document.format == "html":
        return (
            "<!doctype html><html><body>"
            f"<h1>{fields['title']}</h1><dl>"
            f"<dt>Policy reference</dt><dd>{fields['policy_reference']}</dd>"
            f"<dt>Business unit</dt><dd>{fields['business_unit']}</dd></dl>"
            f"<h2>Mandatory control</h2><p>{fields['rule']}</p>"
            f"<h2>Review</h2><p>{fields['review_note']}</p>"
            "</body></html>"
        )
    if document.format == "json":
        return json.dumps(fields, ensure_ascii=False, indent=2) + "\n"
    if document.format == "xml":
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n<policy>\n'
            + "\n".join(
                f"  <{name}>{_xml_escape(value)}</{name}>"
                for name, value in fields.items()
            )
            + "\n</policy>\n"
        )
    if document.format == "csv":
        return (
            "field,value\n"
            + "\n".join(
                f'{name},"{value.replace(chr(34), chr(34) * 2)}"'
                for name, value in fields.items()
            )
            + "\n"
        )
    return (
        "\n".join(
            f"{name.replace('_', ' ').title()}: {value}"
            for name, value in fields.items()
        )
        + "\n"
    )


def _write_corpus(corpus: GeneratedValidationCorpus) -> None:
    expected_paths = {item.relative_path for item in corpus.documents}
    for document in corpus.documents:
        path = corpus.root / document.relative_path
        _require_within(corpus.root, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(path, _render(document))
    for path in corpus.root.rglob("*"):
        if path.is_file() and path.name != _MARKER_NAME:
            relative = path.relative_to(corpus.root).as_posix()
            if relative not in expected_paths:
                raise ValidationHarnessError(
                    "generated corpus contains an unexpected regular file"
                )
    marker = corpus.root / _MARKER_NAME
    payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "kb_id": corpus.kb_id,
        "documents": [
            {
                "index": item.index,
                "relative_path": item.relative_path,
                "source_id": item.source_id,
                "category": item.category,
                "policy_ref": item.policy_ref,
                "team": item.team,
                "revision": item.revision,
                "format": item.format,
            }
            for item in corpus.documents
        ],
    }
    _write_text_atomic(
        marker,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _verify_corpus_files(corpus: GeneratedValidationCorpus) -> None:
    expected = {item.relative_path for item in corpus.documents}
    actual: set[str] = set()
    for path in corpus.root.rglob("*"):
        if path.is_symlink():
            raise ValidationHarnessError("generated corpus cannot contain symlinks")
        if path.is_file() and path.name != _MARKER_NAME:
            actual.add(path.relative_to(corpus.root).as_posix())
    if actual != expected:
        raise ValidationHarnessError("generated corpus files do not match its marker")
    for document in corpus.documents:
        path = corpus.root / document.relative_path
        _require_owned_regular_file(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValidationHarnessError("generated corpus file is unreadable") from exc
        if text != _render(document):
            raise ValidationHarnessError(
                "generated corpus file was modified externally"
            )


def _require_safe_directory(path: Path, *, create: bool) -> None:
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        info = path.lstat()
    except OSError as exc:
        raise ValidationHarnessError("validation root is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValidationHarnessError("validation root must be a real directory")


def _require_owned_regular_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValidationHarnessError("generated corpus file is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValidationHarnessError("generated corpus entry is not a regular file")


def _require_within(root: Path, path: Path) -> None:
    resolved_root = root.resolve(strict=True)
    resolved_parent = path.parent.resolve(strict=False)
    if (
        resolved_parent != resolved_root
        and resolved_root not in resolved_parent.parents
    ):
        raise ValidationHarnessError("generated corpus path escapes its root")


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() and path.is_symlink():
        raise ValidationHarnessError("refusing to replace a symlink")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValidationHarnessError("generated corpus write failed") from exc


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _expect(condition: bool, code: str, violations: list[str]) -> None:
    if not condition:
        violations.append(code)


__all__ = [
    "GeneratedValidationCorpus",
    "MAX_VALIDATION_DOCUMENTS",
    "VALIDATION_SCHEMA_VERSION",
    "ValidationCorpusMutation",
    "ValidationDocument",
    "ValidationHarnessError",
    "ValidationLifecycleReport",
    "ValidationPhase",
    "apply_validation_mutations",
    "default_validation_thresholds",
    "evaluate_validation_corpus",
    "generate_validation_corpus",
    "load_validation_corpus",
    "run_validation_lifecycle",
]
