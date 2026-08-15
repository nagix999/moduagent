"""Download a pinned, checksummed NIST SP 800 cybersecurity corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urljoin, urlsplit

import httpx


NIST_SP800_INDEX_URL = "https://csrc.nist.gov/publications/sp800"
ALLOWED_NIST_HOSTS = frozenset({"csrc.nist.gov", "nvlpubs.nist.gov"})
DEFAULT_USER_AGENT = (
    "moduagent-rag-example/0.6 "
    "(+https://github.com/nagix999/moduagent; public test corpus)"
)
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_DETAIL_BYTES = 2 * 1024 * 1024
MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_DOCUMENTS = 200
SELECTION_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
_FINAL_DETAIL_PATH = re.compile(
    r"^/pubs/sp/800/[A-Za-z0-9-]+(?:/[A-Za-z0-9-]+)*/final/?$"
)
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.pdf$", re.I)


class NISTCorpusError(RuntimeError):
    """The NIST corpus could not be selected or downloaded safely."""


@dataclass(frozen=True, slots=True)
class NISTDocument:
    """One immutable selection entry before or after PDF verification."""

    title: str
    detail_url: str
    download_url: str
    filename: str
    sha256: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("NIST document title must not be empty")
        if len(self.title) > 1_000:
            raise ValueError("NIST document title is too long")
        _require_nist_url(self.detail_url, host="csrc.nist.gov")
        _require_nist_url(self.download_url, host="nvlpubs.nist.gov")
        if _SAFE_FILENAME.fullmatch(self.filename) is None:
            raise ValueError("NIST document filename is unsafe")
        if (
            self.sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("NIST document SHA-256 is invalid")
        if self.size_bytes is not None and (
            type(self.size_bytes) is not int
            or not 1 <= self.size_bytes <= MAX_PDF_BYTES
        ):
            raise ValueError("NIST document size is invalid")


class _DocumentHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str, str]] = []
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._in_h1 = False
        self._h1_text: list[str] = []

    @property
    def title(self) -> str:
        return _normalized_text("".join(self._h1_text))

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag.lower() == "a" and self._anchor_href is None:
            self._anchor_href = attributes.get("href")
            self._anchor_text = []
        if tag.lower() == "h1":
            self._in_h1 = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._anchor_href is not None:
            self.anchors.append(
                (self._anchor_href, _normalized_text("".join(self._anchor_text)))
            )
            self._anchor_href = None
            self._anchor_text = []
        if tag.lower() == "h1":
            self._in_h1 = False

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_text.append(data)
        if self._in_h1:
            self._h1_text.append(data)


class NISTCorpusDownloader:
    """Select current Final SP 800 records and persist verified PDFs."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        request_timeout: float = 60.0,
        max_attempts: int = 3,
        delay_seconds: float = 0.2,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            not isinstance(user_agent, str)
            or not user_agent.strip()
            or len(user_agent) > 512
        ):
            raise ValueError("user_agent must be non-empty and bounded")
        if isinstance(request_timeout, bool) or not isinstance(
            request_timeout, (int, float)
        ):
            raise TypeError("request_timeout must be a number")
        if not 0 < float(request_timeout) <= 600:
            raise ValueError("request_timeout must be between zero and 600 seconds")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between one and five")
        if isinstance(delay_seconds, bool) or not isinstance(
            delay_seconds, (int, float)
        ):
            raise TypeError("delay_seconds must be a number")
        if not 0 <= float(delay_seconds) <= 10:
            raise ValueError("delay_seconds must be between zero and ten seconds")
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        self._client = client
        self._owns_client = client is None
        self.user_agent = user_agent.strip()
        self.request_timeout = float(request_timeout)
        self.max_attempts = max_attempts
        self.delay_seconds = float(delay_seconds)
        self._sleep = sleeper

    def __enter__(self) -> NISTCorpusDownloader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def download(
        self,
        corpus_root: str | os.PathLike[str],
        *,
        count: int = 100,
        refresh_selection: bool = False,
    ) -> tuple[NISTDocument, ...]:
        """Create ``documents/`` plus selection and checksum manifests."""

        if type(count) is not int or not 1 <= count <= MAX_DOCUMENTS:
            raise ValueError(f"count must be between one and {MAX_DOCUMENTS}")
        if type(refresh_selection) is not bool:
            raise TypeError("refresh_selection must be a bool")
        root = _safe_directory(corpus_root)
        documents_root = _safe_directory(root / "documents")
        selection_path = root / "selection.json"
        manifest_path = root / "corpus-manifest.json"
        prior_manifest = _read_manifest(manifest_path, required=False)

        if selection_path.exists() and not refresh_selection:
            selected = _read_selection(selection_path)
            if len(selected) < count:
                raise NISTCorpusError(
                    "saved selection is smaller than requested; use --refresh-selection"
                )
            selected = selected[:count]
        else:
            selected = self._select(count)
            _write_json_atomic(
                selection_path,
                {
                    "schema_version": SELECTION_SCHEMA_VERSION,
                    "source": NIST_SP800_INDEX_URL,
                    "documents": [_selection_dict(item) for item in selected],
                },
            )

        expected_names = {item.filename for item in selected}
        extras = sorted(
            path.name
            for path in documents_root.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".pdf"
            and path.name not in expected_names
        )
        if extras:
            raise NISTCorpusError(
                "documents directory contains PDFs outside the pinned selection; "
                "use a fresh corpus directory"
            )

        prior_by_name = {
            item.filename: item
            for item in prior_manifest
            if item.filename in expected_names
        }
        completed: list[NISTDocument] = []
        total_bytes = 0
        for index, item in enumerate(selected, start=1):
            destination = documents_root / item.filename
            prior = prior_by_name.get(item.filename)
            verified = _verified_existing(destination, prior)
            if verified is None:
                verified = self._download_pdf(item, destination)
            total_bytes += verified.size_bytes or 0
            if total_bytes > MAX_TOTAL_BYTES:
                raise NISTCorpusError("NIST corpus exceeds the 2 GiB aggregate limit")
            completed.append(verified)
            print(f"[{index:03d}/{count:03d}] {verified.filename}", flush=True)
            if index != count and self.delay_seconds:
                self._sleep(self.delay_seconds)

        _write_json_atomic(
            manifest_path,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "domain": "NIST SP 800 cybersecurity and privacy",
                "source": NIST_SP800_INDEX_URL,
                "document_count": len(completed),
                "total_bytes": total_bytes,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "license_note": (
                    "NIST-authored Technical Series works are generally public information; "
                    "retain attribution and review item-specific third-party notices."
                ),
                "documents": [asdict(item) for item in completed],
            },
        )
        return tuple(completed)

    def _select(self, count: int) -> tuple[NISTDocument, ...]:
        listing = self._get_text(NIST_SP800_INDEX_URL, maximum=MAX_INDEX_BYTES)
        parser = _DocumentHTMLParser()
        parser.feed(listing)
        detail_urls: list[str] = []
        seen: set[str] = set()
        for href, _ in parser.anchors:
            resolved = urljoin(NIST_SP800_INDEX_URL, href)
            parsed = urlsplit(resolved)
            if (
                parsed.scheme == "https"
                and parsed.hostname == "csrc.nist.gov"
                and _FINAL_DETAIL_PATH.fullmatch(parsed.path)
                and resolved not in seen
            ):
                detail_urls.append(resolved)
                seen.add(resolved)
        if len(detail_urls) < count:
            raise NISTCorpusError(
                f"NIST listing exposed only {len(detail_urls)} Final SP 800 records"
            )

        selected: list[NISTDocument] = []
        filenames: set[str] = set()
        for detail_url in detail_urls:
            document = self._detail(detail_url)
            if document.filename in filenames:
                continue
            selected.append(document)
            filenames.add(document.filename)
            if len(selected) == count:
                break
            if self.delay_seconds:
                self._sleep(self.delay_seconds)
        if len(selected) != count:
            raise NISTCorpusError(
                "NIST records did not produce enough unique PDF files"
            )
        return tuple(selected)

    def _detail(self, detail_url: str) -> NISTDocument:
        _require_nist_url(detail_url, host="csrc.nist.gov")
        page = self._get_text(detail_url, maximum=MAX_DETAIL_BYTES)
        parser = _DocumentHTMLParser()
        parser.feed(page)
        primary_candidates: list[str] = []
        fallback_candidates: list[str] = []
        for href, anchor_text in parser.anchors:
            resolved = urljoin(detail_url, href)
            parsed = urlsplit(resolved)
            if (
                parsed.scheme == "https"
                and parsed.hostname == "nvlpubs.nist.gov"
                and parsed.path.lower().endswith(".pdf")
            ):
                fallback_candidates.append(resolved)
                if anchor_text.lower() == "download url":
                    primary_candidates.append(resolved)
        candidates = tuple(dict.fromkeys(primary_candidates or fallback_candidates))
        if len(candidates) != 1:
            raise NISTCorpusError("NIST publication page must expose exactly one PDF")
        download_url = candidates[0]
        filename = unquote(Path(urlsplit(download_url).path).name)
        if _SAFE_FILENAME.fullmatch(filename) is None:
            raise NISTCorpusError("NIST PDF filename is unsafe")
        return NISTDocument(
            title=parser.title,
            detail_url=detail_url,
            download_url=download_url,
            filename=filename,
        )

    def _download_pdf(self, item: NISTDocument, destination: Path) -> NISTDocument:
        response = self._request("GET", item.download_url, stream=True)
        temporary_path: Path | None = None
        digest = hashlib.sha256()
        total = 0
        first = b""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{item.filename}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                for chunk in response.iter_bytes(64 * 1024):
                    if not chunk:
                        continue
                    if len(first) < 5:
                        first += chunk[: 5 - len(first)]
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise NISTCorpusError("one NIST PDF exceeds 100 MiB")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total < 5 or first != b"%PDF-":
                raise NISTCorpusError("NIST download is not a PDF")
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            response.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return NISTDocument(
            title=item.title,
            detail_url=item.detail_url,
            download_url=item.download_url,
            filename=item.filename,
            sha256=digest.hexdigest(),
            size_bytes=total,
        )

    def _get_text(self, url: str, *, maximum: int) -> str:
        response = self._request("GET", url, stream=False)
        try:
            content = response.content
            if len(content) > maximum:
                raise NISTCorpusError("NIST HTML response exceeds its size limit")
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise NISTCorpusError("NIST HTML response is not UTF-8") from exc
        finally:
            response.close()

    def _request(self, method: str, url: str, *, stream: bool) -> httpx.Response:
        _require_nist_url(url)
        client = self._get_client()
        current_url = url
        redirects = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                request = client.build_request(
                    method,
                    current_url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "application/pdf,text/html",
                    },
                )
                response = client.send(request, stream=stream)
            except httpx.NetworkError as exc:
                if attempt == self.max_attempts:
                    raise NISTCorpusError("NIST request failed") from exc
                self._sleep(min(2 ** (attempt - 1), 4))
                continue
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                response.close()
                if not location or redirects >= 3:
                    raise NISTCorpusError("NIST redirect is invalid")
                current_url = urljoin(current_url, location)
                _require_nist_url(current_url)
                redirects += 1
                continue
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                response.close()
                if attempt == self.max_attempts:
                    raise NISTCorpusError(
                        f"NIST request failed with HTTP {response.status_code}"
                    )
                self._sleep(min(2 ** (attempt - 1), 4))
                continue
            if not 200 <= response.status_code <= 299:
                status = response.status_code
                response.close()
                raise NISTCorpusError(f"NIST request failed with HTTP {status}")
            _require_nist_url(str(response.url))
            return response
        raise NISTCorpusError("NIST request retry budget was exhausted")

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.request_timeout,
                follow_redirects=False,
            )
        return self._client


def download_nist_corpus(
    corpus_root: str | os.PathLike[str],
    *,
    count: int = 100,
    refresh_selection: bool = False,
) -> tuple[NISTDocument, ...]:
    """Convenience entry point for scripts and notebooks."""

    with NISTCorpusDownloader() as downloader:
        return downloader.download(
            corpus_root,
            count=count,
            refresh_selection=refresh_selection,
        )


def _require_nist_url(value: str, *, host: str | None = None) -> None:
    if not isinstance(value, str) or len(value) > 2_048:
        raise NISTCorpusError("NIST URL is invalid")
    parsed = urlsplit(value)
    expected_hosts = {host} if host is not None else ALLOWED_NIST_HOSTS
    if (
        parsed.scheme != "https"
        or parsed.hostname not in expected_hosts
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise NISTCorpusError("NIST URL is outside the approved HTTPS hosts")


def _safe_directory(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise NISTCorpusError("corpus directories cannot be symbolic links")
        resolved = path.resolve(strict=True)
        if not stat.S_ISDIR(resolved.stat().st_mode):
            raise NISTCorpusError("corpus path must be a directory")
        return resolved
    except NISTCorpusError:
        raise
    except OSError as exc:
        raise NISTCorpusError("corpus directory is not accessible") from exc


def _selection_dict(value: NISTDocument) -> dict[str, str]:
    return {
        "title": value.title,
        "detail_url": value.detail_url,
        "download_url": value.download_url,
        "filename": value.filename,
    }


def _read_selection(path: Path) -> tuple[NISTDocument, ...]:
    value = _read_json(path)
    if value.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise NISTCorpusError("NIST selection schema is unsupported")
    raw_documents = value.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise NISTCorpusError("NIST selection has no documents")
    try:
        documents = tuple(
            NISTDocument(
                title=item["title"],
                detail_url=item["detail_url"],
                download_url=item["download_url"],
                filename=item["filename"],
            )
            for item in raw_documents
            if isinstance(item, Mapping)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NISTCorpusError("NIST selection is malformed") from exc
    if len(documents) != len(raw_documents) or len(
        {item.filename for item in documents}
    ) != len(documents):
        raise NISTCorpusError("NIST selection contains invalid or duplicate documents")
    return documents


def _read_manifest(path: Path, *, required: bool) -> tuple[NISTDocument, ...]:
    if not path.exists() and not required:
        return ()
    value = _read_json(path)
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise NISTCorpusError("NIST corpus manifest schema is unsupported")
    raw_documents = value.get("documents")
    if not isinstance(raw_documents, list):
        raise NISTCorpusError("NIST corpus manifest is malformed")
    try:
        documents = tuple(
            NISTDocument(**item) for item in raw_documents if isinstance(item, Mapping)
        )
    except (TypeError, ValueError) as exc:
        raise NISTCorpusError("NIST corpus manifest is malformed") from exc
    if len(documents) != len(raw_documents):
        raise NISTCorpusError("NIST corpus manifest contains invalid documents")
    return documents


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 2 * 1024 * 1024
        ):
            raise NISTCorpusError("NIST metadata file is unsafe")
        value = json.loads(path.read_text(encoding="utf-8"))
    except NISTCorpusError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NISTCorpusError("NIST metadata file could not be decoded") from exc
    if not isinstance(value, Mapping):
        raise NISTCorpusError("NIST metadata root must be an object")
    return value


def _verified_existing(path: Path, prior: NISTDocument | None) -> NISTDocument | None:
    if (
        not path.exists()
        or prior is None
        or prior.sha256 is None
        or prior.size_bytes is None
    ):
        return None
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != prior.size_bytes
        ):
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            first = source.read(5)
            if first != b"%PDF-":
                return None
            digest.update(first)
            for chunk in iter(lambda: source.read(64 * 1024), b""):
                digest.update(chunk)
        return prior if digest.hexdigest() == prior.sha256 else None
    except OSError:
        return None


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


__all__ = [
    "ALLOWED_NIST_HOSTS",
    "DEFAULT_USER_AGENT",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_DOCUMENTS",
    "NISTCorpusDownloader",
    "NISTCorpusError",
    "NISTDocument",
    "NIST_SP800_INDEX_URL",
    "SELECTION_SCHEMA_VERSION",
    "download_nist_corpus",
]
