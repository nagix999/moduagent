from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event, Thread
from types import MappingProxyType
from typing import Any, Callable, TypeVar

from moduagent.skills.errors import SkillError


__all__ = [
    "SkillResourceBinaryError",
    "SkillResourceChangedError",
    "SkillResourceCursorError",
    "SkillResourceError",
    "SkillResourceKind",
    "SkillResourceLoader",
    "SkillResourceMatch",
    "SkillResourceNotFoundError",
    "SkillResourcePage",
    "SkillResourcePathError",
    "SkillResourceSearchResult",
    "SkillResourceSecurityError",
    "SkillResourceTooLargeError",
]


DEFAULT_MAX_FILE_BYTES = 1024 * 1024
DEFAULT_MAX_READ_BYTES = 64 * 1024
DEFAULT_MAX_SEARCH_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_SEARCH_RESULTS = 20
DEFAULT_MAX_MATCH_CHARS = 1000
_ALLOWED_DIRECTORIES = frozenset({"references", "assets"})
_CURSOR_VERSION = 1


class SkillResourceError(SkillError):
    """Base class for safe Skill resource access failures."""


class SkillResourcePathError(SkillResourceError):
    """A resource path is invalid or outside the public resource directories."""


class SkillResourceNotFoundError(SkillResourceError):
    """A requested resource does not exist."""


class SkillResourceSecurityError(SkillResourceError):
    """A resource cannot be accessed safely."""


class SkillResourceBinaryError(SkillResourceError):
    """A resource is not UTF-8 text."""


class SkillResourceTooLargeError(SkillResourceError):
    """A resource or operation exceeds its configured byte limit."""


class SkillResourceCursorError(SkillResourceError):
    """A read or search cursor is invalid for the requested resource."""


class SkillResourceChangedError(SkillResourceError):
    """A resource does not match its pinned digest."""


class SkillResourceKind(str, Enum):
    REFERENCE = "reference"
    ASSET = "asset"


@dataclass(frozen=True, slots=True)
class SkillResourcePage:
    """One bounded UTF-8 page from a Skill resource.

    Cursors and sizes are byte based so the same limits apply regardless of the
    number of Unicode code points in a document. ``next_cursor`` always points
    at a valid UTF-8 boundary.
    """

    path: str
    kind: SkillResourceKind
    content: str
    cursor: int
    next_cursor: int | None
    size_bytes: int
    returned_bytes: int
    digest: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class SkillResourceMatch:
    path: str
    kind: SkillResourceKind
    line_number: int
    column: int
    text: str
    digest: str
    text_truncated: bool = False


@dataclass(frozen=True, slots=True)
class SkillResourceSearchResult:
    query: str
    matches: tuple[SkillResourceMatch, ...]
    cursor: str | None
    next_cursor: str | None
    scanned_files: int
    scanned_bytes: int
    digest: str
    truncated: bool


class SkillResourceLoader:
    """Safely read and search text resources under one active Skill root.

    Only files below ``references/`` and ``assets/`` are visible. In
    particular, this class never exposes ``SKILL.md`` or ``scripts/``. Resource
    paths use forward slashes independent of the host platform.

    The loader is intentionally independent from registry and runtime types so
    a registry can construct one after it has selected and pinned a Skill.
    """

    def __init__(
        self,
        skill_root: str | os.PathLike[str],
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_search_bytes: int = DEFAULT_MAX_SEARCH_BYTES,
        max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
        max_match_chars: int = DEFAULT_MAX_MATCH_CHARS,
        expected_digests: Mapping[str, str] | None = None,
    ) -> None:
        self._root_descriptor = -1
        self._max_file_bytes = _positive_int(max_file_bytes, "max_file_bytes")
        self._max_read_bytes = _positive_int(max_read_bytes, "max_read_bytes")
        self._max_search_bytes = _positive_int(max_search_bytes, "max_search_bytes")
        self._max_search_results = _positive_int(
            max_search_results, "max_search_results"
        )
        self._max_match_chars = _positive_int(max_match_chars, "max_match_chars")
        self._expected_digests = _normalize_expected_digests(expected_digests)
        if (
            not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
            or os.listdir not in os.supports_fd
        ):
            raise SkillResourceSecurityError(
                "secure Skill resources require openat, O_NOFOLLOW, and fd listing"
            )

        root = Path(skill_root).absolute()
        flags = (
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | os.O_DIRECTORY
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            root_descriptor = os.open(root, flags)
        except FileNotFoundError as exc:
            raise SkillResourceNotFoundError(
                f"Skill root does not exist: {root}"
            ) from exc
        except OSError as exc:
            raise SkillResourceSecurityError(
                "Skill root cannot be opened safely"
            ) from exc
        root_info = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_info.st_mode):
            os.close(root_descriptor)
            raise SkillResourcePathError("Skill root must be a directory")
        self._root = root
        self._root_descriptor = root_descriptor

    @property
    def skill_root(self) -> Path:
        return self._root

    def close(self) -> None:
        """Close the pinned Skill root descriptor. This method is idempotent."""

        descriptor = self._root_descriptor
        if descriptor >= 0:
            self._root_descriptor = -1
            os.close(descriptor)

    def __enter__(self) -> "SkillResourceLoader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    def read(
        self,
        path: str | os.PathLike[str],
        *,
        cursor: int = 0,
        max_bytes: int | None = None,
        expected_digest: str | None = None,
    ) -> SkillResourcePage:
        """Return a bounded page, rejecting non-text and changed resources."""

        relative = _validate_resource_path(path)
        self._require_pinned(relative)
        cursor = _non_negative_int(cursor, "cursor")
        limit = self._max_read_bytes
        if max_bytes is not None:
            limit = _positive_int(max_bytes, "max_bytes")
            if limit > self._max_read_bytes:
                raise SkillResourceTooLargeError(
                    "max_bytes exceeds the configured read limit"
                )

        payload, digest = self._read_text(relative)
        _check_expected_digest(expected_digest, digest)
        if cursor > len(payload):
            raise SkillResourceCursorError("cursor is past the end of the resource")
        try:
            payload[:cursor].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillResourceCursorError(
                "cursor does not point to a UTF-8 boundary"
            ) from exc

        end = min(len(payload), cursor + limit)
        while end > cursor:
            try:
                content = payload[cursor:end].decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            if cursor < len(payload):
                raise SkillResourceTooLargeError(
                    "max_bytes is too small for the next UTF-8 character"
                )
            content = ""

        truncated = end < len(payload)
        return SkillResourcePage(
            path=relative,
            kind=_resource_kind(relative),
            content=content,
            cursor=cursor,
            next_cursor=end if truncated else None,
            size_bytes=len(payload),
            returned_bytes=end - cursor,
            digest=digest,
            truncated=truncated,
        )

    async def aread(
        self,
        path: str | os.PathLike[str],
        *,
        cursor: int = 0,
        max_bytes: int | None = None,
        expected_digest: str | None = None,
    ) -> SkillResourcePage:
        """Asynchronous wrapper that keeps file IO off the event loop."""

        operation = partial(
            self.read,
            path,
            cursor=cursor,
            max_bytes=max_bytes,
            expected_digest=expected_digest,
        )
        return await _run_sync_in_daemon(operation)

    def search(
        self,
        query: str,
        *,
        path: str | os.PathLike[str] | None = None,
        cursor: str | int | None = None,
        max_results: int | None = None,
        max_scan_bytes: int | None = None,
        case_sensitive: bool = False,
    ) -> SkillResourceSearchResult:
        """Search allowed text resources with bounded, resumable pagination.

        The returned cursor is opaque and is tied to the query, search path,
        case-sensitivity setting, and current ordered resource path list.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if "\x00" in query:
            raise ValueError("query cannot contain NUL")

        result_limit = self._max_search_results
        if max_results is not None:
            result_limit = _positive_int(max_results, "max_results")
            if result_limit > self._max_search_results:
                raise SkillResourceTooLargeError(
                    "max_results exceeds the configured search limit"
                )
        scan_limit = self._max_search_bytes
        if max_scan_bytes is not None:
            scan_limit = _positive_int(max_scan_bytes, "max_scan_bytes")
            if scan_limit > self._max_search_bytes:
                raise SkillResourceTooLargeError(
                    "max_scan_bytes exceeds the configured search limit"
                )
        candidates = self._search_candidates(path)
        self._validate_search_snapshot(path, candidates)
        scope = _search_scope(query, path, case_sensitive, candidates)
        start_file, start_line, original_cursor = _decode_search_cursor(cursor, scope)
        if start_file > len(candidates):
            raise SkillResourceCursorError("search cursor is past the resource list")

        needle = query if case_sensitive else query.casefold()
        matches: list[SkillResourceMatch] = []
        scanned_files = 0
        scanned_bytes = 0
        next_position: tuple[int, int] | None = None
        scan_hasher = hashlib.sha256()
        scan_hasher.update(scope.encode("ascii"))

        for file_index in range(start_file, len(candidates)):
            relative = candidates[file_index]
            size = self._safe_size(relative)
            if scanned_files and scanned_bytes + size > scan_limit:
                next_position = (file_index, 0)
                break

            payload, resource_digest = self._read_text(relative)
            if scanned_bytes + len(payload) > scan_limit:
                # The constructor and per-call checks guarantee that a single
                # valid file can fit. This protects against a concurrent growth.
                raise SkillResourceTooLargeError(
                    "resource exceeds the remaining search scan budget"
                )
            scanned_files += 1
            scanned_bytes += len(payload)
            scan_hasher.update(relative.encode("utf-8"))
            scan_hasher.update(b"\x00")
            scan_hasher.update(resource_digest.encode("ascii"))

            text = payload.decode("utf-8")
            lines = text.splitlines()
            line_offset = start_line if file_index == start_file else 0
            if line_offset > len(lines):
                raise SkillResourceCursorError(
                    "search cursor is past the end of a resource"
                )

            for line_index in range(line_offset, len(lines)):
                line = lines[line_index]
                haystack = line if case_sensitive else line.casefold()
                column = haystack.find(needle)
                if column < 0:
                    continue
                if len(matches) >= result_limit:
                    next_position = (file_index, line_index)
                    break
                snippet, snippet_truncated = _bounded_match_text(
                    line, column, self._max_match_chars
                )
                matches.append(
                    SkillResourceMatch(
                        path=relative,
                        kind=_resource_kind(relative),
                        line_number=line_index + 1,
                        column=column + 1,
                        text=snippet,
                        digest=resource_digest,
                        text_truncated=snippet_truncated,
                    )
                )
            if next_position is not None:
                break
            start_line = 0

        next_cursor = (
            _encode_search_cursor(*next_position, scope)
            if next_position is not None
            else None
        )
        return SkillResourceSearchResult(
            query=query,
            matches=tuple(matches),
            cursor=original_cursor,
            next_cursor=next_cursor,
            scanned_files=scanned_files,
            scanned_bytes=scanned_bytes,
            digest=f"sha256:{scan_hasher.hexdigest()}",
            truncated=next_cursor is not None,
        )

    async def asearch(
        self,
        query: str,
        *,
        path: str | os.PathLike[str] | None = None,
        cursor: str | int | None = None,
        max_results: int | None = None,
        max_scan_bytes: int | None = None,
        case_sensitive: bool = False,
    ) -> SkillResourceSearchResult:
        """Asynchronous wrapper that keeps file IO off the event loop."""

        operation = partial(
            self.search,
            query,
            path=path,
            cursor=cursor,
            max_results=max_results,
            max_scan_bytes=max_scan_bytes,
            case_sensitive=case_sensitive,
        )
        return await _run_sync_in_daemon(operation)

    def _search_candidates(self, path: str | os.PathLike[str] | None) -> list[str]:
        if path is not None:
            relative = _validate_resource_path(path)
            descriptor = self._open_relative(relative, expected_directory=None)
            try:
                mode = os.fstat(descriptor).st_mode
            finally:
                os.close(descriptor)
            if stat.S_ISREG(mode):
                return [relative]
            if stat.S_ISDIR(mode):
                return self._walk_directory(relative)
            raise SkillResourceSecurityError(
                f"Skill resource must be a regular file or directory: {relative}"
            )

        candidates: list[str] = []
        for directory_name in sorted(_ALLOWED_DIRECTORIES):
            try:
                descriptor = self._open_relative(
                    directory_name, expected_directory=True
                )
            except SkillResourceNotFoundError:
                continue
            else:
                os.close(descriptor)
            candidates.extend(self._walk_directory(directory_name))
        return sorted(candidates)

    def _require_pinned(self, relative: str) -> None:
        if (
            self._expected_digests is not None
            and relative not in self._expected_digests
        ):
            raise SkillResourceChangedError(
                f"Skill resource is not part of the pinned package: {relative}"
            )

    def _validate_search_snapshot(
        self,
        path: str | os.PathLike[str] | None,
        candidates: list[str],
    ) -> None:
        if self._expected_digests is None:
            return
        if path is None:
            expected = set(self._expected_digests)
        else:
            relative = _validate_resource_path(path)
            expected = {
                candidate
                for candidate in self._expected_digests
                if candidate == relative or candidate.startswith(f"{relative}/")
            }
        if set(candidates) != expected:
            raise SkillResourceChangedError(
                "Skill resource paths changed after package activation"
            )

    def _walk_directory(self, relative_directory: str) -> list[str]:
        directory_descriptor = self._open_relative(
            relative_directory, expected_directory=True
        )
        files: list[str] = []

        def visit(descriptor: int, prefix: tuple[str, ...]) -> None:
            try:
                names = sorted(os.listdir(descriptor))
            except OSError as exc:
                raise SkillResourceSecurityError(
                    f"cannot enumerate Skill resource directory: {'/'.join(prefix)}"
                ) from exc
            for name in names:
                _validate_discovered_name(name)
                child_parts = (*prefix, name)
                relative = PurePosixPath(*child_parts).as_posix()
                try:
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    # Concurrent deletion is harmless. A later cursor snapshot
                    # will contain only resources that still exist.
                    continue
                except OSError as exc:
                    raise SkillResourceSecurityError(
                        f"cannot inspect Skill resource: {relative}"
                    ) from exc
                if stat.S_ISLNK(info.st_mode):
                    raise SkillResourceSecurityError(
                        f"symbolic links are not allowed in Skill resources: {relative}"
                    )
                if stat.S_ISDIR(info.st_mode):
                    child_descriptor = self._open_child(
                        descriptor,
                        name,
                        expected_directory=True,
                        display_path=relative,
                    )
                    try:
                        visit(child_descriptor, child_parts)
                    finally:
                        os.close(child_descriptor)
                elif stat.S_ISREG(info.st_mode):
                    child_descriptor = self._open_child(
                        descriptor,
                        name,
                        expected_directory=False,
                        display_path=relative,
                    )
                    os.close(child_descriptor)
                    files.append(relative)
                else:
                    raise SkillResourceSecurityError(
                        f"Skill resource must be a regular file: {relative}"
                    )

        try:
            visit(
                directory_descriptor,
                PurePosixPath(relative_directory).parts,
            )
        finally:
            os.close(directory_descriptor)
        return sorted(files)

    def _duplicate_root(self) -> int:
        descriptor = self._root_descriptor
        if descriptor < 0:
            raise SkillResourceSecurityError("Skill resource loader is closed")
        try:
            return os.dup(descriptor)
        except OSError as exc:
            raise SkillResourceSecurityError(
                "Skill resource root is no longer available"
            ) from exc

    def _open_relative(
        self,
        relative: str,
        *,
        expected_directory: bool | None,
    ) -> int:
        relative = _validate_resource_path(relative)
        parts = PurePosixPath(relative).parts
        descriptor = self._duplicate_root()
        try:
            for index, component in enumerate(parts):
                is_last = index == len(parts) - 1
                component_expected = expected_directory if is_last else True
                next_descriptor = self._open_child(
                    descriptor,
                    component,
                    expected_directory=component_expected,
                    display_path="/".join(parts[: index + 1]),
                )
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_child(
        parent_descriptor: int,
        component: str,
        *,
        expected_directory: bool | None,
        display_path: str,
    ) -> int:
        flags = (
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        )
        if expected_directory is True:
            flags |= os.O_DIRECTORY
        try:
            descriptor = os.open(component, flags, dir_fd=parent_descriptor)
        except FileNotFoundError as exc:
            raise SkillResourceNotFoundError(
                f"Skill resource does not exist: {display_path}"
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise SkillResourceSecurityError(
                    f"symbolic links are not allowed in Skill resources: {display_path}"
                ) from exc
            raise SkillResourceSecurityError(
                f"cannot open Skill resource safely: {display_path}"
            ) from exc

        mode = os.fstat(descriptor).st_mode
        valid = (
            stat.S_ISDIR(mode)
            if expected_directory is True
            else stat.S_ISREG(mode)
            if expected_directory is False
            else stat.S_ISDIR(mode) or stat.S_ISREG(mode)
        )
        if not valid:
            os.close(descriptor)
            expected = (
                "directory"
                if expected_directory is True
                else "regular file"
                if expected_directory is False
                else "regular file or directory"
            )
            raise SkillResourceSecurityError(
                f"Skill resource must be a {expected}: {display_path}"
            )
        return descriptor

    def _safe_size(self, relative: str) -> int:
        descriptor = self._open_relative(relative, expected_directory=False)
        try:
            size = os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
        if size > self._max_file_bytes:
            raise SkillResourceTooLargeError(
                f"Skill resource exceeds {self._max_file_bytes} bytes"
            )
        return size

    def _read_text(self, relative: str) -> tuple[bytes, str]:
        descriptor = self._open_relative(relative, expected_directory=False)
        try:
            info = os.fstat(descriptor)
            if info.st_size > self._max_file_bytes:
                raise SkillResourceTooLargeError(
                    f"Skill resource exceeds {self._max_file_bytes} bytes"
                )
            with os.fdopen(descriptor, "rb", closefd=True) as resource_file:
                descriptor = -1
                payload = resource_file.read(self._max_file_bytes + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        if len(payload) > self._max_file_bytes:
            raise SkillResourceTooLargeError(
                f"Skill resource exceeds {self._max_file_bytes} bytes"
            )
        if _looks_binary(payload):
            raise SkillResourceBinaryError("Skill resource must be UTF-8 text")
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillResourceBinaryError(
                "Skill resource must be valid UTF-8 text"
            ) from exc
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if self._expected_digests is not None:
            self._require_pinned(relative)
            _check_expected_digest(self._expected_digests[relative], digest)
        return payload, digest


def _validate_resource_path(path: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise TypeError("resource path must be a string or path-like value") from exc
    if not isinstance(raw, str):
        raise TypeError("resource path must resolve to a string")
    if not raw:
        raise SkillResourcePathError("resource path cannot be empty")
    if "\x00" in raw:
        raise SkillResourcePathError("resource path cannot contain NUL")
    if "\\" in raw:
        raise SkillResourcePathError("resource paths must use forward slashes")
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
        raise SkillResourcePathError("absolute resource paths are not allowed")
    raw_parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise SkillResourcePathError(
            "resource paths cannot contain empty, '.' or '..' components"
        )
    if raw_parts[0] not in _ALLOWED_DIRECTORIES:
        raise SkillResourcePathError(
            "only references/ and assets/ resources can be accessed"
        )
    return PurePosixPath(*raw_parts).as_posix()


def _resource_kind(relative: str) -> SkillResourceKind:
    if PurePosixPath(relative).parts[0] == "references":
        return SkillResourceKind.REFERENCE
    return SkillResourceKind.ASSET


def _validate_discovered_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "\x00" in name
        or "/" in name
        or "\\" in name
    ):
        raise SkillResourceSecurityError(
            "Skill resource contains a non-portable path component"
        )


def _looks_binary(payload: bytes) -> bool:
    if b"\x00" in payload:
        return True
    allowed_controls = {9, 10, 12, 13}
    return any(byte < 32 and byte not in allowed_controls for byte in payload)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SkillResourceCursorError(f"{name} must be a non-negative integer")
    return value


def _normalize_expected_digests(
    value: Mapping[str, str] | None,
) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("expected_digests must be a mapping or None")
    normalized: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        path = _validate_resource_path(raw_path)
        if not isinstance(raw_digest, str) or not raw_digest.strip():
            raise ValueError(f"expected digest must be non-empty: {path}")
        digest = raw_digest.strip().lower()
        if not digest.startswith("sha256:"):
            digest = f"sha256:{digest}"
        algorithm, _, hexadecimal = digest.partition(":")
        if (
            algorithm != "sha256"
            or len(hexadecimal) != 64
            or any(character not in "0123456789abcdef" for character in hexadecimal)
        ):
            raise ValueError(f"expected digest must be SHA-256: {path}")
        if path in normalized:
            raise ValueError(f"duplicate expected resource digest: {path}")
        normalized[path] = digest
    return MappingProxyType(normalized)


def _check_expected_digest(expected: str | None, actual: str) -> None:
    if expected is None:
        return
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("expected_digest must be a non-empty string")
    normalized = expected.strip().lower()
    if not normalized.startswith("sha256:"):
        normalized = f"sha256:{normalized}"
    if normalized != actual:
        raise SkillResourceChangedError(
            f"Skill resource digest mismatch: expected {normalized}, got {actual}"
        )


def _bounded_match_text(line: str, column: int, limit: int) -> tuple[str, bool]:
    if len(line) <= limit:
        return line, False
    start = max(0, column - limit // 3)
    end = min(len(line), start + limit)
    start = max(0, end - limit)
    return line[start:end], True


def _search_scope(
    query: str,
    path: str | os.PathLike[str] | None,
    case_sensitive: bool,
    candidates: list[str],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(query.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(str(case_sensitive).encode("ascii"))
    hasher.update(b"\x00")
    if path is not None:
        hasher.update(os.fspath(path).encode("utf-8"))
    for relative in candidates:
        hasher.update(b"\x00")
        hasher.update(relative.encode("utf-8"))
    return hasher.hexdigest()


def _encode_search_cursor(file_index: int, line_index: int, scope: str) -> str:
    value = json.dumps(
        {"v": _CURSOR_VERSION, "f": file_index, "l": line_index, "s": scope},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_search_cursor(
    cursor: str | int | None, scope: str
) -> tuple[int, int, str | None]:
    if cursor is None or cursor == 0:
        return 0, 0, None
    if not isinstance(cursor, str) or not cursor:
        raise SkillResourceCursorError("search cursor must be an opaque string")
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.b64decode(cursor + padding, altchars=b"-_", validate=True).decode(
            "utf-8"
        )
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        if value.get("v") != _CURSOR_VERSION or value.get("s") != scope:
            raise ValueError
        file_index = _non_negative_int(value.get("f"), "cursor file index")
        line_index = _non_negative_int(value.get("l"), "cursor line index")
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillResourceCursorError(
            "search cursor is invalid or belongs to a different search"
        ) from exc
    return file_index, line_index, cursor


T = TypeVar("T")


async def _run_sync_in_daemon(function: Callable[[], T]) -> T:
    """Run bounded file IO without relying on the process default executor."""

    completed = Event()
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            completed.set()

    Thread(target=run, daemon=True).start()
    while not completed.is_set():
        await asyncio.sleep(0.001)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
