"""Bounded and deterministic filesystem discovery for the RAG example."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .models import ScanError, SourceDocument, stable_digest


DEFAULT_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".md",
        ".txt",
        ".csv",
        ".adoc",
        ".asciidoc",
        ".asc",
        ".vtt",
        ".tex",
        ".eml",
        ".epub",
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }
)
_KB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ScanPolicy:
    """Application-owned limits; none of these are model-controlled."""

    allowed_extensions: frozenset[str] = DEFAULT_EXTENSIONS
    max_files: int = 1_000
    max_file_bytes: int = 100 * 1024 * 1024
    max_total_bytes: int = 2 * 1024 * 1024 * 1024
    max_depth: int = 32
    reject_unsupported: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_extensions:
            raise ValueError("allowed_extensions cannot be empty")
        normalized = frozenset(value.lower() for value in self.allowed_extensions)
        if any(not value.startswith(".") or len(value) > 16 for value in normalized):
            raise ValueError("allowed extensions must be bounded dot suffixes")
        if type(self.max_files) is not int or self.max_files < 1:
            raise ValueError("max_files must be positive")
        if type(self.max_file_bytes) is not int or self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        if type(self.max_total_bytes) is not int or self.max_total_bytes < 1:
            raise ValueError("max_total_bytes must be positive")
        if type(self.max_depth) is not int or not 0 <= self.max_depth <= 256:
            raise ValueError("max_depth must be between 0 and 256")
        if type(self.reject_unsupported) is not bool:
            raise TypeError("reject_unsupported must be a bool")
        object.__setattr__(self, "allowed_extensions", normalized)


def scan_document_directory(
    root: str | os.PathLike[str],
    *,
    kb_id: str = "corporate-assistant",
    policy: ScanPolicy | None = None,
) -> tuple[SourceDocument, ...]:
    """Scan regular documents below ``root`` in normalized path order.

    Symbolic links and duplicate inodes are rejected rather than followed.
    The defaults mirror ordinary Docling Serve input formats. Generic JSON and
    XML are deliberately excluded: Docling accepts only its own JSON schema and
    specific XML dialects, which cannot be identified safely by suffix alone.
    Unsupported regular files are ignored by default, allowing a source tree to
    contain README or application files that are not part of the corpus.
    """

    resolved_policy = policy or ScanPolicy()
    if not isinstance(resolved_policy, ScanPolicy):
        raise TypeError("policy must be a ScanPolicy")
    if not isinstance(kb_id, str) or _KB_ID.fullmatch(kb_id) is None:
        raise ScanError("kb_id must be a safe bounded identifier")

    raw_root = Path(root).expanduser()
    canonical_root, root_descriptor = _open_absolute_directory(raw_root)
    candidates: list[tuple[str, Path, os.stat_result, str]] = []
    seen_paths: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    seen_ids: set[str] = set()
    total_bytes = 0
    try:
        if not stat.S_ISDIR(os.fstat(root_descriptor).st_mode):
            raise ScanError("document root must be a directory")
        for current, directory_names, file_names, current_descriptor in os.fwalk(
            ".",
            topdown=True,
            onerror=_raise_walk_error,
            follow_symlinks=False,
            dir_fd=root_descriptor,
        ):
            directory_parts = _relative_directory_parts(current)
            depth = len(directory_parts)
            directory_names.sort()
            file_names.sort()
            if depth >= resolved_policy.max_depth:
                if directory_names:
                    raise ScanError("document directory exceeds the depth limit")
                directory_names[:] = []

            for directory_name in directory_names:
                info = os.stat(
                    directory_name,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(info.st_mode):
                    raise ScanError(
                        "symbolic links are not allowed in the document tree"
                    )
                if not stat.S_ISDIR(info.st_mode):
                    raise ScanError("document tree contains a non-directory entry")

            for filename in file_names:
                info = os.stat(
                    filename,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(info.st_mode):
                    raise ScanError(
                        "symbolic links are not allowed in the document tree"
                    )
                if not stat.S_ISREG(info.st_mode):
                    raise ScanError("document tree contains a non-regular file")
                if any(ord(character) < 32 for character in filename):
                    raise ScanError("document filename contains control characters")
                extension = Path(filename).suffix.lower()
                if extension not in resolved_policy.allowed_extensions:
                    if resolved_policy.reject_unsupported:
                        raise ScanError(f"unsupported document extension: {extension}")
                    continue
                relative_path = _normalized_relative_parts((*directory_parts, filename))
                if relative_path in seen_paths:
                    raise ScanError("normalized document paths collide")
                if not 1 <= info.st_size <= resolved_policy.max_file_bytes:
                    raise ScanError("document size exceeds the per-file limit")
                sha256, final_info = _hash_regular_file_at(
                    current_descriptor,
                    filename,
                    expected=info,
                    maximum=resolved_policy.max_file_bytes,
                )
                identity = (final_info.st_dev, final_info.st_ino)
                if identity in seen_inodes:
                    raise ScanError("hard-linked duplicate documents are not allowed")
                total_bytes += final_info.st_size
                if total_bytes > resolved_policy.max_total_bytes:
                    raise ScanError("document size exceeds the batch limit")
                source_id = "src_" + stable_digest(kb_id, relative_path)[:32]
                if source_id in seen_ids:
                    raise ScanError("source ID collision")
                path = canonical_root.joinpath(*relative_path.split("/"))
                candidates.append((relative_path, path, final_info, sha256))
                if len(candidates) > resolved_policy.max_files:
                    raise ScanError(
                        f"document count exceeds the limit of {resolved_policy.max_files}"
                    )
                seen_paths.add(relative_path)
                seen_inodes.add(identity)
                seen_ids.add(source_id)
    except ScanError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ScanError("document directory changed or could not be scanned") from exc
    finally:
        os.close(root_descriptor)

    result: list[SourceDocument] = []
    for relative_path, path, final_info, sha256 in sorted(candidates):
        result.append(
            SourceDocument(
                kb_id=kb_id,
                source_id="src_" + stable_digest(kb_id, relative_path)[:32],
                root=canonical_root,
                path=path,
                relative_path=relative_path,
                media_type=mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                size_bytes=final_info.st_size,
                mtime_ns=final_info.st_mtime_ns,
                sha256=sha256,
                device=final_info.st_dev,
                inode=final_info.st_ino,
            )
        )
    return tuple(result)


def read_source_bytes(
    source: SourceDocument,
    *,
    policy: ScanPolicy | None = None,
) -> bytes:
    """Recheck a scanned file and return bytes only if it is unchanged."""

    if not isinstance(source, SourceDocument):
        raise TypeError("source must be a SourceDocument")
    resolved_policy = policy or ScanPolicy()
    if source.size_bytes > resolved_policy.max_file_bytes:
        raise ScanError("document exceeds the active read policy")
    try:
        descriptor = _open_source_descriptor(source)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            _verify_source_identity(source, before)
            content = stream.read(resolved_policy.max_file_bytes + 1)
            after = os.fstat(stream.fileno())
    except ScanError:
        raise
    except OSError as exc:
        raise ScanError("source could not be rechecked before processing") from exc
    _verify_source_identity(source, after)
    if len(content) > resolved_policy.max_file_bytes:
        raise ScanError("document size exceeds the per-file limit")
    if len(content) != source.size_bytes:
        raise ScanError("source changed after directory scan")
    if hashlib.sha256(content).hexdigest() != source.sha256:
        raise ScanError("source changed after directory scan")
    return content


def _open_absolute_directory(path: Path) -> tuple[Path, int]:
    """Open every absolute path component without following a symbolic link."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ScanError("safe directory traversal is unsupported on this platform")
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise ScanError("document root is not accessible") from exc
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(os.sep, flags)
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ScanError("document root contains an unsafe component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return absolute, descriptor
    except ScanError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ScanError(
            "document root is inaccessible or contains a symbolic link"
        ) from exc


def _open_source_descriptor(source: SourceDocument) -> int:
    canonical_root, descriptor = _open_absolute_directory(source.root)
    if canonical_root != source.root:
        os.close(descriptor)
        raise ScanError("source root identity is invalid")
    directory_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    parts = tuple(source.relative_path.split("/"))
    try:
        for component in parts[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        source_descriptor = os.open(parts[-1], file_flags, dir_fd=descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise ScanError("source path changed after directory scan") from exc
    os.close(descriptor)
    return source_descriptor


def _relative_directory_parts(current: str) -> tuple[str, ...]:
    path = Path(current)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ScanError("document walk escaped its configured root")
    return tuple(part for part in path.parts if part not in {"", "."})


def _raise_walk_error(error: OSError) -> None:
    raise ScanError("document directory changed during traversal") from error


def _normalized_relative_parts(parts: tuple[str, ...]) -> str:
    normalized = unicodedata.normalize("NFC", "/".join(parts))
    if not normalized or normalized.startswith("/") or "\\" in normalized:
        raise ScanError("document path is not a normalized relative path")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ScanError("document path contains an unsafe component")
    if any(ord(character) < 32 for character in normalized):
        raise ScanError("document path contains control characters")
    if len(normalized.encode("utf-8")) > 4_096:
        raise ScanError("document path exceeds its length limit")
    return normalized


def _hash_regular_file_at(
    directory_descriptor: int,
    filename: str,
    *,
    expected: os.stat_result,
    maximum: int,
) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    total = 0
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(filename, flags, dir_fd=directory_descriptor)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ScanError("document must remain a regular file")
            if (before.st_dev, before.st_ino) != (expected.st_dev, expected.st_ino):
                raise ScanError("document changed during directory scan")
            while block := stream.read(1024 * 1024):
                total += len(block)
                if total > maximum:
                    raise ScanError("document size exceeds the per-file limit")
                digest.update(block)
            after = os.fstat(stream.fileno())
    except ScanError:
        raise
    except OSError as exc:
        raise ScanError("document could not be read during directory scan") from exc
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        raise ScanError("document changed during directory scan")
    if (
        after.st_size != total
        or before.st_size != total
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise ScanError("document changed during directory scan")
    return digest.hexdigest(), after


def _verify_source_identity(source: SourceDocument, info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ScanError("source must remain a regular file")
    if (info.st_dev, info.st_ino) != (source.device, source.inode):
        raise ScanError("source changed after directory scan")
    if info.st_size != source.size_bytes or info.st_mtime_ns != source.mtime_ns:
        raise ScanError("source changed after directory scan")
