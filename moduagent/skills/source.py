from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from moduagent.skills.errors import (
    SkillDigestMismatchError,
    SkillLimitError,
    SkillNotFoundError,
    SkillValidationError,
)
from moduagent.skills.models import (
    SkillArtifact,
    SkillDescriptor,
    SkillLimits,
    SkillRef,
)
from moduagent.skills.validation import validate_relative_path, validate_skill_package


_FILESYSTEM_SOURCE_NAMESPACE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")


@runtime_checkable
class SkillSource(Protocol):
    """Source of content-addressed skill packages."""

    def discover(self) -> tuple[SkillDescriptor, ...]: ...

    def load(self, ref: SkillRef) -> SkillArtifact: ...


@runtime_checkable
class ResourceReadableSkillSource(SkillSource, Protocol):
    """A source that can safely return active reference and asset bytes."""

    def read_resource(
        self,
        ref: SkillRef,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes: ...


class FilesystemSkillSource:
    """Discover skills from a directory without executing packaged scripts."""

    def __init__(
        self,
        root: str | Path,
        *,
        source_id: str | None = None,
        strict: bool = True,
        limits: SkillLimits | None = None,
    ) -> None:
        self._root_descriptor = -1
        if (
            not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
            or os.listdir not in os.supports_fd
        ):
            raise SkillValidationError(
                "secure filesystem Skills require openat, O_NOFOLLOW, and fd listing"
            )
        raw_root = Path(root).expanduser()
        if raw_root.is_symlink():
            raise SkillValidationError("skill source root cannot be a symbolic link")
        if not raw_root.exists():
            raise SkillValidationError(f"skill source root does not exist: {raw_root}")
        if not raw_root.is_dir():
            raise SkillValidationError(
                f"skill source root is not a directory: {raw_root}"
            )
        self.root = raw_root.absolute()
        try:
            root_descriptor = os.open(
                self.root,
                os.O_RDONLY
                | os.O_CLOEXEC
                | os.O_NOFOLLOW
                | os.O_DIRECTORY
                | getattr(os, "O_NONBLOCK", 0),
            )
        except OSError as exc:
            raise SkillValidationError(
                "skill source root cannot be opened safely"
            ) from exc
        if not stat.S_ISDIR(os.fstat(root_descriptor).st_mode):
            os.close(root_descriptor)
            raise SkillValidationError("skill source root is not a directory")
        self._root_descriptor = root_descriptor
        self.source_id = _normalize_filesystem_source_id(source_id)
        self.strict = strict
        self.limits = limits or SkillLimits()
        self._paths: Mapping[str, Path] = MappingProxyType({})

    def close(self) -> None:
        descriptor = self._root_descriptor
        if descriptor >= 0:
            self._root_descriptor = -1
            os.close(descriptor)

    def __enter__(self) -> "FilesystemSkillSource":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass

    def discover(self) -> tuple[SkillDescriptor, ...]:
        artifacts: list[tuple[SkillArtifact, Path]] = []
        for skill_dir in self._candidate_directories():
            package = self._snapshot(skill_dir)
            artifact = validate_skill_package(
                package,
                source_id=self._descriptor_source_id(skill_dir.name),
                expected_name=skill_dir.name,
                strict=self.strict,
                limits=self.limits,
            )
            artifacts.append((artifact, skill_dir))

        paths: dict[str, Path] = {}
        descriptors: list[SkillDescriptor] = []
        for artifact, path in sorted(
            artifacts, key=lambda item: item[0].descriptor.name
        ):
            name = artifact.descriptor.name
            if name in paths:
                raise SkillValidationError(f"duplicate skill in source: {name}")
            paths[name] = path
            descriptors.append(artifact.descriptor)
        self._paths = MappingProxyType(paths)
        return tuple(descriptors)

    def load(self, ref: SkillRef) -> SkillArtifact:
        path = self._path_for(ref.name)
        artifact = validate_skill_package(
            self._snapshot(path),
            source_id=self._descriptor_source_id(path.name),
            expected_name=path.name,
            strict=self.strict,
            limits=self.limits,
        )
        _verify_ref(ref, artifact.descriptor)
        return artifact

    def path_for(self, ref: SkillRef) -> Path:
        """Return the pinned package root for a trusted resource loader.

        The complete package is validated before the path is exposed. Callers
        must still restrict reads to ``references/`` and ``assets/``.
        """

        path, _ = self.resource_snapshot(ref)
        return path

    def resource_snapshot(self, ref: SkillRef) -> tuple[Path, Mapping[str, str]]:
        """Pin the allowed resource paths and content digests for one revision."""

        path = self._path_for(ref.name)
        package = self._snapshot(path)
        artifact = validate_skill_package(
            package,
            source_id=self._descriptor_source_id(path.name),
            expected_name=path.name,
            strict=self.strict,
            limits=self.limits,
        )
        _verify_ref(ref, artifact.descriptor)
        resource_paths = (*artifact.references, *artifact.assets)
        digests = {
            resource_path: (
                f"sha256:{hashlib.sha256(package[resource_path]).hexdigest()}"
            )
            for resource_path in resource_paths
        }
        return path, MappingProxyType(digests)

    def read_resource(
        self,
        ref: SkillRef,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        normalized_path = validate_relative_path(path)
        skill_dir = self._path_for(ref.name)
        package = self._snapshot(skill_dir)
        artifact = validate_skill_package(
            package,
            source_id=self._descriptor_source_id(skill_dir.name),
            expected_name=skill_dir.name,
            strict=self.strict,
            limits=self.limits,
        )
        _verify_ref(ref, artifact.descriptor)
        if normalized_path not in {*artifact.references, *artifact.assets}:
            raise SkillNotFoundError(
                f"skill resource is not an active reference or asset: {normalized_path}"
            )
        limit = (
            self.limits.max_resource_bytes_per_read if max_bytes is None else max_bytes
        )
        if limit <= 0:
            raise ValueError("max_bytes must be greater than zero")
        content = package[normalized_path]
        if len(content) > limit:
            raise SkillLimitError(
                f"skill resource exceeds read limit ({limit} bytes): {normalized_path}"
            )
        return content

    def _candidate_directories(self) -> tuple[Path, ...]:
        root_descriptor = self._duplicate_root()
        try:
            if self._has_skill_document(root_descriptor, self.root.as_posix()):
                return (self.root,)
            try:
                names = sorted(os.listdir(root_descriptor))
            except OSError as exc:
                raise SkillValidationError(
                    "cannot enumerate skill source root"
                ) from exc
            candidates: list[Path] = []
            for name in names:
                _validate_filesystem_component(name)
                display = (self.root / name).as_posix()
                try:
                    info = os.stat(
                        name,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise SkillValidationError(
                        f"cannot inspect skill source entry: {display}"
                    ) from exc
                if stat.S_ISLNK(info.st_mode):
                    raise SkillValidationError(
                        f"symbolic links are not allowed in skill sources: {display}"
                    )
                if not stat.S_ISDIR(info.st_mode):
                    continue
                child_descriptor = self._open_child(
                    root_descriptor,
                    name,
                    directory=True,
                    display_path=display,
                )
                try:
                    if self._has_skill_document(child_descriptor, display):
                        candidates.append(self.root / name)
                finally:
                    os.close(child_descriptor)
            return tuple(candidates)
        finally:
            os.close(root_descriptor)

    def _path_for(self, name: str) -> Path:
        if not self._paths:
            self.discover()
        path = self._paths.get(name)
        if path is None:
            raise SkillNotFoundError(f"unknown skill: {name}")
        return path

    def _descriptor_source_id(self, skill_name: str) -> str:
        if self.source_id is None:
            return f"filesystem://{skill_name}"
        return f"filesystem://{self.source_id}/{skill_name}"

    def _snapshot(self, skill_dir: Path) -> dict[str, bytes]:
        try:
            relative_root = skill_dir.absolute().relative_to(self.root)
        except ValueError as exc:
            raise SkillValidationError(
                "skill directory is outside the source root"
            ) from exc
        directory_descriptor = self._duplicate_root()
        try:
            for index, component in enumerate(relative_root.parts):
                next_descriptor = self._open_child(
                    directory_descriptor,
                    component,
                    directory=True,
                    display_path="/".join(relative_root.parts[: index + 1]),
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
        except BaseException:
            os.close(directory_descriptor)
            raise

        package: dict[str, bytes] = {}
        total_bytes = 0

        def visit(descriptor: int, prefix: tuple[str, ...]) -> None:
            nonlocal total_bytes
            try:
                names = sorted(os.listdir(descriptor))
            except OSError as exc:
                raise SkillValidationError(
                    f"cannot enumerate skill package: {'/'.join(prefix) or '.'}"
                ) from exc
            for name in names:
                _validate_filesystem_component(name)
                parts = (*prefix, name)
                relative = PurePosixPath(*parts).as_posix()
                validate_relative_path(relative)
                try:
                    info = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise SkillValidationError(
                        f"cannot inspect skill package entry: {relative}"
                    ) from exc
                if stat.S_ISLNK(info.st_mode):
                    raise SkillValidationError(
                        f"symbolic links are not allowed in skill packages: {relative}"
                    )
                if stat.S_ISDIR(info.st_mode):
                    child_descriptor = self._open_child(
                        descriptor,
                        name,
                        directory=True,
                        display_path=relative,
                    )
                    try:
                        visit(child_descriptor, parts)
                    finally:
                        os.close(child_descriptor)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise SkillValidationError(
                        f"unsupported skill package entry: {relative}"
                    )
                if len(package) >= self.limits.max_package_files:
                    raise SkillValidationError(
                        "skill package exceeds max_package_files "
                        f"({self.limits.max_package_files})"
                    )
                file_descriptor = self._open_child(
                    descriptor,
                    name,
                    directory=False,
                    display_path=relative,
                )
                try:
                    opened_info = os.fstat(file_descriptor)
                    remaining = self.limits.max_package_bytes - total_bytes
                    is_resource = relative.split("/", 1)[0] in {
                        "references",
                        "assets",
                    }
                    read_limit = remaining
                    if is_resource:
                        read_limit = min(
                            read_limit,
                            self.limits.max_resource_file_bytes,
                        )
                    if opened_info.st_size < 0 or opened_info.st_size > remaining:
                        self._raise_package_limit()
                    if (
                        is_resource
                        and opened_info.st_size > self.limits.max_resource_file_bytes
                    ):
                        self._raise_resource_file_limit(relative)
                    with os.fdopen(
                        file_descriptor, "rb", closefd=True
                    ) as resource_file:
                        file_descriptor = -1
                        content = resource_file.read(read_limit + 1)
                finally:
                    if file_descriptor >= 0:
                        os.close(file_descriptor)
                if len(content) > remaining:
                    self._raise_package_limit()
                if is_resource and len(content) > self.limits.max_resource_file_bytes:
                    self._raise_resource_file_limit(relative)
                package[relative] = content
                total_bytes += len(content)

        try:
            visit(directory_descriptor, ())
        finally:
            os.close(directory_descriptor)
        return package

    def _duplicate_root(self) -> int:
        descriptor = self._root_descriptor
        if descriptor < 0:
            raise SkillValidationError("skill source is closed")
        try:
            return os.dup(descriptor)
        except OSError as exc:
            raise SkillValidationError("skill source root is unavailable") from exc

    @staticmethod
    def _open_child(
        parent_descriptor: int,
        component: str,
        *,
        directory: bool,
        display_path: str,
    ) -> int:
        flags = (
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        )
        if directory:
            flags |= os.O_DIRECTORY
        try:
            descriptor = os.open(component, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise SkillValidationError(
                f"cannot safely open skill package entry: {display_path}"
            ) from exc
        mode = os.fstat(descriptor).st_mode
        valid = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
        if not valid:
            os.close(descriptor)
            raise SkillValidationError(
                f"unsupported skill package entry: {display_path}"
            )
        return descriptor

    @classmethod
    def _has_skill_document(cls, descriptor: int, display_path: str) -> bool:
        try:
            info = os.stat(
                "SKILL.md",
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise SkillValidationError(
                f"cannot inspect SKILL.md under {display_path}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise SkillValidationError(
                f"SKILL.md cannot be a symbolic link: {display_path}"
            )
        return stat.S_ISREG(info.st_mode)

    def _raise_resource_file_limit(self, relative: str) -> None:
        raise SkillValidationError(
            "skill resource exceeds max_resource_file_bytes "
            f"({self.limits.max_resource_file_bytes}): {relative}"
        )

    def _raise_package_limit(self) -> None:
        raise SkillValidationError(
            f"skill package exceeds max_package_bytes ({self.limits.max_package_bytes})"
        )


SkillPackage = str | bytes | Mapping[str, str | bytes]


class InMemorySkillSource:
    """Immutable in-process source, useful for embedded skills and tests."""

    def __init__(
        self,
        skills: Mapping[str, SkillPackage],
        *,
        source_id: str = "default",
        strict: bool = True,
        limits: SkillLimits | None = None,
    ) -> None:
        if not source_id.strip():
            raise ValueError("source_id cannot be empty")
        self.source_id = source_id.strip()
        self.strict = strict
        self.limits = limits or SkillLimits()

        artifacts: dict[str, SkillArtifact] = {}
        packages: dict[str, Mapping[str, bytes]] = {}
        for expected_name, raw_package in skills.items():
            package = _coerce_memory_package(raw_package)
            skill_source_id = f"memory://{self.source_id}/{expected_name}"
            artifact = validate_skill_package(
                package,
                source_id=skill_source_id,
                expected_name=expected_name,
                strict=self.strict,
                limits=self.limits,
            )
            if artifact.descriptor.name in artifacts:
                raise SkillValidationError(
                    f"duplicate skill in source: {artifact.descriptor.name}"
                )
            artifacts[artifact.descriptor.name] = artifact
            packages[artifact.descriptor.name] = MappingProxyType(dict(package))

        self._artifacts: Mapping[str, SkillArtifact] = MappingProxyType(artifacts)
        self._packages: Mapping[str, Mapping[str, bytes]] = MappingProxyType(packages)

    def discover(self) -> tuple[SkillDescriptor, ...]:
        return tuple(
            self._artifacts[name].descriptor for name in sorted(self._artifacts)
        )

    def load(self, ref: SkillRef) -> SkillArtifact:
        artifact = self._artifacts.get(ref.name)
        if artifact is None:
            raise SkillNotFoundError(f"unknown skill: {ref.name}")
        _verify_ref(ref, artifact.descriptor)
        return artifact

    def read_resource(
        self,
        ref: SkillRef,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        artifact = self.load(ref)
        normalized_path = validate_relative_path(path)
        if normalized_path not in {*artifact.references, *artifact.assets}:
            raise SkillNotFoundError(
                f"skill resource is not an active reference or asset: {normalized_path}"
            )
        content = self._packages[ref.name][normalized_path]
        limit = (
            self.limits.max_resource_bytes_per_read if max_bytes is None else max_bytes
        )
        if limit <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if len(content) > limit:
            raise SkillLimitError(
                f"skill resource exceeds read limit ({limit} bytes): {normalized_path}"
            )
        return bytes(content)


def _coerce_memory_package(value: SkillPackage) -> dict[str, bytes]:
    if isinstance(value, str):
        return {"SKILL.md": value.encode("utf-8")}
    if isinstance(value, bytes):
        return {"SKILL.md": bytes(value)}
    if not isinstance(value, Mapping):
        raise SkillValidationError("in-memory skill must be Markdown or a file mapping")

    package: dict[str, bytes] = {}
    for path, content in value.items():
        normalized = validate_relative_path(path)
        if isinstance(content, str):
            package[normalized] = content.encode("utf-8")
        elif isinstance(content, bytes):
            package[normalized] = bytes(content)
        else:
            raise SkillValidationError(
                f"in-memory skill file must be bytes or text: {normalized}"
            )
    return package


def _verify_ref(ref: SkillRef, descriptor: SkillDescriptor) -> None:
    if ref.source_id != descriptor.source_id:
        raise SkillDigestMismatchError(
            f"skill source changed for {ref.name}: {ref.source_id!r}"
        )
    if ref.digest != descriptor.digest:
        raise SkillDigestMismatchError(f"skill content changed for {ref.name}")
    if ref.version != descriptor.version:
        raise SkillDigestMismatchError(f"skill version changed for {ref.name}")


def _normalize_filesystem_source_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("source_id must be a string or None")
    normalized = value.strip()
    prefix = "filesystem://"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    normalized = normalized.strip("/")
    if not normalized or not _FILESYSTEM_SOURCE_NAMESPACE.fullmatch(normalized):
        raise ValueError("source_id must contain URI-safe logical namespace components")
    return normalized


def _validate_filesystem_component(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
    ):
        raise SkillValidationError(
            "skill package contains a non-portable path component"
        )
