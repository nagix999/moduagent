"""Small, dependency-free ``.env`` loader for the runnable example."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path


MAX_ENV_FILE_BYTES = 64 * 1024
MAX_ENV_VALUE_CHARS = 8_192
MAX_SECRET_FILE_BYTES = 8_192
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvironmentFileError(ValueError):
    """A local environment file is missing, unsafe, or malformed."""


def environment_secret(name: str) -> str | None:
    """Resolve ``NAME`` or safely read ``NAME_FILE`` without logging either."""

    if not isinstance(name, str) or _ENV_NAME.fullmatch(name) is None:
        raise ValueError("secret environment name is invalid")
    direct = os.getenv(name)
    path = os.getenv(f"{name}_FILE")
    if direct is not None and path is not None:
        raise EnvironmentFileError(f"configure only one of {name} and {name}_FILE")
    if direct is not None:
        return _secret_value(direct)
    if path is None:
        return None
    raw_path = Path(path).expanduser()
    try:
        descriptor = os.open(
            raw_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise EnvironmentFileError(f"{name}_FILE could not be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SECRET_FILE_BYTES:
            raise EnvironmentFileError(f"{name}_FILE must be a bounded regular file")
        raw = os.read(descriptor, MAX_SECRET_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_SECRET_FILE_BYTES:
        raise EnvironmentFileError(f"{name}_FILE exceeds its size limit")
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise EnvironmentFileError(f"{name}_FILE must contain UTF-8 text") from exc
    return _secret_value(value)


def _secret_value(value: str) -> str:
    if (
        not value
        or len(value) > MAX_ENV_VALUE_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EnvironmentFileError("secret value is empty or invalid")
    return value


def load_environment_file(
    path: str | os.PathLike[str] = ".env",
    *,
    required: bool = False,
    override: bool = False,
) -> dict[str, str]:
    """Load literal key/value pairs without interpolation or command execution.

    Existing process variables win unless ``override`` is explicitly true.  The
    parser accepts ``KEY=value``, optional ``export``, and single/double quoted
    values.  Dollar expressions stay literal; this function never evaluates a
    shell expression.
    """

    if type(required) is not bool or type(override) is not bool:
        raise TypeError("required and override must be bool values")
    raw_path = Path(path).expanduser()
    try:
        descriptor = os.open(
            raw_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        if required:
            raise EnvironmentFileError(f"environment file not found: {raw_path}")
        return {}
    except OSError as exc:
        raise EnvironmentFileError(
            "environment file could not be opened safely"
        ) from exc

    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise EnvironmentFileError("environment file must be a regular file")
        if info.st_size > MAX_ENV_FILE_BYTES:
            raise EnvironmentFileError("environment file exceeds 64 KiB")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(16_384, MAX_ENV_FILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_ENV_FILE_BYTES:
                raise EnvironmentFileError("environment file exceeds 64 KiB")
    finally:
        os.close(descriptor)

    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvironmentFileError("environment file must be UTF-8") from exc

    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise EnvironmentFileError(
                f"environment line {line_number} must contain '='"
            )
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if _ENV_NAME.fullmatch(name) is None:
            raise EnvironmentFileError(
                f"environment line {line_number} has an invalid variable name"
            )
        if name in parsed:
            raise EnvironmentFileError(
                f"environment line {line_number} duplicates {name}"
            )
        value = _parse_value(raw_value, line_number)
        if len(value) > MAX_ENV_VALUE_CHARS:
            raise EnvironmentFileError(
                f"environment line {line_number} has an oversized value"
            )
        if any(ord(character) < 32 and character not in "\t" for character in value):
            raise EnvironmentFileError(
                f"environment line {line_number} contains control characters"
            )
        parsed[name] = value

    for name, value in parsed.items():
        if override or name not in os.environ:
            os.environ[name] = value
    return parsed


def _parse_value(raw_value: str, line_number: int) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] not in {"'", '"'}:
        marker = re.search(r"\s+#", value)
        return (value[: marker.start()] if marker else value).rstrip()

    quote = value[0]
    characters: list[str] = []
    escaped = False
    closing_index: int | None = None
    for index, character in enumerate(value[1:], start=1):
        if quote == '"' and escaped:
            characters.append(
                {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(
                    character,
                    "\\" + character,
                )
            )
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if character == quote:
            closing_index = index
            break
        characters.append(character)
    if escaped or closing_index is None:
        raise EnvironmentFileError(
            f"environment line {line_number} has an unterminated quote"
        )
    remainder = value[closing_index + 1 :].strip()
    if remainder and not remainder.startswith("#"):
        raise EnvironmentFileError(
            f"environment line {line_number} has trailing content"
        )
    return "".join(characters)


__all__ = [
    "EnvironmentFileError",
    "MAX_ENV_FILE_BYTES",
    "MAX_ENV_VALUE_CHARS",
    "MAX_SECRET_FILE_BYTES",
    "environment_secret",
    "load_environment_file",
]
