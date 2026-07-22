class SkillError(Exception):
    """Base class for all skill subsystem failures."""


class SkillValidationError(SkillError, ValueError):
    """A skill package or model selection is malformed."""


class SkillNotFoundError(SkillError, LookupError):
    """A requested skill does not exist in the catalog snapshot."""


class SkillDigestMismatchError(SkillError):
    """The skill content no longer matches its pinned reference."""


class SkillSelectionError(SkillError):
    """A skill selector returned an invalid selection."""


class SkillLimitError(SkillError):
    """A configured skill limit would be exceeded."""
