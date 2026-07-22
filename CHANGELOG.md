# Changelog

## 0.2.0

- Added official `SKILL.md` compatible Agent Skills packages.
- Added filesystem and in-memory Skill sources with immutable package and catalog digests.
- Added explicit, model-based, and hybrid Skill selection.
- Applied active Skill instructions consistently to PLAN, ACT, and FINALIZE phases.
- Added bounded `references/` and text `assets/` read/search tools.
- Pinned filesystem resource paths and SHA-256 digests, with POSIX `openat`/`O_NOFOLLOW` traversal protection against path races.
- Added Skill-specific catalog, instruction, resource-read, byte, and token limits.
- Added mount-independent filesystem source IDs and enforceable catalog lock files.
- Restricted active runs to the intersection of registered and Skill-declared tools; `ToolAuthorizer` is still enforced at execution.
- Added Skill lifecycle events without logging instruction or resource contents.
- Upgraded checkpoints to schema v2 with v1 read compatibility and pinned Skill state.
- Added `moduagent skills init|validate|inspect|lock`.
- Added PyYAML as a runtime dependency.
- Added Python 3.10-3.13 CI coverage.
- Bundled scripts remain non-executable in this release.

## 0.1.1

- Initial public release of the ModuAgent runtime.
