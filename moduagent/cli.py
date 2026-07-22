from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from moduagent.skills import SkillError, SkillRegistry, validate_skill_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moduagent")
    commands = parser.add_subparsers(dest="command", required=True)
    skills = commands.add_parser("skills", help="Create and validate Agent Skills")
    skill_commands = skills.add_subparsers(dest="skill_command", required=True)

    init = skill_commands.add_parser("init", help="Create a minimal SKILL.md")
    init.add_argument("name")
    init.add_argument("--path", default="./skills")
    init.add_argument(
        "--resources",
        default="references,assets",
        help="Comma-separated directories: references,assets,scripts",
    )

    validate = skill_commands.add_parser(
        "validate", help="Validate one Skill or a Skill directory"
    )
    validate.add_argument("path")
    validate.add_argument("--no-strict", action="store_true")

    inspect = skill_commands.add_parser(
        "inspect", help="Print the immutable Skill catalog"
    )
    inspect.add_argument("path")
    inspect.add_argument("--json", action="store_true", dest="as_json")

    lock = skill_commands.add_parser(
        "lock", help="Write catalog names, versions, sources, and digests"
    )
    lock.add_argument("path")
    lock.add_argument("--output", default="skills.lock.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.skill_command == "init":
            return _init_skill(args.name, Path(args.path), args.resources)
        if args.skill_command == "validate":
            registry = SkillRegistry.from_paths(
                Path(args.path),
                strict=not args.no_strict,
            )
            print(
                f"valid: {len(registry)} skill(s), "
                f"catalog_digest={registry.catalog_digest}"
            )
            return 0
        if args.skill_command == "inspect":
            registry = SkillRegistry.from_paths(Path(args.path))
            payload = _catalog_payload(registry)
            if args.as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                for item in payload["skills"]:
                    print(
                        f"{item['name']} {item['version'] or '-'} "
                        f"{item['digest']} {item['description']}"
                    )
            return 0
        if args.skill_command == "lock":
            root = Path(args.path)
            registry = SkillRegistry.from_paths(root)
            output = Path(args.output)
            if not output.is_absolute():
                output = (
                    root.parent / output
                    if (root / "SKILL.md").is_file()
                    else root / output
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(_catalog_payload(registry), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            print(output)
            return 0
    except (OSError, SkillError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError("unknown command")


def _init_skill(name: str, root: Path, resources: str) -> int:
    name = validate_skill_name(name)
    selected = tuple(value.strip() for value in resources.split(",") if value.strip())
    allowed = {"references", "assets", "scripts"}
    unknown = set(selected).difference(allowed)
    if unknown:
        raise ValueError(
            f"unsupported resource directory: {', '.join(sorted(unknown))}"
        )
    skill_root = root / name
    skill_file = skill_root / "SKILL.md"
    if skill_file.exists():
        raise FileExistsError(f"Skill already exists: {skill_file}")
    skill_root.mkdir(parents=True, exist_ok=True)
    for directory in selected:
        (skill_root / directory).mkdir(exist_ok=True)
    skill_file.write_text(
        "---\n"
        f"name: {name}\n"
        "description: Describe what this Skill does and when to use it.\n"
        "---\n\n"
        f"# {name}\n\n"
        "Describe the workflow here.\n",
        encoding="utf-8",
    )
    print(skill_root)
    return 0


def _catalog_payload(registry: SkillRegistry) -> dict[str, object]:
    return {
        "schema_version": 1,
        "catalog_digest": registry.catalog_digest,
        "skills": [
            {
                "name": descriptor.name,
                "description": descriptor.description,
                "version": descriptor.version,
                "digest": descriptor.digest,
                "source_id": descriptor.source_id,
                "allowed_tools": sorted(descriptor.allowed_tools),
            }
            for descriptor in registry
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
