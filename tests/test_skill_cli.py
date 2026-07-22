from __future__ import annotations

import json
from pathlib import Path

import pytest

from moduagent.cli import main
from moduagent.skills import SkillDigestMismatchError, SkillRegistry


def test_skill_cli_init_validate_inspect_and_lock(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "skills"

    assert main(["skills", "init", "invoice-review", "--path", str(root)]) == 0
    skill = root / "invoice-review"
    assert (skill / "SKILL.md").is_file()
    assert (skill / "references").is_dir()
    assert (skill / "assets").is_dir()

    assert main(["skills", "validate", str(root)]) == 0
    assert "valid: 1 skill(s)" in capsys.readouterr().out

    assert main(["skills", "inspect", str(root), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["skills"][0]["name"] == "invoice-review"

    assert main(["skills", "lock", str(root)]) == 0
    lockfile = root / "skills.lock.json"
    assert lockfile.is_file()
    locked = json.loads(lockfile.read_text(encoding="utf-8"))
    assert locked["catalog_digest"].startswith("sha256:")
    assert (
        SkillRegistry.from_paths(root, lockfile=lockfile).catalog_digest
        == locked["catalog_digest"]
    )

    nested_lock = root / "deploy" / "skills.lock.json"
    assert (
        main(
            [
                "skills",
                "lock",
                str(root),
                "--output",
                "deploy/skills.lock.json",
            ]
        )
        == 0
    )
    assert nested_lock.is_file()

    skill_file = skill / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8") + "\nChanged.\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillDigestMismatchError):
        SkillRegistry.from_paths(root, lockfile=lockfile)


def test_skill_cli_rejects_overwrite_and_bad_names(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "skills"
    assert main(["skills", "init", "valid-name", "--path", str(root)]) == 0
    capsys.readouterr()

    assert main(["skills", "init", "valid-name", "--path", str(root)]) == 2
    assert "already exists" in capsys.readouterr().err

    assert main(["skills", "init", "Bad_Name", "--path", str(root)]) == 2
    assert "skill name" in capsys.readouterr().err
