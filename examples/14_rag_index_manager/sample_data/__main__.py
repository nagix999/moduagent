"""Download a pinned NIST SP 800 corpus for example 14."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from ..environment import EnvironmentFileError, load_environment_file
from .nist import NISTCorpusError, download_nist_corpus


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(argv) if argv is not None else sys.argv[1:]
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env-file", default=os.getenv("RAG_ENV_FILE", ".env"))
    environment_args, _ = bootstrap.parse_known_args(values)
    explicit = (
        any(
            value == "--env-file" or value.startswith("--env-file=") for value in values
        )
        or "RAG_ENV_FILE" in os.environ
    )
    load_environment_file(environment_args.env_file, required=explicit)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=environment_args.env_file)
    parser.add_argument(
        "--output",
        default=os.getenv(
            "RAG_SAMPLE_CORPUS_ROOT",
            "examples/14_rag_index_manager/.runtime/nist-cybersecurity",
        ),
        help="corpus root containing documents/ and checksum manifests",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=int(os.getenv("RAG_SAMPLE_DOCUMENT_COUNT", "100")),
    )
    parser.add_argument(
        "--refresh-selection",
        action="store_true",
        help="replace the saved URL selection; use a fresh output directory",
    )
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        documents = download_nist_corpus(
            Path(args.output),
            count=args.count,
            refresh_selection=args.refresh_selection,
        )
        print(
            f"ready: {len(documents)} NIST documents in {Path(args.output) / 'documents'}"
        )
    except (
        EnvironmentFileError,
        NISTCorpusError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
