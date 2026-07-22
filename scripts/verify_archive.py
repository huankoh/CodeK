#!/usr/bin/env python3
"""Fail if a CodeK release contains common archival or privacy hazards."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "DATA_AVAILABILITY.md",
    "requirements.txt",
    "data/README.md",
    "data/MANIFEST.tsv",
}
FORBIDDEN_NAMES = {".DS_Store", "Thumbs.db", "temporary.csv"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt"}
RAW_DATA_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather", ".vcf", ".vcf.gz"}
ALLOWED_DATA_FILES = {"data/MANIFEST.tsv"}


def fail(message: str) -> None:
    print(f"ERROR: {message}")


def main() -> int:
    errors = 0

    for relative in sorted(REQUIRED):
        if not (ROOT / relative).is_file():
            fail(f"missing required file: {relative}")
            errors += 1

    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.name in FORBIDDEN_NAMES or ".ipynb_checkpoints" in path.parts:
            fail(f"generated metadata is present: {relative}")
            errors += 1
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"model checkpoint is present: {relative}")
            errors += 1
        if relative.startswith("data/") and relative not in ALLOWED_DATA_FILES:
            lower = path.name.lower()
            if any(lower.endswith(suffix) for suffix in RAW_DATA_SUFFIXES):
                fail(f"data payload is present in the public source tree: {relative}")
                errors += 1

    try:
        json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid .zenodo.json: {exc}")
        errors += 1

    for notebook in ROOT.rglob("*.ipynb"):
        try:
            json.loads(notebook.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"invalid notebook {notebook.relative_to(ROOT)}: {exc}")
            errors += 1

    if errors:
        print(f"Archive verification failed with {errors} error(s).")
        return 1
    print("Archive verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
