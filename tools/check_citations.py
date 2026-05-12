#!/usr/bin/env python3
"""Structural validator for ``recvae/<file>.py:NN(-MM)`` citations.

Walks ``docs/``, ``tutorials/`` and the repository root, scans every ``.md``
and ``.py`` file, and verifies that each citation of the form
``recvae/<file>.py:N`` or ``recvae/<file>.py:N-M`` points at a real file and
in-bounds line range. The check is intentionally *structural* — it does not
verify the cited content is semantically correct, only that the target file
exists and the referenced lines are within the file's current length. This
catches the most common form of citation rot (deleted files, dropped lines)
without forcing edits on every internal refactor.

Exit codes
----------
0
    All citations validated.
1
    One or more citations are stale. Each offending location is printed in
    the form ``path:line: <reason>: <citation>``.

The script uses only the standard library and is designed to run in well
under five seconds on the canonical repository checkout.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Match references like ``recvae/model.py:174-185`` or ``recvae/data.py:42``.
# - Group 1: the file path relative to the repo root (e.g. ``recvae/model.py``).
# - Group 2: starting line (1-indexed).
# - Group 3 (optional): ending line of an inclusive range.
CITATION_RE = re.compile(r"(recvae/[a-z_]+\.py):(\d+)(?:-(\d+))?")

# Files and directories scanned for citations. Kept narrow on purpose: the
# repo's ``recvae/`` source itself doesn't cite line numbers, and ``tests/``
# / ``examples/`` / ``notebooks/`` are out of scope for the citation policy.
SCAN_ROOTS = ("docs", "tutorials")
SCAN_FILE_GLOBS = ("*.md", "*.py")

# Files that live at the repository root and may legitimately contain
# citations. Currently empty — the root ``README.md`` does not cite lines
# numbers and is intentionally not modifiable by docs tooling.
ROOT_FILES: tuple[str, ...] = ()


def iter_scan_files(repo_root: Path):
    """Yield every documentation file we want to scan for citations."""
    for sub in SCAN_ROOTS:
        base = repo_root / sub
        if not base.is_dir():
            continue
        for pattern in SCAN_FILE_GLOBS:
            yield from base.rglob(pattern)
    for name in ROOT_FILES:
        path = repo_root / name
        if path.is_file():
            yield path


def line_count_cache(repo_root: Path):
    """Return a callable ``count(rel_path) -> int`` that memoizes results."""
    cache: dict[str, int] = {}

    def count(rel_path: str) -> int:
        if rel_path in cache:
            return cache[rel_path]
        target = repo_root / rel_path
        try:
            # Counting newlines is cheaper than splitlines for big files and
            # matches what the citation author would have visually counted.
            with target.open("rb") as fh:
                n = sum(1 for _ in fh)
        except FileNotFoundError:
            n = -1
        cache[rel_path] = n
        return n

    return count


def check_file(path: Path, repo_root: Path, line_count) -> list[str]:
    """Return a list of human-readable error messages for ``path``."""
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip non-text files defensively; nothing in scope is binary, but a
        # stray asset shouldn't crash the linter.
        return errors

    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in CITATION_RE.finditer(line):
            rel_path, start_s, end_s = match.group(1), match.group(2), match.group(3)
            citation = match.group(0)
            n_lines = line_count(rel_path)

            if n_lines < 0:
                errors.append(
                    f"{path}:{lineno}: target file does not exist: {citation}",
                )
                continue

            start = int(start_s)
            end = int(end_s) if end_s is not None else start

            if start < 1 or end < start:
                errors.append(
                    f"{path}:{lineno}: malformed line range: {citation}",
                )
                continue
            if end > n_lines:
                errors.append(
                    f"{path}:{lineno}: line {end} > file length {n_lines}: {citation}",
                )
                continue

    return errors


def main() -> int:
    """Walk the repo, validate citations, and report the result."""
    # ``tools/check_citations.py`` lives one directory below the repo root.
    repo_root = Path(__file__).resolve().parent.parent

    line_count = line_count_cache(repo_root)

    errors: list[str] = []
    total = 0
    for path in iter_scan_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # Count citations independently of the per-file checker so the final
        # summary reflects the universe of citations, not just clean ones.
        total += sum(1 for _ in CITATION_RE.finditer(text))
        errors.extend(check_file(path, repo_root, line_count))

    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        print(
            f"FAIL: {len(errors)} stale citation(s) out of {total} checked.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {total} citations validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
