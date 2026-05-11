"""Shared helper for tutorial demo scripts.

Importing this module makes two things true:

1. The repo root is on ``sys.path``, so ``import recvae`` works whether or
   not you ran ``pip install -e .`` first. This minimizes friction for
   readers who want to ``python tutorials/03_encoder_3dconv/build_encoder.py``
   directly.
2. PyTorch is set to use a small number of CPU threads. Tutorial demos use
   real (91, 109, 91) volumes through the actual encoder/decoder where
   relevant, and capping threads avoids saturating a learner's laptop.

This file deliberately stays small. It is not a framework — it is a
``sys.path`` shim plus a couple of plotting/timing helpers used by some
demos.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from contextlib import contextmanager

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep CPU demos polite. Override with TUTORIAL_THREADS env var if you want.
try:
    import torch  # noqa: E402

    _threads = int(os.environ.get("TUTORIAL_THREADS", "2"))
    torch.set_num_threads(max(1, _threads))
except Exception:  # pragma: no cover - torch may not be importable at parse time
    pass


@contextmanager
def section(title: str):
    """Print a banner around a block of demo output.

    Usage::

        with section("encoder shape chain"):
            ...
    """
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"-- done in {dt:.2f}s")


def banner(label: str) -> None:
    """Single-line banner, lighter than :func:`section`."""
    print(f"\n--- {label} ---")
