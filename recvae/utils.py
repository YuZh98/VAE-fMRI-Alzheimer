"""Device, seed, and DataLoader utilities."""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Pin RNG state for Python, NumPy, and PyTorch (CPU + all CUDA + MPS).

    In addition to seeding the libraries the original notebook touched,
    this also sets:

    - ``PYTHONHASHSEED`` (via os.environ) so dict/set iteration order from
      hash-based containers is deterministic when the process starts fresh.
    - ``CUBLAS_WORKSPACE_CONFIG`` so cuBLAS matmuls are reproducible.
    - ``torch.use_deterministic_algorithms(True, warn_only=True)`` so any
      remaining nondeterministic kernel warns instead of silently drifting.
    - ``torch.mps.manual_seed`` when running on Apple Silicon.

    Both environment variables are set with ``setdefault`` so an explicit
    user override survives.

    Caveat
    ------
    Bit-exact reproducibility is only guaranteed within a single device
    class. A run seeded on CPU will not reproduce bitwise on CUDA or MPS,
    and vice versa, because each backend has its own kernel implementations
    and rounding behavior. Seeding gets you reproducibility within one
    hardware/library combination, not across them.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS backend availability + torch.mps.manual_seed presence vary by
        # torch build; suppress AttributeError on older versions.
        with contextlib.suppress(Exception):  # pragma: no cover
            torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True keeps ops that lack a deterministic kernel from raising;
    # they instead warn. Older torch versions don't support the kwarg.
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_default_device() -> torch.device:
    """Pick CUDA if available, then MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(data, device: torch.device, dtype: torch.dtype = torch.float32):
    """Recursively move tensor(s) to ``device``. Handles list/tuple inputs.

    Index tensors keep their integer dtype; everything else is cast to
    ``dtype`` (default float32).
    """
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device, dtype) for x in data)
    if data.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool):
        return data.to(device=device, non_blocking=True)
    return data.to(device=device, dtype=dtype, non_blocking=True)


class DeviceDataLoader:
    """Wrap a DataLoader to move every batch to a device on iteration."""

    def __init__(self, dl: DataLoader, device: torch.device):
        self.dl = dl
        self.device = device

    def __iter__(self) -> Iterable:
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self) -> int:
        return len(self.dl)
