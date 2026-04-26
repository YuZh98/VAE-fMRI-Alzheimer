"""Device, seed, and DataLoader utilities."""

from __future__ import annotations

import random
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Pin RNG state for Python, NumPy, and PyTorch (CPU + all CUDA devices).

    Also enables deterministic cuDNN. This can slow training but makes runs
    reproducible. The original notebook only called ``torch.manual_seed`` —
    cuDNN and the CUDA RNG were left unpinned.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
