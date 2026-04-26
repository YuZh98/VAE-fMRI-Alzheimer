"""Recurrent VAE for temporal fMRI representation learning.

See README.md for architecture and usage.
"""

from .config import Config
from .data import (
    FMRIDataset,
    build_dataloader,
    list_subject_files,
    load_subject_volumes,
    normalize_per_subject,
)
from .model import RecVAEModel
from .train import evaluate, fit
from .utils import (
    DeviceDataLoader,
    get_default_device,
    set_seed,
    to_device,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DeviceDataLoader",
    "FMRIDataset",
    "RecVAEModel",
    "build_dataloader",
    "evaluate",
    "fit",
    "get_default_device",
    "list_subject_files",
    "load_subject_volumes",
    "normalize_per_subject",
    "set_seed",
    "to_device",
]
