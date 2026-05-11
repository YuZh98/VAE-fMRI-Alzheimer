"""Recurrent VAE for temporal fMRI representation learning.

See README.md for architecture and usage.
"""

from .checkpoint import load_run, save_run
from .config import Config
from .data import (
    FMRIDataset,
    build_dataloader,
    list_subject_files,
    load_subject_volumes,
    normalize_per_subject,
    synthetic_cohort,
)
from .evaluation import evaluate_held_out, split_subjects
from .losses import KLRecVAELoss, RecVAELoss
from .model import RecVAEModel, RolloutOutput
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
    "KLRecVAELoss",
    "RecVAELoss",
    "RecVAEModel",
    "RolloutOutput",
    "build_dataloader",
    "evaluate",
    "evaluate_held_out",
    "fit",
    "get_default_device",
    "list_subject_files",
    "load_run",
    "load_subject_volumes",
    "normalize_per_subject",
    "save_run",
    "set_seed",
    "split_subjects",
    "synthetic_cohort",
    "to_device",
]
