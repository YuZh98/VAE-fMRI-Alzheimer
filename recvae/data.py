"""Data loading and preprocessing for fMRI volumes."""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _lazy_nib():
    """Import nibabel only when actually loading NIfTI files.

    Lets synthetic-data tests run without nibabel installed.
    """
    import nibabel as nib  # noqa: WPS433

    return nib


def list_subject_files(directory: str, exclude_prefix: str = "norm_") -> list:
    """Return sorted NIfTI filenames in ``directory``.

    Filenames starting with ``exclude_prefix`` are skipped — this matches the
    original notebook's ``remove_norm`` behavior of ignoring pre-normalized
    duplicates that lived alongside the raw scans.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"data directory not found: {directory}")
    return sorted(
        f
        for f in os.listdir(directory)
        if not f.startswith(exclude_prefix)
        and (f.endswith(".nii") or f.endswith(".nii.gz"))
    )


def load_subject_volumes(
    directory: str,
    filenames: Sequence[str],
    tol_time: int = 120,
) -> torch.Tensor:
    """Load each NIfTI into a 5D tensor and stack along subject dimension.

    Parameters
    ----------
    directory : root directory containing the files
    filenames : sequence of NIfTI filenames within ``directory``
    tol_time : truncate each volume to this many timepoints. Volumes with
        fewer timepoints raise ``ValueError`` (the original notebook silently
        produced shape mismatches on short volumes).

    Returns
    -------
    torch.Tensor of shape ``(N, 1, X, Y, Z, T)`` and dtype float32.
    """
    if not filenames:
        raise ValueError("filenames is empty")
    nib = _lazy_nib()
    tensors = []
    for name in filenames:
        path = os.path.join(directory, name)
        arr = nib.load(path).get_fdata()
        if arr.shape[-1] < tol_time:
            raise ValueError(
                f"{name}: only {arr.shape[-1]} timepoints, need >= {tol_time}",
            )
        # leading channel dim, then truncate time
        t = torch.from_numpy(arr[..., :tol_time]).float().unsqueeze(0)
        tensors.append(t)
    return torch.stack(tensors, dim=0)


def normalize_per_subject(
    volumes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-subject min-max normalize each volume to ``[-1, 1]``.

    NOTE
    ----
    This matches the original notebook's normalization. Per-subject statistics
    destroy absolute-intensity differences across subjects, which may matter
    for downstream classification. Considered a research decision and left
    unchanged here; see README for context.

    Parameters
    ----------
    volumes : ``(N, 1, X, Y, Z, T)`` float tensor (modified in place).

    Returns
    -------
    normalized : same tensor, scaled to ``[-1, 1]`` per subject.
    max_values : ``(N,)`` per-subject max before normalization.
    min_values : ``(N,)`` per-subject min before normalization.
    """
    if volumes.dim() != 6:
        raise ValueError(f"expected 6D tensor, got shape {tuple(volumes.shape)}")
    max_values = torch.amax(volumes, dim=(1, 2, 3, 4, 5))
    min_values = torch.amin(volumes, dim=(1, 2, 3, 4, 5))
    span = max_values - min_values
    # Constant-volume safety: avoid 0/0. The original notebook would NaN here.
    span = torch.where(span == 0, torch.ones_like(span), span)
    for i in range(volumes.shape[0]):
        volumes[i] = 2 * ((volumes[i] - min_values[i]) / span[i] - 0.5)
    return volumes, max_values, min_values


class FMRIDataset(Dataset):
    """Yields ``(volume, index)`` pairs.

    The integer index is needed at training time to look up the subject's
    noise vector ``z_vectors[index]`` on the model.
    """

    def __init__(self, volumes: torch.Tensor):
        if volumes.dim() != 6:
            raise ValueError(
                f"expected 6D tensor (N,C,X,Y,Z,T), got shape {tuple(volumes.shape)}",
            )
        self.volumes = volumes

    def __len__(self) -> int:
        return self.volumes.shape[0]

    def __getitem__(self, idx: int):
        return self.volumes[idx], idx


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> DataLoader:
    """Construct a DataLoader with deterministic shuffle if ``seed`` given.

    Uses a CPU-side ``torch.Generator``. The original notebook constructed
    the generator with ``device='cuda'`` which crashes on CPU-only systems
    and is unnecessary — DataLoader shuffle indices are CPU-side integers.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
