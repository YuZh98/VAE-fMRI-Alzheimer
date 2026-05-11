"""Data loading and preprocessing for fMRI volumes."""

from __future__ import annotations

import os
from collections.abc import Sequence

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
        if not f.startswith(exclude_prefix) and (f.endswith(".nii") or f.endswith(".nii.gz"))
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    seed: int | None = None,
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


def synthetic_cohort(
    n_cn: int = 4,
    n_ad: int = 4,
    T: int = 16,
    spatial: tuple[int, int, int] = (91, 109, 91),
    cohort_effect: float = 0.3,
    noise_std: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic fMRI-like cohort with two subgroups.

    Each subject has a low-rank spatial pattern with an AR(1) temporal
    profile per spatial component. The AD cohort additionally has a regional
    damping mask multiplied into the spatial pattern, mimicking atrophy:
    in a hemispheric ROI the pattern amplitude is reduced by
    ``cohort_effect``. Independent Gaussian noise of std ``noise_std`` is
    added per voxel-per-timestep.

    The function is deterministic in ``seed``: same seed produces identical
    output; different seeds produce statistically independent draws.

    Parameters
    ----------
    n_cn : number of control (label=0) subjects.
    n_ad : number of AD-like (label=1) subjects.
    T : number of timepoints per subject.
    spatial : (X, Y, Z) spatial extent. The default matches the canonical
              encoder's hardcoded shape; callers passing this to a real
              :class:`recvae.RecVAEModel` must keep the default.
    cohort_effect : fractional damping applied in the AD region. 0 disables
                    the AD/CN distinction; 1 zeroes the AD region entirely.
    noise_std : per-voxel additive Gaussian noise std.
    seed : RNG seed for reproducibility.

    Returns
    -------
    volumes : ``(N, 1, X, Y, Z, T)`` float32 tensor, N = n_cn + n_ad.
    labels  : ``(N,)`` int64 tensor, 0 for CN and 1 for AD, in the order
              ``[CN, ..., CN, AD, ..., AD]``.
    """
    if n_cn < 0 or n_ad < 0:
        raise ValueError(f"cohort sizes must be non-negative, got {n_cn}, {n_ad}")
    if (n_cn + n_ad) == 0:
        raise ValueError("at least one subject is required")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")

    X, Y, Z = spatial

    # Use a local generator so we don't disturb the global torch RNG.
    gen = torch.Generator().manual_seed(seed)

    # Low-rank spatial basis: K random 3D fields. Each subject mixes them
    # with their own coefficient vector. K small keeps the cohort dimension
    # statistically meaningful at modest sizes.
    K = 4
    basis = torch.randn(K, X, Y, Z, generator=gen, dtype=torch.float32)

    # AR(1) coefficient for the per-component temporal signal. Fixed across
    # subjects so the structure of the signal is comparable; subjects differ
    # in their spatial mixing weights and noise realization.
    ar_phi = 0.7

    # AD damping mask: damp a hemispheric slab so the cohort difference is
    # localized in space (a crude but recognizable atrophy pattern).
    damp_mask = torch.ones(X, Y, Z, dtype=torch.float32)
    half_x = X // 2
    damp_mask[:half_x] = 1.0 - cohort_effect

    N = n_cn + n_ad
    volumes = torch.empty(N, 1, X, Y, Z, T, dtype=torch.float32)

    for i in range(N):
        is_ad = i >= n_cn
        coeffs = torch.randn(K, generator=gen, dtype=torch.float32)
        # Per-subject spatial pattern: weighted sum of basis fields.
        pattern = (coeffs.view(K, 1, 1, 1) * basis).sum(dim=0)  # (X, Y, Z)
        if is_ad:
            pattern = pattern * damp_mask

        # Per-component AR(1) temporal series, K independent series, mixed
        # by the same coefficients so the spatial pattern temporally
        # modulates as a single 1-D scalar series.
        t_signal = torch.empty(T, dtype=torch.float32)
        prev = torch.randn((), generator=gen, dtype=torch.float32)
        for t in range(T):
            eps = torch.randn((), generator=gen, dtype=torch.float32)
            prev = ar_phi * prev + eps
            t_signal[t] = prev

        # Broadcast: (X, Y, Z, T) = (X, Y, Z, 1) * (T,)
        vol_4d = pattern.unsqueeze(-1) * t_signal.view(1, 1, 1, T)
        # Add per-voxel Gaussian noise.
        noise = torch.randn(X, Y, Z, T, generator=gen, dtype=torch.float32) * noise_std
        vol_4d = vol_4d + noise

        volumes[i, 0] = vol_4d

    labels = torch.zeros(N, dtype=torch.long)
    labels[n_cn:] = 1
    return volumes, labels
