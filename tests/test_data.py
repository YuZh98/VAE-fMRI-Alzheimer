"""Tests for data loading and preprocessing."""

from __future__ import annotations

import pytest
import torch

from recvae import (
    FMRIDataset,
    build_dataloader,
    list_subject_files,
    normalize_per_subject,
)


# ---------- normalize_per_subject ----------

def test_normalization_maps_to_unit_interval():
    vol = torch.rand(3, 1, 4, 4, 4, 5) * 100 + 50  # arbitrary positive range
    norm, mx, mn = normalize_per_subject(vol.clone())
    assert torch.isclose(norm.amax(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(norm.amin(), torch.tensor(-1.0), atol=1e-5)
    assert mx.shape == (3,)
    assert mn.shape == (3,)


def test_normalization_handles_constant_volume():
    """Original notebook would NaN here; the package guards against div-by-0."""
    vol = torch.full((2, 1, 3, 3, 3, 4), 5.0)
    norm, mx, mn = normalize_per_subject(vol.clone())
    assert torch.isfinite(norm).all()
    assert torch.allclose(mx, torch.tensor([5.0, 5.0]))
    assert torch.allclose(mn, torch.tensor([5.0, 5.0]))


def test_normalization_rejects_wrong_rank():
    with pytest.raises(ValueError, match="6D"):
        normalize_per_subject(torch.zeros(2, 1, 3, 3, 3))


def test_normalization_per_subject_is_independent():
    """Subject 0 ranges in [0,10], subject 1 in [100,200]; both must end in [-1,1]."""
    vol = torch.zeros(2, 1, 2, 2, 2, 2)
    vol[0] = torch.linspace(0, 10, vol[0].numel()).reshape(vol[0].shape)
    vol[1] = torch.linspace(100, 200, vol[1].numel()).reshape(vol[1].shape)
    norm, _, _ = normalize_per_subject(vol)
    for i in range(2):
        assert torch.isclose(norm[i].amax(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(norm[i].amin(), torch.tensor(-1.0), atol=1e-5)


# ---------- FMRIDataset ----------

def test_dataset_yields_index_and_volume():
    vol = torch.zeros(3, 1, 2, 2, 2, 2)
    ds = FMRIDataset(vol)
    assert len(ds) == 3
    item, idx = ds[1]
    assert item.shape == (1, 2, 2, 2, 2)
    assert idx == 1


def test_dataset_rejects_wrong_rank():
    with pytest.raises(ValueError, match="6D"):
        FMRIDataset(torch.zeros(3, 2, 2, 2))


# ---------- build_dataloader ----------

def test_dataloader_seed_gives_deterministic_order():
    vol = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1, 1, 1, 1)
    ds = FMRIDataset(vol)
    dl_a = build_dataloader(ds, batch_size=4, shuffle=True, seed=123)
    dl_b = build_dataloader(ds, batch_size=4, shuffle=True, seed=123)
    a = [tuple(idx.tolist()) for _, idx in dl_a]
    b = [tuple(idx.tolist()) for _, idx in dl_b]
    assert a == b, (a, b)


# ---------- list_subject_files ----------

def test_list_subject_files_filters_norm_prefix(tmp_path):
    (tmp_path / "mainimage_1.nii.gz").touch()
    (tmp_path / "mainimage_2.nii").touch()
    (tmp_path / "norm_fmri_img_1.nii.gz").touch()
    (tmp_path / "ignored.txt").touch()
    names = list_subject_files(str(tmp_path))
    assert names == ["mainimage_1.nii.gz", "mainimage_2.nii"]


def test_list_subject_files_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_subject_files(str(tmp_path / "does_not_exist"))
