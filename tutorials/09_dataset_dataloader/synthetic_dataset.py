"""Demo: subclass Dataset, build DataLoader, verify seeded shuffle determinism."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch
from torch.utils.data import DataLoader, Dataset


class TinyFMRIDataset(Dataset):
    """Minimal stand-in for recvae.FMRIDataset.

    Same contract: 6D tensor in, ``(volume, idx)`` out. Kept here so this
    demo is readable on its own without flipping to recvae/data.py.
    """

    def __init__(self, volumes: torch.Tensor):
        if volumes.dim() != 6:
            raise ValueError(f"expected 6D, got {tuple(volumes.shape)}")
        self.volumes = volumes

    def __len__(self) -> int:
        return self.volumes.shape[0]

    def __getitem__(self, idx: int):
        return self.volumes[idx], idx


def iterate(loader: DataLoader) -> list:
    """Return the index tensor seen on each batch, as plain Python lists."""
    seen = []
    for vol, idx in loader:
        print(f"  vol.shape={tuple(vol.shape)}  idx={idx.tolist()}")
        seen.append(idx.tolist())
    return seen


def main() -> int:
    N, C, X, Y, Z, T = 6, 1, 8, 8, 8, 4
    volumes = torch.randn(N, C, X, Y, Z, T)

    with section("construct dataset"):
        ds = TinyFMRIDataset(volumes)
        print(f"len(ds) = {len(ds)}")
        vol0, idx0 = ds[0]
        print(f"ds[0] -> vol.shape={tuple(vol0.shape)}, idx={idx0}")

    with section("two loaders, same seed -> identical shuffle"):
        gen_a = torch.Generator().manual_seed(123)
        gen_b = torch.Generator().manual_seed(123)
        loader_a = DataLoader(ds, batch_size=2, shuffle=True, generator=gen_a)
        loader_b = DataLoader(ds, batch_size=2, shuffle=True, generator=gen_b)

        banner("loader_a (seed=123)")
        seen_a = iterate(loader_a)
        banner("loader_b (seed=123)")
        seen_b = iterate(loader_b)

        assert seen_a == seen_b, (
            f"same seed should produce same batch order; got {seen_a} vs {seen_b}"
        )
        print("ok: same seed -> same batch order")

    with section("different seed -> different shuffle"):
        gen_c = torch.Generator().manual_seed(999)
        loader_c = DataLoader(ds, batch_size=2, shuffle=True, generator=gen_c)
        banner("loader_c (seed=999)")
        seen_c = iterate(loader_c)
        # We don't assert inequality strictly (some seed pairs could collide on
        # tiny N), but print the comparison so the reader sees the difference.
        same = seen_a == seen_c
        print(f"seed=123 vs seed=999 produced same order? {same}")

    with section("device argument footgun"):
        # The original notebook used torch.Generator(device='cuda'), which
        # crashes on CPU-only systems. The CPU-side generator above works
        # everywhere because shuffle indices are CPU integers regardless of
        # where the tensors live. We don't actually construct a cuda generator
        # here (we may have no cuda); we just narrate the choice.
        print("torch.Generator() (CPU): works on every platform.")
        print("torch.Generator(device='cuda'): crashes on CPU-only systems.")
        print("DataLoader shuffle indices are CPU-side, so CPU generator is correct.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
