"""Pitfall: DataLoader generator with device='cuda'.

The legacy notebook used:

    g = torch.Generator(device='cuda').manual_seed(seed)
    DataLoader(..., generator=g)

That crashes on CPU-only machines and is unnecessary anyway:
DataLoader's shuffle indices are host-side Python ints.

We do NOT instantiate Generator(device='cuda') here (it would raise on
CPU-only). Instead we show that a CPU generator gives reproducible
shuffle order, which is the actual goal.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def collect_order(seed: int) -> list:
    """Build a DataLoader with a CPU generator and return the shuffle order."""
    data = torch.arange(10).unsqueeze(1).float()
    ds = TensorDataset(data)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, generator=g)
    return [int(batch[0].item()) for batch in dl]


def main() -> int:
    with section("the unsafe legacy pattern"):
        print("g = torch.Generator(device='cuda').manual_seed(seed)")
        print("DataLoader(..., generator=g)")
        print()
        print("On a CPU-only machine, torch.Generator(device='cuda') raises:")
        print("  RuntimeError: Device type CUDA is not supported for ...")
        print("It is also unnecessary -- DataLoader indices are host-side.")

    with section("the safe pattern (matches recvae/data.py:141-147)"):
        # Note: NO device= argument. The generator lives on CPU.
        g = torch.Generator().manual_seed(2022)
        print(f"generator      = {g}")
        print(f"generator.device = {g.device}")

    with section("reproducibility: same seed -> same order"):
        order_a = collect_order(seed=2022)
        order_b = collect_order(seed=2022)
        order_c = collect_order(seed=9999)
        print(f"seed=2022 run 1: {order_a}")
        print(f"seed=2022 run 2: {order_b}")
        print(f"seed=9999      : {order_c}")
        assert order_a == order_b, "same seed should reproduce same order"
        assert order_a != order_c, "different seeds should differ"
        banner("same seed reproduces; different seed differs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
