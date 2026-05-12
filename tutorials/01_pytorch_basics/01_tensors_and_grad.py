"""Tensors, dtype, device, autograd, and the no_grad context."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch


def main() -> int:
    with section("tensor metadata"):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        print(f"x          = {x}")
        print(f"x.dtype    = {x.dtype}")
        print(f"x.device   = {x.device}")
        print(f"x.requires_grad = {x.requires_grad}")

    with section("autograd: forward + backward"):
        y = (x * x).sum()
        banner("forward graph recorded")
        # y = sum(x_i^2). dy/dx_i = 2 x_i. Expect grad = [2, 4, 6].
        print(f"y          = {y.item():.4f}")
        print(f"y.grad_fn  = {y.grad_fn}")
        y.backward()
        banner("backward filled x.grad")
        print(f"x.grad     = {x.grad}")
        assert torch.allclose(x.grad, torch.tensor([2.0, 4.0, 6.0]))

    with section("torch.no_grad disables graph recording"):
        # Mirrors @torch.no_grad() on updating_F at recvae/model.py:269.
        with torch.no_grad():
            z = (x * x).sum()
        print(f"z          = {z.item():.4f}")
        print(f"z.requires_grad = {z.requires_grad}  (False because of no_grad)")
        print(f"z.grad_fn  = {z.grad_fn}  (None — nothing was recorded)")
        assert z.requires_grad is False
        assert z.grad_fn is None

    with section("casting and devices"):
        a = torch.zeros(2, 3)
        print(f"default a.dtype  = {a.dtype}")
        print(f"default a.device = {a.device}")
        a64 = a.to(dtype=torch.float64)
        print(f"after .to float64: a64.dtype = {a64.dtype}")
        # We do not move to GPU here — keep the demo CPU-only and deterministic.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
