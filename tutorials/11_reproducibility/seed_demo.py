"""Demonstrate that set_seed pins every RNG we care about.

Three claims this script verifies:

1. Same seed + same sequence -> bitwise-equal outputs.
2. Different seed -> different outputs.
3. The RNG state advances between draws: two consecutive torch.randn
   calls differ even without re-seeding.
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch
from torch import nn

from recvae import set_seed


def draw_and_forward() -> tuple[torch.Tensor, torch.Tensor]:
    """One unit of reproducible work.

    - sample a 4-vector,
    - build a tiny Linear(4, 4) (its init pulls from the same RNG),
    - run it on a fixed input,
    - return both tensors.
    """
    x = torch.randn(4)
    layer = nn.Linear(4, 4)
    # Fixed input so any output difference is due to RNG-driven init.
    fixed_in = torch.ones(4)
    y = layer(fixed_in)
    return x, y


def main() -> int:
    with section("run 1: set_seed(2022)"):
        set_seed(2022)
        x1, y1 = draw_and_forward()
        print(f"randn       : {x1}")
        print(f"linear(ones): {y1}")

    with section("run 2: set_seed(2022) again -> must equal run 1"):
        set_seed(2022)
        x2, y2 = draw_and_forward()
        print(f"randn       : {x2}")
        print(f"linear(ones): {y2}")
        # Deterministic re-execution: == is fine, no allclose tolerance needed.
        assert torch.equal(x1, x2), "randn drifted across re-seed"
        assert torch.equal(y1, y2), "Linear output drifted across re-seed"
        # torch.allclose works too and is the safer idiom in general.
        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)
        print("ok: bitwise-equal to run 1")

    with section("run 3: set_seed(1234) -> must differ"):
        set_seed(1234)
        x3, y3 = draw_and_forward()
        print(f"randn       : {x3}")
        print(f"linear(ones): {y3}")
        assert not torch.equal(x1, x3), "different seeds produced identical randn"
        assert not torch.equal(y1, y3), "different seeds produced identical Linear out"
        print("ok: differs from run 1")

    with section("RNG state advances: consecutive draws differ"):
        # Note: we do NOT call set_seed here. PyTorch's global RNG just keeps
        # advancing from whatever state it currently has, so two consecutive
        # torch.randn calls draw from different RNG positions and produce
        # different tensors.
        a = torch.randn(4)
        b = torch.randn(4)
        banner("two consecutive torch.randn(4) calls")
        print(f"a: {a}")
        print(f"b: {b}")
        assert not torch.equal(a, b), "two randn draws happened to match (vanishingly unlikely)"
        print("ok: consecutive draws differ as expected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
