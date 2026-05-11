"""Pitfall: wrong ConvTranspose3d output_padding silently produces the
wrong spatial shape.

Builds a small ConvTranspose3d chain with intentionally wrong
output_padding, prints the off-by-one shape, then rebuilds with the
correct values. Uses tiny spatial sizes so the demo runs instantly.

This is a scaled-down version of Lesson 04's chain. The point here is
the *silence* of the bug: PyTorch produces a tensor, it just has the
wrong size.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402


def upsample_chain(op2: int, op3: int) -> nn.Sequential:
    """Two stride-2 transposed convs with given output_paddings.

    Each layer does out = 2*in + output_padding for (k=4, s=2, p=1).
    """
    return nn.Sequential(
        nn.ConvTranspose3d(2, 2, kernel_size=4, stride=2, padding=1,
                           output_padding=op2, bias=False),
        nn.ConvTranspose3d(2, 1, kernel_size=4, stride=2, padding=1,
                           output_padding=op3, bias=False),
    )


def main() -> int:
    # We want to recover spatial shape (1, 1, 11, 13, 11) from a (1, 2, 3, 3, 3)
    # encoded feature.
    #
    #   layer 1: 3 -> 2*3 + op  -> we need 5 (so op=-1? no: revisit chain)
    #
    # Pick a feasible chain: input (1, 2, 2, 3, 2) -> target (1, 1, 9, 13, 9).
    # Per-axis math with out = 2*in + op:
    #     X: 2 -> 4(op=0)  -> 9(op=1)
    #     Y: 3 -> 7(op=1)  -> 13(op=-1)  -- NOT VALID, op>=0
    #
    # Simpler legal chain: input (1, 2, 2, 3, 2) -> target (1, 1, 8, 12, 8).
    #     X: 2 -> 4(op=0) -> 8(op=0)
    #     Y: 3 -> 6(op=0) -> 12(op=0)
    #     Z: 2 -> 4(op=0) -> 8(op=0)
    # Scalar op=0 throughout works. To create a silent off-by-one we will
    # *target* (1, 1, 8, 13, 8) -- which requires per-dim Y op on the second
    # layer to be 1, not 0.

    target = (1, 1, 8, 13, 8)
    x = torch.randn(1, 2, 2, 3, 2)

    with section("wrong: scalar output_padding=0 on both layers"):
        wrong = upsample_chain(op2=0, op3=0)
        out = wrong(x)
        print(f"input shape    : {tuple(x.shape)}")
        print(f"after layer 1  : {tuple(wrong[0](x).shape)}")
        print(f"after layer 2  : {tuple(out.shape)}")
        print(f"target shape   : {target}")
        match = tuple(out.shape) == target
        print(f"matches target : {match}  <-- Y dim is 12, not 13 (off by one)")
        assert not match, "this branch should NOT match the target"

        # The error only surfaces when you try to combine with a tensor of
        # the actual target shape.
        banner("attempting (target - out).pow(2).sum()")
        fake_target = torch.randn(*target)
        try:
            loss = (fake_target - out).pow(2).sum()
            print(f"loss = {loss.item()}  (UNEXPECTED success)")
        except RuntimeError as exc:
            print(f"RuntimeError at the loss line, not the decoder:\n  {exc}")

    with section("correct: per-dimension output_padding (0, 1, 0) on layer 2"):
        # Layer 2: 2*(4) + op = 8 + op, want X=8 (op=0), Y=13 (op=1), Z=8 (op=0).
        right = nn.Sequential(
            nn.ConvTranspose3d(2, 2, kernel_size=4, stride=2, padding=1,
                               output_padding=0, bias=False),
            nn.ConvTranspose3d(2, 1, kernel_size=4, stride=2, padding=1,
                               output_padding=(0, 1, 0), bias=False),
        )
        out = right(x)
        print(f"after layer 1  : {tuple(right[0](x).shape)}")
        print(f"after layer 2  : {tuple(out.shape)}")
        print(f"target shape   : {target}")
        match = tuple(out.shape) == target
        print(f"matches target : {match}")
        assert match, "correct chain should match target"

    with section("takeaways"):
        print("- PyTorch does NOT raise when output_padding is wrong.")
        print("- The bad shape silently propagates downstream.")
        print("- The traceback points at the operation that finally fails")
        print("  (often loss subtraction), NOT the offending decoder layer.")
        print("- Fix: print every layer's output shape when building a new")
        print("  decoder. See recvae/model.py:13-20 for the recvae chain.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
