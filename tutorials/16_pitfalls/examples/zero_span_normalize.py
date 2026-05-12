"""Pitfall: zero-span min-max normalization produces NaN.

The legacy notebook used per-subject min-max:

    norm[i] = 2 * ((x[i] - min[i]) / (max[i] - min[i]) - 0.5)

When a subject's volume is constant (all-zero, masking bug, ...), the
denominator is 0 and PyTorch produces NaN -- which then poisons every
gradient downstream.

The fix in recvae/data.py:100-102 substitutes span=1 in the degenerate
slots. A constant subject ends up at zero post-normalization, but no
NaN.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402


def naive_normalize(volumes: torch.Tensor) -> torch.Tensor:
    """Per-subject min-max to [-1, 1]. UNSAFE: divides by zero span."""
    out = volumes.clone()
    max_v = torch.amax(out, dim=(1, 2, 3, 4, 5))
    min_v = torch.amin(out, dim=(1, 2, 3, 4, 5))
    for i in range(out.shape[0]):
        out[i] = 2 * ((out[i] - min_v[i]) / (max_v[i] - min_v[i]) - 0.5)
    return out


def safe_normalize(volumes: torch.Tensor) -> torch.Tensor:
    """Same as naive_normalize but with the recvae/data.py:100-102 guard."""
    out = volumes.clone()
    max_v = torch.amax(out, dim=(1, 2, 3, 4, 5))
    min_v = torch.amin(out, dim=(1, 2, 3, 4, 5))
    span = max_v - min_v
    span = torch.where(span == 0, torch.ones_like(span), span)
    for i in range(out.shape[0]):
        out[i] = 2 * ((out[i] - min_v[i]) / span[i] - 0.5)
    return out


def main() -> int:
    # Two subjects: one real, one constant-zero (the bad subject).
    torch.manual_seed(0)
    real = torch.randn(1, 1, 4, 4, 4, 2)
    constant = torch.zeros(1, 1, 4, 4, 4, 2)
    volumes = torch.cat([real, constant], dim=0)

    with section("input"):
        print(f"volumes shape       : {tuple(volumes.shape)}  (N=2 subjects)")
        print(f"subject 0 range     : [{volumes[0].min():.3f}, {volumes[0].max():.3f}]")
        print(f"subject 1 range     : [{volumes[1].min():.3f}, {volumes[1].max():.3f}]   <-- constant")

    with section("naive normalize -> NaN on subject 1"):
        naive = naive_normalize(volumes)
        print(f"naive[0] has nan?   : {torch.isnan(naive[0]).any().item()}")
        print(f"naive[1] has nan?   : {torch.isnan(naive[1]).any().item()}   <-- bug")
        print(f"naive[1] all nan?   : {torch.isnan(naive[1]).all().item()}")
        assert torch.isnan(naive[1]).all()

        banner("downstream effect: gradient poisoning")
        # Show that the NaN propagates through a non-trivial op chain.
        # weight is upstream of naive in a real model; here we mimic that
        # by multiplying naive by a trainable scalar and summing squares.
        weight = torch.ones(1, requires_grad=True)
        loss = ((naive * weight) ** 2).sum()
        loss.backward()
        print(f"loss              : {loss.item()}   (NaN poisons the loss)")
        print(f"weight.grad       : {weight.grad.item()}   (NaN poisons upstream grads)")
        assert torch.isnan(weight.grad).any()

    with section("safe normalize (matches recvae/data.py:100-102)"):
        safe = safe_normalize(volumes)
        print(f"safe[0] has nan?    : {torch.isnan(safe[0]).any().item()}")
        print(f"safe[1] has nan?    : {torch.isnan(safe[1]).any().item()}   <-- fixed")
        # span=1 substitution: constant x at value v lands at 2*((v - v)/1 - 0.5) = -1.
        print(f"safe[1] all -1?     : {(safe[1] == -1).all().item()}   <-- constant -> -1, finite")
        print(f"safe[1] range       : [{safe[1].min():.3f}, {safe[1].max():.3f}]")
        assert not torch.isnan(safe).any()

        weight = torch.ones(1, requires_grad=True)
        loss = ((safe * weight) ** 2).sum()
        loss.backward()
        print(f"loss              : {loss.item():.4f}   (finite)")
        print(f"weight.grad       : {weight.grad.item():.4f}   (finite)")
        assert not torch.isnan(weight.grad).any()

    with section("takeaway"):
        print("- Any 'divide by range' normalization can silently produce NaN.")
        print("- The NaN poisons all downstream gradients, so the bug surfaces")
        print("  as 'loss is nan starting at batch 7' -- not as a clear error.")
        print("- Fix: replace zero span with 1 (the constant subject ends up")
        print("  at -1 post-normalization, which is finite and harmless).")
        print("- recvae/data.py:100-102 does exactly this.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
