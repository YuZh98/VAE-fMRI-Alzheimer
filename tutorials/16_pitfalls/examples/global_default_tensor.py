"""Pitfall: torch.set_default_tensor_type('torch.cuda.FloatTensor').

Demonstrates the SAFE pattern (explicit device=) rather than actually
calling set_default_tensor_type — that call crashes on CPU-only systems,
which is precisely the bug being illustrated.

Run this on any machine and it should exit 0.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    with section("the unsafe legacy pattern (do NOT run on CPU-only)"):
        unsafe = """
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# eps = torch.randn(4, 10)   # implicitly lands on CUDA
"""
        print(unsafe.strip())
        print("Problem: 'torch.cuda.FloatTensor' does not exist on CPU-only")
        print("         or Apple-Silicon (MPS) machines. Importing the module")
        print("         that contains the set_default_tensor_type call will")
        print("         raise immediately. The setting is process-wide global.")

    with section("the safe pattern (matches recvae/model.py:135-146)"):
        # Always pass device= and dtype= explicitly. The tensor lands on
        # exactly the device you asked for, with the dtype you asked for,
        # no global side effects.
        cpu = torch.device("cpu")
        mu_h = torch.zeros(4, 10, device=cpu, dtype=torch.float32)
        eps = torch.randn(mu_h.shape, device=mu_h.device, dtype=mu_h.dtype)

        print(f"mu_h.device  = {mu_h.device}")
        print(f"mu_h.dtype   = {mu_h.dtype}")
        print(f"eps.device   = {eps.device}      (inherited from mu_h)")
        print(f"eps.dtype    = {eps.dtype}    (inherited from mu_h)")

        banner("a tensor explicitly placed on a chosen device")
        # Show that explicit device= really is explicit: ask for CPU.
        chosen = torch.device("cpu")
        x = torch.zeros(2, 3, device=chosen)
        print(f"x            = {x}")
        print(f"x.device     = {x.device}")
        assert x.device == chosen

    with section("how the recvae package handles this"):
        print("recvae/model.py:135-146 -- reparametrize() passes device=mu_h.device")
        print("recvae/data.py:142-148  -- DataLoader uses CPU generator")
        print("recvae/utils.py:29-35   -- get_default_device picks CUDA/MPS/CPU")
        print("No torch.set_default_tensor_type anywhere in the package.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
