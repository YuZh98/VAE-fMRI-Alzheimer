"""Pitfall: plain self.foo = torch.randn(...) is invisible to PyTorch.

Defines a Module with three attributes:
  - self.weight       : nn.Parameter  (trainable, in state_dict)
  - self.const        : registered buffer (not trainable, in state_dict)
  - self.plain        : plain torch.Tensor (invisible to all of the above)

Prints which appear in .parameters(), .buffers(), .state_dict(), and
demonstrates that .plain does NOT round-trip through torch.save /
torch.load(state_dict).
"""

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402


class ToyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Trainable. Will be in parameters() and state_dict().
        self.weight = nn.Parameter(torch.full((3,), 1.0))
        # Not trainable, but persistent. In state_dict(), follows .to(device).
        self.register_buffer("const", torch.full((3,), 2.0))
        # PLAIN TENSOR -- INVISIBLE. Not in parameters(), not in buffers(),
        # not in state_dict(), not moved by .to(device).
        self.plain = torch.full((3,), 3.0)


def main() -> int:
    torch.manual_seed(0)

    with section("which attrs does PyTorch see?"):
        m = ToyModule()
        param_names = [n for n, _ in m.named_parameters()]
        buffer_names = [n for n, _ in m.named_buffers()]
        sd_keys = list(m.state_dict().keys())

        print(f"named_parameters: {param_names}")
        print(f"named_buffers   : {buffer_names}")
        print(f"state_dict keys : {sd_keys}")

        assert "weight" in param_names
        assert "const" in buffer_names
        assert "plain" not in param_names
        assert "plain" not in buffer_names
        assert "plain" not in sd_keys, "plain should be missing"

    with section("the optimizer sees only Parameters"):
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        # One param group with one tensor (self.weight).
        n_managed = sum(p.numel() for g in opt.param_groups for p in g["params"])
        print(f"optimizer manages {n_managed} numbers (just self.weight)")
        print(f"self.plain.numel() = {m.plain.numel()}   (NOT managed)")

    with section("save / load round-trip: plain does NOT survive"):
        # Mutate all three so a successful round-trip would be detectable.
        with torch.no_grad():
            m.weight.fill_(10.0)
            m.const.fill_(20.0)
            m.plain.fill_(30.0)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(m.state_dict(), tmp_path)

            m2 = ToyModule()  # plain reinit'd to 3.0 by __init__
            m2.load_state_dict(torch.load(tmp_path, weights_only=True))

            print(f"after load: m2.weight = {m2.weight.data.tolist()}   (10.0 -- round-tripped)")
            print(f"after load: m2.const  = {m2.const.tolist()}   (20.0 -- round-tripped)")
            print(f"after load: m2.plain  = {m2.plain.tolist()}    (still 3.0 -- DROPPED)")

            assert torch.equal(m2.weight.data, torch.full((3,), 10.0))
            assert torch.equal(m2.const, torch.full((3,), 20.0))
            assert torch.equal(m2.plain, torch.full((3,), 3.0)), \
                "plain attribute should NOT have round-tripped"
            banner("plain tensor was silently dropped from the checkpoint")
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    with section("how recvae avoids this (recvae/model.py:156-163)"):
        print("self.z_vectors = nn.Parameter(...)         # learnable subject noise")
        print("self.register_buffer('F_mat', ...)         # closed-form, no grad")
        print()
        print("Both appear in state_dict(); both move with .to(device);")
        print("only z_vectors gets gradients from .backward().")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
