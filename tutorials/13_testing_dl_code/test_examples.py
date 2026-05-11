"""Five illustrative pytest tests on a toy nn.Module.

The categories mirror what tests/test_model.py applies to RecVAEModel:
shape contract, gradient flow, determinism, parameter membership, and
error path. Kept on a toy Linear(4, 4) so the file is readable end to
end.

Runs as:

    python tutorials/13_testing_dl_code/test_examples.py
    pytest tutorials/13_testing_dl_code/test_examples.py -v

Not picked up by the project's default pytest run because
pyproject.toml sets testpaths=["tests"]. Intentional: this file is a
tutorial, not part of the real suite.
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section

import pytest
import torch
from torch import nn


class ToyModel(nn.Module):
    """Small stand-in for any real model: a Linear with a registered buffer."""

    def __init__(self, in_dim: int = 4, out_dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        # Buffer (like F_mat on RecVAEModel): part of state, but not learned.
        self.register_buffer("scale", torch.ones(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.fc.in_features:
            raise ValueError(
                f"expected last dim {self.fc.in_features}, got {x.shape[-1]}",
            )
        return self.fc(x) * self.scale


# 1. SHAPE CONTRACT -----------------------------------------------------------
def test_forward_preserves_batch_and_out_dim():
    model = ToyModel(in_dim=4, out_dim=4)
    x = torch.randn(8, 4)
    y = model(x)
    assert y.shape == (8, 4), y.shape


# 2. GRADIENT FLOW ------------------------------------------------------------
def test_backward_fills_param_grads():
    model = ToyModel()
    x = torch.randn(2, 4)
    loss = model(x).pow(2).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite grad"


# 3. DETERMINISM --------------------------------------------------------------
def test_seeded_forward_is_reproducible():
    def run() -> torch.Tensor:
        torch.manual_seed(7)
        model = ToyModel()
        x = torch.randn(3, 4)
        return model(x)

    a = run()
    b = run()
    assert torch.equal(a, b), "same seed produced different outputs"


# 4. PARAMETER MEMBERSHIP -----------------------------------------------------
def test_param_vs_buffer_membership():
    model = ToyModel()
    param_names = {n for n, _ in model.named_parameters()}
    buffer_names = {n for n, _ in model.named_buffers()}
    state_keys = set(model.state_dict().keys())

    assert "fc.weight" in param_names
    assert "fc.bias" in param_names
    assert "scale" not in param_names, "buffer leaked into .parameters()"
    assert "scale" in buffer_names, "buffer missing from .named_buffers()"
    # Both Parameters AND Buffers must be in state_dict — that is the contract.
    assert {"fc.weight", "fc.bias", "scale"} <= state_keys


# 5. ERROR PATH ---------------------------------------------------------------
def test_forward_raises_on_wrong_last_dim():
    model = ToyModel(in_dim=4)
    with pytest.raises(ValueError, match="expected last dim 4"):
        model(torch.randn(2, 5))


if __name__ == "__main__":
    with section("running illustrative tests via pytest.main"):
        # Exit code from pytest is propagated so this script signals
        # pass/fail to the caller (CI will check exit 0).
        sys.exit(pytest.main([__file__, "-v"]))
