"""Round-trip a RecVAEModel through save -> load and verify equality.

The key check: with both models in .eval() (so BatchNorm uses running
stats from the state_dict, not freshly computed batch stats), forwarding
the same input through both must produce element-wise identical output.

Runs on CPU in well under a minute with train_size=2 and tol_time=4.
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import os
import tempfile

import torch

from recvae import Config, RecVAEModel, set_seed


def main() -> int:
    set_seed(0)
    cfg = Config(tol_time=4)
    B, T = 2, 4

    with section("build model A and run one forward"):
        model_a = RecVAEModel(train_size=B, cfg=cfg)
        x = torch.randn(B, 1, 91, 109, 91, T)
        h0 = torch.zeros(B, model_a.latent_dim)
        which = torch.arange(B, dtype=torch.long)
        _ = model_a(x, h0, which)
        print(f"forward ok on input shape {tuple(x.shape)}")

    with section("save state_dict to a temp file"):
        # delete=False so we control teardown and can re-open by path on Windows.
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.close()
        ckpt_path = tmp.name
        torch.save(model_a.state_dict(), ckpt_path)
        size = os.path.getsize(ckpt_path)
        print(f"wrote {ckpt_path} ({size} bytes)")

    try:
        with section("inspect what is in the state_dict"):
            sd = model_a.state_dict()
            banner(f"{len(sd)} keys total")
            for name, tensor in sd.items():
                print(f"  {name:50s} {tuple(tensor.shape)}")
            assert "z_vectors" in sd, "Parameter z_vectors must be in state_dict"
            assert "F_mat" in sd, "Buffer F_mat must be in state_dict"

        with section("build model B and load_state_dict"):
            model_b = RecVAEModel(train_size=B, cfg=cfg)
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model_b.load_state_dict(loaded)
            print("load ok (strict=True by default)")

        with section("eval mode + same input -> identical outputs"):
            # eval() flips BatchNorm to running-stats mode so the forward is
            # deterministic given identical weights/buffers.
            model_a.eval()
            model_b.eval()
            # Reparametrize samples eps ~ N(0, I); re-seed so both models draw
            # the same epsilon sequence.
            set_seed(123)
            _, mus_a, _, _ = model_a(x, h0, which)
            set_seed(123)
            _, mus_b, _, _ = model_b(x, h0, which)
            for t, (a, b) in enumerate(zip(mus_a, mus_b)):
                assert torch.equal(a, b), f"timestep {t} reconstructions differ"
            print(f"all {len(mus_a)} reconstructions match element-wise")
    finally:
        # tempfile.NamedTemporaryFile(delete=False) requires manual cleanup.
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
            print(f"\ncleaned up {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
