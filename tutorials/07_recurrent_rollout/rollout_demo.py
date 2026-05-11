"""Roll RecVAEModel over a few timesteps and inspect what comes out.

Uses batch=2 and tol_time=4 so the full (91, 109, 91) encoder/decoder
still runs in well under a minute on a CPU.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402

from recvae import Config, RecVAEModel  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    cfg = Config(tol_time=4)
    model = RecVAEModel(train_size=2, cfg=cfg)
    model.eval()  # avoid BN stat updates from a single forward pass

    B, T, D = 2, cfg.tol_time, cfg.latent_dim
    x = torch.randn(B, 1, 91, 109, 91, T)
    h_0 = torch.zeros(B, D)
    which_ones = torch.tensor([0, 1])

    with section(f"forward(x, h_0, which_ones)  B={B}, T={T}, D={D}"):
        with torch.no_grad():
            x_list, mu_history, h_history, gh_history = model(x, h_0, which_ones)

        banner("list lengths")
        print(f"len(x_list)    = {len(x_list)}")
        print(f"len(mu_history) = {len(mu_history)}")
        print(f"len(h_history) = {len(h_history)}")
        print(f"len(gh_history)= {len(gh_history)}")

        banner("per-element shapes")
        print(f"x_list[0]    : {tuple(x_list[0].shape)}    (one timeslice, T axis sliced off)")
        print(f"mu_history[0]: {tuple(mu_history[0].shape)} (decoder output)")
        print(f"h_history[0] : {tuple(h_history[0].shape)}              (posterior latent)")
        print(f"gh_history[0]: {tuple(gh_history[0].shape)}              (g(h_{{t-1}}) = F @ h)")

    with section("latent state evolves across t"):
        # Stack list of (B, D) into (B, T, D) for inspection.
        h_stacked = torch.stack(h_history, dim=1)
        print(f"stacked h_history shape: {tuple(h_stacked.shape)}")
        banner("per-timestep mean / std across batch")
        for t in range(T):
            ht = h_stacked[:, t]
            print(
                f"t={t}: mean={ht.mean().item():+.4f}  "
                f"std={ht.std().item():.4f}"
            )
        # Quick sanity check: at t=0 g(h_0) is zero because h_0 is zeros.
        print(f"\ngh_history[0] norm (should be 0): "
              f"{gh_history[0].abs().sum().item():.4f}")


if __name__ == "__main__":
    main()
