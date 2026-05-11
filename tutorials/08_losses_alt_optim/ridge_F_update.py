"""Demo: closed-form ridge update for F vs. hand-rolled torch.linalg.solve."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch

from recvae import Config, RecVAEModel


def manual_F_ridge(
    h_history: torch.Tensor,
    h0: torch.Tensor,
    rho: float,
    sig_h: float,
) -> torch.Tensor:
    """Reference implementation of the same math as RecVAEModel.updating_F.

    Mirrors recvae/model.py. Kept here as a separate function so a reader
    can diff the two and confirm they agree line for line.

    Derivation
    ----------
    The SGD ``loss2`` in :meth:`RecVAEModel.training_step` is averaged over
    batch and time:

        loss2 = (1 / (2 N T sig_h^2)) * sum_{n,t} ||h_t - g(h_{t-1})||^2,

    and the F-only regularizer is ``rho * ||F||_F^2``. Differentiating the
    sum with respect to F at zero gives the normal equation

        (X^T X + 2 N T sig_h^2 rho I) F^T = X^T Y.

    The ``2 N T`` factor that comes from the per-volume averaging cannot
    be dropped — without it the closed form solves a different objective
    than the SGD loss.
    """
    N, T, D = h_history.shape
    Y = h_history.reshape(-1, D)
    # Build X as h_history shifted by one timestep, with h0 prepended at t=0.
    x_shifted = torch.empty_like(h_history)
    x_shifted[:, 0] = h0
    x_shifted[:, 1:] = h_history[:, :-1]
    X = x_shifted.reshape(-1, D)

    # The factor 2 * N * T * sig_h^2 comes from how loss2 is averaged in
    # training_step (per-volume mean).
    rho_I = 2 * N * T * (sig_h ** 2) * rho * torch.eye(D, dtype=h_history.dtype)

    # Normal equations: (X^T X + rho_I) F^T = X^T Y.
    F_T = torch.linalg.solve(X.T @ X + rho_I, X.T @ Y)
    return F_T.T


def main() -> int:
    torch.manual_seed(0)

    N, T, D = 4, 8, 5
    rho = 0.3

    with section("synthesize posterior history"):
        h_history = torch.randn(N, T, D)
        h0 = torch.randn(1, D)
        print(f"h_history shape: {tuple(h_history.shape)}")
        print(f"h0 shape       : {tuple(h0.shape)}")
        print(f"rho            : {rho}")

    with section("manual ridge solve"):
        cfg = Config(latent_dim=D, tol_time=T)
        F_manual = manual_F_ridge(h_history, h0, rho, sig_h=cfg.sig_h)
        banner("F_manual (D x D)")
        print(F_manual)

    with section("RecVAEModel.updating_F"):
        # train_size matches N so the model's z_vectors size is consistent,
        # but updating_F itself does not touch z_vectors.
        model = RecVAEModel(train_size=N, cfg=cfg)
        model.updating_F(h_history, h0, rho=rho)
        banner("model.F_mat after update")
        print(model.F_mat)

    with section("compare"):
        max_abs_diff = (model.F_mat - F_manual).abs().max().item()
        print(f"max |F_model - F_manual| = {max_abs_diff:.3e}")
        assert torch.allclose(model.F_mat, F_manual, atol=1e-5), (
            "closed-form F update disagrees with hand-rolled ridge solve"
        )
        print("ok: model.F_mat matches the hand solve")

    with section("residual sanity"):
        # ||Y - X F^T|| should be finite and non-negative; smoke check only.
        Y = h_history.reshape(-1, D)
        x_shifted = torch.empty_like(h_history)
        x_shifted[:, 0] = h0
        x_shifted[:, 1:] = h_history[:, :-1]
        X = x_shifted.reshape(-1, D)
        residual = (Y - X @ F_manual.T).norm().item()
        print(f"||Y - X F^T||_F = {residual:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
