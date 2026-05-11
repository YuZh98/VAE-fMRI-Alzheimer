"""Sketch of the proper KL term that would replace loss2.

Implements the closed form of KL(N(mu_h, sigma_h^2 I) || N(gh_prev, sig_h^2 I)),
verifies it is non-negative for random inputs, and shows where it would
slot into RecVAEModel.training_step. Does NOT modify recvae/model.py.

Reader exercise: take temporal_kl_term below, plug it into training_step
at the marked location (see the TODO comment), retrain, compare to the
current loss2 behavior.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402


def temporal_kl_term(
    mu_h: torch.Tensor,
    log_var_h: torch.Tensor,
    gh_prev: torch.Tensor,
    sig_h: float,
) -> torch.Tensor:
    """Closed-form KL(q || p) for two diagonal Gaussians.

        q(h_t) = N(mu_h, sigma_h^2 I)        # inference output
        p(h_t) = N(gh_prev, sig_h^2 I)       # temporal prior

    The per-element formula is

        kl_d = log(sig_h / sigma_h_d)
             + (sigma_h_d^2 + (mu_h_d - gh_prev_d)^2) / (2 sig_h^2)
             - 1/2

    Summed over the latent dimension and averaged across the batch.

    Parameters
    ----------
    mu_h     : (B, D) posterior mean from hidden2mu
    log_var_h: (B, D) posterior log-variance from hidden2log_var
    gh_prev  : (B, D) g(h_{t-1}) = h_{t-1} @ F^T
    sig_h    : scalar fixed std of the temporal prior

    Returns
    -------
    scalar KL summed over latent dim and averaged over batch.
    """
    # sigma_h_d^2 = exp(log_var_h)
    var_q = torch.exp(log_var_h)
    log_sig_q = log_var_h / 2  # log(sigma_h_d) = log_var_h / 2

    # term-by-term, then sum over latent dim D, then mean over batch B.
    per_dim = (
        torch.log(torch.tensor(sig_h, device=mu_h.device, dtype=mu_h.dtype)) - log_sig_q
        + (var_q + (mu_h - gh_prev) ** 2) / (2 * sig_h ** 2)
        - 0.5
    )
    return per_dim.sum(dim=-1).mean()


def main() -> int:
    torch.manual_seed(0)

    B, D = 8, 10
    sig_h = 1.0

    with section("KL >= 0 for random inputs"):
        for trial in range(5):
            mu_h = torch.randn(B, D)
            log_var_h = torch.randn(B, D) * 0.5  # keeps sigma in a reasonable range
            gh_prev = torch.randn(B, D)
            kl = temporal_kl_term(mu_h, log_var_h, gh_prev, sig_h)
            print(f"trial {trial}: KL = {kl.item():.4f}")
            assert kl.item() >= -1e-6, f"KL should be non-negative, got {kl.item()}"
        banner("all trials non-negative")

    with section("limiting case: q == p -> KL = 0"):
        # If mu_h == gh_prev and exp(log_var_h/2) == sig_h, then q == p and KL=0.
        mu_h = torch.randn(B, D)
        gh_prev = mu_h.clone()
        # sigma = sig_h => log_var = 2 * log(sig_h)
        log_var_h = torch.full((B, D), 2 * torch.log(torch.tensor(sig_h)).item())
        kl = temporal_kl_term(mu_h, log_var_h, gh_prev, sig_h)
        print(f"KL(q == p) = {kl.item():.6e}   (should be ~0)")
        assert abs(kl.item()) < 1e-5, f"KL(q == p) should be ~0, got {kl.item()}"

    with section("KL grows when q drifts from p"):
        mu_h = torch.zeros(B, D)
        log_var_h = torch.zeros(B, D)  # sigma_h = 1.0 = sig_h
        for offset in [0.0, 0.5, 1.0, 2.0, 4.0]:
            gh_prev = torch.full((B, D), offset)
            kl = temporal_kl_term(mu_h, log_var_h, gh_prev, sig_h)
            print(f"||mu - gh|| ~ {offset:.1f}  ->  KL = {kl.item():.4f}")

    with section("differentiability sanity check"):
        # Confirm we can backprop through the KL.
        mu_h = torch.randn(B, D, requires_grad=True)
        log_var_h = torch.randn(B, D, requires_grad=True)
        gh_prev = torch.randn(B, D, requires_grad=True)
        kl = temporal_kl_term(mu_h, log_var_h, gh_prev, sig_h)
        kl.backward()
        print(f"mu_h.grad      finite: {torch.isfinite(mu_h.grad).all().item()}")
        print(f"log_var_h.grad finite: {torch.isfinite(log_var_h.grad).all().item()}")
        print(f"gh_prev.grad   finite: {torch.isfinite(gh_prev.grad).all().item()}")

    with section("integration stub (do not run as-is)"):
        # This block shows where temporal_kl_term would slot into
        # RecVAEModel.training_step. It is NOT executed; it is reference
        # material for the reader to wire up.
        sketch = '''
# Inside RecVAEModel.training_step (recvae/model.py:257-302):
#
#     x_list, mu_history, h_history, gh_history = self(batch, h_0, which_ones)
#     ...
#     # TODO: integrate this into RecVAEModel.training_step.
#     # Replace the current loss2 (MSE proxy) with the closed-form KL:
#     #
#     #     loss2 = sum_t temporal_kl_term(mu_h_t, log_var_h_t,
#     #                                    gh_history[t], cfg.sig_h)
#     #     loss2 = loss2 / len(h_history)   # average over time
#     #
#     # That requires keeping mu_h and log_var_h from each step. Currently
#     # vae_step only returns the sampled h. Add a fourth return list, e.g.
#     # mu_h_history, log_var_h_history, and accumulate them in forward().
#
#     loss = loss1 + loss2 + loss_z
'''
        print(sketch.strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
