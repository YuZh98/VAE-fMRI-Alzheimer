"""Hand-roll the VAE reparameterization trick and verify gradient flow.

Also confirm that ``torch.randn`` defaults to CPU, and that the only way
to land an ``eps`` tensor on a non-CPU device is to pass ``device=...``
explicitly.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402

from recvae import Config, RecVAEModel  # noqa: E402


def reparametrize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """z = mu + sigma * eps, with eps ~ N(0, I) on mu's device/dtype."""
    sigma = torch.exp(log_var / 2)
    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype)
    return mu + sigma * eps


def main() -> None:
    torch.manual_seed(0)

    with section("from-scratch reparametrize on (2, 10) mu, log_var"):
        mu = torch.zeros(2, 10, requires_grad=True)
        log_var = torch.zeros(2, 10, requires_grad=True)
        z = reparametrize(mu, log_var)
        print(f"z shape : {tuple(z.shape)}")
        print(f"z device: {z.device}")
        print(f"z dtype : {z.dtype}")
        print(f"z[0, :5]: {z[0, :5].detach().tolist()}")

        banner("gradient flow")
        # Sum to a scalar so we can backprop. mu and log_var should both
        # receive gradients because z = mu + exp(log_var/2) * eps.
        loss = z.sum()
        loss.backward()
        print(f"mu.grad is None     : {mu.grad is None}")
        print(f"log_var.grad is None: {log_var.grad is None}")
        print(f"mu.grad[0, :5]      : {mu.grad[0, :5].tolist()}")
        print(f"log_var.grad[0, :5] : {log_var.grad[0, :5].tolist()}")
        # d/dmu z.sum() = 1 elementwise, so each mu.grad entry is 1.
        # d/dlogvar z.sum() = 0.5 * exp(log_var/2) * eps, which depends on eps.

    with section("RecVAEModel.reparametrize matches in shape/device"):
        cfg = Config(tol_time=3)
        model = RecVAEModel(train_size=2, cfg=cfg)
        mu2 = torch.zeros(2, cfg.latent_dim)
        log_var2 = torch.zeros(2, cfg.latent_dim)
        z2 = model.reparametrize(mu2, log_var2)
        print(f"shape : {tuple(z2.shape)}")
        print(f"device: {z2.device}")
        print(f"dtype : {z2.dtype}")

    with section("torch.randn defaults to CPU regardless of context"):
        # No `device=` argument -> CPU. This is what the old notebook
        # relied on `set_default_tensor_type` to override globally.
        eps_default = torch.randn(2, 10)
        print(f"default torch.randn device: {eps_default.device}")

        # Passing device= is the only correct way to control placement.
        eps_cpu_explicit = torch.randn(2, 10, device=torch.device("cpu"))
        print(f"explicit cpu device       : {eps_cpu_explicit.device}")
        print("(this script targets CPU; the same call with device='cuda'")
        print(" or device='mps' would land there if the device exists)")


if __name__ == "__main__":
    main()
