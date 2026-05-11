"""Replace every BatchNorm3d in the model with GroupNorm.

Variation from canonical
------------------------
- BatchNorm with small batch sizes (B=2-4) is statistically noisy. A
  common production swap is GroupNorm, which normalizes per-sample over
  groups of channels and so does not depend on batch size.
- Walk the model with ``named_modules()`` and swap each ``BatchNorm3d``
  for ``GroupNorm(num_groups=min(channels, 8), num_channels=channels)``
  in place. Verify a forward pass still produces the right output shape.
- Train for 2 epochs on a tiny cohort and print loss-decrease.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    fit,
    set_seed,
    synthetic_cohort,
)


def swap_bn_for_gn(model: nn.Module) -> int:
    """In-place replace BatchNorm3d -> GroupNorm. Returns number swapped."""
    count = 0
    # Iterate over (parent_name, parent_module) so we can setattr() on
    # the parent. We collect first to avoid mutating during iteration.
    to_swap = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                to_swap.append((module, child_name, child.num_features))
    for parent, child_name, num_features in to_swap:
        gn = nn.GroupNorm(num_groups=min(num_features, 8),
                          num_channels=num_features)
        setattr(parent, child_name, gn)
        count += 1
    return count


def main() -> int:
    set_seed(2022)
    vols, _ = synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=2022)
    cfg = Config(tol_time=4, epochs=2, batch_size=2, learning_rate=1e-5)
    model = RecVAEModel(train_size=vols.shape[0], cfg=cfg)

    n_swapped = swap_bn_for_gn(model)
    print(f"swapped {n_swapped} BatchNorm3d modules for GroupNorm")

    # Sanity: model has no BatchNorm3d remaining.
    leftover = [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm3d)]
    assert not leftover, f"unexpected leftover BN modules: {leftover}"

    # Forward-pass shape check before training.
    model.eval()
    with torch.no_grad():
        h0 = torch.zeros(1, cfg.latent_dim)
        out = model(vols, h0.expand(vols.size(0), -1), torch.arange(vols.size(0)))
    expected_mu = (vols.size(0), cfg.tol_time, 1, 91, 109, 91)
    print(f"forward pass output shape: {tuple(out.mu.shape)} "
          f"(expected {expected_mu})")
    assert tuple(out.mu.shape) == expected_mu

    # Short training run on the swapped model.
    dl = build_dataloader(FMRIDataset(vols), batch_size=2, shuffle=True, seed=2022)
    h0 = torch.zeros(1, cfg.latent_dim)
    result = fit(model, dl, h0, cfg=cfg, epochs=cfg.epochs)
    hist = result["train_loss_history"]
    print(f"loss history: {[round(x, 3) for x in hist]}")
    print(f"delta(first -> last): {hist[-1] - hist[0]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
