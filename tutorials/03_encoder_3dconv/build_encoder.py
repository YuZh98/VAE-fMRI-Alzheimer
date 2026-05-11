"""Build the encoder from scratch, then compare shapes against RecVAEModel."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch
from torch import nn

from recvae import Config, RecVAEModel


def build_from_scratch(cfg: Config) -> nn.Sequential:
    """Mirror RecVAEModel.encoder1..encoder5 (recvae/model.py:50-75)."""
    return nn.Sequential(
        nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(0.2, inplace=True),
        ),
        nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(0.2, inplace=True),
        ),
        nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
        ),
        nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 6 * 5, cfg.enc_out_dim),
            nn.Tanh(),
        ),
    )


def main() -> int:
    cfg = Config()
    x = torch.randn(1, 1, 91, 109, 91)

    with section("from-scratch encoder: stage-by-stage shapes"):
        scratch = build_from_scratch(cfg)
        scratch.eval()  # BN with batch=1: use running stats.
        h = x
        banner("input")
        print(tuple(h.shape))
        for i, stage in enumerate(scratch, start=1):
            h = stage(h)
            label = f"encoder{i}"
            print(f"after {label}: shape={tuple(h.shape)}")
        scratch_out_shape = tuple(h.shape)
        assert scratch_out_shape == (1, cfg.enc_out_dim), scratch_out_shape

    with section("real RecVAEModel.encode on the same input"):
        model = RecVAEModel(train_size=1, cfg=cfg)
        model.eval()
        with torch.no_grad():
            real_out = model.encode(x)
        real_out_shape = tuple(real_out.shape)
        print(f"real encode output shape: {real_out_shape}")
        # Same architecture -> same output shape. Weights differ, values differ.
        assert real_out_shape == scratch_out_shape, (real_out_shape, scratch_out_shape)

    with section("summary"):
        print(f"both encoders produced shape {scratch_out_shape}")
        print("(values differ — different random weights — but shapes match)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
