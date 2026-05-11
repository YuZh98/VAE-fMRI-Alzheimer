"""A toy nn.Module mirroring encoder1 from recvae/model.py:51-55."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch
from torch import nn


class ToyEncoderBlock(nn.Module):
    """Conv3d -> BatchNorm3d -> LeakyReLU, the canonical encoder unit.

    Same structure as ``RecVAEModel.encoder1`` (recvae/model.py:51-55), but
    with arbitrary in/out channel counts so we can use small inputs here.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def main() -> int:
    with section("instantiate toy encoder block"):
        block = ToyEncoderBlock(in_channels=1, out_channels=4)
        n_params = sum(p.numel() for p in block.parameters())
        print(block)
        print(f"trainable parameters: {n_params}")

    with section("forward on (1, 1, 16, 16, 16)"):
        # Small cube — this is a concept demo, not the real 91^3 volume.
        x = torch.randn(1, 1, 16, 16, 16)
        # BatchNorm with batch=1 in train mode would error on the running
        # statistics update for some configs; switch to eval so the demo is
        # robust regardless of BN internals.
        block.eval()
        y = block(x)
        banner("shapes")
        print(f"input  shape: {tuple(x.shape)}")
        print(f"output shape: {tuple(y.shape)}")
        # k=4, s=2, p=1 halves an even dim: 16 -> 8.
        assert tuple(y.shape) == (1, 4, 8, 8, 8)

    with section("module tree"):
        # nn.Module discovers submodules by attribute assignment.
        for name, child in block.named_children():
            print(f"{name:6s} -> {type(child).__name__}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
