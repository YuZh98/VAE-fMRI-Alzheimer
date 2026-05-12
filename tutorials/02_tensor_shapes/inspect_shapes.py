"""Walk the encoder/decoder spatial chain on a (1,1,91,109,91) tensor."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch
from torch import nn


ENCODER_CHAIN = [
    (45, 54, 45),
    (22, 27, 22),
    (11, 13, 11),
    (5, 6, 5),
]
DECODER_CHAIN = [
    (11, 13, 11),
    (22, 27, 22),
    (45, 54, 45),
    (91, 109, 91),
]


def main() -> int:
    with section("encoder: Conv3d k=4 s=2 p=1, channels 1->4->8->16->32"):
        encoder = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.Conv3d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
        )
        x = torch.randn(1, 1, 91, 109, 91)
        banner("input")
        print(tuple(x.shape))
        for stage, (layer, expected) in enumerate(zip(encoder, ENCODER_CHAIN), start=1):
            x = layer(x)
            spatial = tuple(x.shape[-3:])
            print(f"after encoder{stage}: shape={tuple(x.shape)}  spatial={spatial}  expected={expected}")
            assert spatial == expected, f"stage {stage}: got {spatial}, expected {expected}"

    with section("decoder: ConvTranspose3d k=4 s=2 p=1, with output_padding"):
        # output_padding values mirror recvae/model.py:129-152 exactly.
        decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1, output_padding=(0, 1, 0), bias=False),
            nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1, output_padding=(1, 0, 1), bias=False),
            nn.ConvTranspose3d(4, 1, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
        )
        h = torch.randn(1, 32, 5, 6, 5)
        banner("input")
        print(tuple(h.shape))
        for stage, (layer, expected) in enumerate(zip(decoder, DECODER_CHAIN), start=2):
            h = layer(h)
            spatial = tuple(h.shape[-3:])
            print(f"after decoder{stage}: shape={tuple(h.shape)}  spatial={spatial}  expected={expected}")
            assert spatial == expected, f"stage {stage}: got {spatial}, expected {expected}"

    with section("summary"):
        print("encoder: 91 -> 45 -> 22 -> 11 -> 5   (X axis)")
        print("decoder:  5 -> 11 -> 22 -> 45 -> 91  (X axis)")
        print("all asserts passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
