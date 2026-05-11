"""Walk a tensor through the RecVAE decoder, layer by layer.

Build the four ConvTranspose3d layers from scratch (no recvae import), push
a (1, 32, 5, 6, 5) volume through, and print the shape at every stage.

Then deliberately swap in the WRONG ``output_padding`` on one layer to see
the off-by-one failure mode.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402


def build_decoder(
    op2=1,
    op3=(0, 1, 0),
    op4=(1, 0, 1),
    op5=1,
):
    """Return the four-layer transposed-conv stack with given output_paddings."""
    return nn.ModuleList([
        nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1,
                           output_padding=op2, bias=False),
        nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1,
                           output_padding=op3, bias=False),
        nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1,
                           output_padding=op4, bias=False),
        nn.ConvTranspose3d(4, 1, kernel_size=4, stride=2, padding=1,
                           output_padding=op5, bias=False),
    ])


def run_chain(layers, x):
    """Push x through layers, printing shape after each."""
    print(f"input          : {tuple(x.shape)}")
    for i, layer in enumerate(layers, start=2):
        x = layer(x)
        print(f"after decoder{i}: {tuple(x.shape)}")
    return x


def main() -> None:
    torch.manual_seed(0)

    with section("correct output_padding -> recovers (91, 109, 91)"):
        x = torch.randn(1, 32, 5, 6, 5)
        layers = build_decoder()
        out = run_chain(layers, x)
        target = (1, 1, 91, 109, 91)
        print(f"target shape   : {target}")
        print(f"match          : {tuple(out.shape) == target}")

    with section("wrong output_padding on decoder3 -> silent off-by-one"):
        # decoder3 should use output_padding=(0, 1, 0) because Y=13 is odd.
        # If we naively use scalar 0, the Y dim becomes 26 instead of 27 and
        # the error only surfaces when we try to compute reconstruction loss.
        x = torch.randn(1, 32, 5, 6, 5)
        layers = build_decoder(op3=0)
        out = run_chain(layers, x)
        # The Y off-by-one at decoder3 (26 instead of 27) doubles at each
        # subsequent stride-2 layer, so by decoder5 Y is 105, not 109.
        print(f"final shape    : {tuple(out.shape)}  <-- Y collapsed to 105")

        banner("attempting reconstruction loss against true shape")
        fake_x = torch.randn(1, 1, 91, 109, 91)
        try:
            loss = (fake_x - out).pow(2).sum()
            print(f"loss computed (this should NOT happen): {loss.item():.4f}")
        except RuntimeError as exc:
            # This is the bug surfacing at the loss line, not the decoder.
            print(f"RuntimeError at loss line:\n  {exc}")


if __name__ == "__main__":
    main()
