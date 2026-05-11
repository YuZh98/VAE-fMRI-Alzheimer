"""Compare nn.Parameter, register_buffer, and plain tensor attributes.

Demonstrate which of the three appear in ``.parameters()``, in
``.named_buffers()``, and in ``.state_dict()``, then prove which round-
trip through ``torch.save`` / ``torch.load`` and which silently do not.
"""

import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402


class Tiny(nn.Module):
    """One Parameter, one buffer, one plain tensor attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((3,), 1.0))
        self.register_buffer("running_mean", torch.full((3,), 2.0))
        # Plain attribute -> NOT tracked by any module bookkeeping.
        self.cache = torch.full((3,), 3.0)


def main() -> None:
    m = Tiny()

    with section("membership in the three module registries"):
        params = dict(m.named_parameters())
        bufs = dict(m.named_buffers())
        sd = m.state_dict()

        for name in ("weight", "running_mean", "cache"):
            print(
                f"{name:14s} | in .parameters(): {str(name in params):5s}"
                f" | in .named_buffers(): {str(name in bufs):5s}"
                f" | in .state_dict(): {str(name in sd):5s}"
            )

    with section("round-trip through torch.save / torch.load"):
        # Mutate every attribute on the original so we can tell whether a
        # value got persisted (mutation survives reload) vs reinitialized
        # (fresh module's init value reappears).
        with torch.no_grad():
            m.weight.fill_(7.0)
            m.running_mean.fill_(8.0)
        m.cache = torch.full((3,), 9.0)

        banner("before save")
        print(f"weight       : {m.weight.detach().tolist()}")
        print(f"running_mean : {m.running_mean.tolist()}")
        print(f"cache        : {m.cache.tolist()}")

        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pt", delete=False
        )
        tmp.close()
        try:
            torch.save(m.state_dict(), tmp.name)

            fresh = Tiny()
            fresh.load_state_dict(torch.load(tmp.name, weights_only=True))

            banner("after load into a fresh Tiny()")
            print(f"weight       : {fresh.weight.detach().tolist()}  (Parameter, persisted)")
            print(f"running_mean : {fresh.running_mean.tolist()}  (buffer, persisted)")
            print(f"cache        : {fresh.cache.tolist()}  (plain attr, NOT persisted)")
            print()
            print(
                "Note: 'cache' is the init value 3.0, not the mutated 9.0 — "
                "torch.save never saw it because it was never in state_dict()."
            )
        finally:
            os.unlink(tmp.name)


if __name__ == "__main__":
    main()
