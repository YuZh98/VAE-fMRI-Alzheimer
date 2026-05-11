"""Demo: device picker, explicit device= in tensor constructors, DeviceDataLoader."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from recvae import DeviceDataLoader, get_default_device, to_device


class LabeledTensorDataset(Dataset):
    """Yield (tensor, integer_label) pairs.

    The label exercises to_device's integer-dtype preservation: it must
    arrive on the target device with its integer dtype intact, otherwise
    indexing operations downstream would silently break.
    """

    def __init__(self, n: int, dim: int):
        self.x = torch.randn(n, dim)
        self.y = torch.arange(n, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def main() -> int:
    with section("device picker"):
        device = get_default_device()
        print(f"get_default_device() -> {device}")
        print(f"cuda available?  {torch.cuda.is_available()}")
        print(f"mps available?   {torch.backends.mps.is_available()}")

    with section("explicit device on tensor construction"):
        # No global set_default_tensor_type. The device flows through the
        # explicit device= argument and the module's .to(device) call.
        linear = nn.Linear(4, 4).to(device)
        x = torch.randn(2, 4, device=device)
        y = linear(x)
        print(f"input device : {x.device}")
        print(f"output device: {y.device}")
        print(f"output dtype : {y.dtype}")
        assert y.device.type == device.type, "output landed on wrong device"

    with section("DeviceDataLoader moves each batch"):
        ds = LabeledTensorDataset(n=6, dim=4)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        ddl = DeviceDataLoader(loader, device)

        banner("iterate one epoch")
        for i, (xb, yb) in enumerate(ddl):
            print(
                f"  batch {i}: x.shape={tuple(xb.shape)} x.device={xb.device} "
                f"x.dtype={xb.dtype} | y={yb.tolist()} y.device={yb.device} "
                f"y.dtype={yb.dtype}"
            )
            assert xb.device.type == device.type
            assert yb.device.type == device.type
            # Float tensors cast to float32; integer labels keep their integer dtype.
            assert xb.dtype == torch.float32, f"expected float32, got {xb.dtype}"
            assert yb.dtype == torch.long, f"expected long (int64), got {yb.dtype}"
        print("ok: all batches on device with correct dtypes")

    with section("to_device on a nested structure"):
        # Sanity-check the recursive branch: list/tuple of tensors.
        nested = [torch.randn(2, 3), torch.tensor([1, 2, 3], dtype=torch.long)]
        moved = to_device(nested, device)
        print(f"floats: device={moved[0].device} dtype={moved[0].dtype}")
        print(f"ints  : device={moved[1].device} dtype={moved[1].dtype}")
        assert moved[1].dtype == torch.long, "integer dtype must be preserved"

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
