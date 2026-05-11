"""Smoke test: confirm Python, torch, recvae, and a trivial tensor op all work."""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tutorials._tutorial_utils import section, banner

import torch

import recvae
from recvae import get_default_device


def main() -> int:
    with section("environment"):
        print(f"python  : {sys.version.split()[0]}")
        print(f"torch   : {torch.__version__}")
        print(f"recvae  : {recvae.__version__}")

    with section("device"):
        device = get_default_device()
        print(f"picked device: {device}")

    with section("trivial tensor op"):
        banner("torch.eye(3) @ torch.eye(3)")
        result = torch.eye(3) @ torch.eye(3)
        print(result)
        assert torch.allclose(result, torch.eye(3)), "identity @ identity must equal identity"
        print("ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
