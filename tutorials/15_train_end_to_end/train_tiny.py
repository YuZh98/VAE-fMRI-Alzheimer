"""End-to-end training demo on synthetic data.

Wires together Config, set_seed, FMRIDataset, build_dataloader,
DeviceDataLoader, RecVAEModel, and fit. Trains for 3 epochs on 4
synthetic subjects with tol_time=4, then exercises a state_dict
save/reload round-trip.

Runs on CPU in well under 90s. No real fMRI data required.
"""

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    DeviceDataLoader,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    fit,
    get_default_device,
    normalize_per_subject,
    set_seed,
)


def main() -> int:
    set_seed(2022)
    # get_default_device() picks CUDA / MPS / CPU. CI runs on CPU; on a
    # Mac dev machine this resolves to MPS. RecVAE has known MPS-backend
    # quirks (the .item() call inside fit() can index oddly), so we
    # demote to CPU for the demo. The pattern below is the production
    # one — swap the conditional out for a plain assignment in real code.
    device = get_default_device()
    if device.type == "mps":
        banner("MPS detected; demoting to CPU for demo stability")
        device = torch.device("cpu")

    # Demo-sized config. lr=1e-5 is ten times the production default
    # (1e-6) — enough to move the loss visibly in three epochs without
    # diverging to NaN on random inputs. We initially tried 1e-4 but it
    # NaN'd by epoch 3 on this synthetic data, so we backed off to 1e-5.
    cfg = Config(tol_time=4, epochs=3, batch_size=2, learning_rate=1e-5)

    with section("synthesize 4 subjects, tol_time=4 timesteps"):
        N = 4
        volumes = torch.randn(N, 1, 91, 109, 91, cfg.tol_time)
        print(f"raw volumes shape : {tuple(volumes.shape)}")
        volumes, vmax, vmin = normalize_per_subject(volumes)
        print(f"after normalize   : min={volumes.min():.3f}, max={volumes.max():.3f}")
        print(f"per-subject vmax  : {vmax.tolist()}")
        print(f"per-subject vmin  : {vmin.tolist()}")

    with section("build dataset + dataloader + device wrapper"):
        ds = FMRIDataset(volumes)
        dl = build_dataloader(ds, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
        dl = DeviceDataLoader(dl, device)
        print(f"dataset size      : {len(ds)}")
        print(f"batches per epoch : {len(dl)}")
        print(f"device            : {device}")

    with section("instantiate RecVAEModel and h0"):
        model = RecVAEModel(train_size=N, cfg=cfg).to(device)
        h0 = torch.zeros(1, cfg.latent_dim, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model parameters  : {n_params:,}")
        print(f"F_mat buffer      : {tuple(model.F_mat.shape)} on {model.F_mat.device}")
        print(f"z_vectors param   : {tuple(model.z_vectors.shape)}")

    with section("train for 3 epochs (SGD inside batches, ridge update at epoch end)"):
        result = fit(model, dl, h0, cfg=cfg, epochs=cfg.epochs)
        history = result["train_loss_history"]
        print(f"train_loss_history: {history}")
        assert isinstance(history, list) and len(history) > 0, history

    with section("save / reload state_dict, confirm forward pass matches"):
        # F_mat (buffer) and z_vectors (Parameter) both live in state_dict
        # thanks to the refactor (Lesson 14). Verifying round-trip exercises
        # both.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(model.state_dict(), tmp_path)
            banner(f"wrote checkpoint to {tmp_path}")

            model2 = RecVAEModel(train_size=N, cfg=cfg).to(device)
            model2.load_state_dict(torch.load(tmp_path, map_location=device, weights_only=True))

            # Eval mode so BatchNorm uses running stats; pin RNG so the
            # reparameterization sample is the same for both models.
            model.eval()
            model2.eval()

            first_batch, first_idx = next(iter(dl))
            which = first_idx.long()
            h_batch = h0.expand(first_batch.size(0), -1)

            torch.manual_seed(0)
            _, mu1, _, _ = model(first_batch, h_batch, which)
            torch.manual_seed(0)
            _, mu2, _, _ = model2(first_batch, h_batch, which)

            agree = torch.allclose(mu1[0], mu2[0], atol=1e-6)
            print(f"first reconstruction matches: {agree}")
            assert agree, "state_dict round-trip changed model output"

            # Also confirm F_mat and z_vectors round-tripped exactly.
            assert torch.equal(model.F_mat, model2.F_mat), "F_mat mismatch"
            assert torch.equal(model.z_vectors, model2.z_vectors), "z_vectors mismatch"
            banner("F_mat and z_vectors round-tripped exactly")
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
            banner("cleaned up tempfile")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
