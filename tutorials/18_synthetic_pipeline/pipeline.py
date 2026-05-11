"""End-to-end pipeline on synthetic data: generate -> train -> held-out eval.

Wires `synthetic_cohort` (Lesson 09 + recvae/data.py) into the same loop
demonstrated in Lesson 15, then adds the proper held-out evaluation
implemented in `recvae.evaluate_held_out`.

No real fMRI data, no external dependencies beyond `recvae` itself.
Runs on CPU in well under 60 seconds.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tutorials._tutorial_utils import section, banner  # noqa: E402

import torch  # noqa: E402

from recvae import (  # noqa: E402
    Config,
    DeviceDataLoader,
    FMRIDataset,
    RecVAEModel,
    build_dataloader,
    evaluate_held_out,
    fit,
    get_default_device,
    normalize_per_subject,
    set_seed,
    synthetic_cohort,
)


def main() -> int:
    set_seed(2022)
    # MPS has known quirks with this model (see Lesson 15). Demote to CPU.
    device = get_default_device()
    if device.type == "mps":
        banner("MPS detected; demoting to CPU for demo stability")
        device = torch.device("cpu")

    # Demo-sized config. lr=1e-5 matches Lesson 15's rationale: ten times the
    # production default so the loss moves visibly in two epochs without
    # diverging to NaN on random-ish input.
    cfg = Config(tol_time=4, epochs=2, batch_size=2, learning_rate=1e-5)

    with section("synthesize cohort (2 CN + 2 AD, tol_time=4)"):
        volumes, labels = synthetic_cohort(n_cn=2, n_ad=2, T=cfg.tol_time, seed=2022)
        print(f"volumes shape : {tuple(volumes.shape)}")
        print(f"labels        : {labels.tolist()}  (0=CN, 1=AD)")
        volumes, vmax, vmin = normalize_per_subject(volumes)
        print(f"after norm    : min={volumes.min():.3f}, max={volumes.max():.3f}")

    with section("build dataset, loader, model"):
        ds = FMRIDataset(volumes)
        dl = build_dataloader(ds, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
        dl = DeviceDataLoader(dl, device)
        model = RecVAEModel(train_size=len(ds), cfg=cfg).to(device)
        h0 = torch.zeros(1, cfg.latent_dim, device=device)
        print(f"device        : {device}")
        print(f"train_size    : {len(ds)}")
        print(f"latent_dim    : {cfg.latent_dim}")

    with section(f"train for {cfg.epochs} epochs"):
        result = fit(model, dl, h0, cfg=cfg, epochs=cfg.epochs)
        history = result["train_loss_history"]
        print(f"loss history  : {history}")
        if len(history) >= 2:
            delta = history[0] - history[-1]
            print(f"loss change   : {delta:+.4f} (first - last)")

    with section("held-out evaluation on a NEW synthetic cohort"):
        # Different seed => statistically independent draw from the same
        # generator. The model has never seen these subjects.
        vol_test, lab_test = synthetic_cohort(n_cn=1, n_ad=1, T=cfg.tol_time, seed=4242)
        vol_test, _, _ = normalize_per_subject(vol_test)
        print(f"test volumes  : {tuple(vol_test.shape)}")
        print(f"test labels   : {lab_test.tolist()}")
        eval_out = evaluate_held_out(
            model,
            vol_test.to(device),
            h0,
            inner_steps=5,
            inner_lr=1e-3,
        )
        print(f"held-out MSE  : {eval_out['recon_mse']:.6f}")
        print(f"z_test shape  : {tuple(eval_out['z_test'].shape)}")
        print(f"h_test shape  : {tuple(eval_out['h_test'].shape)}")

    banner("pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
