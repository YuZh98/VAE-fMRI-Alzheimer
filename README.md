# RecVAE — Recurrent 3D-Conv VAE for fMRI (Alzheimer's / ADNI)

> A PyTorch reference implementation + an 18-lesson tutorial series for
> people learning to model temporal fMRI data with deep variational
> autoencoders.

[![CI](https://github.com/YuZh98/VAE-fMRI-Alzheimer/actions/workflows/tutorials.yml/badge.svg)](https://github.com/YuZh98/VAE-fMRI-Alzheimer/actions/workflows/tutorials.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python ≥3.9](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch ≥2.0](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuZh98/VAE-fMRI-Alzheimer/blob/main/notebooks/RecVAE_on_synthetic.ipynb)

## Why this repo

This is the answer to "where do I start if I want to model fMRI with a VAE
in PyTorch?" It pairs a clean, tested, device-agnostic recurrent 3D-conv
VAE implementation with a synthetic-data on-ramp and an 18-lesson tutorial
series, so you can read, run, and extend the model without needing ADNI
access. Every example runs on CPU in seconds, and continuous integration
keeps every script working on every push.

## Who this is for

- **ML practitioner** new to neuroimaging — read [`docs/background/fmri_101.md`](docs/background/fmri_101.md) for the domain context.
- **Neuroimaging researcher** new to PyTorch — start with [`tutorials/`](tutorials/) and the synthetic notebook.
- **Course instructor** — every demo runs on CPU in seconds; CI guarantees they all work.

## Try it without data

```bash
git clone https://github.com/YuZh98/VAE-fMRI-Alzheimer
cd VAE-fMRI-Alzheimer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v                                              # 36 tests, ~5-20s, CPU-only
python tutorials/15_train_end_to_end/train_tiny.py     # synthetic, ~3s
```

> Note: the local directory name after clone is `VAE-fMRI-Alzheimer` (matches the GitHub repo); the Python package importable as `recvae`.

For a full end-to-end run on synthetic data, see
[`notebooks/RecVAE_on_synthetic.ipynb`](notebooks/RecVAE_on_synthetic.ipynb).

## 30-minute tour

1. Run `python tutorials/00_setup/verify_torch.py` (5s) — confirms environment.
2. Run `pytest -q` (~5-20s) — confirms the package works.
3. Open `tutorials/15_train_end_to_end/train_tiny.py` and run it (~3s) — your first end-to-end training run on synthetic data.
4. Open `notebooks/RecVAE_on_synthetic.ipynb` (run in Colab or locally) — full pipeline: synthesize data → train → held-out evaluation → linear-probe CN vs AD.
5. Browse `tutorials/README.md` and pick a lesson — the 18 lessons cover everything from 3D-conv arithmetic to research extensions.

## Tutorial series

[`tutorials/`](tutorials/) contains 18 hands-on lessons covering tensor
shapes, 3D-conv arithmetic, the reparameterization trick, recurrent
rollouts, alternating optimization, reproducibility, testing DL code, and
research extensions. Every lesson has a short README and at least one
runnable script. CI runs every script nightly (see `.github/workflows/tutorials.yml`); tests run on every push/PR via `tests.yml`.

## Visuals

| Loss curve | Reconstruction | Latent trajectory |
|---|---|---|
| ![loss](docs/assets/loss_curve.png) | ![recon](docs/assets/recon_slice.png) | ![latents](docs/assets/latent_trajectory.png) |

Generated from `notebooks/RecVAE_on_synthetic.ipynb` (5 epochs on 8 synthetic
subjects, CPU). Regenerate with `python tools/make_figures.py`.

## Architecture

The model is a **3D-convolutional VAE with a latent recurrence**:

```
                    ┌──────────────┐
   x_t ─────────────► 3D-CNN encode├──┐
   (1,91,109,91)    └──────────────┘  │  (B, 100)
                                      ▼
                       ┌────────────────────────────┐
   h_{t-1} ───────────►│  hidden2mu, hidden2log_var │
   (B, 10)             └────────────────────────────┘
                                      │
                                      ▼      ε ~ N(0, I)   (reparam)
                                 mu_h, log_var_h ────────────┐
                                                              ▼
                                                      h_t = mu_h + σ_h ε   (B, 10)
                                                              │
                                       z_s (subject noise) ───┤
                                                              ▼
                                                      h_t + z_s  ─────────┐
                                                                          ▼
                                                              ┌─────────────────────┐
                                                              │  3D-CNN decode      │
                                                              └─────────────────────┘
                                                                          │
                                                                          ▼
                                                                  μ_t  (B,1,91,109,91)

   Linear temporal prior: g(h) = h F^T  →  used in loss term  ‖h_t − g(h_{t-1})‖^2
```

For each subject's fMRI sequence (T=120 timepoints), the encoder collapses
each volume to a 100-dim feature; the inference head combines that with the
previous latent state `h_{t-1}` to produce a Gaussian posterior over `h_t`;
the decoder reconstructs the volume from `h_t + z_s` (where `z_s` is a
subject-specific noise vector, L1-regularized for sparsity).

The transition matrix **F** is **not** trained by gradient descent — it is
re-solved every epoch in closed form by ridge regression on the accumulated
posterior trajectory. The rest of the network (encoder, inference head,
decoder, `z_s`) is trained by SGD.

For design rationale, see [`docs/background/this_models_design.md`](docs/background/this_models_design.md).

## Loss

| Term     | Form                                    | How it's optimized       |
|----------|-----------------------------------------|--------------------------|
| `loss1`  | Per-volume reconstruction MSE / σ_x²    | SGD                      |
| `loss2`  | ‖h_t − g(h_{t-1})‖² / σ_h² (temporal)   | SGD                      |
| `loss_z` | λ_z · ‖z‖₁ (subject-noise sparsity)     | SGD                      |
| `loss_F` | ρ · ‖F‖_F² (reported, not back-propped) | Closed form (ridge)      |

> **Note.** `loss2` is an MSE proxy for the temporal prior, not the canonical
> VAE KL divergence. This is MAP-style point estimation of the latent path,
> not full variational inference. See "Known limitations" below.

## Repository layout

```
VAE-fMRI-Alzheimer/
├── recvae/                  # main Python package (extracted from notebooks)
│   ├── config.py            # Config dataclass — all hyperparameters
│   ├── data.py              # NIfTI loading, normalization, Dataset, DataLoader
│   ├── model.py             # RecVAEModel (encoder/decoder/inference/F update)
│   ├── train.py             # fit() and evaluate()
│   └── utils.py             # set_seed, device picker, DeviceDataLoader
├── tutorials/               # 18-lesson hands-on series (synthetic data, CPU-only)
├── examples/                # short end-to-end driver scripts
├── docs/                    # background notes (fMRI 101, design rationale, related work)
├── notebooks/
│   ├── RecVAE_on_fMRI.ipynb       # canonical driver (uses recvae package)
│   ├── RecVAE_on_synthetic.ipynb  # synthetic-data end-to-end demo
│   └── Input_images.ipynb         # data-inspection helpers
├── tests/                   # pytest suite (36 tests, ~5-20s, CPU-only)
├── legacy/                  # archived earlier iterations (V1–V4)
├── pyproject.toml           # PEP 621 metadata + pytest config
├── requirements.txt         # runtime deps
├── requirements-dev.txt     # + pytest, nbstripout
├── LICENSE                  # Apache-2.0
└── CITATION.cff             # how to cite this repo
```

## Hyperparameters

All in [`recvae/config.py`](recvae/config.py). Defaults match the canonical
notebook (RecVAE_on_fMRI.ipynb / Version4.ipynb):

| Field           | Default | Meaning                                         |
|-----------------|---------|-------------------------------------------------|
| `enc_out_dim`   | 100     | Encoder output dim before inference head        |
| `latent_dim`    | 10      | Posterior latent state dim, also the subject-noise dim (`z_vectors` shape `(N_train, latent_dim)`) |
| `tol_time`      | 120     | Truncate every fMRI sequence to this many points|
| `sig_x/sig_h/sig_z` | 1.0 | Fixed observation/process noise scales          |
| `rho`           | 0.1     | Ridge penalty for closed-form F update          |
| `lambda_z`      | 10.0    | L1 weight on subject noise                      |
| `batch_size`    | 4       |                                                 |
| `learning_rate` | 1e-6    | SGD lr                                          |
| `epochs`        | 500     |                                                 |
| `seed`          | 2022    |                                                 |

## Known limitations

The package preserves the original notebook's training math exactly. A few
choices flagged in the code review are **not** changed here because they
affect experimental results and need an owner decision:

- **No proper train/test split.** The original `test_loader` wraps the same
  dataset as `train_loader`. Reported reconstruction losses are training
  losses. Subject-level k-fold CV is recommended before any quantitative
  claim.
- **`loss2` is MSE, not KL.** The canonical VAE ELBO has
  `KL(q(h_t|·) || p(h_t|h_{t-1}))`. The current loss does point estimation
  on the latent path. See the `TODO(research)` comment in
  [`recvae/model.py`](recvae/model.py).
- **Noise scales are fixed, not learned.** Making `sig_x/sig_h/sig_z`
  trainable parameters would let the model calibrate its own loss weighting.
- **Per-subject min-max normalization** destroys absolute-intensity
  differences across subjects, which may matter for downstream
  classification.
- **SGD@1e-6 is a slow optimizer.** AdamW@~1e-4 with a scheduler is likely
  to converge faster.
- **No KL annealing / β-VAE.** Without an explicit KL term this is moot,
  but worth noting if you add one.

## What this is / what this isn't

This is:
- A reference implementation of a recurrent 3D-conv VAE for fMRI in PyTorch.
- A teaching artifact with 18 lessons + a synthetic on-ramp.
- A starting point you can fork and extend.

This is NOT:
- A published method paper (no preprint yet).
- A SOTA model — see [`docs/background/vae_for_neuroimaging.md`](docs/background/vae_for_neuroimaging.md) for stronger alternatives.
- A clinical tool — the included synthetic data is fictional and ADNI use
  is for learning, not diagnosis.

## Maintainers

- Yu Zheng ([@YuZh98](https://github.com/YuZh98)) — primary author and maintainer.

Open an issue if you'd like to be added as a co-maintainer.

## Citation

If you use this code in your research or teaching, please cite via the
GitHub "Cite this repository" button (driven by [`CITATION.cff`](CITATION.cff)).

## Acknowledgements

Subject scans come from the [ADNI](http://adni.loni.usc.edu/) study; data
use requires acceptance of the ADNI Data Use Agreement.
