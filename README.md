# fMRI_Project

Recurrent VAE for temporal fMRI representation learning, applied to
Alzheimer's-related cohorts (Cognitively Normal vs Alzheimer's Disease) from
the ADNI study.

## What it does

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

### Loss

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
fMRI_Project/
├── recvae/                  # main Python package (extracted from notebooks)
│   ├── config.py            # Config dataclass — all hyperparameters
│   ├── data.py              # NIfTI loading, normalization, Dataset, DataLoader
│   ├── model.py             # RecVAEModel (encoder/decoder/inference/F update)
│   ├── train.py             # fit() and evaluate()
│   └── utils.py             # set_seed, device picker, DeviceDataLoader
├── tests/                   # pytest suite (23 tests, ~5s, CPU-only)
├── notebooks/
│   ├── RecVAE_on_fMRI.ipynb # canonical driver (uses recvae package)
│   └── Input_images.ipynb   # data-inspection helpers
├── legacy/                  # archived earlier iterations (V1–V4)
├── pyproject.toml           # PEP 621 metadata + pytest config
├── requirements.txt         # runtime deps
└── requirements-dev.txt     # + pytest, nbstripout
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

(Or `pip install -e .[dev]` once you've initialized the package layout.)

## Usage

### Run the tests

```bash
pytest -v
```

All 23 tests pass on CPU in a few seconds — they use synthetic tensors and
do not require nibabel or real fMRI data.

### Train on real data

```python
import torch
from recvae import (
    Config, RecVAEModel, FMRIDataset, build_dataloader,
    DeviceDataLoader, fit, get_default_device, set_seed,
    list_subject_files, load_subject_volumes, normalize_per_subject,
)

set_seed(2022)
device = get_default_device()

# Point these at your local copies; do NOT commit the paths.
DIR_CN = "/path/to/CN"
DIR_AD = "/path/to/AD"

cn_files = list_subject_files(DIR_CN)
ad_files = list_subject_files(DIR_AD)

cn = load_subject_volumes(DIR_CN, cn_files, tol_time=120)
ad = load_subject_volumes(DIR_AD, ad_files, tol_time=120)
volumes = torch.cat([cn, ad], dim=0)

volumes, _, _ = normalize_per_subject(volumes)

cfg = Config()  # defaults match the canonical notebook
ds = FMRIDataset(volumes)
dl = build_dataloader(ds, cfg.batch_size, shuffle=True, seed=cfg.seed)
dl = DeviceDataLoader(dl, device)

model = RecVAEModel(train_size=len(ds), cfg=cfg).to(device)
h0 = torch.zeros(1, cfg.latent_dim, device=device)

history = fit(model, dl, h0, cfg=cfg, epochs=500)
```

### Save / load

```python
torch.save(model.state_dict(), "recvae_state.pt")

model2 = RecVAEModel(train_size=len(ds), cfg=cfg).to(device)
model2.load_state_dict(torch.load("recvae_state.pt", map_location=device))
```

`F_mat` (Buffer) and `z_vectors` (Parameter) are both included in
`state_dict()`, so the snapshot fully captures model state.

## Hyperparameters

All in [`recvae/config.py`](recvae/config.py). Defaults match the canonical
notebook (RecVAE_on_fMRI.ipynb / Version4.ipynb):

| Field           | Default | Meaning                                         |
|-----------------|---------|-------------------------------------------------|
| `enc_out_dim`   | 100     | Encoder output dim before inference head        |
| `latent_dim`    | 10      | Posterior latent state dim                      |
| `z_dim`         | 10      | Subject-specific noise dim                      |
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

## What changed during the refactor

Earlier this branch:

- Extracted the model and training loop into the `recvae/` package (no
  more 80% code duplication across V1/V2/V3/V4).
- Made the model device-agnostic (CUDA / MPS / CPU) — removed
  `torch.set_default_tensor_type('torch.cuda.FloatTensor')` and
  `DataLoader(generator=torch.Generator(device='cuda'))`.
- Added explicit `device=` and `dtype=` to the reparameterization sample
  (`torch.randn` was previously CPU-allocated then implicitly moved).
- Pinned all RNG state in `set_seed` (Python, NumPy, torch CPU, all CUDA
  devices, cuDNN deterministic).
- Promoted `z_vectors` to an `nn.Parameter` and `F_mat` to a Buffer —
  both serialize correctly in `state_dict()` now.
- Validated NIfTI timepoint count instead of silently producing
  shape-mismatched stacks.
- Guarded `normalize_per_subject` against constant-volume division by zero.
- Added a 23-test pytest suite covering shapes, gradient flow, F update,
  normalization edge cases, and dataloader determinism.
- Archived V1–V4 notebooks to `legacy/` and stripped output blobs to keep
  the repo small.

See `git log refactor/cleanup-and-modularize` for per-commit detail.

## Citation / data

Subject scans come from the [ADNI](http://adni.loni.usc.edu/) study. Do not
commit subject IDs or filesystem paths into source control.
