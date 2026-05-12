# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-in-Colab badge and synthetic notebook bootstrap so the notebooks
  run on a fresh Colab kernel without a local checkout.
- `docs/assets/` with three reproducible visuals (loss curve,
  reconstruction slice, latent trajectory).
- `tools/check_citations.py` plus a CI guard that fails when
  `docs/background/` references drift away from the cited line numbers.
- `tools/make_figures.py` to regenerate the `docs/assets/` visuals from
  a single pinned-seed run.
- Per-epoch progress logging in `fit()`.
- `lambda_z` parameter on `evaluate_held_out` so held-out evaluation
  uses the same L1 penalty as training.
- `--plot` flag on `train_tiny.py` and `pipeline.py` for quick
  smoke-plotting of loss + a reconstruction slice.
- `[examples]` optional-dependency group pulling in `scikit-learn` for
  the `examples/classify_from_latents.py` driver.
- `.github/CODEOWNERS`.

### Changed
- CI split into a fast `tests.yml` (pytest matrix on every push/PR)
  and a nightly `tutorials.yml` (full tutorial demo loop +
  `examples/`). Python 3.9 dropped from the matrix (EOL).
- Upper-bound version pins added in `pyproject.toml`
  (`torch<3`, `numpy<3`, `nibabel<6`, `matplotlib<4`).

### Fixed
- Tutorial 08 ridge formula now matches the implementation (the `N*T`
  scaling factor was missing from the LaTeX).
- Stale `file:line` citations under `docs/background/` refreshed to
  point at the current source.
- "MPS detected; demoting" log message reworded to explain *why* the
  demotion happens rather than just announcing it.

## [0.1.0] - 2026-05-11

Initial public release. The recurrent 3D-conv VAE was extracted from a
canonical Jupyter notebook into a tested, device-agnostic Python package,
paired with an 18-lesson tutorial series and a synthetic on-ramp.

### Added
- `recvae/` package extracted from the canonical notebook
  (`config`, `data`, `model`, `train`, `losses`, `evaluation`, `utils`).
- Device-agnostic execution: picks CUDA, MPS, or CPU at runtime.
- Pinned RNG state via `recvae.utils.set_seed` for reproducible runs.
- Timepoint validation: every fMRI sequence is verified against
  `Config.tol_time` before training.
- Zero-span normalization guard so constant volumes do not divide by zero
  during per-subject min-max normalization.
- 34-test pytest suite (grown from the original 23 tests during refactor).
- Synthetic cohort generator so the model can be exercised without ADNI
  access.
- Held-out evaluation utilities in `recvae.evaluation`.
- `RolloutOutput` NamedTuple returned by `RecVAEModel.forward` so
  downstream code can pull named fields instead of positional tuples.
- 18-lesson tutorial series under `tutorials/`, each lesson with a short
  README and at least one runnable script. CI runs every script on every
  push.
- Background documentation under `docs/background/` (fMRI 101, design
  rationale, related work).
- Variants menu under `examples/` showing short end-to-end driver
  scripts.

### Changed
- `z_vectors` promoted to `nn.Parameter` (was a plain tensor) so the
  optimizer manages it as a first-class parameter.
- `F_mat` registered as a `buffer` (was a plain tensor) so it travels
  with `.to(device)` and `state_dict()` but is not updated by SGD.
- F-ridge update: corrected the scaling on the closed-form ridge solve
  so the penalty `rho` has the documented effect.
- Loss assembly decoupled from the model into `recvae.losses`, making
  each term (`loss1`, `loss2`, `loss_z`, `loss_F`) independently
  testable.

### Fixed
- F-ridge scaling correction (see Changed above) — the previous code
  scaled the ridge penalty in a way that made `rho` effectively
  meaningless.

[Unreleased]: https://github.com/YuZh98/VAE-fMRI-Alzheimer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/YuZh98/VAE-fMRI-Alzheimer/releases/tag/v0.1.0
