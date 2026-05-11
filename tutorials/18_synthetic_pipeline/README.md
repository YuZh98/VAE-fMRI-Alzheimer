# Lesson 18: Putting it all together — synthetic pipeline

## What you'll learn

- How `synthetic_cohort` (a deterministic generator producing CN- and
  AD-like volumes) lets you exercise the full RecVAE pipeline without
  any external data.
- How to compose `synthetic_cohort` -> `normalize_per_subject` ->
  `FMRIDataset` -> `build_dataloader` -> `RecVAEModel` -> `fit` ->
  `evaluate_held_out` into one ~100-line script.
- The difference between training-time evaluation (the loss history
  returned by `fit`) and *held-out* evaluation, where a fresh `z_s`
  is fit for unseen subjects while every other parameter is frozen.
- Why a held-out cohort drawn from the same generator with a
  *different seed* is the cleanest way to demonstrate generalization
  on synthetic data.

## Where this lives in the repo

- `recvae/data.py:synthetic_cohort` — the generator. Returns
  `volumes` of shape `(N, 1, 91, 109, 91, T)` and `labels` of shape
  `(N,)` with 0=CN, 1=AD.
- `recvae/evaluation.py:evaluate_held_out` — the held-out scoring
  function. Freezes encoder/decoder/inference head/`F_mat`/training
  `z_vectors`, then SGD-optimizes a fresh `z_test` for the test
  subjects against the reconstruction loss.
- `recvae/train.py:fit` — the same training loop Lesson 15 used.
- `tutorials/15_train_end_to_end/train_tiny.py` — the immediate
  predecessor; the present script adds (a) labeled synthetic cohorts
  and (b) the held-out eval step.
- `notebooks/RecVAE_on_synthetic.ipynb` — the notebook companion to
  this lesson. Same pipeline, with a CN-vs-AD linear probe added at
  the end.

## The concept

Up to Lesson 15 every demo trained on a single tensor of random noise:
useful for exercising shapes and the optimizer, useless for asking the
model anything about *generalization*. The `synthetic_cohort` generator
fixes that. It produces:

- A low-rank spatial pattern per subject (random mixture of `K=4` shared
  basis fields).
- An AR(1) temporal modulation of that pattern.
- For AD subjects, a hemispheric damping mask multiplied into the
  pattern (the `cohort_effect` parameter) — a crude but recognizable
  stand-in for atrophy.
- Per-voxel additive Gaussian noise.

So CN and AD volumes differ in a structured, *spatially localized* way.
That is enough signal for a linear probe on the model's latents to
separate the two cohorts at training-set size N=8, and enough to make
held-out reconstruction MSE behave sensibly: a fresh draw from the
generator should reconstruct cleanly because the model has learned
the basis.

The pipeline this lesson scripts is:

```
synthetic_cohort(seed=2022)        # train cohort
  -> normalize_per_subject         # min-max to [-1, 1] per subject
  -> FMRIDataset + build_dataloader
  -> RecVAEModel(train_size=N)
  -> fit(model, dl, h0, epochs=2)  # SGD + closed-form F alternation
synthetic_cohort(seed=4242)        # held-out cohort, different seed
  -> normalize_per_subject
  -> evaluate_held_out(...)        # fits z_test only, model frozen
```

### Why a different seed for the held-out cohort

The generator is deterministic in its `seed` argument. Calling it twice
with `seed=2022` gives you literally the same tensor — that would not
test generalization. Two *different* seeds give you statistically
independent draws from the same generative process, which is exactly
what you want to measure out-of-sample reconstruction on. The held-out
MSE should be similar in magnitude to the final training reconstruction
loss; if it is wildly higher you have a training-set memorization
problem.

### Why `evaluate_held_out` re-fits `z_s`

`z_vectors` is per-subject by design: each row is the subject-specific
offset added to the latent state before decoding. Training subjects
have their `z_s` learned during `fit`; held-out subjects do not, so the
inference-time procedure is to re-fit *just* `z_test` against the
reconstruction loss while everything else stays frozen. See
`recvae/evaluation.py:evaluate_held_out` for the implementation — it
snapshots and restores every `requires_grad` flag on the model so the
call has no side effects.

## Code walk

`pipeline.py` in this directory:

1. `set_seed(2022)`, `get_default_device()` (demoted to CPU on MPS, same
   as Lesson 15).
2. `Config(tol_time=4, epochs=2, batch_size=2, learning_rate=1e-5)`.
3. `synthetic_cohort(n_cn=2, n_ad=2, T=4, seed=2022)` — 4 training
   subjects, half CN half AD.
4. `normalize_per_subject(volumes)`.
5. `FMRIDataset` + `build_dataloader(seed=cfg.seed)` +
   `DeviceDataLoader(..., device)`.
6. `RecVAEModel(train_size=4, cfg=cfg)`, `h0 = zeros(1, latent_dim)`.
7. `fit(model, dl, h0, cfg=cfg, epochs=2)` — alternating optimization.
   Print the loss history and the per-epoch delta.
8. `synthetic_cohort(n_cn=1, n_ad=1, T=4, seed=4242)` — fresh draw.
9. `evaluate_held_out(model, ..., inner_steps=5, inner_lr=1e-3)`.
   Print the final reconstruction MSE and shapes of `z_test`/`h_test`.

Total wall time: under 10 seconds on CPU.

## Run it

```bash
python tutorials/18_synthetic_pipeline/pipeline.py
```

Expected output: four banner sections (synthesize, build, train,
held-out eval), each printing a small dict of stats. The training loss
history should show a decrease (e.g. ~32k -> ~10k on this seed); the
held-out MSE should land in the same order of magnitude as the final
training reconstruction loss (the `loss1` component of `fit`'s loss).

## Why this approach

Lesson 15 trained on `torch.randn` and could not say anything about
generalization because successive `randn` draws have no shared
structure. Without a shared generative process there is no "same
distribution" for a held-out set to come from.

The `synthetic_cohort` generator solves that. It is deliberately not
neuroscientifically accurate — it is a *toy* signal with enough
structure (low-rank spatial basis, AR(1) temporal modulation,
cohort-localized damping) to:

1. Make training reduce a real reconstruction loss, not a noise floor.
2. Make held-out scoring meaningful — a fresh draw from the same
   generator has the same statistical structure as training.
3. Make a CN-vs-AD distinction *exist* — the AD damping mask is
   detectable in the latents (the notebook companion demonstrates this
   with a linear probe).

This lets the whole pipeline be exercised in CI, on CPU, without anyone
needing access to a clinical dataset.

## Exercise (optional)

1. Change `seed=4242` to `seed=2022` (same as training). The held-out
   MSE should drop substantially because you are scoring on training
   examples. Confirm this and explain why.
2. Set `cohort_effect=0.0` in both `synthetic_cohort` calls. Now CN and
   AD are statistically identical. The held-out MSE should be roughly
   unchanged (reconstruction does not depend on the cohort label), but
   any downstream cohort classifier (see the notebook) should drop to
   chance. Verify.
3. Bump `epochs=2` to `epochs=10`. Plot the training loss and the
   held-out MSE per epoch (call `evaluate_held_out` inside a `callbacks`
   list — `fit` accepts one). Watch for overfitting: training loss
   keeps falling while held-out MSE plateaus or rises.

## Further reading

- Lesson 09 — `Dataset` / `DataLoader` mechanics and per-subject
  normalization.
- Lesson 15 — the predecessor end-to-end script; same loop, no
  cohort labels, no held-out eval.
- Lesson 17 #4 — research notes on subject-aware splits and on the
  two design choices for handling `z_s` on held-out subjects (zero
  it out vs. re-fit it).
- `notebooks/RecVAE_on_synthetic.ipynb` — same pipeline as a notebook,
  with an added linear probe on the latents and a PCA baseline.
