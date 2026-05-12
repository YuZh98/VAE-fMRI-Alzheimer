# Lesson 17: Research extensions

## What you'll learn

The six follow-on research items the refactor deliberately left open,
why each was left open, and how each would be implemented. By the end
you can read the `TODO(research)` comments in the `recvae/` source and
explain what they are asking for.

This is the "what to do next" lesson. It does not change the model. It
maps the territory between the current implementation and a paper-grade
RecVAE.

## Where this lives in the repo

- Root `README.md` "Known limitations" — author's framing of these.
- `recvae/losses.py:73-74` — the MSE-only `loss2` that should become a
  proper KL term.
- `recvae/config.py:32-33` — `TODO(research)` on making `sig_x`,
  `sig_h`, `sig_z` learnable.
- `recvae/train.py:52-53` — `TODO(research)` on swapping SGD@1e-6 for
  AdamW + cosine.
- `recvae/data.py:79-85` — note on per-subject min-max normalization
  destroying inter-subject intensity differences.

## The concept

The refactor preserved the canonical notebook's training math exactly.
That is a feature: it lets you compare numbers from old notebook runs
against the package. But the canonical math has known issues, and a
real paper-grade RecVAE would address them. The six items below are
prioritized; the first three change training behavior, the last three
change experimental hygiene.

### 1. Real KL term (replace `loss2`)

**Limitation.** `loss2` in `recvae/losses.py:73-74` is

```
sum_t ||h_t - g(h_{t-1})||^2 / (2 B T sig_h^2)
```

This is MAP point estimation on the latent path. The canonical VAE
ELBO requires `KL(q(h_t | x_t, h_{t-1}) || p(h_t | h_{t-1}))` where
`p` is the temporal prior `N(g(h_{t-1}), sig_h^2 I)`.

**TODO citation.** The MSE form is preserved in `recvae/losses.py:73-74`;
the canonical KL alternative is implemented as `KLRecVAELoss` at
`recvae/losses.py:90-163` and is opt-in via the `loss_fn=` argument to
`training_step`.

**Fix sketch.** The KL between two diagonal Gaussians has a closed
form. With `q = N(mu_h, sigma_h^2 I)` (the inference output) and
`p = N(gh_prev, sig_h^2 I)`:

```
KL(q || p) = sum_d [ log(sig_h / sigma_h_d) + (sigma_h_d^2 + (mu_h_d - gh_prev_d)^2) / (2 sig_h^2) - 1/2 ]
```

`kl_term_sketch.py` in this lesson implements that closed form and
verifies it is non-negative for random inputs. The slot in
`RecVAEModel.training_step` where it would replace `loss2` is marked
with a `TODO` in that file.

### 2. Learnable noise scales

**Limitation.** `sig_x`, `sig_h`, `sig_z` are fixed floats in `Config`.
They weight the loss terms but the model cannot tune them.

**TODO citation.** `recvae/config.py:32-33`.

**Fix sketch.** Move the three sigmas off `Config` and onto `RecVAEModel`
as `nn.Parameter`s. Train an unconstrained log-sigma and apply
`softplus` (or `exp`) to keep them positive:

```python
self.log_sig_x = nn.Parameter(torch.zeros(1))
self.log_sig_h = nn.Parameter(torch.zeros(1))
self.log_sig_z = nn.Parameter(torch.zeros(1))

sig_x = torch.nn.functional.softplus(self.log_sig_x) + 1e-3
```

Update `training_step` to use `sig_x`/`sig_h`/`sig_z` from the module
instead of `cfg`. `updating_F` also reads `cfg.sig_h`; switch it to read
the module's current value. Add a per-epoch log of the learned sigmas
so you can see them calibrate.

### 3. AdamW + cosine annealing

**Limitation.** `fit` uses `torch.optim.SGD` at `lr=1e-6` for 500 epochs.
That is slow even on a small dataset, and the canonical noise scales
amplify the issue.

**TODO citation.** `recvae/train.py:52-53`.

**Fix sketch.** Two-line change to `fit`:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
...
for epoch in range(epochs):
    ...
    scheduler.step()
```

Typical learning rate: `1e-4`. The cosine schedule decays to zero over
`epochs`. Consider warmup for the first 5% of epochs. Validate that
loss curves converge faster than SGD on a held-out split (see #4).

### 4. Subject-level k-fold cross-validation

**Limitation.** The canonical notebook's `test_loader` wraps the same
`Dataset` as `train_loader`. Reported reconstruction losses are
training losses, not held-out test losses. The notebook also has
nothing equivalent to subject-aware splits — naive index splits could
leak the same subject across train and test if there were repeated
scans.

**TODO citation.** Root `README.md` "Known limitations".

**Fix sketch.**

1. Group subjects by `subject_id` (e.g. parse from filename).
2. Split *subject_ids*, not file indices, into K folds.
3. For each fold, build the train and test loaders, train fresh, log
   train and test reconstruction loss per epoch.
4. `z_vectors` is per-subject — for held-out subjects you have two
   options:
   - **Zero out z for test subjects.** Cleanest; the model evaluates
     "what would this look like without subject offsets?".
   - **Infer a test `z` post hoc.** Run an inner SGD over a fresh
     `nn.Parameter` `z_test` while holding the rest of the model
     frozen, then evaluate.

   Document the choice; it matters.

### 5. Normalization scheme

**Limitation.** `normalize_per_subject` in `recvae/data.py:74-105`
rescales each subject to `[-1, 1]` based on that subject's own
min/max. This destroys absolute-intensity differences across subjects
— differences that may carry information about the cohort the subject
came from (CN vs AD).

**Fix sketch.** Two reasonable alternatives:

- **Z-score per subject.** Subtract per-subject mean, divide by
  per-subject std. Preserves shape of the intensity distribution but
  still loses absolute scale.
- **Global z-score.** Compute `mean` and `std` once across the whole
  training set; apply to every subject. Preserves inter-subject scale.
  Requires that you NOT recompute stats from test data (use the train
  stats with `map_location` semantics).

The right choice depends on the downstream task. For unconditional
reconstruction the current scheme is defensible; for cohort
classification you almost certainly want global stats.

### 6. KL annealing / beta-VAE

**Limitation.** Beta-VAE-style weighting on the KL term is moot when
there is no proper KL term (see #1). It belongs on the roadmap as a
follow-up to #1.

**Fix sketch.** After #1 is done, multiply the KL term by a `beta`
scalar that you ramp from 0 to 1 over the first ~10% of training:

```python
beta = min(1.0, epoch / (0.1 * epochs))
loss = recon + beta * kl + loss_z
```

Beta-VAE-style annealing prevents the KL from collapsing the encoder
to the prior early in training, before the decoder has learned to use
the latent code.

## Code walk

`kl_term_sketch.py` in this directory:

1. Implements `temporal_kl_term(mu_h, log_var_h, gh_prev, sig_h)`
   returning a tensor. The closed form of `KL(N(mu_h, sigma_h^2 I) ||
   N(gh_prev, sig_h^2 I))`.
2. Verifies the result is non-negative for several random inputs.
3. Verifies the limiting case: when `mu_h == gh_prev` and
   `sigma_h == sig_h`, the KL is zero.
4. Includes a `# TODO: integrate this into RecVAEModel.training_step`
   stub that shows where in the existing loss it would slot.

It does *not* modify `recvae/model.py`. It is a sketch — the reader is
expected to finish the integration by replacing `loss2` in
`training_step`.

## Run it

```bash
python tutorials/17_extensions/kl_term_sketch.py
```

Output: prints the KL value for several random inputs, the limiting
case, and the integration stub.

## Why this approach

The alternative would be to commit a version of `recvae/` that
implements all six items. We deliberately did not:

- **Each item is a research decision.** Switching to AdamW changes the
  numbers in every paper plot. The maintainer should make that call,
  not the refactor.
- **Each item interacts with the others.** A real KL term changes the
  loss scale, which changes the optimal `lr`. Learnable noise scales
  change which loss term dominates. Cleanest path is one PR per item.
- **The current code is reproducible.** It matches what the canonical
  notebook does, modulo correctness fixes. Anyone trying to reproduce
  an old result wants that exact behavior, not "what we now think is
  better".

So Lesson 17 marks the territory but does not enter it. Follow the
TODOs.

## Further reading

- Kingma and Welling, *Auto-Encoding Variational Bayes* (2014) — the
  canonical VAE ELBO derivation that #1 fixes.
- Higgins et al., *beta-VAE* (2017) — the disentanglement motivation
  for #6.
- Krueger et al., *Bayesian Hypernetworks* (2017) — learnable noise
  scales in the variational setting (#2).
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization* (2019,
  AdamW) and *SGDR* (2017, cosine annealing) — for #3.
- `recvae/model.py:334-386` and `recvae/train.py:19-159` — the code
  these extensions modify.
