# This Model's Design — Choices and Their Justifications

For the reader walking through `recvae/model.py` line by line, asking
"why is it this way and not the other way." Every section flags whether
the choice is **standard** (boring, do-as-everyone-does) or **unusual**
(a deliberate departure that you should understand before reusing).
Several are honest weak spots flagged as `TODO(research)` in the code.

## Architecture summary

The model is a recurrent VAE over 4-D fMRI volumes
`(B, 1, 91, 109, 91, T=120)`.

- **Encoder** — five stages. Four `Conv3d` blocks with `kernel=4`,
  `stride=2`, `padding=1`, `BatchNorm3d`, and `LeakyReLU(0.2)`
  (`recvae/model.py:90-109`), followed by a `Flatten + Linear + Tanh`
  to `enc_out_dim = 100` (`recvae/model.py:110-114`). Spatial chain
  `91 → 45 → 22 → 11 → 5`.
- **Inference head** — concatenates the 100-dim encoder feature with
  the previous latent state `h_{t-1}` (10-dim) and emits Gaussian
  posterior parameters `(μ_h, log σ_h^2)` via two `Linear` layers
  (`recvae/model.py:116-119`). The reparameterized sample is the new
  latent `h_t`.
- **Decoder** — symmetric to the encoder. `Linear + Unflatten` back to
  `(32, 5, 6, 5)` then four `ConvTranspose3d` stages with
  carefully-chosen `output_padding` because the spatial extents are
  odd (`recvae/model.py:121-146`). Final `Tanh` matches the `[-1, 1]`
  range of the normalized input.
- **Latent recurrence** — a linear temporal prior
  `g(h) = h F^T` (`recvae/model.py:187-189`). `F` is registered as a
  Buffer (`recvae/model.py:152-155`) because it is updated by
  closed-form ridge, not gradient descent.
- **Per-subject offsets** — an `nn.Parameter` of shape
  `(N_train, latent_dim)` indexed by the subject's position in the
  training set, added into the latent state before decoding
  (`recvae/model.py:148-150`, `recvae/model.py:207`).

## Why 3-D conv (not 2-D slice, not 1-D timeseries)

**STANDARD.** fMRI volumes have local spatial structure along all three
anatomical axes: cortical folding curves through 3-D space, subcortical
structures sit at fixed 3-D positions, and tract-level neighborhoods are
inherently 3-D. A `Conv3d` kernel respects that local 3-D neighborhood.
The alternatives:

- **2-D slice-by-slice with `Conv2d`.** Loses the anisotropy across the
  through-plane axis — two voxels that are immediate neighbors but in
  different slices look infinitely far apart to the network. Some
  papers do this for memory reasons, but it sacrifices structure.
- **1-D voxel timeseries.** Treats each voxel as an independent signal
  and throws away spatial context entirely. Useful for ROI-level
  models with a known parcellation, useless for whole-brain
  representation learning.

Concretely: with `91 × 109 × 91 ≈ 9 × 10^5` voxels, a 1-D approach
either explodes parameter count (one network per voxel) or shares
weights so aggressively that the spatial dimension is moot. 3-D conv is
the right level of weight sharing.

## Why recurrent on the latent (not on volumes)

**STANDARD.** Running an RNN directly over the volume sequence would
mean carrying a hidden state of comparable size to the input volume,
which is unaffordable in both memory and parameters. The latent
dynamical-systems trick — run the recurrence on the **encoded** state,
not the raw observation — is the same move state-space models have made
for decades, and what DKF/VRNN/SRNN all do. Here the latent is 10-D, so
the temporal model `g: ℝ^10 → ℝ^10` has 100 parameters, vs the
~10^5 it would have if we tried to do recurrence on the encoder's
100-dim feature directly or the ~10^11 on the raw voxel grid.

## Why a LINEAR temporal prior

**UNUSUAL CHOICE.** Most recurrent-VAE papers use a nonlinear
transition (MLP, GRU, or attention) and learn it by gradient. This repo
uses

```
g(h) = h F^T
```

with `F ∈ ℝ^{10×10}`. The defense:

1. **Simplest non-trivial dynamics.** A linear transition is the
   minimum interesting choice — anything simpler is the identity or
   zero. It is the latent-dynamical-systems analogue of using a
   linear-Gaussian state-space model (Kalman filter) instead of a
   particle filter.
2. **Identifiable up to similarity.** `F` and its similarity transform
   `S F S^{-1}` (with `S` invertible) produce equivalent dynamics in
   different latent bases. That is the well-known and well-understood
   identifiability structure of linear state-space models; nonlinear
   transitions don't enjoy it.
3. **Tractable in closed form.** Minimizing
   `(1/2σ_h^2) Σ ‖h_t − F h_{t-1}‖^2 + ρ ‖F‖_F^2` is **ridge
   regression** in `F`, which has an analytic minimizer (see next
   section).

The cost: a linear transition cannot capture nonlinear dynamics — limit
cycles, multi-stable regimes, regime-switching, anything where the
update rule depends on where you are in the latent space. For
resting-state fMRI at 6-minute timescales, whether that matters is an
open empirical question.

## Why CLOSED-FORM ridge update on F (not SGD)

**UNUSUAL CHOICE.** This is the pedagogical centerpiece of the model.
The ridge objective

```
min_F (1/(2 σ_h^2 N T)) Σ ‖h_t − F h_{t-1}‖^2 + ρ ‖F‖_F^2
```

has a closed-form minimizer via the normal equation

```
(X^T X + 2 N T σ_h^2 ρ I) F^T = X^T Y
```

where `Y` stacks all posterior states and `X` stacks the same shifted
by one (with the shared `h_0` prepended). See `recvae/model.py:261-319`
for the derivation in the docstring and the implementation.

Why solve it analytically each epoch instead of letting SGD drift `F`
toward this same minimum?

- **At the end of each epoch, `F` is at the true optimum** given the
  current posterior trajectory, not "wherever SGD got to in one step."
- **One fewer learning rate to tune.** `F` would otherwise need its
  own step size, decay, and possibly its own optimizer state — a
  surprising fraction of recurrent-VAE failure modes are mismatches
  between the transition's learning rate and the rest of the network's.
- **Teaching value.** It makes the alternation between SGD on `θ` and
  closed-form on `F` concrete: a coordinate-descent style algorithm
  with one analytic block.

The cost: this only works because the transition is linear. A
nonlinear transition would not admit a closed form and would need
gradient updates. See `tutorials/08_losses_alt_optim/` for an extended
walkthrough.

## Why subject-specific offsets `z_s` as `nn.Parameter`

**UNUSUAL CHOICE WITH A REAL DOWNSIDE.** The idea is sound: subjects
differ from each other in scanner-gain, head-size scaling, baseline
motion characteristics, and slow drift — variation that has nothing to
do with the cognitive or clinical signal of interest. Absorbing those
into a per-subject offset that the decoder adds in latent space lets
the rest of the network focus on the shared structure.

The implementation choice is the problem:

```python
self.z_vectors = nn.Parameter(torch.randn(train_size, latent_dim) * sig_z)
```

`z_vectors` is **indexed by the subject's position in the training
set** (`recvae/model.py:207`). That means `z_s` for a new subject does
not exist in the model — there is no way to encode a held-out subject's
offset without re-running optimization.

The repo works around this in `recvae/evaluation.py:109-...` by
**re-fitting `z_s` for held-out subjects** by a few SGD steps over the
reconstruction loss with the rest of the network frozen, which is
acceptable as long as you budget the compute for it.

Standard alternative practice: **amortize the offset.** Encode `z_s`
from a small per-subject summary (e.g., the temporal mean volume) so a
forward pass extracts the offset directly, without any inner
optimization. A sketch of this is forward-referenced in
`examples/amortized_z.py` (yet to land).

## Why L1 sparsity on `z_s`

**UNUSUAL CHOICE WITH AN INTERNAL INCONSISTENCY.** The training loss
includes `loss_z = λ_z ‖z_vectors‖_1` (`recvae/losses.py:78`,
`recvae/config.py:41`). The motivation is to encourage only a few
dimensions of the offset to fire per subject — a sparse subject code.

The inconsistency: `z_vectors` is **initialized from a Gaussian** with
scale `sig_z` (`recvae/model.py:148-150`). A Gaussian prior is the
maximum-entropy distribution for a given variance and corresponds to
**L2** (ridge) regularization in MAP terms. An **L1** penalty
corresponds to a **Laplace** prior. The model uses one at
initialization and the other during training. They are not
mathematically the same regularizer. This isn't a bug — both
distributions are fine choices — but documenting the mismatch matters
if you start tuning `lambda_z` or comparing to other priors.

## Why MSE not KL on the latent transition (`loss2`)

**UNUSUAL CHOICE BY OMISSION.** The canonical VAE ELBO has a KL term:

```
KL(q(h_t | x_{1:t}, h_{t-1}) || p(h_t | h_{t-1}))
```

This repo replaces that term with the squared distance between the
sampled `h_t` and the prior mean `g(h_{t-1})`
(`recvae/losses.py`). That is point estimation of the latent path
(MAP-style), not full variational inference. The encoder still produces
a `log_var_h`, but `loss2` doesn't use it — it falls into the
reconstruction term only.

A canonical KL implementation is provided in `KLRecVAELoss`
(`recvae/losses.py:92-...`) and discussed in
`tutorials/08_losses_alt_optim/why_mse_not_kl.md`. Switching to it is
one of the cleanest small experiments to run on this codebase.

The cost of the MSE form: you lose the variance-calibrating effect of
the KL term. Without it, the posterior variance is free to shrink to
near-zero ("posterior collapse" is the wrong word here, but the failure
mode is in that direction) and there is no longer a clean ELBO to quote.

## Why BatchNorm3d

**STANDARD with a small-batch caveat.** Five `BatchNorm3d` layers in
the encoder and decoder normalize across the batch and spatial axes per
channel (`recvae/model.py:91-146`). Standard practice in 3-D
convolutional networks because (a) deep networks have internal
covariate shift that batch normalization mitigates, (b) it has a mild
regularization effect, and (c) `LeakyReLU` after BN is well-trodden.

The caveat: `batch_size = 4` (`recvae/config.py:44`) is small. Batch
statistics computed from 4 volumes are noisy, and at eval time
BatchNorm switches to running statistics that may not match the
training-time per-batch distribution. **GroupNorm** or **InstanceNorm**
would be more robust at this batch size. A swap is sketched in
`examples/groupnorm_swap.py` (yet to land).

## Why SGD@1e-6

**SLOW CHOICE inherited from the canonical notebook.** The hyperparameter
table shows `learning_rate = 1e-6` with vanilla SGD
(`recvae/config.py:45`, `recvae/train.py:74`). For a network of this
size, a learning rate that small means almost imperceptible per-step
movement; 500 epochs is a lot of compute for a slow optimizer.

A modern reset would use **AdamW** at `~1e-4` with a cosine schedule
and likely converge much faster to a comparable or better loss. The
code is structured so the optimizer is a one-line swap; see
`examples/swap_adamw.py` (yet to land). The slow setting is preserved
in `Config` because changing it would change the experimental results,
which need an owner decision, not a cleanup.

## What I'd change if I were starting over

All of these are already flagged as `TODO(research)` somewhere in the
code:

1. **Real KL term for `loss2`.** Use `KLRecVAELoss` so the training
   objective is a proper ELBO and `log_var_h` actually participates.
2. **Learnable `σ`'s.** Make `sig_x`, `sig_h`, `sig_z` trainable
   `nn.Parameter`s (with `softplus` to stay positive) so the model
   calibrates its own loss term weighting instead of having them as
   fixed hyperparameters.
3. **Amortized `z_s`.** Encode the subject offset from data instead of
   indexing into a Parameter table, so the model generalizes to held-out
   subjects without an inner optimization loop.
4. **GroupNorm or InstanceNorm.** Drop the batch-size dependence of
   BatchNorm; the network would be more reliable across batch sizes
   and at eval time.
5. **AdamW + cosine schedule.** Replace SGD@1e-6 with a modern
   optimizer and learning-rate schedule.
6. **Proper subject-level CV everywhere.** The codebase has the
   utilities (`recvae.evaluation.split_subjects`), but the canonical
   notebook does not yet use them — using the same dataset for train
   and test inflates every loss number it reports.

None of these are interesting research projects on their own; they are
the bar to clear before any quantitative claim about model performance.
That bar is deliberately left unmet in the canonical notebook so the
teaching surface stays focused on engineering, not on results.
