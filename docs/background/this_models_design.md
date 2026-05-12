# This Model's Design — Choices and Their Justifications

For the reader walking through `recvae/model.py` line by line asking
"why is it this way and not the other way." Several of the choices below
are flagged as `TODO(research)` in the code, which is the polite way of
saying they could be done better.

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
  `(32, 5, 6, 5)` then four `ConvTranspose3d` stages with chosen
  `output_padding` because the spatial extents are odd
  (`recvae/model.py:121-154`). Final `Tanh` matches the `[-1, 1]` range
  of the normalized input.
- **Latent recurrence** — a linear temporal prior `g(h) = h F^T`
  (`recvae/model.py:195-197`). `F` is registered as a Buffer
  (`recvae/model.py:160-163`) because it is updated by closed-form
  ridge, not gradient descent.
- **Per-subject offsets** — an `nn.Parameter` of shape
  `(N_train, latent_dim)` indexed by the subject's position in the
  training set, added into the latent state before decoding
  (`recvae/model.py:156-158`, `recvae/model.py:215`).

## Why 3-D conv

fMRI volumes have local spatial structure along all three anatomical
axes: cortical folding curves through 3-D space, subcortical structures
sit at fixed positions, tract-level neighborhoods are inherently 3-D. A
`Conv3d` kernel respects that local 3-D neighborhood.

The alternatives are worse. Slice-by-slice `Conv2d` loses anisotropy
across the through-plane axis: two voxels that are immediate neighbors
in different slices look infinitely far apart to the network. Some
papers do this for memory reasons but pay for it in structure. A 1-D
voxel timeseries throws away spatial context entirely; it is fine for
ROI-level models with a known parcellation and useless for whole-brain
representation learning. With `91 × 109 × 91 ≈ 9 × 10^5` voxels, a 1-D
approach either explodes parameter count or shares weights so
aggressively the spatial dimension is moot. 3-D conv sits at the right
level of weight sharing.

## Why recurrent on the latent (not on volumes)

Running an RNN directly over the volume sequence means carrying a
hidden state comparable in size to the input volume, which is
unaffordable in memory and parameters. The latent dynamical-systems
trick — recurrence on the **encoded** state, not the raw observation —
is the same move state-space models have made for decades, and the
move DKF/VRNN/SRNN all share. Here the latent is 10-D, so the temporal
model `g: ℝ^10 → ℝ^10` has 100 parameters, against the ~10^5 it would
have on the encoder's 100-dim feature or the ~10^11 on the raw voxel
grid.

## Why a linear temporal prior with closed-form F

Most recurrent-VAE papers use a nonlinear transition (MLP, GRU, or
attention) and learn it by gradient. This repo uses

```
g(h) = h F^T
```

with `F ∈ ℝ^{10×10}`, and solves `F` analytically per epoch. Three
reasons:

1. **Simplest non-trivial dynamics.** A linear transition is the
   minimum interesting choice — anything simpler is the identity or
   zero. It is the latent-dynamical-systems analogue of using a
   linear-Gaussian state-space model (Kalman filter) instead of a
   particle filter.
2. **Identifiable up to similarity.** `F` and its similarity transform
   `S F S^{-1}` (with `S` invertible) produce equivalent dynamics in
   different latent bases. That is the well-understood identifiability
   structure of linear state-space models; nonlinear transitions don't
   enjoy it.
3. **Tractable in closed form.** Minimizing
   `(1/2σ_h^2) Σ ‖h_t − F h_{t-1}‖^2 + ρ ‖F‖_F^2` is ridge regression
   in `F` with the analytic solution

   ```
   (X^T X + 2 N T σ_h^2 ρ I) F^T = X^T Y
   ```

   where `Y` stacks all posterior states and `X` stacks the same
   shifted by one (with the shared `h_0` prepended). See
   `recvae/model.py:269-331` for the derivation and the implementation.

Solving analytically beats letting SGD drift `F` toward the same
minimum for two reasons. At the end of each epoch `F` is at the true
optimum given the current posterior trajectory, not "wherever SGD got
to in one step." And `F` no longer needs its own step size, decay, and
optimizer state; a surprising fraction of recurrent-VAE failure modes
trace back to mismatches between the transition's learning rate and
the rest of the network's.

The cost: a linear transition cannot capture nonlinear dynamics (limit
cycles, multi-stable regimes, regime switching), and the closed form
only works because the transition is linear. For resting-state fMRI at
6-minute timescales, whether that matters is an open empirical
question. See `tutorials/08_losses_alt_optim/` for the extended
walkthrough.

## Why subject-specific offsets as `nn.Parameter`

Subjects differ from each other in scanner gain, head-size scaling,
baseline motion characteristics, and slow drift — variation that has
nothing to do with the cognitive or clinical signal of interest.
Absorbing those into a per-subject offset added in latent space lets
the rest of the network focus on shared structure.

The implementation is the problem:

```python
self.z_vectors = nn.Parameter(torch.randn(train_size, latent_dim) * sig_z)
```

`z_vectors` is indexed by the subject's position in the training set
(`recvae/model.py:215`). `z_s` for a new subject does not exist in the
model. There is no way to encode a held-out subject's offset without
re-running optimization.

The workaround in `recvae/evaluation.py:108-205` is to re-fit `z_s` for
held-out subjects by a few SGD steps over the reconstruction loss with
the rest of the network frozen. Acceptable if you budget the compute.

The standard alternative is to amortize the offset: encode `z_s` from a
small per-subject summary (e.g. the temporal mean volume) so a forward
pass extracts the offset directly, without an inner optimization loop.
A sketch sits at `examples/amortized_z.py` (yet to land).

## Why L1 sparsity on `z_s`

The training loss includes `loss_z = λ_z ‖z_vectors‖_1`
(`recvae/losses.py:77`, `recvae/config.py:40`). The motivation is to
encourage only a few dimensions of the offset to fire per subject — a
sparse subject code.

There is a small mathematical inconsistency worth flagging: `z_vectors`
is initialized from a Gaussian with scale `sig_z`
(`recvae/model.py:156-158`). A Gaussian prior is the maximum-entropy
distribution for a given variance and corresponds to L2 (ridge)
regularization in MAP terms; an L1 penalty corresponds to a Laplace
prior. The model uses one at initialization and the other during
training. Not a bug, both distributions are fine choices, but the
mismatch matters if you start tuning `lambda_z` or comparing to other
priors.

## Why MSE not KL on the latent transition (`loss2`)

The canonical VAE ELBO has a KL term:

```
KL(q(h_t | x_{1:t}, h_{t-1}) || p(h_t | h_{t-1}))
```

This repo replaces it with the squared distance between the sampled
`h_t` and the prior mean `g(h_{t-1})` (`recvae/losses.py`). That is
point estimation of the latent path (MAP-style), not full variational
inference. The encoder still produces `log_var_h`, but `loss2` doesn't
use it; the variance falls into the reconstruction term only.

A canonical KL implementation is provided in `KLRecVAELoss`
(`recvae/losses.py:90-163`) and discussed in
`tutorials/08_losses_alt_optim/why_mse_not_kl.md`. Switching to it is
one of the smaller experiments worth running on this codebase.

The cost of the MSE form: you lose the variance-calibrating effect of
the KL term. Without it, the posterior variance is free to shrink to
near-zero and there is no longer a clean ELBO to quote.

## Why BatchNorm3d

Five `BatchNorm3d` layers in the encoder and decoder normalize across
batch and spatial axes per channel (`recvae/model.py:90-154`). This is
standard in 3-D convolutional networks: deep networks have internal
covariate shift that BN mitigates, BN has a mild regularization effect,
and `LeakyReLU` after BN is well-trodden.

The caveat: `batch_size = 4` (`recvae/config.py:43`) is small. Batch
statistics from 4 volumes are noisy, and at eval time BatchNorm
switches to running statistics that may not match the training-time
per-batch distribution. **GroupNorm** or **InstanceNorm** would be more
robust at this batch size. A swap is sketched in
`examples/groupnorm_swap.py` (yet to land).

## Why SGD@1e-6

The hyperparameter table sets `learning_rate = 1e-6` with vanilla SGD
(`recvae/config.py:44`, `recvae/train.py:85`). For a network of this
size that is almost imperceptible per-step movement, and 500 epochs is
a lot of compute for a slow optimizer. The setting is inherited from
the canonical notebook and preserved in `Config` because changing it
would change the experimental results, which needs an owner decision
rather than a cleanup.

A modern reset would use AdamW at ~1e-4 with a cosine schedule and
likely converge faster to a comparable or better loss. The code is
structured so the optimizer is a one-line swap; see
`examples/swap_adamw.py` (yet to land).

## What I'd change if I were starting over

The two that matter most:

1. **A real KL term for `loss2`.** Use `KLRecVAELoss` so the training
   objective is a proper ELBO and `log_var_h` actually participates.
   The current MSE proxy is the single largest gap between this
   codebase and a textbook recurrent VAE.
2. **Subject-level CV plumbed through the canonical notebook.** The
   utility exists (`recvae.evaluation.split_subjects`) but the canonical
   notebook still uses the same dataset for `train_loader` and
   `test_loader`. Until that changes, every loss number the notebook
   reports is a training loss, and any quantitative claim about model
   performance is structurally unsupported.

Both are tagged `TODO(research)` in the code.
