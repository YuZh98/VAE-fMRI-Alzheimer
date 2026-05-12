# Lesson 08: Losses and alternating optimization

## What you'll learn

The four loss terms that drive RecVAE training, why one of them (the
transition matrix `F`) is not trained by gradient descent, and how the
training loop alternates per-epoch SGD with a closed-form ridge update.
By the end you can read `training_step` and `updating_F` in
`recvae/model.py` and explain the math behind each line.

## Where this lives in the repo

- `recvae/model.py:334-386` — `training_step`: composes the four loss terms.
- `recvae/model.py:269-331` — `updating_F`: closed-form ridge solve.
- `recvae/train.py:105-130` — alternation: SGD inside the batch loop, then one `updating_F` call at the end of each epoch.
- `recvae/config.py` — `sig_x`, `sig_h`, `lambda_z`, `rho` defaults.

## The four terms

RecVAE's objective has four pieces. Three are added into a scalar loss
that gets `.backward()`-ed; the fourth is reported only and solved in
closed form. Per batch of `B` subjects rolled out over `T` timesteps:

1. `loss1` — per-volume reconstruction MSE divided by `sig_x^2`.
   ```
   loss1 = sum_t ||x_t - mu_t||^2 / (2 * B * T * sig_x^2)
   ```
   The `2 * B * T` denominator averages over batch and timesteps; the
   `sig_x^2` weight reflects the per-pixel observation noise of a
   Gaussian likelihood. See `recvae/losses.py:70-71`.

2. `loss2` — temporal-prior MSE divided by `sig_h^2`.
   ```
   loss2 = sum_t ||h_t - g(h_{t-1})||^2 / (2 * B * T * sig_h^2)
   ```
   where `g(h) = h @ F^T` is the linear transition. This term pulls the
   posterior path `h_1, h_2, ...` toward a sequence predicted by `F`.
   See `recvae/losses.py:73-74`.

   This is *not* the KL term a canonical VAE ELBO would have. It is an
   MSE point-estimate proxy that does MAP-style inference on the latent
   path. The opt-in KL alternative lives in `KLRecVAELoss` at
   `recvae/losses.py:90-163`. See `why_mse_not_kl.md` in this lesson
   for what a real KL term would look like.

3. `loss_z` — L1 sparsity on the subject-specific noise vectors.
   ```
   loss_z = lambda_z * ||z||_1
   ```
   `z_vectors` is a `(N_train, latent_dim)` Parameter (see
   [Lesson 06](../06_param_vs_buffer/README.md)). L1 keeps most subject
   offsets near zero so the shared decoder carries the work. See
   `recvae/losses.py:77`.

4. `loss_F` — Frobenius regularizer on `F`.
   ```
   loss_F = rho * ||F||_F^2
   ```
   Reported but not back-propped. `loss_F` is computed and returned in
   the loss dict for logging, but the total loss that gets
   `.backward()`-ed is `loss1 + loss2 + loss_z` only. `F_mat` is a
   buffer (no `requires_grad`); it is updated by `updating_F` instead.
   See `recvae/losses.py:76, 79`.

## Why alternating optimization

If we hold `F` fixed and let SGD update encoder/decoder/inference/z, we
are still solving a hard nonlinear problem (the encoder and decoder are
deep nets). But if we hold those nets fixed and ask "what `F` minimizes
`loss2 + loss_F`?", the answer is exact: it's a ridge regression.

Write out the parts of the loss that depend on `F`. The `loss2` term is
averaged over batch and time (`1 / (2 * B * T * sig_h^2)`), while
`loss_F = rho * ||F||_F^2` carries no such averaging. To match the two
on a single scale, multiply through by `2 * N * T * sig_h^2`:

```
J(F) = sum_{n,t} ||h_{n,t} - F h_{n,t-1}||^2 + 2 * N * T * sig_h^2 * rho * ||F||_F^2
```

Stack the posterior states across all subjects and timesteps. Let `Y`
be the `(N*T, D)` matrix of `h_{n,t}` and `X` the same thing shifted
one step (`h_{n,t-1}`, with `h_0` prepended at `t=0`). Then

```
J(F) = ||Y - X F^T||_F^2 + 2 N T sig_h^2 rho ||F||_F^2
```

Setting the gradient w.r.t. `F^T` to zero gives the normal equations:

```
(X^T X + 2 N T sig_h^2 rho I) F^T = X^T Y
F^T = (X^T X + 2 N T sig_h^2 rho I)^{-1} X^T Y
```

The `2*N*T` factor traces back to `loss2`'s normalization by `2*B*T`;
multiplying through to clear the denominator gives the matched ridge
update. Drop the factor and the closed form solves a *different*
objective than the SGD loss — the resulting `F` would be biased toward
zero by a factor of `N*T`.

This is the standard ridge-regression closed form. Gradient descent
would slowly approach the same answer; one linear solve gets you there
in one step. That is exactly the operation at `recvae/model.py:311-331`.

## How the loop alternates

`recvae/train.py:105-130`:

1. For each epoch, iterate over batches. For each batch:
   - Run `training_step` to get `loss = loss1 + loss2 + loss_z`.
   - Stash the posterior trajectory `h_history` (detached) into a big
     `(N_train, T, D)` tensor indexed by subject id.
   - `loss.backward()`, `optimizer.step()` — this updates encoder,
     decoder, inference head, and `z_vectors`.
2. After the epoch finishes, call `model.updating_F(h_history_history,
   h0, rho)`. This is wrapped in `@torch.no_grad()` and reads the
   trajectory accumulated during the SGD pass.

So `F` lags one epoch behind the rest of the parameters. That is fine:
the SGD parameters and `F` improve in lockstep over many epochs, and
the lag matters less than the fact that `F` is always at the optimum
for the latents *as seen at the end of the previous epoch*.

## Run it

```bash
python tutorials/08_losses_alt_optim/ridge_F_update.py
```

The demo synthesizes a small `(N=4, T=8, D=5)` posterior history,
solves the ridge system by hand with `torch.linalg.solve`, then runs
`RecVAEModel.updating_F` on the same inputs and asserts the buffer
matches the hand solve.

## Why this approach

Two reasons to prefer closed-form over SGD when you can use it:

1. **Exactness.** One linear solve gives the global minimum of the
   sub-problem. SGD can only approach it, and with `lr=1e-6` (the
   notebook default) it would approach very slowly.
2. **Decoupling.** Holding `F` at its optimum given the current latents
   means SGD on the rest of the model sees a well-behaved temporal
   regularizer instead of fighting `F` for the same gradient signal.

The trade-off is that the closed form only works for the
linear-Gaussian sub-problem. Replace `g(h) = h @ F^T` with a nonlinear
transition (an MLP, an RNN cell) and you lose the closed form and fall
back to gradient descent on everything.

## Exercise (optional)

Modify `ridge_F_update.py` to plot the residual `||Y - X F^T||_F` as a
function of `rho` from `1e-4` to `1e2`. You should see classic
bias/variance behavior: tiny `rho` overfits to `h_history` and gives
small training residual but unstable `F`; large `rho` shrinks `F`
toward zero and the residual climbs.

## Further reading

- Bishop, *PRML*, Section 3.1.4 — ridge regression as MAP under a
  Gaussian prior.
- Kingma and Welling, "Auto-Encoding Variational Bayes" (2014) — the
  canonical ELBO derivation that `loss2` is approximating.
- `recvae/losses.py:90-163` — `KLRecVAELoss`, the opt-in canonical-KL
  alternative to `loss2`'s MSE proxy.
