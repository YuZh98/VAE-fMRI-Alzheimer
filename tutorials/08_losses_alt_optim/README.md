# Lesson 08: Losses and alternating optimization

## What you'll learn

The four loss terms that drive RecVAE training, why one of them (the
transition matrix `F`) is *not* trained by gradient descent, and how the
training loop alternates per-epoch SGD with a closed-form ridge update.
By the end you can read `training_step` and `updating_F` in
`recvae/model.py` and explain the math behind each line.

## Where this lives in the repo

- `recvae/model.py:257-302` — `training_step`: composes the four loss terms.
- `recvae/model.py:209-254` — `updating_F`: closed-form ridge solve.
- `recvae/train.py:73-103` — alternation: SGD inside the batch loop, then
  one `updating_F` call at the end of each epoch.
- `recvae/config.py` — `sig_x`, `sig_h`, `lambda_z`, `rho` defaults.

## The concept

RecVAE's objective has four pieces. Three of them are added into a scalar
loss that gets `.backward()`-ed; the fourth is reported only and solved
in closed form.

### The four terms

Per batch of `B` subjects rolled out over `T` timesteps:

1. `loss1` — per-volume reconstruction MSE divided by `sig_x^2`.
   ```
   loss1 = sum_t ||x_t - mu_t||^2 / (2 * B * T * sig_x^2)
   ```
   The `2 * B * T` denominator averages over batch and timesteps; the
   `sig_x^2` weight reflects the per-pixel observation noise of a
   Gaussian likelihood. See `recvae/model.py:281-283`.

2. `loss2` — temporal-prior MSE divided by `sig_h^2`.
   ```
   loss2 = sum_t ||h_t - g(h_{t-1})||^2 / (2 * B * T * sig_h^2)
   ```
   where `g(h) = h @ F^T` is the linear transition. This term pulls the
   posterior path `h_1, h_2, ...` toward a sequence predicted by the
   transition matrix `F`. See `recvae/model.py:285-286`.

   **Important caveat.** This is *not* the KL term a canonical VAE ELBO
   would have. It is an MSE point-estimate proxy that does MAP-style
   inference on the latent path. The TODO at `recvae/model.py:273-276`
   flags this explicitly. See `why_mse_not_kl.md` in this lesson for
   what a real KL term would look like.

3. `loss_z` — L1 sparsity on the subject-specific noise vectors.
   ```
   loss_z = lambda_z * ||z||_1
   ```
   `z_vectors` is a `(N_train, latent_dim)` Parameter (see Lesson 06).
   L1 keeps most subject offsets near zero so the shared decoder
   carries the work. See `recvae/model.py:289`.

4. `loss_F` — Frobenius regularizer on `F`.
   ```
   loss_F = rho * ||F||_F^2
   ```
   **Reported but not back-propped.** `loss_F` is computed and returned
   in the loss dict for logging, but the total loss that gets
   `.backward()`-ed is `loss1 + loss2 + loss_z` only. `F_mat` is a buffer
   (no `requires_grad`); it is updated by `updating_F` instead. See
   `recvae/model.py:288, 291`.

### Why alternating optimization

If we held `F` fixed and let SGD update encoder/decoder/inference/z, we
would still be solving a hard nonlinear problem (the encoder and decoder
are deep nets). But if we hold those nets fixed and ask "what `F`
minimizes `loss2 + loss_F`?", the answer is exact: it's a ridge
regression.

Concretely, write out the parts of the loss that depend on `F`. Drop
constants and grouping factors and just look at the structure:

```
J(F) = sum_{n,t} ||h_{n,t} - F h_{n,t-1}||^2 + 2 * sig_h^2 * rho * ||F||_F^2
```

Stack the posterior states across all subjects and timesteps. Let
`Y` be the `(N*T, D)` matrix of `h_{n,t}` and `X` the same thing
shifted one step (`h_{n,t-1}`, with `h_0` prepended at `t=0`). Then

```
J(F) = ||Y - X F^T||_F^2 + 2 sig_h^2 rho ||F||_F^2
```

Setting the gradient w.r.t. `F^T` to zero gives the normal equations:

```
(X^T X + 2 sig_h^2 rho I) F^T = X^T Y
F^T = (X^T X + 2 sig_h^2 rho I)^{-1} X^T Y
```

This is the standard ridge-regression closed form. Gradient descent
would slowly approach the same answer; one linear solve gets you there
in one step. That is exactly the operation in `recvae/model.py:236-254`.

### How the loop alternates

`recvae/train.py:73-103`:

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

## Code walk

Open `recvae/model.py` and read these blocks together with the math
above:

- `recvae/model.py:281-291` — the four-line composition of `loss1`,
  `loss2`, `loss_z`, `loss_F`, and the deliberate omission of `loss_F`
  from the summed `loss`.
- `recvae/model.py:243-247` — building `Y` and `X` by reshaping
  `h_history_history` and shifting it by one timestep.
- `recvae/model.py:249-254` — the actual solve, `torch.linalg.solve(XX +
  rho_I, XY)`, and the in-place `F_mat.copy_` that preserves buffer
  registration.

Then read `recvae/train.py:73-103` and identify the two updates:
`optimizer.step()` (line 90) is the SGD step; `model.updating_F(...)`
(line 103) is the closed-form step.

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

1. **Exactness.** One linear solve gives you the global minimum of the
   sub-problem. SGD can only approach it, and with `lr=1e-6` (the
   notebook default), it would approach very slowly.
2. **Decoupling.** Holding `F` to its optimum given the current latents
   means SGD on the rest of the model sees a well-behaved temporal
   regularizer instead of fighting `F` for the same gradient signal.

The trade-off is that the closed form only works for the linear-Gaussian
sub-problem. If you ever replaced `g(h) = h @ F^T` with a nonlinear
transition (an MLP, an RNN cell), you would lose the closed form and
fall back to gradient descent on everything.

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
- `recvae/model.py:273-276` — the in-source TODO that calls out the
  KL-vs-MSE substitution as a research item.
