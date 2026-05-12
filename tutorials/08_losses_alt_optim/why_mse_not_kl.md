# Why `loss2` is an MSE and not a KL term

## Short answer

The canonical VAE ELBO has a KL divergence as its regularizer. The
RecVAE notebook this repo descends from used a squared-error term in
its place. That choice is preserved in `recvae/losses.py:73-74`, and
the canonical-KL alternative is provided as the opt-in `KLRecVAELoss`
at `recvae/losses.py:90-163`.

This file explains what the canonical form would look like and why the
substitution matters.

## What an ELBO would look like

For a VAE with prior `p(h_t | h_{t-1})` and approximate posterior
`q(h_t | x_t, h_{t-1})`, the ELBO per timestep is

```
ELBO_t = E_{q}[ log p(x_t | h_t) ]  -  KL( q(h_t | x_t, h_{t-1}) || p(h_t | h_{t-1}) )
```

The first term is the negative reconstruction loss (matched by `loss1`
under a Gaussian observation model). The second term is the KL between
posterior and prior. In RecVAE the prior is a Gaussian centered on the
linear transition:

```
p(h_t | h_{t-1}) = N( g(h_{t-1}), sig_h^2 I )    with    g(h) = h @ F^T
```

and the posterior comes from the inference head:

```
q(h_t | x_t, h_{t-1}) = N( mu_h, diag(exp(log_var_h)) )
```

For two diagonal Gaussians the KL has a closed form. With `D = latent_dim`,
posterior variance `sigma^2_d = exp(log_var_h)_d`, posterior mean
`mu`, prior mean `m = g(h_{t-1})`, and prior variance `s^2 = sig_h^2`:

```
KL = 0.5 * sum_d [
        log(s^2 / sigma^2_d)
      + (sigma^2_d + (mu_d - m_d)^2) / s^2
      - 1
    ]
```

That is the term `loss2` *would* be if RecVAE implemented the ELBO.

## What `loss2` actually computes

`recvae/losses.py:73-74`:

```python
loss2 = sum((h - gh).pow(2).sum() for h, gh in zip(h_history, gh_history))
loss2 = loss2 / (self.sig_h**2) / denom
```

That is `||h_t - g(h_{t-1})||^2 / sig_h^2`, summed over time and
averaged over the batch. Note that `h_t` here is the *sampled* posterior
state from `reparametrize`, not the posterior mean. The variance terms
in the KL above are gone; only the squared-mean-offset survives. That
is what you get if you treat `h_t` as a point estimate and put a
Gaussian prior with mean `g(h_{t-1})` and variance `sig_h^2` on it —
i.e. MAP inference on the latent path.

So the current code is doing maximum-a-posteriori estimation of the
latent trajectory, not variational inference over it. The inference
head still produces `mu_h` and `log_var_h` and the reparam trick still
runs (so gradients flow), but the regularizer ignores the variance.

## Why it matters

- **Theoretical.** The current objective is not a lower bound on the
  marginal likelihood. The relationship to maximum likelihood is
  weaker than for a true ELBO.
- **Practical.** Without the `0.5 * sigma^2 / sig_h^2` and `-0.5 *
  log(sigma^2)` terms, there is no pressure on the posterior variance.
  `log_var_h` can drift arbitrarily; the model gets no signal to
  calibrate its uncertainty.
- **Numerical.** The MSE form is simpler and harder to blow up than
  a KL with `exp(log_var)` terms, which is part of why the original
  notebook used it.

## Sketch of a proper KL term

A drop-in replacement, computed inside `forward` (so we have access to
`mu_h`, `log_var_h` per timestep — currently they are discarded after
`reparametrize`):

```python
# Per-timestep KL between q(h_t|.) = N(mu_h, diag(exp(log_var_h)))
# and p(h_t|h_{t-1}) = N(g(h_prev), sig_h^2 I).
sigma2 = torch.exp(log_var_h)         # (B, D)
m_prior = self.g_transform(h_prev)    # (B, D)
s2 = self.cfg.sig_h ** 2
kl = 0.5 * (
    torch.log(torch.full_like(sigma2, s2)) - log_var_h
    + (sigma2 + (mu_h - m_prior).pow(2)) / s2
    - 1
).sum(dim=1)                          # (B,)
```

Then `loss2_kl = kl.mean()` (or summed over time and averaged over
batch as `loss2` currently is). Doing this properly requires plumbing
`mu_h` and `log_var_h` out of `vae_step` so `forward` can collect them;
that refactor is out of scope for this lesson.

## What to take away

- `loss2` is *not* the KL term of a textbook VAE; it is an MSE proxy
  that does MAP-style point estimation of the latent trajectory.
- `KLRecVAELoss` at `recvae/losses.py:90-163` implements the proper KL
  alternative; swap it in via the `loss_fn=` argument to
  `RecVAEModel.training_step`. The wider change — propagating `mu_h`
  and `log_var_h` out of `vae_step` so the KL term can use them — is a
  research-grade extension, not a one-line fix.
- Until that change is made, treat the inference head's `log_var_h` as
  a parameter the model is free to ignore.
