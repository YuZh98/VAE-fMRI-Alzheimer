# Lesson 05: VAE reparameterization

## What you'll learn

- A one-paragraph sketch of the variational lower bound (ELBO).
- Why you cannot backprop through a sampling step `z ~ N(mu, sigma)`.
- The reparameterization trick: rewrite the sample as
  `z = mu + sigma * eps` with `eps ~ N(0, I)`. Now gradients flow through
  `mu` and `sigma`.
- Why `eps` must be allocated on the same `device` and `dtype` as `mu` —
  and why you should *never* rely on `torch.set_default_tensor_type` for
  this.

## Where this lives in the repo

`recvae/model.py:135-146`. The method docstring also calls out exactly
which mistake the canonical notebook made.

## The concept

### ELBO in one paragraph

A VAE models data `x` through a latent `z` with a learned prior `p(z)`
and a learned decoder `p(x | z)`. The marginal `log p(x)` is intractable,
so we maximize a lower bound by introducing an inference network
`q(z | x)`:

```
log p(x) >= E_{q(z|x)}[ log p(x | z) ] - KL( q(z | x) || p(z) )
            \_______ reconstruction _______/   \____ regularizer ____/
```

The expectation is approximated by a single Monte Carlo sample
`z ~ q(z | x)`. To train, we need a gradient of the bound with respect to
the parameters of `q`, including `mu` and `sigma`.

### Why you can't differentiate through a sample

`z ~ N(mu, sigma)` is a random draw. The operation "sample" is not a
differentiable function of `mu` and `sigma`: even if you tabulated it,
the *value* of `z` depends on a random number generator, not on `mu` in
any analytic way. PyTorch's autograd has no edge to attach a gradient
to.

### Reparameterization

Push the randomness outside the parameters:

```
eps ~ N(0, I)      <-- random, independent of (mu, sigma)
z   = mu + sigma * eps
```

Now `z` is a deterministic function of `mu`, `sigma`, and a noise tensor
`eps`. Gradients flow through `mu` (additive) and `sigma` (multiplicative)
exactly like through any other op. Only `eps` is "non-differentiable",
but we don't need its gradient — it isn't a parameter.

In code this is two lines:

```python
sigma = torch.exp(log_var / 2)
eps   = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype)
z     = mu + sigma * eps
```

(`log_var` instead of `sigma` directly because it can be any real number
without breaking positivity.)

### Why `device` and `dtype` must be explicit

The original notebook contained:

```python
eps = torch.randn(size=mu.shape)   # lands on CPU regardless of mu's device
```

It worked anyway only because the notebook also called
`torch.set_default_tensor_type('torch.cuda.FloatTensor')` at the top,
which makes every fresh tensor land on CUDA by default. That global flag
is now deprecated and has several footguns:

- It does not exist on MPS (Apple Silicon).
- It makes *every* helper script and *every* third-party library you
  import on this process also default to CUDA, which is hostile to
  composability.
- It silently breaks if no CUDA device is present — the model fails to
  construct, not in `reparametrize`.

The fix in `recvae/model.py:144-145` is:

```python
eps = torch.randn(mu_h.shape, device=mu_h.device, dtype=mu_h.dtype)
```

This is device- and dtype-agnostic: `eps` lives wherever `mu` lives, in
whatever precision `mu` uses. No global state needed.

## Code walk

`reparam_demo.py`:

1. Implements `reparametrize(mu, log_var)` from scratch (one screen of
   code).
2. Runs it on a tiny `(2, 10)` `mu` and `log_var` and prints the result.
3. Calls `.backward()` on a sum of the output and prints `mu.grad` and
   `log_var.grad` — proving gradients flow through both parameters.
4. Confirms the same answer (in shape and device) when calling
   `RecVAEModel.reparametrize` from the real package.
5. Demonstrates with a printout that `torch.randn(2, 10)` lands on CPU by
   default, regardless of where you happen to be working — and that
   passing `device=...` is the only correct way to land elsewhere.

## Run it

```bash
python tutorials/05_vae_reparam/reparam_demo.py
```

## Why this approach

Reparameterization is the simplest of a family of "pathwise gradient"
tricks for differentiating through random samples (see Kingma & Welling
2013 for VAEs, Mohamed et al. 2020 for a survey). For Gaussian latents
it's a one-liner; for other distributions you may need rejection sampling
or score-function estimators (REINFORCE), which have much higher
variance.

## Further reading

- Kingma & Welling, "Auto-Encoding Variational Bayes" (arXiv:1312.6114).
  Sections 2.3-2.4 derive the reparameterization trick.
- Doersch, "Tutorial on Variational Autoencoders" (arXiv:1606.05908).
  Section 2 has the ELBO derivation with one less leap.
- `recvae/model.py:135-146` — the implementation you've now read.
- Lesson 10 in this tutorial series covers `device`-agnostic code in
  more depth, including why `torch.set_default_tensor_type` is bad
  practice.
