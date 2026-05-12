# Variational Autoencoders for Neuroimaging

A tour for researchers familiar with PCA, ICA, and the mass-univariate
GLM who want to understand why a variational autoencoder (VAE) exists,
what it does that linear methods cannot, and where this repository's
model sits in the landscape.

## The latent-variable framing

Classical neuroimaging methods are mostly variations on linear
decomposition:

- **PCA** finds orthogonal directions of maximum variance. It is a
  matrix factorization `X ≈ U S V^T` with no probabilistic model of the
  observation noise on `X` and no assumed prior on the components.
- **ICA** drops orthogonality, imposes statistical independence on the
  components, and is still linear. The model is `X = A S` with `S`
  non-Gaussian and (usually approximately) independent across
  components.
- **The GLM** regresses `X` on a design matrix that encodes
  experimental conditions. It is a supervised model that asks "where in
  the brain does signal track condition $c$", not "what is the
  underlying structure of $X$".

All three are linear in the latent factors. None give a probabilistic
generative model from which you can sample new fMRI-like volumes,
evaluate the likelihood of a held-out scan, or interpolate between two
scans through a smooth latent space.

The latent-variable framing posits a generative process on the data:

```
z ~ p(z)            # prior over latents
x ~ p(x | z; θ)      # likelihood, parameterized by θ
```

`p(x | z; θ)` is the decoder; you build the entire model around it. The
posterior `p(z | x)` is intractable for any interesting `p(x | z; θ)`,
so you cannot just maximize the marginal likelihood `p(x; θ)` directly.

## VAE in one page

The variational autoencoder (Kingma and Welling, 2014; Rezende et al.,
2014) trains the generative model above by introducing an inference
network `q(z | x; φ)` — the encoder — that approximates the intractable
posterior. The training objective is the evidence lower bound (ELBO):

```
ELBO(x) = E_{q(z|x)}[log p(x|z)] − KL(q(z|x) || p(z))
```

- The first term is the expected log-likelihood of `x` given a latent
  sample. For Gaussian likelihoods this is (negative) MSE up to scaling
  and pushes the decoder to reconstruct.
- The second term is the KL regularizer that keeps the approximate
  posterior close to the prior, which keeps the latent space organized.

`q(z | x)` is typically a diagonal Gaussian whose mean and log-variance
the encoder outputs. To backpropagate gradients through a stochastic
sample, the reparameterization trick writes `z = μ + σ · ε` with
`ε ~ N(0, I)`, shunting the randomness to an external noise variable.
See `recvae/model.py:182-193` and `tutorials/05_vae_reparam/` for this
repo's implementation.

The result is a model you can train by SGD, sample from, evaluate
likelihoods on, and use the encoder of as a learned dimensionality
reduction.

## Why a VAE for fMRI

A VAE buys things over PCA/ICA/GLM that matter for fMRI. It is a
probabilistic generative model with continuous latents, so you get a
proper density over `x` (modulo the variational bound), the latent
space has a prior, and you can interpolate between latents, draw new
samples, and ask whether a new scan is typical of the training
distribution. The latents themselves are compressed representations
(10-dimensional per timestep is ~10^5 times smaller than the input
volume) and become attractive features for downstream classification
(CN vs AD), regression (age, cognitive scores), or clustering. The
decoder gives `p(x | z)` explicitly rather than a point prediction,
which matters for calibrated uncertainty, anomaly scoring, or
evidence-style inference. A recurrent variant carries a latent state
forward in time so slow fluctuations and dynamic functional
connectivity have a place to live; a vanilla VAE treats each timepoint
independently and throws temporal dynamics away.

## The recurrent VAE family

Several published architectures extend the VAE to sequences:

- **DKF — Deep Kalman Filter** (Krishnan, Shalit, Sontag 2015). The
  generative model is a Gaussian state-space model with a learned
  nonlinear transition `z_t ~ p(z_t | z_{t-1})` and a learned emission
  `x_t ~ p(x_t | z_t)`. Inference is amortized by an encoder
  conditioned on `(x_t, z_{t-1})` or on the full sequence. The
  transition is the generative bottleneck: everything that happens
  across time has to flow through `z`.
- **VRNN — Variational RNN** (Chung, Kastner, Dinh, Goel, Courville,
  Bengio 2015). A deterministic RNN runs over the data; at each step
  the RNN state is concatenated with `x_t` to produce a stochastic
  `z_t`, which feeds back into the RNN. The deterministic hidden
  carries most of the dynamics, the latent carries the stochasticity.
- **SRNN — Stochastic RNN** (Fraccaro, Sønderby, Paquet, Winther 2016).
  A two-track architecture: a deterministic forward RNN over inputs and
  a separate stochastic chain of `z_t`'s, with a backward smoother for
  posterior inference. More expressive temporal dependencies than DKF,
  at the cost of architectural complexity.

This repository's `RecVAEModel` (`recvae/model.py:72-386`) is closest to
**DKF**: temporal structure lives entirely in the latent recurrence,
the emission is the 3-D conv decoder, and the encoder/inference head
amortizes the posterior. The substantive differences:

- The transition is linear (`g(h) = h F^T`) rather than an MLP or GRU.
- `F` is not trained by gradient. It is re-solved by ridge regression
  in closed form once per epoch (`recvae/model.py:269-331`).
- Per-subject offsets `z_s` (`recvae/model.py:156-158`) are added in
  the latent space to absorb subject-specific variation.

These choices are discussed in `this_models_design.md`.

## Related fMRI deep learning

A short orientation list, not a ranking:

- **Kim, Zhu, Chang, Bao, Wager, Calhoun, 2021. "Representation
  learning of resting-state fMRI with variational autoencoder",
  *NeuroImage*** — applies a VAE to ICA-reduced resting-state data; an
  early and frequently cited template for VAE-on-fMRI.
- **Qiang, Dong, Ge, Liang, Ge, Liu, 2021. "Modeling task-based fMRI
  data via deep belief network with layer-wise pretraining",
  *Frontiers in Neuroscience*** — earlier deep generative approach
  (DBN, not VAE), included to show the lineage.
- **Li, Zhou, Dvornek, Zhang, Gao, Zhuang, Scheinost, Staib, Ventola,
  Duncan, 2021. "BrainGNN: interpretable brain graph neural network
  for fMRI analysis", *Medical Image Analysis*** — graph neural network
  operating on parcellated, region-of-interest-level connectivity
  rather than voxels.
- **Caro, Oliveira-Fonoff, Park, Jia, Jadi, Norman-Haignere, Pillow,
  Hasson, 2023+. "BrainLM" / "fMRI-PTE"** — transformer-based
  large-scale pretraining over many subjects and datasets; the current
  direction of travel for foundation models in fMRI.
- **fMRI-Transformer / neuroformer family** — generic transformer
  architectures applied to fMRI timeseries with masked-modeling
  pretraining.

The right baseline depends on data scale, the downstream task, and how
much pretraining data is available.

## Stronger alternatives to this repo's model

This repository is a teaching scaffold. For peer-reviewed predictive
performance on ADNI the model here is not the best option. Stronger
choices, ordered roughly by capacity needed:

- **Classical functional connectivity + ML.** Compute the Pearson
  correlation matrix between parcellated region timeseries, feed the
  flattened upper triangle into a regularized linear classifier or
  gradient-boosted trees. With careful subject-level CV and matched
  demographics this is a remarkably strong baseline that many deep
  models fail to clear.
- **3D-CNN classifier.** Skip the generative model entirely; train a
  3-D CNN supervised on volume → label. Simpler to tune, fewer moving
  parts, competitive on cohort-classification tasks.
- **3D autoencoder + downstream classifier.** Pretrain a deterministic
  autoencoder for reconstruction, then train a classifier on the
  bottleneck features. Strictly more capacity per parameter than a VAE
  if you don't need the generative properties.
- **Pretrained transformer backbones** — BrainLM, fMRI-PTE,
  neuroformer-style models. Best results published to date on
  cohort-level tasks when you have access to large pretraining corpora.

The recurrent-VAE-with-closed-form-ridge formulation here is
interesting as a study object: alternating optimization, latent
dynamics, and the gap between MAP point estimation and proper
variational inference, in code small enough to read in an afternoon.

## Reading list

Foundational:

- Kingma, D. P., and Welling, M. (2014). Auto-encoding variational
  Bayes. *International Conference on Learning Representations*.
  arXiv:1312.6114.
- Rezende, D. J., Mohamed, S., and Wierstra, D. (2014). Stochastic
  backpropagation and approximate inference in deep generative models.
  *International Conference on Machine Learning*. arXiv:1401.4082.

Recurrent / sequential VAEs:

- Krishnan, R. G., Shalit, U., and Sontag, D. (2015). Deep Kalman
  filters. arXiv:1511.05121.
- Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A. C., and
  Bengio, Y. (2015). A recurrent latent variable model for sequential
  data. *NeurIPS*. arXiv:1506.02216.
- Fraccaro, M., Sønderby, S. K., Paquet, U., and Winther, O. (2016).
  Sequential neural models with stochastic layers. *NeurIPS*.
  arXiv:1605.07571.

fMRI applications:

- Kim, J.-H., Zhu, B., Chang, M., Bao, R., Wager, T. D., Calhoun, V.
  D. (2021). Representation learning of resting-state fMRI with
  variational autoencoder. *NeuroImage*, 241, 118423.
- Qiang, N., Dong, Q., Ge, F., Liang, H., Ge, B., Zhang, S., Liu, T.
  (2021). Modeling task-based fMRI data via deep belief network with
  layer-wise pretraining. *Frontiers in Neuroscience*, 14, 588296.
- Li, X., Zhou, Y., Dvornek, N. C., Zhang, M., Gao, S., Zhuang, J.,
  Scheinost, D., Staib, L. H., Ventola, P., and Duncan, J. S. (2021).
  BrainGNN: Interpretable brain graph neural network for fMRI analysis.
  *Medical Image Analysis*, 74, 102233.
- Caro, J. O., et al. (2023). BrainLM: A foundation model for brain
  activity recordings. *bioRxiv* / *ICLR* preprint.

Critique of fMRI ML practice:

- Varoquaux, G. (2018). Cross-validation failure: small sample sizes
  lead to large error bars. *NeuroImage*, 180, 68-77. Why CV estimates
  are noisier than they look.
- Marek, S., et al. (2022). Reproducible brain-wide association studies
  require thousands of individuals. *Nature*, 603, 654-660. Why the
  sample-size bar is much higher than the field has assumed.
