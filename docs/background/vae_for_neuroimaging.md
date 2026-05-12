# Variational Autoencoders for Neuroimaging

For researchers familiar with PCA, ICA, and the mass-univariate GLM who
want a fast tour of why a variational autoencoder (VAE) exists, what it
does that linear methods cannot, and where this repository's model fits
in the landscape.

## The latent-variable framing

Classical neuroimaging methods are mostly variations on **linear
decomposition**:

- **PCA** finds orthogonal directions of maximum variance. It is a
  matrix factorization `X ≈ U S V^T` with no probabilistic model of the
  observation noise on `X`, and no assumed prior on the components.
- **ICA** drops orthogonality, imposes statistical independence on the
  components, and is still linear. The model is `X = A S` with `S`
  non-Gaussian and (usually approximately) independent across
  components.
- **The GLM** regresses `X` on a *design matrix* that encodes
  experimental conditions. It is a supervised model that asks "where in
  the brain does signal track condition $c$", not "what is the
  underlying structure of $X$".

All three are linear in the latent factors. None give you a probabilistic
generative model from which you can **sample** new fMRI-like volumes,
**evaluate** the likelihood of a held-out scan, or **interpolate**
between two scans through a smooth latent space.

The latent-variable framing puts a posited generative process on the
data:

```
z ~ p(z)            # prior over latents
x ~ p(x | z; θ)      # likelihood, parameterized by θ
```

`p(x | z; θ)` is the decoder; you build the entire model around it. The
catch is that the posterior `p(z | x)` is intractable for any
interesting `p(x | z; θ)`, so you cannot just maximize the marginal
likelihood `p(x; θ)` directly.

## VAE in one page

The variational autoencoder (Kingma and Welling, 2014; Rezende et al.,
2014) trains the generative model above by introducing an **inference
network** `q(z | x; φ)` — the *encoder* — that approximates the
intractable posterior. The training objective is the **evidence lower
bound (ELBO)**:

```
ELBO(x) = E_{q(z|x)}[log p(x|z)] − KL(q(z|x) || p(z))
```

- The first term is the **expected log-likelihood** of `x` given a
  latent sample — for Gaussian likelihoods this is (negative) MSE up to
  scaling, and it pushes the decoder to reconstruct.
- The second term is the **KL regularizer** that keeps the
  approximate posterior close to the prior, which keeps the latent
  space organized.

`q(z | x)` is typically a diagonal Gaussian whose mean and log-variance
the encoder outputs. To backpropagate gradients through a stochastic
sample, the **reparameterization trick** writes
`z = μ + σ · ε` with `ε ~ N(0, I)` so the randomness is shunted to an
external noise variable. See `recvae/model.py:182-193` and
`tutorials/05_vae_reparam/` for this repo's implementation.

The result is a model that you can train by SGD, sample from, evaluate
likelihoods on, and use the encoder of as a learned dimensionality
reduction.

## Why a VAE for fMRI

Four pragmatic reasons specifically:

1. **Probabilistic generative model with continuous latents.** Unlike
   PCA/ICA you get a proper density over `x` (modulo the variational
   bound), and unlike a deterministic autoencoder the latent space has
   a prior. You can interpolate between latents and decode
   intermediates, draw new samples, and ask whether a new scan is
   typical of the training distribution.

2. **Latents as compressed representations.** A 10-dimensional latent
   per timestep is ~10^5 times smaller than the input volume. Once
   trained, those latents are an attractive feature for downstream
   classification (CN vs AD), regression (age, cognitive scores), or
   clustering.

3. **Built-in noise model.** The decoder gives you `p(x | z)`
   explicitly, not just a point prediction. That matters whenever you
   want calibrated uncertainty, anomaly scoring, or evidence-style
   inference.

4. **Recurrent latents capture temporal structure.** A vanilla VAE
   treats each timepoint independently and throws away the temporal
   dynamics of resting-state fMRI. A recurrent variant carries a latent
   state forward in time, so slow fluctuations and dynamic functional
   connectivity have a place to live.

## The recurrent VAE family

Several published architectures extend the VAE to sequences. Roughly:

- **DKF — Deep Kalman Filter** (Krishnan, Shalit, Sontag 2015). The
  generative model is a Gaussian state-space model with a learned
  nonlinear transition `z_t ~ p(z_t | z_{t-1})` and a learned emission
  `x_t ~ p(x_t | z_t)`. Inference is amortized by an encoder that
  conditions on `(x_t, z_{t-1})` or on the full sequence. The
  transition is **the** generative bottleneck — everything that
  happens across time has to flow through `z`.

- **VRNN — Variational RNN** (Chung, Kastner, Dinh, Goel, Courville,
  Bengio 2015). A deterministic RNN runs over the data; at each step,
  the RNN state is concatenated with `x_t` to produce a stochastic
  `z_t`, which then feeds back into the RNN. The deterministic hidden
  carries most of the dynamics; the latent carries the stochasticity.

- **SRNN — Stochastic RNN** (Fraccaro, Sønderby, Paquet, Winther 2016).
  A two-track architecture: a deterministic forward RNN over inputs and
  a separate stochastic chain of `z_t`'s, with a backward smoother for
  posterior inference. More expressive temporal dependencies than DKF
  at the cost of architectural complexity.

This repository's `RecVAEModel` (`recvae/model.py:72-386`) is closest in
spirit to **DKF**: the temporal structure lives entirely in the latent
recurrence, the emission is the 3-D conv decoder, and the
encoder/inference head amortizes the posterior. The substantive
differences are:

- The transition is **linear** (`g(h) = h F^T`) rather than an MLP or
  GRU.
- `F` is not trained by gradient — it is re-solved by **ridge
  regression in closed form** once per epoch
  (`recvae/model.py:269-331`).
- Per-subject offsets `z_s` (`recvae/model.py:156-158`) are added in
  the latent space to absorb subject-specific variation.

These choices are unusual and discussed in `this_models_design.md`.

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
  for fMRI analysis", *Medical Image Analysis*** — graph neural
  network operating on parcellated, region-of-interest-level
  connectivity rather than voxels.
- **Caro, Oliveira-Fonoff, Park, Jia, Jadi, Norman-Haignere, Pillow,
  Hasson, 2023+. "BrainLM" / "fMRI-PTE"** — transformer-based
  large-scale pretraining over many subjects and datasets; the current
  direction of travel for foundation models in fMRI.
- **fMRI-Transformer / neuroformer family** — generic transformer
  architectures applied to fMRI timeseries with masked-modeling
  pretraining.

No claim about which is "best" is made; the right baseline depends on
data scale, the downstream task, and how much pretraining data is
available.

## Stronger alternatives to this repo's model

Honest framing: this repository is a teaching scaffold. If your goal is
peer-reviewed predictive performance on ADNI, the model here is **not**
your best option. Stronger choices, ordered roughly by capacity needed:

- **Classical functional connectivity + ML.** Compute the Pearson
  correlation matrix between parcellated region timeseries, feed the
  flattened upper triangle into a regularized linear classifier or
  gradient-boosted trees. With careful subject-level CV and matched
  demographics this is a remarkably strong baseline that many deep
  models fail to clear.
- **3D-CNN classifier.** Skip the generative model entirely; train a
  3-D CNN supervised on volume → label. Simpler to tune, fewer moving
  parts, and competitive on cohort-classification tasks.
- **3D autoencoder + downstream classifier.** Pretrain a deterministic
  autoencoder for reconstruction, then train a classifier on the
  bottleneck features. Strictly more capacity per parameter than a VAE
  if you don't need the generative properties.
- **Pretrained transformer backbones** — BrainLM, fMRI-PTE,
  neuroformer-style models. Best results published to date on
  cohort-level tasks when you have access to large pretraining
  corpora.

The recurrent-VAE-with-closed-form-ridge formulation here is interesting
**as a study object**: it exposes alternating optimization, latent
dynamics, and the gap between MAP point estimation and proper
variational inference in code small enough to read in an afternoon.
That is its actual purpose.

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
  lead to large error bars. *NeuroImage*, 180, 68-77. — Why your CV
  estimates are noisier than you think.
- Marek, S., et al. (2022). Reproducible brain-wide association
  studies require thousands of individuals. *Nature*, 603, 654-660. —
  Why the sample-size bar is much higher than the field has assumed.
