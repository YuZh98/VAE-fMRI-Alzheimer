# Background Docs

Three bridging notes for crossing the gap between machine-learning
practice and neuroimaging practice in this repository.

- [`fmri_101.md`](fmri_101.md) — What fMRI data is, what the volumes
  look like, what preprocessing this repo does *not* do, and the
  ADNI-specific data-handling constraints. Read this if you've never
  loaded a NIfTI file.
- [`vae_for_neuroimaging.md`](vae_for_neuroimaging.md) — A tour of
  variational autoencoders for a reader who already knows PCA, ICA, and
  the GLM, plus how the recurrent-VAE family sits among other deep
  learning approaches to fMRI.
- [`this_models_design.md`](this_models_design.md) — A line-by-line
  walkthrough of why each architectural and training choice in
  `recvae/model.py` was made, marking each choice as standard or
  unusual and flagging the honest weak spots.
