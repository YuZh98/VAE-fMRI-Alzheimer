# Legacy notebooks

These notebooks are kept for historical reference only. They are **not**
maintained and may not run with the current `recvae` package.

| Notebook | Status | Notes |
|----------|--------|-------|
| `Version1.ipynb` | Archived | Earliest prototype. `g_transform` is a 3-layer MLP with Frobenius-norm parametrization. Uses `loss2 = 0`. |
| `Version2.ipynb` | Archived | Second prototype with subject-specific `F_s`, `latent_dim=50`, trains on only 4 images (`imgs = [...][0,1,2,5]`). |
| `Version3.ipynb` | Archived | Switches to matrix `g_transform`. Notable ablation: `z_dim=2` (sparse subject noise). |
| `Version4.ipynb` | Archived | Final un-cleaned version. Functionally equivalent to `notebooks/RecVAE_on_fMRI.ipynb` (the canonical version). |

The progression V1 → V4 documents the iterative refinement of the model:
- V1: MLP transition, no temporal-prior loss
- V2: subject-specific transition matrices `F_s`
- V3: introduces sparse subject noise `z_s` with `z_dim=2`
- V4: shared `F`, `z_dim = latent_dim = 10`

For the current implementation, see `notebooks/RecVAE_on_fMRI.ipynb` and the
`recvae/` package.

If you need to inspect or rerun a legacy notebook, restore the repository to
the commit prior to the legacy move:

    git log --oneline --diff-filter=R -- legacy/Version1.ipynb
