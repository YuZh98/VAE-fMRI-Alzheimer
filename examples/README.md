# examples/

Short runnable variant scripts that fork the canonical RecVAE in
pedagogically useful directions. Each file is self-contained, runs on CPU
in under 90 seconds, and exits 0.

Every script seeds the RNG with `2022`, uses `synthetic_cohort` for data,
and starts with a bootstrap shim that puts the repo root on `sys.path`
so `import recvae` works without `pip install -e .`.

| script | one-line pitch |
| --- | --- |
| [`classify_from_latents.py`](classify_from_latents.py) | Train RecVAE, mean-pool h_t to a subject vector, then probe CN-vs-AD with logistic regression and compare to a flattened-volume PCA baseline. |
| [`swap_adamw.py`](swap_adamw.py) | Compare canonical SGD@1e-6 against AdamW@1e-4 + cosine annealing; shows both `opt_func=` and a custom-loop pattern. |
| [`with_kl_term.py`](with_kl_term.py) | Train once with the default `RecVAELoss` (MSE proxy) and once with `KLRecVAELoss` (closed-form KL on the temporal prior), side-by-side. |
| [`learnable_sigma.py`](learnable_sigma.py) | Subclass `RecVAEModel` so `sig_x`, `sig_h`, `sig_z` become softplus-reparameterized learnable parameters. |
| [`groupnorm_swap.py`](groupnorm_swap.py) | Walk the model, swap every `BatchNorm3d` for `GroupNorm`, then verify forward shape + train briefly. |
| [`cohort_specific_F.py`](cohort_specific_F.py) | Keep `F_cn` and `F_ad` transition matrices and route each subject through the right one based on its label. |
| [`amortized_z.py`](amortized_z.py) | Replace per-subject `z_vectors` with a small 3D-conv encoder `Z_Enc`; time amortized inference vs `evaluate_held_out`'s inner SGD. |

Run any example directly:

```bash
.venv/bin/python examples/classify_from_latents.py
```
