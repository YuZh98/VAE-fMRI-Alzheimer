# Before / After

Seven concrete examples. Each shows the legacy notebook idiom on the
left and the `recvae/` rewrite on the right, with file:line citations
so you can open the package source and read the real thing.

The "Before" snippets are paraphrases of the patterns in
`legacy/Version4.ipynb` (and its near-identical predecessors V1-V3).
They are not byte-for-byte transcriptions — notebook cells contain a
lot of incidental scaffolding — but they are faithful to the
notebooks' behavior and gotchas.

---

## 1. RNG seeding

**Before (notebook):**

```python
seed = 2022
torch.manual_seed(seed)
```

This only pins torch's CPU RNG. Python's `random`, NumPy's `np.random`,
the CUDA RNG on each device, and cuDNN's autotune all keep using fresh
randomness. A "seeded" notebook run is still non-reproducible whenever
any of those is touched (data shuffles using NumPy, dropout on CUDA,
etc).

**After (`recvae/utils.py:13-26`):**

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Five RNGs pinned plus the cuDNN determinism toggle. Cost: cuDNN may run
slower (no autotune). Benefit: identical numbers across reruns,
including on multi-GPU.

---

## 2. Default device for new tensors

**Before (notebook):**

```python
torch.set_default_tensor_type('torch.cuda.FloatTensor')
...
eps = torch.randn(size=mu_h.shape)  # implicitly lands on CUDA
```

`set_default_tensor_type` is a process-wide global. It crashes on a
CPU-only machine (the tensor type does not exist), it breaks MPS
(Apple Silicon), and it makes every `torch.randn`, `torch.zeros`, etc.
in the same Python process implicitly device-bound.

**After (`recvae/model.py:135-146`, especially line 145):**

```python
def reparametrize(self, mu_h, log_var_h):
    sigma_h = torch.exp(log_var_h / 2)
    eps = torch.randn(mu_h.shape, device=mu_h.device, dtype=mu_h.dtype)
    return mu_h + sigma_h * eps
```

No global state. `eps` lands on whatever device `mu_h` is on, with
matching dtype. Works on CPU, CUDA, MPS, and any future backend without
changes.

---

## 3. DataLoader generator

**Before (notebook):**

```python
generator = torch.Generator(device='cuda').manual_seed(seed)
train_loader = DataLoader(ds, batch_size=4, shuffle=True, generator=generator)
```

`torch.Generator(device='cuda')` crashes immediately on a CPU-only
machine. It is also unnecessary — the indices a `DataLoader` shuffles
are Python ints living in the host process; they never touch the GPU.

**After (`recvae/data.py:130-148`, especially 142-148):**

```python
def build_dataloader(dataset, batch_size, shuffle=True, seed=None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
```

CPU generator. Works everywhere. Same shuffle order for a given seed
regardless of where the data lives.

---

## 4. `z_vectors` as a Parameter

**Before (notebook):**

```python
class RecVAEModel(nn.Module):
    def __init__(self, train_size, ...):
        super().__init__()
        ...
        self.z_vectors = torch.randn(train_size, z_dim) * sig_z
```

A plain tensor attribute. Three consequences:

- Not in `model.parameters()`, so the optimizer was passed a *second*
  param group `[model.z_vectors]` by hand. Easy to forget.
- Not in `model.state_dict()`, so `torch.save(model.state_dict(), ...)`
  silently drops the learned subject offsets.
- Not moved by `model.to(device)`. If you move the model to CUDA,
  `z_vectors` stays on CPU and the next forward pass crashes.

**After (`recvae/model.py:109-111`):**

```python
z_init = torch.randn(train_size, cfg.latent_dim) * cfg.sig_z
self.z_vectors = nn.Parameter(z_init)
```

Now `z_vectors` is a real `nn.Parameter`: in `parameters()`, in
`state_dict()`, follows `to(device)`. The optimizer line in
`recvae/train.py:58` collapses from "two param groups" to one
`opt_func(model.parameters(), lr=lr)`.

---

## 5. `F_mat` as a buffer

**Before (notebook):**

```python
self.F_mat = torch.rand(latent_dim, latent_dim)
```

Same problem as `z_vectors`. Worse, `F_mat` is updated by a closed-form
solve every epoch, so any state to be checkpointed *is* the current
`F_mat`. Saving without it means a reloaded model has a randomly
initialized `F` and you have to retrain from scratch.

**After (`recvae/model.py:113-116`):**

```python
self.register_buffer("F_mat", torch.rand(cfg.latent_dim, cfg.latent_dim))
```

`register_buffer` is the right tool here: `F_mat` is part of model
state but has no gradient. Now it appears in `state_dict()` and moves
with `.to(device)`, but is not handed to the optimizer. The in-place
update at `recvae/model.py:254` (`self.F_mat.copy_(new_F_T.T)`)
preserves the registration.

---

## 6. NIfTI timepoint validation

**Before (notebook):**

```python
for name in filenames:
    arr = nib.load(os.path.join(directory, name)).get_fdata()
    t = torch.from_numpy(arr[..., :120]).float().unsqueeze(0)
    tensors.append(t)
volumes = torch.stack(tensors, dim=0)
```

If one of the NIfTI files has fewer than 120 timepoints, `arr[..., :120]`
silently returns whatever's there (say 90 timepoints). The `torch.stack`
then either crashes with an unhelpful message about non-matching sizes
along dim -1, or — if all volumes happen to be short by the same amount
— it succeeds and you train on the wrong data without knowing.

**After (`recvae/data.py:65-68`):**

```python
if arr.shape[-1] < tol_time:
    raise ValueError(
        f"{name}: only {arr.shape[-1]} timepoints, need >= {tol_time}",
    )
```

Fail loud at the first short file, with a message that names the file
and the actual count. The error is at *load* time, not deep inside
`stack` or — worse — silently downstream.

---

## 7. Zero-span min-max normalization

**Before (notebook):**

```python
max_values = torch.amax(volumes, dim=(1,2,3,4,5))
min_values = torch.amin(volumes, dim=(1,2,3,4,5))
for i in range(volumes.shape[0]):
    volumes[i] = 2 * ((volumes[i] - min_values[i]) /
                      (max_values[i] - min_values[i]) - 0.5)
```

If a subject's volume is constant (all-zero, all-mask, masking bug
upstream), `max_values[i] == min_values[i]` and the denominator is 0.
PyTorch's float division yields NaN; the NaN then poisons every
gradient downstream and the loss becomes `nan` on the first batch
that touches that subject. Diagnosing this from "loss is nan" is
miserable.

**After (`recvae/data.py:101-103`):**

```python
span = max_values - min_values
span = torch.where(span == 0, torch.ones_like(span), span)
```

Replace the zero span with 1 in exactly the degenerate slots. A
constant subject ends up at `-1` post-normalization (because `(x -
min) / 1 = 0`, then `2 * (0 - 0.5) = -1`), but no NaN, no gradient
poisoning, and the rest of the batch trains normally.

---

## Reading order

If you read these top to bottom you've traced the whole refactor:
how the package handles RNG state (1), device placement (2, 3),
model-state registration (4, 5), data validation (6), and numerical
robustness (7). Each piece is small. Together they make the
difference between "research notebook" and "library you can depend
on".
