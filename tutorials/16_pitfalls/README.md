# Lesson 16: Pitfalls

## What you'll learn

Five concrete bugs that show up in PyTorch fMRI / 3D-CNN code, with a
small demonstration script for each. Each pitfall is a real-world
recurrence: every one of these is something the legacy V1-V4 notebooks
did (or could trivially do) that the `recvae/` refactor fixed.

For each, you'll see:

- A two-line problem statement,
- A "how to recognize it" hint so you can spot the pattern in code
  review,
- A pointer to a small script in `examples/` you can run yourself.

## Where this lives in the repo

The five pitfalls correspond to five fixes in the refactor — see
`tutorials/14_refactor_notebook_to_pkg/before_after.md` for the
side-by-side before/after of each:

- `recvae/model.py:135-146` — `reparametrize` uses explicit `device=` /
  `dtype=` (pitfall 1).
- `recvae/data.py:142-148` — DataLoader generator is CPU-side (pitfall 2).
- `recvae/model.py:13-20`, `:83-107` — decoder `output_padding` chain
  (pitfall 3).
- `recvae/data.py:101-103` — zero-span guard in `normalize_per_subject`
  (pitfall 4).
- `recvae/model.py:109-116` — `nn.Parameter` for `z_vectors`,
  `register_buffer` for `F_mat` (pitfall 5).

## The concept

Five idioms that look fine at first glance but quietly produce wrong
behavior. The common thread: PyTorch will *not* warn you about any of
them. They surface as a crash on a different machine, NaN gradients,
shape errors in the loss line, or model weights that vanish on
checkpoint reload.

### 1. Global default tensor type

**Problem.** `torch.set_default_tensor_type('torch.cuda.FloatTensor')`
is a process-wide global. Once set, every new tensor (`torch.zeros`,
`torch.randn`, etc.) implicitly lands on CUDA. Code that worked on the
author's GPU box crashes on import on a CPU-only laptop, because the
`torch.cuda.FloatTensor` type does not exist there. It also has no MPS
equivalent.

**How to recognize it.** Look near the top of an old notebook for
`set_default_tensor_type`. Or look for `torch.randn(size, ...)` calls
that never pass `device=` — they are relying on this global.

**See.** `examples/global_default_tensor.py`.

### 2. DataLoader generator with `device='cuda'`

**Problem.** `torch.Generator(device='cuda').manual_seed(seed)` and
then `DataLoader(..., generator=g)` was the legacy idiom for "seed the
data shuffle". It crashes immediately on a CPU-only machine and is
unnecessary — DataLoader's shuffle indices are Python ints on the host,
they never touch the GPU.

**How to recognize it.** Search for `torch.Generator(device=` in any
`DataLoader` construction. CPU generator is the right answer.

**See.** `examples/dataloader_cuda_generator.py`.

### 3. Silent shape mismatch from wrong `ConvTranspose3d` `output_padding`

**Problem.** Transposed convolutions with the wrong `output_padding`
produce a tensor one voxel off along the affected axis. PyTorch does
not error; the off-by-one propagates downstream and only surfaces when
the reconstruction loss tries to subtract two tensors of different
shape. The traceback points at the loss, not the offending decoder
layer.

**How to recognize it.** When the encoder/decoder spatial dims do not
match. Always print the output shape after each decoder layer when
writing new ones; the right `output_padding` is determined by the
parity of the *target* size at each layer.

**See.** `examples/silent_shape_mismatch.py`.

### 4. Zero-span min-max normalization producing NaN

**Problem.** `(x - x.min()) / (x.max() - x.min())` produces NaN when
the input is constant (e.g. an all-zero subject from a masking bug
upstream). The NaN poisons every downstream gradient and the loss
becomes `nan` on the first batch that touches that subject.

**How to recognize it.** Any min-max normalization that does not guard
against zero span. The first sign is "loss is nan starting at batch 7"
in training logs — by which point the bad subject is hard to track
down.

**See.** `examples/zero_span_normalize.py`.

### 5. Assigning `torch.Tensor` as `self.foo` (instead of `nn.Parameter`
   or `register_buffer`)

**Problem.** A plain tensor attribute on an `nn.Module` is invisible to
the optimizer, missing from `state_dict()`, and not moved by
`.to(device)`. If you train and `torch.save(model.state_dict(), ...)`,
the saved checkpoint silently drops that attribute. Loading it later
gives you a model with that field re-initialized to whatever the
constructor sets — typically `torch.rand` or `torch.zeros`.

**How to recognize it.** Look for `self.something = torch.randn(...)`
in any `__init__`. If "something" needs to be learned by the optimizer
it should be `nn.Parameter(...)`; if it needs to *travel* with the
model (move with `.to(device)`, persist in `state_dict()`) but should
not get gradients, it should be `self.register_buffer("something",
...)`.

**See.** `examples/tensor_attribute_not_param.py`.

## Code walk

Each `examples/*.py` script is self-contained: imports
`tutorials._tutorial_utils.section`, demonstrates the safe pattern,
prints observable output, exits 0.

Recommended reading order:

1. `tensor_attribute_not_param.py` — fastest to understand, sets up
   vocabulary (`parameters`, `buffers`, `state_dict`).
2. `global_default_tensor.py` — explains why explicit `device=` won
   everywhere in `recvae/`.
3. `dataloader_cuda_generator.py` — a tiny gotcha with a tiny fix.
4. `zero_span_normalize.py` — a numerical bug masquerading as a NaN
   loss.
5. `silent_shape_mismatch.py` — biggest aha-moment of the lesson.

## Run it

```bash
for f in tutorials/16_pitfalls/examples/*.py; do
  echo "==> $f"
  python "$f" || exit 1
done
```

CI does NOT run these (they are one directory deeper than the
`tutorials/*/*.py` glob CI uses). They are meant to be run by you,
individually, while reading the corresponding pitfall.

## Why this approach

We could have written this as a long "things to avoid" prose document.
We chose a script per pitfall instead because each one can be made
*visible* in code: print the parameter count with and without
`nn.Parameter`, show the NaN vs the safe value, show the shape after
the bad `output_padding`. Once seen, these bugs become recognizable on
sight in a code review — which is the point of the lesson.

## Further reading

- `tutorials/14_refactor_notebook_to_pkg/before_after.md` — the
  before/after for each of these pitfalls in the `recvae/` codebase.
- PyTorch docs:
  - `torch.set_default_tensor_type` — deprecated since 2.0; this is
    why.
  - `torch.nn.Module.register_buffer` — the canonical way to attach
    non-learnable state.
  - `torch.nn.ConvTranspose3d` shape formula — for pitfall 3.
- Lesson 06 (`tutorials/06_param_vs_buffer/`) — the conceptual
  background for pitfall 5.
- Lesson 04 (`tutorials/04_decoder_convtranspose/`) — the conceptual
  background for pitfall 3.
