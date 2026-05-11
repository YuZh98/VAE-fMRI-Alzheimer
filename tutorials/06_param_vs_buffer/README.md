# Lesson 06: Parameter vs buffer

## What you'll learn

- The three ways to attach a tensor to an `nn.Module` and what each does:
  `nn.Parameter`, `register_buffer`, and a plain `self.foo = tensor`.
- Which ones get gradients, appear in `.parameters()`, appear in
  `.state_dict()`, and move with `.to(device)`.
- The case study in `recvae/model.py`: `z_vectors` is a *Parameter*
  (trained with SGD), but `F_mat` is a *buffer* (updated with a closed-
  form ridge solve and so has no gradient).

## Where this lives in the repo

`recvae/model.py:109-117`:

```python
# z_vectors: trained by SGD -> Parameter
z_init = torch.randn(train_size, cfg.z_dim) * cfg.sig_z
self.z_vectors = nn.Parameter(z_init)

# F_mat: updated in closed form -> buffer
self.register_buffer("F_mat", torch.rand(cfg.latent_dim, cfg.latent_dim))
```

The closed-form update is at `recvae/model.py:208-254`. Note that it
runs under `@torch.no_grad()` and writes back with `self.F_mat.copy_(...)`
— never via assignment.

## The concept

Comparison table — three ways to put a tensor on an `nn.Module`:

```
                          nn.Parameter   register_buffer   plain attribute
gets a gradient                YES             NO                NO
appears in .parameters()       YES             NO                NO
appears in .named_buffers()    NO              YES               NO
appears in .state_dict()       YES             YES               NO
moves with .to(device)         YES             YES               NO
saved by torch.save(model.sd)  YES             YES               NO
```

The four properties on the right line are the punchline:

- A **`nn.Parameter`** is for tensors you train with autograd. The model
  optimizer sees them via `model.parameters()`.
- A **buffer** is for tensors that are part of the model state — they
  need to be saved with the model and moved with the model — but are
  *not* updated by gradient descent. Examples: BatchNorm's running mean
  and variance, an attention causal mask, a precomputed lookup table,
  or, in this repo, a transition matrix solved in closed form.
- A **plain attribute** (`self.foo = some_tensor`) is for a temporary
  cache that you don't want persisted and don't want moved. It will
  silently *not* travel with the rest of the module to GPU, which is
  almost always a bug if the value matters.

### Case study: `z_vectors` vs `F_mat`

Both are 2D learned tensors. So why does one get to be a Parameter and
the other doesn't?

- `z_vectors` enters the loss through `mu_x = decode(h + z_vectors[i])`.
  A gradient of the reconstruction loss with respect to `z_vectors[i]`
  is well-defined and SGD-trainable.
- `F_mat` enters the loss through `||h_t - g(h_{t-1})||^2`, with
  `g(h) = h @ F^T`. We *could* train it with SGD, but there's a much
  better option: the loss in `F` is quadratic, so the optimum has a
  closed-form ridge-regression solution. We solve for it directly
  (`recvae/model.py:208-254`) instead of relying on SGD to crawl toward
  it. Closed-form means no autograd needed, so a buffer is more
  honest than a Parameter — a Parameter would also work but would
  mislead readers into thinking it's gradient-trained.

A second reason to prefer a buffer here: writing `self.F_mat = ...` (plain
assignment) on a Parameter is undefined-behavior-adjacent (it replaces the
Parameter object, breaks `model.parameters()` membership, and unhooks
optimizers). Buffers are designed for in-place updates via `.copy_`.

## Code walk

`serialize_demo.py`:

1. Defines a tiny `nn.Module` with three tensor attributes:
   `self.weight` (Parameter), `self.running_mean` (buffer), and
   `self.cache` (plain attribute).
2. Prints membership in `.parameters()`, `.named_buffers()`, and
   `.state_dict()` for all three.
3. Saves the state_dict to a `tempfile.NamedTemporaryFile`, instantiates
   a *fresh* module, and loads the saved dict back.
4. Confirms `weight` and `running_mean` round-trip with the saved values
   but `cache` does not — the fresh module's `cache` retains the
   initial value because it wasn't persisted.
5. Cleans up the temp file.

## Run it

```bash
python tutorials/06_param_vs_buffer/serialize_demo.py
```

## Why this approach

You will occasionally see codebases stash everything as a Parameter and
just freeze it with `requires_grad=False` when they don't want gradients.
That works but loses information: a reader has to mentally re-derive
which tensors are *meant* to be trained. Using buffer vs Parameter as a
*declaration of intent* makes the model self-documenting.

## Further reading

- PyTorch docs: `torch.nn.Module.register_buffer` and `torch.nn.Parameter`.
- `recvae/model.py:109-117` — the two-line declaration this lesson is
  about.
- `recvae/model.py:208-254` — `updating_F`, the closed-form update that
  motivates choosing a buffer over a Parameter.
