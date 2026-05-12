# Lesson 12: Save and load

## What you'll learn

How to checkpoint a PyTorch model the right way: serialize
`state_dict()`, not the whole module; use `map_location` to load onto a
different device than you trained on; and understand exactly what is
inside a `state_dict` for `RecVAEModel` (Parameters *and* Buffers —
including the closed-form-updated `F_mat`).

## Where this lives in the repo

- `recvae/model.py:156-163` — the registration of `z_vectors` (Parameter)
  and `F_mat` (Buffer). Both show up in `state_dict()`.
- Root `README.md` "Save / load" section — the one-liner recipe for
  saving and restoring a trained model.

## The concept

### What `state_dict()` is

`state_dict()` is an ordered mapping from string keys to tensors. It
contains **everything** the model needs to reproduce its state:

- Every `nn.Parameter` registered on the module or its submodules.
- Every buffer registered via `register_buffer`.

For `RecVAEModel` that means: all encoder/decoder/inference-head weights
and biases, every `BatchNorm3d` running mean and running variance
(buffers), the `z_vectors` Parameter (subject-specific noise vectors),
and the `F_mat` Buffer (the linear temporal-prior transition matrix).
See `recvae/model.py:156-163`:

```python
z_init = torch.randn(train_size, cfg.latent_dim) * cfg.sig_z
self.z_vectors = nn.Parameter(z_init)
self.register_buffer("F_mat", torch.rand(cfg.latent_dim, cfg.latent_dim))
```

`F_mat` is a Buffer because it is updated by closed-form ridge regression
(see Lesson 06 and `recvae/model.py:269-331`), not by gradient descent.
But it is still part of the model's state, so it still needs to be
checkpointed — that is exactly why `register_buffer` is the right tool:
buffers are serialized in `state_dict()` and moved by `.to(device)` just
like parameters, the only difference is that they do not appear in
`.parameters()` and the optimizer ignores them.

### Why save `state_dict()` and not the whole `nn.Module`

`torch.save(model, ...)` (pickling the module directly) appears
convenient but has two real problems:

1. **Brittle to refactoring.** The pickle file records the fully
   qualified class path (`recvae.model.RecVAEModel`). Rename the file,
   move the class, restructure the package, and the old checkpoint
   cannot be loaded without monkey-patching `sys.modules`.
2. **Security.** `torch.load` calls `pickle.load`, which can execute
   arbitrary code during deserialization. Loading a `state_dict` is
   safer because the recipient code builds the module instance first,
   then the load is a structured copy of tensors into known slots.

Saving `state_dict()` decouples the artifact from the code. The
checkpoint is just a dict of tensors; you reconstruct the module from
your current source tree and load the tensors into it.

### `map_location`

`torch.load(path, map_location='cpu')` forces all tensors to load on
CPU regardless of where they were saved. This is the standard idiom for
loading a CUDA-trained checkpoint on a CPU box. You can pass a device
object (`torch.device('cuda:0')`) or a function, but `'cpu'` is the
right default for portability.

Without `map_location`, `torch.load` tries to restore the original
device, which fails (or burns GPU memory you did not want) when the
saving and loading environments differ.

### Strict vs non-strict load

`load_state_dict(state_dict, strict=True)` (the default) errors if the
keys do not match exactly. `strict=False` reports missing and unexpected
keys but does not error. Use it when:

- You are fine-tuning: the head is new, so you expect to miss the old
  head's keys and have unexpected keys in the new head.
- You added a layer to the model and want to load the old weights into
  the unchanged layers and leave the new one freshly initialized.

For a vanilla "save model, load same model later" workflow, leave it
strict — `strict=False` silently masks shape and naming bugs you
otherwise want to know about.

## Code walk

`checkpoint_demo.py` does the full round-trip in under a minute on CPU:

1. Build a `RecVAEModel(train_size=2, cfg=Config(tol_time=4))` — small
   enough to run fast.
2. Run one forward pass on `torch.randn(2, 1, 91, 109, 91, 4)` to make
   sure the model is actually exercised before we serialize.
3. `torch.save(model.state_dict(), tmpfile)`.
4. Build a *second* model instance with the same config.
5. `torch.load(tmpfile, map_location='cpu')`, then
   `model2.load_state_dict(state_dict)`.
6. Put both models in `.eval()` so `BatchNorm3d` uses its running stats
   (which were copied through the `state_dict`) instead of recomputing
   batch statistics on the forward pass — that is what makes the two
   forward passes deterministically identical.
7. Forward the *same* input through both. Assert element-wise equality.
8. Print the names and sizes of state-dict keys so you can see exactly
   what got saved (parameters from `nn.Linear`/`nn.Conv3d`/`BatchNorm3d`,
   `BatchNorm3d` running stats, `z_vectors`, `F_mat`).
9. Delete the temp file.

The crucial step is #6. In training mode, `BatchNorm3d` recomputes
batch statistics from the input — even with identical weights, two
re-runs on the same input will agree, but only because the input
matches. The point of `.eval()` is to use the *stored* running stats
(also part of the state dict), so we are testing the full state
round-trip, not just the parameter copy.

## Run it

```bash
python tutorials/12_save_load/checkpoint_demo.py
```

Expected output:

- One line listing each state-dict key and its tensor shape (you should
  see `z_vectors` of shape `(2, 10)` and `F_mat` of shape `(10, 10)`
  among the encoder/decoder weights).
- Confirmation that the two models produce element-wise identical
  reconstruction tensors.
- Exit code 0.

## Why this approach

Checkpointing `state_dict()` is the single most important habit when you
move a project from "works on my notebook" to "survives a refactor."
Every paper-shaped artifact — a fine-tuned model, an ablation point on
a curve, a reviewer's reproducibility request — is a `state_dict` file
plus the source commit that produced it. Pickle-the-module
checkpoints rot the moment you rename anything.

## Exercise (optional)

Modify `checkpoint_demo.py` to call `model2.load_state_dict(state_dict,
strict=False)` after deleting one key from the state dict (e.g.
`del state_dict['F_mat']`). Print the `missing_keys` and
`unexpected_keys` returned by `load_state_dict` and verify that the two
models' outputs are now different.

## Further reading

- [Saving and Loading Models — PyTorch tutorials](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [`torch.nn.Module.state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict)
- [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html) — note the `weights_only` argument for safer loading of untrusted files.
