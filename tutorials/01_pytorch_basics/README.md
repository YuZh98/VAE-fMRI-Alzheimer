# Lesson 01: PyTorch basics

## What you'll learn

A one-page refresher on the three PyTorch primitives every `recvae/` file
leans on: tensors with `dtype` and `device`, autograd via `.backward()` and
`.grad`, and `nn.Module` subclassing. The lesson assumes you have seen these
before; the goal is to point at where each shows up in the package.

## Where this lives in the repo

- Tensors with explicit `device` and `dtype`: `recvae/model.py:191-193`
  (`torch.randn(mu_h.shape, device=mu_h.device, dtype=mu_h.dtype)`).
- Autograd off-switch: `recvae/model.py:269` (`@torch.no_grad()` on
  `updating_F`), and `recvae/train.py:162` (`@torch.no_grad()` on `evaluate`).
- `nn.Module` subclass: `recvae/model.py:72` (`class RecVAEModel(nn.Module)`).
- A `nn.Sequential` block: `recvae/model.py:90-94` — Conv3d + BatchNorm3d
  + LeakyReLU, the canonical encoder unit.

## The concept

**Tensor.** A multi-dimensional array with attached metadata: `dtype` (e.g.
`torch.float32`), `device` (`cpu`, `cuda`, `mps`), and a `requires_grad`
flag. Operations on tensors compose into a graph for autograd.

**Autograd.** When `requires_grad=True` flows through an op, PyTorch records
that op. Calling `.backward()` on a scalar walks the recorded graph and
accumulates gradients into the `.grad` attribute of leaf tensors. Wrap a
block in `torch.no_grad()` (or decorate with `@torch.no_grad()`) to skip
recording — used in `recvae` when updating `F_mat` in closed form and when
evaluating.

**`nn.Module`.** A container that registers any `nn.Parameter` or
`nn.Module` attribute as a tracked sub-component. `model.parameters()` walks
the tree; `model.to(device)` moves every tracked tensor; `state_dict()`
serializes them. Override `forward(self, ...)` to define the computation.

## Code walk

`01_tensors_and_grad.py` builds two tensors, multiplies them, calls
`.backward()`, and inspects `.grad`. It also flips through `dtype`, `device`,
`requires_grad`, and shows that `torch.no_grad()` suppresses graph
recording (the result has no `grad_fn`).

`02_first_module.py` defines a toy `nn.Module` whose `forward()` mirrors
the `encoder1` block from `recvae/model.py:90-94`: a Conv3d, then BatchNorm3d,
then LeakyReLU. It runs a `(1, 1, 16, 16, 16)` cube through and prints the
input and output shapes. The cube is small on purpose — concept demos do
not need the full 91 x 109 x 91 volume.

## Run it

```bash
python tutorials/01_pytorch_basics/01_tensors_and_grad.py
python tutorials/01_pytorch_basics/02_first_module.py
```

Both should exit 0 in under a second.

## Why this approach

The `recvae/` code uses these three primitives constantly. Anchoring the
abstract idea to a concrete code citation makes the rest of the tutorials
self-locating: when you see `nn.Module` in Lesson 06, you already know
where to look in `model.py`.

## Further reading

- [PyTorch autograd tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [`torch.nn` API reference](https://pytorch.org/docs/stable/nn.html)
- [`torch.no_grad` docs](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
