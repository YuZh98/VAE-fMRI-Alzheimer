# Lesson 03: Encoder with Conv3d

## What you'll learn

How a `Conv3d` layer arrives at its output spatial dimensions, why `(k=4,
s=2, p=1)` is the canonical "halving" block and what it does on an odd
input like 91, and what `BatchNorm3d` and `LeakyReLU(0.2)` add on top.
By the end you can reproduce `RecVAEModel.encoder1`-`encoder5` from
memory.

## Where this lives in the repo

- The full encoder stack: `recvae/model.py:50-75`.
- The encode method that runs it: `recvae/model.py:119-125`.
- The flatten + Linear that turns the 4D feature map into a 1D feature
  vector: `recvae/model.py:71-75`.

## The concept

**Conv3d output formula.** For each spatial axis,

```
out = floor((in + 2*p - k) / s) + 1
```

With `k=4, s=2, p=1` this simplifies to `floor((in - 2)/2) + 1`. Two cases:

```
even input  (e.g. 22): floor((22-2)/2)+1 = 10+1 = 11   (exact halving rounds down)
odd  input  (e.g. 91): floor((91-2)/2)+1 = 44+1 = 45   (also rounds down by floor)
```

For even inputs the halving is symmetric. For odd inputs the floor
introduces a fractional offset — that offset is what the decoder's
`output_padding` arguments compensate for (Lesson 04).

**BatchNorm3d.** Normalizes each channel across `(B, X, Y, Z)`: subtract a
per-channel running mean, divide by a running standard deviation, then apply
a learnable affine `gamma * z + beta`. In `train()` mode it uses batch
statistics and updates running stats; in `eval()` mode it uses the running
stats only. Two consequences worth knowing:

- With `BatchNorm3d` and N=1, the layer still normalizes over the
  spatial extents `(D, H, W)`, which gives plenty of samples per
  channel — the layer does useful work. The pathological case is
  `BatchNorm1d` on a `(1, C)` input, which has one sample per
  channel and produces undefined batch statistics. Tutorial demos
  use `.eval()` defensively where any ambiguity exists.
- BN's running stats live in `state_dict` as buffers, not parameters; they
  move with `.to(device)` but do not appear in `optimizer.step()`.

**LeakyReLU(0.2).** `f(x) = x if x > 0 else 0.2 * x`. The small positive
slope on the negative side keeps gradients flowing through neurons that
would otherwise saturate at zero. `0.2` is a common default; `recvae/`
uses it at every encoder and decoder activation (`recvae/model.py:54, 59,
64, 69, 87, 92, 97, 102`).

## Code walk

`build_encoder.py` builds two encoders.

The first is a from-scratch `nn.Sequential` mirroring `encoder1`-`encoder5`
of `RecVAEModel`. It is wrapped in a `with section(...)` block that pushes
`torch.randn(1, 1, 91, 109, 91)` through stage by stage and prints each
intermediate shape.

The second is the real `RecVAEModel.encode()` on the same input. We compare
**shapes only** — both encoders have random weights, so output values will
differ. The from-scratch path is just a literacy check that you can
reproduce the architecture without reading `model.py`.

## Run it

```bash
python tutorials/03_encoder_3dconv/build_encoder.py
```

Expect both encoders to produce a `(1, enc_out_dim=100)` final feature
vector and the assert to pass.

## Why this approach

The encoder stack is repetitive — five copy-pasted blocks with a stride-2
halving in each. Building one yourself once lets you read the package
version as "four halvings then a flatten" instead of two dozen lines of
PyTorch boilerplate.

## Exercise (optional)

See [`exercise.md`](exercise.md). Change `encoder3` from `stride=2` to
`stride=3` and predict the new spatial shape with the formula before
verifying.

## Further reading

- [`torch.nn.Conv3d` docs](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
- [`torch.nn.BatchNorm3d` docs](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html)
- [`torch.nn.LeakyReLU` docs](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
- A guide to convolution arithmetic: https://arxiv.org/abs/1603.07285
