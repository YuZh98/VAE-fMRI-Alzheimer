# Lesson 02: Tensor shapes

## What you'll learn

The `(B, C, X=91, Y=109, Z=91, T=120)` shape convention used everywhere
in `recvae/`. Why channel is second, why time is last, and what the
spatial dim chain `91 -> 45 -> 22 -> 11 -> 5` looks like coming down the
encoder and back up the decoder.

## Where this lives in the repo

- The convention block is documented at the top of the model module:
  `recvae/model.py:1-21`.
- The encoder slices time off the input with `x[..., t]` at
  `recvae/model.py:196` — only possible because T is the last axis.
- Encoder stages: `recvae/model.py:50-75`.
- Decoder stages: `recvae/model.py:82-107`.

## The concept

PyTorch's 3D convolution expects `(B, C, D, H, W)` — batch, channel, then
three spatial dims. RecVAE extends this to a temporal series:

```
(B, C, X, Y, Z, T)
 |   |  |  |  |  |
 |   |  |  |  |  +-- time (T=120)
 |   |  +--+--+----- spatial volume (91 x 109 x 91)
 |   +-------------- channel (1: fMRI is single-channel)
 +------------------ batch
```

**Why channel-first.** `torch.nn` convolution and batchnorm modules are all
channel-first. Doing it any other way would mean an explicit transpose at
every conv layer.

**Why T-last.** `forward()` rolls over time one step at a time. Slicing the
trailing axis with `x[..., t]` returns a contiguous `(B, C, X, Y, Z)`
view — exactly the shape `encode()` wants. If T were anywhere else the
slice would either be non-contiguous or need an explicit `.permute()`.

**The spatial chain.** Every encoder stage uses `Conv3d(k=4, s=2, p=1)`.
The output formula is

```
out = floor((in + 2*p - k) / s) + 1
    = floor((in + 2 - 4) / 2) + 1
    = floor((in - 2) / 2) + 1
```

Applied to 91:

```
91 -> floor((91-2)/2)+1 = floor(89/2)+1 = 44+1 = 45
45 -> floor((45-2)/2)+1 = floor(43/2)+1 = 21+1 = 22
22 -> floor((22-2)/2)+1 = 10+1 = 11
11 -> floor((11-2)/2)+1 = 4+1  = 5
```

So the X axis walks `91 -> 45 -> 22 -> 11 -> 5`. The same chain on Y goes
`109 -> 54 -> 27 -> 13 -> 6` and Z is identical to X. The decoder
(`ConvTranspose3d` with the same k/s/p) walks it in reverse — Lesson 03 and
Lesson 04 cover the details, including the `output_padding` values needed
for the odd-dim cases.

## Code walk

`inspect_shapes.py` does two things:

1. Builds a `nn.Sequential` of four raw `nn.Conv3d` layers with the same
   stride/padding pattern as the real encoder. Pushes a
   `(1, 1, 91, 109, 91)` tensor through and asserts the spatial shape after
   each stage matches the documented chain.
2. Builds the matching `nn.ConvTranspose3d` chain with the correct
   `output_padding` values from `recvae/model.py:82-107` and confirms the
   inverse chain lands back on `(91, 109, 91)`.

## Run it

```bash
python tutorials/02_tensor_shapes/inspect_shapes.py
```

Each stage prints its output shape. All asserts must pass.

## Why this approach

Convolutional networks fail silently if a shape is one off. The package
has hard-coded `Linear(32 * 5 * 6 * 5, ...)` at the encoder-to-MLP boundary
(`recvae/model.py:73`), so a shape drift anywhere upstream surfaces as a
matmul error, not a meaningful one. The clearest defense is a written
shape chain that the code reproduces exactly — which is what this lesson
is.

## Exercise (optional)

Change the input to `(1, 1, 95, 109, 91)` and predict, then verify, the new
chain on the X axis. Hint: `(95-2)/2 = 46.5 -> floor -> 46`, then `+1 -> 47`.

## Further reading

- [`torch.nn.Conv3d` docs](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
- [`torch.nn.ConvTranspose3d` docs](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)
- `recvae/model.py:1-21` — the convention block this lesson mirrors.
