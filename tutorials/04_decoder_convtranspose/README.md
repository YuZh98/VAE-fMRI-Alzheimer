# Lesson 04: Decoder — ConvTranspose3d

## What you'll learn

- The `ConvTranspose3d` output-shape formula and how it differs from `Conv3d`.
- Why `output_padding` is required when the encoder downsamples odd spatial
  dimensions (91, 109, 91) and you want the decoder to recover them exactly.
- Why a single layer needs *per-dimension* `output_padding` (the `(0, 1, 0)`
  and `(1, 0, 1)` gotchas) — not just a scalar.
- Why a wrong `output_padding` is a silent bug: it produces a tensor of the
  wrong shape that only blows up later, deep inside the loss computation.

## Where this lives in the repo

The decoder is built in `recvae/model.py:122-154`. Note the per-dimension
tuples on `decoder3` and `decoder4`:

```
ConvTranspose3d(..., output_padding=(0, 1, 0))   # decoder3
ConvTranspose3d(..., output_padding=(1, 0, 1))   # decoder4
```

The reason is documented in the module docstring at `recvae/model.py:13-20`.

## The concept

A `Conv3d(kernel_size=4, stride=2, padding=1)` halves each spatial dim and
rounds *down* on odd sizes. That means several different input sizes can map
to the same output size, and a transposed conv can't undo the operation
unambiguously without a hint. That hint is `output_padding`.

The output size of `ConvTranspose3d` along one dimension is:

```
out = (in - 1) * stride - 2 * padding + kernel_size + output_padding
```

For the values used here (`stride=2, padding=1, kernel_size=4`) this
simplifies to:

```
out = 2 * in + output_padding
```

so `output_padding` is just the off-by-one knob that lets a transposed conv
target either an even or an odd output along each dimension independently.

### The chain in this repo

The encoder takes `(91, 109, 91)` down to `(5, 6, 5)`:

```
X:  91 -> 45 -> 22 -> 11 -> 5
Y: 109 -> 54 -> 27 -> 13 -> 6
Z:  91 -> 45 -> 22 -> 11 -> 5
```

The decoder reverses this. Per layer, given `out = 2*in + output_padding`:

```
                X dim                Y dim                Z dim
decoder2:  5  -> 11  (op=1)    6  -> 13  (op=1)    5  -> 11  (op=1)   -> output_padding=1     (scalar OK)
decoder3: 11  -> 22  (op=0)   13  -> 27  (op=1)   11  -> 22  (op=0)   -> output_padding=(0,1,0)
decoder4: 22  -> 45  (op=1)   27  -> 54  (op=0)   22  -> 45  (op=1)   -> output_padding=(1,0,1)
decoder5: 45  -> 91  (op=1)   54  -> 109 (op=1)   45  -> 91  (op=1)   -> output_padding=1     (scalar OK)
```

The asymmetry comes from `Y=109` having a different parity history than
`X=Z=91`. If you copy a decoder built for cubic `(64, 64, 64)` inputs and
apply it to `(91, 109, 91)`, the shape errors will not appear until your
reconstruction loss tries to subtract two tensors of different shape.

### Why a wrong `output_padding` is a silent bug

PyTorch will *not* raise when you give a bad `output_padding`. It will
produce a tensor that is one voxel too small or too large along that
dimension. The error only surfaces later when, e.g.:

```
loss1 = (x - mu_x).pow(2).sum()
```

tries to subtract `(B, 1, 91, 109, 91)` from `(B, 1, 91, 108, 91)`. The
traceback points at the *loss* line, not the offending decoder layer. This
is exactly the kind of bug worth practicing on — once seen, it becomes
recognizable on sight.

## Code walk

The demo script `shape_walkthrough.py`:

1. Rebuilds the four-layer transposed-conv chain from scratch (no `recvae`
   imports needed for the construction).
2. Feeds a `(1, 32, 5, 6, 5)` tensor through it and prints the shape after
   each layer. You should see the chain reach `(1, 1, 91, 109, 91)`.
3. Deliberately substitutes the *wrong* `output_padding` on one layer and
   re-runs, printing the off-by-one resulting shape so you see the failure
   mode concretely. The downstream loss subtraction is also exercised so you
   see the eventual `RuntimeError`.

## Run it

```bash
python tutorials/04_decoder_convtranspose/shape_walkthrough.py
```

## Why this approach

The alternative is to use `Conv3d` everywhere and rely on bilinear
upsampling. That trades exact-shape control for a learned filter that has
to also do the upsample. `ConvTranspose3d` keeps the upsampling and the
learned filter in one op, at the cost of having to think about
`output_padding` once per layer per non-trivial spatial dim.

## Exercise (optional)

See `exercise.md`.

## Further reading

- PyTorch docs: `torch.nn.ConvTranspose3d` — the formula above is in the
  "Shape" section.
- Dumoulin & Visin, "A guide to convolution arithmetic for deep learning"
  (arXiv:1603.07285). The figures in section 4 are the clearest mental
  model of what transposed conv does to a single feature map.
- `recvae/model.py:122-154` — the actual decoder this lesson is teaching.
