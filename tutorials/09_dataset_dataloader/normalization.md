# Per-subject normalization in RecVAE

## What `normalize_per_subject` does

`recvae/data.py:74-105` rescales each subject's full 6D volume to the
range `[-1, 1]` using that subject's own min and max. In pseudocode:

```
for i in range(N):
    span_i = max(volumes[i]) - min(volumes[i])
    volumes[i] = 2 * ((volumes[i] - min_i) / span_i - 0.5)
```

The output is float in `[-1, 1]`, ready to feed the encoder (which ends
in `Tanh` and so expects roughly that range).

## What "per-subject" means and why it matters

There are several reasonable normalization scales for a multi-subject
fMRI dataset:

- **Per-voxel, across the dataset.** Standardize each voxel using the
  mean and std over all subjects and timepoints. Preserves *relative*
  intensity differences across subjects.
- **Per-subject, all voxels.** What this code does. Each subject is
  independently scaled to `[-1, 1]`. Destroys absolute-intensity
  differences across subjects.
- **Per-volume (per-timepoint).** Each timepoint of each subject is
  independently scaled. Destroys absolute differences within a subject
  too.

The current choice — per subject, across all voxels and timepoints — is
inherited from the canonical notebook. The note at
`recvae/data.py:79-85` flags it as a research decision:

> Per-subject statistics destroy absolute-intensity differences across
> subjects, which may matter for downstream classification. Considered
> a research decision and left unchanged here.

If you ever wire a classifier head onto these latents (e.g. healthy vs
disease), think hard about whether the per-subject scaling has thrown
away the signal you wanted to learn from. The encoder cannot recover an
intensity dimension that has been normalized out of the input.

## The zero-span guard

`recvae/data.py:100-102`:

```python
span = max_values - min_values
# Constant-volume safety: avoid 0/0. The original notebook would NaN here.
span = torch.where(span == 0, torch.ones_like(span), span)
```

If a subject's volume happens to be constant — every voxel at every
timepoint is the same value, which would happen with a zeroed-out
placeholder or a corrupted scan — then `span == 0` and the division
in the normalization step would propagate `NaN`. One `NaN` voxel will
contaminate every subsequent loss and gradient.

The fix replaces `0` with `1` in the divisor. The numerator
`volumes[i] - min_i` is also zero in that case (constant volume), so
the final result for that subject is `2 * (0/1 - 0.5) = -1`. Not a
useful sample, but not a `NaN` poison pill either.

This is the kind of guard you want as close to the data ingestion
boundary as possible. The earlier you catch bad inputs, the smaller
the blast radius.

## When to revisit this

A few situations call for revisiting the normalization:

- **Classification objective.** As above, per-subject scaling can
  remove the very signal you are trying to classify on.
- **Across-site comparison.** Different scanners produce different
  baseline intensities. Per-subject normalization absorbs that, which
  is sometimes what you want and sometimes not.
- **Pretrain/finetune mismatch.** If you pretrain with per-subject
  normalization and then finetune on differently-normalized data, the
  encoder's input distribution shifts under it.
