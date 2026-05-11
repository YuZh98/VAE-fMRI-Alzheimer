# Exercise: stride-3 in encoder3

In `build_encoder.py`, the third stage uses `nn.Conv3d(8, 16, kernel_size=4,
stride=2, padding=1)`. Change it to `stride=3` and predict the new spatial
shape **with pen and paper first**, then re-run the script and verify.

## Predict

The input to `encoder3` is `(1, 8, 22, 27, 22)` (the output of `encoder2`).
Use

```
out = floor((in + 2*p - k) / s) + 1
```

with `p=1, k=4, s=3`:

```
X: floor((22 + 2 - 4) / 3) + 1 = floor(20/3) + 1 = 6 + 1 = 7
Y: floor((27 + 2 - 4) / 3) + 1 = floor(25/3) + 1 = 8 + 1 = 9
Z: floor((22 + 2 - 4) / 3) + 1 = 7
```

So you predict `(1, 16, 7, 9, 7)`.

## Verify

Edit `build_encoder.py`. In `build_from_scratch`, change the third
`nn.Sequential` to use `stride=3` in the `Conv3d`. Re-run:

```bash
python tutorials/03_encoder_3dconv/build_encoder.py
```

You should see `after encoder3: shape=(1, 16, 7, 9, 7)`. The script will
then fail downstream — the next stage's `Conv3d` and the final `Linear`
were both sized for the original chain. **That is the lesson:** convs
later in the network compound any upstream shape change, and `Linear`
layers hard-code the flat dimension.

## Restore

Revert the change. (Or `git checkout build_encoder.py`.)

## Extension

What stride keeps the output spatial size **the same** as the input?
Hint: solve `floor((in + 2 - 4) / s) + 1 = in` for small `in` and see
which `s` works in general. (Answer: `s=1`, plus an appropriate padding —
this is the "same" padding regime, useful when you do not want spatial
downsampling.)
