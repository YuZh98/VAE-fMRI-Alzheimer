# Exercise

The current decoder ends at `X = 91`. Suppose you wanted to upsample once
more along X, going from `91 -> 183`.

1. Pick `kernel_size`, `stride`, `padding` consistent with the rest of the
   decoder (`k=4, s=2, p=1`). Using the formula

   ```
   out = (in - 1) * stride - 2 * padding + kernel_size + output_padding
   ```

   what value of `output_padding` along X do you need to land exactly on
   `183`?

2. If you simultaneously wanted Y to go `109 -> 219` and Z to go
   `91 -> 183` with the same layer, what `output_padding` tuple do you
   pass?

3. Add the layer at the end of `build_decoder` in `shape_walkthrough.py`
   (an extra `ConvTranspose3d(1, 1, ...)`) and confirm the printed final
   shape matches your prediction. Note: you will also have to change the
   target shape in the `section("correct ...")` block, since the chain
   now ends at the new size.

### Hints (don't read until you've tried)

- For `91 -> 183`: `out = 2*91 + op = 182 + op`. So `op = 1`.
- For `109 -> 219`: `out = 2*109 + op = 218 + op`. So `op = 1`.
- All three dims happen to need `op = 1` here, so you can pass scalar
  `output_padding=1`. That is *coincidence* — try `91 -> 182` and you'll
  need `op=0` along that dim, forcing a per-dim tuple again.
