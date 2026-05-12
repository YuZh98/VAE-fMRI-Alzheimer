# Lesson 07: Recurrent rollout

## What you'll learn

- How `RecVAEModel.forward` rolls a latent state `h_t` through `T`
  timesteps of fMRI volumes.
- The per-timestep flow:
  `g(h_{t-1}) -> encode(x_t) -> mu_h, log_var_h -> reparam -> h_t -> +z_subject -> decode -> mu_x_t`.
- Why we save `gh_history` separately from `h_history` (the loss uses
  both, and we'd otherwise have to recompute `g` later).
- Why slicing the time axis is `x[..., t]` for this codebase, not
  `x[:, t]`: the convention is `(B, C, X, Y, Z, T)` with `T` last.

## Where this lives in the repo

The rollout itself is at `recvae/model.py:219-266`. The per-timestep
step is factored out as `vae_step` at `recvae/model.py:199-217`. The two
loss terms that consume `mu_history` and (`h_history`, `gh_history`)
are at `recvae/losses.py:70-79`.

## The concept

### The per-step flow

For each timestep `t = 0, 1, ..., T-1`:

```
                                                              recurrent
                                                              part below
        x_t : (B, 1, 91, 109, 91)                             |
          |                                                   |
          v                                                   v
       encode  --enc_x : (B, enc_out_dim)--+              h_{t-1} : (B, D)
                                            |                 |
                                            v                 v
                                            +----- concat ----+
                                                  |
                                                  v
                                          combined : (B, enc_out_dim + D)
                                                  |
                              +-------------------+-------------------+
                              v                                       v
                       hidden2mu  -> mu_h : (B, D)          hidden2log_var -> log_var_h : (B, D)
                              |                                       |
                              +-----------------+---------------------+
                                                v
                                          reparametrize
                                                |
                                                v
                                            h_t : (B, D)
                                                |
                                                v
                       h_tilde = h_t + z_vectors[which_ones]   <-- per-subject bias
                                                |
                                                v
                                             decode
                                                |
                                                v
                                          mu_x_t : (B, 1, 91, 109, 91)
```

The "recurrent" part is just that `h_{t-1}` is concatenated to the
encoded `x_t` before producing the next `h_t`. The encoder itself is
not recurrent — it sees one volume at a time. The recurrence lives in
the two `Linear` heads `hidden2mu` and `hidden2log_var`.

### `gh_history` separate from `h_history`

Inside the loop, before stepping, we apply the temporal prior:

```python
gh_history.append(self.g_transform(h))    # g(h_{t-1})
mu, h = self.vae_step(x_list[t], h, ...)  # produces h_t
h_history.append(h)                       # h_t
```

The training loss has a term `||h_t - g(h_{t-1})||^2` (`recvae/losses.py:73-74`).
If we stored only `h_history`, we'd have to redo all the `g_transform`
calls later. Storing both is cheap (latent dim is 10) and keeps the loss
expression a single zip.

### Why `x[..., t]` and not `x[:, t]`

Shape convention in this repo is `(B, C, X, Y, Z, T)`. The leading
dimensions are batch and channel; the spatial dimensions come next; time
is *last*. So `x[..., t]` slices the time axis. Writing `x[:, t]` would
slice the channel axis (there's only one channel) and produce
`(B, X, Y, Z, T)` — wrong shape and silently wrong meaning. See lesson
02 for the rationale on shape conventions.

The list comprehension `x_list = [x[..., t] for t in range(self.tol_time)]`
front-loads the slicing for clarity; we could equivalently slice inside
the loop.

## Code walk

`rollout_demo.py`:

1. Builds `RecVAEModel(train_size=2, cfg=Config(tol_time=4))`. Four
   timesteps is enough to see the temporal pattern and stays fast on
   CPU.
2. Constructs a synthetic `x` of shape `(2, 1, 91, 109, 91, 4)` and a
   zero initial state `h_0 = (2, 10)`. `which_ones = [0, 1]` selects
   each of the two subjects' `z_vectors` rows.
3. Calls `model(x, h_0, which_ones)`, prints lengths and per-element
   shapes of the four returned lists.
4. Stacks `h_history` into a `(B, T, D)` tensor and prints per-timestep
   mean and standard deviation across the batch — so you can see the
   latent state actually evolves with `t` and isn't constant.

## Run it

```bash
python tutorials/07_recurrent_rollout/rollout_demo.py
```

Expected wall time: a few hundred milliseconds on a 2-thread CPU.

## Why this approach

The alternative is a true RNN (e.g. `nn.GRU`) wrapping the encoded
features. That would be more standard but couples the recurrence to the
encoder and prevents us from doing a closed-form ridge update on the
transition matrix `F` (see lesson 06 on `F_mat` and lesson 08 on
alternating optimization). The hand-rolled loop is verbose but lets the
training loop alternate "SGD on the encoder/decoder/inference heads" and
"closed-form ridge on `F`" cleanly.

## Further reading

- `recvae/model.py:199-217` — `vae_step`, the per-step body.
- `recvae/model.py:219-266` — the rollout itself.
- `recvae/losses.py:70-79` — how the loss consumes the four output
  lists.
- Lesson 08 on losses and alternating optimization picks up where this
  lesson ends.
