# Lesson 15: End-to-end tiny training run

## What you'll learn

- How the eight pieces from Lessons 00-12 (`Config`, `set_seed`, device
  picker, `FMRIDataset`, `build_dataloader`, `DeviceDataLoader`,
  `RecVAEModel`, `fit`) snap together into a complete training run.
- The shape of `fit`'s alternating loop: per-batch SGD on encoder /
  decoder / inference / `z_vectors`, then one closed-form ridge update
  to `F_mat` at the end of each epoch.
- That an end-to-end training run can be exercised on synthetic data,
  on CPU, in under 90 seconds — useful both as a smoke test in CI and
  as a teaching artifact.
- How to save and reload model state correctly: `state_dict()`, a
  `tempfile` location, `load_state_dict`, and a forward-pass equality
  check to confirm round-trip.

## Where this lives in the repo

- `recvae/train.py:19-159` — the `fit` function this demo calls. The
  alternation is on lines 105-130 (batch loop then `updating_F`).
- `recvae/model.py:334-386` — `training_step`, the per-batch loss
  composition.
- `recvae/model.py:269-331` — `updating_F`, the closed-form ridge
  solve called once per epoch by `fit`.
- `recvae/config.py` — the `Config` dataclass.
- `recvae/utils.py` — `set_seed`, `get_default_device`, the
  `DeviceDataLoader` wrapper.
- `recvae/data.py:74-105` — `normalize_per_subject` (per-subject min-max
  to `[-1, 1]`).
- The README's "Train on real data" snippet — same pattern as the demo
  but with real NIfTI files instead of synthetic tensors.

## The concept

`recvae/train.py:fit` is the orchestrator. It does not contain any
math beyond a `for epoch in range(epochs)` outer loop. The math lives
in `RecVAEModel.training_step` (one batch's worth of losses) and
`RecVAEModel.updating_F` (the closed-form solve). `fit` glues them
together with the optimizer and the per-epoch buffer of posterior
trajectories.

The alternation looks like this:

```
for epoch in range(epochs):
    for batch, batch_index in loader:
        loss, loss_dic, h_history = model.training_step(batch, h0, ...)
        h_history_history[which_ones] = stack(h_history).detach()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    model.updating_F(h_history_history, h0, rho)   # closed-form ridge
```

Two updates, two timescales: SGD updates every batch, `F` updates
every epoch. The `h_history_history` buffer is `(N_train, T, D)` and
accumulates the latent trajectory of *every* subject in the training
set during the SGD pass, so the ridge solve has the complete `(X, Y)`
pair it needs.

### Why a small synthetic run is enough to demonstrate this

The whole point of the alternating-optimization architecture is
*structural*: SGD on the nonlinear nets, closed-form on the
linear-Gaussian `F`. That structure is visible after one epoch — you
do not need to train to convergence to see it. With

- `N = 4` synthetic subjects,
- `tol_time = 4` timesteps (instead of 120),
- `batch_size = 2`,
- `epochs = 3`,
- `learning_rate = 1e-5` (ten times the notebook default `1e-6` so the
  loss moves visibly in three epochs; `1e-4` diverges to NaN on this
  random-tensor input by epoch 3, so we backed off),

`fit` returns a non-empty `train_loss_history`, the buffer
`h_history_history` fills with real per-subject trajectories,
`F_mat` is overwritten three times by `updating_F`, and the whole run
takes well under 90s on CPU.

The bigger spatial shape `(1, 91, 109, 91)` is preserved because
the encoder/decoder layer math depends on it (see
[Lesson 04](../04_decoder_convtranspose/README.md) on
`output_padding`). The smaller knobs are `N`, `T`, `batch_size`.

### Why a higher learning rate for the demo

`Config()`'s default `learning_rate = 1e-6` is calibrated for a 500-epoch
training run on the canonical dataset. With 3 epochs on synthetic
random tensors at `lr=1e-6`, you cannot see any change in the loss —
the gradient updates are below the precision of the reconstruction
noise. The demo uses `lr=1e-5` purely so the printed loss history
shows *some* movement. (We initially tried `1e-4` per the lesson plan,
but it diverged to NaN on this random input by epoch 3 — random
tensors have no real signal to descend toward, so the loss surface is
ill-conditioned.) Do not read pedagogical significance into the exact
numbers; they are random tensors.

### Save / load round-trip

The final block of the demo:

1. `torch.save(model.state_dict(), tmp_path)`.
2. Build a fresh `RecVAEModel` with the same `Config` and `train_size`.
3. `model2.load_state_dict(torch.load(tmp_path, map_location=device))`.
4. Confirm `model(...)` and `model2(...)` produce the same first
   reconstruction (under eval mode, with seeded reparameterization).

This exercises *all* the trainable and buffered state: encoder
weights, decoder weights, the inference linears, `z_vectors`
(Parameter), and `F_mat` (buffer). The refactor's promotion of
`z_vectors` to `Parameter` and `F_mat` to a registered buffer is what
makes this round-trip work — in the legacy notebooks both were plain
tensor attributes and would be silently dropped.

## Code walk

`train_tiny.py` in this directory:

1. `set_seed(2022)` and `get_default_device()` —
   [Lesson 11](../11_reproducibility/README.md) patterns.
2. `volumes = torch.randn(N=4, 1, 91, 109, 91, T=4)` — synthetic
   stand-in for `load_subject_volumes`.
3. `normalize_per_subject(volumes)` — same as production.
4. `FMRIDataset` + `build_dataloader(batch_size=2, seed=2022)` +
   `DeviceDataLoader(..., device)`.
5. `RecVAEModel(train_size=4, cfg=Config(tol_time=4, epochs=3,
   batch_size=2, learning_rate=1e-5))`.
6. `h0 = torch.zeros(1, cfg.latent_dim, device=device)`.
7. `fit(model, dl, h0, cfg=cfg, epochs=3)` — does the alternation.
8. Print `train_loss_history`; assert it's a non-empty list.
9. `state_dict()` round-trip via `tempfile`, confirm forward pass
   agreement, clean up the file.

Note what is *not* in the script: any explicit reference to
`updating_F`, to the optimizer, to the per-loss-term breakdown.
Those are all inside `fit`. The user-facing API really is
"build the inputs, call `fit`".

## Run it

```bash
python tutorials/15_train_end_to_end/train_tiny.py
```

Expected output: three epoch lines from `fit` (one per epoch),
followed by the printed `train_loss_history` list, the save/load
banner, and a "round-trip OK" line. Total wall time well under 90
seconds on CPU.

## Why this approach

We could have written this as an integration test in `tests/`. We
chose a script for two reasons:

1. **Visibility.** Tests assert silently. A script prints loss history
   per epoch and the round-trip evidence, so a reader can *see* the
   alternating loop and the state_dict round-trip work.
2. **End-to-end framing.** Tests usually isolate one component.
   `train_tiny.py` deliberately glues everything together — `Config`,
   data, model, optimizer, save/load — exactly as you would in a real
   experiment, just smaller.

The integration-test version of this lives in `tests/test_train.py`;
this script is its didactic cousin.

## Further reading

- README "Train on real data" — same pattern, real NIfTI inputs,
  default config.
- `recvae/train.py:105-130` — the alternation loop, top to bottom.
- [Lesson 08](../08_losses_alt_optim/README.md) — the math behind the
  closed-form `F` update that `fit` calls once per epoch.
- [Lesson 10](../10_device_agnostic/README.md) — device-agnostic code
  and the MPS-backend caveat referenced in the script.
- [Lesson 11](../11_reproducibility/README.md) — RNG seeding and what
  `set_seed` actually pins.
- [Lesson 12](../12_save_load/README.md) — save/load patterns,
  `map_location`, the role of Parameters vs buffers in `state_dict()`.
