# What to test in a DL project — checklist

A short rationale for each category. See `README.md` for the long form
and `tests/` for real applications.

## Test these

- [ ] **Shape contracts.** A forward through every distinct
  encoder/decoder/head produces the expected shape. Catches silent
  off-by-one bugs in stride/padding/output_padding.
- [ ] **Round-trip identities.** Encode-then-decode preserves the
  spatial extent; save-then-load reproduces the output bitwise (in
  eval mode). Catches dimension-arithmetic and serialization bugs.
- [ ] **Gradient flow.** Every parameter that should learn receives a
  finite gradient after `loss.backward()`. Catches accidental
  `.detach()` and dangling parameters.
- [ ] **Device handling.** Sampled tensors land on the input's device,
  not some global default. Catches `set_default_tensor_type` residue
  and hardcoded `device='cuda'`.
- [ ] **Parameter vs buffer registration.** Things that should be in
  `.parameters()` are; things that should not, aren't; everything is
  in `state_dict()`. Catches refactors that change tensor categories
  by accident.
- [ ] **Math identities.** Closed-form expressions in the code match
  the math (e.g. `g(h) == h @ F^T`, the ridge solution against a
  hand-written reference). Catches algebraic errors that produce no
  crash and no visible symptom.
- [ ] **Determinism.** Two re-runs with the same seed produce
  identical outputs. Provides a baseline for every other test that
  uses `torch.allclose` or `torch.equal`.
- [ ] **Error paths.** Every `raise` in the source has a test that
  triggers it. Locks the API contract.
- [ ] **Data preprocessing edge cases.** Constant inputs, empty
  inputs, wrong rank, wrong dtype. Catches silent NaN / silent shape
  drift in the pipeline.

## Skip these

- [ ] Exact loss values after N epochs. Flaky across BLAS/cuDNN/driver
  versions.
- [ ] Convergence on tiny synthetic data. Tests the optimizer, not your
  code.
- [ ] Tests that re-implement the function under test. You will
  write the same bug twice.
- [ ] `torch.equal` on float tensors across hardware. Use
  `torch.allclose` with a tolerance.
- [ ] Tests that depend on global state set by *other* tests. Use
  fixtures; assume nothing about order.
