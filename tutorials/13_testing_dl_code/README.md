# Lesson 13: Testing DL code

## What you'll learn

What is worth testing in a deep-learning codebase, what is *not*, and
how to write each kind of test in pytest. The examples are grounded in
the existing `tests/` suite for this repo so you can see each pattern
applied to real code.

## Where this lives in the repo

- `tests/test_model.py` — shape, gradient, device-handling, parameter
  registration, math-identity, and error-path tests for `RecVAEModel`.
- `tests/test_data.py` — preprocessing edge cases (constant-volume
  guard, rank validation, deterministic shuffle).
- `tests/test_train.py` — end-to-end smoke tests for `fit()` and
  `evaluate()`.
- `tests/conftest.py` — shared fixtures and an autouse `set_seed(0)`
  that makes the whole suite reproducible.

## The concept

A DL test suite is **not** an experiment-tracking system. It does not
ask "did the model achieve 92% accuracy?" — that question lives in your
metrics dashboard. It asks "is the *machinery* still wired correctly?"
The machinery is shape arithmetic, device handling, gradient plumbing,
serialization, and a small number of math identities. All of that can
be tested with synthetic tensors in a few seconds.

Below are the categories that pay rent, each with a concrete example
from `tests/test_model.py`.

### 1. Shape contracts

Most deep-learning bugs are silent shape bugs. A `Conv3d` with the
wrong `output_padding` produces a tensor that is one voxel off; the
model still trains, the loss still goes down, the reconstructions look
plausible — and a downstream skip connection misaligns. Test the
shape, not the values.

`tests/test_model.py:11-18`:

```python
def test_encoder_decoder_shape_chain(model):
    x = torch.randn(2, 1, 91, 109, 91)
    enc = model.encode(x)
    assert enc.shape == (2, model.cfg.enc_out_dim)
    h = torch.randn(2, model.latent_dim)
    dec = model.decode(h)
    assert dec.shape == (2, 1, 91, 109, 91)
```

This catches every broken stride/padding/output_padding in one assertion.

### 2. Gradient flow

A common bug: a tensor is computed but accidentally detached, or a
parameter is registered on the model but never used in the forward
pass. The model still runs, but the parameter never learns.

`tests/test_train.py:17-34`:

```python
loss, _, _ = model.training_step(x, h0, which)
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert grads, "no parameter received a gradient"
```

Combine this with a check that *specific* parameters received grad
(`tests/test_train.py:37-46` does this for `z_vectors`) to catch the
case where the loss does not depend on some parameter you expected it to.

### 3. Device handling

A test can pin "no global state leaks." `RecVAEModel.reparametrize`
samples `eps` and must put it on the same device as the input — the
original notebook relied on `torch.set_default_tensor_type` to make
this happen, which is a global side effect we deliberately removed.

`tests/test_model.py:42-51`:

```python
mu = torch.zeros(2, model.latent_dim)
out = model.reparametrize(mu, torch.zeros_like(mu))
assert out.device == mu.device
assert out.dtype == mu.dtype
```

A CPU-only test can still catch this regression: if someone hardcodes
`torch.randn(..., device='cuda')`, the test fails on CPU because the
shapes won't match the input device.

### 4. Parameter vs buffer registration

`tests/test_model.py:54-64`:

```python
def test_z_vectors_is_parameter_with_correct_size(model):
    assert isinstance(model.z_vectors, torch.nn.Parameter)
    assert model.z_vectors.requires_grad

def test_F_mat_is_buffer_no_grad(model):
    assert "F_mat" in dict(model.named_buffers())
    assert model.F_mat.requires_grad is False
```

This kind of test guards against a refactor that "promotes" `F_mat` to
a Parameter (so the optimizer would start trying to learn it via
gradient descent, fighting the closed-form ridge update), or
"demotes" `z_vectors` to a buffer (so the optimizer would stop
updating it). Both bugs are easy to introduce and silent at runtime.

### 5. Determinism

`tests/conftest.py:15-18` declares an autouse fixture:

```python
@pytest.fixture(autouse=True)
def _pin_seed():
    set_seed(0)
```

So every test starts from the same RNG state and any tensor comparison
across re-runs is bitwise stable. Without this fixture, a test that
asserts `torch.allclose(out_a, out_b)` could be flaky when one of the
ops is RNG-sensitive.

### 6. Math identities

When you have a closed-form expression in the code, test that the code
matches the math. `tests/test_model.py:67-72` verifies
`g(h) == h @ F^T`; `tests/test_model.py:85-103` solves the ridge
system by hand and compares against `updating_F`.

These are the highest-value tests in the suite because they catch
algebraic errors that *do* affect training but produce no visible crash.

### 7. Error paths

`tests/test_model.py:33-39`:

```python
with pytest.raises(ValueError, match="timepoints"):
    model(x, h0, which)
```

For every `raise ValueError(...)` in the source, write a test that
triggers it. This locks the API: future refactors cannot remove the
guard without breaking a test, and you have a working example of the
error condition for documentation.

### Anti-patterns

The other side of the question: what *not* to test.

- **Exact loss values after N epochs.** Tests that assert
  `loss < 0.42` after 10 epochs on synthetic data are flaky. Tiny
  differences in op ordering across PyTorch versions, BLAS libraries, or
  CUDA drivers shift the value. You also conflate "math is right" with
  "training converged," which is a much weaker statement.
- **Convergence on tiny synthetic data.** A 4-subject, 3-timepoint
  dataset has nothing to converge to. Asserting "the loss went down" on
  data this small is testing torch.optim.SGD, not your model.
- **Tests that re-implement the code under test.** If your test for
  `decode` is `decode_again = lambda h: deco1(deco2(...(h)))` and then
  asserts equality, you have written the same bug twice. The shape
  contract test in category 1 avoids this — it asserts an *external*
  property of the function (its output shape) without re-implementing
  the function.
- **Snapshot tests on float tensors with no tolerance.** Comparing a
  saved float tensor across hardware with `torch.equal` will fail; use
  `torch.allclose` with a sensible `atol` if you do snapshot at all.

## Code walk

`test_examples.py` is a self-contained pytest file that demonstrates
five of the categories above on a toy `nn.Linear(4, 4)` model — small
enough that the file is readable end to end, but exercising the same
patterns the real suite uses on `RecVAEModel`.

The file also has an `if __name__ == "__main__":` block so you can run
it directly with `python tutorials/13_testing_dl_code/test_examples.py`
without needing pytest's discovery machinery. This is the same pattern
the rest of the tutorial demos use — every script must run as a plain
Python module.

Note: `pyproject.toml` declares `testpaths = ["tests"]`, so the
project's `pytest -q` invocation will not pick up
`tutorials/13_testing_dl_code/test_examples.py` automatically. That is
intentional — the tutorial file is illustrative, not part of the real
test suite. To run it through pytest:

```bash
pytest tutorials/13_testing_dl_code/test_examples.py -v
```

## Run it

```bash
# As a script (no pytest CLI needed):
python tutorials/13_testing_dl_code/test_examples.py

# Or through pytest, explicitly pointed at the file:
pytest tutorials/13_testing_dl_code/test_examples.py -v
```

Both should report five passing tests and exit 0.

## Why this approach

You want a CI suite that runs in seconds and tells you whether the
*plumbing* still works. Loss curves and benchmark scores belong in a
separate, slower pipeline (or a research notebook). Mixing the two
gives you a slow CI that flakes on hardware variation, which trains
people to ignore failing builds — the worst possible failure mode for
a research codebase.

The categorization above is the minimum bar. With those seven test
types in place you can refactor aggressively, because regressions in
the plumbing will be caught before they reach a training run.

## Exercise (optional)

Look at `recvae/data.py` and pick a function that has *no* test in
`tests/test_data.py`. Write one. The function `load_subject_volumes` is
a good candidate — see `tests/test_data.py` for the existing patterns
and `recvae/data.py:38-71` for the source.

## Further reading

- [pytest documentation](https://docs.pytest.org/)
- [PyTorch testing utilities](https://pytorch.org/docs/stable/testing.html) — `torch.testing.assert_close` is the recommended replacement for `torch.allclose` in tests, with richer failure messages.
- See also `tests/conftest.py` for the autouse `set_seed` fixture and the shared `model` / `small_cfg` / `synthetic_volumes` fixtures used across the suite.
- [`what_to_test.md`](what_to_test.md) — a heuristic checklist of test categories.
