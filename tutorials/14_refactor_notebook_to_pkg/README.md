# Lesson 14: Refactor notebook to package

The four notebooks in `legacy/` (`Version1.ipynb` through
`Version4.ipynb`) are the prehistory of this repository. Each is a
copy-modify of the previous one, which is the natural way a research
notebook evolves: you have a working prototype, you copy the whole
thing into V2, you change one hyperparameter or one block, and now
you have two copies of every cell that wasn't the focus of the change.
By V4, an estimated ~80% of cells were byte-identical or near-identical
across the four files. Fix a bug in V2's data loader and you have not
fixed it in V3 or V4.

The arc the four notebooks trace:

1. **V1** — a working prototype. `g_transform` is a 3-layer MLP and
   `loss2 = 0`.
2. **V2** — copy V1, change one thing. Subject-specific transition
   matrices `F_s`, `latent_dim = 50`, only 4 images.
3. **V3** — copy V2, change another thing. Matrix `g_transform`,
   `z_dim = 2`.
4. **V4** — copy V3, change a third thing. Shared `F`,
   `z_dim = latent_dim = 10`. This is what the canonical notebook (and
   `recvae/`) descended from.

The refactor extracts `recvae/` from `legacy/Version4.ipynb`, keeping
the math intact and rebuilding the engineering underneath it. The
result is one source of truth for each concept rather than four near-
copies, and a pytest suite that exercises the package on synthetic
tensors without needing nibabel or real fMRI data.

## Where this lives in the repo

- `legacy/Version{1,2,3,4}.ipynb` — the four predecessor notebooks. See `legacy/README.md` for a one-line summary of each.
- `recvae/config.py` — the extracted `Config` dataclass.
- `recvae/data.py` — NIfTI loading, normalization, `FMRIDataset`, `build_dataloader`.
- `recvae/model.py` — `RecVAEModel` (encoder, decoder, inference head, closed-form `F` update).
- `recvae/train.py` — `fit` and `evaluate`.
- `recvae/utils.py` — `set_seed`, `get_default_device`, `to_device`, `DeviceDataLoader`.

## How to recognize "ripe for extraction"

A research notebook is a good place to discover an idea; it is a poor
place to maintain one. Specific tells that a notebook has crossed the
line:

- **Cells duplicated across notebooks** with one-line edits. Move them
  to a module and parametrize the edit.
- **The same hyperparameter in multiple cells** (e.g. `latent_dim = 10`
  baked into encoder, decoder, and loss). Pull into a `Config`.
- **Logic with no test coverage** last "verified" by visually squinting
  at a printed shape. Move it to a function and write a pytest shape
  test.
- **Mixed concerns.** A single cell loads NIfTI files, normalizes them,
  builds a `DataLoader`, and instantiates the model. Split.
- **Global state leaking in.** `torch.set_default_tensor_type(...)`,
  notebook-cell ordering, manually re-running a cell to "reset" things.
  Move to a module where execution order is deterministic.
- **Magic constants without a name.** `(91, 109, 91)` appearing in
  three cells should be `cfg.spatial_shape` once.

## What the refactor actually did

The math did not change. The loss decomposition, the alternating
SGD-plus-closed-form-ridge update, the encoder/decoder topology, the
default hyperparameters — all preserved. Run an experiment from the
canonical notebook against `recvae/` and expect the same numbers, up
to the RNG fix that makes them more reproducible.

What actually changed (all visible in the diff):

1. **Eliminated the ~80% code duplication across V1/V2/V3/V4.** Each
   prior notebook was a copy-modify of the previous; `recvae/` has one
   source of truth for each concept.
2. **Modularized.** `Config` dataclass for hyperparams; `data.py` for
   loading and normalization; `model.py` for architecture and the
   closed-form `F` update; `train.py` for `fit` and `evaluate`;
   `utils.py` for device picker and seed.
3. **Device-agnostic.** Removed `torch.set_default_tensor_type
   ('torch.cuda.FloatTensor')` and the `DataLoader` with
   `generator=torch.Generator(device='cuda')`. The same code now runs
   on CPU, CUDA, and MPS.
4. **Added a pytest suite.** 23 tests, CPU-only, finishes in ~5
   seconds. Uses synthetic tensors so it does not require nibabel or
   real fMRI data to run in CI.
5. **Stripped notebook outputs.** `nbstripout` keeps embedded image and
   tensor outputs out of git history. The repo stays small, diffs stay
   readable.
6. **Pinned all RNG state.** `set_seed` covers Python `random`, NumPy,
   torch CPU, all CUDA devices, and toggles cuDNN deterministic. The
   notebook only called `torch.manual_seed`.
7. **`z_vectors` is now `nn.Parameter`, `F_mat` is a registered buffer.**
   In the notebooks they were plain `self.z_vectors = torch.randn(...)`
   tensors: invisible to `.parameters()`, missing from `state_dict()`,
   not moved by `.to(device)`. The refactor makes them serialize and
   migrate correctly.
8. **Validated NIfTI timepoint count.** Notebooks silently stacked
   tensors of mismatched length and produced shape errors much later.
   `load_subject_volumes` now raises `ValueError` immediately.
9. **Guarded `normalize_per_subject` against zero span.** A constant
   volume previously produced NaN through `(x - min) / 0`; we now
   substitute span = 1 in that degenerate case.

Several of these are correctness fixes, not just ergonomic wins: the
`state_dict` round-trip, the absence of silent shape mismatches, and
the NaN guard all change behavior in cases the notebooks got wrong.

## Code walk

Read `before_after.md` in this lesson — it lists seven concrete
before/after pairs with file:line citations into `recvae/`. Each pair
shows one notebook idiom and the package-level rewrite of it.
Suggested reading order:

1. RNG seeding (utils).
2. Device tensors (model `reparametrize`).
3. DataLoader generator (data).
4. `z_vectors` promotion (model `__init__`).
5. `F_mat` registration (model `__init__`).
6. Timepoint validation (data loading).
7. Zero-span normalization (data normalization).

Then skim the four legacy notebooks and notice how much of each is a
verbatim copy of the previous. That is the duplication the `recvae/`
package collapses.

## Run it

No demo script for this lesson; the artifact is the package itself.
Browse `recvae/` and run the test suite:

```bash
.venv/bin/pytest -q
```

23 tests, ~5 seconds. Each test exercises one chunk of the refactored
package against a synthetic tensor — the kind of coverage the
notebooks never had.

## Why this approach

Two alternatives we did *not* take:

- **Keep the notebook, add tests.** Notebooks are a hostile environment
  for pytest: cell ordering matters, cells share global state, and
  importing a notebook into a test file is awkward. You also still pay
  the duplication tax across V1-V4.
- **Rewrite from scratch.** Tempting; you lose continuity with earlier
  experiments. The refactor preserved exact training math so numbers
  from the canonical notebook still mean something against the
  package.

Extract the modules, keep the math, add tests — the cheapest way to
make a research codebase maintainable without abandoning its history.
The legacy notebooks stay archived in `legacy/` for reference; nobody
runs them, but you can read them to see how the model evolved.

## Further reading

- `legacy/README.md` — one-paragraph history of V1 through V4.
- Jeremy Howard, *nbdev* — alternative approach that keeps the notebook
  as the source of truth and generates modules from it. Worth knowing
  about even if you don't use it.
- `recvae/` itself — the deliverable of this refactor. Read top to
  bottom once.
