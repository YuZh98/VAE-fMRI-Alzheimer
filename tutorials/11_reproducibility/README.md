# Lesson 11: Reproducibility

## What you'll learn

Why "I set `torch.manual_seed`" is not enough, and how to pin every RNG
source that a PyTorch training run actually touches: Python, NumPy, the
PyTorch CPU generator, every CUDA device, cuDNN, and the `DataLoader`
shuffle generator. You will also learn the determinism/performance
trade-off that cuDNN forces on you, and the CPU-vs-CUDA gotcha that the
original notebook hit in its `DataLoader` generator.

## Where this lives in the repo

- `recvae/utils.py:15-58` — `set_seed(seed)` pins all the RNG sources
  described below.
- `recvae/data.py:141-147` — `build_dataloader` seeds a CPU-side
  `torch.Generator()` for deterministic shuffles.
- `tests/conftest.py:15-18` — an autouse fixture calls `set_seed(0)`
  before every test so the suite is bitwise reproducible.

## The concept

A modern training step pulls random numbers from several independent
RNGs. Pinning one of them is not enough; the run is reproducible only if
every RNG it consumes is pinned.

The sources you need to think about:

| Source                                       | Used by                                          |
|----------------------------------------------|--------------------------------------------------|
| Python `random.seed`                         | `random.shuffle`, anything in stdlib             |
| NumPy `np.random.seed`                       | `np.random.*`, many scientific libs              |
| PyTorch CPU `torch.manual_seed`              | `torch.randn`, dropout, init, CPU sampling       |
| PyTorch CUDA `torch.cuda.manual_seed_all`    | all GPU samplers, on every visible device        |
| cuDNN `torch.backends.cudnn.deterministic`   | conv algorithm selection                          |
| cuDNN `torch.backends.cudnn.benchmark`       | autotuner (non-deterministic when on)             |
| `DataLoader` `generator=`                    | per-epoch shuffle order                           |

`set_seed` in this repo handles the first five:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The original notebook only called `torch.manual_seed`. CUDA samplers and
cuDNN algorithm selection were left unpinned, so a re-run on the same
machine produced numerically different loss curves.

### The determinism / performance trade-off

cuDNN ships several implementations of each convolution. With
`benchmark = True` (opt-in; default is `False`) cuDNN times
implementations on the first batch and picks the fastest one — but
that timing depends on machine state, so two runs on different
hardware can pick different algorithms and produce different bitwise
results. Pin it to `False` for reproducibility and set
`deterministic = True`. Worse, some algorithms are themselves
non-deterministic across runs even for the same shape.

Forcing `deterministic = True` and `benchmark = False` tells cuDNN to
pick a deterministic algorithm even if it is slower. For a research
codebase where you want re-runnable experiments, that is the right
default. For a production training run where wall-clock matters more
than bitwise reproducibility, you would set `benchmark = True` and
accept the variation.

### The `DataLoader` generator gotcha

`DataLoader(shuffle=True)` uses a global RNG for the per-epoch shuffle
unless you pass `generator=`. Passing a fresh seeded generator makes the
shuffle reproducible across runs. The catch is **what device the
generator lives on**.

`recvae/data.py:141-147` constructs the generator with no device
argument, which means CPU:

```python
generator = torch.Generator().manual_seed(seed) if seed is not None else None
```

The earlier notebook wrote `torch.Generator(device='cuda')`. That works
on a GPU box but **crashes on a CPU-only machine**, and there is no
reason for the shuffle generator to live on GPU — the shuffle indices
are tiny CPU-side integers consumed by the `DataLoader` workers, not
GPU tensors. Always construct the shuffle generator on CPU.

### What `set_seed` does *not* fix

- Non-deterministic ops outside cuDNN (e.g. `scatter_add` on CUDA without
  `torch.use_deterministic_algorithms(True)`).
- Multi-worker `DataLoader` ordering when `worker_init_fn` is not set.
- Floating-point non-associativity on GPU (`(a + b) + c` may differ from
  `a + (b + c)` across runs if reductions happen in parallel).

For bit-exact reproducibility on GPU the standard escalation is
`torch.use_deterministic_algorithms(True)` plus the
`CUBLAS_WORKSPACE_CONFIG` environment variable. This repo does not go
that far; `set_seed` is enough for the CPU tutorial demos and for most
research re-runs.

## Code walk

`seed_demo.py` shows the minimum convincing demonstration: same seed
gives bitwise-equal tensors, different seed gives different tensors, no
seed gives non-reproducible tensors.

Three runs in the script:

1. `set_seed(2022)`, then `torch.randn(4)`, build `nn.Linear(4, 4)`,
   forward an input — record outputs.
2. `set_seed(2022)` again, repeat the same sequence — assert outputs are
   element-wise equal to run 1.
3. `set_seed(1234)` — same sequence, assert outputs differ.

Then a fourth check: the RNG state advances between draws, so two
consecutive `torch.randn(4)` calls produce different tensors even
without re-seeding. This is the baseline — randomness is the default;
reproducibility is the thing you opt into.

## Run it

```bash
python tutorials/11_reproducibility/seed_demo.py
```

Expected output:

- Two identical 4-vectors for runs 1 and 2 (the `torch.randn(4)` and the
  `Linear` output).
- A different 4-vector for run 3.
- Two different 4-vectors from consecutive `torch.randn` draws (RNG
  state advances between calls).
- Exit code 0.

## Why this approach

Reproducibility is a debugging tool, not a vanity metric. When a loss
curve changes between two runs and you cannot tell whether you changed
the *model* or the *RNG*, you cannot make progress. Pinning everything
removes the second variable so a delta in output reflects only a delta
in code.

The reason `set_seed` lives in `recvae/utils.py` and not at the top of
every script is that it is a one-line policy: pin everything, accept the
cuDNN slowdown, never debate it again.

## Exercise (optional)

Remove the `torch.backends.cudnn.deterministic = True` line from a local
copy of `set_seed`. On a CUDA machine, train one epoch twice and compare
the loss histories. On CPU you will not see a difference — cuDNN only
affects CUDA convolutions. This is why the trade-off is real but
GPU-specific.

## Further reading

- [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html)
- [cuDNN deterministic mode](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.deterministic)
- `torch.use_deterministic_algorithms` for the stricter setting.
