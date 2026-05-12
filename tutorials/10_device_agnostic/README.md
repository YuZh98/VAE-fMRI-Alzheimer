# Lesson 10: Device-agnostic code

## What you'll learn

How RecVAE picks a device, why it never calls
`torch.set_default_tensor_type('torch.cuda.FloatTensor')`, and how the
small `DeviceDataLoader` wrapper keeps training-loop code clean of
explicit `.to(device)` calls.

## Where this lives in the repo

- `recvae/utils.py:61-67` — `get_default_device()`: CUDA -> MPS -> CPU.
- `recvae/utils.py:70-80` — `to_device()`: recursive move that preserves
  integer dtypes for index tensors.
- `recvae/utils.py:83-95` — `DeviceDataLoader`: thin wrapper that yields
  device-resident batches.
- `recvae/model.py:191-193` — `reparametrize`: builds `eps` with
  `device=mu_h.device, dtype=mu_h.dtype` instead of relying on a global
  default tensor type.

## The concept

### The device priority

`get_default_device()`:

```python
if torch.cuda.is_available():
    return torch.device("cuda")
if torch.backends.mps.is_available():
    return torch.device("mps")
return torch.device("cpu")
```

CUDA first because dedicated NVIDIA GPUs are the fastest path. MPS
second to pick up Apple Silicon GPUs. CPU last so the code always has
*some* device to fall back on. The function is pure: it does not
mutate any global state, it just returns a `torch.device` you can pass
around.

### Why `torch.set_default_tensor_type('torch.cuda.FloatTensor')` is a bad pattern

The notebook this repo descends from set the default tensor type to
CUDA at the top of the file. After that line ran, every bare
`torch.tensor(...)` and `torch.randn(...)` call would land on the GPU
without being told to.

That looks convenient but is a maintenance disaster:

1. **Global state.** Every module that imports `torch` after this point
   sees the changed default. Other libraries (e.g. `numpy` interop,
   matplotlib plotting code that does an intermediate `torch.tensor`)
   may break in surprising ways.
2. **Crashes on CPU-only systems.** No CUDA -> `torch.cuda.FloatTensor`
   does not exist -> `RuntimeError` at import time. The tutorial
   examples in this repo all run on a CPU laptop, which this pattern
   would forbid.
3. **Breaks tests.** Pytest runs on whatever the CI image provides
   (here, CPU). The default-tensor-type pattern means the *whole* test
   process is forced onto CUDA, so the suite cannot run at all.
4. **Action at a distance.** When a bug appears 20 imports later
   ("why is this tensor on CUDA when I never asked?") the cause is the
   single line at the top of a different file. That kind of bug is
   hard to find.

The correct pattern is to pass `device=` explicitly to every tensor
constructor that needs it. `recvae/model.py:192` is the canonical
example:

```python
eps = torch.randn(mu_h.shape, device=mu_h.device, dtype=mu_h.dtype)
```

`eps` lives wherever `mu_h` lives. No global state, works on any
device, and the intent is obvious to anyone reading the code.

### The `DeviceDataLoader` wrapper

If you have to call `.to(device)` on every batch inside every
training-loop iteration, you will eventually forget. `DeviceDataLoader`
(`recvae/utils.py:83-95`) wraps a plain `DataLoader` and moves each
batch on iteration:

```python
class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)
```

It is the smallest object that does one thing well. The training loop
just iterates as if the batches were already on the right device.

### Preserving integer dtypes

`to_device` (`recvae/utils.py:70-80`) recurses through list/tuple
batches and moves each tensor. The wrinkle is at lines 78-80:

```python
if data.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool):
    return data.to(device=device, non_blocking=True)
return data.to(device=device, dtype=dtype, non_blocking=True)
```

Integer (and bool) tensors are moved *without* a dtype cast. Float
tensors are cast to the configured float dtype.

That distinction matters because index tensors must stay integral.
`FMRIDataset.__getitem__` returns `(volume, idx)`; the loader collates
the indices into a `LongTensor`; that `LongTensor` is then used as
`self.z_vectors[which_ones]` in `recvae/model.py:215`. If `to_device`
indiscriminately cast everything to float, the indexing would crash
(`IndexError: tensors used as indices must be long, int, byte or bool`) or
silently return wrong rows.

## Code walk

Open `recvae/utils.py` and read the three functions together:

- `get_default_device` — just a priority list, no side effects.
- `to_device` — recursion + the integer-preservation branch.
- `DeviceDataLoader` — twelve lines, one job.

Then check `recvae/model.py:191-193` to see how the reparam trick
preserves device/dtype without needing any global default.

## Run it

```bash
python tutorials/10_device_agnostic/device_demo.py
```

The demo prints the picked device, runs a tiny `nn.Linear` forward
pass on it, then builds a `DataLoader` over synthetic `(volume, idx)`
samples, wraps it with `DeviceDataLoader`, and asserts each batch
lands on the picked device with the correct dtypes (float for
volumes, integer for indices).

## Why this approach

Two principles:

1. **Explicit beats implicit.** Pass `device=...` (and `dtype=...`)
   wherever you create a tensor. The cost is a handful of extra
   characters per call; the payoff is code that works on every
   device and doesn't surprise the next reader.
2. **Push device handling to the boundary.** Inside the training step,
   the model and the optimizer should not care which device they're
   on. The device decision is made once at startup
   (`get_default_device()`); the boundary that enforces it is
   `DeviceDataLoader` for input data and explicit `.to(device)` for
   the model. Everything between those boundaries is device-agnostic.

## Exercise (optional)

Add a `cpu` override to `device_demo.py`: accept a `--cpu` command-line
flag and skip the device picker, forcing CPU. Confirm the rest of the
script behaves identically. This is a useful pattern for debugging
device-specific failures and for keeping CI predictable.

## Further reading

- [`torch.set_default_tensor_type` docs](https://pytorch.org/docs/stable/generated/torch.set_default_tensor_type.html)
  (read the note about "Setting a non-default tensor type can cause
  unexpected results...").
- `recvae/utils.py:70-80` — the integer-dtype preservation, in
  context.
- Lesson 09 (`09_dataset_dataloader`) — where the index tensors
  come from in the first place.
