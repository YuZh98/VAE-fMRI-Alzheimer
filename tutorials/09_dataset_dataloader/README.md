# Lesson 09: Dataset and DataLoader

## What you'll learn

How `torch.utils.data.Dataset` and `DataLoader` plug into RecVAE's
training loop, why the dataset yields `(volume, idx)` instead of a bare
volume, and how to get *deterministic* shuffling that does not blow up
on CPU-only hardware.

## Where this lives in the repo

- `recvae/data.py:108-126` — `FMRIDataset`: subclasses `Dataset`, returns
  `(volume, idx)`.
- `recvae/data.py:129-147` — `build_dataloader`: wraps a `Dataset` in a
  `DataLoader` with an optional seeded `torch.Generator` for reproducible
  shuffle.
- `recvae/data.py:74-105` — `normalize_per_subject`: per-subject min-max
  to `[-1, 1]`, with a guard against constant volumes.
- `recvae/train.py:105-106` — consumer side: the training loop unpacks
  `for batch, batch_index in train_loader`.
- `recvae/model.py:215` — what `idx` is *for*: `self.z_vectors[which_ones]`
  looks up each subject's noise vector.

## The concept

### Why subclass `Dataset` instead of holding tensors in a list

A `Dataset` is a tiny interface — two methods, `__len__` and
`__getitem__` — but it buys you several things at once:

1. **Lazy access.** `__getitem__(idx)` only materializes what the loader
   actually asks for. With NIfTI files on disk that means we never need
   the full corpus in memory.
2. **Composability.** Anything that conforms to the interface plugs into
   `DataLoader`, `Subset`, `ConcatDataset`, `random_split`, samplers,
   and so on. Holding tensors in a Python list cuts you off from all of
   that.
3. **Indexing semantics.** The integer index is the natural place to
   carry along subject identity.

`FMRIDataset.__getitem__` returns `(self.volumes[idx], idx)`. The first
slot is the data; the second is the index *into the training set*.
That index is what `training_step` needs to look up the per-subject
noise vector — see `recvae/model.py:215`:

```python
h_tilde = h + self.z_vectors[which_ones]
```

`z_vectors` is a `(N_train, latent_dim)` Parameter; `which_ones` is the
batch of indices that just came off the loader. Without the index in
each sample, the model would have no way to associate the right
subject-specific noise with the right volume in a shuffled batch.

### What `__getitem__` and `__len__` must return

- `__len__(self) -> int`: the number of *samples* in the dataset
  (subjects, in our case — not timepoints, because each subject's full
  `(1, X, Y, Z, T)` volume is one sample).
- `__getitem__(self, idx)`: whatever your training step expects per
  sample. Tuples are fine; nested dicts are fine. `DataLoader` calls
  its `collate_fn` to stack samples into a batch; the default collate
  understands tensors and tuples/dicts containing tensors.

### Deterministic shuffle: the seeded `Generator` gotcha

`DataLoader(..., shuffle=True)` will pick a fresh permutation each epoch
by sampling from `torch.default_generator`. To make that reproducible
you pass a `torch.Generator` to `DataLoader(..., generator=...)`. The
generator's manual seed is what controls the permutation.

The footgun: the original notebook constructed the generator as

```python
generator = torch.Generator(device='cuda').manual_seed(seed)
```

That fails on CPU-only systems (`RuntimeError: Device type ... is not
supported for torch.Generator() api`) and is unnecessary in the first
place. `DataLoader` shuffle indices are CPU-side integers; the
generator does not need to live on the same device as the data.

The fix at `recvae/data.py:141-147` is to omit the device argument:

```python
generator = torch.Generator().manual_seed(seed) if seed is not None else None
return DataLoader(dataset, batch_size=batch_size,
                  shuffle=shuffle, generator=generator)
```

Same reproducibility, runs anywhere. The demo in this lesson exercises
both branches and asserts the order is identical across two runs with
the same seed.

## Code walk

Read `recvae/data.py:108-126`. Three things to notice:

- The constructor validates `dim() == 6`. That single guard catches a
  lot of bugs: passing pre-stacked-along-batch tensors, forgetting the
  channel dim, dropping the time axis after a slice.
- `__len__` is `volumes.shape[0]`, the subject count, *not*
  `volumes.numel()` (which would be voxel count).
- `__getitem__` returns the tuple. No collation magic; the default
  `DataLoader` collate stacks the volumes along a new batch dim and
  the indices into a 1-D `LongTensor`.

Then read `recvae/data.py:129-147` (`build_dataloader`) and trace the
`generator` path. Run the demo in this lesson to see it in action.

## Run it

```bash
python tutorials/09_dataset_dataloader/synthetic_dataset.py
```

The script builds a small synthetic 6D tensor, wraps it in a `Dataset`,
makes two DataLoaders with the same seed, iterates each, and asserts
they emit batches in the same order. Then it makes a third loader with
a different seed to show the order changes.

## Why this approach

The `Dataset`/`DataLoader` split is one of PyTorch's nicer pieces of
design. `Dataset` is concerned with "how do I get sample N?", which is
purely about your data. `DataLoader` is concerned with "how do I turn a
stream of samples into batches?", which is purely about training
ergonomics (shuffle, num_workers, drop_last, collate). Keeping them
separate means you can swap one without touching the other.

The `(volume, idx)` pattern in particular is worth internalizing: any
time your training loop needs to look up per-sample state, returning
the index from `__getitem__` is the cleanest path. Trying to thread
batch indices through a `Sampler` and a `collate_fn` works but is more
machinery for the same outcome.

## Exercise (optional)

Modify `synthetic_dataset.py` to add a third column to each sample: a
random integer label in `{0, 1}` representing a class. Confirm the
`DataLoader` collates the label column into a `LongTensor` of shape
`(batch_size,)` without any custom `collate_fn`.

## Further reading

- [PyTorch `torch.utils.data` docs](https://pytorch.org/docs/stable/data.html)
- `normalization.md` in this lesson — what `normalize_per_subject` is
  doing and why the per-subject choice is research-flagged.
- `recvae/train.py:105-116` — the consumer side of the loader, including
  the `h_history_history[which_ones] = ...` indexing trick that needs
  the per-batch indices.
