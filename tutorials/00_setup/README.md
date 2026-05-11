# Lesson 00: Setup

## What you'll learn

How to bring this repo from a cold clone to a working tutorial environment.
By the end you will have a Python virtual environment, the dev dependencies,
and a `verify_torch.py` run that confirms PyTorch can see your hardware.

## Where this lives in the repo

- `requirements.txt`, `requirements-dev.txt` — pinned dependency lists.
- `pyproject.toml` — package metadata for the editable install.
- `recvae/utils.py:29-35` — `get_default_device()` picks CUDA, then MPS, then CPU.
- `tutorials/_tutorial_utils.py` — the `sys.path` shim that lets every demo
  `import recvae` without an editable install.

## The concept

A tutorial repo is only useful if the reader can run it. Four steps:

1. Clone the repo.
2. Create a virtual environment so installs do not contaminate the system Python.
3. Install dev dependencies.
4. Run a known-good script that exercises PyTorch end to end.

If step 4 prints expected output and exits 0, the rest of the lessons will run.

## Code walk

```bash
git clone <repo-url> fMRI_Project
cd fMRI_Project
python3.13 -m venv .venv         # or python3.12; see breakage note below
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .                  # optional; see below
```

`requirements-dev.txt` pulls in `requirements.txt` (torch, numpy, nibabel,
matplotlib) plus pytest and nbstripout.

The editable install (`pip install -e .`) is the standard way to expose the
`recvae` package on the import path. It is **not required** for these tutorials:
every demo starts with `from tutorials._tutorial_utils import section, banner`,
and `_tutorial_utils.py` prepends the repo root to `sys.path` as a side effect
of being imported. Without the editable install you can still run any demo as

```bash
python tutorials/00_setup/verify_torch.py
```

### Python version note

Use Python **3.13** or **3.12**. Python **3.14** is not yet supported — at the
time of writing, PyTorch 2.11 does not publish 3.14 wheels and `pip install`
will either fail or fall back to a source build that needs a CUDA toolchain.
Stick to 3.13 unless you have a specific reason not to.

## Run it

```bash
python tutorials/00_setup/verify_torch.py
```

Expected output (numbers will differ on your machine):

- Python version line
- torch version line
- recvae version line
- the device that `get_default_device()` picked
- the result of `torch.eye(3) @ torch.eye(3)` (the 3x3 identity)

Exit code 0.

## Why this approach

A standalone verifier catches three classes of breakage early: wrong Python
version, broken torch install, and `recvae` not on the import path. Running
it first means later lessons can assume the basics work.

CI runs every demo in the `tutorials/` tree on every push, so `verify_torch.py`
also serves as a smoke test for the whole tutorial set — if it fails in CI,
nothing else is expected to pass.

## Further reading

- [Python venv docs](https://docs.python.org/3/library/venv.html)
- [PyTorch install matrix](https://pytorch.org/get-started/locally/)
- [PEP 660 editable installs](https://peps.python.org/pep-0660/)
