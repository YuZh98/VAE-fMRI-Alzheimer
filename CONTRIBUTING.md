# Contributing

Thanks for your interest in improving RecVAE. This document walks through
local setup, the test/lint/demo loop, and the conventions used across the
repository.

## Development setup

Python 3.9, 3.10, 3.11, 3.12, and 3.13 are all supported (CI exercises
3.9-3.12 on Ubuntu and macOS).

```bash
git clone https://github.com/YuZh98/VAE-fMRI-Alzheimer
cd VAE-fMRI-Alzheimer

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

For an exactly-reproducible environment (e.g. when chasing a regression),
use `requirements.lock` instead of `pip install -e ".[dev]"`:

```bash
pip install -r requirements.lock
pip install -e .
```

## Run the tests

```bash
pytest -v
```

The full suite is CPU-only and runs in a few seconds.

PRs are gated on `tests.yml` (pytest matrix); the full tutorial demo
loop runs nightly via `tutorials.yml`.

## Run every tutorial demo

The full demo loop runs nightly on CI; you do not need to wait for it
on every PR. To run the same loop locally:

```bash
for f in tutorials/*/*.py tutorials/*/*/*.py; do
  base="$(basename "$f")"
  case "$base" in
    _*) continue ;;
  esac
  .venv/bin/python "$f" || exit 1
done
```

Files whose basename starts with `_` (e.g. `_tutorial_utils.py`) are
shared helpers and are skipped.

To run `examples/classify_from_latents.py`, install with
`pip install -e ".[examples]"` (this pulls in `scikit-learn`).

## Lint and type-check

The repo is `ruff`-formatted and `ruff`-linted, with a smoke-level mypy
configuration.

```bash
ruff check recvae/ tests/
ruff format recvae/ tests/        # rewrite in place
ruff format --check recvae/ tests/  # CI-mode (read-only)
mypy recvae/
```

Mypy is intentionally lenient (`ignore_missing_imports = true`,
`disable_error_code = ["import-untyped"]`) — it is a smoke check, not a
strict-mode gate.

## Pre-commit

Install the hooks once per clone:

```bash
pip install pre-commit
pre-commit install
```

The configured hooks run `ruff` (lint + format), `nbstripout` on
notebooks, plus the standard trailing-whitespace, end-of-file, and YAML
checks. They run automatically on `git commit`.

## Notebook hygiene

To keep diffs small and avoid leaking local outputs, run:

```bash
nbstripout --install
```

once per clone. This installs a git filter that strips notebook outputs at
commit time. The `pre-commit` config also strips outputs as a safety net.

## Branching and pull requests

- Cut feature branches off `main`.
- Open the PR back into `main`.
- Use the PR template in `.github/PULL_REQUEST_TEMPLATE.md` (it is
  applied automatically).

Code review is currently routed via `.github/CODEOWNERS` to @YuZh98.
Adding a co-maintainer is welcome — please open an issue if you'd like
to take on review duty for a subtree (e.g. `tutorials/` or `docs/`).

## Adding a tutorial

Drop a script at `tutorials/NN_topic/your_demo.py` (or nested one level
deeper at `tutorials/NN_topic/subtopic/your_demo.py`). CI will pick it up
automatically — no workflow change required.

The convention every demo follows:

1. A short `sys.path` bootstrap so the script runs from any CWD.
2. `from tutorials import _tutorial_utils` for shared helpers
   (seed setup, tiny synthetic generators, plotting, ...).
3. Keep runtime to a few seconds on CPU. Use `TUTORIAL_THREADS` to
   constrain torch threading when it matters.

If your script is a helper that should *not* be executed by CI, prefix
its filename with `_` (e.g. `_helpers.py`).

## Issue and PR templates

- Bug reports: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Feature requests: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Research / methodology questions: `.github/ISSUE_TEMPLATE/research_question.yml`
- Pull request template: `.github/PULL_REQUEST_TEMPLATE.md`

The template chooser is configured in
`.github/ISSUE_TEMPLATE/config.yml` (blank issues are disabled — please
pick a form).

## Code style

`ruff`-enforced. Configuration lives in `[tool.ruff]` inside
`pyproject.toml`. Line length is 100; the rule set is
`E, F, W, I, UP, B, SIM` with `E501` deferred to the formatter.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `chore:` tooling / build / repo plumbing
- `refactor:` code change that is neither a feature nor a fix
- `test:` add or refactor tests

Example: `feat(model): make sig_x learnable`.
