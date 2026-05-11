"""Save and load full training-run snapshots.

A "run snapshot" is everything you need to resume or audit a training
run: the model state, the optimizer state, the config, the loss history,
and the RNG states for every library that produces randomness in the
pipeline. The git SHA and a pip freeze are captured opportunistically so
the snapshot also serves as a record of what code produced it.

Layout of a saved run directory:

    run_dir/
        model_state.pt       # torch state_dict, weights_only-friendly
        optimizer_state.pt   # if an optimizer was passed
        config.json          # dataclass fields, serialized
        history.json         # whatever history dict was passed
        rng_states.pt        # torch CPU+CUDA, numpy, python random
        git_sha.txt          # current git HEAD SHA, if available
        env.txt              # pip freeze output
"""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config
from .model import RecVAEModel


def _git_sha(repo_root: Path) -> str | None:
    """Return ``git rev-parse HEAD`` for ``repo_root`` or None on failure.

    Failure here is non-fatal: most failure modes ("not a git repo",
    "git not on PATH", ...) are uninteresting from the perspective of a
    save-run helper, and we never want a missing git binary to break
    checkpoint saving.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _pip_freeze() -> str:
    """Return the current pip-freeze output (best effort).

    Uses the same Python interpreter the caller is running under so the
    environment recorded matches what actually loaded the model.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _capture_rng_states() -> dict:
    """Snapshot every RNG state we care about."""
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_states(state: dict) -> None:
    """Inverse of :func:`_capture_rng_states`.

    Missing keys are silently skipped — the snapshot may have been written
    on a machine with a different device profile (e.g. CUDA->CPU).
    """
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def save_run(
    run_dir: str | Path,
    model: RecVAEModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    cfg: Config | None = None,
    history: dict | None = None,
) -> Path:
    """Persist a complete training-run snapshot to ``run_dir``.

    Creates ``run_dir`` if it doesn't exist. Each component is written to
    its own file so partial recovery is possible if any one of them is
    later corrupted.

    Returns the resolved Path to ``run_dir`` for caller convenience.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Model state — use the recommended weights-only-friendly call shape.
    torch.save(model.state_dict(), run_dir / "model_state.pt")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), run_dir / "optimizer_state.pt")

    # Config: JSON makes it readable from any non-Python tool. asdict()
    # recursively flattens dataclass fields (including the nested tuple).
    if cfg is not None:
        cfg_payload: dict[str, Any] = dataclasses.asdict(cfg)
        if epoch is not None:
            cfg_payload["__epoch__"] = epoch
        (run_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2))
    elif epoch is not None:
        (run_dir / "config.json").write_text(json.dumps({"__epoch__": epoch}, indent=2))

    if history is not None:
        # History may contain torch tensors — coerce to lists/floats first.
        def _coerce(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_coerce(v) for v in obj]
            return obj

        (run_dir / "history.json").write_text(json.dumps(_coerce(history), indent=2))

    torch.save(_capture_rng_states(), run_dir / "rng_states.pt")

    sha = _git_sha(Path.cwd())
    if sha:
        (run_dir / "git_sha.txt").write_text(sha + "\n")

    env_txt = _pip_freeze()
    if env_txt:
        (run_dir / "env.txt").write_text(env_txt)

    return run_dir


def load_run(
    run_dir: str | Path,
    model: RecVAEModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Inverse of :func:`save_run`.

    Loads model state in-place. Returns a dict with whatever optional
    components were found on disk: ``optimizer_state`` (if loaded into the
    optimizer arg or just returned raw), ``cfg`` (dict form), ``history``,
    ``epoch``, ``git_sha``.

    Parameters
    ----------
    run_dir : directory written by :func:`save_run`.
    model : the model instance to load state into.
    optimizer : if provided, its ``load_state_dict`` is called.
    map_location : passed through to ``torch.load``. Default "cpu" keeps
                   the load operation device-agnostic.
    restore_rng : if True (default), also restore the RNG states. Set to
                  False if you want save/load to be a pure persistence
                  operation that does not touch the global RNG.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    result: dict[str, Any] = {}

    model_path = run_dir / "model_state.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing required file: {model_path}")
    state = torch.load(model_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)

    opt_path = run_dir / "optimizer_state.pt"
    if opt_path.is_file():
        opt_state = torch.load(opt_path, map_location=map_location, weights_only=False)
        result["optimizer_state"] = opt_state
        if optimizer is not None:
            optimizer.load_state_dict(opt_state)

    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        cfg_payload = json.loads(cfg_path.read_text())
        if "__epoch__" in cfg_payload:
            result["epoch"] = cfg_payload.pop("__epoch__")
        if cfg_payload:
            result["cfg"] = cfg_payload

    history_path = run_dir / "history.json"
    if history_path.is_file():
        result["history"] = json.loads(history_path.read_text())

    sha_path = run_dir / "git_sha.txt"
    if sha_path.is_file():
        result["git_sha"] = sha_path.read_text().strip()

    if restore_rng:
        rng_path = run_dir / "rng_states.pt"
        if rng_path.is_file():
            rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
            _restore_rng_states(rng_state)

    return result
