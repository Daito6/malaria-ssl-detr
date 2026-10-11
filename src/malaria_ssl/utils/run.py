"""Helpers shared by the training loops: LR schedule, run metadata, checkpoints, metrics CSV."""
from __future__ import annotations

import csv
import logging
import math
import os
import pickle
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from malaria_ssl.utils.config import resolve_path

log = logging.getLogger(__name__)


def lr_lambda(warmup: int, total: int, min_ratio: float):
    """Linear warmup over `warmup` steps, then cosine decay to `min_ratio` at `total` steps."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(total - warmup, 1), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def run_info() -> dict[str, Any]:
    """Git commit / dirty flag, python and package versions, CUDA and GPU name."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, cwd=resolve_path("."),
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    import importlib.metadata as md
    versions = {}
    for p in ("torch", "torchvision", "transformers", "timm", "albumentations", "pycocotools", "numpy"):
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "packages": versions, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def prev_path(path: Path) -> Path:
    """last.pt -> last.prev.pt"""
    return path.with_name(f"{path.stem}.prev{path.suffix}")


def _write_synced(obj: Any, path: Path) -> Path:
    """torch.save to <path>.tmp and fsync it; returns the .tmp path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    return tmp


def atomic_save(obj: Any, path: Path) -> None:
    """torch.save to <path>.tmp, fsync, then os.replace: `path` is either the old or the new complete file.

    Without the fsync a crash right after the rename can leave a renamed but truncated file.
    """
    os.replace(_write_synced(obj, path), path)


def save_checkpoint(path: Path, keep_prev: bool = False, **state) -> None:
    """Atomic save of a training state; with keep_prev the replaced file is kept as <stem>.prev.pt.

    The previous file is rotated only after the new state is fully written, and it is hard-linked
    (copied if links are unsupported) rather than moved, so `path` never disappears and the resume
    checks for `path` stay valid at every moment.
    """
    tmp = _write_synced(state, path)
    if keep_prev and path.exists():
        prev = prev_path(path)
        prev.unlink(missing_ok=True)
        try:
            os.link(path, prev)
        except OSError:
            shutil.copy2(path, prev)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """torch.load(path); if it is missing or unreadable, fall back to <stem>.prev.pt (one epoch older)."""
    for p in (path, prev_path(path)):
        if not p.exists():
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            log.warning("cannot read checkpoint %s: %s", p, e)
            continue
        if p != path:
            log.warning("falling back to the previous checkpoint %s", p)
        return ck
    raise FileNotFoundError(f"no readable checkpoint: {path} or {prev_path(path)}")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Add / replace the row of `row["epoch"]` (idempotent on resume); new columns are allowed."""
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(r["epoch"]) != row["epoch"]] + [row]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
