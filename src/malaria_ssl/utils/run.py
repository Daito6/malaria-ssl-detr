"""Helpers shared by the training loops: LR schedule, run metadata, checkpoints, metrics CSV."""
from __future__ import annotations

import csv
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from malaria_ssl.utils.config import resolve_path


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


def save_checkpoint(path: Path, **state) -> None:
    """Atomic torch.save (write to .tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


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
