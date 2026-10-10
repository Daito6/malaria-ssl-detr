"""YAML config loading and project paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
