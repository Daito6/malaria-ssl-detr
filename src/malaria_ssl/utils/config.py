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


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load a config with an optional `base:` key (path to a parent config, applied first)."""
    cfg = load_config(resolve_path(path))
    base = cfg.pop("base", None)
    return deep_merge(load_experiment(base), cfg) if base else cfg


def apply_overrides(cfg: dict[str, Any], items: list[str]) -> dict[str, Any]:
    """`--set a.b=value` overrides (value parsed as YAML)."""
    for item in items:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(raw)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
