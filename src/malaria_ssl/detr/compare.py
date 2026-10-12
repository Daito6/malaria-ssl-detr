"""Comparison of finished DETR runs from their saved artifacts (reads only; no model, no GPU)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_TRAINABLE = re.compile(r"trainable parameters: (\d+) of (\d+)")


def run_summary(run_dir: str | Path, split: str = "val", cls: str = "infected") -> dict[str, Any]:
    """One row: AP50 per class with 95% bootstrap CI at the evaluated checkpoint, params and time.

    `train_min` is the sum of per-epoch train + in-training val time from metrics.csv (pure compute:
    it excludes process start-up and any interruption between a crash and --resume).
    """
    run_dir = Path(run_dir)
    m = json.loads((run_dir / f"metrics_{split}.json").read_text(encoding="utf-8"))
    ci = m["bootstrap"]["ci95"]
    row: dict[str, Any] = {"run": run_dir.name, "checkpoint_epoch": m["checkpoint_epoch"]}
    for c in m["coco"]["per_class"]:
        row[f"AP50_{c}"] = m["coco"]["per_class"][c]["AP50"]
        row[f"AP50_{c}_lo"] = ci[f"AP50/{c}"]["low"]
        row[f"AP50_{c}_hi"] = ci[f"AP50/{c}"]["high"]
    found = _TRAINABLE.search((run_dir / "train.log").read_text(encoding="utf-8"))
    row["trainable_params"] = int(found.group(1))
    row["total_params"] = int(found.group(2))
    hist = pd.read_csv(run_dir / "metrics.csv")
    row["epochs"] = len(hist)
    row["train_min"] = (hist["epoch_time_s"].sum() + hist["val_time_s"].fillna(0).sum()) / 60
    row["primary"] = cls
    return row


def val_curve(run_dir: str | Path, cls: str = "infected") -> pd.Series:
    """In-training val AP50 of `cls`, indexed by the number of finished epochs (epoch index + 1)."""
    hist = pd.read_csv(Path(run_dir) / "metrics.csv")
    hist = hist.dropna(subset=[f"val_AP50/{cls}"])
    return pd.Series(hist[f"val_AP50/{cls}"].to_numpy(), index=hist["epoch"].to_numpy() + 1,
                     name=Path(run_dir).name)


def select_by_simplicity(rows: list[dict[str, Any]], cls: str = "infected") -> dict[str, Any]:
    """Leader = highest AP50 of `cls`; tied = point estimate inside the leader's 95% CI;
    winner = the first tied row in the given order (rows are passed simplest first)."""
    key = f"AP50_{cls}"
    leader = max(rows, key=lambda r: r[key])
    lo, hi = leader[f"{key}_lo"], leader[f"{key}_hi"]
    tied = [r["run"] for r in rows if lo <= r[key] <= hi]
    return {"leader": leader["run"], "leader_ci": (lo, hi), "tied": tied, "winner": tied[0]}
