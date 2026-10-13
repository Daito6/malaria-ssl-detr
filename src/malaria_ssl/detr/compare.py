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
    # older (phase 2) runs did not log the parameter count
    row["trainable_params"] = int(found.group(1)) if found else None
    row["total_params"] = int(found.group(2)) if found else None
    hist = pd.read_csv(run_dir / "metrics.csv")
    row["epochs"] = len(hist)
    row["train_min"] = (hist["epoch_time_s"].sum() + hist["val_time_s"].fillna(0).sum()) / 60
    row["primary"] = cls
    return row


def shift_rows(shift_json: str | Path) -> list[dict[str, Any]]:
    """Val and test rows of a reports/tables/shift_<run>.json (scripts/diagnose_shift.py)."""
    s = json.loads(Path(shift_json).read_text(encoding="utf-8"))
    return [{"run": s["run"], "split": split, **s[split]} for split in ("val", "test")]


def val_curve(run_dir: str | Path, cls: str = "infected") -> pd.Series:
    """In-training val AP50 of `cls`, indexed by the number of finished epochs (epoch index + 1)."""
    hist = pd.read_csv(Path(run_dir) / "metrics.csv")
    hist = hist.dropna(subset=[f"val_AP50/{cls}"])
    return pd.Series(hist[f"val_AP50/{cls}"].to_numpy(), index=hist["epoch"].to_numpy() + 1,
                     name=Path(run_dir).name)


def plot_val_curves(curves: pd.DataFrame, colors: dict[str, str], ax=None, mark: dict[str, int] | None = None):
    """In-training val AP50 curves (columns = runs); `mark` = {run: evaluated epoch} drawn as a dot."""
    import matplotlib.pyplot as plt

    from malaria_ssl.data.viz import _recessive_axes

    ax = ax or plt.subplots(figsize=(8, 4.2))[1]
    for run in curves.columns:
        ax.plot(curves.index, curves[run], lw=2, color=colors[run], label=run)
        if mark and run in mark:
            ax.plot(mark[run], curves.loc[mark[run], run], "o", ms=8, color=colors[run], mec="white", mew=2)
    ax.set_xlabel("epochs done")
    ax.set_ylabel("val AP50 infected (in-training)")
    ax.legend(frameon=False, fontsize=9)
    _recessive_axes(ax)
    return ax


def plot_paired_diffs(diff: pd.DataFrame, metric: str, ax=None):
    """Forest plot of paired differences a - b with 95% CI for one metric (rows of a *_diff.csv)."""
    import matplotlib.pyplot as plt

    from malaria_ssl.data.viz import _recessive_axes

    d = diff[diff["metric"] == metric].iloc[::-1]
    ax = ax or plt.subplots(figsize=(7, 0.5 * len(d) + 1))[1]
    y = range(len(d))
    ax.hlines(y, d["low"], d["high"], color="#555555", lw=2)
    ax.plot(d["diff"], y, "o", ms=8, color="#2a78d6", mec="white", mew=2)
    ax.axvline(0, color="#9a9a9a", lw=1, ls="--")
    ax.set_yticks(list(y), [f"{a} − {b}" for a, b in zip(d["a"], d["b"])], fontsize=9)
    ax.set_xlabel(f"paired difference, {metric} (95% CI)")
    _recessive_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    return ax


def select_by_simplicity(rows: list[dict[str, Any]], cls: str = "infected") -> dict[str, Any]:
    """Leader = highest AP50 of `cls`; tied = point estimate inside the leader's 95% CI;
    winner = the first tied row in the given order (rows are passed simplest first)."""
    key = f"AP50_{cls}"
    leader = max(rows, key=lambda r: r[key])
    lo, hi = leader[f"{key}_lo"], leader[f"{key}_hi"]
    tied = [r["run"] for r in rows if lo <= r[key] <= hi]
    return {"leader": leader["run"], "leader_ci": (lo, hi), "tied": tied, "winner": tied[0]}
