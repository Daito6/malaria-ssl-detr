"""Loading run artifacts and plotting for the baseline analysis notebook (reads runs/ only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from malaria_ssl.data.detection import load_split_coco, read_rgb
from malaria_ssl.data.viz import CLASS_COLORS, SERIES, SPLIT_COLORS, _recessive_axes
from malaria_ssl.eval.detection_metrics import iou_matrix
from malaria_ssl.utils.config import load_config, resolve_path

SPLITS = ("val", "test")


# --------------------------------------------------------------------------- loading

def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Config, per-epoch metrics and any metrics_/predictions_ files present in a run dir."""
    run_dir = Path(run_dir)
    run = {"name": run_dir.name, "dir": run_dir,
           "config": yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
           "history": pd.read_csv(run_dir / "metrics.csv"), "metrics": {}, "predictions": {}}
    for s in SPLITS:
        if (p := run_dir / f"metrics_{s}.json").exists():
            run["metrics"][s] = json.loads(p.read_text(encoding="utf-8"))
        if (p := run_dir / f"predictions_{s}.json").exists():
            run["predictions"][s] = json.loads(p.read_text(encoding="utf-8"))
    return run


def gt_for_run(run: dict[str, Any], split: str) -> dict[str, Any]:
    dcfg = load_config(resolve_path(run["config"]["data_config"]))
    return load_split_coco(split, run["config"]["label_mode"], dcfg, resolve_path)


# --------------------------------------------------------------------------- tables

def metrics_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (run, split): COCO metrics, macro F1 and bootstrap 95% CIs when present."""
    rows = []
    for r in runs:
        for s, m in r["metrics"].items():
            c = m["coco"]
            row = {"run": r["name"], "split": s, "label_mode": m["label_mode"], "epoch": m["checkpoint_epoch"],
                   "AP": c["AP"], "AP50": c["AP50"], "AP75": c["AP75"], "AR@300": c["AR@300"],
                   "macro_F1": m["prf"]["macro_f1"]}
            ci = m.get("bootstrap", {}).get("ci95", {})
            for key, col in (("mAP50", "AP50"), ("macro_f1", "macro_F1")):
                if key in ci:
                    row[f"{col}_CI95"] = f"[{ci[key]['low']:.3f}, {ci[key]['high']:.3f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def per_class_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for s, m in run["metrics"].items():
        ci = m.get("bootstrap", {}).get("ci95", {})
        for name, v in m["coco"]["per_class"].items():
            p = m["prf"]["per_class"][name]
            row = {"split": s, "class": name, "n_gt": v["n_gt"], "AP": v["AP"], "AP50": v["AP50"],
                   "AR@300": v["AR@300"], "P": p["precision"], "R": p["recall"], "F1": p["f1"],
                   "thr": p["threshold"]}
            for key, col in ((f"AP50/{name}", "AP50_CI95"), (f"F1/{name}", "F1_CI95")):
                if key in ci and ci[key]["n"]:
                    row[col] = f"[{ci[key]['low']:.3f}, {ci[key]['high']:.3f}]"
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- curves

def plot_training_curves(run: dict[str, Any]):
    h = run["history"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, col, title in zip(axes, ["loss_ce", "loss_bbox", "loss_giou", "loss"],
                              ["CE (final layer)", "L1 box (final layer)", "GIoU (final layer)", "total (weighted, all layers)"]):
        ax.plot(h["epoch"], h[col], color=SERIES[0], lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        _recessive_axes(ax)
    fig.suptitle(f"{run['name']}: training losses", fontsize=11)
    fig.tight_layout()
    return fig


def plot_lr_gradnorm(run: dict[str, Any]):
    h = run["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(h["epoch"], h["lr"], color=SERIES[0], lw=2, label="transformer")
    axes[0].plot(h["epoch"], h["lr_backbone"], color=SERIES[1], lw=2, label="backbone")
    axes[0].set_yscale("log")
    axes[0].set_title("learning rate (end of epoch)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].plot(h["epoch"], h["grad_norm"], color=SERIES[0], lw=2)
    axes[1].set_title("grad norm before clipping (epoch mean)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        _recessive_axes(ax)
    fig.tight_layout()
    return fig


def plot_val_ap(runs: list[dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, r in enumerate(runs):
        h = r["history"].dropna(subset=["val_AP"])
        ax.plot(h["epoch"], h["val_AP50"], color=SERIES[k], lw=2, marker="o", ms=4, label=f"{r['name']} AP50")
        ax.plot(h["epoch"], h["val_AP"], color=SERIES[k], lw=2, ls="--", label=f"{r['name']} AP")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val")
    ax.set_title("Validation AP over epochs")
    ax.legend(fontsize=8)
    _recessive_axes(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- per-class / confusion / PR

def plot_per_class(run: dict[str, Any]):
    df = per_class_table(run)
    classes = list(dict.fromkeys(df["class"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, metric in zip(axes, ["AP50", "AR@300"]):
        splits = list(dict.fromkeys(df["split"]))
        w = 0.8 / len(splits)
        x = np.arange(len(classes))
        for k, s in enumerate(splits):
            vals = df[df["split"] == s].set_index("class").reindex(classes)[metric].to_numpy()
            bars = ax.bar(x + k * w, np.nan_to_num(vals), w * 0.9, color=SPLIT_COLORS[s], label=s)
            ax.bar_label(bars, labels=["n/a" if np.isnan(v) else f"{v:.2f}" for v in vals], fontsize=7)
        ax.set_xticks(x + w * (len(splits) - 1) / 2, classes, rotation=20)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{metric} per class", fontsize=10)
        ax.legend(fontsize=8)
        _recessive_axes(ax)
    fig.suptitle(run["name"], fontsize=11)
    fig.tight_layout()
    return fig


def plot_confusion(metrics: dict[str, Any], title: str = ""):
    cm = metrics["confusion_iou50"]
    m = np.array(cm["matrix"], dtype=float)
    labels = cm["labels"]
    row_sum = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, row_sum, out=np.zeros_like(m), where=row_sum > 0)
    norm[-1] = 0  # background row has no "total"; show counts only
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j] or (i < len(labels) - 1):
                txt = f"{int(m[i, j])}" + (f"\n{norm[i, j]:.0%}" if i < len(labels) - 1 else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "#222222")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("prediction (background = missed GT)")
    ax.set_ylabel("ground truth (background = false positive)")
    ax.set_title(title or "Confusion matrix @ IoU 0.5 (row %)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_pr_curves(metrics: dict[str, Any], title: str = ""):
    pr = metrics["pr_curve_iou50"] if "pr_curve_iou50" in metrics else metrics["coco"]["pr_curve_iou50"]
    rec = np.array(pr["recall"])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, prec in pr.items():
        if name == "recall":
            continue
        p = np.array(prec, dtype=float)
        if (p < 0).all():
            continue  # class absent from this split
        ap = metrics["coco"]["per_class"][name]["AP50"]
        ax.plot(rec, np.where(p < 0, np.nan, p), lw=2, color=CLASS_COLORS.get(name, "#333333"),
                label=f"{name} (AP50 {ap:.2f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title or "PR curves @ IoU 0.5", fontsize=10)
    ax.legend(fontsize=8)
    _recessive_axes(ax)
    fig.tight_layout()
    return fig


def plot_ap_by_density(runs: list[dict[str, Any]]):
    rows = [{"run": r["name"], "split": s, "group": g, "AP50": v.get("AP50", np.nan), "n": v["n_images"]}
            for r in runs for s, m in r["metrics"].items() for g, v in m["by_density"].items()]
    df = pd.DataFrame(rows)
    keys = list(dict.fromkeys(zip(df["run"], df["split"])))
    groups = list(dict.fromkeys(df["group"]))
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(keys)), 3.8))
    w = 0.8 / len(groups)
    x = np.arange(len(keys))
    for k, g in enumerate(groups):
        sub = df[df["group"] == g].set_index(["run", "split"]).reindex(keys)
        bars = ax.bar(x + k * w, np.nan_to_num(sub["AP50"].to_numpy()), w * 0.9, color=SERIES[k],
                      label=f"{g} cells/image")
        ax.bar_label(bars, labels=[f"{a:.2f}\n(n={int(n)})" if n else "n=0"
                                   for a, n in zip(sub["AP50"].to_numpy(), sub["n"].to_numpy())], fontsize=7)
    ax.set_xticks(x + w * (len(groups) - 1) / 2, [f"{r}\n{s}" for r, s in keys], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("AP50")
    ax.set_title("AP50 by number of cells per image")
    ax.legend(fontsize=8)
    _recessive_axes(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- examples

def select_examples(gt: dict[str, Any], dets: list[dict[str, Any]], thresholds: dict[str, float],
                    n: int = 6, iou_thr: float = 0.5, seed: int = 0) -> list[tuple[int, str]]:
    """Pick images: rare-class misses first, then rare-class false positives, then random.

    "Rare" = every class except the most frequent one in `gt`. Returns [(image_id, reason)].
    """
    names = {c["id"]: c["name"] for c in gt["categories"]}
    counts = pd.Series([a["category_id"] for a in gt["annotations"]]).value_counts()
    rare = set(counts.index[1:]) | (set(names) - set(counts.index))
    thr = {c: thresholds[names[c]] for c in names}
    kept = [d for d in dets if d["score"] >= thr[d["category_id"]]]
    misses, fps = [], []
    for im in gt["images"]:
        g = [a for a in gt["annotations"] if a["image_id"] == im["id"]]
        d = [x for x in kept if x["image_id"] == im["id"]]
        for c in rare:
            gb = np.array([a["bbox"] for a in g if a["category_id"] == c], float).reshape(-1, 4)
            db = np.array([x["bbox"] for x in d if x["category_id"] == c], float).reshape(-1, 4)
            iou = iou_matrix(gb, db)
            if len(gb) and (len(db) == 0 or (iou.max(axis=1) < iou_thr).any()):
                misses.append((im["id"], f"missed {names[c]}"))
            if len(db) and (len(gb) == 0 or (iou.max(axis=0) < iou_thr).any()):
                fps.append((im["id"], f"FP {names[c]}"))
    rng = np.random.default_rng(seed)
    chosen: list[tuple[int, str]] = []
    for pool, k in ((misses, n // 3 + n % 3), (fps, n // 3)):
        pool = [p for p in pool if p[0] not in {c[0] for c in chosen}]
        for i in rng.permutation(len(pool))[:k]:
            chosen.append(pool[i])
    rest = [im["id"] for im in gt["images"] if im["id"] not in {c[0] for c in chosen}]
    for i in rng.permutation(len(rest))[: n - len(chosen)]:
        chosen.append((rest[i], "random"))
    return chosen


def _draw(ax, img, boxes, title):
    ax.imshow(img)
    for name, (x, y, w, h), ls in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ls=ls, lw=0.6 if name in ("red blood cell", "uninfected") else 1.6,
                               edgecolor=CLASS_COLORS.get(name, "black")))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_examples(run: dict[str, Any], split: str, n: int = 6, seed: int = 0):
    """GT (left) vs thresholded predictions (right) for selected images of `split`."""
    gt = gt_for_run(run, split)
    m = run["metrics"][split]
    dets = run["predictions"][split]
    names = {c["id"]: c["name"] for c in gt["categories"]}
    chosen = select_examples(gt, dets, m["thresholds"], n=n, seed=seed)
    dcfg = load_config(resolve_path(run["config"]["data_config"]))
    images_dir = resolve_path(dcfg["paths"]["raw_root"]) / "images"
    info = {im["id"]: im for im in gt["images"]}
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 4.6 * len(chosen)))
    axes = np.atleast_2d(axes)
    for r, (img_id, reason) in enumerate(chosen):
        img = read_rgb(images_dir / info[img_id]["file_name"])
        g = [(names[a["category_id"]], a["bbox"], "-") for a in gt["annotations"] if a["image_id"] == img_id]
        d = [(names[x["category_id"]], x["bbox"], "-") for x in dets
             if x["image_id"] == img_id and x["score"] >= m["thresholds"][names[x["category_id"]]]]
        _draw(axes[r, 0], img, g, f"GT | {info[img_id]['file_name'][:8]} | {reason}")
        _draw(axes[r, 1], img, d, f"prediction (val thresholds) | {len(d)} boxes vs {len(g)} GT")
    handles = [Line2D([0], [0], color=CLASS_COLORS.get(nm, "black"), lw=2, label=nm) for nm in names.values()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig
