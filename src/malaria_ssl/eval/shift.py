"""Val -> test domain-shift diagnostics for a binary DETR run (reads saved predictions only).

For every GT box of the target class (infected): the best score of a target-class detection with
IoU >= iou_thr (0 if none), and the best score of an other-class detection on the same box (shows
whether a missed infected cell was found but called uninfected). Plus PR curves of the target class
from the greedy matching used for P/R/F1, with the val-selected operating point.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from malaria_ssl.data.viz import SPLIT_COLORS, _recessive_axes
from malaria_ssl.eval.detection_metrics import iou_matrix, match_per_class, prf_at


def gt_box_scores(gt: dict[str, Any], dets: list[dict[str, Any]], cat_id: int, iou_thr: float) -> dict[str, np.ndarray]:
    """Per GT box of `cat_id`: best same-class and best other-class detection score at IoU >= iou_thr."""
    dd = defaultdict(list)
    for x in dets:
        dd[x["image_id"]].append(x)
    same, other = [], []
    for a in gt["annotations"]:
        if a["category_id"] != cat_id:
            continue
        d = dd[a["image_id"]]
        if not d:
            same.append(0.0), other.append(0.0)
            continue
        ious = iou_matrix(np.array([x["bbox"] for x in d], float), np.array([a["bbox"]], float))[:, 0]
        scores = np.array([x["score"] for x in d])
        is_same = np.array([x["category_id"] == cat_id for x in d])
        ok = ious >= iou_thr
        same.append(float(scores[ok & is_same].max()) if (ok & is_same).any() else 0.0)
        other.append(float(scores[ok & ~is_same].max()) if (ok & ~is_same).any() else 0.0)
    return {"same": np.asarray(same), "other": np.asarray(other)}


def pr_curve(gt: dict[str, Any], dets: list[dict[str, Any]], cat_id: int, iou_thr: float) -> dict[str, np.ndarray]:
    """Precision / recall / score arrays of `cat_id` over all detections (greedy matching)."""
    per_img = match_per_class(gt, dets, iou_thr)[cat_id]
    s = np.concatenate([r["scores"] for r in per_img.values()])
    t = np.concatenate([r["tp"] for r in per_img.values()])
    n_gt = sum(r["n_gt"] for r in per_img.values())
    order = np.argsort(-s, kind="mergesort")
    s, t = s[order], t[order]
    tp, fp = np.cumsum(t), np.cumsum(~t)
    return {"precision": tp / np.maximum(tp + fp, 1), "recall": tp / max(n_gt, 1), "score": s, "n_gt": n_gt,
            "per_img": per_img}


def summarize(scores: dict[str, np.ndarray], thr: float) -> dict[str, float]:
    s, o = scores["same"], scores["other"]
    return {"n_gt": int(len(s)), "median_score": float(np.median(s)) if len(s) else float("nan"),
            "frac_ge_val_threshold": float(np.mean(s >= thr)),
            "frac_no_target_detection": float(np.mean(s == 0)),
            "frac_other_class_scores_higher": float(np.mean(o > s))}


def plot_shift(scores: dict[str, dict[str, np.ndarray]], curves: dict[str, dict[str, np.ndarray]], thr: float,
               cls: str, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    bins = np.linspace(0, 1, 41)
    for split, sc in scores.items():
        axes[0].hist(sc["same"], bins=bins, density=True, histtype="step", lw=2, color=SPLIT_COLORS[split],
                     label=f"{split} (n={len(sc['same'])})")
        x = np.sort(sc["same"])
        axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), lw=2, color=SPLIT_COLORS[split], label=split)
        c = curves[split]
        axes[2].plot(c["recall"], c["precision"], lw=2, color=SPLIT_COLORS[split], label=split)
        r = prf_at(c["per_img"], thr)
        axes[2].plot(r["recall"], r["precision"], "o", ms=8, color=SPLIT_COLORS[split],
                     label=f"{split} @ val thr: P {r['precision']:.2f} R {r['recall']:.2f}")
    for ax in axes[:2]:
        ax.axvline(thr, color="#555555", ls="--", lw=1, label=f"val threshold {thr:.3f}")
        ax.set_xlabel(f"best {cls} score at IoU >= 0.5 (0 = not detected as {cls})")
    axes[0].set_title(f"GT {cls} cells: score distribution", fontsize=10)
    axes[1].set_title(f"GT {cls} cells: ECDF of the score", fontsize=10)
    axes[2].set_title(f"PR curve, {cls} @ IoU 0.5", fontsize=10)
    axes[2].set_xlabel("recall"), axes[2].set_ylabel("precision")
    axes[2].set_xlim(0, 1), axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.legend(fontsize=8)
        _recessive_axes(ax)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
