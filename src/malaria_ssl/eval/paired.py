"""Paired bootstrap over images for detection runs evaluated on the same split (reads saved predictions).

Each resample draws image ids once and scores every run on that SAME multiset of images, so the
difference metric(a) - metric(b) keeps the image-level correlation between the two runs. Metrics per
class: AP50 (COCO 101-point, exact on the full set; see eval/bootstrap.py) and precision / recall / F1
at each run's own val-selected threshold.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from malaria_ssl.eval.bootstrap import ap50_from_per_image, per_image_iou50
from malaria_ssl.eval.coco_metrics import run_cocoeval
from malaria_ssl.eval.detection_metrics import match_per_class, prf_at

METRICS = ("AP50", "precision", "recall", "f1")


def prepare_run(gt: dict[str, Any], dets: list[dict[str, Any]], cat_id: int, thr: float,
                iou_thr: float = 0.5) -> dict[str, Any]:
    """Per-image matching of one run for `cat_id`, extracted once and reused by every resample."""
    ev = run_cocoeval(gt, dets)
    return {"per_img": {cat_id: per_image_iou50(ev)[cat_id]}, "rec_thrs": ev.params.recThrs,
            "matches": match_per_class(gt, dets, iou_thr)[cat_id], "thr": thr, "cat": cat_id}


def score(run: dict[str, Any], img_ids: list[int]) -> dict[str, float]:
    ap = ap50_from_per_image(run["per_img"], img_ids, run["rec_thrs"])[run["cat"]]
    r = prf_at(run["matches"], run["thr"], img_ids)
    return {"AP50": ap if ap >= 0 else float("nan"),
            "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}


def resamples(img_ids: list[int], n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [rng.choice(img_ids, size=len(img_ids), replace=True).tolist() for _ in range(n)]


def bootstrap_table(runs: dict[str, dict[str, Any]], samples: list[list[int]]) -> dict[str, np.ndarray]:
    """{run name: (n_resamples, len(METRICS)) array} on the shared resamples."""
    return {name: np.array([[score(r, ids)[m] for m in METRICS] for ids in samples]) for name, r in runs.items()}


def summarize_diff(point_a: float, point_b: float, boot_a: np.ndarray, boot_b: np.ndarray) -> dict[str, float]:
    """Point difference, 95% percentile CI and the share of resamples with difference <= 0."""
    d = boot_a - boot_b
    d = d[np.isfinite(d)]
    return {"diff": point_a - point_b, "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)),
            "share_le_0": float(np.mean(d <= 0)), "n": int(d.size)}


def ci(boot: np.ndarray) -> tuple[float, float]:
    v = boot[np.isfinite(boot)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
