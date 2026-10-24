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


def mean_of_runs(points: dict[str, dict[str, float]], boots: dict[str, np.ndarray],
                 members: list[str]) -> tuple[dict[str, float], np.ndarray]:
    """Derived run whose metrics are the mean of `members` (e.g. training seeds): the point is the mean of their
    points, each resample the mean of their values on that SAME resample, so differences stay paired."""
    point = {m: float(np.mean([points[r][m] for r in members])) for m in METRICS}
    return point, np.mean([boots[r] for r in members], axis=0)


def fold_merge(by_fold: dict[int, dict[str, Any]], image_fold: dict[int, int]) -> dict[str, Any]:
    """Cross-fold run (phase 7): every image of fold k is scored with `by_fold[k]`, a run prepared on the whole split
    (each image keeps that run's own detections and its own val threshold). Matching is per image (COCOeval
    evalImgs and match_per_class), so the merge is exact. P / R / F1: each image's matches are cut at its run's
    threshold here, and the merged run is scored at -inf (prf_at only keeps scores >= thr)."""
    runs = list(by_fold.values())
    cat = runs[0]["cat"]
    if any(r["cat"] != cat or not np.array_equal(r["rec_thrs"], runs[0]["rec_thrs"]) for r in runs):
        raise ValueError("fold_merge: runs must share the class and the recall thresholds")
    ids = set(runs[0]["matches"])
    if set(image_fold) != ids or set(image_fold.values()) != set(by_fold):
        raise ValueError("fold_merge: the folds must cover exactly the split's images, one run per fold")
    per_img, matches = {}, {}
    for i in sorted(ids):
        r = by_fold[image_fold[i]]
        per_img[i] = r["per_img"][cat][i]
        m = r["matches"][i]
        keep = m["scores"] >= r["thr"]
        matches[i] = {"scores": m["scores"][keep], "tp": m["tp"][keep], "n_gt": m["n_gt"]}
    return {"per_img": {cat: per_img}, "rec_thrs": runs[0]["rec_thrs"], "matches": matches, "thr": -np.inf,
            "cat": cat}


def parse_fold_merges(items: list[str]) -> dict[str, tuple[str, str]]:
    """--fold-merge NAME=R_F1+R_F2 -> {NAME: (R_F1, R_F2)}; R_Fk = the member whose encoder saw test fold k."""
    out = {}
    for item in items:
        name, _, rhs = item.partition("=")
        members = rhs.split("+")
        if not name or name in out or len(members) != 2 or members[0] == members[1]:
            raise SystemExit(f"bad --fold-merge {item!r}: NAME=R_F1+R_F2 (R_Fk saw test fold k)")
        out[name] = (members[0], members[1])
    return out


def summarize_diff(point_a: float, point_b: float, boot_a: np.ndarray, boot_b: np.ndarray) -> dict[str, float]:
    """Point difference, 95% percentile CI and the share of resamples with difference <= 0."""
    d = boot_a - boot_b
    d = d[np.isfinite(d)]
    return {"diff": point_a - point_b, "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5)),
            "share_le_0": float(np.mean(d <= 0)), "n": int(d.size)}


def ci(boot: np.ndarray) -> tuple[float, float]:
    v = boot[np.isfinite(boot)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
