"""Bootstrap 95% CIs over images for AP50 and F1.

AP50: the per-image matching from COCOeval.evaluate (IoU 0.5, area "all", maxDet = max) is
extracted once; each resample only re-accumulates it with COCO's 101-point interpolation, so a
resample costs milliseconds instead of a full COCOeval run. `ap50_from_per_image` on the original
image set reproduces COCOeval's AP50 exactly (tested).
F1: per-image TP/FP/FN at the fixed per-class thresholds, summed over the resampled images.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools.cocoeval import COCOeval

from malaria_ssl.eval.detection_metrics import prf_at


def per_image_iou50(ev: COCOeval) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """{cat_id: {img_id: {"scores", "tp", "ignore", "n_gt"}}} at IoU 0.5 from COCOeval.evalImgs."""
    p = ev.params
    t50 = int(np.argmin(np.abs(p.iouThrs - 0.5)))
    n_area, n_img = len(p.areaRng), len(p.imgIds)
    out: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for k, c in enumerate(p.catIds):
        out[c] = {}
        for i, img in enumerate(p.imgIds):
            e = ev.evalImgs[k * n_area * n_img + 0 * n_img + i]  # area index 0 = "all"
            if e is None:
                out[c][img] = {"scores": np.zeros(0), "tp": np.zeros(0, bool), "ignore": np.zeros(0, bool),
                               "n_gt": 0}
                continue
            dt_ig = np.asarray(e["dtIgnore"])[t50].astype(bool)
            out[c][img] = {"scores": np.asarray(e["dtScores"], float),
                           "tp": np.asarray(e["dtMatches"])[t50] > 0,
                           "ignore": dt_ig,
                           "n_gt": int((~np.asarray(e["gtIgnore"]).astype(bool)).sum())}
    return out


def _ap_101(scores: np.ndarray, tp: np.ndarray, n_gt: int, rec_thrs: np.ndarray) -> float:
    """COCOeval.accumulate for a single (class, IoU, area, maxDet) cell."""
    if n_gt == 0:
        return -1.0
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order]
    tps = np.cumsum(tp).astype(float)
    fps = np.cumsum(~tp).astype(float)
    rc = tps / n_gt
    pr = tps / (fps + tps + np.spacing(1))
    q = np.zeros(len(rec_thrs))
    if len(pr):
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, rec_thrs, side="left")
        valid = inds < len(pr)
        q[valid] = pr[inds[valid]]
    return float(q.mean())


def ap50_from_per_image(per_img: dict[int, dict[int, dict[str, np.ndarray]]], img_ids: list[int],
                        rec_thrs: np.ndarray) -> dict[int, float]:
    """Per-class AP50 over `img_ids` (duplicates allowed, as produced by resampling)."""
    out = {}
    for c, d in per_img.items():
        rows = [d[i] for i in img_ids]
        s = np.concatenate([r["scores"][~r["ignore"]] for r in rows]) if rows else np.zeros(0)
        t = np.concatenate([r["tp"][~r["ignore"]] for r in rows]) if rows else np.zeros(0, bool)
        out[c] = _ap_101(s, t, sum(r["n_gt"] for r in rows), rec_thrs)
    return out


def _ci(values: list[float]) -> dict[str, float]:
    v = np.asarray([x for x in values if np.isfinite(x)])
    if v.size == 0:
        return {"low": float("nan"), "high": float("nan"), "n": 0}
    return {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5)), "n": int(v.size)}


def bootstrap_ci(ev: COCOeval, matches: dict[int, dict[int, dict[str, Any]]], thresholds: dict[int, float],
                 class_names: dict[int, str], n_resamples: int, seed: int) -> dict[str, Any]:
    """95% percentile CIs for mean AP50, per-class AP50, per-class F1 and macro F1."""
    rng = np.random.default_rng(seed)
    per_img = per_image_iou50(ev)
    img_ids = list(ev.params.imgIds)
    rec = ev.params.recThrs
    samples: dict[str, list[float]] = {"mAP50": [], "macro_f1": []}
    for c in per_img:
        samples[f"AP50/{class_names[c]}"] = []
        samples[f"F1/{class_names[c]}"] = []
    for _ in range(n_resamples):
        ids = rng.choice(img_ids, size=len(img_ids), replace=True).tolist()
        ap = ap50_from_per_image(per_img, ids, rec)
        valid = [v for v in ap.values() if v >= 0]
        samples["mAP50"].append(float(np.mean(valid)) if valid else float("nan"))
        f1s = []
        for c in per_img:
            samples[f"AP50/{class_names[c]}"].append(ap[c] if ap[c] >= 0 else float("nan"))
            r = prf_at(matches[c], thresholds[c], ids)
            samples[f"F1/{class_names[c]}"].append(r["f1"] if r["n_gt"] else float("nan"))
            if r["n_gt"]:
                f1s.append(r["f1"])
        samples["macro_f1"].append(float(np.mean(f1s)) if f1s else float("nan"))
    return {"n_resamples": n_resamples, "seed": seed, "ci95": {k: _ci(v) for k, v in samples.items()}}
