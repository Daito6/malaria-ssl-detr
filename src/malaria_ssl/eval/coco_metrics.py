"""COCO-style AP/AR with pycocotools, using maxDets = [1, 10, 300].

The pycocotools default maxDets=100 silently truncates detections on images with >100 cells
(10.5% of BBBC041). Caveat: `COCOeval.summarize` hard-codes maxDets=100 for its first entry
(AP@[.5:.95]), which then returns -1 with custom maxDets. All numbers here are therefore computed
directly from `eval["precision"]` / `eval["recall"]` at the largest maxDet (300); `ev.stats` is
not used.
"""
from __future__ import annotations

import contextlib
import copy
import io
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = [1, 10, 300]


def coco_from_dict(d: dict[str, Any]) -> COCO:
    coco = COCO()
    coco.dataset = copy.deepcopy(d)
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def run_cocoeval(gt: dict[str, Any], dets: list[dict[str, Any]], img_ids: list[int] | None = None,
                 max_dets: list[int] = MAX_DETS) -> COCOeval:
    """Evaluate + accumulate + summarize (quietly); returns the COCOeval object."""
    gt_coco = coco_from_dict(gt)
    ids = sorted(img_ids) if img_ids is not None else sorted(gt_coco.getImgIds())
    keep = set(ids)
    dets = [d for d in dets if d["image_id"] in keep]
    with contextlib.redirect_stdout(io.StringIO()):
        dt_coco = gt_coco.loadRes(copy.deepcopy(dets)) if dets else COCO()
        if not dets:
            dt_coco.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
            dt_coco.createIndex()
        ev = COCOeval(gt_coco, dt_coco, iouType="bbox")
        ev.params.maxDets = list(max_dets)
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def _mean_valid(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(x.mean()) if x.size else float("nan")


def summarize(ev: COCOeval, class_names: dict[int, str]) -> dict[str, Any]:
    """Overall + per-class AP / AP50 / AP75 / AR@max and PR curves at IoU 0.5."""
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    rec = ev.eval["recall"]      # [T, K, A, M]
    t50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    t75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    md = ev.params.maxDets
    out: dict[str, Any] = {
        "AP": _mean_valid(prec[:, :, :, 0, -1]),
        "AP50": _mean_valid(prec[t50, :, :, 0, -1]),
        "AP75": _mean_valid(prec[t75, :, :, 0, -1]),
        **{f"AR@{m}": _mean_valid(rec[:, :, 0, j]) for j, m in enumerate(md)},
        "per_class": {}, "pr_curve_iou50": {"recall": ev.params.recThrs.tolist()},
    }
    for k, cat_id in enumerate(ev.params.catIds):
        name = class_names[cat_id]
        out["per_class"][name] = {
            "AP": _mean_valid(prec[:, :, k, 0, -1]),
            "AP50": _mean_valid(prec[t50, :, k, 0, -1]),
            "AP75": _mean_valid(prec[t75, :, k, 0, -1]),
            f"AR@{ev.params.maxDets[-1]}": _mean_valid(rec[:, k, 0, -1]),
            "n_gt": len(ev.cocoGt.getAnnIds(imgIds=ev.params.imgIds, catIds=[cat_id])),
        }
        out["pr_curve_iou50"][name] = prec[t50, :, k, 0, -1].tolist()
    return out


def evaluate_coco(gt: dict[str, Any], dets: list[dict[str, Any]], img_ids: list[int] | None = None) -> dict[str, Any]:
    names = {c["id"]: c["name"] for c in gt["categories"]}
    return summarize(run_cocoeval(gt, dets, img_ids), names)
