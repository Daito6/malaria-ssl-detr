"""Boxes or classification? Class-agnostic localization quality of a DETR run (reads saved candidates only).

Every GT cell becomes one class; the detections are the two-stage candidates (every query, objectness =
1 - p(no-object), class-agnostic NMS 0.5: runs/<run>/candidates_<split>.json). One COCOeval at IoU 0.5 (greedy
by objectness, one detection per GT, maxDet 300, area "all") gives, per image:
- the per-image matching for the class-agnostic AP50 (re-accumulated per bootstrap resample, eval/bootstrap.py);
- the matched / total GT cells, overall and for the infected GT only (recall by the class-agnostic candidates,
  no score threshold).
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from malaria_ssl.eval.bootstrap import ap50_from_per_image, per_image_iou50
from malaria_ssl.eval.coco_metrics import run_cocoeval

AGNOSTIC_CAT = 1
METRICS = ("agnostic_AP50", "agnostic_recall_all", "agnostic_recall_infected")


def agnostic_gt(gt: dict[str, Any]) -> dict[str, Any]:
    """The same images and boxes with every annotation in one category."""
    g = copy.deepcopy(gt)
    g["categories"] = [{"id": AGNOSTIC_CAT, "name": "cell"}]
    for a in g["annotations"]:
        a["category_id"] = AGNOSTIC_CAT
    return g


def prepare_agnostic(gt: dict[str, Any], candidates: list[dict[str, Any]], infected_cat: int) -> dict[str, Any]:
    """Per-image class-agnostic matching of one run's candidates, extracted once and reused by every resample."""
    dets = [{"image_id": c["image_id"], "bbox": c["bbox"], "score": c["objectness"], "category_id": AGNOSTIC_CAT}
            for c in candidates]
    ev = run_cocoeval(agnostic_gt(gt), dets)
    p = ev.params
    t50 = int(np.argmin(np.abs(p.iouThrs - 0.5)))
    orig_cat = {a["id"]: a["category_id"] for a in gt["annotations"]}
    counts = {}
    for i, img in enumerate(p.imgIds):   # one category, area index 0 ("all")
        e = ev.evalImgs[i]
        if e is None:   # no GT and no detections
            counts[img] = (0, 0, 0, 0)
            continue
        valid = ~np.asarray(e["gtIgnore"]).astype(bool)
        matched = (np.asarray(e["gtMatches"])[t50] > 0) & valid
        infected = np.array([orig_cat[g] == infected_cat for g in e["gtIds"]], bool) & valid
        counts[img] = (int(matched.sum()), int(valid.sum()), int((matched & infected).sum()), int(infected.sum()))
    return {"per_img": per_image_iou50(ev), "rec_thrs": p.recThrs, "counts": counts}


def score_agnostic(prep: dict[str, Any], img_ids: list[int]) -> dict[str, float]:
    """Metrics over `img_ids` (duplicates allowed, as produced by resampling)."""
    ap = ap50_from_per_image(prep["per_img"], img_ids, prep["rec_thrs"])[AGNOSTIC_CAT]
    c = np.array([prep["counts"][i] for i in img_ids]).sum(axis=0)
    return {"agnostic_AP50": ap if ap >= 0 else float("nan"),
            "agnostic_recall_all": c[0] / c[1] if c[1] else float("nan"),
            "agnostic_recall_infected": c[2] / c[3] if c[3] else float("nan")}
