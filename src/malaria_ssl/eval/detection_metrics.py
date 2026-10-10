"""Detection-level precision / recall / F1 at a fixed IoU and confusion matrix.

Per-class greedy matching: detections of class c, sorted by score, are matched to the unmatched GT
box of class c with the highest IoU >= iou_thr. Score thresholds are chosen per class on val
(max F1) and then applied unchanged to test.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xywh boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def _group(gt: dict[str, Any], dets: list[dict[str, Any]]):
    g = defaultdict(list)
    for a in gt["annotations"]:
        g[(a["image_id"], a["category_id"])].append(a["bbox"])
    d = defaultdict(list)
    for x in dets:
        d[(x["image_id"], x["category_id"])].append((x["score"], x["bbox"]))
    return g, d


def match_per_class(gt: dict[str, Any], dets: list[dict[str, Any]], iou_thr: float
                    ) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """{cat_id: {image_id: {"scores": (D,), "tp": (D,) bool, "n_gt": int}}} for every GT image."""
    g, d = _group(gt, dets)
    img_ids = [im["id"] for im in gt["images"]]
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for c in (cat["id"] for cat in gt["categories"]):
        per_img = {}
        for i in img_ids:
            gboxes = np.asarray(g.get((i, c), []), dtype=float).reshape(-1, 4)
            dd = sorted(d.get((i, c), []), key=lambda x: -x[0])
            scores = np.array([s for s, _ in dd], dtype=float)
            dboxes = np.asarray([b for _, b in dd], dtype=float).reshape(-1, 4)
            tp = np.zeros(len(dd), dtype=bool)
            if len(dd) and len(gboxes):
                ious = iou_matrix(dboxes, gboxes)
                taken = np.zeros(len(gboxes), dtype=bool)
                for j in range(len(dd)):
                    cand = np.where(~taken, ious[j], -1.0)
                    k = int(cand.argmax())
                    if cand[k] >= iou_thr:
                        tp[j], taken[k] = True, True
            per_img[i] = {"scores": scores, "tp": tp, "n_gt": len(gboxes)}
        out[c] = per_img
    return out


def prf_at(per_img: dict[int, dict[str, Any]], thr: float, img_ids=None) -> dict[str, float]:
    ids = per_img.keys() if img_ids is None else img_ids
    tp = fp = n_gt = 0
    for i in ids:
        r = per_img[i]
        keep = r["scores"] >= thr
        t = int(r["tp"][keep].sum())
        tp += t
        fp += int(keep.sum()) - t
        n_gt += r["n_gt"]
    fn = n_gt - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    rcl = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * rcl / (p + rcl) if n_gt and p + rcl > 0 else 0.0
    return {"precision": p, "recall": rcl, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt}


def best_f1_threshold(per_img: dict[int, dict[str, Any]]) -> tuple[float, float]:
    """Score threshold maximizing F1 (over all observed scores). Returns (threshold, f1)."""
    scores = np.concatenate([r["scores"] for r in per_img.values()]) if per_img else np.zeros(0)
    tps = np.concatenate([r["tp"] for r in per_img.values()]) if per_img else np.zeros(0, bool)
    n_gt = sum(r["n_gt"] for r in per_img.values())
    if n_gt == 0 or scores.size == 0:
        return 1.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], tps[order]
    ctp = np.cumsum(t)
    cfp = np.cumsum(~t)
    f1 = 2 * ctp / (ctp + cfp + n_gt)  # = 2PR/(P+R)
    # only cut between distinct scores
    last_of_value = np.r_[s[1:] != s[:-1], True]
    f1 = np.where(last_of_value, f1, -1)
    k = int(f1.argmax())
    return float(s[k]), float(f1[k])


def choose_thresholds(matches: dict[int, dict[int, dict[str, Any]]]) -> dict[int, float]:
    return {c: best_f1_threshold(per_img)[0] for c, per_img in matches.items()}


def prf_report(matches, thresholds: dict[int, float], class_names: dict[int, str], img_ids=None
               ) -> dict[str, Any]:
    per = {class_names[c]: {"threshold": thresholds[c], **prf_at(m, thresholds[c], img_ids)}
           for c, m in matches.items()}
    valid = [v["f1"] for v in per.values() if v["n_gt"] > 0]
    return {"per_class": per, "macro_f1": float(np.mean(valid)) if valid else float("nan")}


def confusion_matrix(gt: dict[str, Any], dets: list[dict[str, Any]], thresholds: dict[int, float],
                     iou_thr: float) -> dict[str, Any]:
    """Class-agnostic greedy matching of thresholded detections to GT at IoU >= iou_thr.

    Rows = GT class + "background" (unmatched detections = false positives);
    columns = predicted class + "background" (unmatched GT = missed).
    """
    cats = [c["id"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    names = {c["id"]: c["name"] for c in gt["categories"]}
    idx = {c: i for i, c in enumerate(cats)}
    bg = len(cats)
    m = np.zeros((bg + 1, bg + 1), dtype=int)
    gts = defaultdict(list)
    for a in gt["annotations"]:
        gts[a["image_id"]].append(a)
    dd = defaultdict(list)
    for x in dets:
        if x["score"] >= thresholds[x["category_id"]]:
            dd[x["image_id"]].append(x)
    for im in gt["images"]:
        g = gts[im["id"]]
        d = sorted(dd[im["id"]], key=lambda x: -x["score"])
        ious = iou_matrix(np.array([x["bbox"] for x in d], float).reshape(-1, 4),
                          np.array([a["bbox"] for a in g], float).reshape(-1, 4))
        taken = np.zeros(len(g), dtype=bool)
        for j, x in enumerate(d):
            if len(g):
                cand = np.where(~taken, ious[j], -1.0)
                k = int(cand.argmax())
                if cand[k] >= iou_thr:
                    taken[k] = True
                    m[idx[g[k]["category_id"]], idx[x["category_id"]]] += 1
                    continue
            m[bg, idx[x["category_id"]]] += 1
        for k in np.where(~taken)[0]:
            m[idx[g[k]["category_id"]], bg] += 1
    labels = [names[c] for c in cats] + ["background"]
    return {"labels": labels, "matrix": m.tolist(), "rows": "ground truth", "cols": "prediction"}
