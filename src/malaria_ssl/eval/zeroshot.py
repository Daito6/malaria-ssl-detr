"""Zero-shot evaluation on an external domain (NIH-NLM): stitching tile predictions, patient-level bootstrap,
crop-level calibration and parasitemia error. Nothing here is fitted on the target domain."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torchvision.ops import nms

from malaria_ssl.data.nih import in_core, seam_nms
from malaria_ssl.eval.detection_metrics import ioa_matrix, iou_matrix


# --------------------------------------------------------------------------- stitching

def core_mask(q: dict[str, np.ndarray], tiles: list[dict[str, Any]], factor: float) -> np.ndarray:
    """Queries (boxes in ORIGINAL frame pixels, `tile` = index into `tiles`) whose centre lies in their tile's core
    (core coordinates are in the resized frame, hence `factor`)."""
    keep = np.zeros(len(q["tile"]), bool)
    for t in np.unique(q["tile"]):
        sel = np.where(q["tile"] == t)[0]
        keep[sel] = in_core(q["box"][sel] * factor, tiles[t]["core"])
    return keep


def end_to_end_detections(q: dict[str, np.ndarray], keep: np.ndarray, seam_iou: float,
                          label_to_cat: list[int]) -> dict[str, np.ndarray]:
    """End-to-end detections as on BBBC041 (every query; class = argmax of the real classes, score = its prob),
    after the core filter and the cross-tile seam NMS per image and class. Keeps p(class) of every detection."""
    idx = np.where(keep)[0]
    probs = q["probs"][idx, :-1]
    label = probs.argmax(1)
    score = probs.max(1)
    out = np.zeros(len(idx), bool)
    for img in np.unique(q["image_id"][idx]):
        for lab in range(probs.shape[1]):
            sel = np.where((q["image_id"][idx] == img) & (label == lab))[0]
            if len(sel):
                out[sel[seam_nms(q["box"][idx[sel]], score[sel], q["tile"][idx[sel]], seam_iou)]] = True
    idx, label, score = idx[out], label[out], score[out]
    return {"image_id": q["image_id"][idx], "box": q["box"][idx], "category_id": np.asarray(label_to_cat)[label],
            "score": score, "probs": q["probs"][idx, :-1]}


def candidates(q: dict[str, np.ndarray], keep: np.ndarray, nms_iou: float) -> dict[str, np.ndarray]:
    """Two-stage candidates: objectness = 1 - p(no-object), core filter, class-agnostic NMS per frame
    (as two_stage.nms_per_image), sorted by image then objectness."""
    idx = np.where(keep)[0]
    obj = 1 - q["probs"][idx, -1]
    out = []
    for img in np.unique(q["image_id"][idx]):
        sel = np.where(q["image_id"][idx] == img)[0]
        b = torch.as_tensor(q["box"][idx[sel]], dtype=torch.float32)
        k = nms(torch.cat([b[:, :2], b[:, :2] + b[:, 2:]], 1), torch.as_tensor(obj[sel], dtype=torch.float32), nms_iou)
        k = sel[k.numpy()]
        out.append(k[np.argsort(-obj[k], kind="mergesort")])
    k = np.concatenate(out) if out else np.zeros(0, int)
    return {"image_id": q["image_id"][idx[k]], "box": q["box"][idx[k]], "objectness": obj[k]}


def to_coco_dets(image_id: np.ndarray, box: np.ndarray, score: np.ndarray, cat) -> list[dict[str, Any]]:
    cats = np.broadcast_to(np.asarray(cat), score.shape)
    return [{"image_id": int(i), "category_id": int(c), "bbox": [float(v) for v in b], "score": float(s)}
            for i, b, s, c in zip(image_id, box, score, cats)]


# --------------------------------------------------------------------------- patient bootstrap

def patient_resamples(image_patient: dict[int, str], image_ids: list[int], n: int, seed: int) -> list[list[int]]:
    """Resample PATIENTS with replacement; each resample is the list of the drawn patients' images."""
    by_patient = defaultdict(list)
    for i in sorted(image_ids):
        by_patient[image_patient[i]].append(i)
    patients = sorted(by_patient)
    rng = np.random.default_rng(seed)
    return [[i for p in rng.choice(patients, size=len(patients), replace=True) for i in by_patient[p]]
            for _ in range(n)]


def rows_of_images(image_id: np.ndarray, sample: list[int]) -> np.ndarray:
    """Row indices of the items (crops) of every image in `sample`, duplicates repeated."""
    order = np.argsort(image_id, kind="mergesort")
    ids, start = np.unique(image_id[order], return_index=True)
    end = np.r_[start[1:], len(order)]
    pos = {int(i): (s, e) for i, s, e in zip(ids, start, end)}
    parts = [order[pos[i][0]:pos[i][1]] for i in sample if i in pos]
    return np.concatenate(parts) if parts else np.zeros(0, int)


# --------------------------------------------------------------------------- crop level

def ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Expected calibration error of p(infected) against the binary label, equal-width bins on [0, 1]."""
    if len(y) == 0:
        return float("nan")
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins)
    conf = np.bincount(b, weights=p, minlength=n_bins)
    acc = np.bincount(b, weights=y.astype(float), minlength=n_bins)
    nz = cnt > 0
    return float(np.abs(acc[nz] - conf[nz]).sum() / len(y))


def matched_p_infected(gt_image: np.ndarray, gt_box: np.ndarray, det_image: np.ndarray, det_box: np.ndarray,
                       det_p_inf: np.ndarray, iou_thr: float) -> np.ndarray:
    """End-to-end protocol for crop-level metrics: per GT box, p(infected) of the detection with the highest
    IoU (>= iou_thr) with it; 0 when no detection reaches iou_thr."""
    out = np.zeros(len(gt_box))
    det_rows = defaultdict(list)
    for j, i in enumerate(det_image):
        det_rows[int(i)].append(j)
    gt_rows = defaultdict(list)
    for j, i in enumerate(gt_image):
        gt_rows[int(i)].append(j)
    for img, g in gt_rows.items():
        d = det_rows.get(img, [])
        if not d:
            continue
        ious = iou_matrix(gt_box[g], det_box[d])
        best = ious.argmax(1)
        hit = ious[np.arange(len(g)), best] >= iou_thr
        out[np.asarray(g)[hit]] = det_p_inf[np.asarray(d)[best[hit]]]
    return out


# --------------------------------------------------------------------------- parasitemia

def counted(det_image: np.ndarray, det_box: np.ndarray, gt: dict[str, Any], iou_thr: float) -> np.ndarray:
    """Mask of predicted boxes that count in a parasitemia estimate: dropped when they cover >= iou_thr of their
    area with an ignore (ambiguous) region and match no labelled cell (IoU >= iou_thr), as in match_per_class."""
    cells, ign = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        (ign if a.get("iscrowd", 0) else cells)[a["image_id"]].append(a["bbox"])
    keep = np.ones(len(det_box), bool)
    rows = defaultdict(list)
    for j, i in enumerate(det_image):
        rows[int(i)].append(j)
    for img, r in rows.items():
        if not ign.get(img):
            continue
        b = det_box[r]
        hit_ign = (ioa_matrix(b, np.asarray(ign[img], float)) >= iou_thr).any(1)
        c = np.asarray(cells.get(img, []), float).reshape(-1, 4)
        hit_cell = (iou_matrix(b, c) >= iou_thr).any(1) if len(c) else np.zeros(len(r), bool)
        keep[np.asarray(r)[hit_ign & ~hit_cell]] = False
    return keep


def parasitemia_estimate(image_ids: list[int], inf_image: np.ndarray, cell_image: np.ndarray) -> dict[int, float]:
    """Per image: (# infected predictions) / (# cell predictions); 0 when no cell is predicted."""
    n_inf = defaultdict(int, zip(*np.unique(inf_image, return_counts=True))) if len(inf_image) else defaultdict(int)
    n_cell = defaultdict(int, zip(*np.unique(cell_image, return_counts=True))) if len(cell_image) else defaultdict(int)
    return {i: (n_inf[i] / n_cell[i] if n_cell[i] else 0.0) for i in image_ids}


def parasitemia_errors(est: dict[int, float], true: dict[int, float], sample: list[int]) -> dict[str, float]:
    """Mean |estimated - true| parasitemia and mean signed error over the images of `sample`."""
    d = np.array([est[i] - true[i] for i in sample])
    return {"parasitemia_mae": float(np.abs(d).mean()), "parasitemia_bias": float(d.mean())}
