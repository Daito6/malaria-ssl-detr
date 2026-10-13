"""Two-stage diagnostic: DETR localizes (class-agnostic candidates), a frozen-encoder crop classifier classifies.

Candidates: every DETR query with objectness = 1 - p(no-object), class-agnostic NMS per image. Crops: the probe
window (224 at long side `long_side`, mean of the patch tokens inside the box), see ssl/probe.py. Classifier:
the probe's class-balanced linear model, binary infected vs uninfected, fitted on TRAIN GT crops only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torchvision.ops import nms

from malaria_ssl.ssl.probe import linear_predict

VARIANTS = ("det_x_p", "p")  # infected score = objectness * p(infected) | p(infected)


def nms_per_image(cands: list[dict[str, Any]], iou: float) -> list[dict[str, Any]]:
    """Class-agnostic NMS per image on objectness; keeps the input order of the survivors' images."""
    by_img = defaultdict(list)
    for c in cands:
        by_img[c["image_id"]].append(c)
    out = []
    for img in sorted(by_img):
        cs = by_img[img]
        xywh = torch.tensor([c["bbox"] for c in cs], dtype=torch.float32)
        xyxy = torch.cat([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], dim=1)
        keep = nms(xyxy, torch.tensor([c["objectness"] for c in cs]), iou)
        out.extend(cs[i] for i in sorted(keep.tolist(), key=lambda i: -cs[i]["objectness"]))
    return out


def candidate_crops(cands: list[dict[str, Any]], cache_index: dict[str, Any], long_side: int) -> dict[str, np.ndarray]:
    """Probe crop set for the candidates (same keys as ssl.probe.build_crop_set; label unused = 0),
    grouped by image in cache-row order so every image is decoded once."""
    row_of = {im["id"]: i for i, im in enumerate(cache_index["images"])}
    size_of = {im["id"]: (im["width"], im["height"]) for im in cache_index["images"]}
    order = sorted(range(len(cands)), key=lambda i: row_of[cands[i]["image_id"]])
    boxes, rows, img_ids = [], [], []
    for i in order:
        c = cands[i]
        w0, h0 = size_of[c["image_id"]]
        s = long_side / max(w0, h0)
        x, y, w, h = c["bbox"]
        boxes.append([x * s, y * s, (x + w) * s, (y + h) * s])
        rows.append(row_of[c["image_id"]])
        img_ids.append(c["image_id"])
    return {"box": np.asarray(boxes, np.float32), "row": np.asarray(rows), "image_id": np.asarray(img_ids),
            "label": np.zeros(len(order), np.int64), "order": np.asarray(order)}


def fit_p_infected(train_x: torch.Tensor, train_infected: torch.Tensor, eval_x: list[torch.Tensor],
                   l2: float, max_iter: int) -> list[np.ndarray]:
    """p(infected) for each eval set from the probe's linear model on 2 classes (0 uninfected, 1 infected)."""
    sizes = [len(x) for x in eval_x]
    p = linear_predict(train_x.float(), train_infected.long(), torch.cat(eval_x).float(), 2, l2, max_iter)[:, 1]
    return [a.numpy() for a in torch.split(p.cpu(), sizes)]


def scored_detections(cands: list[dict[str, Any]], p_infected: np.ndarray, variant: str,
                      infected_cat: int) -> list[dict[str, Any]]:
    """Infected-class COCO detections, one per candidate (p_infected aligned with `cands`)."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    return [{"image_id": c["image_id"], "category_id": infected_cat, "bbox": c["bbox"],
             "score": float(c["objectness"] * p if variant == "det_x_p" else p)}
            for c, p in zip(cands, p_infected)]
