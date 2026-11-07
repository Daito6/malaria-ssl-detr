"""The backend returns NMS survivors, so a client-side score threshold can never show the same cell twice."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision.ops import box_iou

from app.inference import NMS_IOU, class_agnostic_nms
from malaria_ssl.data.detection import read_rgb

THRESHOLDS = np.linspace(0, 1, 21)


def _xyxy(dets) -> torch.Tensor:
    b = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32).reshape(-1, 4)
    return torch.cat([b[:, :2], b[:, :2] + b[:, 2:]], dim=1)


def _max_pair_iou(dets) -> float:
    if len(dets) < 2:
        return 0.0
    iou = box_iou(_xyxy(dets), _xyxy(dets))
    iou.fill_diagonal_(0)
    return float(iou.max())


def _visible(dets, t: float):
    """What the client shows at threshold t (frontend/src/lib/filter.ts applies the same rule)."""
    return [d for d in dets if d["score"] >= t]


def test_nms_removes_cross_class_duplicates():
    dets = [
        {"label": "uninfected", "score": 0.90, "bbox": [10, 10, 50, 50]},
        {"label": "infected", "score": 0.95, "bbox": [12, 11, 50, 49]},  # same cell, other class
        {"label": "uninfected", "score": 0.40, "bbox": [11, 10, 48, 50]},  # same cell, low score
        {"label": "uninfected", "score": 0.80, "bbox": [200, 200, 50, 50]},
    ]
    out = class_agnostic_nms(dets)
    assert [d["score"] for d in out] == [0.95, 0.80]
    assert out[0]["label"] == "infected"
    for t in THRESHOLDS:
        assert _max_pair_iou(_visible(out, t)) <= NMS_IOU


def test_nms_empty():
    assert class_agnostic_nms([]) == []


@pytest.mark.parametrize("model_id", ["a_coco_detr_binary", "b_vit_imagenet", "two_stage_dino"])
def test_model_output_has_no_duplicates_at_any_threshold(pool, val_image_path, model_id):
    result = pool.run(model_id, read_rgb(val_image_path))
    dets = result.detections
    assert dets, "no detections on a val image"
    assert [d["score"] for d in dets] == sorted((d["score"] for d in dets), reverse=True)
    assert _max_pair_iou(dets) <= result.nms_iou + 1e-6
    prev = None
    for t in THRESHOLDS:  # raising the threshold only removes boxes, it never adds or moves one
        vis = {tuple(d["bbox"]) for d in _visible(dets, t)}
        if prev is not None:
            assert vis <= prev
        prev = vis
