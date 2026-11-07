"""Parity of app/inference.py with the evaluation pipeline (evaluate.py) on one val image.

Part of every app test run on purpose: if malaria_ssl (e.g. detr/build.py, changed by later phases) stops matching,
these tests fail instead of the app silently drifting.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from torchvision.ops import box_iou

from app.crop_classifier import MIN_FEATURE_COS, TOL_AP, TOL_P
from app.tests.conftest import VAL_IMAGE_ID
from malaria_ssl.data.detection import DetectionDataset, build_transforms, collate_detr, load_split_coco, read_rgb
from malaria_ssl.eval.inference import predict
from malaria_ssl.utils.config import load_config, resolve_path

DETR_MODELS = ["a_coco_detr_binary", "b_vit_imagenet", "b_vit_joint"]


def _xyxy(boxes) -> torch.Tensor:
    b = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return torch.cat([b[:, :2], b[:, :2] + b[:, 2:]], dim=1)


@pytest.fixture(scope="module")
def val_gt():
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gt = load_split_coco("val", "binary", dcfg, resolve_path)
    return {
        **gt,
        "images": [im for im in gt["images"] if im["id"] == VAL_IMAGE_ID],
        "annotations": [a for a in gt["annotations"] if a["image_id"] == VAL_IMAGE_ID],
    }, dcfg


@pytest.mark.parametrize("model_id", DETR_MODELS)
def test_detr_matches_evaluate_pipeline(pool, val_gt, val_image_path, model_id):
    """Same weights, same device: DetectionDataset + collate_detr + predict (evaluate.py) == the app, query by query."""
    gt, dcfg = val_gt
    pred = pool.get(model_id)
    cfg = yaml.safe_load(pool.entry(model_id).files["config"].read_text("utf-8"))
    ds = DetectionDataset(
        gt, resolve_path(dcfg["paths"]["raw_root"]) / "images", build_transforms(cfg["augment"], train=False)
    )
    assert ds.class_names == pred.class_names
    ref = predict(pred.model, [collate_detr([ds[0]])], torch.device("cpu"), ds.label_to_cat, None)
    app = pred.detections(read_rgb(val_image_path))
    assert len(app) == len(ref) == cfg["model"]["num_queries"]
    names = dict(zip(ds.label_to_cat, ds.class_names, strict=True))
    assert [d["label"] for d in app] == [names[d["category_id"]] for d in ref]
    np.testing.assert_allclose([d["score"] for d in app], [d["score"] for d in ref], atol=1e-6)
    np.testing.assert_allclose([d["bbox"] for d in app], [d["bbox"] for d in ref], atol=1e-3)


@pytest.mark.parametrize("model_id", DETR_MODELS)
def test_detr_close_to_saved_val_predictions(pool, val_image_path, model_id):
    """CPU fp32 vs the saved predictions_val.json (CUDA bf16): per query they may differ (bf16 can swap which of two
    near-duplicate queries owns a cell), so compare the confident boxes at the val thresholds as sets."""
    entry = pool.entry(model_id)
    thr = entry.thresholds
    saved = [
        d
        for d in json.loads((entry.files["config"].parent / "predictions_val.json").read_text("utf-8"))
        if d["image_id"] == VAL_IMAGE_ID
    ]
    cat_name = {1: "uninfected", 2: "infected"}
    s = [(cat_name[d["category_id"]], d["bbox"]) for d in saved if d["score"] >= thr[cat_name[d["category_id"]]]]
    a = [
        (d["label"], d["bbox"])
        for d in pool.get(model_id).detections(read_rgb(val_image_path))
        if d["score"] >= thr[d["label"]]
    ]
    for label in ("uninfected", "infected"):
        sb = [b for lab, b in s if lab == label]
        ab = [b for lab, b in a if lab == label]
        assert abs(len(sb) - len(ab)) <= max(2, 0.05 * len(sb)), (label, len(sb), len(ab))
        if sb:
            matched = (box_iou(_xyxy(sb), _xyxy(ab)).max(1).values >= 0.7).float().mean().item() if ab else 0.0
            assert matched >= 0.95, (label, matched)


def test_two_stage_classifier_verified(pool):
    """The cached classifier passed its checks (refit == linear_predict, recorded val AP50 / threshold, features)."""
    meta = pool.get("two_stage_dino").classifier["meta"]
    c = meta["checks"]
    assert c["p_max_abs_diff"] <= TOL_P
    assert abs(c["val_ap50"] - c["val_ap50_reference"]) <= TOL_AP
    assert abs(c["val_threshold"] - c["val_threshold_reference"]) <= TOL_AP
    assert c["feature_cos_min"] >= MIN_FEATURE_COS


def test_two_stage_features_match_saved_now(pool, val_image_path):
    """Recomputed in every run (the cache only stores the result of its build): app crop features of the val
    image's saved candidates vs the saved features."""
    pred = pool.get("two_stage_dino")
    entry = pool.entry("two_stage_dino")
    cands = json.loads(entry.files["reference_candidates"].read_text("utf-8"))["candidates"]
    feats = torch.load(entry.files["reference_features"], map_location="cpu", weights_only=False)["features"]
    idx = [i for i, c in enumerate(cands) if c["image_id"] == VAL_IMAGE_ID]
    f = pred.features(read_rgb(val_image_path), [cands[i] for i in idx])
    cos = F.cosine_similarity(f, feats[idx].float(), dim=1)
    assert cos.min().item() >= MIN_FEATURE_COS
