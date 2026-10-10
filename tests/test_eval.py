"""Evaluation module tests."""
import numpy as np
import pytest
import torch

from malaria_ssl.data.detection import load_split_coco
from malaria_ssl.eval.bootstrap import ap50_from_per_image, per_image_iou50
from malaria_ssl.eval.coco_metrics import evaluate_coco, run_cocoeval, summarize
from malaria_ssl.eval.detection_metrics import best_f1_threshold, confusion_matrix, match_per_class, prf_report
from malaria_ssl.eval.inference import outputs_to_detections
from malaria_ssl.utils.config import load_config, resolve_path

DATA = load_config(resolve_path("configs/data.yaml"))
HAVE_DATA = (resolve_path(DATA["paths"]["coco_dir"]) / "val.json").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="processed data not found")


def _gt_as_dets(gt):
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
            for a in gt["annotations"]]


@needs_data
@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_gt_as_predictions_gives_perfect_ap_ar_on_dense_images(mode):
    gt = load_split_coco("val", mode, DATA, resolve_path)
    per_img = np.bincount([a["image_id"] for a in gt["annotations"]])
    assert per_img.max() > 100  # the split contains dense images
    m = evaluate_coco(gt, _gt_as_dets(gt))
    assert m["AP"] == pytest.approx(1.0) and m["AP50"] == pytest.approx(1.0)
    assert m["AR@300"] == pytest.approx(1.0)
    for name, v in m["per_class"].items():
        assert v["AP"] == pytest.approx(1.0), name
    # the pycocotools default (100 dets/img) would truncate the dense images
    ev = run_cocoeval(gt, _gt_as_dets(gt), max_dets=[1, 10, 100])
    assert ev.stats[8] < 1.0


@needs_data
def test_gt_as_predictions_perfect_f1_and_diagonal_confusion():
    gt = load_split_coco("val", "multiclass", DATA, resolve_path)
    dets = _gt_as_dets(gt)
    matches = match_per_class(gt, dets, 0.5)
    thr = {c: best_f1_threshold(m)[0] for c, m in matches.items()}
    rep = prf_report(matches, thr, {c["id"]: c["name"] for c in gt["categories"]})
    assert all(v["f1"] == pytest.approx(1.0) for v in rep["per_class"].values())
    cm = np.array(confusion_matrix(gt, dets, thr, 0.5)["matrix"])
    assert cm.sum() == len(gt["annotations"]) and np.trace(cm[:-1, :-1]) == cm.sum()


@needs_data
def test_bootstrap_accumulation_matches_cocoeval():
    gt = load_split_coco("val", "multiclass", DATA, resolve_path)
    rng = np.random.default_rng(0)
    dets = []
    for a in gt["annotations"]:
        if rng.random() < 0.15:
            continue  # missed
        x, y, w, h = a["bbox"]
        j = rng.normal(0, 0.15, 4) * [w, h, w, h]
        cat = a["category_id"] if rng.random() > 0.1 else int(rng.integers(1, 8))
        dets.append({"image_id": a["image_id"], "category_id": cat,
                     "bbox": [x + j[0], y + j[1], max(w + j[2], 1), max(h + j[3], 1)], "score": float(rng.random())})
    for im in gt["images"]:  # random false positives
        for _ in range(5):
            dets.append({"image_id": im["id"], "category_id": int(rng.integers(1, 8)),
                         "bbox": [float(rng.uniform(0, 1400)), float(rng.uniform(0, 1000)), 100.0, 100.0],
                         "score": float(rng.random())})
    ev = run_cocoeval(gt, dets)
    ap = ap50_from_per_image(per_image_iou50(ev), list(ev.params.imgIds), ev.params.recThrs)
    prec = ev.eval["precision"][0, :, :, 0, -1]
    for k, c in enumerate(ev.params.catIds):
        ref = prec[:, k][prec[:, k] > -1].mean() if (prec[:, k] > -1).any() else -1.0
        assert ap[c] == pytest.approx(ref, abs=1e-12), c
    valid = [v for v in ap.values() if v >= 0]
    assert np.mean(valid) == pytest.approx(ev.stats[1], abs=1e-12)
    # our summary equals pycocotools' own where the latter is valid (default maxDets)
    ev100 = run_cocoeval(gt, dets, max_dets=[1, 10, 100])
    s = summarize(ev100, {c["id"]: c["name"] for c in gt["categories"]})
    assert [s["AP"], s["AP50"], s["AP75"], s["AR@1"], s["AR@10"], s["AR@100"]] == pytest.approx(
        [ev100.stats[i] for i in (0, 1, 2, 6, 7, 8)], abs=1e-12)


def test_outputs_mapped_back_to_original_pixels():
    # original 1600x1200 resized to 800x600 (scale 0.5); one box at original (400, 300, 200, 100)
    t = {"image_id": 5, "size": (600, 800), "scale": (0.5, 0.5), "orig_size": (1200, 1600)}
    cx, cy, w, h = (400 + 100) * 0.5 / 800, (300 + 50) * 0.5 / 600, 200 * 0.5 / 800, 100 * 0.5 / 600
    logits = torch.tensor([[[0.0, 5.0, -5.0]]])  # class 1 wins, last = no-object
    dets = outputs_to_detections(logits, torch.tensor([[[cx, cy, w, h]]]), [t], label_to_cat=[1, 2])
    assert dets[0]["category_id"] == 2 and dets[0]["image_id"] == 5
    assert dets[0]["bbox"] == pytest.approx([400, 300, 200, 100], abs=1e-3)
    assert dets[0]["score"] == pytest.approx(torch.softmax(logits[0, 0], -1)[1].item())
