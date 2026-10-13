"""Domain-shift diagnostic tests on a toy binary GT."""
import numpy as np

from malaria_ssl.eval.shift import gt_box_scores, pr_curve, summarize

GT = {"images": [{"id": 1}], "categories": [{"id": 1, "name": "uninfected"}, {"id": 2, "name": "infected"}],
      "annotations": [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10]},
                      {"image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10]},
                      {"image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10]}]}
DETS = [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.8},   # infected called uninfected
        {"image_id": 1, "category_id": 2, "bbox": [200, 200, 10, 10], "score": 0.95}]  # false positive


def test_gt_box_scores_and_summary():
    s = gt_box_scores(GT, DETS, cat_id=2, iou_thr=0.5)
    assert np.allclose(s["same"], [0.9, 0.0]) and np.allclose(s["other"], [0.0, 0.8])
    m = summarize(s, thr=0.5)
    assert m["n_gt"] == 2 and m["frac_ge_val_threshold"] == 0.5
    assert m["frac_no_target_detection"] == 0.5 and m["frac_other_class_scores_higher"] == 0.5


def test_pr_curve_order_and_values():
    c = pr_curve(GT, DETS, cat_id=2, iou_thr=0.5)
    assert np.allclose(c["score"], [0.95, 0.9])
    assert np.allclose(c["precision"], [0.0, 0.5]) and np.allclose(c["recall"], [0.0, 0.5])


def test_summary_outcomes_partition():
    scores = {"same": np.array([0.0, 0.0, 0.3, 0.9, 0.5]), "other": np.array([0.0, 0.8, 0.6, 0.2, 0.5])}
    s = summarize(scores, thr=0.5)
    assert s["frac_no_detection_any_class"] == 0.2      # box 0
    assert s["frac_other_class_scores_higher"] == 0.4   # boxes 1, 2
    assert s["frac_target_scores_highest"] == 0.4       # boxes 3, 4 (tie goes to the target)
    total = s["frac_no_detection_any_class"] + s["frac_other_class_scores_higher"] + s["frac_target_scores_highest"]
    assert np.isclose(total, 1.0)
