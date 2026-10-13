import numpy as np
import pytest
import torch

from malaria_ssl.eval.inference import outputs_to_candidates
from malaria_ssl.eval.two_stage import candidate_crops, fit_p_infected, nms_per_image, scored_detections


def test_candidates_objectness_and_boxes():
    logits = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])  # 2 queries; last column = no-object
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1]]])
    t = {"image_id": 7, "size": (100, 200), "scale": (0.5, 0.5), "orig_size": (200, 400)}
    c = outputs_to_candidates(logits, boxes, [t])
    p_bg = torch.tensor([0.0, 0.0, 5.0]).softmax(0)[-1].item()
    assert np.isclose(c[1]["objectness"], 1 - p_bg)
    assert np.isclose(c[0]["objectness"], 1 - torch.tensor([2.0, 0.0, 0.0]).softmax(0)[-1].item())
    assert np.allclose(c[0]["bbox"], [160, 60, 80, 80])  # cxcywh of the 200x100 resized image -> x2 original


def test_nms_keeps_best_of_overlapping_and_all_images():
    cands = [{"image_id": 1, "bbox": [0, 0, 10, 10], "objectness": 0.6},
             {"image_id": 1, "bbox": [1, 1, 10, 10], "objectness": 0.9},
             {"image_id": 1, "bbox": [50, 50, 10, 10], "objectness": 0.2},
             {"image_id": 2, "bbox": [0, 0, 10, 10], "objectness": 0.3}]
    kept = nms_per_image(cands, 0.5)
    assert [(c["image_id"], c["objectness"]) for c in kept] == [(1, 0.9), (1, 0.2), (2, 0.3)]


def test_candidate_crops_scale_and_order():
    index = {"images": [{"id": 5, "width": 1600, "height": 1200}, {"id": 3, "width": 1600, "height": 1200}]}
    cands = [{"image_id": 3, "bbox": [160, 160, 80, 40], "objectness": 1.0},
             {"image_id": 5, "bbox": [0, 0, 16, 16], "objectness": 1.0}]
    crops = candidate_crops(cands, index, 800)
    assert list(crops["order"]) == [1, 0] and list(crops["row"]) == [0, 1]
    assert np.allclose(crops["box"][1], [80, 80, 120, 100])  # x0, y0, x1, y1 at long side 800


def test_scores_and_classifier():
    cands = [{"image_id": 1, "bbox": [0, 0, 1, 1], "objectness": 0.5}]
    assert scored_detections(cands, np.array([0.4]), "det_x_p", 2)[0]["score"] == pytest.approx(0.2)
    assert scored_detections(cands, np.array([0.4]), "p", 2)[0]["score"] == pytest.approx(0.4)
    x = torch.cat([torch.randn(50, 4) + 3, torch.randn(50, 4) - 3])
    y = torch.cat([torch.ones(50), torch.zeros(50)])
    p, = fit_p_infected(x, y, [torch.tensor([[3.0] * 4, [-3.0] * 4])], 1e-3, 100)
    assert p[0] > 0.9 and p[1] < 0.1
