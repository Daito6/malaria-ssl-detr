import numpy as np

from malaria_ssl.eval.box_vs_cls import agnostic_gt, prepare_agnostic, score_agnostic

INF, UNINF = 2, 1
# image 1: one infected + one uninfected cell; image 2: two uninfected; image 3: one infected
GT = {"images": [{"id": i, "width": 100, "height": 100} for i in (1, 2, 3)],
      "categories": [{"id": UNINF, "name": "uninfected"}, {"id": INF, "name": "infected"}],
      "annotations": [{"id": k, "image_id": img, "category_id": c, "bbox": b, "area": b[2] * b[3], "iscrowd": 0}
                      for k, (img, c, b) in enumerate([(1, INF, [10, 10, 20, 20]), (1, UNINF, [50, 50, 20, 20]),
                                                       (2, UNINF, [10, 10, 20, 20]), (2, UNINF, [50, 50, 20, 20]),
                                                       (3, INF, [30, 30, 20, 20])], start=1)]}


def _cand(img, bbox, obj):
    return {"image_id": img, "bbox": bbox, "objectness": obj}


def test_agnostic_gt_merges_classes_and_keeps_the_input():
    g = agnostic_gt(GT)
    assert {a["category_id"] for a in g["annotations"]} == {1} and len(g["categories"]) == 1
    assert {a["category_id"] for a in GT["annotations"]} == {UNINF, INF}


def test_perfect_candidates():
    cands = [_cand(a["image_id"], a["bbox"], 0.9) for a in GT["annotations"]]
    s = score_agnostic(prepare_agnostic(GT, cands, INF), [1, 2, 3])
    assert np.isclose(s["agnostic_AP50"], 1.0)
    assert s["agnostic_recall_all"] == 1.0 and s["agnostic_recall_infected"] == 1.0


def test_missed_infected_cell_and_false_positive():
    cands = [_cand(1, [10, 10, 20, 20], 0.9), _cand(1, [50, 50, 20, 20], 0.8),   # image 1 both found
             _cand(2, [10, 10, 20, 20], 0.7), _cand(2, [50, 50, 20, 20], 0.6),
             _cand(3, [80, 80, 15, 15], 0.95)]                                    # image 3: infected missed, FP
    prep = prepare_agnostic(GT, cands, INF)
    s = score_agnostic(prep, [1, 2, 3])
    assert s["agnostic_recall_all"] == 4 / 5 and s["agnostic_recall_infected"] == 1 / 2
    # the top-scored candidate is a false positive -> precision 0 at the first recall levels
    assert 0 < s["agnostic_AP50"] < 0.8
    assert score_agnostic(prep, [1, 2]) == {"agnostic_AP50": 1.0, "agnostic_recall_all": 1.0,
                                            "agnostic_recall_infected": 1.0}
    # duplicated images (bootstrap resample) count twice
    d = score_agnostic(prep, [1, 3, 3])
    assert d["agnostic_recall_infected"] == 1 / 3 and d["agnostic_recall_all"] == 2 / 4


def test_localization_is_class_blind():
    # a box on the infected cell counts whatever the run would call it: candidates carry no class
    cands = [_cand(3, [31, 31, 20, 20], 0.5)]
    s = score_agnostic(prepare_agnostic(GT, cands, INF), [3])
    assert s["agnostic_recall_infected"] == 1.0 and np.isclose(s["agnostic_AP50"], 1.0)


def test_image_without_gt_or_candidates():
    gt = {**GT, "images": GT["images"] + [{"id": 4, "width": 100, "height": 100}]}
    s = score_agnostic(prepare_agnostic(gt, [], INF), [4])
    assert np.isnan(s["agnostic_recall_all"]) and np.isnan(s["agnostic_recall_infected"])
