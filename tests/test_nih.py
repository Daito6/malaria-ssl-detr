import csv
from collections import Counter

import numpy as np
import pytest

from malaria_ssl.data.nih import (agnostic_coco, check_patients, convert, core_bounds, in_core, make_tiles,
                                  parasitemia, seam_nms, tile_starts)
from malaria_ssl.eval.bootstrap import ap50_from_per_image, per_image_iou50
from malaria_ssl.eval.coco_metrics import run_cocoeval
from malaria_ssl.eval.detection_metrics import match_per_class, prf_at
from malaria_ssl.eval.zeroshot import (candidates, core_mask, ece, end_to_end_detections, matched_p_infected,
                                       parasitemia_estimate, patient_resamples)
from malaria_ssl.utils.config import resolve_path

CATS = [{"id": 1, "name": "uninfected"}, {"id": 2, "name": "parasitized"}, {"id": 3, "name": "white blood cell"}]


def _sources():
    polys = {"images": [{"id": 1, "file_name": "IMG_1.jpg", "width": 100, "height": 50}],
             "categories": CATS,
             "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
                             {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 0, 10, 10], "iscrowd": 0},
                             {"id": 3, "image_id": 1, "category_id": 3, "bbox": [40, 0, 10, 10], "iscrowd": 0}]}
    points = {"images": [{"id": 1, "file_name": "IMG_2.jpg", "width": 100, "height": 50}],
              "categories": CATS + [{"id": 4, "name": "ambiguous"}],
              "annotations": [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [60, 0, 10, 10], "iscrowd": 0},
                              {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 20, 10, 10], "iscrowd": 0}]}
    patients = [{"subset": "polys", "file_name": "IMG_1.jpg", "patient_id": "P1", "path": "Polygon Set/P1/img/IMG_1.jpg"},
                {"subset": "points", "file_name": "IMG_2.jpg", "patient_id": "P2", "path": "Point Set/P2/img/IMG_2.jpg"}]
    return {"polys": polys, "points": points}, patients


def test_convert_ids_classes_and_ignore_regions():
    sources, patients = _sources()
    coco = convert(sources, patients)
    assert [im["id"] for im in coco["images"]] == [1, 1001]                       # points offset +1000
    assert [im["patient_id"] for im in coco["images"]] == ["P1", "P2"]
    assert coco["images"][1]["file_name"] == "Point Set/P2/img/IMG_2.jpg"
    real = [(a["image_id"], a["category_id"]) for a in coco["annotations"] if not a["iscrowd"]]
    assert real == [(1, 1), (1, 2), (1001, 1)]                                     # WBC dropped, parasitized -> 2
    crowd = [a for a in coco["annotations"] if a["iscrowd"]]
    assert sorted(a["category_id"] for a in crowd) == [1, 2] and {a["image_id"] for a in crowd} == {1001}
    assert len({a["id"] for a in coco["annotations"]}) == len(coco["annotations"])  # unique annotation ids
    agn = agnostic_coco(coco)
    assert sum(a["iscrowd"] for a in agn["annotations"]) == 1 and len(agn["annotations"]) == 4
    per_image, per_patient = parasitemia(coco)
    assert per_image[0]["parasitemia"] == 0.5 and per_image[1]["n_ambiguous"] == 1
    assert {p["patient_id"]: p["parasitemia"] for p in per_patient} == {"P1": 0.5, "P2": 0.0}


def test_convert_rejects_unknown_class():
    sources, patients = _sources()
    sources["polys"]["categories"] = CATS + [{"id": 9, "name": "platelet"}]
    with pytest.raises(ValueError):
        convert(sources, patients)


def test_patient_table_of_the_real_data():
    path = resolve_path("data/processed/nih_nlm/patients.csv")
    if not path.exists():
        pytest.skip("run scripts/prepare_nih.py first")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert check_patients(rows) == {"polys": 33, "points": 160}
    per_patient = Counter(r["patient_id"] for r in rows)
    assert len(per_patient) == 193 and set(per_patient.values()) == {5}
    polys = {r["patient_id"] for r in rows if r["subset"] == "polys"}
    points = {r["patient_id"] for r in rows if r["subset"] == "points"}
    assert not polys & points and len(rows) == 965


def _gt_with_ignore():
    return {"images": [{"id": 1, "width": 200, "height": 100}],
            "categories": [{"id": 1, "name": "uninfected"}, {"id": 2, "name": "infected"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 0, 20, 20], "area": 400, "iscrowd": 1}]}


def test_ignore_region_is_neither_tp_nor_fp():
    gt = _gt_with_ignore()
    dets = [{"image_id": 1, "category_id": 2, "bbox": [102, 2, 16, 16], "score": 0.9},   # inside the ignore region
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20], "score": 0.5},     # TP
            {"image_id": 1, "category_id": 2, "bbox": [150, 50, 20, 20], "score": 0.4}]  # FP (background)
    ev = run_cocoeval(gt, dets)
    per = per_image_iou50(ev)
    # ranked: ignored, TP, FP -> precision 1 up to recall 1: AP50 = 1 (an FP at 0.9 would give 0.5)
    assert ap50_from_per_image(per, [1], ev.params.recThrs)[2] == pytest.approx(1.0)
    m = match_per_class(gt, dets, 0.5)[2][1]
    assert m["n_gt"] == 1 and list(m["scores"]) == [0.5, 0.4] and list(m["tp"]) == [True, False]
    r = prf_at(match_per_class(gt, dets, 0.5)[2], 0.45)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    without = {**gt, "annotations": gt["annotations"][:1]}
    assert prf_at(match_per_class(without, dets, 0.5)[2], 0.45)["fp"] == 1


def test_ignore_region_uses_ioa_of_the_detection():
    gt = _gt_with_ignore()
    big = [{"image_id": 1, "category_id": 2, "bbox": [90, 0, 60, 60], "score": 0.9}]    # IoA 400 / 3600 < 0.5
    m = match_per_class(gt, big, 0.5)[2][1]
    assert list(m["tp"]) == [False]                                                     # stays a false positive


def test_tiles_cover_and_cores_partition_the_frame():
    assert tile_starts(2291, 800, 160) == [0, 640, 1280, 1491]
    assert tile_starts(500, 800, 160) == [0]
    starts = tile_starts(1289, 600, 160)
    cores = core_bounds(starts, 600, 1289)
    assert cores[0][0] == 0 and cores[-1][1] == 1289
    assert all(a[1] == b[0] for a, b in zip(cores, cores[1:]))
    for s, (lo, hi) in zip(starts, cores):   # each core lies inside its tile
        assert s <= lo and hi <= s + 600
    tiles = make_tiles(2291, 1289, (800, 600), 160)
    assert len(tiles) == 12
    pts = np.random.default_rng(0).uniform([0, 0], [2291, 1289], size=(500, 2))
    boxes = np.c_[pts - 1, np.full((500, 2), 2.0)]
    owners = np.stack([in_core(boxes, t["core"]) for t in tiles]).sum(0)
    assert (owners == 1).all()


def test_stitching_keeps_a_seam_cell_once():
    tiles = make_tiles(1440, 600, (800, 600), 160)       # starts 0 and 640, cut at 720
    # the same cell (resized frame x 700..740) seen by both tiles; factor 0.5 -> original x 1400..1480
    q = {"image_id": np.array([1, 1, 1]), "tile": np.array([0, 1, 1]),
         "box": np.array([[1400, 100, 80, 80], [1402, 100, 80, 80], [100, 100, 80, 80]], float),
         "probs": np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.9, 0.05, 0.05]])}
    keep = core_mask(q, tiles, 0.5)
    assert list(keep) == [False, True, False]   # centre x 1440 * 0.5 = 720 -> tile 1; x 140 * 0.5 = 70 -> tile 0
    q["tile"] = np.array([0, 1, 0])
    keep = core_mask(q, tiles, 0.5)
    assert list(keep) == [False, True, True]
    e2e = end_to_end_detections(q, np.ones(3, bool), 0.5, [1, 2])   # without the core filter: seam NMS
    assert sorted(e2e["score"].round(2).tolist()) == [0.8, 0.9]    # tile-1 duplicate removed, other cell kept
    c = candidates(q, np.ones(3, bool), 0.5)
    assert len(c["objectness"]) == 2


def test_seam_nms_never_suppresses_within_a_tile():
    b = np.array([[0, 0, 10, 10], [1, 0, 10, 10], [0, 1, 10, 10]], float)
    keep = seam_nms(b, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), 0.5)
    assert list(keep) == [True, True, False]


def test_patient_resamples_draw_whole_patients():
    image_patient = {1: "A", 2: "A", 3: "B", 4: "B", 5: "C"}
    for s in patient_resamples(image_patient, list(image_patient), 20, 0):
        c = Counter(image_patient[i] for i in s)
        assert sum(c.values()) == len(s)
        assert all(c[p] % {"A": 2, "B": 2, "C": 1}[p] == 0 for p in c)
        assert len(set(c)) <= 3


def test_ece_and_matched_scores_and_parasitemia():
    assert ece(np.array([1, 0]), np.array([1.0, 0.0]), 10) == 0.0
    assert ece(np.array([0, 0]), np.array([0.95, 0.95]), 10) == pytest.approx(0.95)
    p = matched_p_infected(np.array([1, 1]), np.array([[0, 0, 10, 10], [50, 50, 10, 10]], float),
                           np.array([1, 1]), np.array([[1, 0, 10, 10], [0, 0, 30, 30]], float),
                           np.array([0.7, 0.2]), 0.5)
    assert list(p) == [0.7, 0.0]
    est = parasitemia_estimate([1, 2], np.array([1]), np.array([1, 1, 1, 1]))
    assert est == {1: 0.25, 2: 0.0}
