"""scripts/robustness_curve.py: the grid is the one fixed in reports/robustness_plan.md; cached predictions round-trip.
Nothing here runs the curve."""

from __future__ import annotations

import importlib.util

import numpy as np

from malaria_ssl.utils.config import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("robustness_curve", PROJECT_ROOT / "scripts/robustness_curve.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def test_grid_matches_plan():
    conds = rc.conditions(["stain", "blur", "jpeg", "brightness_contrast", "scale_down", "scale_up", "stain_full"])
    assert len(conds) == 26 and conds[0] == ("none", 0.0)
    assert ("stain_full", 1.0) in conds and ("stain_full", 0.5) not in conds
    assert rc.GRID == (0.25, 0.5, 0.75, 1.0) and rc.N_RESAMPLES == 1000 and rc.SEED == 42
    part1 = rc.conditions(["stain", "blur", "jpeg"])
    assert len(part1) == 13


def test_cached_predictions_round_trip(tmp_path):
    p = tmp_path / "x.npz"
    np.savez_compressed(
        p,
        image_id=np.array([2, 2]),
        label=np.array([0, 1], np.int8),
        score=np.array([0.9, 0.5], np.float32),
        bbox=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32),
    )
    dets = rc.load_dets(p, ["uninfected", "infected"], {"uninfected": 1, "infected": 2})
    assert dets == [
        {"image_id": 2, "category_id": 1, "score": np.float32(0.9).item(), "bbox": [1.0, 2.0, 3.0, 4.0]},
        {"image_id": 2, "category_id": 2, "score": 0.5, "bbox": [5.0, 6.0, 7.0, 8.0]},
    ]


def test_cond_file_keeps_the_strength():
    """Regression: Path.with_suffix turned blur_0.25 into blur_0.npz, so aggregate found no predictions."""
    for kind, s in rc.conditions(["blur", "stain_full"]):
        assert rc.cond_file("a_coco_detr_binary", kind, s, ".npz").name == f"{rc.cond_name(kind, s)}.npz"
    assert rc.cond_file("m", "blur", 0.25, ".json").name == "blur_0.25.json"
