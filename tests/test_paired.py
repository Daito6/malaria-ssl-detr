import json

import numpy as np
import pytest

from malaria_ssl.eval.paired import (METRICS, bootstrap_table, fold_merge, mean_of_runs, prepare_run, resamples, score,
                                     summarize_diff)
from malaria_ssl.ssl.data import load_image_folds
from malaria_ssl.ssl.encoders import ssl_run_target_fold

GT = {"images": [{"id": i, "width": 100, "height": 100} for i in (1, 2, 3, 4)],
      "categories": [{"id": 1, "name": "uninfected"}, {"id": 2, "name": "infected"}],
      "annotations": [{"id": i, "image_id": i, "category_id": 2, "bbox": [10, 10, 20, 20], "area": 400,
                       "iscrowd": 0} for i in (1, 2, 3, 4)]}


def _dets(found):
    return [{"image_id": i, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 0.9} for i in found]


def test_paired_bootstrap_difference():
    a = prepare_run(GT, _dets([1, 2, 3, 4]), 2, thr=0.5)
    b = prepare_run(GT, _dets([1, 2]), 2, thr=0.5)
    ids = [1, 2, 3, 4]
    pa, pb = score(a, ids), score(b, ids)
    assert pa["recall"] == 1.0 and pb["recall"] == 0.5
    assert np.isclose(pa["AP50"], 1.0)
    samples = resamples(ids, 200, seed=0)
    boot = bootstrap_table({"a": a, "b": b, "a2": a}, samples)
    d = summarize_diff(pa["recall"], pb["recall"], boot["a"][:, 2], boot["b"][:, 2])
    assert np.isclose(d["diff"], 0.5) and d["low"] >= 0 and d["high"] <= 1
    same = summarize_diff(pa["recall"], pa["recall"], boot["a"][:, 2], boot["a2"][:, 2])
    assert same["low"] == same["high"] == 0 and same["share_le_0"] == 1.0  # identical runs on shared resamples


def test_mean_of_runs_is_paired_mean():
    ids = [1, 2, 3, 4]
    runs = {"a": prepare_run(GT, _dets([1, 2, 3, 4]), 2, thr=0.5), "b": prepare_run(GT, _dets([1, 2]), 2, thr=0.5),
            "c": prepare_run(GT, _dets([3]), 2, thr=0.5)}
    points = {n: score(r, ids) for n, r in runs.items()}
    boots = bootstrap_table(runs, resamples(ids, 50, seed=0))
    # the mean of a run with itself is that run
    p, bt = mean_of_runs(points, boots, ["a", "a"])
    assert p == points["a"] and np.array_equal(bt, boots["a"])
    # three members: the point and every resample are the plain means
    p, bt = mean_of_runs(points, boots, ["a", "b", "c"])
    for j, m in enumerate(METRICS):
        assert np.isclose(p[m], np.mean([points[n][m] for n in "abc"]))
        assert np.allclose(bt[:, j], (boots["a"][:, j] + boots["b"][:, j] + boots["c"][:, j]) / 3, equal_nan=True)
    assert np.isclose(p["recall"], (1.0 + 0.5 + 0.25) / 3)


def test_fold_merge_scores_each_fold_with_its_run():
    ids = [1, 2, 3, 4]
    folds = {1: 1, 2: 1, 3: 2, 4: 2}
    full = prepare_run(GT, _dets([1, 2, 3, 4]), 2, thr=0.5)
    half = prepare_run(GT, _dets([1, 2]), 2, thr=0.5)
    # fold 1 (images 1, 2) from `half`, fold 2 (3, 4) from `full`: every GT is found
    m = fold_merge({1: half, 2: full}, folds)
    assert score(m, ids)["recall"] == 1.0 and np.isclose(score(m, ids)["AP50"], 1.0)
    # swapped: images 3, 4 come from `half`, which has no detections there
    m = fold_merge({1: full, 2: half}, folds)
    assert score(m, ids)["recall"] == 0.5
    # equals a single run prepared on the union of the per-fold detections (same threshold)
    union = prepare_run(GT, _dets([1, 2]), 2, thr=0.5)
    assert score(m, ids) == score(union, ids)
    # a run merged with itself is that run, also on every resample
    same = fold_merge({1: full, 2: full}, folds)
    samples = resamples(ids, 50, seed=0)
    boot = bootstrap_table({"run": full, "merged": same}, samples)
    assert np.array_equal(boot["run"], boot["merged"], equal_nan=True)


def test_fold_merge_keeps_each_runs_own_threshold():
    ids = [1, 2, 3, 4]
    folds = {1: 1, 2: 1, 3: 2, 4: 2}
    low = prepare_run(GT, _dets([1, 2, 3, 4]), 2, thr=0.5)
    high = prepare_run(GT, _dets([1, 2, 3, 4]), 2, thr=0.95)   # scores 0.9 < 0.95: nothing kept at its threshold
    m = score(fold_merge({1: low, 2: high}, folds), ids)
    assert m["recall"] == 0.5 and m["precision"] == 1.0
    assert np.isclose(m["AP50"], 1.0)   # AP50 is threshold-free
    with pytest.raises(ValueError):
        fold_merge({1: low, 2: high}, {1: 1, 2: 1, 3: 2})   # image 4 has no fold
    with pytest.raises(ValueError):
        fold_merge({1: low, 2: high}, {1: 1, 2: 1, 3: 1, 4: 1})   # fold 2 has no images


def test_load_image_folds_and_ssl_run_target_fold(tmp_path):
    p = tmp_path / "folds.json"
    p.write_text(json.dumps({"folds": {"1": [{"id": 1, "file_name": "a.jpg"}], "2": [{"id": 2, "file_name": "b.jpg"}]}}))
    assert load_image_folds(p) == {1: 1, 2: 2}
    p.write_text(json.dumps({"folds": {"1": [{"id": 1, "file_name": "a.jpg"}], "2": [{"id": 1, "file_name": "a.jpg"}]}}))
    with pytest.raises(ValueError):
        load_image_folds(p)
    run = tmp_path / "ssl_run"
    run.mkdir()
    (run / "config.yaml").write_text("target:\n  folds_file: x.json\n  fold: 2\n  fraction: 0.5\n")
    assert ssl_run_target_fold(run) == 2
    (run / "config.yaml").write_text("model:\n  init: dino\n")
    assert ssl_run_target_fold(run) is None
    with pytest.raises(FileNotFoundError):
        ssl_run_target_fold(tmp_path / "missing")
