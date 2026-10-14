import numpy as np

from malaria_ssl.eval.paired import METRICS, bootstrap_table, mean_of_runs, prepare_run, resamples, score, summarize_diff

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
