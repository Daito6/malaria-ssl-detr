"""Probe tests: metrics against hand-computed cases, classifiers on separable features, patch masks."""
import numpy as np
import pytest
import torch

from malaria_ssl.ssl.probe import (
    apply_stain_shift, auroc, average_precision, bootstrap_metrics, build_stain_shift, compute_metrics,
    knn_predict, linear_predict, macro_f1, select_run, stratified_subset, window_and_patch_mask,
)
from malaria_ssl.utils.config import load_experiment


def test_auroc_hand_computed_with_ties():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert auroc(y, np.array([0.1, 0.5, 0.5, 0.9])) == pytest.approx(0.875)  # one tied pos/neg pair = 1/2
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_average_precision_hand_computed():
    y = np.array([1, 0, 1, 0])
    # ranks: 1(P) 2(N) 3(P) 4(N): AP = 0.5*1 + 0.5*(2/3)
    assert average_precision(y, np.array([0.9, 0.8, 0.7, 0.1])) == pytest.approx(0.5 + 1 / 3)
    # all tied: one threshold, precision 0.5 at recall 1
    assert average_precision(y, np.full(4, 0.3)) == pytest.approx(0.5)
    assert average_precision(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.1])) == pytest.approx(1.0)


def test_macro_f1_and_stratified_subset():
    y = np.array([0, 0, 1, 1, 2])
    assert macro_f1(y, y, 3) == 1.0
    assert macro_f1(y, np.array([0, 1, 1, 1, 2]), 3) == pytest.approx((2 / 3 + 0.8 + 1) / 3)
    yy = np.repeat(np.arange(3), [500, 20, 3])
    idx = stratified_subset(yy, 0.1, 3, seed=0)
    assert np.bincount(yy[idx]).tolist() == [50, 2, 1]
    assert np.array_equal(idx, stratified_subset(yy, 0.1, 3, seed=0))


def _separable(n_per_class=60, k=3, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(k, dim, generator=g) * 6
    y = torch.arange(k).repeat_interleave(n_per_class)
    return centres[y] + torch.randn(len(y), dim, generator=g), y


def test_probes_on_separable_features_give_auroc_one():
    xtr, ytr = _separable(seed=0)
    xte, yte = _separable(seed=0)
    xte = xte + 0.1 * torch.randn_like(xte)
    infected = np.array([False, True, True])
    names = ["a", "b", "c"]
    for probs in (knn_predict(xtr, ytr, xte, 3, k=20, temperature=0.07),
                  linear_predict(xtr, ytr, xte, 3, l2=1e-3, max_iter=200)):
        m = compute_metrics(yte.numpy(), probs.numpy(), infected, names)
        assert m["auroc_infected"] > 0.999 and m["auprc_infected"] > 0.999
        assert m["balanced_acc"] > 0.99 and m["macro_f1"] > 0.99
    ci = bootstrap_metrics(yte.numpy(), probs.numpy(), np.arange(len(yte)) // 10, infected, names, 50, 0)
    assert ci["auroc_infected"]["low"] > 0.99


def test_patch_mask_inside_box_and_fallback():
    # box well inside the image: window centred on it; patch centres 300..348 step 16 -> 4 x 4 inside
    origin, m = window_and_patch_mask(np.array([300.0, 200.0, 348.0, 248.0]), 800, 600, 224, 16)
    assert origin == (212, 112)
    grid = m.reshape(14, 14)
    ys, xs = np.nonzero(grid)
    centres_x, centres_y = 212 + xs * 16 + 8, 112 + ys * 16 + 8
    assert m.sum() == 16 and np.all((centres_x >= 300) & (centres_x <= 348)) and np.all((centres_y >= 200) & (centres_y <= 248))
    # box at the image corner: window is clamped inside the image
    origin, _ = window_and_patch_mask(np.array([0.0, 0.0, 30.0, 30.0]), 800, 600, 224, 16)
    assert origin == (0, 0)
    # tiny box between patch centres -> nearest patch
    origin, m = window_and_patch_mask(np.array([401.0, 301.0, 403.0, 303.0]), 800, 600, 224, 16)
    assert m.sum() == 1


def test_stain_shift_is_fixed_per_image_and_changes_colours():
    scfg = load_experiment("configs/ssl/probe.yaml")["probe"]["stain_shift"]
    rng = np.random.default_rng(0)
    img = rng.integers(60, 230, size=(120, 160, 3), dtype=np.uint8)
    t = build_stain_shift(scfg)
    a = apply_stain_shift(img, t, scfg["seed"], image_id=7)
    _ = apply_stain_shift(img, t, scfg["seed"], image_id=8)        # other images do not disturb the stream
    assert np.array_equal(a, apply_stain_shift(img, build_stain_shift(scfg), scfg["seed"], image_id=7))
    assert not np.array_equal(a, apply_stain_shift(img, t, scfg["seed"], image_id=8))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.abs(a.astype(float) - img).mean() > 5


def _fake_probe(clean, clean_low, stain, stain_low, ll_mean, ll_std, f1):
    lin = {"metrics": {"auprc_infected": clean, "macro_f1": f1},
           "ci95": {"auprc_infected": {"low": clean_low, "high": 1.0}}}
    st = {"metrics": {"auprc_infected": stain}, "ci95": {"auprc_infected": {"low": stain_low, "high": 1.0}}}
    return {"methods": {"linear": lin}, "stain_shift": {"methods": {"linear": st}},
            "low_label": {"0.1": {"linear": {"mean": {"auprc_infected": ll_mean}, "std": {"auprc_infected": ll_std}}}}}


def test_select_run_lexicographic_with_ci_ties():
    res = {"a": _fake_probe(0.95, 0.92, 0.78, 0.75, 0.90, 0.01, 0.5),
           "b": _fake_probe(0.94, 0.91, 0.85, 0.80, 0.85, 0.01, 0.6),   # tied with a on clean, wins stain
           "c": _fake_probe(0.90, 0.85, 0.99, 0.98, 0.99, 0.01, 0.9)}   # outside a's clean CI -> dropped
    sel = select_run(res)
    assert sel["steps"][0]["tied"] == ["a", "b"]
    assert sel["winner"] == "b" and len(sel["steps"]) == 2
    # stain tie too (a inside b's stain CI) -> 10%-label decides
    res["a"] = _fake_probe(0.95, 0.92, 0.84, 0.75, 0.90, 0.01, 0.5)
    sel = select_run(res)
    assert [s["criterion"] for s in sel["steps"]] == ["auprc_clean", "auprc_stain", "auprc_10pct"]
    assert sel["winner"] == "a"
