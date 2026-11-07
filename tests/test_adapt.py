import numpy as np
import torch

from malaria_ssl.eval.adapt import (adabn, bbse_prior, cbr, coral, em_prior, irc, match_gt_to_candidates,
                                    pooled_parasitemia, prior_adjust, quantile_threshold, robust_moments)
from malaria_ssl.ssl.probe import linear_fit, linear_predict


def _data(n=4000, d=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(n, d, generator=g) * 2 + 1
    tgt = torch.randn(n, d, generator=g) @ torch.diag(torch.linspace(0.5, 3, d)) - 4
    return src, tgt


def test_adabn_matches_source_moments_on_cells():
    src, tgt = _data()
    cells = np.ones(len(tgt), bool)
    cells[::3] = False
    out = adabn(tgt, cells, src)
    assert torch.allclose(out[cells].mean(0), src.mean(0), atol=1e-4)
    assert torch.allclose(out[cells].std(0, unbiased=False), src.std(0, unbiased=False), atol=1e-4)


def test_coral_matches_source_covariance():
    src, tgt = _data()
    cells = np.ones(len(tgt), bool)
    out = coral(tgt, cells, src, lam=0.0).double()
    cs = torch.cov(src.double().T, correction=0)
    assert torch.allclose(torch.cov(out.T, correction=0), cs, atol=1e-3)


def test_irc_per_group_median_and_shrink():
    src, tgt = _data()
    group = np.repeat(np.arange(4), len(tgt) // 4)
    tgt = tgt + torch.as_tensor(group, dtype=torch.float32)[:, None] * 5   # per-image offsets
    cells = np.ones(len(tgt), bool)
    out = irc(tgt, cells, group, src, shrink=0.0, scale=True)
    ms, _ = robust_moments(src)
    for g in range(4):
        med, _ = robust_moments(out[group == g])
        assert torch.allclose(med, ms, atol=1e-3)
    # with a huge pseudo-count every group gets the pooled statistics -> per-group offsets survive
    out2 = irc(tgt, cells, group, src, shrink=1e9, scale=False)
    d = robust_moments(out2[group == 3])[0] - robust_moments(out2[group == 0])[0]
    assert torch.allclose(d.float(), torch.full_like(d.float(), 15.0), atol=0.5)


def test_prior_methods_recover_the_target_prior():
    rng = np.random.default_rng(0)
    pi_t, n = 0.07, 200000
    y = rng.random(n) < pi_t
    # calibrated balanced-prior posterior of a 1-d Gaussian problem: logit = 2 * s with s ~ N(+-1, 1)
    s = np.where(y, 1.0, -1.0) + rng.standard_normal(n)
    p = 1 / (1 + np.exp(-2 * s))
    assert abs(em_prior(p) - pi_t) < 0.005
    src_y = rng.random(n) < 0.5
    src_p = 1 / (1 + np.exp(-2 * (np.where(src_y, 1.0, -1.0) + rng.standard_normal(n))))
    tpr, fpr = (src_p[src_y] >= 0.5).mean(), (src_p[~src_y] >= 0.5).mean()
    assert abs(bbse_prior(p, tpr, fpr) - pi_t) < 0.005
    assert np.allclose(prior_adjust(p, 0.5), p)


def test_quantile_threshold_fraction():
    s = np.linspace(0, 1, 1001)
    assert abs((s >= quantile_threshold(s, 0.1)).mean() - 0.1) < 2e-3
    assert (s >= quantile_threshold(s, 0.0)).sum() == 0
    assert (s.astype(np.float32) >= quantile_threshold(s, 1e-9)).sum() == 0


def test_cbr_falls_back_without_infected_pseudo_labels():
    src, tgt = _data()
    cells = np.ones(len(tgt), bool)
    out, fell = cbr(tgt, cells, src, np.zeros(len(src)), lambda x: np.zeros(len(x)), "hard", 2)
    assert fell and torch.allclose(out, adabn(tgt, cells, src))


def test_linear_fit_equals_linear_predict():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(300, 5, generator=g), (torch.rand(300, generator=g) < 0.2).long()
    xt = torch.randn(50, 5, generator=g)
    assert torch.equal(linear_fit(x, y, 2, 1e-3, 100)(xt), linear_predict(x, y, xt, 2, 1e-3, 100))


def test_match_and_pooled_parasitemia():
    gi, gb = np.array([1, 1, 2]), np.array([[0, 0, 10, 10], [50, 50, 10, 10], [0, 0, 10, 10]], float)
    ci, cb = np.array([1, 1, 2]), np.array([[1, 1, 10, 10], [200, 200, 5, 5], [30, 30, 10, 10]], float)
    assert match_gt_to_candidates(gi, gb, ci, cb, 0.5).tolist() == [0, -1, -1]
    est = pooled_parasitemia([1, 2, 3], {1: 2, 2: 0}, {1: 10, 2: 10, 3: 0}, {1: "a", 2: "a", 3: "b"})
    assert est == {"a": 0.1, "b": 0.0}
