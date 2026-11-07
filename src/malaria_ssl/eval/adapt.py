"""EXPLORATORY unlabeled target adaptation of the two-stage crop classifier (scripts/adapt.py).

Feature methods map the target crop features x (N, d) to the source feature distribution before the fixed linear
classifier; statistics are computed on the "cells" (candidates with objectness >= the BBBC041-val cell threshold)
and the map is applied to every candidate. Source statistics come from the BBBC041 val candidates (cells) of the
same box run. Prior methods re-estimate the infected fraction of the target cells from p(infected) only.
No target label is read here.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch

from malaria_ssl.eval.detection_metrics import iou_matrix
from malaria_ssl.ssl.probe import auroc

MAD_SCALE = 1.4826   # MAD -> std of a normal


# --------------------------------------------------------------------------- feature statistics

def moments(x: torch.Tensor, w: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(weighted) mean and variance per dimension, float64."""
    x = x.double()
    if w is None:
        return x.mean(0), x.var(0, unbiased=False)
    w = w.double() / w.double().sum()
    m = (w[:, None] * x).sum(0)
    return m, (w[:, None] * (x - m) ** 2).sum(0)


def robust_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Median and MAD-based std per dimension (float64)."""
    x = x.double()
    med = x.median(0).values
    return med, MAD_SCALE * (x - med).abs().median(0).values


def shrunk_cov(x: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance + lam * (trace / d) * I, float64."""
    x = x.double()
    m = x.mean(0)
    c = torch.cov((x - m).T, correction=0)
    d = c.shape[0]
    return m, c + lam * torch.trace(c) / d * torch.eye(d, dtype=c.dtype)


def _sqrtm(c: torch.Tensor, inverse: bool) -> torch.Tensor:
    val, vec = torch.linalg.eigh(c)
    val = val.clamp_min(1e-12)
    return vec @ torch.diag(val ** (-0.5 if inverse else 0.5)) @ vec.T


def _eps(v: torch.Tensor) -> torch.Tensor:
    return v.clamp_min(1e-12)


# --------------------------------------------------------------------------- feature methods

def adabn(x: torch.Tensor, cells: np.ndarray, src: torch.Tensor) -> torch.Tensor:
    """Global standardization: target cell mean / std -> source cell mean / std (diagonal)."""
    mt, vt = moments(x[cells])
    ms, vs = moments(src)
    return ((x.double() - mt) / _eps(vt).sqrt() * vs.sqrt() + ms).float()


def coral(x: torch.Tensor, cells: np.ndarray, src: torch.Tensor, lam: float) -> torch.Tensor:
    """CORAL: whiten with the (shrunk) target cell covariance, re-colour with the source covariance."""
    mt, ct = shrunk_cov(x[cells], lam)
    ms, cs = shrunk_cov(src, lam)
    a = _sqrtm(ct, inverse=True) @ _sqrtm(cs, inverse=False)
    return ((x.double() - mt) @ a + ms).float()


def balanced_moments(x: torch.Tensor, p: np.ndarray, mode: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Mean / variance of the equal-weight mixture of the two pseudo-classes (soft: weights p, 1 - p; hard: p >= 0.5).
    None if the infected pseudo-class is (almost) empty."""
    pt = torch.as_tensor(p, dtype=torch.float64)
    w1 = pt if mode == "soft" else (pt >= 0.5).double()
    if w1.sum() < 10 or (1 - w1).sum() < 10:
        return None
    m0, v0 = moments(x, 1 - w1)
    m1, v1 = moments(x, w1)
    return (m0 + m1) / 2, (v0 + v1) / 2 + (m1 - m0) ** 2 / 4


def cbr(x: torch.Tensor, cells: np.ndarray, src: torch.Tensor, p_src: np.ndarray,
        predict: Callable[[torch.Tensor], np.ndarray], mode: str, iters: int) -> tuple[torch.Tensor, bool]:
    """Class-balanced re-standardization: balanced pseudo-class moments of the target cells -> those of the source
    cells (source pseudo-labels = the classifier on the clean source). Starts from AdaBN; `iters` re-labellings.
    Returns (features, fell_back): if a pseudo-class is empty the AdaBN map of the last step is kept."""
    bs = balanced_moments(src, p_src, mode)
    out = adabn(x, cells, src)
    if bs is None:
        return out, True
    for _ in range(iters):
        bt = balanced_moments(x[cells], predict(out[cells]), mode)
        if bt is None:
            return out, True
        out = ((x.double() - bt[0]) / _eps(bt[1]).sqrt() * bs[1].sqrt() + bs[0]).float()
    return out, False


def irc(x: torch.Tensor, cells: np.ndarray, group: np.ndarray, src: torch.Tensor, shrink: float,
        scale: bool) -> torch.Tensor:
    """In-image robust centring: per group (image or patient) the median (and MAD std if `scale`) of its cells ->
    source cell median / MAD std. Group statistics are shrunk toward the pooled target cell statistics with weight
    `shrink` (pseudo-count; 0 = the group alone). Groups without cells use the pooled statistics."""
    ms, ss = robust_moments(src)
    mg_all, sg_all = robust_moments(x[cells])
    out = torch.empty_like(x)
    rows = defaultdict(list)
    for j, g in enumerate(group):
        rows[g].append(j)
    xd = x.double()
    for g, r in rows.items():
        r = np.asarray(r)
        c = r[cells[r]]
        n = len(c)
        if n:
            mg, sg = robust_moments(x[c])
            mg = (n * mg + shrink * mg_all) / (n + shrink)
            sg = (n * sg + shrink * sg_all) / (n + shrink)
        else:
            mg, sg = mg_all, sg_all
        z = xd[r] - mg
        out[r] = (z / _eps(sg) * ss + ms if scale else z + ms).float()
    return out


# --------------------------------------------------------------------------- prior (label-shift) methods

def prior_adjust(p: np.ndarray, pi_t: float, pi_s: float = 0.5) -> np.ndarray:
    """Posterior under the target prior (Saerens et al. 2002); the classifier is class-balanced, so pi_s = 0.5."""
    a = p * pi_t / pi_s
    b = (1 - p) * (1 - pi_t) / (1 - pi_s)
    return a / np.maximum(a + b, 1e-12)


def em_prior(p: np.ndarray, pi_s: float = 0.5, max_iter: int = 1000, tol: float = 1e-8) -> float:
    """EM estimate of the target infected fraction from the cells' p(infected)."""
    pi = pi_s
    for _ in range(max_iter):
        new = float(prior_adjust(p, pi, pi_s).mean())
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def bbse_prior(p_target: np.ndarray, tpr: float, fpr: float) -> float:
    """Black-box shift estimation (binary, hard rule p >= 0.5): q = tpr pi + fpr (1 - pi) -> pi, clipped to [0, 1]."""
    q = float((p_target >= 0.5).mean())
    if tpr - fpr <= 1e-9:
        return float("nan")
    return float(np.clip((q - fpr) / (tpr - fpr), 0.0, 1.0))


def quantile_threshold(scores: np.ndarray, frac: float) -> float:
    """Threshold t with mean(scores >= t) ~ frac (the frac-quantile from the top); frac ~ 0 -> inf (no positive)."""
    k = int(round(frac * len(scores))) if np.isfinite(frac) else 0
    if k <= 0:
        return float("inf")
    return float(np.sort(scores)[::-1][min(k, len(scores)) - 1])


# --------------------------------------------------------------------------- matching / aggregation helpers

def match_gt_to_candidates(gt_image: np.ndarray, gt_box: np.ndarray, c_image: np.ndarray, c_box: np.ndarray,
                           iou_thr: float) -> np.ndarray:
    """Per GT box: index of the candidate with the highest IoU (>= iou_thr), -1 if none (boxes xywh)."""
    out = np.full(len(gt_box), -1)
    crow, grow = defaultdict(list), defaultdict(list)
    for j, i in enumerate(c_image):
        crow[int(i)].append(j)
    for j, i in enumerate(gt_image):
        grow[int(i)].append(j)
    for img, g in grow.items():
        d = crow.get(img, [])
        if not d:
            continue
        ious = iou_matrix(gt_box[g], c_box[d])
        best = ious.argmax(1)
        hit = ious[np.arange(len(g)), best] >= iou_thr
        out[np.asarray(g)[hit]] = np.asarray(d)[best[hit]]
    return out


def pooled_parasitemia(image_ids: list[int], n_inf: dict[int, int], n_cell: dict[int, int],
                       group: dict[int, str]) -> dict[str, float]:
    """Per group (patient): sum of infected predictions / sum of cell predictions over its images."""
    a, b = defaultdict(int), defaultdict(int)
    for i in image_ids:
        a[group[i]] += n_inf.get(i, 0)
        b[group[i]] += n_cell.get(i, 0)
    return {g: (a[g] / b[g] if b[g] else 0.0) for g in a}


def auroc_or_nan(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    return float(auroc(y, s)) if 0 < y.sum() < len(y) else float("nan")
