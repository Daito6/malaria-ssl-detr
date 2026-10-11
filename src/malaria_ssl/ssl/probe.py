"""Cell-level probe of a frozen ViT encoder, without DETR.

Crops: for every GT box (train / val; `exclude` classes dropped) a `window` x `window` crop of the
long-side-800 image (the DETR eval scale) centred on the box and shifted to stay inside the image.
Feature = mean of the final-normed patch tokens whose centres fall inside the box (fallback: the
patch nearest to the box centre). RBC boxes are subsampled per split; all other classes are kept.

Classifiers (fitted on train crops, evaluated on val crops), identical for every encoder:
    kNN: k nearest train crops by cosine similarity, votes weighted by exp(sim / T);
    linear: multinomial logistic regression on standardised features, class-balanced weights,
            L2 penalty, full-batch L-BFGS (torch).
Binary score P(infected) = sum of the probabilities of the infected classes.

Metrics (numpy, no sklearn): binary AUROC / AUPRC / balanced accuracy; multiclass balanced accuracy,
macro-F1, per-class recall. 95% CIs by bootstrap over val IMAGES (crops of one image are correlated).
Low-label: class-stratified fractions of the train crops (>= 1 per class), several seeds.

Stain-shifted probe: the same classifiers (fitted on CLEAN train crops) are also evaluated on the val
crops after a fixed, seeded, strong photometric shift (HED stain jitter, brightness / contrast /
saturation / hue, mild blur, JPEG). The shift is drawn once per val image (seed = (shift seed, image
id)) and applied to the whole long-side-800 image before cropping, like a slide-level stain change.

Lambda selection (`select_run`): linear AUPRC infected on clean val -> stain-shifted linear AUPRC ->
10%-label linear AUPRC -> linear macro-F1. At each step the runs whose value lies inside the leader's
interval (95% bootstrap CI; mean - std over seeds for the 10%-label step) stay tied and go to the
next criterion; the last criterion is decided by the point estimate.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from malaria_ssl.data.detection import HEDJitter, load_split_coco
from malaria_ssl.ssl.data import normalize

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- crop set

def probe_classes(data_cfg: dict[str, Any], exclude: list[str]) -> tuple[list[str], list[int], np.ndarray]:
    """Class names (COCO id order), their category ids, and the infected flag per class."""
    cats = [c for c in sorted(data_cfg["categories"], key=lambda c: c["id"]) if c["name"] not in exclude]
    mapping = data_cfg["binary"]["mapping"]
    return [c["name"] for c in cats], [c["id"] for c in cats], np.array([mapping[c["name"]] == "infected" for c in cats])


def build_crop_set(split: str, data_cfg: dict[str, Any], pcfg: dict[str, Any], cache_index: dict[str, Any],
                   resolve) -> dict[str, np.ndarray]:
    """Boxes of the probe set: cache row, image id, label, box (x0, y0, x1, y1) at long side `long_side`."""
    names, cat_ids, _ = probe_classes(data_cfg, pcfg["exclude"])
    label_of = {c: i for i, c in enumerate(cat_ids)}
    coco = load_split_coco(split, "multiclass", data_cfg, resolve)
    row_of = {im["id"]: i for i, im in enumerate(cache_index["images"])}
    size_of = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    anns = sorted((a for a in coco["annotations"] if a["category_id"] in label_of), key=lambda a: a["id"])
    rbc = [a for a in anns if names[label_of[a["category_id"]]] == "red blood cell"]
    rest = [a for a in anns if names[label_of[a["category_id"]]] != "red blood cell"]
    n_rbc = pcfg["rbc_per_split"][split]
    rng = np.random.default_rng(pcfg["seed"])
    if n_rbc < len(rbc):
        rbc = [rbc[i] for i in sorted(rng.choice(len(rbc), size=n_rbc, replace=False))]
    keep = sorted(rbc + rest, key=lambda a: (a["image_id"], a["id"]))
    boxes, rows, img_ids, labels = [], [], [], []
    for a in keep:
        w0, h0 = size_of[a["image_id"]]
        s = pcfg["long_side"] / max(w0, h0)
        x, y, w, h = a["bbox"]
        boxes.append([x * s, y * s, (x + w) * s, (y + h) * s])
        rows.append(row_of[a["image_id"]])
        img_ids.append(a["image_id"])
        labels.append(label_of[a["category_id"]])
    return {"box": np.asarray(boxes, np.float32), "row": np.asarray(rows), "image_id": np.asarray(img_ids),
            "label": np.asarray(labels)}


def window_and_patch_mask(box: np.ndarray, img_w: int, img_h: int, window: int, patch: int
                          ) -> tuple[tuple[int, int], np.ndarray]:
    """Window origin (box-centred, clamped inside the image) and the (g*g,) mask of patches whose
    centres fall inside the box (nearest patch if none)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(round(min(max(cx - window / 2, 0), img_w - window)))
    y0 = int(round(min(max(cy - window / 2, 0), img_h - window)))
    g = window // patch
    centres = np.arange(g) * patch + patch / 2
    px, py = np.meshgrid(centres + x0, centres + y0)
    inside = (px >= box[0]) & (px <= box[2]) & (py >= box[1]) & (py <= box[3])
    if not inside.any():
        inside = np.zeros_like(inside)
        d = (px - cx) ** 2 + (py - cy) ** 2
        inside.flat[int(np.argmin(d))] = True
    return (x0, y0), inside.reshape(-1).astype(np.float32)


def build_stain_shift(scfg: dict[str, Any]) -> A.Compose:
    """Strong photometric shift of the stain-shifted probe; every transform is always applied."""
    cj = scfg["color_jitter"]
    return A.Compose([
        HEDJitter(sigma=scfg["hed_sigma"], p=1.0),
        A.ColorJitter(brightness=cj["brightness"], contrast=cj["contrast"], saturation=cj["saturation"],
                      hue=cj["hue"], p=1.0),
        A.GaussianBlur(sigma_limit=tuple(scfg["blur_sigma"]), p=1.0),
        A.ImageCompression(quality_range=tuple(scfg["jpeg_quality"]), p=1.0),
    ])


def apply_stain_shift(img: np.ndarray, transform: A.Compose, seed: int, image_id: int) -> np.ndarray:
    """Shift of one image, fixed by (seed, image id)."""
    transform.set_random_seed(int(np.random.default_rng([seed, int(image_id)]).integers(0, 2**31 - 1)))
    return transform(image=img)["image"]


@torch.no_grad()
def extract_features(encoder: torch.nn.Module, cache: np.ndarray, crops: dict[str, np.ndarray],
                     pcfg: dict[str, Any], device: torch.device, amp: torch.dtype | None,
                     shift: tuple[A.Compose, int] | None = None) -> torch.Tensor:
    """(N, dim) float32 features for the crop set, in crop order. `shift` = (transform, seed) applies
    the stain shift to each image before cropping."""
    encoder.eval()
    window, patch = pcfg["window"], encoder.patch_embed.patch_size[0]
    prefix = encoder.num_prefix_tokens
    hc, wc = cache.shape[1:3]
    s = pcfg["long_side"] / max(hc, wc)
    w, h = round(wc * s), round(hc * s)
    feats = torch.zeros(len(crops["label"]), encoder.embed_dim)
    batch_x, batch_m, batch_i = [], [], []

    def flush():
        x = torch.stack(batch_x).to(device, non_blocking=True)
        m = torch.from_numpy(np.stack(batch_m)).to(device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            tok = encoder.forward_features(x)[:, prefix:].float()
        feats[batch_i] = ((tok * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)).cpu()
        batch_x.clear(), batch_m.clear(), batch_i.clear()

    current_row, img = None, None
    for i in range(len(crops["label"])):
        if crops["row"][i] != current_row:
            current_row = crops["row"][i]
            img = cv2.resize(np.asarray(cache[current_row]), (w, h), interpolation=cv2.INTER_AREA)
            if shift is not None:
                img = apply_stain_shift(img, *shift, crops["image_id"][i])
        (x0, y0), m = window_and_patch_mask(crops["box"][i], w, h, window, patch)
        batch_x.append(normalize(img[y0:y0 + window, x0:x0 + window]))
        batch_m.append(m)
        batch_i.append(i)
        if len(batch_x) == pcfg["batch_size"]:
            flush()
    if batch_x:
        flush()
    return feats


# --------------------------------------------------------------------------- classifiers

def knn_predict(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, n_classes: int,
                k: int, temperature: float, chunk: int = 4096) -> torch.Tensor:
    """Class probabilities from a cosine, exp(sim/T)-weighted kNN vote."""
    tr = F.normalize(train_x.float(), dim=1)
    k = min(k, len(tr))
    out = []
    for s in range(0, len(test_x), chunk):
        te = F.normalize(test_x[s:s + chunk].float(), dim=1)
        sim, idx = (te @ tr.t()).topk(k, dim=1)
        votes = torch.zeros(len(te), n_classes, device=te.device)
        votes.scatter_add_(1, train_y[idx], torch.exp(sim / temperature))
        out.append(votes / votes.sum(1, keepdim=True))
    return torch.cat(out)


def linear_predict(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, n_classes: int,
                   l2: float, max_iter: int) -> torch.Tensor:
    """Class probabilities from a class-balanced, L2-regularised multinomial logistic regression."""
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
    xtr, xte = (train_x - mean) / std, (test_x - mean) / std
    counts = torch.bincount(train_y, minlength=n_classes).float()
    cw = torch.where(counts > 0, len(train_y) / (n_classes * counts.clamp_min(1)), torch.zeros_like(counts))
    lin = torch.nn.Linear(xtr.shape[1], n_classes).to(xtr.device)
    torch.nn.init.zeros_(lin.weight), torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(xtr), train_y, weight=cw) + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return lin(xte).softmax(1)


# --------------------------------------------------------------------------- metrics

def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count 1/2)."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _average_ranks(np.asarray(s, float))
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AUPRC as average precision over distinct score thresholds (step-wise, no interpolation)."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, float), kind="mergesort")
    ys, ss = y[order], np.asarray(s, float)[order]
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]   # last index of each distinct score
    tp = np.cumsum(ys)[last]
    precision = tp / (last + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def per_class_recall(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(n_classes)])


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    f1 = []
    for c in range(n_classes):
        if not (y == c).any():
            continue
        tp = np.sum((pred == c) & (y == c))
        fp, fn = np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def compute_metrics(y: np.ndarray, probs: np.ndarray, infected: np.ndarray, names: list[str]) -> dict[str, float]:
    k = len(names)
    pred = probs.argmax(1)
    y_bin, s_bin = infected[y], probs[:, infected].sum(1)
    rec = per_class_recall(y, pred, k)
    bin_rec = [np.mean(infected[pred][y_bin == v] == v) for v in (False, True) if (y_bin == v).any()]
    out = {"auroc_infected": auroc(y_bin, s_bin), "auprc_infected": average_precision(y_bin, s_bin),
           "balanced_acc_binary": float(np.mean(bin_rec)),
           "balanced_acc": float(np.nanmean(rec)), "macro_f1": macro_f1(y, pred, k)}
    out.update({f"recall/{n}": float(r) for n, r in zip(names, rec)})
    return out


def bootstrap_metrics(y: np.ndarray, probs: np.ndarray, image_ids: np.ndarray, infected: np.ndarray,
                      names: list[str], n_resamples: int, seed: int) -> dict[str, dict[str, float]]:
    """95% percentile CIs, resampling val images with replacement."""
    groups = defaultdict(list)
    for i, im in enumerate(image_ids):
        groups[im].append(i)
    imgs = sorted(groups)
    members = [np.asarray(groups[im]) for im in imgs]
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(n_resamples):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(imgs), size=len(imgs))])
        for key, v in compute_metrics(y[idx], probs[idx], infected, names).items():
            samples[key].append(v)
    out = {}
    for key, v in samples.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        out[key] = ({"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} if v.size
                    else {"low": float("nan"), "high": float("nan")})
    return out


def stratified_subset(y: np.ndarray, fraction: float, n_classes: int, seed: int) -> np.ndarray:
    """Indices of a class-stratified `fraction` of y (at least one per present class)."""
    if fraction >= 1:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_classes):
        members = np.nonzero(y == c)[0]
        if len(members):
            idx.append(rng.choice(members, size=max(1, int(round(fraction * len(members)))), replace=False))
    return np.sort(np.concatenate(idx))


# --------------------------------------------------------------------------- protocol

class Prober:
    """Builds the train / val crop sets once; `evaluate(encoder)` extracts features and fits probes."""

    def __init__(self, cfg: dict[str, Any], data_cfg: dict[str, Any], caches: dict[str, Path], resolve,
                 device: torch.device):
        self.cfg, self.pcfg, self.device = cfg, cfg["probe"], device
        self.names, _, self.infected = probe_classes(data_cfg, self.pcfg["exclude"])
        self.caches = {s: np.load(p, mmap_mode="r") for s, p in caches.items()}
        self.crops = {}
        for split, path in caches.items():
            index = json.loads(path.with_name(f"{split}_index.json").read_text(encoding="utf-8"))
            self.crops[split] = build_crop_set(split, data_cfg, self.pcfg, index, resolve)
        self.amp = torch.bfloat16 if device.type == "cuda" else None

    def counts(self) -> dict[str, dict[str, int]]:
        return {s: {n: int((c["label"] == i).sum()) for i, n in enumerate(self.names)} for s, c in self.crops.items()}

    def features(self, encoder: torch.nn.Module, stain_shift: bool = False) -> dict[str, torch.Tensor]:
        """Clean train / val features, plus `val_stain` (stain-shifted val crops) if requested."""
        feats = {s: extract_features(encoder, self.caches[s], self.crops[s], self.pcfg, self.device, self.amp)
                 for s in ("train", "val")}
        if stain_shift:
            sc = self.pcfg["stain_shift"]
            feats["val_stain"] = extract_features(encoder, self.caches["val"], self.crops["val"], self.pcfg,
                                                  self.device, self.amp, shift=(build_stain_shift(sc), sc["seed"]))
        return feats

    def _fit(self, feats: dict[str, torch.Tensor], train_idx: np.ndarray,
             eval_sets: tuple[str, ...] = ("val",)) -> dict[str, dict[str, np.ndarray]]:
        """{eval set: {method: val probabilities}}; one fit per method, applied to every eval set."""
        k = len(self.names)
        xtr = feats["train"][train_idx].to(self.device)
        ytr = torch.as_tensor(self.crops["train"]["label"][train_idx], device=self.device)
        xte = torch.cat([feats[s] for s in eval_sets]).to(self.device)
        kc, lc = self.pcfg["knn"], self.pcfg["linear"]
        probs = {"knn": knn_predict(xtr, ytr, xte, k, kc["k"], kc["temperature"]).cpu().numpy(),
                 "linear": linear_predict(xtr, ytr, xte, k, lc["l2"], lc["max_iter"]).cpu().numpy()}
        bounds = np.cumsum([0] + [len(feats[s]) for s in eval_sets])
        return {s: {m: p[bounds[j]:bounds[j + 1]] for m, p in probs.items()} for j, s in enumerate(eval_sets)}

    def _entry(self, probs: np.ndarray, full: bool) -> dict[str, Any]:
        y, img = self.crops["val"]["label"], self.crops["val"]["image_id"]
        entry = {"metrics": compute_metrics(y, probs, self.infected, self.names)}
        if full:
            entry["ci95"] = bootstrap_metrics(y, probs, img, self.infected, self.names,
                                              self.pcfg["bootstrap"], self.pcfg["seed"])
        return entry

    def evaluate(self, encoder: torch.nn.Module, full: bool = True,
                 feats: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        """Light (full=False): 100% labels, point estimates. Full: + bootstrap CIs, low-label and, if
        `feats` has `val_stain` (computed by default when the config has `stain_shift`), the
        stain-shifted probe."""
        feats = feats or self.features(encoder, stain_shift=full and "stain_shift" in self.pcfg)
        y = self.crops["val"]["label"]
        ytr = self.crops["train"]["label"]
        all_idx = np.arange(len(ytr))
        res: dict[str, Any] = {"classes": self.names, "infected": [n for n, f in zip(self.names, self.infected) if f],
                               "n_crops": self.counts()}
        fitted = self._fit(feats, all_idx, ("val", "val_stain") if "val_stain" in feats else ("val",))
        res["methods"] = {m: self._entry(p, full) for m, p in fitted["val"].items()}
        if "val_stain" in fitted:
            res["stain_shift"] = {"config": self.pcfg["stain_shift"],
                                  "methods": {m: self._entry(p, full) for m, p in fitted["val_stain"].items()}}
        if full:
            ll = self.pcfg["low_label"]
            res["low_label"] = {}
            for frac in ll["fractions"]:
                if frac >= 1:
                    continue
                runs = defaultdict(list)
                for s in range(ll["seeds"]):
                    idx = stratified_subset(ytr, frac, len(self.names), seed=self.pcfg["seed"] + s)
                    for method, probs in self._fit(feats, idx)["val"].items():
                        runs[method].append({"n_train": int(len(idx)),
                                             **compute_metrics(y, probs, self.infected, self.names)})
                res["low_label"][str(frac)] = {m: summarize_seeds(r) for m, r in runs.items()}
        return res


def summarize_seeds(runs: list[dict[str, float]]) -> dict[str, Any]:
    keys = runs[0].keys()
    return {"per_seed": runs,
            "mean": {k: float(np.nanmean([r[k] for r in runs])) for k in keys},
            "std": {k: float(np.nanstd([r[k] for r in runs])) for k in keys}}


def results_table(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat rows (eval_set, method, label_fraction, metric, value, ci_low, ci_high, std) for the CSV."""
    rows = []
    for eval_set, methods in (("val", res["methods"]), ("val_stain", res.get("stain_shift", {}).get("methods", {}))):
        for method, e in methods.items():
            for k, v in e["metrics"].items():
                ci = e.get("ci95", {}).get(k, {})
                rows.append({"eval_set": eval_set, "method": method, "label_fraction": 1.0, "metric": k,
                             "value": v, "ci_low": ci.get("low"), "ci_high": ci.get("high"), "std": None})
    for frac, by_method in res.get("low_label", {}).items():
        for method, s in by_method.items():
            for k, v in s["mean"].items():
                rows.append({"eval_set": "val", "method": method, "label_fraction": float(frac), "metric": k,
                             "value": v, "ci_low": None, "ci_high": None, "std": s["std"][k]})
    return rows


# --------------------------------------------------------------------------- lambda selection

def selection_criteria(res: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """(value, lower end of the tie interval) of each lambda-selection criterion, in priority order."""
    lin, stain = res["methods"]["linear"], res["stain_shift"]["methods"]["linear"]
    ll = res["low_label"]["0.1"]["linear"]
    return {"auprc_clean": (lin["metrics"]["auprc_infected"], lin["ci95"]["auprc_infected"]["low"]),
            "auprc_stain": (stain["metrics"]["auprc_infected"], stain["ci95"]["auprc_infected"]["low"]),
            "auprc_10pct": (ll["mean"]["auprc_infected"], ll["mean"]["auprc_infected"] - ll["std"]["auprc_infected"]),
            "macro_f1": (lin["metrics"]["macro_f1"], lin["metrics"]["macro_f1"])}


def select_run(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Lexicographic lambda selection with interval ties (see module docstring). `results`: {tag: probe json}."""
    crit = {t: selection_criteria(r) for t, r in results.items()}
    cands, steps = sorted(results), []
    for name in next(iter(crit.values())):
        leader = max(cands, key=lambda t: crit[t][name][0])
        low = crit[leader][name][1]
        cands = [t for t in cands if crit[t][name][0] >= low]
        steps.append({"criterion": name, "leader": leader, "tied": cands})
        if len(cands) == 1:
            break
    return {"winner": leader, "steps": steps, "criteria": {t: {k: v[0] for k, v in c.items()} for t, c in crit.items()}}
