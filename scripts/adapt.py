"""EXPLORATORY unlabeled target adaptation of the two-stage scheme (configs/adapt.yaml, src/malaria_ssl/eval/adapt.py).

    python scripts/adapt.py select                         # CPU: grid on stain-shifted BBBC041 val -> adapt_selection.csv
    python scripts/adapt.py evaluate --target bbbc_test    # CPU: needs `selected` in the config
    python scripts/adapt.py evaluate --target nih

Reads only saved crop features: runs/two_stage/<tag>/{train, <run>_val, <run>_val_stain, <run>_test}.pt and
runs/nih_zeroshot/features/<tag>_features.pt (configs/queues/adapt_features.txt). The linear crop classifier (probe
hyperparameters) is fitted once per encoder on the clean BBBC041 train GT crops. Every method sees only unlabeled
target crops (see the config header); target labels are read only for the metrics.
evaluate: infected AP50 (COCOeval, maxDets from the config; NIH ambiguous = ignore regions), P / R / F1 at the
method's threshold, crop AUPRC / ECE of p(infected) on the labelled cells matched to a candidate (best IoU >= 0.5),
parasitemia MAE / bias per image and per patient (pooled counts), screening AUROC (patient has >= 1 infected cell;
BBBC041: image), all with 1000 paired resamples (NIH patients, BBBC041 images) and differences to `none`;
NIH: the same per true image-parasitemia bin for `bins_methods`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml

from malaria_ssl.data.detection import load_split_coco
from malaria_ssl.eval.adapt import (adabn, auroc_or_nan, bbse_prior, cbr, coral, em_prior, irc,
                                    match_gt_to_candidates, prior_adjust, quantile_threshold)
from malaria_ssl.eval.bootstrap import ap50_from_per_image
from malaria_ssl.eval.detection_metrics import best_f1_threshold, match_per_class, prf_at
from malaria_ssl.eval.paired import prepare_run, summarize_diff
from malaria_ssl.eval.zeroshot import counted, to_coco_dets
from malaria_ssl.ssl.probe import linear_fit
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

log = logging.getLogger("adapt")
INF = 2   # infected category id (BBBC041 binary = NIH binary)
METRICS = ["AP50", "precision", "recall", "f1", "crop_auprc", "crop_ece", "par_img_mae", "par_img_bias",
           "par_pat_mae", "par_pat_bias", "screen_auroc"]
BIN_METRICS = ["AP50", "precision", "recall", "crop_auprc", "par_img_mae", "par_img_bias"]


def sha256(path) -> str:
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- data

def classifier(tag: str, pcfg):
    tr = torch.load(resolve_path("runs/two_stage") / tag / "train.pt", weights_only=False)
    f = linear_fit(tr["clean"].float(), tr["infected"].long(), 2, pcfg["linear"]["l2"], pcfg["linear"]["max_iter"])
    return lambda x: f(x.float())[:, 1].numpy().astype(np.float64)


def bbbc_split(split: str, run: str, tag: str, limit=None, stain=False):
    """Candidates of a BBBC041 split: image_id, box (xywh), objectness, features; and the binary GT."""
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gt = load_split_coco(split, "binary", dcfg, resolve_path)
    cands = json.loads((resolve_path(f"runs/{run}") / f"candidates_{split}.json").read_text(encoding="utf-8"))["candidates"]
    name = f"{run}_{split}_stain.pt" if stain else f"{run}_{split}.pt"
    x = torch.load(resolve_path("runs/two_stage") / tag / name, weights_only=False)["features"]
    if len(x) != len(cands):
        raise SystemExit(f"{tag} {name}: features do not match the candidates")
    c = {"image_id": np.array([d["image_id"] for d in cands]), "box": np.array([d["bbox"] for d in cands], float),
         "obj": np.array([d["objectness"] for d in cands], float), "x": x.float()}
    if limit:
        keep = set(sorted(im["id"] for im in gt["images"])[:limit])
        gt = {**gt, "images": [im for im in gt["images"] if im["id"] in keep],
              "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}
        m = np.isin(c["image_id"], list(keep))
        c = {k: v[torch.as_tensor(m)] if k == "x" else v[m] for k, v in c.items()}
    return c, gt


def cell_threshold(gt, c) -> float:
    """Class-agnostic max-F1 objectness threshold on BBBC041 val (as nih_zeroshot.py)."""
    agn = {**gt, "categories": [{"id": 1, "name": "cell"}], "annotations": [{**a, "category_id": 1} for a in gt["annotations"]]}
    dets = to_coco_dets(c["image_id"], c["box"], c["obj"], 1)
    return best_f1_threshold(match_per_class(agn, dets, 0.5)[1])[0]


def gt_arrays(gt):
    anns = sorted((a for a in gt["annotations"] if not a.get("iscrowd", 0)), key=lambda a: a["id"])
    return {"image_id": np.array([a["image_id"] for a in anns]), "box": np.array([a["bbox"] for a in anns], float),
            "infected": np.array([a["category_id"] == INF for a in anns])}


def cell_labels(gt, c, iou) -> np.ndarray:
    """Per candidate: 1 infected / 0 uninfected if it is the best-IoU candidate of a labelled cell, else -1."""
    g = gt_arrays(gt)
    m = match_gt_to_candidates(g["image_id"], g["box"], c["image_id"], c["box"], iou)
    lab = np.full(len(c["obj"]), -1)
    lab[m[m >= 0]] = g["infected"][m >= 0].astype(int)
    return lab


def load_target(target: str, run: str, tag: str, cfg, limit=None):
    """Target candidates, GT, true parasitemia per image / patient and the image -> patient map."""
    if target == "bbbc_test":
        c, gt = bbbc_split("test", run, tag, limit)
        g = gt_arrays(gt)
        ids = sorted(im["id"] for im in gt["images"])
        n_inf = pd.Series(g["infected"]).groupby(g["image_id"]).sum()
        n_all = pd.Series(g["infected"]).groupby(g["image_id"]).size()
        true_img = {i: float(n_inf.get(i, 0) / n_all[i]) if i in n_all else 0.0 for i in ids}
        return c, gt, ids, {i: str(i) for i in ids}, true_img, {str(i): v for i, v in true_img.items()}
    ncfg = load_config(resolve_path(cfg["nih_config"]))
    d = resolve_path(ncfg["nih"]["processed_dir"])
    gt = json.loads((d / "nih_binary.json").read_text(encoding="utf-8"))
    ids = sorted(im["id"] for im in gt["images"])[:limit] if limit else sorted(im["id"] for im in gt["images"])
    keep = set(ids)
    gt = {**gt, "images": [im for im in gt["images"] if im["id"] in keep],
          "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}
    nih_out = resolve_path(ncfg["eval"]["out_dir"] + ("_smoke" if limit else ""))   # as nih_zeroshot.py --limit
    cand = np.load(nih_out / run / "candidates.npz")
    feats = torch.load(nih_out / "features" / f"{tag}_features.pt", weights_only=False)
    x = feats[run]
    if len(x) != len(cand["objectness"]):
        raise SystemExit(f"{tag} {run}: NIH features do not match the candidates")
    m = np.isin(cand["image_id"], ids)
    c = {"image_id": cand["image_id"][m], "box": cand["box"][m].astype(float), "obj": cand["objectness"][m].astype(float),
         "x": x[torch.as_tensor(m)].float()}
    patient = {im["id"]: im["patient_id"] for im in gt["images"]}
    true_img = {r["image_id"]: r["parasitemia"] for r in pd.read_csv(d / "parasitemia_images.csv").to_dict("records")
                if r["image_id"] in keep}
    pats = set(patient.values())
    true_pat = {r["patient_id"]: r["parasitemia"] for r in pd.read_csv(d / "parasitemia_patients.csv").to_dict("records")
                if r["patient_id"] in pats}
    return c, gt, ids, patient, true_img, true_pat


# --------------------------------------------------------------------------- methods

def transform(base: str, params: dict, c: dict, cells: np.ndarray, group: np.ndarray, src_x, src_p, predict):
    """Adapted features of `c` (base feature method); info about fall-backs."""
    if base == "none":
        return c["x"], {}
    if base == "adabn":
        return adabn(c["x"], cells, src_x), {}
    if base == "coral":
        return coral(c["x"], cells, src_x, params["lam"]), {}
    if base == "cbr":
        x, fell = cbr(c["x"], cells, src_x, src_p, predict, params["mode"], params["iters"])
        return x, {"cbr_fallback": fell}
    if base in ("irc", "irc_patient"):
        return irc(c["x"], cells, group, src_x, params["shrink"], params["scale"]), {}
    raise ValueError(base)


def split_method(m: str) -> tuple[str, str]:
    base, _, prior = m.partition("+")
    return base, prior


def params_of(base: str, selected: dict) -> dict:
    key = "irc" if base == "irc_patient" else base
    return selected.get(key, {}) if selected else {}


def grid_points(grid: dict) -> list[tuple[str, dict]]:
    out = [("none", {}), ("adabn", {})]
    for base, g in grid.items():
        keys = list(g)
        out += [(base, dict(zip(keys, vals))) for vals in product(*(g[k] for k in keys))]
    return out


def prepare(gt, dets, thr, cfg):
    """AP on the top maxDets per image, P / R / F1 matches on all detections (as nih_zeroshot.py)."""
    by = {}
    for d in dets:
        by.setdefault(d["image_id"], []).append(d)
    top = [d for v in by.values() for d in sorted(v, key=lambda d: -d["score"])[:cfg["max_dets"]]]
    r = prepare_run(gt, top, INF, thr, cfg["iou"])
    r["matches"] = match_per_class(gt, dets, cfg["iou"])[INF]
    return r


# --------------------------------------------------------------------------- select (stain-shifted BBBC041 val)

def cmd_select(args, cfg) -> None:
    pcfg = load_experiment(cfg["probe_config"])["probe"]
    rows = []
    points = grid_points(cfg["grid"])
    for tag in cfg["encoders"]:
        predict = classifier(tag, pcfg)
        for run in cfg["runs"]:
            src, gt = bbbc_split("val", run, tag, args.limit)
            tgt, _ = bbbc_split("val", run, tag, args.limit, stain=True)
            tau = cell_threshold(gt, src)
            src_cells, cells = src["obj"] >= tau, tgt["obj"] >= tau
            src_x, src_p = src["x"][torch.as_tensor(src_cells)], predict(src["x"][torch.as_tensor(src_cells)])
            ids = sorted(im["id"] for im in gt["images"])
            lab = cell_labels(gt, tgt, cfg["iou"])
            true_pi = float(lab[cells & (lab >= 0)].mean())
            for base, params in points:
                t0 = time.time()
                x, info = transform(base, params, tgt, cells, tgt["image_id"], src_x, src_p, predict)
                p = predict(x)
                s = tgt["obj"] * p
                dets = to_coco_dets(tgt["image_id"], tgt["box"], s, INF)
                thr, f1 = best_f1_threshold(match_per_class(gt, dets, cfg["iou"])[INF])
                r = prepare_run(gt, dets, INF, thr, cfg["iou"])
                ap = ap50_from_per_image(r["per_img"], ids, r["rec_thrs"])[INF]
                row = {"encoder": tag, "run": run, "method": base, "params": json.dumps(params, sort_keys=True),
                       "val_stain_AP50": ap, "val_stain_maxF1": f1, "threshold": thr, **info}
                if base == "none":   # prior estimates on the stain-shifted val (diagnostic; val labels allowed)
                    row.update({"true_cell_pi": true_pi, "em_pi": em_prior(p[cells])})
                rows.append(row)
                log.info("%s %s %s %s: AP50 %.4f maxF1 %.4f (%.1f s)", tag, run, base, params, ap, f1, time.time() - t0)
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "params"], sort=False)["val_stain_AP50"].agg(["mean", "std", "count"]).reset_index()
    agg["selected"] = False
    for base in cfg["grid"]:   # max mean AP50; exact ties -> first in grid order
        sub = agg[agg["method"] == base]
        agg.loc[sub["mean"].round(4).idxmax(), "selected"] = True
    suffix = "_smoke" if args.limit else ""
    out = resolve_path(cfg["tables"] + suffix)
    df.to_csv(f"{out}_selection_runs.csv", index=False, float_format="%.4f")
    agg.to_csv(f"{out}_selection.csv", index=False, float_format="%.4f")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("selected:", {r["method"]: json.loads(r["params"]) for _, r in agg[agg["selected"]].iterrows()})


def cmd_control(args, cfg) -> None:
    """Selected params on BBBC041 val only: stain-shifted val and clean val (val as its own target), thresholds as in
    `evaluate` (none: clean-val max F1; none_stainthr and feature methods: stain-val max F1 after the method)."""
    sel = check_selected(cfg)
    pcfg = load_experiment(cfg["probe_config"])["probe"]
    rows = []
    for tag in cfg["encoders"]:
        predict = classifier(tag, pcfg)
        for run in cfg["runs"]:
            src, gt = bbbc_split("val", run, tag, args.limit)
            stain, _ = bbbc_split("val", run, tag, args.limit, stain=True)
            tau = cell_threshold(gt, src)
            src_cells, stain_cells = src["obj"] >= tau, stain["obj"] >= tau
            src_x = src["x"][torch.as_tensor(src_cells)]
            src_p = predict(src_x)
            ids = sorted(im["id"] for im in gt["images"])
            for m in ["none", "none_stainthr", "adabn", "coral", "cbr", "irc"]:
                base = "none" if m == "none_stainthr" else m
                params = params_of(base, sel)
                scores = {}
                for dom, c, cells in (("stain", stain, stain_cells), ("clean", src, src_cells)):
                    x, _ = transform(base, params, c, cells, c["image_id"], src_x, src_p, predict)
                    scores[dom] = to_coco_dets(c["image_id"], c["box"], c["obj"] * predict(x), INF)
                matches = {d: match_per_class(gt, scores[d], cfg["iou"])[INF] for d in scores}
                best = {d: best_f1_threshold(matches[d]) for d in scores}
                thr = best["clean"][0] if m == "none" else best["stain"][0]
                row = {"encoder": tag, "run": run, "method": m, "params": json.dumps(params, sort_keys=True),
                       "threshold": thr}
                for d in scores:
                    r = prepare_run(gt, scores[d], INF, thr, cfg["iou"])
                    prf = prf_at(matches[d], thr, ids)
                    row.update({f"{d}_AP50": ap50_from_per_image(r["per_img"], ids, r["rec_thrs"])[INF],
                                f"{d}_F1": prf["f1"], f"{d}_P": prf["precision"], f"{d}_R": prf["recall"],
                                f"{d}_maxF1": best[d][1]})
                rows.append(row)
                log.info("%s %s %s: stain AP50 %.4f F1 %.4f | clean AP50 %.4f F1 %.4f", tag, run, m,
                         row["stain_AP50"], row["stain_F1"], row["clean_AP50"], row["clean_F1"])
    df = pd.DataFrame(rows)
    out = resolve_path(cfg["tables"] + ("_smoke" if args.limit else ""))
    df.to_csv(f"{out}_selection_control.csv", index=False, float_format="%.4f")
    cols = [c for c in df.columns if c.startswith(("stain_", "clean_"))]
    print(df.groupby("method", sort=False)[cols].agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.4f}"))


def check_selected(cfg) -> dict:
    sel = cfg.get("selected")
    if not sel:
        raise SystemExit("configs/adapt.yaml: `selected` is empty; run `select` and copy the choice first")
    agg = pd.read_csv(resolve_path(cfg["tables"] + "_selection.csv"))
    table = {r["method"]: json.loads(r["params"]) for _, r in agg[agg["selected"]].iterrows()}
    if table != sel:
        raise SystemExit(f"`selected` {sel} differs from adapt_selection.csv {table}")
    return sel


# --------------------------------------------------------------------------- evaluate

class Sampler:
    """Paired resamples: NIH patients (all images of a drawn patient), BBBC041 images (patient = image)."""

    def __init__(self, ids, patient, n, seed):
        self.ids = list(ids)
        self.pos = {i: k for k, i in enumerate(self.ids)}
        self.patients = sorted(set(patient[i] for i in self.ids))
        self.ppos = {p: k for k, p in enumerate(self.patients)}
        self.img_pat = np.array([self.ppos[patient[i]] for i in self.ids])
        by = {p: [] for p in self.patients}
        for i in self.ids:
            by[patient[i]].append(self.pos[i])
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n):
            draw = rng.choice(len(self.patients), size=len(self.patients), replace=True)
            w_pat = np.bincount(draw, minlength=len(self.patients))
            self.samples.append((w_pat[self.img_pat], w_pat))
        self.full = (np.ones(len(self.ids), int), np.ones(len(self.patients), int))


class Crops:
    """Weighted AUPRC / ECE over labelled cells; image multiplicities as weights (= duplicating the rows)."""

    def __init__(self, img_pos, y, p, bins):
        order = np.argsort(-p, kind="mergesort")
        self.img, self.y, self.p = img_pos[order], y[order].astype(float), p[order]
        self.last = np.r_[np.nonzero(np.diff(self.p))[0], len(self.p) - 1]
        self.bin = np.clip((self.p * bins).astype(int), 0, bins - 1)
        self.bins = bins

    def score(self, w_img):
        w = w_img[self.img].astype(float)
        pos = (w * self.y).sum()
        if pos == 0 or w.sum() == 0:
            return float("nan"), float("nan")
        ctp, cnt = np.cumsum(w * self.y)[self.last], np.cumsum(w)[self.last]
        keep = cnt > 0
        prec, rec = ctp[keep] / cnt[keep], ctp[keep] / pos
        ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
        n = np.bincount(self.bin, weights=w, minlength=self.bins)
        conf = np.bincount(self.bin, weights=w * self.p, minlength=self.bins)
        acc = np.bincount(self.bin, weights=w * self.y, minlength=self.bins)
        return ap, float(np.abs(acc - conf).sum() / w.sum())


def metrics(r, crops, est_img, true_img, est_pat, true_pat, pat_label, ids, w_img, w_pat, nih: bool):
    sample = [i for i, k in zip(ids, w_img) for _ in range(k)]
    ap = ap50_from_per_image(r["per_img"], sample, r["rec_thrs"])[INF]
    prf = prf_at(r["matches"], r["thr"], sample)
    auprc, ece = crops.score(w_img)
    e = est_img - true_img
    wi = w_img / max(w_img.sum(), 1)
    out = {"AP50": ap if ap >= 0 else float("nan"), "precision": prf["precision"], "recall": prf["recall"],
           "f1": prf["f1"], "crop_auprc": auprc, "crop_ece": ece,
           "par_img_mae": float((wi * np.abs(e)).sum()), "par_img_bias": float((wi * e).sum())}
    if nih:
        ep = est_pat - true_pat
        wp = w_pat / max(w_pat.sum(), 1)
        out.update({"par_pat_mae": float((wp * np.abs(ep)).sum()), "par_pat_bias": float((wp * ep).sum()),
                    "screen_auroc": auroc_or_nan(np.repeat(pat_label, w_pat), np.repeat(est_pat, w_pat))})
    else:   # BBBC041: no patients; screening per image
        out.update({"par_pat_mae": float("nan"), "par_pat_bias": float("nan"),
                    "screen_auroc": auroc_or_nan(np.repeat(true_img > 0, w_img), np.repeat(est_img, w_img))})
    return out


def cmd_evaluate(args, cfg) -> None:
    sel = check_selected(cfg)
    pcfg = load_experiment(cfg["probe_config"])["probe"]
    nih = args.target == "nih"
    methods = [m for m in cfg["methods"] if nih or split_method(m)[0] != "irc_patient"]
    suffix = "_smoke" if args.limit else ""
    out = resolve_path(f"{cfg['tables']}_{args.target}{suffix}")
    points, boots, bin_points, bin_boots, info_rows = {}, {}, {}, {}, []
    sampler = edges = None
    for tag in cfg["encoders"]:
        predict = classifier(tag, pcfg)
        for run in cfg["runs"]:
            t_run = time.time()
            src, gt_val = bbbc_split("val", run, tag)
            stain, _ = bbbc_split("val", run, tag, stain=True)
            tau = cell_threshold(gt_val, src)
            src_cells, stain_cells = src["obj"] >= tau, stain["obj"] >= tau
            src_x = src["x"][torch.as_tensor(src_cells)]
            src_p = predict(src_x)
            val_lab = cell_labels(gt_val, src, cfg["iou"])
            val_ids = sorted(im["id"] for im in gt_val["images"])

            c, gt, ids, patient, true_img_d, true_pat_d = load_target(args.target, run, tag, cfg, args.limit)
            if sampler is None:
                sampler = Sampler(ids, patient, args.n or cfg["bootstrap"], cfg["seed"])
                true_img = np.array([true_img_d[i] for i in ids])
                true_pat = np.array([true_pat_d[p] for p in sampler.patients])
                pat_label = true_pat > 0
                edges = np.asarray(cfg["parasitemia_bins"])
                img_bin = np.where(true_img == 0, 0, np.searchsorted(edges[1:], true_img, side="right") + 1)
            cells = c["obj"] >= tau
            cnt = counted(c["image_id"], c["box"], gt, cfg["iou"])   # NIH: ambiguous regions do not count (evaluation)
            img_pos = np.array([sampler.pos[int(i)] for i in c["image_id"]])
            groups = {"irc": c["image_id"], "irc_patient": np.array([patient[int(i)] for i in c["image_id"]])}
            g = gt_arrays(gt)
            gm = match_gt_to_candidates(g["image_id"], g["box"], c["image_id"], c["box"], cfg["iou"])
            gpos = np.array([sampler.pos[int(i)] for i in g["image_id"][gm >= 0]])
            n_cell = np.bincount(img_pos[cells & cnt], minlength=len(ids))
            feats_cache = {}
            for m in methods:
                t0 = time.time()
                base, prior = split_method(m)
                params = params_of(base, sel)
                key_base = "none" if base == "none_stainthr" else base
                if key_base not in feats_cache:
                    x, info = transform(key_base, params, c, cells, groups.get(key_base), src_x, src_p, predict)
                    feats_cache[key_base] = (predict(x), info)
                p, info = feats_cache[key_base]
                pi = float("nan")
                if prior:
                    if prior == "em":
                        pi = em_prior(p[cells])
                    else:   # tpr / fpr on labelled clean-val cells after the same base method (val as its own target)
                        xv, _ = transform(key_base, params, src, src["obj"] >= tau, src["image_id"], src_x, src_p, predict)
                        pv, lv = predict(xv), val_lab
                        ok = (src["obj"] >= tau) & (lv >= 0)
                        pi = bbse_prior(p[cells], float((pv[ok & (lv == 1)] >= 0.5).mean()),
                                        float((pv[ok & (lv == 0)] >= 0.5).mean()))
                    p = prior_adjust(p, pi)
                s = c["obj"] * p
                if prior:
                    thr = quantile_threshold(s[cells], pi)
                elif base == "none":   # clean-val max F1 (nih_zeroshot / two_stage)
                    pv = predict(src["x"])
                    thr = best_f1_threshold(match_per_class(gt_val, to_coco_dets(src["image_id"], src["box"],
                                                                                src["obj"] * pv, INF), cfg["iou"])[INF])[0]
                else:   # max F1 on stain-shifted val after the same method
                    vgroup = stain["image_id"]
                    xv, _ = transform(key_base, params, stain, stain_cells, vgroup, src_x, src_p, predict)
                    sv = stain["obj"] * predict(xv)
                    thr = best_f1_threshold(match_per_class(gt_val, to_coco_dets(stain["image_id"], stain["box"], sv, INF),
                                                            cfg["iou"])[INF])[0]
                r = prepare(gt, to_coco_dets(c["image_id"], c["box"], s, INF), thr, cfg)
                inf = (s >= thr) & cnt
                n_inf = np.bincount(img_pos[inf], minlength=len(ids))
                est_img = np.where(n_cell > 0, n_inf / np.maximum(n_cell, 1), 0.0)
                pi_cell = np.bincount(sampler.img_pat, weights=n_inf, minlength=len(sampler.patients))
                pc_cell = np.bincount(sampler.img_pat, weights=n_cell, minlength=len(sampler.patients))
                est_pat = np.where(pc_cell > 0, pi_cell / np.maximum(pc_cell, 1), 0.0)
                crops = Crops(gpos, g["infected"][gm >= 0], p[gm[gm >= 0]], cfg["ece_bins"])
                key = (run, tag, m)
                args_m = (r, crops, est_img, true_img, est_pat, true_pat, pat_label, ids)
                points[key] = metrics(*args_m, *sampler.full, nih)
                boots[key] = np.array([[metrics(*args_m, wi, wp, nih)[k] for k in METRICS] for wi, wp in sampler.samples])
                if nih and m in cfg["bins_methods"]:
                    for b in range(len(edges) + 1):
                        inb = (img_bin == b).astype(int)
                        if not inb.any():
                            continue
                        bp = metrics(*args_m, inb, sampler.full[1], nih)
                        bin_points[key + (b,)] = {k: bp[k] for k in BIN_METRICS}
                        bin_boots[key + (b,)] = np.array([[metrics(*args_m, wi * inb, wp, nih)[k] for k in BIN_METRICS]
                                                         for wi, wp in sampler.samples])
                info_rows.append({"run": run, "encoder": tag, "method": m, "params": json.dumps(params, sort_keys=True),
                                  "threshold": thr, "cell_threshold": tau, "pi_hat": pi,
                                  "pred_pos_frac_cells": float((s[cells] >= thr).mean()),
                                  "true_cell_pi": float(g["infected"].mean()), **info})
                log.info("%s %s %s: AP50 %.4f R %.4f par_img_bias %.4f pi %.4f (%.0f s)", run, tag, m,
                         points[key]["AP50"], points[key]["recall"], points[key]["par_img_bias"], pi, time.time() - t0)
            log.info("%s %s done in %.0f s", run, tag, time.time() - t_run)

    def with_means(pt, bt):   # seed mean of the box runs on the same resamples
        for k in list(pt):
            if k[0] != cfg["runs"][0]:
                continue
            rest = k[1:]
            members = [(r, *rest) for r in cfg["runs"]]
            if all(mk in pt for mk in members):
                pt[(cfg["mean_name"], *rest)] = {mm: float(np.mean([pt[mk][mm] for mk in members])) for mm in pt[k]}
                bt[(cfg["mean_name"], *rest)] = np.mean([bt[mk] for mk in members], axis=0)
        return pt, bt

    points, boots = with_means(points, boots)
    bin_points, bin_boots = with_means(bin_points, bin_boots)

    def ci(v):
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

    rows, diffs = [], []
    for (run, tag, m), pt in points.items():
        for j, k in enumerate(METRICS):
            lo, hi = ci(boots[(run, tag, m)][:, j])
            rows.append({"target": args.target, "run": run, "encoder": tag, "method": m, "metric": k,
                         "value": pt[k], "ci_low": lo, "ci_high": hi})
            if m != "none" and np.isfinite(pt[k]) and np.isfinite(points[(run, tag, "none")][k]):
                d = summarize_diff(pt[k], points[(run, tag, "none")][k], boots[(run, tag, m)][:, j],
                                   boots[(run, tag, "none")][:, j])
                diffs.append({"target": args.target, "run": run, "encoder": tag, "pair": f"{m} - none", "metric": k,
                              **d, "excludes_0": bool(d["low"] > 0 or d["high"] < 0)})
    bins_rows, bins_diffs = [], []
    labels = ["0", "(0, 1%)", "[1%, 5%)", "[5%, 20%)", ">= 20%"]
    for (run, tag, m, b), pt in bin_points.items():
        n_img = int((img_bin == b).sum())
        for j, k in enumerate(BIN_METRICS):
            lo, hi = ci(bin_boots[(run, tag, m, b)][:, j])
            bins_rows.append({"run": run, "encoder": tag, "method": m, "bin": labels[b], "n_images": n_img,
                              "metric": k, "value": pt[k], "ci_low": lo, "ci_high": hi})
            ref = (run, tag, "none", b)
            if m != "none" and np.isfinite(pt[k]) and np.isfinite(bin_points[ref][k]):
                d = summarize_diff(pt[k], bin_points[ref][k], bin_boots[(run, tag, m, b)][:, j], bin_boots[ref][:, j])
                bins_diffs.append({"run": run, "encoder": tag, "pair": f"{m} - none", "bin": labels[b], "n_images": n_img,
                                   "metric": k, **d, "excludes_0": bool(d["low"] > 0 or d["high"] < 0)})
    pd.DataFrame(rows).to_csv(f"{out}_runs.csv", index=False, float_format="%.4f")
    pd.DataFrame(diffs).to_csv(f"{out}_diff.csv", index=False, float_format="%.4f")
    pd.DataFrame(info_rows).to_csv(f"{out}_info.csv", index=False, float_format="%.4f")
    if nih:
        pd.DataFrame(bins_rows).to_csv(f"{out}_bins.csv", index=False, float_format="%.4f")
        pd.DataFrame(bins_diffs).to_csv(f"{out}_bins_diff.csv", index=False, float_format="%.4f")
    meta = {"exploratory": True, "target": args.target, "n_resamples": len(sampler.samples),
            "resampling": "patients" if nih else "images", "seed": cfg["seed"], "n_images": len(sampler.ids),
            "n_patients": len(sampler.patients) if nih else None, "selected": sel,
            "config_sha256": sha256(args.config), "limit": args.limit}
    (out.parent / f"{out.name}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {out}_runs.csv, _diff.csv, _info.csv{', _bins.csv, _bins_diff.csv' if nih else ''}, _meta.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/adapt.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images of every split / target")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    sub.add_parser("control", help="selected params on stain-shifted and clean BBBC041 val (no target data)")
    e = sub.add_parser("evaluate")
    e.add_argument("--target", required=True, choices=["nih", "bbbc_test"])
    e.add_argument("--n", type=int, default=None, help="override the number of resamples (smoke)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(resolve_path(args.config).read_text(encoding="utf-8"))
    {"select": cmd_select, "control": cmd_control, "evaluate": cmd_evaluate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
