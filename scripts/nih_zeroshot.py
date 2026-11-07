"""EXPLORATORY zero-shot evaluation of BBBC041-trained models on NIH-NLM thin smears (P. falciparum).

    python scripts/nih_zeroshot.py predict --run a_coco_detr_binary            # GPU: DETR on the NIH tiles
    python scripts/nih_zeroshot.py features --tag ssl_v2_b                     # GPU: crop classifier p(infected)
    python scripts/nih_zeroshot.py evaluate                                    # CPU: metrics + patient bootstrap

Nothing is fitted or selected on NIH. Frames are resized by `scale.factor` (cells at the BBBC041 eval pixel size,
prepare_nih.py) and cut into 800x600 tiles (the BBBC041 eval input); tiles almost outside the circular field of view
are skipped. predict: every query of every tile -> runs/nih_zeroshot/<run>/queries.npz, then (CPU) the
end-to-end detections (core filter + cross-tile seam NMS per class, every query kept otherwise, as on BBBC041) and
the two-stage candidates (core filter + class-agnostic NMS 0.5 per frame).
features: the probe crop features (224 window around the box at the resized-frame scale, patch-token mean) of the
NIH GT cells and of the candidates; the linear classifier is fitted on the clean BBBC041 TRAIN GT crops
(runs/two_stage/<tag>/train.pt) and p(infected) is stored for NIH and for the BBBC041 VAL candidates (score variant
and threshold are chosen on BBBC041 val exactly as in two_stage.py evaluate).
evaluate: infected AP50 (COCOeval, maxDets 300, ambiguous = ignore regions), P / R / F1 at the BBBC041-val
threshold, class-agnostic AP50 of the boxes, crop-level AUPRC / ECE of p(infected) on the NIH GT cells (two-stage:
the classifier on GT crops; end-to-end A: p(infected) of the best-IoU detection, 0 if none -> a different protocol),
parasitemia error per image; 95% CIs and paired differences from 1000 resamples of PATIENTS.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD, collate_detr, contiguous_labels, load_split_coco
from malaria_ssl.data.nih import agnostic_coco, in_core, subset_coco
from malaria_ssl.detr.build import build_model
from malaria_ssl.detr.trainer import AMP_DTYPES
from malaria_ssl.eval.detection_metrics import best_f1_threshold, match_per_class
from malaria_ssl.eval.inference import _boxes_xywh
from malaria_ssl.eval.paired import METRICS, ci, prepare_run, score, summarize_diff
from malaria_ssl.eval.two_stage import VARIANTS, fit_p_infected, scored_detections
from malaria_ssl.eval.zeroshot import (candidates, core_mask, counted, ece, end_to_end_detections, matched_p_infected,
                                       parasitemia_errors, parasitemia_estimate, patient_resamples, rows_of_images,
                                       to_coco_dets)
from malaria_ssl.ssl.encoders import load_encoder
from malaria_ssl.ssl.probe import average_precision, extract_features
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

log = logging.getLogger("nih_zeroshot")
INF = 2   # infected category id (BBBC041 binary = NIH binary)


def load_nih(cfg):
    d = resolve_path(cfg["nih"]["processed_dir"])
    coco = json.loads((d / "nih_binary.json").read_text(encoding="utf-8"))
    index = json.loads((d / "cache_index.json").read_text(encoding="utf-8"))
    tiles = json.loads((d / "tiles.json").read_text(encoding="utf-8"))
    return coco, d / index["file"], index, tiles


def limited(ids: list[int], limit: int | None) -> list[int]:
    return sorted(ids)[:limit] if limit else sorted(ids)


def out_root(cfg, args):
    """runs/nih_zeroshot, or runs/nih_zeroshot_smoke for --limit runs (never mixed with the full outputs)."""
    return resolve_path(cfg["eval"]["out_dir"] + ("_smoke" if args.limit else ""))


# --------------------------------------------------------------------------- predict (GPU)

class TileDataset(Dataset):
    """Model-input tiles of the resized NIH frames (ImageNet normalization, as DetectionDataset)."""

    def __init__(self, cache_path, tiles: list[dict], tile_ids: list[int]):
        self.path, self.tiles, self.ids, self.arr = cache_path, tiles, tile_ids, None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, k: int):
        if self.arr is None:
            self.arr = np.load(self.path, mmap_mode="r")
        t = self.tiles[self.ids[k]]
        img = np.asarray(self.arr[t["row"], t["y0"]:t["y1"], t["x0"]:t["x1"]])
        h, w = img.shape[:2]
        x = torch.from_numpy(((img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())
        return x, {"image_id": self.ids[k], "size": (h, w), "orig_size": (h, w), "scale": (1.0, 1.0)}


def cmd_predict(args, cfg) -> None:
    coco, cache, index, tiles = load_nih(cfg)
    factor, ecfg = cfg["scale"]["factor"], cfg["eval"]
    run_dir = resolve_path(f"runs/{args.run}")
    rcfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    out = out_root(cfg, args) / args.run
    out.mkdir(parents=True, exist_ok=True)
    keep_ids = set(limited([im["id"] for im in coco["images"]], args.limit))
    tile_ids = [i for i, t in enumerate(tiles["tiles"]) if t["keep"] and t["image_id"] in keep_ids]
    _, label_to_cat, names = contiguous_labels(coco)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(rcfg["model"], names, rcfg["seed"])
    model.load_state_dict(ck["model"])
    device = torch.device("cuda")
    model.to(device).eval()
    amp = AMP_DTYPES[rcfg["train"]["amp_dtype"]]
    loader = DataLoader(TileDataset(cache, tiles["tiles"], tile_ids), batch_size=rcfg["eval"]["batch_size"],
                        shuffle=False, collate_fn=collate_detr, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=True)
    img_ids, tile_idx, boxes, probs = [], [], [], []
    with torch.no_grad():
        for b, batch in enumerate(loader):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            pm = batch["pixel_mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                o = model(pv, pm)
            p = o.logits.float().softmax(-1).cpu().numpy()
            for k, t in enumerate(batch["labels"]):
                tile = tiles["tiles"][t["image_id"]]
                bx = np.asarray(_boxes_xywh(o.pred_boxes[k], t), np.float64)
                bx[:, 0] += tile["x0"]
                bx[:, 1] += tile["y0"]
                boxes.append(bx / factor)   # resized frame -> original NIH pixels
                probs.append(p[k])
                tile_idx.append(np.full(len(bx), t["image_id"]))
                img_ids.append(np.full(len(bx), tile["image_id"]))
            if (b + 1) % 200 == 0:
                log.info("%s: %d / %d batches", args.run, b + 1, len(loader))
    q = {"image_id": np.concatenate(img_ids), "tile": np.concatenate(tile_idx), "box": np.concatenate(boxes),
         "probs": np.concatenate(probs).astype(np.float32)}
    np.savez_compressed(out / "queries.npz", **q, checkpoint_epoch=ck["epoch"], label_to_cat=label_to_cat)
    del model
    torch.cuda.empty_cache()
    stitch(args.run, cfg, q, tiles, label_to_cat, out)


def stitch(run: str, cfg, q, tiles, label_to_cat, out) -> None:
    ecfg = cfg["eval"]
    keep = core_mask(q, tiles["tiles"], cfg["scale"]["factor"])
    e2e = end_to_end_detections(q, keep, ecfg["seam_nms_iou"], label_to_cat)
    cand = candidates(q, keep, ecfg["candidate_nms_iou"])
    np.savez_compressed(out / "e2e.npz", **e2e)
    np.savez_compressed(out / "candidates.npz", **cand)
    n_img = len(np.unique(q["image_id"]))
    log.info("%s: %d queries, %d in cores -> %d end-to-end detections, %d candidates (%.1f / frame)", run,
             len(keep), keep.sum(), len(e2e["score"]), len(cand["objectness"]), len(cand["objectness"]) / n_img)


# --------------------------------------------------------------------------- features (GPU, classifier on CPU)

def gt_cells(coco) -> dict[str, np.ndarray]:
    """Labelled NIH cells (ignore regions excluded), annotation-id order."""
    anns = sorted((a for a in coco["annotations"] if not a["iscrowd"]), key=lambda a: a["id"])
    return {"ann_id": np.array([a["id"] for a in anns]), "image_id": np.array([a["image_id"] for a in anns]),
            "box": np.array([a["bbox"] for a in anns], float), "infected": np.array([a["category_id"] == INF for a in anns])}


def cmd_features(args, cfg) -> None:
    coco, cache, index, _ = load_nih(cfg)
    ecfg = cfg["eval"]
    scfg = load_experiment(cfg["probe_config"])
    factor = cfg["scale"]["factor"]
    keep_ids = set(limited([im["id"] for im in coco["images"]], args.limit))
    row_of = {i: r for r, i in enumerate(index["ids"])}
    gt = gt_cells(coco)
    sel = np.isin(gt["image_id"], list(keep_ids))
    sets = {"gt": (gt["image_id"][sel], gt["box"][sel])}
    for run in ecfg["runs"]:
        c = np.load(out_root(cfg, args) / run / "candidates.npz")
        m = np.isin(c["image_id"], list(keep_ids))
        sets[run] = (c["image_id"][m], c["box"][m])
    names = list(sets)
    image_id = np.concatenate([sets[n][0] for n in names])
    box = np.concatenate([sets[n][1] for n in names]) * factor
    order = np.argsort([row_of[int(i)] for i in image_id], kind="stable")   # every frame decoded once
    crops = {"box": np.c_[box[:, :2], box[:, :2] + box[:, 2:]][order].astype(np.float32),
             "row": np.array([row_of[int(i)] for i in image_id])[order], "image_id": image_id[order],
             "label": np.zeros(len(order), np.int64)}
    pcfg = {**scfg["probe"], "long_side": max(index["shape"][1:3])}   # crops at the resized-frame scale
    kind, value = ecfg["encoder_specs"][args.tag].split(":", 1)   # ref:<name> | run:<ssl run dir>
    enc, _, source = load_encoder(scfg, **({"ref": value} if kind == "ref" else {"run": str(resolve_path(value))}),
                                  tag=args.tag)
    norm = (source["input_mean"], source["input_std"]) if "input_mean" in source else None
    device = torch.device("cuda")
    enc.to(device)
    f = extract_features(enc, np.load(cache, mmap_mode="r"), crops, pcfg, device, torch.bfloat16, input_norm=norm)
    enc.cpu()
    feats = torch.zeros_like(f)
    feats[torch.as_tensor(order)] = f
    sizes = [len(sets[n][0]) for n in names]
    nih = dict(zip(names, torch.split(feats, sizes)))
    d = resolve_path("runs/two_stage") / args.tag
    tr = torch.load(d / "train.pt", weights_only=False)
    if ecfg["training"] != "clean":
        raise SystemExit("only the clean crop classifier is implemented here")
    val = {run: torch.load(d / f"{run}_val.pt", weights_only=False)["features"] for run in ecfg["runs"]}
    evals = [val[r] for r in ecfg["runs"]] + [nih[n] for n in names]
    p = fit_p_infected(tr["clean"], tr["infected"], evals, scfg["probe"]["linear"]["l2"],
                       scfg["probe"]["linear"]["max_iter"])
    k = len(ecfg["runs"])
    outp = {f"bbbc_val_{r}": p[j] for j, r in enumerate(ecfg["runs"])}
    outp.update({f"nih_{n}": p[k + j] for j, n in enumerate(names)})
    out = out_root(cfg, args) / "features"
    out.mkdir(parents=True, exist_ok=True)
    if args.save_features:   # adaptation (scripts/adapt.py): the crop features themselves, fp16
        torch.save({**{n: nih[n].half() for n in names}, "gt_ann_id": torch.as_tensor(gt["ann_id"][sel]),
                    "source": source}, out / f"{args.tag}_features.pt")
        old = out / f"{args.tag}.npz"
        if old.exists():   # the evaluated p(infected) stay; record how well the re-extraction reproduces them
            f = np.load(old)
            dev = {k: float(np.abs(f[k] - v).max()) for k, v in outp.items()}
            (out / f"{args.tag}_reextract_check.json").write_text(json.dumps(dev, indent=1), encoding="utf-8")
            log.info("%s: features saved; max |p - saved p| %s", args.tag, dev)
            return
    np.savez_compressed(out / f"{args.tag}.npz", **outp,
                        gt_ann_id=gt["ann_id"][sel], source=json.dumps(source))
    log.info("%s: NIH crops %s", args.tag, dict(zip(names, sizes)))


# --------------------------------------------------------------------------- evaluate (CPU)

def top_k_per_image(dets: list[dict], k: int) -> list[dict]:
    """COCOeval keeps the top maxDets detections per image and class; pre-cutting gives the same AP, faster."""
    by = {}
    for d in dets:
        by.setdefault((d["image_id"], d["category_id"]), []).append(d)
    return [d for v in by.values() for d in sorted(v, key=lambda d: -d["score"])[:k]]


def prepare(gt, dets, cat, thr, ecfg):
    """prepare_run with AP on the top maxDets per image and P / R / F1 on all detections (as on BBBC041)."""
    r = prepare_run(gt, top_k_per_image(dets, ecfg["max_dets"]), cat, thr, ecfg["iou"])
    r["matches"] = match_per_class(gt, dets, ecfg["iou"])[cat]
    return r


def bbbc_val(run: str):
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gt = load_split_coco("val", "binary", dcfg, resolve_path)
    cands = json.loads((resolve_path(f"runs/{run}") / "candidates_val.json").read_text(encoding="utf-8"))["candidates"]
    return gt, cands


def select_variant(gt_val, cands_val, p_val, ecfg):
    """Score variant by BBBC041 val infected AP50, threshold by val max F1 (two_stage.py classify)."""
    ids = [im["id"] for im in gt_val["images"]]
    best = None
    for v in VARIANTS:
        dets = scored_detections(cands_val, p_val, v, INF)
        thr = best_f1_threshold(match_per_class(gt_val, dets, 0.5)[INF])[0]
        ap = score(prepare_run(gt_val, dets, INF, thr), ids)["AP50"]
        if best is None or ap > best[2]:
            best = (v, thr, ap)
    return best


def cmd_evaluate(args, cfg) -> None:
    coco, _, _, tiles = load_nih(cfg)
    ecfg, factor = cfg["eval"], cfg["scale"]["factor"]
    out_dir = out_root(cfg, args)
    all_ids = limited([im["id"] for im in coco["images"]], args.limit)
    coco = subset_coco(coco, set(all_ids))
    agn = agnostic_coco(coco)
    patient = {im["id"]: im["patient_id"] for im in coco["images"]}
    subset_ids = {"all": all_ids, **{s: [im["id"] for im in coco["images"] if im["subset"] == s] for s in ("polys", "points")}}
    subset_ids = {s: v for s, v in subset_ids.items() if s in ecfg["subsets"] and v}
    samples = {s: patient_resamples(patient, ids, args.n or ecfg["bootstrap"], ecfg["seed"]) for s, ids in subset_ids.items()}
    true_par = {r["image_id"]: r["parasitemia"] for r in pd.read_csv(
        resolve_path(cfg["nih"]["processed_dir"]) / "parasitemia_images.csv").to_dict("records")}
    gt = gt_cells(coco)
    suffix = "_smoke" if args.limit else ""

    points, crop_points, selection = {}, {}, []   # method key -> (prepared run, parasitemia estimate)
    feats = {t: np.load(out_dir / "features" / f"{t}.npz") for t in ecfg["encoders"]}
    for t, f in feats.items():
        if not np.array_equal(f["gt_ann_id"], gt["ann_id"]):
            raise SystemExit(f"{t}: GT crop order differs")
        crop_points[("encoder", t)] = f["nih_gt"]
    for run in ecfg["runs"]:
        d = out_dir / run
        e2e, cand = dict(np.load(d / "e2e.npz")), dict(np.load(d / "candidates.npz"))
        for x in (e2e, cand):
            m = np.isin(x["image_id"], all_ids)
            for k in x:
                x[k] = x[k][m]
        gt_val, cands_val = bbbc_val(run)
        # cells: class-agnostic boxes, threshold = max F1 of the class-agnostic matching on BBBC041 val
        val_agn = {**gt_val, "categories": [{"id": 1, "name": "cell"}],
                   "annotations": [{**a, "category_id": 1} for a in gt_val["annotations"]]}
        tau = best_f1_threshold(match_per_class(val_agn, [{**c, "category_id": 1, "score": c["objectness"]}
                                                           for c in cands_val], 0.5)[1])[0]
        cand_cnt = counted(cand["image_id"], cand["box"], coco, ecfg["iou"])
        cell = (cand["objectness"] >= tau) & cand_cnt
        points[(run, "agnostic")] = (prepare(agn, to_coco_dets(cand["image_id"], cand["box"], cand["objectness"], 1),
                                             1, tau, ecfg), None)
        thr_e2e = json.loads((resolve_path(f"runs/{run}") / "thresholds.json").read_text(encoding="utf-8"))["infected"]
        e2e_dets = to_coco_dets(e2e["image_id"], e2e["box"], e2e["score"], e2e["category_id"])
        inf = (e2e["category_id"] == INF) & (e2e["score"] >= thr_e2e) & counted(e2e["image_id"], e2e["box"], coco, ecfg["iou"])
        points[(run, "e2e")] = (prepare(coco, [x for x in e2e_dets if x["category_id"] == INF], INF, thr_e2e, ecfg),
                                parasitemia_estimate(all_ids, e2e["image_id"][inf], cand["image_id"][cell]))
        crop_points[(run, "e2e")] = matched_p_infected(gt["image_id"], gt["box"], e2e["image_id"], e2e["box"],
                                                       e2e["probs"][:, 1], ecfg["iou"])
        selection.append({"run": run, "method": "e2e", "variant": "-", "threshold": thr_e2e, "cell_threshold": tau})
        for t, f in feats.items():
            p_val, p = f[f"bbbc_val_{run}"], f[f"nih_{run}"]
            if len(p) != len(cand["objectness"]):
                raise SystemExit(f"{t} {run}: features do not match the candidates")
            variant, thr, val_ap = select_variant(gt_val, cands_val, p_val, ecfg)
            s = cand["objectness"] * p if variant == "det_x_p" else p
            inf = (s >= thr) & cand_cnt
            points[(run, t)] = (prepare(coco, to_coco_dets(cand["image_id"], cand["box"], s, INF), INF, thr, ecfg),
                                parasitemia_estimate(all_ids, cand["image_id"][inf], cand["image_id"][cell]))
            selection.append({"run": run, "method": t, "variant": variant, "threshold": thr, "cell_threshold": tau,
                              "bbbc_val_AP50": val_ap})
            log.info("%s %s: variant %s thr %.4f (BBBC val AP50 %.4f)", run, t, variant, thr, val_ap)

    det_metrics = [*METRICS, "parasitemia_mae", "parasitemia_bias"]

    def det_score(key, sample):
        r, est = points[key]
        s = score(r, sample)
        if est is not None:
            s.update(parasitemia_errors(est, true_par, sample))
        return [s.get(m, np.nan) for m in det_metrics]

    crop_metrics = ["auprc", "ece"]

    def crop_score(p, sample):
        rows = rows_of_images(gt["image_id"], sample)
        return [average_precision(gt["infected"][rows].astype(int), p[rows]), ece(gt["infected"][rows], p[rows], ecfg["ece_bins"])]

    methods = ["e2e", "agnostic", *ecfg["encoders"]]
    table, crop_table, diffs = [], [], []
    for sname, ids in subset_ids.items():
        log.info("bootstrap %s: %d images, %d patients", sname, len(ids), len({patient[i] for i in ids}))
        pt = {k: np.array(det_score(k, ids)) for k in points}
        bt = {k: np.array([det_score(k, s) for s in samples[sname]]) for k in points}
        for m in methods:   # seed mean of the box runs, paired on the same resamples
            pt[("a_3s", m)] = np.mean([pt[(r, m)] for r in ecfg["runs"]], axis=0)
            bt[("a_3s", m)] = np.mean([bt[(r, m)] for r in ecfg["runs"]], axis=0)
        cpt = {k: np.array(crop_score(p, ids)) for k, p in crop_points.items()}
        cbt = {k: np.array([crop_score(p, s) for s in samples[sname]]) for k, p in crop_points.items()}
        cpt[("a_3s", "e2e")] = np.mean([cpt[(r, "e2e")] for r in ecfg["runs"]], axis=0)
        cbt[("a_3s", "e2e")] = np.mean([cbt[(r, "e2e")] for r in ecfg["runs"]], axis=0)
        base = {"subset": sname, "n_images": len(ids), "n_patients": len({patient[i] for i in ids})}
        for (run, m), v in pt.items():
            for j, met in enumerate(det_metrics):
                if np.isfinite(v[j]):
                    lo, hi = ci(bt[(run, m)][:, j])
                    table.append({**base, "run": run, "method": m, "metric": met, "value": v[j], "ci_low": lo, "ci_high": hi})
        for (who, m), v in cpt.items():
            protocol = "classifier on GT crops" if who == "encoder" else "best-IoU detection, 0 if none"
            for j, met in enumerate(crop_metrics):
                lo, hi = ci(cbt[(who, m)][:, j])
                crop_table.append({**base, "method": m if who == "encoder" else f"e2e_{who}", "protocol": protocol,
                                   "metric": met, "value": v[j], "ci_low": lo, "ci_high": hi})
        for pair in ecfg["pairs"]:
            a, b = pair.split(":")
            for run in [*ecfg["runs"], "a_3s"]:
                for j, met in enumerate(det_metrics):
                    if np.isfinite(pt[(run, a)][j]) and np.isfinite(pt[(run, b)][j]):
                        d = summarize_diff(pt[(run, a)][j], pt[(run, b)][j], bt[(run, a)][:, j], bt[(run, b)][:, j])
                        diffs.append({**base, "run": run, "pair": f"{a} - {b}", "metric": met, **d,
                                      "excludes_0": d["low"] > 0 or d["high"] < 0})
            if "e2e" not in (a, b):   # crop-level: encoder - encoder only (the e2e protocol differs)
                for j, met in enumerate(crop_metrics):
                    d = summarize_diff(cpt[("encoder", a)][j], cpt[("encoder", b)][j], cbt[("encoder", a)][:, j],
                                       cbt[("encoder", b)][:, j])
                    diffs.append({**base, "run": "-", "pair": f"{a} - {b}", "metric": f"crop_{met}", **d,
                                  "excludes_0": d["low"] > 0 or d["high"] < 0})

    # coverage diagnostics: labelled cells whose centre is in a skipped tile's core; frames with > maxDets cells
    skipped = [t for t in tiles["tiles"] if not t["keep"] and t["image_id"] in set(all_ids)]
    lost = sum(in_core(gt["box"][gt["image_id"] == t["image_id"]] * factor, t["core"]).sum() for t in skipped)
    n_cells = pd.Series(gt["image_id"]).value_counts()
    meta = {"exploratory": True, "n_resamples": len(next(iter(samples.values()))), "seed": ecfg["seed"],
            "resampling": "patients", "skipped_tiles": len(skipped), "gt_cells_in_skipped_tiles": int(lost),
            "frames_with_more_cells_than_max_dets": int((n_cells > ecfg["max_dets"]).sum()),
            "n_gt_cells": int(len(gt["ann_id"])), "n_gt_infected": int(gt["infected"].sum()),
            "n_ignore_regions": int(sum(1 for a in agn["annotations"] if a["iscrowd"])), "scale_factor": factor}
    out = resolve_path(ecfg["tables"] + suffix)
    pd.DataFrame(table).to_csv(f"{out}_runs.csv", index=False, float_format="%.4f")
    pd.DataFrame(crop_table).to_csv(f"{out}_crops.csv", index=False, float_format="%.4f")
    pd.DataFrame(diffs).to_csv(f"{out}_diff.csv", index=False, float_format="%.4f")
    pd.DataFrame(selection).to_csv(f"{out}_selection.csv", index=False, float_format="%.4f")
    (out.parent / f"{out.name}.json").write_text(json.dumps(
        {"meta": meta, "selection": selection, "runs": table, "crops": crop_table, "diffs": diffs},
        indent=1, default=float), encoding="utf-8")
    log.info("meta %s", meta)
    print(f"wrote {out}_runs.csv, {out}_crops.csv, {out}_diff.csv, {out}_selection.csv, {out}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/nih_zeroshot.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N NIH images only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--workers", type=int, default=4)
    f = sub.add_parser("features")
    f.add_argument("--tag", required=True)
    f.add_argument("--save-features", action="store_true",
                   help="also save the crop features (<tag>_features.pt); an existing <tag>.npz is kept and compared")
    e = sub.add_parser("evaluate")
    e.add_argument("--n", type=int, default=None, help="override the number of resamples (smoke)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve_path(args.config))
    {"predict": cmd_predict, "features": cmd_features, "evaluate": cmd_evaluate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
