"""Two-stage diagnostic: DETR boxes (class-agnostic) + frozen-encoder crop classifier (evaluation only).

    python scripts/two_stage.py candidates --run runs/a_coco_detr_binary            # GPU, val + test
    python scripts/two_stage.py candidates --run runs/a_coco_detr_binary --splits train   # phase 7 SSL cell boxes
    python scripts/two_stage.py features --runs a_coco_detr_binary b_vit_imagenet \
        --encoders ref:imagenet run:runs/ssl_con_only run:runs/ssl_mae_only run:runs/ssl_joint   # GPU
    python scripts/two_stage.py evaluate --runs a_coco_detr_binary b_vit_imagenet \
        --tags ref_imagenet ssl_con_only ssl_mae_only ssl_joint --encoder-runs ref_imagenet:b_vit_imagenet \
        ssl_con_only:b_vit_con ssl_mae_only:b_vit_mae ssl_joint:b_vit_joint --out reports/tables/phase4_two_stage   # CPU

candidates: every query, objectness = 1 - p(no-object), class-agnostic NMS (IoU `--nms-iou`, fixed 0.5) ->
runs/<run>/candidates_{val,test}.json.
features: probe crop features (224 window at long side 800, mean of patch tokens in the box) of the TRAIN GT
probe crops (clean, and stain-shifted with the probe's `stain_shift` transform, one fixed shift per train image)
and of the candidates -> runs/two_stage/<tag>/.
evaluate: binary linear classifier (probe hyperparameters) fitted on TRAIN GT crops, `clean` or `stain`
(clean + shifted train crops). Per (DETR run, encoder, training): the score variant (objectness x p(infected) or
p(infected)) is chosen by val infected AP50, the threshold by val max F1; both are applied unchanged to test.
Test: infected AP50 / P / R / F1 with 95% CIs and paired differences (shared resamples of test images) vs the
end-to-end DETR (its predictions_test.json at its own val threshold) and encoder - encoder two-stage (`--pairs`,
default joint - ImageNet).
--mean NAME=R1+R2+R3 (training seeds): mean of the members on the same resamples, each member at its own threshold /
score variant; two-stage means (all Ri in --runs) are compared with the end-to-end mean of every group.
--fold-merge NAME=TAG_F1+TAG_F2 (phase 7 cross-fold scheme, test only): TAG_Fk is an SSL encoder that saw the
unlabeled images of test fold k (checked against target.fold of runs/<TAG_Fk>/config.yaml). Each member gets its own
score variant and threshold on val as usual; NAME scores the fold-1 test images with TAG_F2 and the fold-2 images
with TAG_F1. The members' own whole-test rows are dropped (they include the seen fold), so TAG_Fk must not be in
--tags; NAME is usable in --pairs and in the two-stage --mean groups (test rows only). Cross-fold END-TO-END DETR
runs are merged by paired_detr_diff.py --fold-merge, not here.
--pairs with --mean: every encoder pair is also compared on the seed mean of each two-stage group (shared resamples).
folds (diagnostic): test metrics of each encoder on each test fold separately (seen / unseen by an SSL encoder),
same classifier / variant / threshold as evaluate; writes <out>.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
import yaml

from malaria_ssl.data.detection import DetectionDataset, build_transforms, load_split_coco
from malaria_ssl.detr.build import build_model
from malaria_ssl.detr.trainer import AMP_DTYPES, data_config, eval_loader
from malaria_ssl.eval.detection_metrics import best_f1_threshold, match_per_class
from malaria_ssl.eval.inference import predict_candidates
from malaria_ssl.eval.paired import (METRICS, bootstrap_table, ci, fold_merge, parse_fold_merges, prepare_run,
                                     resamples, score, summarize_diff)
from malaria_ssl.eval.two_stage import (VARIANTS, candidate_crops, fit_p_infected, nms_per_image,
                                        scored_detections)
from malaria_ssl.ssl.data import ensure_caches, load_image_folds
from malaria_ssl.ssl.encoders import load_encoder, ssl_run_target_fold
from malaria_ssl.ssl.probe import build_crop_set, build_stain_shift, extract_features, probe_classes
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

log = logging.getLogger("two_stage")
SPLITS = ("val", "test")
CLS = "infected"


def cmd_candidates(args) -> None:
    run_dir = resolve_path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    dcfg = data_config(cfg)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = None
    for split in args.splits:
        gt = load_split_coco(split, cfg["label_mode"], dcfg, resolve_path)
        ds = DetectionDataset(gt, resolve_path(dcfg["paths"]["raw_root"]) / "images",
                              build_transforms(cfg["augment"], train=False))
        if model is None:
            model = build_model(cfg["model"], ds.class_names, cfg["seed"])
            model.load_state_dict(ck["model"])
            model.to(device)
        cands = predict_candidates(model, eval_loader(ds, cfg), device, AMP_DTYPES[cfg["train"]["amp_dtype"]])
        kept = nms_per_image(cands, args.nms_iou)
        (run_dir / f"candidates_{split}.json").write_text(json.dumps(
            {"nms_iou": args.nms_iou, "checkpoint_epoch": ck["epoch"], "candidates": kept}), encoding="utf-8")
        obj = np.array([c["objectness"] for c in kept])
        log.info("%s %s: %d queries -> %d after NMS (%.1f / image); objectness >= 0.5: %d, >= 0.05: %d",
                 run_dir.name, split, len(cands), len(kept), len(kept) / len(gt["images"]), (obj >= 0.5).sum(),
                 (obj >= 0.05).sum())


def load_candidates(run: str, split: str) -> list[dict]:
    d = json.loads((resolve_path(f"runs/{run}") / f"candidates_{split}.json").read_text(encoding="utf-8"))
    return d["candidates"]


def cmd_features(args) -> None:
    cfg = load_experiment(args.config)
    pcfg = cfg["probe"]
    dcfg = load_config(resolve_path(cfg["data_config"]))
    device = torch.device("cuda")
    caches = ensure_caches(cfg, resolve_path, ("train", *SPLITS), allow_test=True)
    arrays = {s: np.load(p, mmap_mode="r") for s, p in caches.items()}
    index = {s: json.loads(p.with_name(f"{s}_index.json").read_text(encoding="utf-8")) for s, p in caches.items()}
    train = build_crop_set("train", dcfg, pcfg, index["train"], resolve_path)
    _, _, infected = probe_classes(dcfg, pcfg["exclude"])
    sc = pcfg["stain_shift"]
    amp = torch.bfloat16
    for spec in args.encoders:
        kind, value = spec.split(":", 1)
        enc, tag, source = load_encoder(cfg, **{{"ref": "ref", "run": "run", "detr": "detr_run"}[kind]: value})
        enc.to(device)
        norm = (source["input_mean"], source["input_std"]) if "input_mean" in source else None
        out = resolve_path("runs/two_stage") / tag
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "train.pt").exists():
            feats = {"clean": extract_features(enc, arrays["train"], train, pcfg, device, amp, input_norm=norm),
                     "stain": extract_features(enc, arrays["train"], train, pcfg, device, amp,
                                               shift=(build_stain_shift(sc), sc["seed"]), input_norm=norm)}
            torch.save({**feats, "infected": torch.as_tensor(infected[train["label"]]),
                        "image_id": torch.as_tensor(train["image_id"]), "source": source}, out / "train.pt")
            log.info("%s: train crops %d (clean + stain)", tag, len(train["label"]))
        for run in args.runs:
            for split in SPLITS:
                path = out / f"{run}_{split}.pt"
                if path.exists():
                    continue
                cands = load_candidates(run, split)
                crops = candidate_crops(cands, index[split], pcfg["long_side"])
                f = extract_features(enc, arrays[split], crops, pcfg, device, amp, input_norm=norm)
                feats = torch.zeros_like(f)
                feats[torch.as_tensor(crops["order"])] = f  # back to candidate order
                torch.save({"features": feats, "n": len(cands)}, path)
                log.info("%s: %s %s candidates %d", tag, run, split, len(cands))
        enc.cpu()


def evaluate_one(gt, cands, p_inf, variant, cat, thr=None):
    dets = scored_detections(cands, p_inf, variant, cat)
    if thr is None:
        thr = best_f1_threshold(match_per_class(gt, dets, 0.5)[cat])[0]
    return dets, thr


def classify(tag, run, training, pcfg, gt_val, cands_val, val_ids, cat):
    """Fit the crop classifier of `tag` (train GT crops, `clean` or clean + `stain`), p(infected) of the candidates
    of `run`; the score variant is chosen by val infected AP50 and the threshold by val max F1."""
    d = resolve_path("runs/two_stage") / tag
    tr = torch.load(d / "train.pt", weights_only=False)
    ev = {s: torch.load(d / f"{run}_{s}.pt", weights_only=False)["features"] for s in SPLITS}
    x = tr["clean"] if training == "clean" else torch.cat([tr["clean"], tr["stain"]])
    y = tr["infected"] if training == "clean" else torch.cat([tr["infected"], tr["infected"]])
    p_val, p_test = fit_p_infected(x, y, [ev["val"], ev["test"]], pcfg["linear"]["l2"], pcfg["linear"]["max_iter"])
    val_ap = {}
    for v in VARIANTS:
        dets, thr = evaluate_one(gt_val, cands_val, p_val, v, cat)
        val_ap[v] = (score(prepare_run(gt_val, dets, cat, thr), val_ids)["AP50"], thr)
    variant = max(VARIANTS, key=lambda v: val_ap[v][0])
    return {"val": p_val, "test": p_test}, variant, val_ap[variant][1], val_ap


def cmd_evaluate(args) -> None:
    pcfg = load_experiment(args.config)["probe"]
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gts = {s: load_split_coco(s, "binary", dcfg, resolve_path) for s in SPLITS}
    cat = {c["name"]: c["id"] for c in gts["val"]["categories"]}[CLS]
    img_ids = {s: [im["id"] for im in gts[s]["images"]] for s in SPLITS}
    samples = {s: resamples(img_ids[s], args.n, args.seed) for s in SPLITS}
    torch.manual_seed(0)
    rows, prepared = [], {}
    same_encoder = dict(x.split(":") for x in args.encoder_runs)   # tag -> its own end-to-end vit_detr run
    merges = parse_fold_merges(args.fold_merge)
    member_tags = [t for m in merges.values() for t in m]
    if {*merges, *member_tags} & {*args.tags, *same_encoder}:
        raise SystemExit("--fold-merge: NAME and its members must not be in --tags or --encoder-runs")
    for name, members in merges.items():
        for k, tag in enumerate(members, start=1):
            if ssl_run_target_fold(resolve_path(f"runs/{tag}")) != k:
                raise SystemExit(f"--fold-merge {name}: {tag} did not train on test fold {k}")
    image_fold = load_image_folds(resolve_path(args.folds)) if merges else {}
    test_tags = [*args.tags, *merges]   # what test rows / pairs / means can refer to
    if args.pairs is None:   # default (phase 4): joint - ImageNet when both are evaluated
        pairs = [("ssl_joint", "ref_imagenet")] if {"ssl_joint", "ref_imagenet"} <= set(args.tags) else []
    else:
        pairs = [tuple(x.split(":")) for x in args.pairs]
        if any(len(pr) != 2 or not set(pr) <= set(test_tags) for pr in pairs):
            raise SystemExit("--pairs must be A:B with A, B in --tags or --fold-merge")
    means = {}   # NAME -> member runs (training seeds); box means need every member in --runs
    for item in args.mean:
        name, _, rhs = item.partition("=")
        members = rhs.split("+")
        if not name or name in means or len(members) < 2:
            raise SystemExit(f"bad --mean {item!r}: NAME=R1+R2[+...]")
        means[name] = members
    for run in dict.fromkeys([*args.runs, *same_encoder.values(), *(r for m in means.values() for r in m)]):
        run_dir = resolve_path(f"runs/{run}")
        thr_e2e = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))[CLS]
        for s in SPLITS:   # end-to-end reference at its own val threshold
            dets = json.loads((run_dir / f"predictions_{s}.json").read_text(encoding="utf-8"))
            prepared[(run, "end_to_end", "-", s)] = prepare_run(gts[s], dets, cat, thr_e2e)
    for run in args.runs:
        cands = {s: load_candidates(run, s) for s in SPLITS}
        for tag in [*args.tags, *member_tags]:
            for training in ("clean", "stain"):
                p, variant, thr, val_ap = classify(tag, run, training, pcfg, gts["val"], cands["val"],
                                                   img_ids["val"], cat)
                for s in SPLITS:
                    dets = scored_detections(cands[s], p[s], variant, cat)
                    prepared[(run, tag, training, s)] = prepare_run(gts[s], dets, cat, thr)
                rows.append({"run": run, "encoder": tag, "training": training, "variant": variant,
                             **{f"val_AP50_{v}": val_ap[v][0] for v in VARIANTS}, "threshold": thr})
                log.info("%s %s %s: variant %s (val AP50 %s)", run, tag, training, variant,
                         {v: round(a, 4) for v, (a, _) in val_ap.items()})
        for name, (tag_f1, tag_f2) in merges.items():   # each test fold scored by the encoder that did not see it
            for training in ("clean", "stain"):
                prepared[(run, name, training, "test")] = fold_merge(
                    {1: prepared[(run, tag_f2, training, "test")], 2: prepared[(run, tag_f1, training, "test")]},
                    image_fold)
                for tag in (tag_f1, tag_f2):   # whole-test rows of a member include the fold its encoder saw
                    del prepared[(run, tag, training, "test")]

    boots = {s: bootstrap_table({k: r for k, r in prepared.items() if k[3] == s}, samples[s]) for s in SPLITS}
    table = []
    for key, r in prepared.items():
        run, tag, training, s = key
        pt = score(r, img_ids[s])
        row = {"run": run, "encoder": tag, "training": training, "split": s}
        for j, m in enumerate(METRICS):
            row[m] = pt[m]
            row[f"{m}_lo"], row[f"{m}_hi"] = ci(boots[s][key][:, j])
        table.append(row)
    table = pd.DataFrame(table).merge(pd.DataFrame(rows), on=["run", "encoder", "training"], how="left")

    diffs = []

    def add_diff(a, b, label):
        for j, m in enumerate(METRICS):
            pa, pb = score(prepared[a], img_ids["test"])[m], score(prepared[b], img_ids["test"])[m]
            diffs.append({"run": a[0], "comparison": label, "training": a[2], "metric": f"test_{m}_{CLS}",
                          **summarize_diff(pa, pb, boots["test"][a][:, j], boots["test"][b][:, j])})

    for run in args.runs:
        for training in ("clean", "stain"):
            for tag in test_tags:
                add_diff((run, tag, training, "test"), (run, "end_to_end", "-", "test"), f"{tag} two-stage - end-to-end")
                if tag in same_encoder:   # same encoder, different use: frozen crop classifier vs inside DETR
                    add_diff((run, tag, training, "test"), (same_encoder[tag], "end_to_end", "-", "test"),
                             f"{tag} two-stage - {same_encoder[tag]} end-to-end")
            for a, b in pairs:   # encoder - encoder, both two-stage on the same candidates
                add_diff((run, a, training, "test"), (run, b, training, "test"), f"{a} - {b} (two-stage)")

    # --mean: derived runs (mean of training seeds) on the same resamples; members keep their own thresholds /
    # score variants. A group whose members are all in --runs gets two-stage means; every group gets the
    # end-to-end mean. Two-stage means are compared with their own end-to-end mean and with every other group.
    mean_points, mean_rows = {}, []
    box_groups = [g for g, m in means.items() if set(m) <= set(args.runs)]
    for g, members in means.items():
        variants = [("end_to_end", "-")] + ([(t, tr) for t in test_tags for tr in ("clean", "stain")]
                                            if g in box_groups else [])
        for tag, training in variants:
            for s in SPLITS:
                keys = [(r, tag, training, s) for r in members]
                if any(k not in prepared for k in keys):   # fold-merged tags exist on test only
                    continue
                pts = [score(prepared[k], img_ids[s]) for k in keys]
                point = {m: float(np.mean([p[m] for p in pts])) for m in METRICS}
                boot = np.mean([boots[s][k] for k in keys], axis=0)
                mean_points[(g, tag, training, s)] = (point, boot)
                row = {"run": g, "encoder": tag, "training": training, "split": s, "members": "+".join(members)}
                for j, m in enumerate(METRICS):
                    row[m] = point[m]
                    row[f"{m}_lo"], row[f"{m}_hi"] = ci(boot[:, j])
                mean_rows.append(row)

    def add_mean_diff(a, b, label):
        (pa, ba), (pb, bb) = mean_points[a], mean_points[b]
        for j, m in enumerate(METRICS):
            diffs.append({"run": a[0], "comparison": label, "training": a[2], "metric": f"test_{m}_{CLS}",
                          **summarize_diff(pa[m], pb[m], ba[:, j], bb[:, j])})

    for g in box_groups:
        for training in ("clean", "stain"):
            for tag in test_tags:
                for h in means:
                    add_mean_diff((g, tag, training, "test"), (h, "end_to_end", "-", "test"),
                                  f"{tag} two-stage ({g}) - end-to-end ({h})")
            for a, b in pairs:   # encoder - encoder on the seed mean of the same boxes, shared resamples
                add_mean_diff((g, a, training, "test"), (g, b, training, "test"), f"{a} - {b} (two-stage, {g})")
    if mean_rows:
        table = pd.concat([table, pd.DataFrame(mean_rows)], ignore_index=True)
    diffs = pd.DataFrame(diffs)

    out = resolve_path(args.out)
    table.to_csv(f"{out}.csv", index=False, float_format="%.4f")
    diffs.to_csv(f"{out}_diff.csv", index=False, float_format="%.4f")
    (out.parent / f"{out.name}.json").write_text(json.dumps(
        {"class": CLS, "n_resamples": args.n, "seed": args.seed, "n_images": {s: len(v) for s, v in img_ids.items()},
         "selection": rows, "means": means, "table": table.to_dict("records"), "diffs": diffs.to_dict("records")},
        indent=2),
        encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 500):
        print(table[["run", "encoder", "training", "variant", "split", "AP50", "AP50_lo", "AP50_hi", "precision",
                     "recall", "f1"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        d = diffs[diffs["metric"].isin([f"test_AP50_{CLS}", f"test_recall_{CLS}"])]
        print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}.csv, {out}_diff.csv, {out}.json")


def cmd_folds(args) -> None:
    """Per-test-fold two-stage metrics of each encoder (diagnostic of the cross-fold scheme): same classifier,
    score variant and val threshold as `evaluate`; CIs from resamples of the fold's images."""
    pcfg = load_experiment(args.config)["probe"]
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gts = {s: load_split_coco(s, "binary", dcfg, resolve_path) for s in SPLITS}
    cat = {c["name"]: c["id"] for c in gts["val"]["categories"]}[CLS]
    val_ids = [im["id"] for im in gts["val"]["images"]]
    image_fold = load_image_folds(resolve_path(args.folds))
    fold_ids = {k: sorted(i for i, f in image_fold.items() if f == k) for k in sorted(set(image_fold.values()))}
    samples = {k: resamples(ids, args.n, args.seed) for k, ids in fold_ids.items()}
    torch.manual_seed(0)
    rows = []
    for run in args.runs:
        cands = {s: load_candidates(run, s) for s in SPLITS}
        for tag in args.tags:
            run_dir = resolve_path(f"runs/{tag}")   # references (ref_*) have no run dir and saw no test fold
            seen = ssl_run_target_fold(run_dir) if (run_dir / "config.yaml").exists() else None
            for training in args.training:
                p, variant, thr, _ = classify(tag, run, training, pcfg, gts["val"], cands["val"], val_ids, cat)
                r = prepare_run(gts["test"], scored_detections(cands["test"], p["test"], variant, cat), cat, thr)
                for k, ids in fold_ids.items():
                    pt = score(r, ids)
                    boot = bootstrap_table({"r": r}, samples[k])["r"]
                    row = {"run": run, "encoder": tag, "training": training, "variant": variant, "threshold": thr,
                           "fold": k, "encoder_saw_fold": seen, "seen": seen == k}
                    for j, m in enumerate(METRICS):
                        row[m] = pt[m]
                        row[f"{m}_lo"], row[f"{m}_hi"] = ci(boot[:, j])
                    rows.append(row)
                    log.info("%s %s %s fold %d (encoder saw %s): AP50 %.4f P %.4f R %.4f", run, tag, training, k,
                             seen, pt["AP50"], pt["precision"], pt["recall"])
    table = pd.DataFrame(rows)
    out = resolve_path(args.out)
    table.to_csv(f"{out}.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 500):
        print(table[["run", "encoder", "training", "fold", "seen", "AP50", "AP50_lo", "AP50_hi", "precision",
                     "recall", "f1"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--run", required=True)
    c.add_argument("--nms-iou", type=float, default=0.5)
    c.add_argument("--splits", nargs="+", default=list(SPLITS), choices=["train", *SPLITS],
                   help="train: class-agnostic cell boxes for the phase-7 SSL cell crops (eval transform, no labels used)")
    f = sub.add_parser("features")
    f.add_argument("--runs", nargs="+", required=True)
    f.add_argument("--encoders", nargs="+", required=True, help="ref:imagenet | ref:dino | run:<ssl run> | detr:<vit_detr run>")
    f.add_argument("--config", default="configs/ssl/base.yaml")
    e = sub.add_parser("evaluate")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--tags", nargs="+", required=True)
    e.add_argument("--encoder-runs", nargs="*", default=[], metavar="TAG:RUN",
                   help="also compare each two-stage TAG with the end-to-end vit_detr RUN built on the same encoder")
    e.add_argument("--pairs", nargs="*", default=None, metavar="A:B",
                   help="two-stage encoder differences on test (default: ssl_joint:ref_imagenet if both in --tags)")
    e.add_argument("--mean", nargs="*", default=[], metavar="NAME=R1+R2",
                   help="derived run: mean of training seeds on the same resamples (two-stage if all Ri in --runs, "
                        "always end-to-end); two-stage means are compared with every group's end-to-end mean")
    e.add_argument("--fold-merge", nargs="*", default=[], metavar="NAME=TAG_F1+TAG_F2",
                   help="cross-fold encoder (test only): fold-1 images from TAG_F2, fold-2 from TAG_F1 (TAG_Fk saw fold k)")
    e.add_argument("--folds", default="data/splits/test_folds.json")
    e.add_argument("--config", default="configs/ssl/base.yaml")
    e.add_argument("--n", type=int, default=1000)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out", required=True)
    k = sub.add_parser("folds")
    k.add_argument("--runs", nargs="+", required=True)
    k.add_argument("--tags", nargs="+", required=True)
    k.add_argument("--training", nargs="+", default=["clean"], choices=["clean", "stain"])
    k.add_argument("--folds", default="data/splits/test_folds.json")
    k.add_argument("--config", default="configs/ssl/base.yaml")
    k.add_argument("--n", type=int, default=1000)
    k.add_argument("--seed", type=int, default=42)
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    {"candidates": cmd_candidates, "features": cmd_features, "evaluate": cmd_evaluate,
     "folds": cmd_folds}[args.cmd](args)


if __name__ == "__main__":
    main()
