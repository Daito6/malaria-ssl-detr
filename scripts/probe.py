"""Cell-level kNN / linear probe of a ViT encoder on train -> val crops (no DETR; test only via --eval-split).

Usage:
    python scripts/probe.py --run runs/ssl_joint                 # runs/<name>/checkpoints/encoder_final.pt
    python scripts/probe.py --encoder runs/ssl_joint/checkpoints/encoder_ep020.pt --tag ssl_joint_ep020
    python scripts/probe.py --ref random                         # random-init ViT-S/16 reference
    python scripts/probe.py --ref imagenet                       # ImageNet ViT-S/16 (timm) reference
    python scripts/probe.py --run runs/ssl_joint --eval-split test   # FINAL evaluation on test crops (no tuning)
    python scripts/probe.py --detr-run runs/b_vit_joint           # the ViT after DETR fine-tuning (tag detr_<run>)
    python scripts/probe.py --run runs/ssl_imagenet_da_f1 --eval-split test --eval-fold 2   # held-out test fold only

Also runs the stain-shifted probe (same classifiers, fitted on clean train crops, evaluated on val
crops with the fixed seeded shift of `probe.stain_shift`): `stain_shift` in the JSON, rows with
eval_set=val_stain in the CSV.

Input normalization: crops are ImageNet-normalized (as in SSL pretraining and DETR); the ImageNet
reference is re-mapped to its own timm statistics (augreg ViT: mean = std = 0.5), recorded in `source`.

--eval-fold k (with --eval-split test) keeps only the test crops of fold k of the folds file
(scripts/make_test_folds.py); tag test_fold<k>_<tag>. Used to evaluate encoders adapted on the other fold.

Writes reports/tables/probe_<tag>.{json,csv} and runs/probe/<tag>/features_{train,val,val_stain}.pt.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from malaria_ssl.ssl.data import ensure_caches, load_fold_images
from malaria_ssl.ssl.encoders import load_encoder
from malaria_ssl.ssl.probe import Prober, results_table
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

log = logging.getLogger("probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run")
    src.add_argument("--encoder")
    src.add_argument("--ref", choices=["random", "imagenet"])
    src.add_argument("--detr-run", help="vit_detr run dir: probe its ViT after DETR fine-tuning (best.pt)")
    parser.add_argument("--config", default="configs/ssl/base.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"],
                        help="test = final evaluation on test crops (fit on train; no stain shift; writes probe_test_<tag>)")
    parser.add_argument("--eval-fold", type=int, choices=[1, 2], default=None,
                        help="with --eval-split test: evaluate on the test crops of this fold only")
    parser.add_argument("--folds-file", default="data/splits/test_folds.json")
    args = parser.parse_args()
    if args.eval_fold and args.eval_split != "test":
        parser.error("--eval-fold needs --eval-split test")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_experiment(args.config)
    device = torch.device(args.device)
    encoder, tag, source = load_encoder(cfg, args.ref, args.run, args.encoder, args.detr_run, args.tag)
    encoder.to(device)
    on_test = args.eval_split == "test"
    caches = ensure_caches(cfg, resolve_path, ("train", args.eval_split), allow_test=on_test)
    fold_ids = ([im["id"] for im in load_fold_images(resolve_path(args.folds_file), args.eval_fold)]
                if args.eval_fold else None)
    prober = Prober(cfg, load_config(resolve_path(cfg["data_config"])), caches, resolve_path, device, args.eval_split,
                    eval_image_ids=fold_ids)
    if on_test:
        tag = f"test_fold{args.eval_fold}_{tag}" if args.eval_fold else f"test_{tag}"
    log.info("crops per class: %s", prober.counts())

    t0 = time.time()
    norm = (source["input_mean"], source["input_std"]) if "input_mean" in source else None  # None = ImageNet stats
    feats = prober.features(encoder, stain_shift=not on_test, input_norm=norm)
    t_feat = time.time() - t0
    res = prober.evaluate(encoder, full=True, feats=feats)
    if args.eval_fold:
        res["eval_fold"] = {"fold": args.eval_fold, "folds_file": Path(args.folds_file).as_posix(),
                            "n_images": len(fold_ids)}
    res.update(tag=tag, source=source, probe_config=cfg["probe"],
               time_s={"features": round(t_feat, 1), "total": round(time.time() - t0, 1)})

    out_dir = resolve_path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"probe_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame(results_table(res)).to_csv(out_dir / f"probe_{tag}.csv", index=False)
    feat_dir = resolve_path("runs/probe") / tag
    feat_dir.mkdir(parents=True, exist_ok=True)
    for split, f in feats.items():
        c = prober.crops[split.removesuffix("_stain")]
        torch.save({"features": f, "label": torch.as_tensor(c["label"]), "image_id": torch.as_tensor(c["image_id"]),
                    "classes": prober.names}, feat_dir / f"features_{split}.pt")
    for m, e in res["methods"].items():
        log.info("%s: %s", m, {k: round(v, 4) for k, v in e["metrics"].items()})
    for m, e in res.get("stain_shift", {}).get("methods", {}).items():
        log.info("%s (stain-shifted val): %s", m, {k: round(v, 4) for k, v in e["metrics"].items()})
    log.info("wrote %s", out_dir / f"probe_{tag}.json")


if __name__ == "__main__":
    main()
