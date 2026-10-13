"""Cell-level kNN / linear probe of a ViT encoder on train -> val crops (no DETR; test only via --eval-split).

Usage:
    python scripts/probe.py --run runs/ssl_joint                 # runs/<name>/checkpoints/encoder_final.pt
    python scripts/probe.py --encoder runs/ssl_joint/checkpoints/encoder_ep020.pt --tag ssl_joint_ep020
    python scripts/probe.py --ref random                         # random-init ViT-S/16 reference
    python scripts/probe.py --ref imagenet                       # ImageNet ViT-S/16 (timm) reference
    python scripts/probe.py --run runs/ssl_joint --eval-split test   # FINAL evaluation on test crops (no tuning)

Also runs the stain-shifted probe (same classifiers, fitted on clean train crops, evaluated on val
crops with the fixed seeded shift of `probe.stain_shift`): `stain_shift` in the JSON, rows with
eval_set=val_stain in the CSV.

Input normalization: crops are ImageNet-normalized (as in SSL pretraining and DETR); the ImageNet
reference is re-mapped to its own timm statistics (augreg ViT: mean = std = 0.5), recorded in `source`.

Writes reports/tables/probe_<tag>.{json,csv} and runs/probe/<tag>/features_{train,val,val_stain}.pt.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import timm
import torch
import yaml

from malaria_ssl.ssl.data import ensure_caches
from malaria_ssl.ssl.models import build_encoder
from malaria_ssl.ssl.probe import Prober, results_table
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

log = logging.getLogger("probe")


def load_encoder(args, cfg) -> tuple[torch.nn.Module, str, dict]:
    refs = cfg["probe"]["references"]
    if args.ref == "random":
        torch.manual_seed(refs["random_init_seed"])
        return build_encoder(cfg["model"]["encoder"]), "ref_random_init", {"ref": "random", "seed": refs["random_init_seed"]}
    if args.ref == "imagenet":   # fed with its own input normalization (augreg: mean = std = 0.5)
        pc = timm.models.get_pretrained_cfg(refs["imagenet"])
        return (build_encoder(refs["imagenet"], pretrained=True), "ref_imagenet",
                {"ref": "imagenet", "timm": refs["imagenet"], "input_mean": list(pc.mean), "input_std": list(pc.std)})
    if args.run:
        run_dir = Path(args.run)
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        path, name, tag = run_dir / "checkpoints" / "encoder_final.pt", run_cfg["model"]["encoder"], run_dir.name
    else:
        path, name, tag = Path(args.encoder), cfg["model"]["encoder"], Path(args.encoder).stem
    enc = build_encoder(name)
    enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return enc, args.tag or tag, {"encoder_path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run")
    src.add_argument("--encoder")
    src.add_argument("--ref", choices=["random", "imagenet"])
    parser.add_argument("--config", default="configs/ssl/base.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"],
                        help="test = final evaluation on test crops (fit on train; no stain shift; writes probe_test_<tag>)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_experiment(args.config)
    device = torch.device(args.device)
    encoder, tag, source = load_encoder(args, cfg)
    encoder.to(device)
    on_test = args.eval_split == "test"
    caches = ensure_caches(cfg, resolve_path, ("train", args.eval_split), allow_test=on_test)
    prober = Prober(cfg, load_config(resolve_path(cfg["data_config"])), caches, resolve_path, device, args.eval_split)
    if on_test:
        tag = f"test_{tag}"
    log.info("crops per class: %s", prober.counts())

    t0 = time.time()
    norm = (source["input_mean"], source["input_std"]) if "input_mean" in source else None  # None = ImageNet stats
    feats = prober.features(encoder, stain_shift=not on_test, input_norm=norm)
    t_feat = time.time() - t0
    res = prober.evaluate(encoder, full=True, feats=feats)
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
