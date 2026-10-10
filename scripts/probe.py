"""Cell-level kNN / linear probe of a ViT encoder on train -> val crops (no DETR, no test split).

Usage:
    python scripts/probe.py --run runs/ssl_joint                 # runs/<name>/checkpoints/encoder_final.pt
    python scripts/probe.py --encoder runs/ssl_joint/checkpoints/encoder_ep020.pt --tag ssl_joint_ep020
    python scripts/probe.py --ref random                         # random-init ViT-S/16 reference
    python scripts/probe.py --ref imagenet                       # ImageNet ViT-S/16 (timm) reference

Writes reports/tables/probe_<tag>.{json,csv} and runs/probe/<tag>/features_{train,val}.pt.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
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
    if args.ref == "imagenet":
        return build_encoder(refs["imagenet"], pretrained=True), "ref_imagenet", {"ref": "imagenet", "timm": refs["imagenet"]}
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_experiment(args.config)
    device = torch.device(args.device)
    encoder, tag, source = load_encoder(args, cfg)
    encoder.to(device)
    caches = ensure_caches(cfg, resolve_path)
    prober = Prober(cfg, load_config(resolve_path(cfg["data_config"])), caches, resolve_path, device)
    log.info("crops per class: %s", prober.counts())

    t0 = time.time()
    feats = prober.features(encoder)
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
        c = prober.crops[split]
        torch.save({"features": f, "label": torch.as_tensor(c["label"]), "image_id": torch.as_tensor(c["image_id"]),
                    "classes": prober.names}, feat_dir / f"features_{split}.pt")
    for m, e in res["methods"].items():
        log.info("%s: %s", m, {k: round(v, 4) for k, v in e["metrics"].items()})
    log.info("wrote %s", out_dir / f"probe_{tag}.json")


if __name__ == "__main__":
    main()
