"""Evaluate a trained run on one split.

Writes runs/<name>/predictions_<split>.json and runs/<name>/metrics_<split>.json.
Per-class score thresholds are chosen on val (max F1) and saved to thresholds.json; evaluating
test requires that val was evaluated first and reuses those thresholds unchanged.

Usage:
    python scripts/evaluate.py runs/a_coco_detr_binary --split val --bootstrap
    python scripts/evaluate.py runs/a_coco_detr_binary --split test --bootstrap 200
    python scripts/evaluate.py runs/a_coco_detr_binary --split val --long-side 1016 --bootstrap   # scale probe

--long-side N overrides the eval input long side (default augment.eval_long_side = 800), i.e. rescales the
image content by N/800 at the model input. It is a diagnostic: the val thresholds in thresholds.json are
reused unchanged (also for val), nothing is written to thresholds.json, and outputs get a `_ls<N>` suffix.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import yaml

from malaria_ssl.data.detection import DetectionDataset, build_transforms, load_split_coco
from malaria_ssl.detr.build import build_model
from malaria_ssl.detr.trainer import AMP_DTYPES, data_config, eval_loader
from malaria_ssl.eval.inference import predict
from malaria_ssl.eval.report import evaluate_split
from malaria_ssl.utils.config import resolve_path

log = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--bootstrap", type=int, nargs="?", const=200, default=0,
                        help="bootstrap resamples for 95%% CIs (flag alone = 200)")
    parser.add_argument("--long-side", type=int, default=None,
                        help="diagnostic eval input long side; reuses the val thresholds; outputs *_ls<N>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg = data_config(cfg)
    thr_path = run_dir / "thresholds.json"
    fixed_thr = args.split != "val" or args.long_side is not None
    if fixed_thr and not thr_path.exists():
        parser.error("thresholds.json not found: evaluate --split val first (thresholds are chosen on val only)")
    suffix = f"{args.split}_ls{args.long_side}" if args.long_side else args.split
    if args.long_side:
        cfg["augment"]["eval_long_side"] = args.long_side

    gt = load_split_coco(args.split, cfg["label_mode"], dcfg, resolve_path)
    cells = Counter(a["image_id"] for a in load_split_coco(args.split, "multiclass", dcfg, resolve_path)["annotations"])
    cells = {im["id"]: cells[im["id"]] for im in gt["images"]}
    ds = DetectionDataset(gt, resolve_path(dcfg["paths"]["raw_root"]) / "images",
                          build_transforms(cfg["augment"], train=False))

    ck = torch.load(run_dir / "checkpoints" / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    if ck["class_names"] != ds.class_names:
        raise ValueError(f"checkpoint classes {ck['class_names']} != dataset classes {ds.class_names}")
    device = torch.device("cuda")
    model = build_model(cfg["model"], ds.class_names, cfg["seed"])
    model.load_state_dict(ck["model"])
    model.to(device)
    log.info("loaded %s checkpoint (epoch %d)", args.checkpoint, ck["epoch"])

    dets = predict(model, eval_loader(ds, cfg), device, ds.label_to_cat, AMP_DTYPES[cfg["train"]["amp_dtype"]])
    (run_dir / f"predictions_{suffix}.json").write_text(json.dumps(dets), encoding="utf-8")

    thresholds = None
    if fixed_thr:
        by_name = json.loads(thr_path.read_text(encoding="utf-8"))
        name_to_cat = {c["name"]: c["id"] for c in gt["categories"]}
        thresholds = {name_to_cat[n]: t for n, t in by_name.items()}
    metrics, thresholds = evaluate_split(gt, dets, cells, thresholds, cfg["eval"]["iou_threshold"],
                                         args.bootstrap, cfg["seed"])
    if not fixed_thr:
        thr_path.write_text(json.dumps(metrics["thresholds"], indent=2), encoding="utf-8")
    metrics.update({"split": args.split, "eval_long_side": cfg["augment"]["eval_long_side"],
                    "checkpoint": args.checkpoint, "checkpoint_epoch": ck["epoch"],
                    "label_mode": cfg["label_mode"], "n_images": len(gt["images"]),
                    "n_gt_boxes": len(gt["annotations"])})
    (run_dir / f"metrics_{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    c = metrics["coco"]
    log.info("%s: AP %.4f | AP50 %.4f | AP75 %.4f | AR@300 %.4f | macro F1 %.4f", suffix, c["AP"], c["AP50"],
             c["AP75"], c["AR@300"], metrics["prf"]["macro_f1"])
    for name, v in metrics["prf"]["per_class"].items():
        log.info("  %-15s AP50 %.4f  P %.3f R %.3f F1 %.3f (thr %.3f, n_gt %d)", name,
                 c["per_class"][name]["AP50"], v["precision"], v["recall"], v["f1"], v["threshold"], v["n_gt"])


if __name__ == "__main__":
    main()
