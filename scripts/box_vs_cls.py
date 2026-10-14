"""Boxes or classification: class-agnostic AP50 / recall of a DETR run's candidates vs its infected AP50 (CPU).

    python scripts/box_vs_cls.py --runs a_coco_detr_binary b_vit_imagenet b_vit_dino_full \
        --out reports/tables/phase6_box_vs_cls

Needs, per run: candidates_{val,test}.json (scripts/two_stage.py candidates), predictions_{val,test}.json and
thresholds.json (scripts/evaluate.py). Per split, every run is scored on the SAME bootstrap resamples of images:
- agnostic_AP50: all GT cells as one class, candidates scored by objectness (eval/box_vs_cls.py);
- agnostic_recall_all / agnostic_recall_infected: share of all / infected GT cells matched (IoU 0.5) by a
  class-agnostic candidate, no score threshold;
- infected_AP50: the end-to-end infected AP50 (predictions_<split>.json; COCO 101-point).
Writes <out>.csv (point, 95% CI per run x split x metric) and <out>.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from malaria_ssl.data.detection import load_split_coco
from malaria_ssl.eval.box_vs_cls import METRICS, prepare_agnostic, score_agnostic
from malaria_ssl.eval.paired import ci, prepare_run, resamples, score
from malaria_ssl.utils.config import load_config, resolve_path

CLS = "infected"
SPLITS = ("val", "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run names under runs/")
    ap.add_argument("--n", type=int, default=1000, help="bootstrap resamples of images (shared by all runs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dcfg = load_config(resolve_path(args.data_config))
    rows = []
    for split in SPLITS:
        gt = load_split_coco(split, "binary", dcfg, resolve_path)
        cat = {c["name"]: c["id"] for c in gt["categories"]}[CLS]
        img_ids = [im["id"] for im in gt["images"]]
        samples = resamples(img_ids, args.n, args.seed)
        for run in args.runs:
            run_dir = resolve_path(f"runs/{run}")
            cands = json.loads((run_dir / f"candidates_{split}.json").read_text(encoding="utf-8"))
            agn = prepare_agnostic(gt, cands["candidates"], cat)
            dets = json.loads((run_dir / f"predictions_{split}.json").read_text(encoding="utf-8"))
            thr = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))[CLS]
            e2e = prepare_run(gt, dets, cat, thr)

            def all_metrics(ids):
                return {**score_agnostic(agn, ids), "infected_AP50": score(e2e, ids)["AP50"]}

            point = all_metrics(img_ids)
            boot = [all_metrics(ids) for ids in samples]
            for m in (*METRICS, "infected_AP50"):
                lo, hi = ci(np.array([b[m] for b in boot]))
                rows.append({"run": run, "split": split, "metric": m, "value": point[m], "ci_low": lo, "ci_high": hi,
                             "checkpoint_epoch": cands["checkpoint_epoch"], "nms_iou": cands["nms_iou"]})
            print(f"{run} {split}: " + ", ".join(f"{m} {point[m]:.4f}" for m in point))

    table = pd.DataFrame(rows)
    out = resolve_path(args.out)
    table.to_csv(f"{out}.csv", index=False, float_format="%.4f")
    (out.parent / f"{out.name}.json").write_text(json.dumps(
        {"class": CLS, "iou": 0.5, "n_resamples": args.n, "seed": args.seed, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}.csv, {out}.json")


if __name__ == "__main__":
    main()
