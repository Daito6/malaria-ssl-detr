"""Val vs test confidence of true infected cells and the test PR curve for a binary DETR run.

Needs runs/<name>/predictions_{val,test}.json and thresholds.json (from scripts/evaluate.py).
Diagnostic only: nothing here is tuned on test.

Usage:
    python scripts/diagnose_shift.py runs/a_coco_detr_binary
Writes reports/figures/shift_<name>.png and reports/tables/shift_<name>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")

from malaria_ssl.data.detection import load_split_coco  # noqa: E402
from malaria_ssl.eval.shift import gt_box_scores, plot_shift, pr_curve, summarize  # noqa: E402
from malaria_ssl.utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--cls", default="infected")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if cfg["label_mode"] != "binary":
        parser.error("expects a binary run")
    dcfg = load_config(resolve_path(cfg["data_config"]))
    thr = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))[args.cls]
    iou = cfg["eval"]["iou_threshold"]
    scores, curves, summary = {}, {}, {"run": run_dir.name, "class": args.cls, "val_threshold": thr, "iou": iou}
    for split in ("val", "test"):
        gt = load_split_coco(split, "binary", dcfg, resolve_path)
        cat = {c["name"]: c["id"] for c in gt["categories"]}[args.cls]
        dets = json.loads((run_dir / f"predictions_{split}.json").read_text(encoding="utf-8"))
        scores[split] = gt_box_scores(gt, dets, cat, iou)
        curves[split] = pr_curve(gt, dets, cat, iou)
        summary[split] = summarize(scores[split], thr)
    fig = plot_shift(scores, curves, thr, args.cls, f"{run_dir.name}: {args.cls} cells, val vs test")
    fig_path = resolve_path("reports/figures") / f"shift_{run_dir.name}.png"
    fig.savefig(fig_path, dpi=120)
    tab_path = resolve_path("reports/tables") / f"shift_{run_dir.name}.json"
    tab_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {fig_path} and {tab_path}")


if __name__ == "__main__":
    main()
