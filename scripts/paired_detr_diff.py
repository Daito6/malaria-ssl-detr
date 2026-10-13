"""Per-run val/test table and paired test bootstrap of differences for binary DETR runs (CPU, reads only).

Needs, per run: metrics_{val,test}.json, predictions_test.json, thresholds.json (scripts/evaluate.py) and
reports/tables/shift_<run>.json (scripts/diagnose_shift.py). Nothing is tuned: every run is scored at its
own val-selected threshold.

Usage:
    python scripts/paired_detr_diff.py --runs a_coco_detr_binary b_vit_random b_vit_imagenet b_vit_mae \
        b_vit_con b_vit_joint --pairs b_vit_joint:b_vit_imagenet b_vit_joint:b_vit_con ... \
        --out reports/tables/phase4_main
Writes <out>_runs.csv (AP50 CIs from evaluate.py; test P/R/F1 CIs from the shared resamples here),
<out>_diff.csv and <out>.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from malaria_ssl.data.detection import load_split_coco
from malaria_ssl.eval.paired import METRICS, bootstrap_table, ci, prepare_run, resamples, score, summarize_diff
from malaria_ssl.utils.config import load_config, resolve_path

CLS = "infected"


def run_row(name: str, boot: np.ndarray, point: dict[str, float]) -> dict:
    run_dir = resolve_path(f"runs/{name}")
    row = {"run": name}
    for split in ("val", "test"):
        m = json.loads((run_dir / f"metrics_{split}.json").read_text(encoding="utf-8"))
        b = m["bootstrap"]["ci95"]
        if split == "val":
            row["best_epoch"] = m["checkpoint_epoch"] + 1
            row["val_recall"] = m["prf"]["per_class"][CLS]["recall"]  # point only, at the val-selected threshold
        for c in ("infected", "uninfected"):
            row[f"{split}_AP50_{c}"] = m["coco"]["per_class"][c]["AP50"]
            row[f"{split}_AP50_{c}_lo"], row[f"{split}_AP50_{c}_hi"] = b[f"AP50/{c}"]["low"], b[f"AP50/{c}"]["high"]
        if split == "test":
            p = m["prf"]["per_class"][CLS]
            row["threshold"] = p["threshold"]
            for k, j in (("precision", 1), ("recall", 2), ("f1", 3)):
                assert abs(p[k] - point[k]) < 1e-9, (name, k, p[k], point[k])  # same matching as evaluate.py
                row[f"test_{k}"] = p[k]
                row[f"test_{k}_lo"], row[f"test_{k}_hi"] = ci(boot[:, j])
    stain = run_dir / "metrics_val_stain.json"  # evaluate.py --stain-shift (synthetic stain-shifted val), optional
    if stain.exists():
        m = json.loads(stain.read_text(encoding="utf-8"))
        b = m["bootstrap"]["ci95"][f"AP50/{CLS}"]
        row["val_stain_AP50_infected"] = m["coco"]["per_class"][CLS]["AP50"]
        row["val_stain_AP50_infected_lo"], row["val_stain_AP50_infected_hi"] = b["low"], b["high"]
        row["val_stain_recall"] = m["prf"]["per_class"][CLS]["recall"]
        row["selected_on"] = m["checkpoint_selected_on"]
    sh = json.loads((resolve_path("reports/tables") / f"shift_{name}.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        row[f"{split}_median_score"] = sh[split]["median_score"]
    for k in ("frac_ge_val_threshold", "frac_no_target_detection", "frac_other_class_scores_higher"):
        row[f"test_{k}"] = sh["test"][k]
    for split in ("val", "test"):  # mutually exclusive outcomes per infected GT box
        for k in ("frac_no_detection_any_class", "frac_other_class_scores_higher", "frac_target_scores_highest"):
            row[f"{split}_excl_{k}"] = sh[split][k]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], metavar="A:B")
    ap.add_argument("--n", type=int, default=1000, help="bootstrap resamples of test images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    gt = load_split_coco("test", "binary", load_config(resolve_path(args.data_config)), resolve_path)
    cat = {c["name"]: c["id"] for c in gt["categories"]}[CLS]
    runs = {}
    for name in args.runs:
        run_dir = resolve_path(f"runs/{name}")
        dets = json.loads((run_dir / "predictions_test.json").read_text(encoding="utf-8"))
        thr = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))[CLS]
        runs[name] = prepare_run(gt, dets, cat, thr)
    img_ids = [im["id"] for im in gt["images"]]
    samples = resamples(img_ids, args.n, args.seed)
    boots = bootstrap_table(runs, samples)
    points = {name: score(r, img_ids) for name, r in runs.items()}

    table = pd.DataFrame([run_row(name, boots[name], points[name]) for name in args.runs])
    diffs = []
    for pair in args.pairs:
        a, b = pair.split(":")
        for j, m in enumerate(METRICS):
            diffs.append({"a": a, "b": b, "metric": f"test_{m}_{CLS}",
                          **summarize_diff(points[a][m], points[b][m], boots[a][:, j], boots[b][:, j])})
    diff = pd.DataFrame(diffs, columns=None if diffs else ["a", "b", "metric"])

    out = resolve_path(args.out)
    table.to_csv(f"{out}_runs.csv", index=False, float_format="%.4f")
    diff.to_csv(f"{out}_diff.csv", index=False, float_format="%.4f")
    meta = {"split": "test", "class": CLS, "n_images": len(img_ids), "n_gt": int(len(
        [a for a in gt["annotations"] if a["category_id"] == cat])), "n_resamples": args.n, "seed": args.seed,
            "runs": table.to_dict("records"), "diffs": diffs}
    (out.parent / f"{out.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.T.to_string(float_format=lambda v: f"{v:.4f}"))
        print(diff.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}_runs.csv, {out}_diff.csv, {out}.json")


if __name__ == "__main__":
    main()
