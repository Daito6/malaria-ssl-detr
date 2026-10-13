"""Scale diagnostic on VAL: the same model at different eval input long sides, at its val thresholds (CPU).

Needs runs/<name>/predictions_val.json (default scale) and predictions_val_ls<N>.json from
    python scripts/evaluate.py runs/<name> --split val --long-side <N> --bootstrap
Rescaling the eval long side from 800 to N changes the cells' size at the model input by N/800 (the eval
transform resizes every image to a fixed long side, so resizing the image file itself would change nothing).
The thresholds are the ones chosen at the default scale; nothing is re-tuned. Test is not used.

Usage:
    python scripts/scale_diagnostic.py --runs a_coco_detr_binary b_vit_imagenet b_vit_joint \
        --long-sides 640 840 1016 --out reports/tables/phase4_scale
Writes <out>.csv: infected AP50 / recall / precision with 95% CIs and the paired difference vs the default
scale (same val-image resamples), plus <out>.json.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import yaml

from malaria_ssl.data.detection import load_split_coco
from malaria_ssl.eval.paired import METRICS, bootstrap_table, ci, prepare_run, resamples, score, summarize_diff
from malaria_ssl.utils.config import load_config, resolve_path

CLS = "infected"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--long-sides", nargs="+", type=int, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    gt = load_split_coco("val", "binary", load_config(resolve_path(args.data_config)), resolve_path)
    cat = {c["name"]: c["id"] for c in gt["categories"]}[CLS]
    img_ids = [im["id"] for im in gt["images"]]
    samples = resamples(img_ids, args.n, args.seed)
    rows = []
    for name in args.runs:
        run_dir = resolve_path(f"runs/{name}")
        base = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["augment"]["eval_long_side"]
        thr = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))[CLS]
        runs = {}
        for ls in [base, *args.long_sides]:
            pred = "predictions_val.json" if ls == base else f"predictions_val_ls{ls}.json"
            dets = json.loads((run_dir / pred).read_text(encoding="utf-8"))
            runs[ls] = prepare_run(gt, dets, cat, thr)
        boots = bootstrap_table(runs, samples)
        points = {ls: score(r, img_ids) for ls, r in runs.items()}
        for ls in runs:
            row = {"run": name, "long_side": ls, "factor": ls / base, "threshold": thr}
            for j, m in enumerate(METRICS):
                row[f"{m}_{CLS}"] = points[ls][m]
                row[f"{m}_{CLS}_lo"], row[f"{m}_{CLS}_hi"] = ci(boots[ls][:, j])
                if ls != base:
                    d = summarize_diff(points[ls][m], points[base][m], boots[ls][:, j], boots[base][:, j])
                    row.update({f"d{m}_vs_base": d["diff"], f"d{m}_lo": d["low"], f"d{m}_hi": d["high"],
                                f"d{m}_share_le_0": d["share_le_0"]})
            rows.append(row)
    df = pd.DataFrame(rows)
    out = resolve_path(args.out)
    df.to_csv(f"{out}.csv", index=False, float_format="%.4f")
    (out.parent / f"{out.name}.json").write_text(json.dumps(
        {"split": "val", "class": CLS, "n_images": len(img_ids), "n_resamples": args.n, "seed": args.seed,
         "rows": rows}, indent=2), encoding="utf-8")
    cols = ["run", "factor", "AP50_infected", "AP50_infected_lo", "AP50_infected_hi", "dAP50_vs_base", "dAP50_lo",
            "dAP50_hi", "recall_infected", "recall_infected_lo", "recall_infected_hi", "drecall_vs_base",
            "drecall_lo", "drecall_hi"]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}.csv, {out}.json")


if __name__ == "__main__":
    main()
