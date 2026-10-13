"""Comparison table of finished DETR runs from metrics_<split>.json / metrics.csv / train.log (reads only).

Usage (phase 4 step 4, runs listed simplest first for the tie-break):
    python scripts/compare_detr_runs.py --runs b_vit_imagenet b_vit_imagenet_s32 b_vit_imagenet_llrd \
        b_vit_imagenet_lora b_vit_imagenet_lora_lr5e4 --split val --select \
        --out reports/tables/phase4_recipe.csv --curves reports/tables/phase4_recipe_curves.csv

--select applies the step-4 rule: leader = highest AP50 of --cls; tied = point estimate inside the
leader's 95% bootstrap CI; winner = first tied run in the --runs order.
"""
from __future__ import annotations

import argparse

import pandas as pd

from malaria_ssl.detr.compare import run_summary, select_by_simplicity, shift_rows, val_curve
from malaria_ssl.utils.config import resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run names under runs/")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cls", default="infected")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--curves", help="optional csv of in-training val AP50 of --cls per epoch")
    ap.add_argument("--shift-out", help="optional csv of the val/test diagnose_shift fractions (shift_<run>.json)")
    args = ap.parse_args()

    rows = [run_summary(resolve_path(f"runs/{r}"), args.split, args.cls) for r in args.runs]
    df = pd.DataFrame(rows).drop(columns="primary")
    out = resolve_path(args.out)
    df.to_csv(out, index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}")

    if args.curves:
        curves = pd.concat([val_curve(resolve_path(f"runs/{r}"), args.cls) for r in args.runs], axis=1)
        curves.index.name = "epochs_done"
        curves.to_csv(resolve_path(args.curves), float_format="%.4f")
        print(curves.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"wrote {resolve_path(args.curves)}")

    if args.shift_out:
        tables = resolve_path("reports/tables")
        sh = pd.DataFrame([r for run in args.runs for r in shift_rows(tables / f"shift_{run}.json")])
        sh.to_csv(resolve_path(args.shift_out), index=False, float_format="%.4f")
        print(sh.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"wrote {resolve_path(args.shift_out)}")

    if args.select:
        sel = select_by_simplicity(rows, args.cls)
        lo, hi = sel["leader_ci"]
        print(f"leader {sel['leader']} ({args.split} AP50 {args.cls} CI [{lo:.4f}, {hi:.4f}]); "
              f"tied {sel['tied']}; winner {sel['winner']}")


if __name__ == "__main__":
    main()
