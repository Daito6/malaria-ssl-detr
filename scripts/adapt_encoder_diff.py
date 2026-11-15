"""EXPLORATORY paired encoder differences of `adapt.py evaluate` from its saved (run, encoder) parts (CPU, no features).

    python scripts/adapt_encoder_diff.py --parts reports/tables/adapt_nih_parts \
        --pairs ssl_v2_b:ref_dino ssl_v2_b_seed43:ref_dino --out reports/tables/adapt_nih_subset_encdiff.csv

For every box run present for both encoders and every method: A - B per metric, 95% percentile CI on the shared
resamples (NIH: patients). The parts must carry the same resample fingerprint (checked), so the differences are paired.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from malaria_ssl.eval.paired import summarize_diff

METRICS = ["AP50", "precision", "recall", "f1", "crop_auprc", "crop_ece", "par_img_mae", "par_img_bias",
           "par_pat_mae", "par_pat_bias", "screen_auroc"]   # column order of the boots arrays (scripts/adapt.py)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B encoder tags, difference A - B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = {}
    for f in sorted(args.parts.glob("*__*.pt")):
        run, tag = f.stem.split("__", 1)
        part = torch.load(f, weights_only=False)
        if not part.get("complete"):
            raise SystemExit(f"{f} is incomplete")
        parts[(run, tag)] = part
    fps = {p["shared"]["fingerprint"] for p in parts.values()}
    if len(fps) != 1:
        raise SystemExit(f"parts have {len(fps)} different resample fingerprints: not paired")

    rows = []
    for pair in args.pairs:
        a, b = pair.split(":")
        runs = sorted({r for r, t in parts if t == a} & {r for r, t in parts if t == b})
        if not runs:
            raise SystemExit(f"no box run has both {a} and {b}")
        for run in runs:
            pa, pb = parts[(run, a)], parts[(run, b)]
            for (_, _, m), pt in pa["points"].items():
                for j, k in enumerate(METRICS):
                    xa, xb = pt[k], pb["points"][(run, b, m)][k]
                    if not (np.isfinite(xa) and np.isfinite(xb)):
                        continue
                    d = summarize_diff(xa, xb, pa["boots"][(run, a, m)][:, j], pb["boots"][(run, b, m)][:, j])
                    rows.append({"run": run, "method": m, "pair": f"{a} - {b}", "metric": k, **d,
                                 "excludes_0": bool(d["low"] > 0 or d["high"] < 0)})
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.4f")
    print(f"wrote {args.out}: {len(rows)} rows, {len(parts)} parts, fingerprint {fps.pop()[:12]}")


if __name__ == "__main__":
    main()
