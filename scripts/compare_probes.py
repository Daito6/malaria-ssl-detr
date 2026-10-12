"""Comparison table of probe results (val crops only) from reports/tables/probe_<tag>.json.

Usage:
    python scripts/compare_probes.py --tags ref_random_init ref_imagenet ssl_mae_only ssl_con_only ssl_joint \
        --out reports/tables/phase3_final_comparison.csv

Columns: linear AUPRC infected (clean, stain-shifted) with 95% bootstrap CI, linear AUPRC infected at 10%
labels (mean, std over seeds), linear macro-F1 and kNN balanced accuracy with 95% CI. Reads only.
"""
from __future__ import annotations

import argparse

from malaria_ssl.ssl.viz import comparison_table, load_probes
from malaria_ssl.utils.config import resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probes = load_probes(args.tags)
    missing = set(args.tags) - {p["tag"] for p in probes}
    if missing:
        raise SystemExit(f"missing probe json: {sorted(missing)}")
    df = comparison_table(probes)
    out = resolve_path(args.out)
    df.to_csv(out, index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
