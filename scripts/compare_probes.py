"""Comparison table of probe results (val crops only) from reports/tables/probe_<tag>.json.

Usage:
    python scripts/compare_probes.py --tags ref_random_init ref_imagenet ssl_mae_only ssl_con_only ssl_joint \
        --out reports/tables/phase3_final_comparison.csv

--val-test: the same classifiers on val vs test crops (linear and kNN; AUPRC / AUROC infected, macro-F1,
binary balanced accuracy, 95% CIs).

Columns: linear AUPRC infected (clean, stain-shifted) with 95% bootstrap CI, linear AUPRC infected at 10%
labels (mean, std over seeds), linear macro-F1 and kNN balanced accuracy with 95% CI. Reads only.
"""
from __future__ import annotations

import argparse

from malaria_ssl.ssl.viz import comparison_table, load_probes, val_test_probe_table
from malaria_ssl.utils.config import resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-test", action="store_true",
                    help="val vs test crops (needs probe_test_<tag>.json from probe.py --eval-split test)")
    args = ap.parse_args()

    if args.val_test:
        df = val_test_probe_table(args.tags)
        df.to_csv(resolve_path(args.out), index=False, float_format="%.4f")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"wrote {resolve_path(args.out)}")
        return
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
