"""Split the test images into two folds for the self-supervised domain-adaptation experiment (phase 5).

Folds are balanced by the number of infected cells (`make_folds`: sort by count, random fold within each
consecutive pair; fixed seed). The test annotations are read HERE only to count infected cells per image;
the folds file itself holds image ids and file names only, so the SSL loader never sees any label.

Usage:
    python scripts/make_test_folds.py            # data/splits/test_folds.json + reports/tables/phase5_test_folds.csv

Idempotent: an existing folds file that differs from the recomputed one is an error (never silently
re-split after runs have used it).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from malaria_ssl.data.detection import load_split_coco
from malaria_ssl.ssl.data import make_folds
from malaria_ssl.utils.config import load_config, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/splits/test_folds.json")
    ap.add_argument("--table", default="reports/tables/phase5_test_folds.csv")
    args = ap.parse_args()

    data_cfg = load_config(resolve_path("configs/data.yaml"))
    coco = load_split_coco("test", "multiclass", data_cfg, resolve_path)
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    mapping = data_cfg["binary"]["mapping"]
    infected = Counter(a["image_id"] for a in coco["annotations"] if mapping[cat_name[a["category_id"]]] == "infected")
    cells = Counter(a["image_id"] for a in coco["annotations"])
    images = sorted(coco["images"], key=lambda im: im["id"])
    fold = make_folds([im["id"] for im in images], infected, args.seed)

    folds = {"seed": args.seed, "split": "test",
             "balance": "sorted by number of infected cells, random fold within consecutive pairs",
             "folds": {str(f): [{"id": im["id"], "file_name": im["file_name"]} for im in images if fold[im["id"]] == f]
                       for f in (1, 2)}}
    out = resolve_path(args.out)
    if out.exists():
        if json.loads(out.read_text(encoding="utf-8")) != folds:
            raise SystemExit(f"{out} exists and differs from the recomputed folds; refusing to overwrite")
    else:
        out.write_text(json.dumps(folds, indent=1), encoding="utf-8")

    df = pd.DataFrame([{"image_id": im["id"], "file_name": im["file_name"], "fold": fold[im["id"]],
                        "n_infected": infected[im["id"]], "n_cells": cells[im["id"]]} for im in images])
    df.to_csv(resolve_path(args.table), index=False)
    summary = df.groupby("fold").agg(images=("image_id", "size"), infected=("n_infected", "sum"),
                                     images_with_infected=("n_infected", lambda v: int((v > 0).sum())),
                                     cells=("n_cells", "sum"))
    print(summary.to_string())
    print(f"wrote {out} and {resolve_path(args.table)}")


if __name__ == "__main__":
    main()
