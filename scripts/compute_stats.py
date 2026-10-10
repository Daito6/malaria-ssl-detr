"""Compute dataset statistics and write JSON + CSV tables to reports/tables/.

Usage:
    python scripts/compute_stats.py --config configs/data.yaml [--out reports/tables]
"""
from __future__ import annotations

import argparse
import json

from malaria_ssl.data.stats import compute_stats, load_coco, save_stats
from malaria_ssl.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="reports/tables")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    coco_dir = resolve_path(cfg["paths"]["coco_dir"])
    cocos = {s: load_coco(coco_dir / f"{s}.json") for s in ("train", "val", "test")}
    summary, tables = compute_stats(cocos)
    save_stats(summary, tables, resolve_path(args.out))
    print(json.dumps({k: summary[k] for k in ("splits", "boxes_per_image", "max_boxes_image")}, indent=2))
    print(tables["class_counts"].to_string())


if __name__ == "__main__":
    main()
