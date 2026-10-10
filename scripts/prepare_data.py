"""Prepare BBBC041: download + extract, convert annotations to COCO, make splits.

Usage:
    python scripts/prepare_data.py --config configs/data.yaml [--steps download,convert,split]
"""
from __future__ import annotations

import argparse
import json
import logging

from malaria_ssl.data.convert import convert_dataset, save_json, to_binary
from malaria_ssl.data.download import fetch_dataset
from malaria_ssl.data.splits import make_splits
from malaria_ssl.data.stats import load_coco
from malaria_ssl.utils.config import load_config, resolve_path

ALL_STEPS = ("download", "convert", "split")
log = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--steps", default=",".join(ALL_STEPS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(resolve_path(args.config))
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")

    if "download" in steps:
        d = cfg["download"]
        fetch_dataset(d["url"], resolve_path(d["zip_path"]), resolve_path(d["extract_dir"]))

    coco_dir = resolve_path(cfg["paths"]["coco_dir"])
    if "convert" in steps:
        summary = convert_dataset(cfg, resolve_path)
        # binary mode is derived on the fly; here we only report what it would drop
        summary["binary_dropped"] = {}
        for name in ("train_full", "test"):
            _, dropped = to_binary(load_coco(coco_dir / f"{name}.json"),
                                   cfg["binary"]["mapping"], cfg["binary"]["categories"])
            summary["binary_dropped"][name] = dropped
            log.info("binary mode drops in %s: %s", name, dropped)
        save_json(summary, coco_dir.parent / "convert_summary.json")
        log.info("conversion summary:\n%s", json.dumps(summary, indent=2))

    if "split" in steps:
        meta = make_splits(cfg, resolve_path)
        log.info("split sizes: %s (val fraction %.3f)", meta["n_images"], meta["val_fraction_actual"])
        log.info("near-duplicate pairs: %s", meta["near_duplicate_pairs"])
        log.info("val class counts: %s", meta["class_counts"]["val"])


if __name__ == "__main__":
    main()
