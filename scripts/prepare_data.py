"""Prepare BBBC041: download + extract, convert annotations to COCO, make splits.

Usage:
    python scripts/prepare_data.py --config configs/data.yaml [--steps download,convert,split]
"""
from __future__ import annotations

import argparse
import logging

from malaria_ssl.data.download import fetch_dataset
from malaria_ssl.utils.config import load_config, resolve_path

ALL_STEPS = ("download",)


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


if __name__ == "__main__":
    main()
