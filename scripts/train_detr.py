"""Train a DETR model.

Usage:
    python scripts/train_detr.py --config configs/experiments/a_coco_detr_binary.yaml
    python scripts/train_detr.py --config ... --resume
    python scripts/train_detr.py --config ... --name dry_run --set train.epochs=2 --set train.max_steps=50
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from malaria_ssl.detr.trainer import Trainer
from malaria_ssl.utils.config import apply_overrides, load_experiment, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default=None, help="run name (default: config file stem)")
    parser.add_argument("--resume", action="store_true", help="continue from runs/<name>/checkpoints/last.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    name = args.name or Path(args.config).stem
    run_dir = resolve_path("runs") / name
    if args.resume:
        with (run_dir / "config.yaml").open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        if (run_dir / "checkpoints" / "last.pt").exists():
            parser.error(f"{run_dir} already has checkpoints; use --resume or another --name")
        cfg = apply_overrides(load_experiment(args.config), args.set)
    cfg["run_name"] = name

    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")])
    trainer = Trainer(cfg, run_dir)
    trainer.setup(resume=args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
