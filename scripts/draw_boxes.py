"""Save ~12 images with drawn GT boxes for manual verification.

Picks one image per rare class first (so every non-RBC class is shown), then fills up with
random images. Images are drawn at original resolution.

Usage:
    python scripts/draw_boxes.py --config configs/data.yaml [--n 12] [--out reports/figures/box_check]
"""
from __future__ import annotations

import argparse
import random

from malaria_ssl.data.stats import load_coco
from malaria_ssl.data.viz import save_box_check
from malaria_ssl.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--out", default="reports/figures/box_check")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    rng = random.Random(cfg["seed"])
    images_dir = resolve_path(cfg["paths"]["raw_root"]) / "images"
    coco_dir = resolve_path(cfg["paths"]["coco_dir"])
    out_dir = resolve_path(args.out)
    for old in [*out_dir.glob("*.png"), *out_dir.glob("*.jpg")]:
        old.unlink()

    # pool of (split, image_id) over train/val/test; rare classes guaranteed first
    cocos = {s: load_coco(coco_dir / f"{s}.json") for s in ("train", "val", "test")}
    chosen: list[tuple[str, int, str]] = []
    for cat in cocos["train"]["categories"]:
        if cat["name"] == "red blood cell":
            continue
        cands = sorted({(s, a["image_id"]) for s, c in cocos.items() for a in c["annotations"]
                        if a["category_id"] == cat["id"]})
        cands = [c for c in cands if c not in {(s, i) for s, i, _ in chosen}]
        if cands:
            s, i = rng.choice(cands)
            chosen.append((s, i, cat["name"]))
    all_imgs = sorted((s, im["id"]) for s, c in cocos.items() for im in c["images"])
    rest = [x for x in all_imgs if x not in {(s, i) for s, i, _ in chosen}]
    for s, i in rng.sample(rest, max(args.n - len(chosen), 0)):
        chosen.append((s, i, "random"))

    for k, (s, i, reason) in enumerate(chosen[: args.n]):
        fname = next(im["file_name"] for im in cocos[s]["images"] if im["id"] == i)
        out = out_dir / f"{k:02d}_{s}_{reason.replace(' ', '-')}_{fname.rsplit('.', 1)[0]}.jpg"
        save_box_check(cocos[s], i, images_dir, out, title=f"{s} | id {i} | {fname} | pick: {reason}")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
