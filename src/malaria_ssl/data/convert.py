"""Convert the official BBBC041 annotations (training.json / test.json) to COCO.

Source format (one record per image):
    {"image": {"pathname": "/images/<file>.png", "shape": {"r": H, "c": W, "channels": 3}},
     "objects": [{"category": "ring",
                  "bounding_box": {"minimum": {"r": y0, "c": x0}, "maximum": {"r": y1, "c": x1}}}, ...]}
where r = row = y and c = column = x. Boxes are kept in ORIGINAL pixel coordinates.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_source(json_path: Path) -> list[dict[str, Any]]:
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def source_bbox_to_coco(bbox: dict[str, dict[str, float]]) -> list[float]:
    """(min_r, min_c, max_r, max_c) -> COCO [x, y, w, h] with x = c, y = r."""
    y0, x0 = bbox["minimum"]["r"], bbox["minimum"]["c"]
    y1, x1 = bbox["maximum"]["r"], bbox["maximum"]["c"]
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def image_file_name(record: dict[str, Any]) -> str:
    """File name relative to the raw `images/` folder."""
    return Path(record["image"]["pathname"]).name


def list_image_files(images_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def assign_image_ids(file_names: list[str]) -> dict[str, int]:
    """Stable ids: 1-based position in the sorted list of ALL image files."""
    return {name: i for i, name in enumerate(sorted(file_names), start=1)}


def build_coco(
    records: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    images_dir: Path,
    image_ids: dict[str, int],
    check_files: bool = True,
) -> dict[str, Any]:
    """Build a multiclass COCO dict from source records.

    The image size is taken from the JSON `shape`; with `check_files` the file md5 is verified against
    the JSON `checksum` and the size against the real file header.
    """
    cat_id = {c["name"]: c["id"] for c in categories}
    images, annotations = [], []
    ann_id = 1
    for rec in records:
        fname = image_file_name(rec)
        h, w = int(rec["image"]["shape"]["r"]), int(rec["image"]["shape"]["c"])
        if check_files:
            path = Path(images_dir) / fname
            if hashlib.md5(path.read_bytes()).hexdigest() != rec["image"]["checksum"]:
                raise ValueError(f"{fname}: md5 does not match the checksum in the annotation file")
            with Image.open(path) as im:
                if im.size != (w, h):
                    raise ValueError(f"{fname}: JSON shape (w={w}, h={h}) != file size {im.size}")
        img_id = image_ids[fname]
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for obj in rec["objects"]:
            if obj["category"] not in cat_id:
                raise KeyError(f"{fname}: unknown category {obj['category']!r}")
            x, y, bw, bh = source_bbox_to_coco(obj["bounding_box"])
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id[obj["category"]],
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1
    images.sort(key=lambda im: im["id"])
    return {"images": images, "annotations": annotations, "categories": copy.deepcopy(categories)}


def orientation_check(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count box corners exceeding image bounds under both r/c interpretations.

    Interpretation A (correct if r=row): r <= H and c <= W.
    Interpretation B (swapped):           r <= W and c <= H.
    """
    out = Counter()
    for rec in records:
        h, w = rec["image"]["shape"]["r"], rec["image"]["shape"]["c"]
        for obj in rec["objects"]:
            r1, c1 = obj["bounding_box"]["maximum"]["r"], obj["bounding_box"]["maximum"]["c"]
            out["boxes"] += 1
            out["violations_r_is_row"] += int(r1 > h or c1 > w)
            out["violations_r_is_col"] += int(r1 > w or c1 > h)
    return dict(out)


def to_binary(
    coco: dict[str, Any],
    mapping: dict[str, str | None],
    binary_categories: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Derive a binary COCO dict; classes mapped to None are dropped.

    Returns the new dict and a per-source-class count of dropped boxes.
    """
    src_name = {c["id"]: c["name"] for c in coco["categories"]}
    missing = set(src_name.values()) - set(mapping)
    if missing:
        raise KeyError(f"binary mapping missing classes: {sorted(missing)}")
    dst_id = {c["name"]: c["id"] for c in binary_categories}
    dropped: Counter[str] = Counter()
    anns = []
    for a in coco["annotations"]:
        name = src_name[a["category_id"]]
        target = mapping[name]
        if target is None:
            dropped[name] += 1
            continue
        anns.append({**a, "category_id": dst_id[target]})
    out = {"images": copy.deepcopy(coco["images"]), "annotations": anns,
           "categories": copy.deepcopy(binary_categories)}
    return out, dict(dropped)


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def describe_unlabeled(images_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    """Size / mode of image files that appear in no annotation file."""
    out = []
    for n in names:
        with Image.open(Path(images_dir) / n) as im:
            out.append({"file_name": n, "width": im.size[0], "height": im.size[1], "mode": im.mode})
    return out


def convert_dataset(cfg: dict[str, Any], resolve) -> dict[str, Any]:
    """Run the full conversion described by the data config. Returns a summary dict."""
    raw_root = resolve(cfg["paths"]["raw_root"])
    images_dir = raw_root / "images"
    coco_dir = resolve(cfg["paths"]["coco_dir"])

    train_rec = load_source(raw_root / "training.json")
    test_rec = load_source(raw_root / "test.json")
    all_files = list_image_files(images_dir)
    image_ids = assign_image_ids(all_files)

    train_names = [image_file_name(r) for r in train_rec]
    test_names = [image_file_name(r) for r in test_rec]
    labeled = set(train_names) | set(test_names)
    summary: dict[str, Any] = {
        "n_image_files": len(all_files),
        "n_train_records": len(train_rec),
        "n_test_records": len(test_rec),
        "n_unique_train": len(set(train_names)),
        "n_unique_test": len(set(test_names)),
        "train_test_overlap": sorted(set(train_names) & set(test_names)),
        "annotated_but_missing_file": sorted(labeled - set(all_files)),
    }
    unlabeled = sorted(set(all_files) - labeled)
    summary["n_unlabeled"] = len(unlabeled)
    if summary["annotated_but_missing_file"]:
        raise FileNotFoundError(f"annotated images missing: {summary['annotated_but_missing_file'][:5]}")

    summary["orientation"] = {
        "train": orientation_check(train_rec), "test": orientation_check(test_rec)
    }

    cats = cfg["categories"]
    train_full = build_coco(train_rec, cats, images_dir, image_ids)
    test = build_coco(test_rec, cats, images_dir, image_ids)
    save_json(train_full, coco_dir / "train_full.json")
    save_json(test, coco_dir / "test.json")
    log.info("train_full: %d images, %d boxes", len(train_full["images"]), len(train_full["annotations"]))
    log.info("test: %d images, %d boxes", len(test["images"]), len(test["annotations"]))

    unl = describe_unlabeled(images_dir, unlabeled)
    save_json(unl, resolve(cfg["paths"]["unlabeled_list"]))
    summary["unlabeled_sizes"] = dict(Counter(f"{u['width']}x{u['height']}" for u in unl))

    # image index over all files
    split_of = {n: "train" for n in train_names} | {n: "test" for n in test_names}
    n_boxes = Counter()
    sizes = {}
    for coco in (train_full, test):
        id2name = {im["id"]: im["file_name"] for im in coco["images"]}
        for im in coco["images"]:
            sizes[im["file_name"]] = (im["width"], im["height"])
        for a in coco["annotations"]:
            n_boxes[id2name[a["image_id"]]] += 1
    for u in unl:
        sizes[u["file_name"]] = (u["width"], u["height"])
    index_path = resolve(cfg["paths"]["images_index"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "file_name", "width", "height", "official_split", "labeled", "n_boxes"])
        for n in all_files:
            wr.writerow([image_ids[n], n, *sizes[n], split_of.get(n, "none"), int(n in labeled), n_boxes[n]])
    return summary
