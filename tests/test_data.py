"""Integrity checks on the generated COCO files and splits (skipped if data not prepared)."""
from collections import Counter

import pytest

from malaria_ssl.data.splits import read_split_file
from malaria_ssl.data.stats import load_coco
from malaria_ssl.utils.config import load_config, resolve_path

CFG = load_config(resolve_path("configs/data.yaml"))
COCO_DIR = resolve_path(CFG["paths"]["coco_dir"])
SPLITS_DIR = resolve_path(CFG["paths"]["splits_dir"])
SPLITS = ("train", "val", "test")

pytestmark = pytest.mark.skipif(
    not all((COCO_DIR / f"{s}.json").exists() for s in SPLITS),
    reason="processed data not found; run scripts/prepare_data.py",
)


@pytest.fixture(scope="module")
def cocos():
    return {s: load_coco(COCO_DIR / f"{s}.json") for s in SPLITS}


@pytest.mark.parametrize("split", SPLITS)
def test_boxes_inside_image_and_positive(cocos, split):
    coco = cocos[split]
    size = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    bad = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        W, H = size[a["image_id"]]
        if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= W and y + h <= H):
            bad.append((a["id"], a["bbox"], (W, H)))
    assert not bad, f"{len(bad)} invalid boxes, e.g. {bad[:5]}"


@pytest.mark.parametrize("split", SPLITS)
def test_unique_ids_and_valid_categories(cocos, split):
    coco = cocos[split]
    img_ids = [im["id"] for im in coco["images"]]
    assert len(img_ids) == len(set(img_ids)), "duplicate image ids"
    ann_ids = [a["id"] for a in coco["annotations"]]
    assert len(ann_ids) == len(set(ann_ids)), "duplicate annotation ids"
    valid_cats = {c["id"] for c in CFG["categories"]}
    assert {c["id"] for c in coco["categories"]} == valid_cats
    assert {a["category_id"] for a in coco["annotations"]} <= valid_cats
    assert {a["image_id"] for a in coco["annotations"]} <= set(img_ids)


def test_image_ids_and_files_unique_across_splits(cocos):
    ids = Counter(im["id"] for c in cocos.values() for im in c["images"])
    files = Counter(im["file_name"] for c in cocos.values() for im in c["images"])
    assert max(ids.values()) == 1
    assert max(files.values()) == 1


def test_split_files_disjoint_and_match_coco(cocos):
    lists = {s: read_split_file(SPLITS_DIR / f"{s}.txt") for s in SPLITS}
    for s in SPLITS:
        assert len(lists[s]) == len(set(lists[s]))
        assert set(lists[s]) == {im["file_name"] for im in cocos[s]["images"]}
    assert not set(lists["train"]) & set(lists["val"])
    assert not set(lists["train"]) & set(lists["test"])
    assert not set(lists["val"]) & set(lists["test"])


def test_train_val_partition_official_train(cocos):
    full = load_coco(COCO_DIR / "train_full.json")
    tv = {im["file_name"] for s in ("train", "val") for im in cocos[s]["images"]}
    assert tv == {im["file_name"] for im in full["images"]}
    assert len(cocos["train"]["annotations"]) + len(cocos["val"]["annotations"]) == len(full["annotations"])
