"""Detection data pipeline tests (real-data tests are skipped if data is not prepared)."""
import numpy as np
import pytest
import torch

from malaria_ssl.data.detection import (
    DetectionDataset, SeededSampler, build_transforms, collate_detr, contiguous_labels,
    coco_for_mode, load_split_coco, median_resized_box_side,
)
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

CFG = load_experiment("configs/detr/base.yaml")
DATA = load_config(resolve_path(CFG["data_config"]))
IMAGES = resolve_path(DATA["paths"]["raw_root"]) / "images"
HAVE_DATA = (resolve_path(DATA["paths"]["coco_dir"]) / "train.json").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="processed data not found")


def _toy_coco():
    cats = [{"id": i + 1, "name": c["name"]} for i, c in enumerate(DATA["categories"])]
    anns = [{"id": k + 1, "image_id": 1, "category_id": k + 1, "bbox": [0, 0, 5, 5]} for k in range(7)]
    return {"images": [{"id": 1, "file_name": "x.png", "width": 10, "height": 10}],
            "annotations": anns, "categories": cats}


def test_label_mapping_multiclass_and_binary():
    coco = _toy_coco()
    cat_to_label, label_to_cat, names = contiguous_labels(coco_for_mode(coco, "multiclass", DATA))
    assert cat_to_label == {i: i - 1 for i in range(1, 8)} and label_to_cat == list(range(1, 8))
    assert names[0] == "red blood cell" and names[6] == "difficult"
    b = coco_for_mode(coco, "binary", DATA)
    _, _, bnames = contiguous_labels(b)
    assert bnames == ["uninfected", "infected"]
    name = {c["id"]: c["name"] for c in b["categories"]}
    got = sorted(name[a["category_id"]] for a in b["annotations"])
    # RBC + leukocyte -> uninfected; 4 parasite stages -> infected; difficult dropped
    assert got == ["infected"] * 4 + ["uninfected"] * 2


@needs_data
@pytest.mark.parametrize("train", [True, False])
def test_transforms_keep_boxes_valid(train):
    coco = load_split_coco("train", "multiclass", DATA, resolve_path)
    ds = DetectionDataset(coco, IMAGES, build_transforms(CFG["augment"], train=train))
    for k in range(6):
        img, t = ds[(k, 1000 + k)]
        h, w = img.shape[1:]
        assert t["size"] == (h, w)
        assert max(h, w) == (CFG["augment"]["eval_long_side"] if not train else max(h, w))
        b = t["boxes"]
        assert b.shape[0] == t["class_labels"].shape[0] and b.shape[0] > 0
        assert torch.all(b[:, 2:] > 0)
        x0, y0 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
        x1, y1 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
        eps = 1e-5
        assert torch.all(x0 >= -eps) and torch.all(y0 >= -eps) and torch.all(x1 <= 1 + eps) and torch.all(y1 <= 1 + eps)
        if not train:  # eval keeps every box
            assert b.shape[0] == len(ds.anns[ds.ids[k]])
        lo, hi, _ = CFG["augment"]["train_long_side"]
        if train:
            assert max(h, w) <= hi


@needs_data
def test_collate_shapes_and_pixel_mask():
    coco = load_split_coco("val", "binary", DATA, resolve_path)
    ds = DetectionDataset(coco, IMAGES, build_transforms(CFG["augment"], train=True))
    items = [ds[(k, k)] for k in range(3)]
    batch = collate_detr(items)
    H = max(t.shape[1] for t, _ in items)
    W = max(t.shape[2] for t, _ in items)
    assert batch["pixel_values"].shape == (3, 3, H, W)
    assert batch["pixel_mask"].shape == (3, H, W)
    for i, (t, _) in enumerate(items):
        m = batch["pixel_mask"][i]
        assert m[: t.shape[1], : t.shape[2]].all()
        assert m.sum() == t.shape[1] * t.shape[2]
        assert torch.equal(batch["pixel_values"][i, :, : t.shape[1], : t.shape[2]], t)
    assert len(batch["labels"]) == 3


@needs_data
def test_resized_box_size_train_vs_test_within_10pct():
    side = CFG["augment"]["eval_long_side"]
    tr = median_resized_box_side(load_split_coco("train", "multiclass", DATA, resolve_path), side)
    te = median_resized_box_side(load_split_coco("test", "multiclass", DATA, resolve_path), side)
    assert abs(te - tr) / tr < 0.10, (tr, te)


@needs_data
def test_same_seed_same_first_batch():
    coco = load_split_coco("train", "binary", DATA, resolve_path)

    def first_batch():
        ds = DetectionDataset(coco, IMAGES, build_transforms(CFG["augment"], train=True))
        sampler = SeededSampler(len(ds), seed=CFG["seed"])
        sampler.set_epoch(3)
        keys = list(sampler)[:2]
        return collate_detr([ds[k] for k in keys])

    a, b = first_batch(), first_batch()
    assert torch.equal(a["pixel_values"], b["pixel_values"])
    for ta, tb in zip(a["labels"], b["labels"]):
        assert ta["image_id"] == tb["image_id"] and torch.equal(ta["boxes"], tb["boxes"])


def test_sampler_epochs_differ_and_cover_all():
    s = SeededSampler(50, seed=1)
    e0 = list(s)
    s.set_epoch(1)
    e1 = list(s)
    assert sorted(i for i, _ in e0) == list(range(50))
    assert [i for i, _ in e0] != [i for i, _ in e1]
    s.set_epoch(0)
    assert list(s) == e0


def test_hed_jitter_seeded_and_changes_image():
    import albumentations as A
    from malaria_ssl.data.detection import HEDJitter
    img = np.random.default_rng(0).integers(30, 230, (32, 32, 3), dtype=np.uint8)
    pipe = A.Compose([HEDJitter(sigma=0.05, p=1.0)])
    pipe.set_random_seed(7)
    a = pipe(image=img)["image"]
    pipe.set_random_seed(7)
    b = pipe(image=img)["image"]
    pipe.set_random_seed(8)
    c = pipe(image=img)["image"]
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.abs(a.astype(int) - img).mean() > 0.5
    # sigma=0 must be (almost) the identity
    ident = A.Compose([HEDJitter(sigma=0.0, p=1.0)])(image=img)["image"]
    assert np.abs(ident.astype(int) - img).max() <= 1
