"""Corruptions: identity at strength 0, determinism, pixel range / format, reuse of the evaluation's stain shift."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from app import corruptions as C
from app.tests.conftest import VAL_IMAGE_ID
from malaria_ssl.data.detection import DetectionDataset, build_transforms, collate_detr, load_split_coco, read_rgb
from malaria_ssl.detr.trainer import val_stain_shift
from malaria_ssl.utils.config import load_config, resolve_path

ALL = C.KINDS + C.CONTROL_KINDS
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def img(val_image_path):
    full = read_rgb(val_image_path)
    return C.model_input(full, "blur", 0.0)  # the clean model input (800 x 600)


@pytest.mark.parametrize("kind", ALL)
def test_strength_zero_is_identity(img, kind):
    out = C.apply(img, kind, 0.0, VAL_IMAGE_ID)
    np.testing.assert_array_equal(out, img)
    assert out is not img
    assert C.long_side(kind, 0.0) == 800


@pytest.mark.parametrize("kind", C.KINDS)
@pytest.mark.parametrize("s", STRENGTHS[1:])
def test_deterministic_range_format(img, kind, s):
    a = C.apply(img, kind, s, VAL_IMAGE_ID)
    b = C.apply(img.copy(), kind, s, VAL_IMAGE_ID)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.uint8 and a.shape == img.shape and a.flags["C_CONTIGUOUS"]
    assert a.min() >= 0 and a.max() <= 255
    if kind not in ("scale_down", "scale_up"):
        assert not np.array_equal(a, img), "a positive strength must change the image"


def test_input_not_modified(img):
    before = img.copy()
    for kind in C.KINDS:
        C.apply(img, kind, 1.0, VAL_IMAGE_ID)
    np.testing.assert_array_equal(img, before)


def test_stain_depends_on_image_id(img):
    assert not np.array_equal(C.apply(img, "stain", 1.0, 1), C.apply(img, "stain", 1.0, 2))


def test_parameter_mapping():
    assert [C.long_side("scale_down", s) for s in STRENGTHS] == [800, 757, 716, 677, 640]
    assert [C.long_side("scale_up", s) for s in STRENGTHS] == [800, 849, 902, 957, 1016]
    assert [C.jpeg_quality(s) for s in STRENGTHS[1:]] == [74, 52, 31, 10]
    assert C.describe("blur", 1.0)["sigma_px"] == 2.0
    d = C.describe("brightness_contrast", 1.0)
    assert d["contrast_factor"] == 0.5 and d["brightness_factor"] == 0.7
    sc = yaml.safe_load(resolve_path("configs/ssl/probe.yaml").read_text("utf-8"))["probe"]["stain_shift"]
    st = C.describe("stain", 1.0)
    assert st["hed_sigma"] == sc["hed_sigma"] and st["hue"] == sc["color_jitter"]["hue"]


def test_monotone_deviation(img):
    """Larger strength moves the image further from the original (same random direction for stain)."""
    for kind in ("stain", "blur", "jpeg", "brightness_contrast"):
        dev = [np.abs(C.apply(img, kind, s, VAL_IMAGE_ID).astype(float) - img).mean() for s in STRENGTHS]
        assert dev[0] == 0 and dev[-1] > dev[1], (kind, dev)


@pytest.mark.parametrize("bad", [("nope", 0.5), ("blur", -0.1), ("blur", 1.5), ("stain_full", 0.5)])
def test_invalid(img, bad):
    with pytest.raises(ValueError):
        C.apply(img, *bad)


def test_rejects_non_rgb_uint8(img):
    with pytest.raises(ValueError):
        C.apply(img.astype(np.float32), "blur", 0.5)


def test_stain_full_is_the_evaluation_stain_shift(pool, val_image_path):
    """At strength 1 stain_full equals DetectionDataset(shift=val_stain_shift) of evaluate.py --stain-shift."""
    entry = pool.entry("a_coco_detr_binary")
    cfg = yaml.safe_load(entry.files["config"].read_text("utf-8"))
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gt = load_split_coco("val", "binary", dcfg, resolve_path)
    gt = {
        **gt,
        "images": [im for im in gt["images"] if im["id"] == VAL_IMAGE_ID],
        "annotations": [a for a in gt["annotations"] if a["image_id"] == VAL_IMAGE_ID],
    }
    ds = DetectionDataset(
        gt,
        resolve_path(dcfg["paths"]["raw_root"]) / "images",
        build_transforms(cfg["augment"], train=False),
        shift=val_stain_shift(cfg),
    )
    ref = collate_detr([ds[0]])["pixel_values"]
    pred = pool.get("a_coco_detr_binary")
    ours = pred.preprocess(read_rgb(val_image_path), None, lambda im: C.apply(im, "stain_full", 1.0, VAL_IMAGE_ID))[
        "pixel_values"
    ]
    assert torch.equal(ours, ref)
