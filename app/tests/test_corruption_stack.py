"""Corruption stacks (app.corruptions.apply_stack): zero strength is the original, order matters, deterministic,
a stack of one is the single corruption. Pure functions only (no model)."""

from __future__ import annotations

import numpy as np
import pytest

from app import corruptions as C
from app import robustness
from app.tests.conftest import VAL_IMAGE_ID
from malaria_ssl.data.detection import read_rgb


@pytest.fixture(scope="module")
def img(val_image_path):
    return read_rgb(val_image_path)


def test_all_zero_stack_is_the_original(img):
    zeros = [(k, 0.0) for k in C.STACK_ORDER]
    assert C.canonical(zeros) == ()
    assert C.stack_long_side(zeros) == 800
    plain = C.stack_model_input(img, [], VAL_IMAGE_ID)
    assert np.array_equal(C.stack_model_input(img, zeros, VAL_IMAGE_ID), plain)
    assert np.array_equal(plain, C.model_input(img, "blur", 0.0))
    assert C.describe_stack(zeros)["identity"] is True


def test_zero_steps_are_dropped(img):
    stack = [("blur", 0.5), ("jpeg", 0.0), ("stain", 0.25)]
    assert C.canonical(stack) == (("blur", 0.5), ("stain", 0.25))
    small = C.model_input(img, "blur", 0.0)
    assert np.array_equal(
        C.apply_stack(small, stack, VAL_IMAGE_ID), C.apply_stack(small, C.canonical(stack), VAL_IMAGE_ID)
    )


@pytest.mark.parametrize("kind", C.KINDS)
def test_stack_of_one_is_the_single_corruption(img, kind):
    assert np.array_equal(
        C.stack_model_input(img, [(kind, 0.75)], VAL_IMAGE_ID), C.model_input(img, kind, 0.75, VAL_IMAGE_ID)
    )
    assert C.stack_long_side([(kind, 0.75)]) == C.long_side(kind, 0.75)


def test_deterministic(img):
    stack = [("scale_down", 0.5), ("stain", 1.0), ("blur", 0.5), ("brightness_contrast", 0.5), ("jpeg", 0.75)]
    a = C.stack_model_input(img, stack, VAL_IMAGE_ID)
    b = C.stack_model_input(img.copy(), list(stack), VAL_IMAGE_ID)
    assert np.array_equal(a, b)
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
    assert max(a.shape[:2]) == C.long_side("scale_down", 0.5)


def test_order_matters(img):
    small = C.model_input(img, "blur", 0.0)
    bj = C.apply_stack(small, [("blur", 1.0), ("jpeg", 1.0)])
    jb = C.apply_stack(small, [("jpeg", 1.0), ("blur", 1.0)])
    assert bj.shape == jb.shape and not np.array_equal(bj, jb)


def test_scale_position(img):
    first = C.stack_model_input(img, [("scale_up", 1.0), ("blur", 0.5)], VAL_IMAGE_ID)
    later = C.stack_model_input(img, [("blur", 0.5), ("scale_up", 1.0)], VAL_IMAGE_ID)
    assert max(first.shape[:2]) == max(later.shape[:2]) == 1016
    assert first.shape == later.shape and not np.array_equal(first, later)
    assert C.stack_long_side([("blur", 0.5), ("scale_up", 1.0)]) == 800  # eval resize at the base, resampled later
    assert C.describe_stack([("blur", 0.5), ("scale_up", 1.0)])["long_side"] == 1016


@pytest.mark.parametrize(
    "bad",
    [
        [("scale_up", 0.5), ("scale_down", 0.5)],
        [("blur", 0.5), ("blur", 0.25)],
        [("stain_full", 1.0)],
        [("nope", 0.5)],
        [("blur", 1.5)],
    ],
)
def test_invalid_stacks(bad):
    with pytest.raises(ValueError):
        C.canonical(bad)


def test_input_not_modified(img):
    small = C.model_input(img, "blur", 0.0)
    before = small.copy()
    C.apply_stack(small, [("stain", 1.0), ("jpeg", 1.0)], VAL_IMAGE_ID)
    assert np.array_equal(small, before)


def test_parse_stack():
    assert robustness.parse_stack("blur:0.5,jpeg:0.25") == [("blur", 0.5), ("jpeg", 0.25)]
    assert robustness.parse_stack("") == []
    with pytest.raises(ValueError):
        robustness.parse_stack("blur")
