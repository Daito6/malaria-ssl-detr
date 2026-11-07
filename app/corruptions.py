"""Image corruptions for the robustness mode and scripts/robustness_curve.py.

Every corruption has a strength s in [0, 1] mapped to real parameters (`describe`); s = 0 returns the input
unchanged. They act on the image AT THE MODEL INPUT, i.e. after the eval resize to the long side, where
evaluate.py --stain-shift applies its shift (DetectionDataset(shift=...)): resizing the whole image would be undone
by the fixed eval resize, so the scale corruptions change that long side instead (`long_side`).

    stain                HEDJitter(0.08 s) + ColorJitter(0.3 s, 0.3 s, 0.3 s, hue 0.05 s): the colour part of the
                         stain-shifted probe (configs/ssl/probe.yaml probe.stain_shift), same transforms
                         (malaria_ssl.data.detection), fixed per image by (seed, image id) via apply_stain_shift
    stain_full           s = 1 only: the full probe stain shift (HED + jitter + blur 0.5-1 + JPEG 30-60) = the
                         synthetic stain-shifted val of evaluate.py --stain-shift (control point)
    blur                 Gaussian, sigma = 2.0 s px at the model input
    jpeg                 JPEG quality round(95 - 85 s)
    brightness_contrast  contrast x(1 - 0.5 s) around the per-channel mean, then brightness x(1 - 0.3 s) (darker only)
    scale_down           eval long side round(800 * 0.8^s): 640 at s = 1 (the --long-side 640 point of phase 4)
    scale_up             eval long side round(800 * 1.27^s): 1016 at s = 1 (raw test / val cell size ratio 1.27)

A STACK is an ordered list of (kind, strength) steps (`apply_stack`): a demonstration of combined corruptions, not
part of the curve protocol. Steps with strength 0 are dropped (`canonical`), so an all-zero stack is the original
input; at most one scale step. A scale step that comes first is the eval resize itself (exactly the single
corruption); a later one resamples the already corrupted input to its long side.

All functions are pure and deterministic: the only randomness (stain) is seeded by (seed, image id).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import albumentations as A
import cv2
import numpy as np

from malaria_ssl.data.detection import HEDJitter, apply_stain_shift, build_stain_shift
from malaria_ssl.utils.config import load_experiment

KINDS = ("stain", "blur", "jpeg", "brightness_contrast", "scale_down", "scale_up")
CONTROL_KINDS = ("stain_full",)
STAIN_CONFIG = "configs/ssl/probe.yaml"
BLUR_SIGMA_MAX = 2.0
JPEG_Q0, JPEG_Q1 = 95, 10
CONTRAST_MAX, BRIGHTNESS_MAX = 0.5, 0.3
SCALE_DOWN, SCALE_UP = 0.8, 1.27
SCALE_KINDS = ("scale_down", "scale_up")
# default order of a stack: scale (the resize) -> stain -> optics -> illumination -> compression last
STACK_ORDER = ("scale_down", "scale_up", "stain", "blur", "brightness_contrast", "jpeg")
MAX_STACK = len(KINDS)

Step = tuple[str, float]


@lru_cache(maxsize=1)
def stain_config() -> dict[str, Any]:
    """probe.stain_shift of configs/ssl/probe.yaml (read only)."""
    return load_experiment(STAIN_CONFIG)["probe"]["stain_shift"]


def _check(kind: str, s: float) -> None:
    if kind not in KINDS + CONTROL_KINDS:
        raise ValueError(f"unknown corruption {kind!r}; one of {KINDS + CONTROL_KINDS}")
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {s}")
    if kind == "stain_full" and s not in (0.0, 1.0):
        raise ValueError("stain_full is defined only at strength 0 or 1")


def long_side(kind: str, s: float, base: int = 800) -> int:
    """Eval long side of the model input under this corruption."""
    _check(kind, s)
    if kind == "scale_down":
        return int(round(base * SCALE_DOWN**s))
    if kind == "scale_up":
        return int(round(base * SCALE_UP**s))
    return base


def describe(kind: str, s: float, base: int = 800) -> dict[str, Any]:
    """The real parameters behind a strength (shown in the UI, written to the curve tables)."""
    _check(kind, s)
    if s == 0:
        return {"identity": True, "long_side": base}
    sc = stain_config()
    cj = sc["color_jitter"]
    if kind == "stain":
        return {
            "hed_sigma": sc["hed_sigma"] * s,
            "brightness": cj["brightness"] * s,
            "contrast": cj["contrast"] * s,
            "saturation": cj["saturation"] * s,
            "hue": cj["hue"] * s,
            "seed": sc["seed"],
            "long_side": base,
        }
    if kind == "stain_full":
        return {"probe_stain_shift": dict(sc), "long_side": base}
    if kind == "blur":
        return {"sigma_px": BLUR_SIGMA_MAX * s, "long_side": base}
    if kind == "jpeg":
        return {"quality": jpeg_quality(s), "long_side": base}
    if kind == "brightness_contrast":
        return {"contrast_factor": 1 - CONTRAST_MAX * s, "brightness_factor": 1 - BRIGHTNESS_MAX * s, "long_side": base}
    return {"scale": long_side(kind, s, base) / base, "long_side": long_side(kind, s, base)}


def jpeg_quality(s: float) -> int:
    return int(round(JPEG_Q0 - (JPEG_Q0 - JPEG_Q1) * s))


def _stain(img: np.ndarray, s: float, image_id: int) -> np.ndarray:
    sc = stain_config()
    cj = sc["color_jitter"]
    t = A.Compose(
        [
            HEDJitter(sigma=sc["hed_sigma"] * s, p=1.0),
            A.ColorJitter(
                brightness=cj["brightness"] * s,
                contrast=cj["contrast"] * s,
                saturation=cj["saturation"] * s,
                hue=cj["hue"] * s,
                p=1.0,
            ),
        ]
    )
    return apply_stain_shift(img, t, sc["seed"], image_id)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _brightness_contrast(img: np.ndarray, s: float) -> np.ndarray:
    f = img.astype(np.float32)
    mean = f.reshape(-1, f.shape[-1]).mean(0)
    f = ((f - mean) * (1 - CONTRAST_MAX * s) + mean) * (1 - BRIGHTNESS_MAX * s)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def apply(img: np.ndarray, kind: str, s: float, image_id: int = 0) -> np.ndarray:
    """Photometric part of a corruption on the model-input image (RGB uint8 HxWx3). Scale corruptions return a copy:
    they act through `long_side`. `image_id` seeds the stain shift (as evaluate.py --stain-shift)."""
    _check(kind, s)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an RGB uint8 HxWx3 image, got {img.dtype} {img.shape}")
    if s == 0 or kind in ("scale_down", "scale_up"):
        return img.copy()
    if kind == "stain":
        out = _stain(img, s, image_id)
    elif kind == "stain_full":
        sc = stain_config()
        out = apply_stain_shift(img, build_stain_shift(sc), sc["seed"], image_id)
    elif kind == "blur":
        out = cv2.GaussianBlur(img, (0, 0), sigmaX=BLUR_SIGMA_MAX * s, sigmaY=BLUR_SIGMA_MAX * s)
    elif kind == "jpeg":
        out = _jpeg(img, jpeg_quality(s))
    else:
        out = _brightness_contrast(img, s)
    return np.ascontiguousarray(out, dtype=np.uint8)


def model_input(img: np.ndarray, kind: str, s: float, image_id: int = 0, base: int = 800) -> np.ndarray:
    """Original image -> what the model sees: eval resize (LongestMaxSize as build_transforms(train=False)) to the
    corruption's long side, then the photometric corruption. Used for the UI preview."""
    resized = A.LongestMaxSize(max_size=long_side(kind, s, base))(image=img)["image"]
    return apply(resized, kind, s, image_id)


def canonical(stack: Sequence[Step]) -> tuple[Step, ...]:
    """Validated stack without strength-0 steps (the cache key of a stack). Raises ValueError."""
    if len(stack) > MAX_STACK:
        raise ValueError(f"a stack has at most {MAX_STACK} steps")
    out: list[Step] = []
    for kind, s in stack:
        _check(kind, float(s))
        if kind in CONTROL_KINDS:
            raise ValueError(f"{kind} is a control point of the curve, not a stack step")
        if any(k == kind for k, _ in out):
            raise ValueError(f"{kind} appears twice in the stack")
        if s > 0:
            out.append((kind, float(s)))
    if sum(k in SCALE_KINDS for k, _ in out) > 1:
        raise ValueError("at most one scale step in a stack")
    return tuple(out)


def stack_long_side(stack: Sequence[Step], base: int = 800) -> int:
    """Eval long side for a stack: the scale step's when it comes first, else the base."""
    st = canonical(stack)
    return long_side(*st[0], base) if st and st[0][0] in SCALE_KINDS else base


def apply_stack(img: np.ndarray, stack: Sequence[Step], image_id: int = 0, base: int = 800) -> np.ndarray:
    """The stack on the model-input image, already resized to `stack_long_side`. A later scale step resamples the
    corrupted image to its long side with the eval resize (LongestMaxSize)."""
    st = canonical(stack)
    out = apply(img, "blur", 0.0)  # validates the image, returns a copy
    for i, (kind, s) in enumerate(st):
        if kind in SCALE_KINDS:
            if i > 0:
                out = np.ascontiguousarray(A.LongestMaxSize(max_size=long_side(kind, s, base))(image=out)["image"])
        else:
            out = apply(out, kind, s, image_id)
    return out


def describe_stack(stack: Sequence[Step], base: int = 800) -> dict[str, Any]:
    """Real parameters of every (non-zero) step and the final long side of the model input."""
    st = canonical(stack)
    scale = [k for k in st if k[0] in SCALE_KINDS]
    return {
        "steps": [{"kind": k, "strength": s, "params": describe(k, s, base)} for k, s in st],
        "long_side": long_side(*scale[0], base) if scale else base,
        "identity": not st,
    }


def stack_model_input(img: np.ndarray, stack: Sequence[Step], image_id: int = 0, base: int = 800) -> np.ndarray:
    """Original image -> what the model sees under a stack (the UI preview)."""
    resized = A.LongestMaxSize(max_size=stack_long_side(stack, base))(image=img)["image"]
    return apply_stack(resized, stack, image_id, base)
