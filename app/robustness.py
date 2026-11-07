"""Robustness mode: val reference images, corrupted previews, a result cache and the robustness curve tables.

Only val images are served (ids from data/processed/coco/val.json, files from the raw image folder, read only);
the test split is never read here. Results are cached in memory by (model, image, canonical stack) or
(model, image, corruption, strength).
"""

from __future__ import annotations

import io
import json
import math
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from app import corruptions
from app.inference import InferenceResult
from malaria_ssl.data.detection import load_split_coco, read_rgb
from malaria_ssl.utils.config import load_config, resolve_path

CURVE_PREFIX = "reports/tables/robustness_val"
CACHE_SIZE = 512


@lru_cache(maxsize=1)
def val_images() -> dict[int, dict[str, Any]]:
    """{image id: {id, file_name, width, height}} of the val split."""
    dcfg = load_config(resolve_path("configs/data.yaml"))
    coco = load_split_coco("val", "multiclass", dcfg, resolve_path)
    return {
        im["id"]: {k: im[k] for k in ("id", "file_name", "width", "height")}
        for im in sorted(coco["images"], key=lambda im: im["id"])
    }


@lru_cache(maxsize=1)
def _raw_dir() -> Path:
    return resolve_path(load_config(resolve_path("configs/data.yaml"))["paths"]["raw_root"]) / "images"


def load_val_image(image_id: int) -> np.ndarray:
    images = val_images()
    if image_id not in images:
        raise KeyError(image_id)
    return read_rgb(_raw_dir() / images[image_id]["file_name"])


def preview_png(image_id: int, stack: list[corruptions.Step], max_side: int | None = None) -> bytes:
    """PNG of the model input under a stack (eval resize + corruptions; one corruption = a stack of one), in memory."""
    img = corruptions.stack_model_input(load_val_image(image_id), stack, image_id)
    im = Image.fromarray(img)
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def parse_stack(text: str) -> list[corruptions.Step]:
    """'blur:0.5,jpeg:0.25' -> [('blur', 0.5), ('jpeg', 0.25)] (query-string form of a stack). Raises ValueError."""
    steps = []
    for part in filter(None, text.split(",")):
        kind, sep, s = part.partition(":")
        if not sep:
            raise ValueError(f"stack step {part!r} is not kind:strength")
        steps.append((kind.strip(), strength_key(float(s))))
    return steps


def strength_key(s: float) -> float:
    """Strengths are cached on a 1e-3 grid (the UI uses steps of 0.05)."""
    return round(float(s), 3)


class ResultCache:
    """LRU of robustness results keyed by (model, image, corruption, strength)."""

    def __init__(self, size: int = CACHE_SIZE):
        self._d: OrderedDict[tuple, InferenceResult] = OrderedDict()
        self._lock = threading.Lock()
        self.size = size

    def get(self, key: tuple) -> InferenceResult | None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key: tuple, value: InferenceResult) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def _clean(v: Any) -> Any:
    return None if isinstance(v, float) and not math.isfinite(v) else v


def read_curve(prefix: str = CURVE_PREFIX) -> dict[str, Any] | None:
    """Rows of reports/tables/robustness_val.csv + the protocol from its .json; None if not computed yet."""
    csv, meta = resolve_path(f"{prefix}.csv"), resolve_path(f"{prefix}.json")
    if not csv.is_file():
        return None
    rows = pd.read_csv(csv).to_dict("records")
    protocol = json.loads(meta.read_text(encoding="utf-8")).get("protocol", {}) if meta.is_file() else {}
    return {
        "source": f"{prefix}.csv",
        "protocol": protocol,
        "rows": [{k: _clean(v) for k, v in r.items()} for r in rows],
    }
