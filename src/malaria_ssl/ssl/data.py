"""Two-view SSL data: decoded-image cache, overlapping random-resized crops, seeded sampling.

Cache: train/val images (all 1600x1200) are decoded once, resized to long side 960 and stored as
one uint8 array per split (`{split}_960.npy`, opened as a memmap) plus `{split}_index.json`
(image ids, file names, original sizes). Reading a crop then touches only the crop's rows.

Views (per sample, all randomness from the sample seed):
    1. training long side L ~ {640, 672, ..., 960} (the DETR train range), scale s = L / 960;
    2. an anchor window of `anchor` px at training scale (anchor / s px in the cache), uniform in
       the image;
    3. two random-resized crops inside the anchor (area fraction in `crop_scale`, aspect ratio in
       `crop_ratio`), each read from the cache and resized once to `out_size` (equivalent to
       resizing the image to L first and cropping, without resizing the whole image);
    4. per-view flips / rot90 / colour jitter / HED jitter / blur / JPEG, ImageNet normalisation.
With anchor 260 and crop scale [0.5, 1] the crop side is 184-260 px at training scale, i.e. the
crop is rescaled by 0.86-1.22x to 224 (about 1.0 on average): cells keep the DETR scale.

Target domain (phase 5): `make_folds` splits the test images into two folds; `build_target_cache` caches
the images of one fold from the folds file (ids + file names) alone, without opening any annotation file;
`MixedTwoViewDataset` + `MixedImageSampler` draw a fixed fraction of the pairs from those images.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterator

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD, HEDJitter, load_split_coco, read_rgb
from malaria_ssl.utils.config import load_config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- cache

def cache_paths(cache_dir: Path, split: str, long_side: int) -> tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{split}_{long_side}.npy", cache_dir / f"{split}_index.json"


def build_cache(split: str, data_cfg: dict[str, Any], cache_cfg: dict[str, Any], resolve,
                allow_test: bool = False) -> tuple[Path, Path]:
    """Decode + resize all images of a split into a uint8 .npy (idempotent). Never writes to raw data.
    `allow_test` is only for the final test-crop probe evaluation (scripts/probe.py --eval-split test)."""
    if split == "test" and not allow_test:
        raise ValueError("the test split is never used for SSL")
    long_side = cache_cfg["long_side"]
    npy, idx_path = cache_paths(resolve(cache_cfg["dir"]), split, long_side)
    coco = load_split_coco(split, "multiclass", data_cfg, resolve)
    images = sorted(coco["images"], key=lambda im: im["id"])
    sizes = {(im["width"], im["height"]) for im in images}
    if len(sizes) != 1:
        raise ValueError(f"{split}: expected one image size, got {sizes}")
    (w0, h0), = sizes
    scale = long_side / max(w0, h0)
    hc, wc = round(h0 * scale), round(w0 * scale)
    index = {"split": split, "long_side": long_side, "scale": scale, "shape": [len(images), hc, wc, 3],
             "images": [{"id": im["id"], "file_name": im["file_name"], "width": im["width"],
                         "height": im["height"]} for im in images]}
    if npy.exists() and idx_path.exists():
        if json.loads(idx_path.read_text(encoding="utf-8")) == index:
            return npy, idx_path
        log.warning("cache index for %s changed, rebuilding", split)
    npy.parent.mkdir(parents=True, exist_ok=True)
    raw_images = resolve(data_cfg["paths"]["raw_root"]) / "images"
    tmp = npy.with_name(npy.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=tuple(index["shape"]))
    for i, im in enumerate(images):
        arr[i] = cv2.resize(read_rgb(raw_images / im["file_name"]), (wc, hc), interpolation=cv2.INTER_AREA)
        if (i + 1) % 200 == 0:
            log.info("cache %s: %d/%d", split, i + 1, len(images))
    arr.flush()
    del arr
    tmp.replace(npy)
    idx_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    log.info("built %s (%s, %.2f GB)", npy, index["shape"], npy.stat().st_size / 1e9)
    return npy, idx_path


def ensure_caches(cfg: dict[str, Any], resolve, splits: tuple[str, ...] = ("train", "val"),
                  allow_test: bool = False) -> dict[str, Path]:
    data_cfg = load_config(resolve(cfg["data_config"]))
    return {s: build_cache(s, data_cfg, cfg["cache"], resolve, allow_test)[0] for s in splits}


# --------------------------------------------------------------------------- views

def sample_windows(rng: np.random.Generator, hc: int, wc: int, cache_long: int,
                   vcfg: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
    """Training long side, anchor and two crop windows (x0, y0, x1, y1) in cache pixels."""
    lo, hi, step = vcfg["train_long_side"]
    L = float(rng.choice(np.arange(lo, hi + 1, step)))
    a = min(vcfg["anchor"] * cache_long / L, hc, wc)
    ax, ay = rng.uniform(0, wc - a), rng.uniform(0, hc - a)
    anchor = np.array([ax, ay, ax + a, ay + a])
    s_lo, s_hi = vcfg["crop_scale"]
    log_r = np.log(vcfg["crop_ratio"])
    crops = np.zeros((2, 4))
    for k in range(2):
        area = rng.uniform(s_lo, s_hi) * a * a
        r = math.exp(rng.uniform(*log_r))
        w, h = min(math.sqrt(area * r), a), min(math.sqrt(area / r), a)
        x0, y0 = ax + rng.uniform(0, a - w), ay + rng.uniform(0, a - h)
        crops[k] = (x0, y0, x0 + w, y0 + h)
    return L, anchor, crops


def read_crop(img: np.ndarray, box: np.ndarray, out_size: int) -> np.ndarray:
    """Integer-rounded crop of `img` (H, W, 3) resized to out_size x out_size."""
    h, w = img.shape[:2]
    x0, y0 = int(np.floor(box[0])), int(np.floor(box[1]))
    x1, y1 = min(max(int(np.ceil(box[2])), x0 + 1), w), min(max(int(np.ceil(box[3])), y0 + 1), h)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    interp = cv2.INTER_AREA if min(x1 - x0, y1 - y0) > out_size else cv2.INTER_LINEAR
    return cv2.resize(crop, (out_size, out_size), interpolation=interp)


def build_view_transform(vcfg: dict[str, Any]) -> A.Compose:
    """Per-view flips / rot90 / photometric augmentations (no geometry beyond the crop)."""
    cj, bl, jp = vcfg["color_jitter"], vcfg["blur"], vcfg["jpeg"]
    t: list[A.BasicTransform] = [
        A.HorizontalFlip(p=vcfg["hflip_p"]),
        A.VerticalFlip(p=vcfg["vflip_p"]),
        A.RandomRotate90(p=vcfg["rot90_p"]),
        A.ColorJitter(brightness=cj["brightness"], contrast=cj["contrast"], saturation=cj["saturation"],
                      hue=cj["hue"], p=cj["p"]),
    ]
    if vcfg["hed_jitter"]["enabled"]:
        t.append(HEDJitter(sigma=vcfg["hed_jitter"]["sigma"], p=vcfg["hed_jitter"]["p"]))
    t.append(A.GaussianBlur(sigma_limit=tuple(bl["sigma"]), p=bl["p"]))
    t.append(A.ImageCompression(quality_range=tuple(jp["quality"]), p=jp["p"]))
    return A.Compose(t)


def normalize(img: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """uint8 HWC RGB -> normalized float CHW tensor (default: ImageNet mean / std)."""
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    return torch.from_numpy(((img.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).copy())


class TwoViewDataset(Dataset):
    """`ds[(image_index, sample_seed)]` -> (view1, view2, crop windows (2, 4) in cache pixels).

    The cache memmap is opened lazily in each worker. `augment=False` keeps the crops but skips
    the per-view augmentations (used for MAE overfitting / fixed monitoring crops).
    """

    def __init__(self, npy_path: Path, vcfg: dict[str, Any], augment: bool = True, image_ids: list[int] | None = None,
                 input_norm: tuple[Any, Any] | None = None):
        self.npy_path = Path(npy_path)
        self.vcfg = vcfg
        self.augment = augment
        self.norm = input_norm or (IMAGENET_MEAN, IMAGENET_STD)   # the encoder's input (mean, std)
        self.transform = build_view_transform(vcfg) if augment else None
        shape = np.load(self.npy_path, mmap_mode="r").shape
        self.n, self.hc, self.wc = shape[0], shape[1], shape[2]
        self.cache_long = max(self.hc, self.wc)
        self.indices = list(range(self.n)) if image_ids is None else list(image_ids)
        self._arr = None

    def __getstate__(self):
        return {**self.__dict__, "_arr": None}

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def arr(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.load(self.npy_path, mmap_mode="r")
        return self._arr

    def __getitem__(self, key: tuple[int, int]):
        idx, seed = key
        rng = np.random.default_rng(seed)
        _, _, crops = sample_windows(rng, self.hc, self.wc, self.cache_long, self.vcfg)
        view_seeds = rng.integers(0, 2**31 - 1, size=2)
        img = self.arr[self.indices[idx]]
        views = []
        for k in range(2):
            v = read_crop(img, crops[k], self.vcfg["out_size"])
            if self.transform is not None:
                self.transform.set_random_seed(int(view_seeds[k]))
                v = self.transform(image=v)["image"]
            views.append(normalize(v, *self.norm))
        return views[0], views[1], torch.from_numpy(crops).float()


class MixedTwoViewDataset(Dataset):
    """Several TwoViewDatasets (e.g. train + unlabeled target images) behind one global image index:
    `ds[(i, seed)]` reads image i - offset of the source that i falls into."""

    def __init__(self, sources: list[TwoViewDataset]):
        self.sources = sources
        self.offsets = np.cumsum([0] + [len(s) for s in sources])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, key: tuple[int, int]):
        idx, seed = key
        k = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return self.sources[k][(idx - int(self.offsets[k]), seed)]


# --------------------------------------------------------------------------- sampling

class RandomImageSampler(Sampler):
    """An epoch = `samples_per_epoch` draws of (random image index, sample seed).

    Draws depend only on (seed, epoch), so batches do not depend on the number of workers.
    """

    def __init__(self, n_images: int, samples_per_epoch: int, seed: int):
        self.n, self.samples, self.seed, self.epoch = n_images, samples_per_epoch, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        idx = rng.integers(0, self.n, size=self.samples)
        seeds = rng.integers(0, 2**31 - 1, size=self.samples)
        return iter(zip(idx.tolist(), seeds.tolist()))


class MixedImageSampler(Sampler):
    """Like RandomImageSampler over [source images | target images]: each draw comes from the target
    images with probability `target_fraction`, then a uniform image of that source. Depends only on
    (seed, epoch)."""

    def __init__(self, n_source: int, n_target: int, target_fraction: float, samples_per_epoch: int, seed: int):
        if not 0 <= target_fraction <= 1:
            raise ValueError("target_fraction must be in [0, 1]")
        self.n_source, self.n_target, self.fraction = n_source, n_target, target_fraction
        self.samples, self.seed, self.epoch = samples_per_epoch, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        is_target = rng.random(self.samples) < self.fraction
        idx = np.where(is_target, self.n_source + rng.integers(0, self.n_target, size=self.samples),
                       rng.integers(0, self.n_source, size=self.samples))
        seeds = rng.integers(0, 2**31 - 1, size=self.samples)
        return iter(zip(idx.tolist(), seeds.tolist()))


# --------------------------------------------------------------------------- target domain (unlabeled)

def make_folds(image_ids: list[int], n_infected: dict[int, int], seed: int) -> dict[int, int]:
    """Two folds of the images, balanced by the number of infected cells: images sorted by
    (n_infected, id), then within each consecutive pair one image goes to each fold at random
    (an odd last image goes to a random fold). Returns {image id: fold (1 or 2)}."""
    order = sorted(image_ids, key=lambda i: (n_infected.get(i, 0), i))
    rng = np.random.default_rng(seed)
    fold = {}
    for s in range(0, len(order), 2):
        pair = order[s:s + 2]
        for im, f in zip(pair, rng.permutation([1, 2])):
            fold[im] = int(f)
    return fold


def load_fold_images(folds_path: Path, fold: int) -> list[dict[str, Any]]:
    """[{id, file_name}] of one fold from the folds file (image ids and file names only, no labels)."""
    folds = json.loads(Path(folds_path).read_text(encoding="utf-8"))["folds"]
    return sorted(folds[str(fold)], key=lambda im: im["id"])


def build_target_cache(folds_path: Path, fold: int, raw_images: Path, cache_cfg: dict[str, Any],
                       resolve) -> tuple[Path, Path]:
    """Unlabeled target-domain cache of ONE fold: decodes only the image files listed for that fold in
    the folds file. Never opens an annotation file. Idempotent; same layout as `build_cache`."""
    images = load_fold_images(folds_path, fold)
    long_side = cache_cfg["long_side"]
    npy, idx_path = cache_paths(resolve(cache_cfg["dir"]), f"target_fold{fold}", long_side)
    h0, w0 = read_rgb(Path(raw_images) / images[0]["file_name"]).shape[:2]
    scale = long_side / max(w0, h0)
    hc, wc = round(h0 * scale), round(w0 * scale)
    index = {"split": f"target_fold{fold}", "long_side": long_side, "scale": scale, "shape": [len(images), hc, wc, 3],
             "images": [{"id": im["id"], "file_name": im["file_name"]} for im in images]}
    if npy.exists() and idx_path.exists():
        if json.loads(idx_path.read_text(encoding="utf-8")) == index:
            return npy, idx_path
        log.warning("target cache index for fold %d changed, rebuilding", fold)
    npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy.with_name(npy.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=tuple(index["shape"]))
    for i, im in enumerate(images):
        img = read_rgb(Path(raw_images) / im["file_name"])
        if img.shape[:2] != (h0, w0):
            raise ValueError(f"target fold {fold}: {im['file_name']} is {img.shape[:2]}, expected {(h0, w0)}")
        arr[i] = cv2.resize(img, (wc, hc), interpolation=cv2.INTER_AREA)
    arr.flush()
    del arr
    tmp.replace(npy)
    idx_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    log.info("built %s (%s)", npy, index["shape"])
    return npy, idx_path


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two (x0, y0, x1, y1) boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
