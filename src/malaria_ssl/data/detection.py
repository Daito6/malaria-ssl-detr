"""Detection dataset, bbox-aware augmentations and collate for HF DetrForObjectDetection.

Targets follow the original DETR convention: boxes are (cx, cy, w, h) normalized by the size of
the resized, UNPADDED image; padding is described by `pixel_mask`. Each target also keeps the
original image id and the per-axis scale so predictions can be mapped back to original pixels.

Determinism: the training sampler yields `(index, sample_seed)` pairs; the dataset reseeds the
albumentations pipeline with `sample_seed` before augmenting, so a batch depends only on
(seed, epoch, position) and not on the number of workers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from malaria_ssl.data.convert import to_binary
from malaria_ssl.data.stats import load_coco

log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LABEL_MODES = ("binary", "multiclass")


# --------------------------------------------------------------------------- labels

def coco_for_mode(coco: dict[str, Any], mode: str, data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the COCO dict in the requested label mode (binary drops classes mapped to null)."""
    if mode == "multiclass":
        return coco
    if mode == "binary":
        out, dropped = to_binary(coco, data_cfg["binary"]["mapping"], data_cfg["binary"]["categories"])
        log.info("binary mode: dropped %s", dropped)
        return out
    raise ValueError(f"label mode must be one of {LABEL_MODES}, got {mode!r}")


def load_split_coco(split: str, mode: str, data_cfg: dict[str, Any], resolve) -> dict[str, Any]:
    coco_dir = resolve(data_cfg["paths"]["coco_dir"])
    return coco_for_mode(load_coco(coco_dir / f"{split}.json"), mode, data_cfg)


def contiguous_labels(coco: dict[str, Any]) -> tuple[dict[int, int], list[int], list[str]]:
    """COCO category id -> label 0..K-1 (sorted by id), the inverse list, and class names."""
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_label = {c["id"]: i for i, c in enumerate(cats)}
    return cat_to_label, [c["id"] for c in cats], [c["name"] for c in cats]


# --------------------------------------------------------------------------- transforms

class HEDJitter(A.ImageOnlyTransform):
    """Stain jitter in Haematoxylin-Eosin-DAB optical-density space (Tellez et al., 2018).

    BBBC041 smears are Giemsa-stained, so this is used as a generic stain-like colour
    perturbation rather than a faithful H&E model. Each channel: s' = s * alpha + beta,
    alpha ~ U(1 - sigma, 1 + sigma), beta ~ U(-sigma, sigma).
    """

    RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
    RGB_FROM_HED = RGB_FROM_HED / np.linalg.norm(RGB_FROM_HED, axis=1, keepdims=True)
    HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)

    def __init__(self, sigma: float = 0.03, p: float = 0.5):
        super().__init__(p=p)
        self.sigma = sigma

    def get_params(self) -> dict[str, Any]:
        rng = self.random_generator
        return {"alpha": rng.uniform(1 - self.sigma, 1 + self.sigma, 3),
                "beta": rng.uniform(-self.sigma, self.sigma, 3)}

    def apply(self, img: np.ndarray, alpha: np.ndarray, beta: np.ndarray, **params) -> np.ndarray:
        rgb = np.clip(img.astype(np.float32) / 255.0, 1e-6, 1.0)
        stains = (-np.log(rgb)) @ self.HED_FROM_RGB
        stains = stains * alpha + beta
        out = np.exp(-(stains @ self.RGB_FROM_HED))
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def stain_shift_transforms(scfg: dict[str, Any]) -> list[A.BasicTransform]:
    """Strong photometric shift of the stain-shifted probe (HED, colour jitter, blur, JPEG), each with p=1."""
    cj = scfg["color_jitter"]
    return [
        HEDJitter(sigma=scfg["hed_sigma"], p=1.0),
        A.ColorJitter(brightness=cj["brightness"], contrast=cj["contrast"], saturation=cj["saturation"],
                      hue=cj["hue"], p=1.0),
        A.GaussianBlur(sigma_limit=tuple(scfg["blur_sigma"]), p=1.0),
        A.ImageCompression(quality_range=tuple(scfg["jpeg_quality"]), p=1.0),
    ]


def build_stain_shift(scfg: dict[str, Any]) -> A.Compose:
    """Fixed-per-image stain shift (stain-shifted probe and the synthetic stain-shifted val)."""
    return A.Compose(stain_shift_transforms(scfg))


def apply_stain_shift(img: np.ndarray, transform: A.Compose, seed: int, image_id: int) -> np.ndarray:
    """Shift of one image, fixed by (seed, image id)."""
    transform.set_random_seed(int(np.random.default_rng([seed, int(image_id)]).integers(0, 2**31 - 1)))
    return transform(image=img)["image"]


def build_transforms(aug: dict[str, Any], train: bool) -> A.Compose:
    """Albumentations pipeline from the `augment` config section."""
    bbox_params = A.BboxParams(format="coco", label_fields=["labels"],
                               min_visibility=aug["min_bbox_visibility"] if train else 0.0,
                               min_width=1, min_height=1, clip=True)
    if not train:
        return A.Compose([A.LongestMaxSize(max_size=aug["eval_long_side"])], bbox_params=bbox_params)

    lo, hi, step = aug["train_long_side"]
    t: list[A.BasicTransform] = []
    crop = aug["crop"]
    if crop["p"] > 0:
        f = crop["max_fraction_per_side"]
        t.append(A.RandomCropFromBorders(crop_left=f, crop_right=f, crop_top=f, crop_bottom=f, p=crop["p"]))
    t += [
        A.LongestMaxSize(max_size=list(range(lo, hi + 1, step))),
        A.HorizontalFlip(p=aug["hflip_p"]),
        A.VerticalFlip(p=aug["vflip_p"]),
        A.RandomRotate90(p=aug["rot90_p"]),
    ]
    cj = aug["color_jitter"]
    t.append(A.ColorJitter(brightness=cj["brightness"], contrast=cj["contrast"], saturation=cj["saturation"],
                           hue=cj["hue"], p=cj["p"]))
    if aug["hed_jitter"]["enabled"]:
        t.append(HEDJitter(sigma=aug["hed_jitter"]["sigma"], p=aug["hed_jitter"]["p"]))
    bl = aug["blur"]
    t.append(A.GaussianBlur(sigma_limit=tuple(bl["sigma"]), p=bl["p"]))
    jp = aug["jpeg"]
    t.append(A.ImageCompression(quality_range=tuple(jp["quality"]), p=jp["p"]))
    sa = aug.get("stain_aug", {})
    if sa.get("p", 0) > 0:  # appended only when on, so the random stream of older configs is unchanged
        t.append(A.Sequential(stain_shift_transforms(sa), p=sa["p"]))
    return A.Compose(t, bbox_params=bbox_params)


# --------------------------------------------------------------------------- dataset

def read_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB uint8 (np.fromfile handles any Windows path)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot decode {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class DetectionDataset(Dataset):
    """COCO split -> (normalized image tensor, DETR target dict).

    `__getitem__` accepts an int index or an `(index, sample_seed)` tuple. `shift` = (stain-shift
    transform, seed) applies a fixed per-image shift (seed, image id) after `transform` (eval only).
    """

    def __init__(self, coco: dict[str, Any], images_dir: Path, transform: A.Compose,
                 image_ids: list[int] | None = None, shift: tuple[A.Compose, int] | None = None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.shift = shift
        self.cat_to_label, self.label_to_cat, self.class_names = contiguous_labels(coco)
        self.images = {im["id"]: im for im in coco["images"]}
        self.ids = sorted(self.images) if image_ids is None else list(image_ids)
        self.anns: dict[int, list[dict[str, Any]]] = {i: [] for i in self.images}
        for a in coco["annotations"]:
            self.anns[a["image_id"]].append(a)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key: int | tuple[int, int]):
        idx, seed = key if isinstance(key, tuple) else (key, None)
        if seed is not None:
            self.transform.set_random_seed(seed)
        info = self.images[self.ids[idx]]
        img = read_rgb(self.images_dir / info["file_name"])
        anns = self.anns[info["id"]]
        out = self.transform(image=img, bboxes=np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4),
                             labels=[self.cat_to_label[a["category_id"]] for a in anns])
        image = out["image"]
        if self.shift is not None:
            image = apply_stain_shift(image, *self.shift, info["id"])
        h, w = image.shape[:2]
        boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
        cxcywh = np.stack([(boxes[:, 0] + boxes[:, 2] / 2) / w, (boxes[:, 1] + boxes[:, 3] / 2) / h,
                           boxes[:, 2] / w, boxes[:, 3] / h], axis=1) if len(boxes) else boxes
        tensor = torch.from_numpy(((image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
                                  .transpose(2, 0, 1).copy())
        target = {
            "class_labels": torch.as_tensor(list(out["labels"]), dtype=torch.long),
            "boxes": torch.from_numpy(cxcywh),
            "image_id": info["id"],
            "orig_size": (info["height"], info["width"]),
            "size": (h, w),
            # per-axis factor resized/original; exact only without crops/rot90 (eval pipeline)
            "scale": (h / info["height"], w / info["width"]),
        }
        return tensor, target


def collate_detr(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> dict[str, Any]:
    """Pad images to the batch max (H, W) at top-left; pixel_mask = 1 on valid pixels."""
    hmax = max(t.shape[1] for t, _ in batch)
    wmax = max(t.shape[2] for t, _ in batch)
    pixel_values = torch.zeros(len(batch), 3, hmax, wmax)
    pixel_mask = torch.zeros(len(batch), hmax, wmax, dtype=torch.long)
    for i, (t, _) in enumerate(batch):
        pixel_values[i, :, : t.shape[1], : t.shape[2]] = t
        pixel_mask[i, : t.shape[1], : t.shape[2]] = 1
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": [tg for _, tg in batch]}


# --------------------------------------------------------------------------- sampling

class SeededSampler(Sampler):
    """Yields `(index, sample_seed)`; order and seeds depend only on (seed, epoch)."""

    def __init__(self, n: int, seed: int, shuffle: bool = True):
        self.n, self.seed, self.shuffle, self.epoch = n, seed, shuffle, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        seeds = rng.integers(0, 2**31 - 1, size=self.n)
        return iter(zip(order.tolist(), seeds.tolist()))


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: seed numpy / python RNGs from torch's per-worker seed."""
    import random
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------- stats

def median_resized_box_side(coco: dict[str, Any], long_side: int) -> float:
    """Median max(w, h) of GT boxes after resizing images to `long_side` (eval pipeline)."""
    scale = {im["id"]: long_side / max(im["width"], im["height"]) for im in coco["images"]}
    sides = [max(a["bbox"][2], a["bbox"][3]) * scale[a["image_id"]] for a in coco["annotations"]]
    return float(np.median(sides)) if sides else float("nan")
