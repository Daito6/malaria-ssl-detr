"""SmearScan inference on one uploaded image: end-to-end DETR and the two-stage scheme.

Independent of scripts/: the model builder, the eval transform (resize to the eval long side), the input
normalization, the padding mask and the output decoding are the functions evaluate.py uses (imported from
malaria_ssl), applied to one image. On CPU this runs in fp32 (evaluate.py ran bf16 autocast on CUDA).

Display post-processing (not part of the evaluation protocol): class-agnostic NMS at IoU 0.5, so a cell is never
drawn twice whatever score threshold the client applies afterwards (every box shown at a threshold t is an NMS
survivor; suppressed boxes always had a lower score than the box that suppressed them).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torchvision.ops import nms

from app import corruptions, crop_classifier
from app.registry import ModelEntry, ModelUnavailable
from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD, build_transforms, collate_detr
from malaria_ssl.detr.build import build_model, vit_input_norm
from malaria_ssl.eval.inference import outputs_to_candidates, outputs_to_detections
from malaria_ssl.eval.two_stage import candidate_crops, nms_per_image
from malaria_ssl.ssl.encoders import load_encoder
from malaria_ssl.ssl.probe import extract_features
from malaria_ssl.utils.config import PROJECT_ROOT, load_experiment

NMS_IOU = 0.5
AMP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}
Progress = Callable[[str, float], None]
Corrupt = Callable[[np.ndarray], np.ndarray]


class InferenceError(ValueError):
    """The image cannot be processed by this model (message is user-facing)."""


def _noop(stage: str, frac: float) -> None:
    pass


def file_version(path: Path) -> dict[str, str]:
    """Identity of a weights file: path, short sha256 and modification time (UTC)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    mtime = datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC).isoformat(timespec="seconds")
    try:
        rel = str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(path)
    return {"file": rel.replace("\\", "/"), "sha256": h.hexdigest()[:12], "modified": mtime}


def _check_norm(actual: dict[str, list[float]], expected: dict[str, list[float]], what: str) -> None:
    if not (
        np.allclose(actual["mean"], expected["mean"], atol=1e-6)
        and np.allclose(actual["std"], expected["std"], atol=1e-6)
    ):
        raise ModelUnavailable(f"{what}: registry normalization {expected} != model {actual}")


def class_agnostic_nms(dets: list[dict[str, Any]], iou: float = NMS_IOU) -> list[dict[str, Any]]:
    """Keep the highest-scoring box among boxes overlapping with IoU > `iou`, whatever their class; sorted by score."""
    if not dets:
        return []
    xywh = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32)
    xyxy = torch.cat([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], dim=1)
    keep = nms(xyxy, torch.tensor([d["score"] for d in dets], dtype=torch.float32), iou)
    return [dets[i] for i in keep.tolist()]  # torchvision returns survivors by decreasing score


class DetrPredictor:
    """A trained DETR run (config.yaml + checkpoint) applied to single images exactly as in evaluate.py."""

    def __init__(self, entry: ModelEntry, device: torch.device):
        entry.require()
        cfg = yaml.safe_load(entry.files["config"].read_text(encoding="utf-8"))
        ck = torch.load(entry.files["checkpoint"], map_location="cpu", weights_only=False)
        self.class_names = list(ck["class_names"])
        model = build_model(cfg["model"], self.class_names, cfg["seed"])
        model.load_state_dict(ck["model"])
        # ViT-DETR: the encoder's own mean / std (ViTBackbone re-maps the ImageNet-normalized input); R50: ImageNet
        actual = vit_input_norm(model) or {
            "mean": [float(v) for v in IMAGENET_MEAN],
            "std": [float(v) for v in IMAGENET_STD],
        }
        _check_norm(actual, entry.normalization, entry.id)
        self.model = model.to(device).eval()
        self.device = device
        self.amp = AMP[cfg["train"]["amp_dtype"]] if device.type == "cuda" else None
        self.augment = cfg["augment"]
        self.eval_long_side = int(cfg["augment"]["eval_long_side"])
        self._transforms: dict[int, Any] = {}
        self.version = {
            **file_version(entry.files["checkpoint"]),
            "epoch": int(ck["epoch"]),
            "selected_on": ck.get("selected_on", "val"),
        }

    def transform(self, long_side: int | None = None):
        """The eval transform of the run (build_transforms(train=False)), optionally at another long side
        (as evaluate.py --long-side)."""
        ls = long_side or self.eval_long_side
        if ls not in self._transforms:
            self._transforms[ls] = build_transforms({**self.augment, "eval_long_side": ls}, train=False)
        return self._transforms[ls]

    def preprocess(self, img: np.ndarray, long_side: int | None = None, corrupt: Corrupt | None = None) -> dict:
        """One RGB uint8 image -> the batch evaluate.py builds (DetectionDataset.__getitem__ + collate_detr).
        `corrupt` acts on the resized image, where DetectionDataset applies its stain `shift`."""
        h0, w0 = img.shape[:2]
        image = self.transform(long_side)(image=img, bboxes=np.zeros((0, 4), np.float32), labels=[])["image"]
        if corrupt is not None:
            image = corrupt(image)
        h, w = image.shape[:2]
        tensor = torch.from_numpy(
            ((image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
        )
        target = {"image_id": 0, "orig_size": (h0, w0), "size": (h, w), "scale": (h / h0, w / w0)}
        return collate_detr([(tensor, target)])

    @torch.no_grad()
    def forward(self, img: np.ndarray, long_side: int | None = None, corrupt: Corrupt | None = None):
        batch = self.preprocess(img, long_side, corrupt)
        pv = batch["pixel_values"].to(self.device)
        pm = batch["pixel_mask"].to(self.device)
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None):
            out = self.model(pv, pm)
        return out.logits, out.pred_boxes, batch["labels"]

    def detections(
        self, img: np.ndarray, long_side: int | None = None, corrupt: Corrupt | None = None
    ) -> list[dict[str, Any]]:
        """Every query as in evaluate.py: label = argmax over real classes, score = its probability, xywh pixels."""
        logits, boxes, targets = self.forward(img, long_side, corrupt)
        dets = outputs_to_detections(logits, boxes, targets, list(range(len(self.class_names))))
        return [{"label": self.class_names[d["category_id"]], "score": d["score"], "bbox": d["bbox"]} for d in dets]

    def candidates(self, img: np.ndarray) -> list[dict[str, Any]]:
        """Class-agnostic candidates as in scripts/two_stage.py: objectness = 1 - p(no-object)."""
        logits, boxes, targets = self.forward(img)
        return outputs_to_candidates(logits, boxes, targets)

    def predict(
        self,
        img: np.ndarray,
        progress: Progress = _noop,
        long_side: int | None = None,
        corrupt: Corrupt | None = None,
    ) -> list[dict[str, Any]]:
        progress("detect", 0.3)
        dets = self.detections(img, long_side, corrupt)
        progress("postprocess", 0.9)
        return class_agnostic_nms(dets)


class TwoStagePredictor:
    """Class-agnostic DETR candidates (NMS as in the evaluation) + linear classifier on frozen-encoder crop features."""

    def __init__(self, entry: ModelEntry, boxes: DetrPredictor, device: torch.device, cache_dir: Path):
        entry.require()
        ssl_cfg = load_experiment(entry.files["ssl_config"])
        self.pcfg = ssl_cfg["probe"]
        self.cache_long_side = ssl_cfg["cache"]["long_side"]
        enc, _, source = load_encoder(ssl_cfg, ref=entry.spec["encoder"]["ref"])
        train_source = torch.load(entry.files["train_features"], map_location="cpu", weights_only=False)["source"]
        if source != train_source:
            raise ModelUnavailable(
                f"{entry.id}: encoder {source} differs from the one of the saved features {train_source}"
            )
        if "input_mean" in source:
            _check_norm({"mean": source["input_mean"], "std": source["input_std"]}, entry.normalization, entry.id)
        self.input_norm = (source["input_mean"], source["input_std"]) if "input_mean" in source else None
        self.encoder = enc.to(device).eval()
        self.device = device
        self.amp = torch.bfloat16 if device.type == "cuda" else None  # as scripts/two_stage.py features
        self.boxes = boxes
        self.variant = entry.spec["variant"]
        ref = json.loads(entry.files["reference_candidates"].read_text(encoding="utf-8"))
        self.nms_iou = float(ref["nms_iou"])
        self.classifier = crop_classifier.load_or_build(entry, self, cache_dir)
        self.version = {
            "boxes": boxes.version,
            "encoder": source.get("timm", entry.spec["encoder"]["ref"]),
            "classifier": self.classifier["meta"]["cache_file"],
            "train_features": file_version(entry.files["train_features"]),
        }

    def features(self, img: np.ndarray, cands: list[dict[str, Any]]) -> torch.Tensor:
        """Probe crop features of the candidates (candidate order): the image goes through the SSL cache resize
        (INTER_AREA to the cache long side), then extract_features as in scripts/two_stage.py features."""
        if not cands:
            return torch.zeros(0, self.encoder.embed_dim)
        h0, w0 = img.shape[:2]
        s = self.cache_long_side / max(w0, h0)
        hc, wc = round(h0 * s), round(w0 * s)
        if min(hc, wc) * self.pcfg["long_side"] / max(hc, wc) < self.pcfg["window"]:
            raise InferenceError(
                f"image aspect ratio too extreme for the {self.pcfg['window']} px crop window "
                f"at long side {self.pcfg['long_side']} ({w0}x{h0})"
            )
        cache = cv2.resize(img, (wc, hc), interpolation=cv2.INTER_AREA)[None]
        index = {"images": [{"id": 0, "width": w0, "height": h0}]}
        crops = candidate_crops([{**c, "image_id": 0} for c in cands], index, self.pcfg["long_side"])
        f = extract_features(self.encoder, cache, crops, self.pcfg, self.device, self.amp, input_norm=self.input_norm)
        feats = torch.zeros_like(f)
        feats[torch.as_tensor(crops["order"])] = f
        return feats

    def predict(self, img: np.ndarray, progress: Progress = _noop) -> list[dict[str, Any]]:
        progress("detect", 0.2)
        cands = nms_per_image([{**c, "image_id": 0} for c in self.boxes.candidates(img)], self.nms_iou)
        progress("classify", 0.5)
        p = crop_classifier.predict_infected(self.classifier, self.features(img, cands))
        progress("postprocess", 0.9)
        dets = []
        for c, pi in zip(cands, p.tolist(), strict=True):
            obj = c["objectness"] if self.variant == "det_x_p" else 1.0
            inf, uninf = obj * pi, obj * (1 - pi)
            dets.append(
                {"label": "infected" if inf >= uninf else "uninfected", "score": max(inf, uninf), "bbox": c["bbox"]}
            )
        dets.sort(key=lambda d: -d["score"])
        return dets  # candidates are already class-agnostic NMS survivors (IoU self.nms_iou)


@dataclass
class InferenceResult:
    model_id: str
    width: int
    height: int
    detections: list[dict[str, Any]]
    nms_iou: float
    device: str
    elapsed_ms: float
    version: dict[str, Any] = field(default_factory=dict)
    corruption: dict[str, Any] | None = None


class ModelPool:
    """Lazily loaded predictors, one per model id (thread-safe; the job queue runs one inference at a time)."""

    def __init__(self, registry: dict[str, ModelEntry], device: str = "cpu", cache_dir: Path | None = None):
        self.registry = registry
        self.device = torch.device(device)
        self.cache_dir = cache_dir or crop_classifier.CACHE_DIR
        self._loaded: dict[str, DetrPredictor | TwoStagePredictor] = {}
        self._lock = threading.RLock()

    def entry(self, model_id: str) -> ModelEntry:
        if model_id not in self.registry:
            raise KeyError(model_id)
        return self.registry[model_id]

    def get(self, model_id: str) -> DetrPredictor | TwoStagePredictor:
        with self._lock:
            if model_id not in self._loaded:
                e = self.entry(model_id)
                e.require()
                if e.kind == "detr":
                    self._loaded[model_id] = DetrPredictor(e, self.device)
                else:
                    self._loaded[model_id] = TwoStagePredictor(
                        e, self.get(e.spec["boxes"]), self.device, self.cache_dir
                    )
            return self._loaded[model_id]

    def run(
        self,
        model_id: str,
        img: np.ndarray,
        progress: Progress = _noop,
        corruption: tuple[str, float, int] | None = None,
        stack: tuple[Sequence[corruptions.Step], int] | None = None,
    ) -> InferenceResult:
        """Inference + display NMS. `corruption` = (kind, strength, image id) or `stack` = (steps, image id) applies
        app.corruptions at the model input (end-to-end DETR models only); a single corruption is a stack of one."""
        t0 = time.perf_counter()
        progress("load_model", 0.05)
        pred = self.get(model_id)
        info = None
        if corruption is not None:
            stack = ([corruption[:2]], corruption[2])
        if stack is None:
            dets = pred.predict(img, progress)
        else:
            if not isinstance(pred, DetrPredictor):
                raise InferenceError(f"{model_id}: corruptions are supported for end-to-end DETR models only")
            steps, image_id = stack
            base = pred.eval_long_side
            ls = corruptions.stack_long_side(steps, base)
            dets = pred.predict(img, progress, ls, lambda im: corruptions.apply_stack(im, steps, image_id, base))
            info = {"image_id": image_id, **corruptions.describe_stack(steps, base)}
            if corruption is not None:
                kind, s, _ = corruption
                info.update(kind=kind, strength=s, params=corruptions.describe(kind, s, base))
        nms_iou = pred.nms_iou if isinstance(pred, TwoStagePredictor) else NMS_IOU
        return InferenceResult(
            model_id=model_id,
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            detections=dets,
            nms_iou=nms_iou,
            device=str(self.device),
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
            version=pred.version,
            corruption=info,
        )
