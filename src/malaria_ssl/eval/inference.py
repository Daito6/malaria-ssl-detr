"""Run a DETR model over a split and return COCO-format detections in ORIGINAL pixel coordinates."""
from __future__ import annotations

from typing import Any

import torch

from malaria_ssl.detr.build import Detr


def outputs_to_detections(logits: torch.Tensor, pred_boxes: torch.Tensor, targets: list[dict[str, Any]],
                          label_to_cat: list[int]) -> list[dict[str, Any]]:
    """Convert raw DETR outputs to COCO detections; keeps every query.

    score = max softmax probability over the real classes (no-object excluded);
    boxes: normalized cxcywh of the resized image -> original pixels via the per-axis scale.
    """
    probs = logits.float().softmax(-1)[..., :-1]
    scores, labels = probs.max(-1)
    dets = []
    for b, t in enumerate(targets):
        h, w = t["size"]
        sy, sx = t["scale"]
        oh, ow = t["orig_size"]
        cx, cy, bw, bh = pred_boxes[b].float().unbind(-1)
        x0 = ((cx - bw / 2) * w / sx).clamp(0, ow)
        y0 = ((cy - bh / 2) * h / sy).clamp(0, oh)
        x1 = ((cx + bw / 2) * w / sx).clamp(0, ow)
        y1 = ((cy + bh / 2) * h / sy).clamp(0, oh)
        xywh = torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1).tolist()
        img_id = int(t["image_id"])
        for box, lab, s in zip(xywh, labels[b].tolist(), scores[b].tolist()):
            dets.append({"image_id": img_id, "category_id": int(label_to_cat[lab]), "bbox": box, "score": s})
    return dets


@torch.no_grad()
def predict(model: Detr, loader, device: torch.device, label_to_cat: list[int],
            amp_dtype: torch.dtype | None = torch.bfloat16) -> list[dict[str, Any]]:
    model.eval()
    dets: list[dict[str, Any]] = []
    for batch in loader:
        pv = batch["pixel_values"].to(device, non_blocking=True)
        pm = batch["pixel_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            out = model(pv, pm)
        dets.extend(outputs_to_detections(out.logits, out.pred_boxes, batch["labels"], label_to_cat))
    return dets
