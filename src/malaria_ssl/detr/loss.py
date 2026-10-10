"""DETR set-prediction loss with configurable class weights.

transformers (v5) rebuilds its criterion inside every forward and only exposes
`eos_coefficient`. We reuse its HungarianMatcher / ImageLoss and reproduce the weighting of
`ForObjectDetectionLoss`, but let the cross-entropy class weights come from the config.
The model is therefore called WITHOUT `labels`, and the loss is computed in fp32.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.loss.loss_for_object_detection import HungarianMatcher, ImageLoss


class DetrCriterion(nn.Module):
    def __init__(self, num_labels: int, loss_cfg: dict[str, Any], num_decoder_layers: int, aux_loss: bool):
        super().__init__()
        matcher = HungarianMatcher(class_cost=loss_cfg["class_cost"], bbox_cost=loss_cfg["bbox_cost"],
                                   giou_cost=loss_cfg["giou_cost"])
        self.image_loss = ImageLoss(matcher=matcher, num_classes=num_labels,
                                    eos_coef=loss_cfg["eos_coefficient"],
                                    losses=["labels", "boxes", "cardinality"])
        weights = loss_cfg.get("class_weights")
        if weights is not None:
            if len(weights) != num_labels:
                raise ValueError(f"class_weights has {len(weights)} entries, expected {num_labels}")
            self.image_loss.empty_weight[:num_labels] = torch.as_tensor(weights, dtype=torch.float)
        base = {"loss_ce": 1.0, "loss_bbox": loss_cfg["bbox_loss_coefficient"],
                "loss_giou": loss_cfg["giou_loss_coefficient"]}
        self.weight_dict = dict(base)
        if aux_loss:
            for i in range(num_decoder_layers - 1):
                self.weight_dict.update({f"{k}_{i}": v for k, v in base.items()})
        self.aux_loss = aux_loss

    def forward(self, outputs, targets: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """`outputs` is a DetrObjectDetectionOutput produced with output of intermediate layers.

        Returns (total weighted loss, dict of all unweighted loss terms).
        """
        out = {"logits": outputs.logits.float(), "pred_boxes": outputs.pred_boxes.float()}
        if self.aux_loss and outputs.auxiliary_outputs is not None:
            out["auxiliary_outputs"] = [{"logits": a["logits"].float(), "pred_boxes": a["pred_boxes"].float()}
                                        for a in outputs.auxiliary_outputs]
        tg = [{"class_labels": t["class_labels"], "boxes": t["boxes"]} for t in targets]
        losses = self.image_loss(out, tg)
        total = sum(losses[k] * self.weight_dict[k] for k in losses if k in self.weight_dict)
        return total, losses
