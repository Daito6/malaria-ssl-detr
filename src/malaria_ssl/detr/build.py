"""DETR model builders, query expansion and optimizer parameter groups.

Auxiliary decoding losses: with `config.auxiliary_loss=True`, transformers' DetrDecoder applies
its final LayerNorm to the running hidden state after EVERY layer, which changes the network
function relative to the original DETR (there the norm is applied only to copies used by the
auxiliary heads). We therefore keep `config.auxiliary_loss=False` (exact original computation)
and collect the intermediate decoder outputs with forward hooks, normalizing copies as in the
original implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import DetrConfig, DetrForObjectDetection

log = logging.getLogger(__name__)

# Trainable ResNet stages, as in the original DETR (stem and layer1 frozen).
TRAINABLE_BACKBONE_STAGES = ("layer2", "layer3", "layer4")


@dataclass
class DetrOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor
    auxiliary_outputs: list[dict[str, torch.Tensor]] | None


class Detr(nn.Module):
    """Wraps DetrForObjectDetection; returns final and (optionally) auxiliary predictions."""

    def __init__(self, hf_model: DetrForObjectDetection, aux_loss: bool):
        super().__init__()
        self.hf = hf_model
        self.aux_loss = aux_loss
        self._inter: list[torch.Tensor] = []
        if aux_loss:
            for layer in self.hf.model.decoder.layers:
                layer.register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        self._inter.append(output[0] if isinstance(output, tuple) else output)

    @property
    def num_labels(self) -> int:
        return self.hf.class_labels_classifier.out_features - 1

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor) -> DetrOutput:
        self._inter = []
        out = self.hf(pixel_values=pixel_values, pixel_mask=pixel_mask)
        aux = None
        if self.aux_loss:
            norm = self.hf.model.decoder.layernorm
            aux = []
            for h in self._inter[:-1]:
                h = norm(h)
                aux.append({"logits": self.hf.class_labels_classifier(h),
                            "pred_boxes": self.hf.bbox_predictor(h).sigmoid()})
        self._inter = []
        return DetrOutput(out.logits, out.pred_boxes, aux)


def expand_queries(old: torch.Tensor, num_queries: int, noise: float, generator: torch.Generator) -> torch.Tensor:
    """Grow a (100, d) query embedding table to (num_queries, d).

    Rows 0..99 are copied unchanged. Row i >= 100 is `old[i % 100] + eps`, eps ~ N(0, (noise*std)^2),
    where std is the std of the pretrained table. Tiling transfers the learned positional priors
    of the COCO queries (each query specializes to an image region and box size) to the new ones;
    the small noise breaks the symmetry so duplicated queries can diverge and the Hungarian matcher
    can assign them to different cells. Random init would discard those priors for 2/3 of the
    queries and is known to slow DETR convergence.
    """
    n_old, d = old.shape
    if num_queries <= n_old:
        return old[:num_queries].clone()
    idx = torch.arange(n_old, num_queries) % n_old
    eps = torch.randn(len(idx), d, generator=generator) * noise * old.std()
    return torch.cat([old, old[idx] + eps], dim=0)


def freeze_backbone_stem(hf_model: DetrForObjectDetection) -> None:
    for name, p in hf_model.model.backbone.named_parameters():
        if not any(f".{s}." in f".{name}" for s in TRAINABLE_BACKBONE_STAGES):
            p.requires_grad_(False)


def load_imagenet_backbone(hf_model: DetrForObjectDetection, timm_name: str) -> None:
    """Copy ImageNet weights from timm into the DETR ResNet backbone (incl. frozen-BN statistics).

    Done explicitly because DetrForObjectDetection(config) re-initializes the backbone; the result
    is verified so a silent mismatch cannot leave the backbone random.
    """
    import timm

    src = timm.create_model(timm_name, pretrained=True).state_dict()
    dst = hf_model.model.backbone.model
    own = dst.state_dict()
    shared = {k: v for k, v in src.items() if k in own}
    missing = sorted(set(own) - set(shared) - {k for k in own if k.endswith("num_batches_tracked")})
    if missing:
        raise RuntimeError(f"backbone keys not found in {timm_name}: {missing[:5]}")
    dst.load_state_dict(shared, strict=False)
    for k, v in shared.items():
        if not torch.equal(dst.state_dict()[k].cpu(), v.cpu()):
            raise RuntimeError(f"backbone weight {k} was not loaded")
    log.info("loaded %d ImageNet backbone tensors from timm %s", len(shared), timm_name)


def _set_head(hf_model: DetrForObjectDetection, class_names: list[str]) -> None:
    k = len(class_names)
    hf_model.class_labels_classifier = nn.Linear(hf_model.config.d_model, k + 1)  # +1 = no-object
    hf_model.config.num_labels = k
    hf_model.config.id2label = dict(enumerate(class_names))
    hf_model.config.label2id = {n: i for i, n in enumerate(class_names)}


def build_model(model_cfg: dict[str, Any], class_names: list[str], seed: int) -> Detr:
    """Build `coco_detr_r50` or `scratch_r50` (see configs/detr/base.yaml)."""
    name = model_cfg["name"]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # new head / random transformer init
    if name == "coco_detr_r50":
        hf = DetrForObjectDetection.from_pretrained(model_cfg["pretrained"])
        hf.config.auxiliary_loss = False  # see module docstring
        _set_head(hf, class_names)
        old = hf.model.query_position_embeddings.weight.data
        emb = nn.Embedding(model_cfg["num_queries"], old.shape[1])
        emb.weight.data.copy_(expand_queries(old, model_cfg["num_queries"], model_cfg["extra_query_noise"], gen))
        hf.model.query_position_embeddings = emb
        hf.config.num_queries = model_cfg["num_queries"]
    elif name == "scratch_r50":
        cfg = DetrConfig.from_pretrained(model_cfg["pretrained"])  # same architecture hyper-parameters
        cfg.backbone_config.use_pretrained_backbone = False
        cfg.num_queries = model_cfg["num_queries"]
        cfg.auxiliary_loss = False
        cfg.num_labels = len(class_names)
        cfg.id2label = dict(enumerate(class_names))
        cfg.label2id = {n: i for i, n in enumerate(class_names)}
        hf = DetrForObjectDetection(cfg)
        load_imagenet_backbone(hf, model_cfg["scratch_backbone"])
    else:
        raise ValueError(f"unknown model name {name!r}")
    freeze_backbone_stem(hf)
    return Detr(hf, aux_loss=model_cfg["auxiliary_loss"])


def param_groups(model: Detr, optim_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Backbone at lr_backbone; transformer, input projection, heads and queries at lr."""
    backbone, rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (backbone if ".backbone." in f".{n}" else rest).append(p)
    return [
        {"params": rest, "lr": optim_cfg["lr"], "weight_decay": optim_cfg["weight_decay"], "name": "transformer"},
        {"params": backbone, "lr": optim_cfg["lr_backbone"], "weight_decay": optim_cfg["weight_decay"],
         "name": "backbone"},
    ]
