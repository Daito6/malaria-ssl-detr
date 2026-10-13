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
import torch.nn.functional as F
from torch import nn
from transformers import DetrConfig, DetrForObjectDetection

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD
from malaria_ssl.utils.config import resolve_path

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


class ViTBackbone(nn.Module):
    """timm ViT as the DETR backbone, with the interface of HF `DetrConvEncoder`:
    forward(pixel_values, pixel_mask) -> [(feature_map (B, C, h, w), mask (B, h, w) bool)].

    - Input normalization: DETR images are normalized with ImageNet mean/std; they are re-mapped
      exactly (per-channel affine) to the mean/std the encoder was trained with.
    - The ViT needs H, W divisible by the patch size, so images are zero-padded at the bottom/right
      (to a multiple of `stride`) and the padding is masked like DETR batch padding. The ViT itself
      does not use the mask (global attention also sees padding); DETR's transformer does.
    - Feature map: final-normed patch tokens (prefix tokens dropped), stride = patch size (16).
      stride 32: 2x2 space-to-depth (C -> 4C), so a 1x1 input projection on top equals a learned
      2x2 / stride-2 convolution, while the mask and DETR's position embedding stay consistent.
    """

    def __init__(self, encoder: nn.Module, enc_mean, enc_std, stride: int = 16):
        super().__init__()
        self.encoder = encoder
        self.frozen = False
        self.patch = encoder.patch_embed.patch_size[0]
        if stride not in (self.patch, 2 * self.patch):
            raise ValueError(f"stride must be {self.patch} or {2 * self.patch}, got {stride}")
        self.stride = stride
        in_mean, in_std = torch.as_tensor(IMAGENET_MEAN), torch.as_tensor(IMAGENET_STD)
        enc_mean, enc_std = torch.as_tensor(enc_mean, dtype=torch.float32), torch.as_tensor(enc_std, dtype=torch.float32)
        # x_enc = (x * in_std + in_mean - enc_mean) / enc_std
        self.register_buffer("scale", (in_std / enc_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((in_mean - enc_mean) / enc_std).view(1, 3, 1, 1), persistent=False)

    def freeze(self) -> None:
        """Fixed feature extractor: no ViT parameter trains and the ViT stays in eval mode (no dropout /
        drop-path), also after model.train(); no autograd graph is built through it."""
        self.frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "ViTBackbone":
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    @property
    def num_channels(self) -> int:
        return self.encoder.embed_dim * (self.stride // self.patch) ** 2

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        x = pixel_values * self.scale + self.shift
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        x = F.pad(x, (0, pw, 0, ph))
        mask = F.pad(pixel_mask, (0, pw, 0, ph))
        gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        tokens = self.encoder.forward_features(x)[:, self.encoder.num_prefix_tokens:]
        fmap = tokens.transpose(1, 2).reshape(x.shape[0], -1, gh, gw)
        if self.stride != self.patch:
            fmap = F.pixel_unshuffle(fmap, self.stride // self.patch)
        mask = F.interpolate(mask[None].float(), size=fmap.shape[-2:]).to(torch.bool)[0]  # as DetrConvEncoder
        return [(fmap, mask)]


def build_vit_backbone(vit_cfg: dict[str, Any]) -> ViTBackbone:
    """`vit_cfg.init`: random | imagenet | path to a timm state dict (e.g. an SSL encoder_final.pt)."""
    import timm

    from malaria_ssl.ssl.models import build_encoder

    init, dpr = vit_cfg["init"], vit_cfg.get("drop_path_rate", 0.0)
    mean, std = IMAGENET_MEAN, IMAGENET_STD  # random init and our SSL encoders use the DETR normalization
    if init == "random":
        enc = build_encoder(vit_cfg["arch"], drop_path_rate=dpr)
    elif init == "imagenet":
        enc = build_encoder(vit_cfg["imagenet"], drop_path_rate=dpr, pretrained=True)
        pc = timm.models.get_pretrained_cfg(vit_cfg["imagenet"])
        mean, std = pc.mean, pc.std
    else:
        from malaria_ssl.ssl.encoders import encoder_input_norm

        enc = build_encoder(vit_cfg["arch"], drop_path_rate=dpr)
        enc.load_state_dict(torch.load(resolve_path(init), map_location="cpu", weights_only=True), strict=True)
        mean, std = encoder_input_norm(resolve_path(init))   # the SSL run's input norm (ImageNet unless recorded)
    log.info("ViT backbone %s (init %s), input mean %s std %s, stride %d",
             vit_cfg["arch"], init, tuple(float(v) for v in mean), tuple(float(v) for v in std), vit_cfg["stride"])
    return ViTBackbone(enc, mean, std, vit_cfg["stride"])


def _set_head(hf_model: DetrForObjectDetection, class_names: list[str]) -> None:
    k = len(class_names)
    hf_model.class_labels_classifier = nn.Linear(hf_model.config.d_model, k + 1)  # +1 = no-object
    hf_model.config.num_labels = k
    hf_model.config.id2label = dict(enumerate(class_names))
    hf_model.config.label2id = {n: i for i, n in enumerate(class_names)}


def apply_lora(hf_model: DetrForObjectDetection, lora_cfg: dict[str, Any]) -> None:
    """LoRA (peft) on the linear layers `lora.targets` of the DETR transformer (encoder and decoder
    layers, self- and cross-attention; the backbone is never matched). The transformer's base weights
    (incl. FFN and norms) are frozen; LoRA weights, backbone, input projection, queries and heads train.
    LoRA B starts at zero, so the model function is unchanged at initialization."""
    from peft import LoraConfig, inject_adapter_in_model

    targets = "|".join(lora_cfg["targets"])
    cfg = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                     target_modules=rf"(encoder|decoder)\.layers\.\d+\..*\.({targets})")
    inject_adapter_in_model(cfg, hf_model.model)
    # peft marks only adapter weights trainable on hf_model.model; set the intended policy explicitly
    for n, p in hf_model.named_parameters():
        in_transformer = n.startswith(("model.encoder.", "model.decoder."))
        p.requires_grad_("lora_" in n or not in_transformer)
    n_lora = sum(p.numel() for n, p in hf_model.named_parameters() if "lora_" in n)
    if n_lora == 0:
        raise RuntimeError(f"LoRA matched no layers for targets {lora_cfg['targets']}")
    log.info("LoRA r=%d alpha=%s on %s: %d adapter parameters", lora_cfg["r"], lora_cfg["alpha"], targets, n_lora)


def _coco_detr(model_cfg: dict[str, Any], class_names: list[str], gen: torch.Generator) -> DetrForObjectDetection:
    """COCO-pretrained DETR with a new class head and the query table expanded to num_queries."""
    hf = DetrForObjectDetection.from_pretrained(model_cfg["pretrained"])
    hf.config.auxiliary_loss = False  # see module docstring
    _set_head(hf, class_names)
    old = hf.model.query_position_embeddings.weight.data
    emb = nn.Embedding(model_cfg["num_queries"], old.shape[1])
    emb.weight.data.copy_(expand_queries(old, model_cfg["num_queries"], model_cfg["extra_query_noise"], gen))
    hf.model.query_position_embeddings = emb
    hf.config.num_queries = model_cfg["num_queries"]
    return hf


def build_model(model_cfg: dict[str, Any], class_names: list[str], seed: int) -> Detr:
    """Build `coco_detr_r50`, `scratch_r50` or `vit_detr` (see configs/detr/base.yaml).

    vit_detr: the COCO-pretrained DETR transformer, queries and box head (new class head as in
    coco_detr_r50) on a timm ViT backbone (`model.vit`); the ResNet is replaced and the input
    projection (2048 -> 256) is re-created for the ViT channels, randomly initialized.
    `model.adapt`: full (the whole transformer trains) | lora (see `apply_lora`); vit_detr only.
    """
    name = model_cfg["name"]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # new head / random transformer init
    if name == "coco_detr_r50":
        hf = _coco_detr(model_cfg, class_names, gen)
    elif name == "vit_detr":
        hf = _coco_detr(model_cfg, class_names, gen)
        backbone = build_vit_backbone(model_cfg["vit"])
        hf.model.backbone = backbone
        hf.model.input_projection = nn.Conv2d(backbone.num_channels, hf.config.d_model, kernel_size=1)
        adapt = model_cfg.get("adapt", "full")
        if adapt == "lora":
            apply_lora(hf, model_cfg["lora"])
        elif adapt != "full":
            raise ValueError(f"unknown adapt {adapt!r} (full | lora)")
        if model_cfg["vit"].get("freeze", False):   # after apply_lora, which marks non-transformer params trainable
            backbone.freeze()
        return Detr(hf, aux_loss=model_cfg["auxiliary_loss"])  # the whole ViT is trainable unless vit.freeze
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


def vit_layer_id(name: str, num_layers: int) -> int:
    """Depth of a timm ViT parameter (name relative to the ViT) for layer-wise lr decay: embeddings and
    prefix tokens 0, `blocks.i` -> i + 1, everything after the blocks (final norm) num_layers + 1."""
    if name.startswith(("cls_token", "reg_token", "pos_embed", "patch_embed")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers + 1


def param_groups(model: Detr, optim_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Groups (order fixed, the trainer logs 0, 1 and 2): transformer + heads + queries at lr, backbone
    at lr_backbone. vit_detr only: the new input projection is a third group at lr_projection (null = lr),
    and with `optim.layer_decay` d the ViT gets layer-wise lr decay: a parameter at depth i (see
    `vit_layer_id`) trains at lr_backbone * d ** (L + 1 - i); depth L + 1 stays in "backbone", the other
    depths follow as "backbone_layer<i>" groups. The R50 models keep two groups, so their existing
    checkpoints still resume."""
    split_proj = isinstance(model.hf.model.backbone, ViTBackbone)
    groups = {"transformer": [], "backbone": [], "projection": []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            key = ("backbone" if ".backbone." in f".{n}" else
                   "projection" if split_proj and ".input_projection." in f".{n}" else "transformer")
            groups[key].append(p)
    lr_proj = optim_cfg.get("lr_projection")
    lrs = {"transformer": optim_cfg["lr"], "backbone": optim_cfg["lr_backbone"],
           "projection": optim_cfg["lr"] if lr_proj is None else lr_proj}
    names = ["transformer", "backbone", "projection"] if split_proj else ["transformer", "backbone"]
    decay = optim_cfg.get("layer_decay")
    if split_proj and decay is not None:
        enc = model.hf.model.backbone.encoder
        top = len(enc.blocks) + 1
        depth = {id(p): vit_layer_id(n, len(enc.blocks)) for n, p in enc.named_parameters()}
        layers = groups.pop("backbone")
        groups["backbone"] = [p for p in layers if depth[id(p)] == top]
        for i in range(top):
            groups[f"backbone_layer{i}"] = [p for p in layers if depth[id(p)] == i]
            lrs[f"backbone_layer{i}"] = optim_cfg["lr_backbone"] * decay ** (top - i)
            names.append(f"backbone_layer{i}")
    return [{"params": groups[k], "lr": lrs[k], "weight_decay": optim_cfg["weight_decay"], "name": k}
            for k in names if groups[k] or k in ("transformer", "backbone")]
