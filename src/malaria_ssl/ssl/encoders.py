"""Loading a frozen ViT encoder for the probe / crop features: references, SSL runs, or a fine-tuned DETR."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import yaml

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD
from malaria_ssl.ssl.models import build_encoder


def load_encoder(cfg: dict[str, Any], ref: str | None = None, run: str | None = None, encoder: str | None = None,
                 detr_run: str | None = None, tag: str | None = None) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    """(encoder, tag, source). Exactly one of ref (random | imagenet), run (SSL run dir), encoder (timm state
    dict path) or detr_run (vit_detr run dir: its ViT after fine-tuning, checkpoint best.pt).
    `source` has input_mean / input_std when the encoder expects a normalization other than ImageNet's."""
    refs = cfg["probe"]["references"]
    if ref == "random":
        torch.manual_seed(refs["random_init_seed"])
        return build_encoder(cfg["model"]["encoder"]), tag or "ref_random_init", {"ref": "random",
                                                                                   "seed": refs["random_init_seed"]}
    if ref == "imagenet":   # fed with its own input normalization (augreg: mean = std = 0.5)
        pc = timm.models.get_pretrained_cfg(refs["imagenet"])
        return (build_encoder(refs["imagenet"], pretrained=True), tag or "ref_imagenet",
                {"ref": "imagenet", "timm": refs["imagenet"], "input_mean": list(pc.mean), "input_std": list(pc.std)})
    if detr_run:
        return _from_detr(Path(detr_run), tag)
    if run:
        run_dir = Path(run)
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        path, name, default_tag = run_dir / "checkpoints" / "encoder_final.pt", run_cfg["model"]["encoder"], run_dir.name
    else:
        path, name, default_tag = Path(encoder), cfg["model"]["encoder"], Path(encoder).stem
    enc = build_encoder(name)
    enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return enc, tag or default_tag, {"encoder_path": str(path)}


def _from_detr(run_dir: Path, tag: str | None) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    """The ViT inside a trained vit_detr run (best.pt) and the input normalization it was fine-tuned with."""
    from malaria_ssl.detr.build import ViTBackbone, build_model

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if cfg["model"]["name"] != "vit_detr":
        raise ValueError(f"{run_dir} is not a vit_detr run")
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(cfg["model"], ck["class_names"], cfg["seed"])
    model.load_state_dict(ck["model"])
    bb = next(m for m in model.modules() if isinstance(m, ViTBackbone))
    # ViTBackbone maps ImageNet-normalized input x to (x * scale + shift); invert to the encoder's mean / std
    std = torch.as_tensor(IMAGENET_STD) / bb.scale.flatten()
    mean = torch.as_tensor(IMAGENET_MEAN) - bb.shift.flatten() * std
    source = {"detr_run": str(run_dir), "checkpoint": "best", "checkpoint_epoch": ck["epoch"],
              "input_mean": [float(v) for v in mean], "input_std": [float(v) for v in std]}
    return bb.encoder, tag or f"detr_{run_dir.name}", source
