"""Loading a frozen ViT encoder for the probe / crop features: references, SSL runs, or a fine-tuned DETR."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import yaml

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD
from malaria_ssl.ssl.models import build_encoder


def is_imagenet_norm(mean, std) -> bool:
    return bool(torch.allclose(torch.as_tensor(mean, dtype=torch.float64), torch.as_tensor(IMAGENET_MEAN, dtype=torch.float64))
                and torch.allclose(torch.as_tensor(std, dtype=torch.float64), torch.as_tensor(IMAGENET_STD, dtype=torch.float64)))


def encoder_input_norm(path: Path) -> tuple[list[float], list[float]]:
    """Input (mean, std) of an exported SSL encoder (runs/<run>/checkpoints/<file>.pt): the run's
    `model.input_norm` (e.g. an ImageNet-initialised run keeps timm's 0.5 / 0.5); ImageNet stats if the
    run config has none (random-init runs) or the file is not inside a run."""
    run_cfg = Path(path).resolve().parent.parent / "config.yaml"
    norm = None
    if run_cfg.exists():
        norm = (yaml.safe_load(run_cfg.read_text(encoding="utf-8")).get("model") or {}).get("input_norm")
    if norm is None:
        return [float(v) for v in IMAGENET_MEAN], [float(v) for v in IMAGENET_STD]
    return [float(v) for v in norm["mean"]], [float(v) for v in norm["std"]]


def ssl_run_target_fold(run_dir: Path) -> int | None:
    """Test fold whose unlabeled images the SSL run trained on (`target.fold` of runs/<run>/config.yaml), None if
    the run had no target domain. Raises if the run config is missing (the fold cannot be checked)."""
    run_cfg = Path(run_dir) / "config.yaml"
    if not run_cfg.exists():
        raise FileNotFoundError(f"{run_cfg}: no SSL run config, cannot tell which test fold the encoder saw")
    target = yaml.safe_load(run_cfg.read_text(encoding="utf-8")).get("target")
    return None if not target else int(target["fold"])


def init_encoder(model_cfg: dict[str, Any], imagenet_name: str, resolve,
                 dino_name: str | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    """SSL encoder and its input normalization. `model_cfg.init`: random (default) | imagenet (timm
    `imagenet_name`, its own mean / std) | dino (timm `dino_name`, its own mean / std; phase 7) | path to a timm
    state dict (continued pretraining; the normalization of the run that produced it).
    Returns (encoder, {"mean": [...], "std": [...]})."""
    init, name, dpr = model_cfg.get("init", "random"), model_cfg["encoder"], model_cfg["drop_path_rate"]
    if init == "random":
        enc, mean, std = build_encoder(name, dpr), IMAGENET_MEAN, IMAGENET_STD
    elif init in ("imagenet", "dino"):
        timm_name = imagenet_name if init == "imagenet" else dino_name
        if timm_name is None:
            raise ValueError("init: dino needs probe.references.dino")
        enc = build_encoder(timm_name, dpr, pretrained=True)
        pc = timm.models.get_pretrained_cfg(timm_name)
        mean, std = pc.mean, pc.std
    else:
        enc = build_encoder(name, dpr)
        enc.load_state_dict(torch.load(resolve(init), map_location="cpu", weights_only=True), strict=True)
        mean, std = encoder_input_norm(resolve(init))
    return enc, {"mean": [float(v) for v in mean], "std": [float(v) for v in std]}


def load_encoder(cfg: dict[str, Any], ref: str | None = None, run: str | None = None, encoder: str | None = None,
                 detr_run: str | None = None, tag: str | None = None) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    """(encoder, tag, source). Exactly one of ref (random | imagenet | dino), run (SSL run dir), encoder (timm state
    dict path) or detr_run (vit_detr run dir: its ViT after fine-tuning, checkpoint best.pt).
    `source` has input_mean / input_std when the encoder expects a normalization other than ImageNet's."""
    refs = cfg["probe"]["references"]
    if ref == "random":
        torch.manual_seed(refs["random_init_seed"])
        return build_encoder(cfg["model"]["encoder"]), tag or "ref_random_init", {"ref": "random",
                                                                                   "seed": refs["random_init_seed"]}
    if ref in ("imagenet", "dino"):   # timm weights, fed with timm's own input normalization (pretrained_cfg;
        name = refs[ref]                # augreg ImageNet: mean = std = 0.5)
        pc = timm.models.get_pretrained_cfg(name)
        return (build_encoder(name, pretrained=True), tag or f"ref_{ref}",
                {"ref": ref, "timm": name, "input_mean": list(pc.mean), "input_std": list(pc.std)})
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
    source = {"encoder_path": str(path)}
    mean, std = encoder_input_norm(path)
    if not is_imagenet_norm(mean, std):   # e.g. continued pretraining from the ImageNet ViT (0.5 / 0.5)
        source.update(input_mean=mean, input_std=std)
    return enc, tag or default_tag, source


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
