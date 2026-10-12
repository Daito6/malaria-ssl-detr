"""DETR training loop: AdamW + warmup/cosine, bf16 autocast, gradient accumulation, periodic
COCO evaluation on val, best/last checkpoints, resume, and per-epoch logging.

Run directory layout (runs/<name>/):
    config.yaml        resolved config (incl. the probed micro-batch)
    run_info.json      git hash / dirty flag, package versions, GPU
    metrics.csv        one row per epoch (losses, lr, grad norm, time, memory, val metrics)
    tb/                TensorBoard logs
    checkpoints/       best.pt (by val AP), last.pt (full training state)
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from malaria_ssl.data.detection import (
    DetectionDataset, SeededSampler, build_transforms, collate_detr, load_split_coco,
    median_resized_box_side, seed_worker,
)
from malaria_ssl.detr.build import Detr, build_model, param_groups
from malaria_ssl.detr.loss import DetrCriterion
from malaria_ssl.eval.coco_metrics import evaluate_coco
from malaria_ssl.eval.inference import predict
from malaria_ssl.utils.config import load_config, resolve_path
from malaria_ssl.utils.run import append_csv, load_checkpoint, lr_lambda, run_info, save_checkpoint
from malaria_ssl.utils.seed import set_seed

log = logging.getLogger(__name__)

AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}


def group_lr_lambdas(optimizer: torch.optim.Optimizer, optim_cfg: dict[str, Any], total: int) -> list:
    """One LR multiplier per param group: warmup + cosine, shared by all groups except `projection`
    (vit_detr input projection), which has its own warmup `warmup_steps_projection` (null = shared)."""
    shared = lr_lambda(optim_cfg["warmup_steps"], total, optim_cfg["min_lr_ratio"])
    w_proj = optim_cfg.get("warmup_steps_projection")
    proj = shared if w_proj is None else lr_lambda(w_proj, total, optim_cfg["min_lr_ratio"])
    return [proj if g.get("name") == "projection" else shared for g in optimizer.param_groups]


# --------------------------------------------------------------------------- data

def data_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return load_config(resolve_path(cfg["data_config"]))


def build_datasets(cfg: dict[str, Any]) -> tuple[DetectionDataset, DetectionDataset, dict[str, Any]]:
    """Train and val datasets. In overfit mode both are the first N train images, no augmentation."""
    dcfg = data_config(cfg)
    images = resolve_path(dcfg["paths"]["raw_root"]) / "images"
    mode = cfg["label_mode"]
    train_coco = load_split_coco("train", mode, dcfg, resolve_path)
    n_over = cfg["train"].get("overfit_images")
    if n_over:
        ids = sorted(im["id"] for im in train_coco["images"])[:n_over]
        keep = set(ids)
        val_coco = {**train_coco, "images": [i for i in train_coco["images"] if i["id"] in keep],
                    "annotations": [a for a in train_coco["annotations"] if a["image_id"] in keep]}
        eval_tf = build_transforms(cfg["augment"], train=False)
        return (DetectionDataset(train_coco, images, eval_tf, ids),
                DetectionDataset(val_coco, images, build_transforms(cfg["augment"], train=False)), val_coco)
    val_coco = load_split_coco("val", mode, dcfg, resolve_path)
    return (DetectionDataset(train_coco, images, build_transforms(cfg["augment"], train=True)),
            DetectionDataset(val_coco, images, build_transforms(cfg["augment"], train=False)), val_coco)


def eval_loader(ds: DetectionDataset, cfg: dict[str, Any]) -> DataLoader:
    tr = cfg["train"]
    return DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, collate_fn=collate_detr,
                      num_workers=tr["num_workers"], persistent_workers=tr["persistent_workers"] and tr["num_workers"] > 0,
                      pin_memory=True)


def train_loader(ds: DetectionDataset, cfg: dict[str, Any], micro_batch: int) -> tuple[DataLoader, SeededSampler]:
    tr = cfg["train"]
    sampler = SeededSampler(len(ds), seed=cfg["seed"], shuffle=True)
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(ds, batch_size=micro_batch, sampler=sampler, collate_fn=collate_detr, drop_last=False,
                        num_workers=tr["num_workers"], worker_init_fn=seed_worker, generator=g,
                        persistent_workers=tr["persistent_workers"] and tr["num_workers"] > 0, pin_memory=True)
    return loader, sampler


def log_resized_box_sizes(cfg: dict[str, Any]) -> dict[str, float]:
    """Median max(w, h) of GT boxes after the eval resize, per split (logged at start)."""
    dcfg = data_config(cfg)
    out = {s: median_resized_box_side(load_split_coco(s, cfg["label_mode"], dcfg, resolve_path),
                                      cfg["augment"]["eval_long_side"]) for s in ("train", "val", "test")}
    log.info("median box side after resize to %d px: %s", cfg["augment"]["eval_long_side"],
             {k: round(v, 1) for k, v in out.items()})
    return out


# --------------------------------------------------------------------------- helpers

def to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [{"class_labels": t["class_labels"].to(device, non_blocking=True),
             "boxes": t["boxes"].to(device, non_blocking=True)} for t in targets]


def probe_micro_batch(model: Detr, criterion: DetrCriterion, cfg: dict[str, Any], device: torch.device,
                      n_boxes: int = 223) -> dict[str, Any]:
    """Largest micro-batch (dividing effective_batch) whose fwd+bwd peak memory fits the budget.

    Worst case: square canvas of the largest train side (rot90 mixes orientations in a batch)
    and `n_boxes` targets per image (max cells in BBBC041). No optimizer step is taken.
    """
    side = cfg["augment"]["train_long_side"][1]
    eff = cfg["train"]["effective_batch"]
    budget = cfg["train"]["probe_memory_fraction"] * torch.cuda.get_device_properties(device).total_memory
    amp = AMP_DTYPES[cfg["train"]["amp_dtype"]]
    model.train()
    results = {}
    chosen = 1
    for mb in [b for b in (8, 4, 2, 1) if eff % b == 0]:
        pv = pm = tg = out = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            pv = torch.randn(mb, 3, side, side, device=device)
            pm = torch.ones(mb, side, side, dtype=torch.long, device=device)
            g = torch.Generator(device="cpu").manual_seed(0)
            xy = torch.rand(mb, n_boxes, 2, generator=g) * 0.9 + 0.05
            tg = [{"class_labels": torch.zeros(n_boxes, dtype=torch.long, device=device),
                   "boxes": torch.cat([xy[i], torch.full((n_boxes, 2), 0.06)], 1).to(device)} for i in range(mb)]
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                out = model(pv, pm)
            loss, _ = criterion(out, tg)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
            results[mb] = round(peak / 2**30, 2)
            ok = peak <= budget
        except torch.OutOfMemoryError:
            results[mb] = "OOM"
            ok = False
        finally:
            model.zero_grad(set_to_none=True)
            del pv, pm, tg, out, loss
            torch.cuda.empty_cache()
        if ok:
            chosen = mb
            break
    log.info("micro-batch probe (peak GiB per candidate): %s -> %d", results, chosen)
    return {"micro_batch": chosen, "probe_peak_gib": results, "probe_canvas": side,
            "budget_gib": round(budget / 2**30, 2)}


# --------------------------------------------------------------------------- training

class Trainer:
    def __init__(self, cfg: dict[str, Any], run_dir: Path, device: torch.device | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device or torch.device("cuda")
        set_seed(cfg["seed"])
        self.train_ds, self.val_ds, self.val_coco = build_datasets(cfg)
        self.class_names = self.train_ds.class_names
        self.model = build_model(cfg["model"], self.class_names, cfg["seed"]).to(self.device)
        self.criterion = DetrCriterion(len(self.class_names), cfg["loss"],
                                       self.model.hf.config.decoder_layers, cfg["model"]["auxiliary_loss"]).to(self.device)
        self.amp = AMP_DTYPES[cfg["train"]["amp_dtype"]]
        self.optimizer = torch.optim.AdamW(param_groups(self.model, cfg["optim"]))
        n_train = sum(p.numel() for g in self.optimizer.param_groups for p in g["params"])
        log.info("trainable parameters: %d of %d; groups %s", n_train, sum(p.numel() for p in self.model.parameters()),
                 {g["name"]: sum(p.numel() for p in g["params"]) for g in self.optimizer.param_groups})
        self.epoch = 0           # next epoch to run
        self.global_step = 0     # optimizer steps done
        self.best_ap = -1.0
        self.scheduler = None    # created once the micro-batch is known

    # -- setup ---------------------------------------------------------------------------
    def setup(self, resume: bool) -> None:
        tr = self.cfg["train"]
        if tr["micro_batch"] == "auto":
            probe = probe_micro_batch(self.model, self.criterion, self.cfg, self.device)
            tr["micro_batch"] = probe["micro_batch"]
            self.cfg["probe"] = probe
        mb, eff = tr["micro_batch"], tr["effective_batch"]
        if eff % mb:
            raise ValueError(f"effective_batch {eff} not divisible by micro_batch {mb}")
        self.accum = eff // mb
        self.loader, self.sampler = train_loader(self.train_ds, self.cfg, mb)
        self.val_loader = eval_loader(self.val_ds, self.cfg)
        self.steps_per_epoch = math.ceil(len(self.loader) / self.accum)
        total = self.steps_per_epoch * tr["epochs"]
        o = self.cfg["optim"]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, group_lr_lambdas(self.optimizer, o, total))
        if resume:
            self.load(self.run_dir / "checkpoints" / "last.pt")
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "run_info.json").write_text(json.dumps(run_info(), indent=2), encoding="utf-8")
            self.cfg["resized_box_median"] = log_resized_box_sizes(self.cfg)
            with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, sort_keys=False)
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(str(self.run_dir / "tb"), purge_step=self.global_step if resume else None)

    # -- state ---------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "epoch": self.epoch, "global_step": self.global_step,
                "best_ap": self.best_ap, "class_names": self.class_names,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state()}}

    def load(self, path: Path) -> None:
        ck = load_checkpoint(path)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.epoch, self.global_step, self.best_ap = ck["epoch"], ck["global_step"], ck["best_ap"]
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"])
        log.info("resumed from %s at epoch %d (step %d)", path, self.epoch, self.global_step)

    # -- loop ----------------------------------------------------------------------------
    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        self.sampler.set_epoch(self.epoch)
        sums: dict[str, float] = defaultdict(float)
        n_micro = 0
        grad_norms = []
        max_steps = self.cfg["train"].get("max_steps")
        torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        pending = 0
        for i, batch in enumerate(self.loader):
            pv = batch["pixel_values"].to(self.device, non_blocking=True)
            pm = batch["pixel_mask"].to(self.device, non_blocking=True)
            tg = to_device(batch["labels"], self.device)
            with torch.autocast("cuda", dtype=self.amp, enabled=self.amp is not None):
                out = self.model(pv, pm)
            loss, parts = self.criterion(out, tg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {self.epoch}, step {self.global_step}")
            (loss / self.accum).backward()
            pending += 1
            n_micro += 1
            sums["loss"] += loss.item()
            for k in ("loss_ce", "loss_bbox", "loss_giou", "cardinality_error"):
                sums[k] += parts[k].item()
            if pending == self.accum or i == len(self.loader) - 1:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["optim"]["grad_clip"])
                grad_norms.append(gn.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                pending = 0
                self.global_step += 1
                self.tb.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.tb.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                if max_steps and self.global_step >= max_steps:
                    break
        dt = time.time() - t0
        stats = {k: v / max(n_micro, 1) for k, v in sums.items()}
        stats.update({
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "lr": self.optimizer.param_groups[0]["lr"],
            "lr_backbone": self.optimizer.param_groups[1]["lr"],
            **({"lr_projection": self.optimizer.param_groups[2]["lr"]} if len(self.optimizer.param_groups) > 2 else {}),
            "epoch_time_s": dt,
            "images_per_s": n_micro * self.cfg["train"]["micro_batch"] / dt,
            "gpu_peak_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "global_step": self.global_step,
        })
        return stats

    def validate(self) -> dict[str, Any]:
        dets = predict(self.model, self.val_loader, self.device, self.val_ds.label_to_cat, self.amp)
        return evaluate_coco(self.val_coco, dets)

    def fit(self) -> None:
        tr = self.cfg["train"]
        metrics_path = self.run_dir / "metrics.csv"
        max_steps = tr.get("max_steps")
        while self.epoch < tr["epochs"]:
            stats = self.train_one_epoch()
            last_epoch = self.epoch == tr["epochs"] - 1 or bool(max_steps and self.global_step >= max_steps)
            row = {"epoch": self.epoch, **stats}
            if (self.epoch + 1) % tr["eval_every"] == 0 or last_epoch:
                t0 = time.time()
                val = self.validate()
                row.update({"val_AP": val["AP"], "val_AP50": val["AP50"], "val_AP75": val["AP75"],
                            "val_AR@300": val["AR@300"], "val_time_s": time.time() - t0})
                row.update({f"val_AP50/{k}": v["AP50"] for k, v in val["per_class"].items()})
                if val["AP"] > self.best_ap:
                    self.best_ap = val["AP"]
                    save_checkpoint(self.run_dir / "checkpoints" / "best.pt", model=self.model.state_dict(),
                                    epoch=self.epoch, val=val, class_names=self.class_names)
                    log.info("new best val AP %.4f at epoch %d", val["AP"], self.epoch)
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "epoch":
                    self.tb.add_scalar(f"epoch/{k}", v, self.epoch)
            append_csv(metrics_path, row)
            log.info("epoch %d: %s", self.epoch, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            self.epoch += 1
            save_checkpoint(self.run_dir / "checkpoints" / "last.pt", keep_prev=True, **self.state())
            if last_epoch:
                break
        self.tb.close()
