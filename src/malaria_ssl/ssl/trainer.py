"""SSL pretraining loop: AdamW + warmup/cosine, bf16 autocast, batch-size probe, resume, monitoring.

Run directory layout (runs/<name>/), same as DETR runs:
    config.yaml        resolved config (incl. the probed batch)
    run_info.json      git hash / dirty flag, package versions, GPU
    metrics.csv        one row per epoch: train losses, effective weights, lr, grad norm, throughput,
                       memory, val losses, collapse diagnostics, light-probe metrics
    tb/                TensorBoard logs
    recon/             MAE reconstruction grids on fixed val crops (original | masked | reconstruction);
                       target_epoch_XXX.png on fixed crops of the unlabeled target images (if `target`)
    checkpoints/       last.pt (full training state), encoder_epXXX.pt, encoder_final.pt (timm state dicts)

No gradient accumulation: it would shrink the set of in-batch negatives of InfoNCE.

`model.init` (random | imagenet | encoder state dict) selects the start point; the encoder's input
normalization is recorded as `model.input_norm` in config.yaml. `target` (folds file + fold + fraction)
adds the unlabeled images of one target fold: a draw comes from them with probability `fraction`.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD, seed_worker
from malaria_ssl.ssl.data import (MixedImageSampler, MixedTwoViewDataset, RandomImageSampler, TwoViewDataset,
                                   build_target_cache, ensure_caches)
from malaria_ssl.ssl.encoders import init_encoder, is_imagenet_norm
from malaria_ssl.ssl.models import SSLModel, info_nce, param_groups
from malaria_ssl.ssl.probe import Prober
from malaria_ssl.utils.config import load_config, resolve_path
from malaria_ssl.utils.run import append_csv, atomic_save, load_checkpoint, lr_lambda, run_info, save_checkpoint
from malaria_ssl.utils.seed import set_seed

log = logging.getLogger(__name__)

AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}
PROBE_METRICS = ("auroc_infected", "auprc_infected", "balanced_acc", "macro_f1")


# --------------------------------------------------------------------------- diagnostics

@torch.no_grad()
def collapse_stats(e1: torch.Tensor, e2: torch.Tensor) -> dict[str, float]:
    """Collapse diagnostics of paired embeddings (N, d).

    std: mean per-dimension std of the L2-normalised embeddings (collapse -> 0; ~1/sqrt(d) if spread);
    alignment: E||n1 - n2||^2 (Wang & Isola 2020); uniformity: log E exp(-2||ni - nj||^2);
    erank: effective rank, exp(entropy) of the normalised singular values (Roy & Vetterli 2007).
    """
    n1, n2 = F.normalize(e1.float(), dim=1), F.normalize(e2.float(), dim=1)
    s = torch.linalg.svdvals(n1 - n1.mean(0))
    p = s / s.sum()
    return {"std": n1.std(0).mean().item(), "std_x_sqrt_d": n1.std(0).mean().item() * math.sqrt(n1.shape[1]),
            "alignment": (n1 - n2).pow(2).sum(1).mean().item(),
            "uniformity": torch.pdist(n1).pow(2).mul(-2).exp().mean().log().item(),
            "erank": torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()}


def to_uint8(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Normalised (3, H, W) -> uint8 (H, W, 3) RGB."""
    img = x.float().cpu().numpy().transpose(1, 2, 0) * np.asarray(std) + np.asarray(mean)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def save_recon_grid(path: Path, original: torch.Tensor, masked: torch.Tensor, recon: torch.Tensor,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
    """Rows: original | masked | reconstruction; one column per crop."""
    rows = [np.concatenate([to_uint8(t, mean, std) for t in batch], axis=1) for batch in (original, masked, recon)]
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"cannot encode {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------- data

class FixedKeysSampler(Sampler):
    """Debug (overfit): cycles over a fixed list of (index, seed) keys, reshuffled per epoch."""

    def __init__(self, keys: list[tuple[int, int]], samples_per_epoch: int, seed: int):
        self.keys, self.samples, self.seed, self.epoch = keys, samples_per_epoch, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        reps = -(-self.samples // len(self.keys))
        order = np.concatenate([rng.permutation(len(self.keys)) for _ in range(reps)])[: self.samples]
        return iter(self.keys[i] for i in order)


def collect(ds: TwoViewDataset, keys: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise fixed pairs (stored as float16 to halve memory)."""
    pairs = [ds[k] for k in keys]
    return (torch.stack([p[0] for p in pairs]).half(), torch.stack([p[1] for p in pairs]).half())


# --------------------------------------------------------------------------- training

class SSLTrainer:
    def __init__(self, cfg: dict[str, Any], run_dir: Path, device: torch.device | None = None,
                 caches: dict[str, Path] | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg["seed"])
        self.caches = caches or ensure_caches(cfg, resolve_path)
        tr, mon = cfg["train"], cfg["monitor"]
        encoder, norm = init_encoder(cfg["model"], cfg["probe"]["references"]["imagenet"], resolve_path)
        cfg["model"]["input_norm"] = norm
        self.norm = (norm["mean"], norm["std"])
        self.model = SSLModel(encoder, cfg).to(self.device)
        self.amp = AMP_DTYPES[tr["amp_dtype"]] if self.device.type == "cuda" else None

        n_over = tr.get("overfit_crops")
        if n_over:
            self.train_ds = TwoViewDataset(self.caches["train"], cfg["views"], augment=False, input_norm=self.norm)
            self.overfit_keys = list(RandomImageSampler(len(self.train_ds), n_over, cfg["seed"]))
            mon_ds, mon_keys = self.train_ds, self.overfit_keys[: mon["val_pairs"]]
        else:
            self.train_ds = TwoViewDataset(self.caches["train"], cfg["views"], augment=True, input_norm=self.norm)
            self.overfit_keys = None
            mon_ds = TwoViewDataset(self.caches["val"], cfg["views"], augment=True, input_norm=self.norm)
            mon_keys = list(RandomImageSampler(len(mon_ds), mon["val_pairs"], mon["monitor_seed"]))
        self.mon_v1, self.mon_v2 = collect(mon_ds, mon_keys)
        self.target_ds, self.target_v1 = None, None
        if cfg.get("target") and not n_over:
            t = cfg["target"]
            npy = self.caches.get("target")
            if npy is None:
                raw_images = resolve_path(load_config(resolve_path(cfg["data_config"]))["paths"]["raw_root"]) / "images"
                npy = build_target_cache(resolve_path(t["folds_file"]), t["fold"], raw_images, cfg["cache"],
                                         resolve_path)[0]
            self.target_ds = TwoViewDataset(npy, cfg["views"], augment=True, input_norm=self.norm)
            self.train_ds = MixedTwoViewDataset([self.train_ds, self.target_ds])
            tkeys = list(RandomImageSampler(len(self.target_ds), mon["recon_crops"], mon["monitor_seed"]))
            self.target_v1 = collect(self.target_ds, tkeys)[0]
            log.info("target fold %d: %d unlabeled images, sampling fraction %.2f", t["fold"], len(self.target_ds),
                     t["fraction"])
        g = torch.Generator().manual_seed(mon["monitor_seed"])
        self.mon_noise = torch.rand(len(mon_keys), (cfg["views"]["out_size"] // self.model.patch) ** 2, generator=g)
        self.prober = None
        if mon["probe_every"]:
            self.prober = Prober(cfg, load_config(resolve_path(cfg["data_config"])),
                                 {s: self.caches[s] for s in ("train", "val")}, resolve_path, self.device)
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0           # next epoch to run
        self.global_step = 0     # optimizer steps done

    # -- setup ---------------------------------------------------------------------------
    def probe_batch(self) -> dict[str, Any]:
        """Largest candidate batch (pairs) whose fwd+bwd peak memory fits the budget."""
        tr = self.cfg["train"]
        size = self.cfg["views"]["out_size"]
        budget = tr["probe_memory_fraction"] * torch.cuda.get_device_properties(self.device).total_memory
        self.model.train()
        results, chosen = {}, None
        for b in sorted(tr["batch_candidates"], reverse=True):
            v1 = v2 = out = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            try:
                v1 = torch.randn(b, 3, size, size, device=self.device)
                v2 = torch.randn(b, 3, size, size, device=self.device)
                with torch.autocast("cuda", dtype=self.amp, enabled=self.amp is not None):
                    out = self.model(v1, v2)
                out["loss"].backward()
                peak = torch.cuda.max_memory_allocated(self.device)
                results[b] = round(peak / 2**30, 2)
                ok = peak <= budget
            except torch.OutOfMemoryError:
                results[b], ok = "OOM", False
            finally:
                self.model.zero_grad(set_to_none=True)
                del v1, v2, out
                torch.cuda.empty_cache()
            if ok:
                chosen = b
                break
        if chosen is None:
            raise RuntimeError(f"no batch candidate fits: {results}")
        log.info("batch probe (peak GiB per candidate, pairs): %s -> %d", results, chosen)
        return {"batch": chosen, "probe_peak_gib": results, "budget_gib": round(budget / 2**30, 2)}

    def setup(self, resume: bool) -> None:
        tr, o = self.cfg["train"], self.cfg["optim"]
        if tr["batch"] == "auto":
            probe = self.probe_batch()
            tr["batch"] = probe["batch"]
            self.cfg["batch_probe"] = probe
        b = tr["batch"]
        self.cfg["lr"] = o["blr"] * b / 256
        self.optimizer = torch.optim.AdamW(param_groups(self.model, o["weight_decay"]), lr=self.cfg["lr"],
                                           betas=tuple(o["betas"]))
        self.steps_per_epoch = tr["samples_per_epoch"] // b
        if self.steps_per_epoch == 0:
            raise ValueError("samples_per_epoch is smaller than the batch")
        total = self.steps_per_epoch * tr["epochs"]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda(max(1, round(o["warmup_fraction"] * total)), total, o["min_lr_ratio"]))
        if self.overfit_keys:
            self.sampler = FixedKeysSampler(self.overfit_keys, tr["samples_per_epoch"], self.cfg["seed"])
        elif self.target_ds is not None:
            n_src = len(self.train_ds) - len(self.target_ds)
            self.sampler = MixedImageSampler(n_src, len(self.target_ds), self.cfg["target"]["fraction"],
                                             tr["samples_per_epoch"], self.cfg["seed"])
        else:
            self.sampler = RandomImageSampler(len(self.train_ds), tr["samples_per_epoch"], self.cfg["seed"])
        nw = tr["num_workers"]
        self.loader = DataLoader(self.train_ds, batch_size=b, sampler=self.sampler, drop_last=True, num_workers=nw,
                                 worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(self.cfg["seed"]),
                                 persistent_workers=tr["persistent_workers"] and nw > 0,
                                 pin_memory=self.device.type == "cuda")
        if resume:
            self.load(self.run_dir / "checkpoints" / "last.pt")
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "run_info.json").write_text(json.dumps(run_info(), indent=2), encoding="utf-8")
            with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, sort_keys=False)
        log.info("batch %d pairs, lr %.2e, %d steps/epoch, %d epochs, %d samples total", b, self.cfg["lr"],
                 self.steps_per_epoch, tr["epochs"], self.steps_per_epoch * b * tr["epochs"])
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(str(self.run_dir / "tb"), purge_step=self.global_step if resume else None)

    # -- state ---------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "epoch": self.epoch, "global_step": self.global_step,
                "rng": rng}

    def load(self, path: Path) -> None:
        ck = load_checkpoint(path)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.epoch, self.global_step = ck["epoch"], ck["global_step"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        if "cuda" in ck["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        log.info("resumed from %s at epoch %d (step %d)", path, self.epoch, self.global_step)

    # -- loop ----------------------------------------------------------------------------
    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        self.sampler.set_epoch(self.epoch)
        sums: dict[str, float] = defaultdict(float)
        n, grad_norms = 0, []
        max_steps = self.cfg["train"].get("max_steps")
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        for v1, v2, _ in self.loader:
            v1 = v1.to(self.device, non_blocking=True)
            v2 = v2.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None):
                out = self.model(v1, v2)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {self.epoch}, step {self.global_step}")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["optim"]["grad_clip"])
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            n += 1
            grad_norms.append(gn.item())
            parts = {"loss": loss.item()}
            for k in ("loss_mae", "loss_con"):
                if out[k] is not None:
                    parts[k] = out[k].item()
            for k, v in parts.items():
                sums[k] += v
                self.tb.add_scalar(f"train/{k}_step", v, self.global_step)
            self.tb.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            if max_steps and self.global_step >= max_steps:
                break
        if cuda:
            torch.cuda.synchronize(self.device)
        dt = time.time() - t0
        b = self.cfg["train"]["batch"]
        stats = {k: v / max(n, 1) for k, v in sums.items()}
        stats.update(self.model.weights())
        stats.update({"lr": self.optimizer.param_groups[0]["lr"],
                      "grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                      "steps": n, "epoch_time_s": dt, "pairs_per_s": n * b / dt,
                      "gpu_peak_gib": torch.cuda.max_memory_allocated(self.device) / 2**30 if cuda else 0.0,
                      "global_step": self.global_step})
        return stats

    @torch.no_grad()
    def monitor(self, recon_path: Path | None) -> dict[str, float]:
        """Val losses and collapse diagnostics on the fixed monitoring pairs (+ optional recon grid)."""
        m = self.model
        m.eval()
        b = self.cfg["train"]["batch"]
        out: dict[str, float] = {}
        mae, con, h1s, h2s, z1s, z2s = [], [], [], [], [], []
        with torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None):
            for s in range(0, len(self.mon_v1), b):
                v1 = self.mon_v1[s:s + b].to(self.device).float()
                v2 = self.mon_v2[s:s + b].to(self.device).float()
                if m.use_mae:
                    mae.append((m.mae_forward(v1, self.mon_noise[s:s + b].to(self.device))["loss"].item(), len(v1)))
                h1, h2 = m.encode_full(v1).float(), m.encode_full(v2).float()
                h1s.append(h1), h2s.append(h2)
                if m.use_con:
                    z1, z2 = m.projector(h1), m.projector(h2)
                    z1s.append(z1.float()), z2s.append(z2.float())
                    con.append((info_nce(z1, z2, m.temperature).item(), len(v1)))
            if recon_path is not None and m.use_mae:
                k = self.cfg["monitor"]["recon_crops"]
                x = self.mon_v1[:k].to(self.device).float()
                masked, recon = m.reconstruct(x, self.mon_noise[:k].to(self.device))
                save_recon_grid(recon_path, x, masked, recon, *self.norm)
                if self.target_v1 is not None:
                    x = self.target_v1.to(self.device).float()
                    masked, recon = m.reconstruct(x, self.mon_noise[:len(x)].to(self.device))
                    save_recon_grid(recon_path.with_name("target_" + recon_path.name), x, masked, recon, *self.norm)
        if mae:
            out["val_loss_mae"] = sum(v * c for v, c in mae) / sum(c for _, c in mae)
        if con:
            out["val_loss_con"] = sum(v * c for v, c in con) / sum(c for _, c in con)
        out.update({f"rep/{k}": v for k, v in collapse_stats(torch.cat(h1s), torch.cat(h2s)).items()})
        if z1s:
            out.update({f"proj/{k}": v for k, v in collapse_stats(torch.cat(z1s), torch.cat(z2s)).items()})
        m.train()
        return out

    def light_probe(self) -> dict[str, float]:
        norm = None if is_imagenet_norm(*self.norm) else self.norm
        res = self.prober.evaluate(self.model.encoder, full=False, input_norm=norm)
        self.model.train()
        return {f"probe/{meth}/{k}": e["metrics"][k] for meth, e in res["methods"].items() for k in PROBE_METRICS}

    def save_encoder(self, name: str) -> None:
        save_checkpoint_plain(self.run_dir / "checkpoints" / name, self.model.encoder_state_dict())

    def fit(self) -> None:
        tr, mon = self.cfg["train"], self.cfg["monitor"]
        metrics_path = self.run_dir / "metrics.csv"
        max_steps = tr.get("max_steps")
        if self.epoch == 0 and self.global_step == 0 and self.model.use_mae:
            self.monitor(self.run_dir / "recon" / "init.png")
        while self.epoch < tr["epochs"]:
            stats = self.train_one_epoch()
            last = self.epoch == tr["epochs"] - 1 or bool(max_steps and self.global_step >= max_steps)
            e1 = self.epoch + 1
            recon = (self.run_dir / "recon" / f"epoch_{e1:03d}.png") if (e1 % mon["recon_every"] == 0 or last) else None
            t0 = time.time()
            row = {"epoch": self.epoch, **stats, **self.monitor(recon)}
            if self.prober is not None and (e1 % mon["probe_every"] == 0 or last):
                row.update(self.light_probe())
            row["monitor_time_s"] = time.time() - t0
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "epoch":
                    self.tb.add_scalar(f"epoch/{k}", v, self.epoch)
            append_csv(metrics_path, row)
            log.info("epoch %d: %s", self.epoch, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            self.epoch += 1
            if mon["save_every"] and e1 % mon["save_every"] == 0:
                self.save_encoder(f"encoder_ep{e1:03d}.pt")
            save_checkpoint(self.run_dir / "checkpoints" / "last.pt", keep_prev=True, **self.state())
            if last:
                break
        if self.epoch >= tr["epochs"] or (max_steps and self.global_step >= max_steps):
            self.save_encoder("encoder_final.pt")
        self.tb.close()


def save_checkpoint_plain(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Atomic save of a bare state dict (loads with torch.load(weights_only=True))."""
    atomic_save(state_dict, path)
