"""Linear crop classifier of the two-stage scheme: refit from the saved TRAIN features, verified, cached.

scripts/two_stage.py evaluate saved the crop features (runs/two_stage/<tag>/train.pt) but not the fitted classifier.
This module refits the same model (the computation of malaria_ssl.ssl.probe.linear_predict: class-balanced,
L2-regularised logistic regression with L-BFGS and the probe's fixed hyperparameters, binary infected vs uninfected,
on clean or clean + stain-shifted train crops) and keeps its parameters in app/cache/<model id>.pt.

Before the cache is written, three checks must pass (results go to app/cache/<model id>.json):
1. `p_max_abs_diff`: the cached parameters reproduce linear_predict itself on the saved val candidate features;
2. `val_ap50` / `val_threshold`: on the saved val candidates and features, infected AP50 and the max-F1 threshold
   equal the values the evaluation recorded (reports/tables/<selection json>);
3. `feature_cos_min`: crop features computed by the app pipeline (this device, fp32 on CPU) from the raw val images
   match the saved features (GPU, bf16) on the candidates of the first `N_FEATURE_IMAGES` val images.
The reference metrics of the scheme stay those in reports/tables; the app does not recompute or display new ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from app.registry import ModelEntry, ModelUnavailable
from malaria_ssl.data.detection import load_split_coco, read_rgb
from malaria_ssl.eval.detection_metrics import best_f1_threshold, match_per_class
from malaria_ssl.eval.paired import prepare_run, score
from malaria_ssl.eval.two_stage import scored_detections
from malaria_ssl.ssl.probe import linear_predict
from malaria_ssl.utils.config import load_config, resolve_path

if TYPE_CHECKING:
    from app.inference import TwoStagePredictor

log = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).with_name("cache")
CACHE_FORMAT = 1
N_FEATURE_IMAGES = 3
TOL_P = 1e-5  # check 1: same computation, float noise only
TOL_AP = 1e-6  # check 2: same features, same classifier -> same AP50 / threshold
MIN_FEATURE_COS = 0.99  # check 3: CPU fp32 vs GPU bf16 features of the same crops


@torch.no_grad()
def _standardize(train_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)


def fit_linear(
    train_x: torch.Tensor, train_y: torch.Tensor, n_classes: int, l2: float, max_iter: int
) -> dict[str, torch.Tensor]:
    """The parameters linear_predict fits internally (same operations, same order)."""
    mean, std = _standardize(train_x)
    xtr = (train_x - mean) / std
    counts = torch.bincount(train_y, minlength=n_classes).float()
    cw = torch.where(counts > 0, len(train_y) / (n_classes * counts.clamp_min(1)), torch.zeros_like(counts))
    lin = torch.nn.Linear(xtr.shape[1], n_classes)
    torch.nn.init.zeros_(lin.weight), torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(xtr), train_y, weight=cw) + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return {"mean": mean, "std": std, "weight": lin.weight.detach().clone(), "bias": lin.bias.detach().clone()}


@torch.no_grad()
def predict_infected(clf: dict[str, Any], feats: torch.Tensor) -> np.ndarray:
    """p(infected) for (N, D) features."""
    if len(feats) == 0:
        return np.zeros(0, np.float32)
    x = (feats.float() - clf["mean"]) / clf["std"]
    return (x @ clf["weight"].t() + clf["bias"]).softmax(1)[:, 1].numpy()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _training_set(train: dict[str, Any], training: str) -> tuple[torch.Tensor, torch.Tensor]:
    """As scripts/two_stage.py evaluate: clean crops, or clean + stain-shifted crops."""
    if training == "clean":
        return train["clean"].float(), train["infected"].long()
    if training == "stain":
        return (
            torch.cat([train["clean"], train["stain"]]).float(),
            torch.cat([train["infected"], train["infected"]]).long(),
        )
    raise ValueError(f"training must be clean or stain, got {training!r}")


def _verify(
    entry: ModelEntry, predictor: TwoStagePredictor, clf: dict[str, Any], x: torch.Tensor, y: torch.Tensor
) -> dict[str, Any]:
    pcfg = predictor.pcfg
    cands = json.loads(entry.files["reference_candidates"].read_text(encoding="utf-8"))["candidates"]
    ref_feats = torch.load(entry.files["reference_features"], map_location="cpu", weights_only=False)["features"]
    if len(ref_feats) != len(cands):
        raise ModelUnavailable(f"{entry.id}: {len(ref_feats)} reference features != {len(cands)} candidates")

    # 1. parameters == linear_predict
    p_ref = linear_predict(x, y, ref_feats.float(), 2, pcfg["linear"]["l2"], pcfg["linear"]["max_iter"])[:, 1].numpy()
    p_app = predict_infected(clf, ref_feats)
    p_diff = float(np.abs(p_ref - p_app).max())

    # 2. val AP50 and threshold as recorded by the evaluation
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gt = load_split_coco("val", "binary", dcfg, resolve_path)
    cat = {c["name"]: c["id"] for c in gt["categories"]}["infected"]
    dets = scored_detections(cands, p_app, entry.spec["variant"], cat)
    thr = best_f1_threshold(match_per_class(gt, dets, 0.5)[cat])[0]
    ap = score(prepare_run(gt, dets, cat, thr), [im["id"] for im in gt["images"]])["AP50"]
    sel = json.loads(entry.files["selection"].read_text(encoding="utf-8"))["selection"]
    row = next(
        r
        for r in sel
        if r["run"] == entry.spec["reference"]["selection"]["run"]
        and r["encoder"] == entry.spec["reference"]["selection"]["encoder"]
        and r["training"] == entry.spec["training"]
    )
    ap_ref = row[f"val_AP50_{entry.spec['variant']}"]

    # 3. app crop features from the raw val images == saved features
    images = {im["id"]: im for im in gt["images"]}
    raw = resolve_path(dcfg["paths"]["raw_root"]) / "images"
    cos = []
    for img_id in sorted({c["image_id"] for c in cands})[:N_FEATURE_IMAGES]:
        idx = [i for i, c in enumerate(cands) if c["image_id"] == img_id]
        f = predictor.features(read_rgb(raw / images[img_id]["file_name"]), [cands[i] for i in idx])
        cos.append(F.cosine_similarity(f, ref_feats[idx].float(), dim=1))
    cos = torch.cat(cos)
    checks = {
        "p_max_abs_diff": p_diff,
        "val_ap50": ap,
        "val_ap50_reference": ap_ref,
        "val_threshold": thr,
        "val_threshold_reference": row["threshold"],
        "feature_cos_min": float(cos.min()),
        "feature_cos_mean": float(cos.mean()),
        "feature_n_crops": int(len(cos)),
        "feature_n_images": N_FEATURE_IMAGES,
        "feature_device": str(predictor.device),
    }
    failed = []
    if p_diff > TOL_P:
        failed.append(f"p(infected) differs from linear_predict by {p_diff:.2e} > {TOL_P}")
    if abs(ap - ap_ref) > TOL_AP or abs(thr - row["threshold"]) > TOL_AP:
        failed.append(f"val AP50 {ap:.6f} / threshold {thr:.6f} != recorded {ap_ref:.6f} / {row['threshold']:.6f}")
    if checks["feature_cos_min"] < MIN_FEATURE_COS:
        failed.append(
            f"app crop features differ from the saved ones: min cosine {checks['feature_cos_min']:.4f} "
            f"< {MIN_FEATURE_COS}"
        )
    if failed:
        raise ModelUnavailable(f"{entry.id}: classifier verification failed: " + "; ".join(failed))
    return checks


def load_or_build(entry: ModelEntry, predictor: TwoStagePredictor, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    """Cached classifier for `entry`; refit + verify + cache when missing or when its inputs changed."""
    pcfg = predictor.pcfg
    key = {
        "format": CACHE_FORMAT,
        "train_features_sha256": _sha(entry.files["train_features"]),
        "training": entry.spec["training"],
        "variant": entry.spec["variant"],
        "l2": pcfg["linear"]["l2"],
        "max_iter": pcfg["linear"]["max_iter"],
    }
    path = Path(cache_dir) / f"{entry.id}.pt"
    if path.exists():
        clf = torch.load(path, map_location="cpu", weights_only=False)
        if clf["meta"]["key"] == key:
            return clf
        log.info("%s: classifier inputs changed, refitting", entry.id)
    train = torch.load(entry.files["train_features"], map_location="cpu", weights_only=False)
    x, y = _training_set(train, entry.spec["training"])
    log.info("%s: fitting the linear classifier on %d train crops (%s)", entry.id, len(y), entry.spec["training"])
    params = fit_linear(x, y, 2, pcfg["linear"]["l2"], pcfg["linear"]["max_iter"])
    checks = _verify(entry, predictor, params, x, y)
    meta = {
        "key": key,
        "model_id": entry.id,
        "n_train": int(len(y)),
        "checks": checks,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "cache_file": f"app/cache/{path.name}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({**params, "meta": meta}, tmp)
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("%s: classifier cached (%s)", entry.id, checks)
    return {**params, "meta": meta}
