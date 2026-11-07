"""EXPLORATORY robustness curve on VAL (never test): infected AP50 / recall vs corruption strength, paired bootstrap.

Protocol fixed in reports/robustness_plan.md before any run. Corruptions are app/corruptions.py, applied at the model
input (after the eval resize, where evaluate.py --stain-shift applies its shift); inference is app/inference.py
DetrPredictor (parity-tested against evaluate.py), every query kept (no NMS, as evaluate.py), fp32 on CPU by default.
Recall / precision / F1 use each run's clean-val thresholds (thresholds.json) unchanged in every condition.

Two steps, so the grid can be run in parts and resumed (predictions are cached per model and condition):
    python scripts/robustness_curve.py predict --corruptions stain blur jpeg               # A, ImageNet, joint
    python scripts/robustness_curve.py predict --corruptions brightness_contrast scale_down scale_up stain_full
    python scripts/robustness_curve.py aggregate          # -> reports/tables/robustness_val{.csv,_diff.csv,.json}
`predict` always includes the clean condition (strength 0, shared by every corruption).
"""

# ruff: noqa: E402  (imports follow the CUDA_VISIBLE_DEVICES / sys.path setup)
from __future__ import annotations

import os
import sys

if "cuda" not in sys.argv:  # CPU by default: hide the GPU before torch is imported ('' would unset the variable)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the `app` package

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app import corruptions as C
from app.inference import DetrPredictor
from app.registry import load_registry
from malaria_ssl.data.detection import load_split_coco, read_rgb
from malaria_ssl.eval.paired import (
    METRICS,
    bootstrap_table,
    ci,
    prepare_run,
    resamples,
    score,
    summarize_diff,
)
from malaria_ssl.utils.config import load_config, resolve_path

log = logging.getLogger("robustness_curve")

# ---- protocol (reports/robustness_plan.md) ----
MODELS = ("a_coco_detr_binary", "b_vit_imagenet", "b_vit_joint")
GRID = (0.25, 0.5, 0.75, 1.0)  # + the shared clean point 0
CONTROL = (("stain_full", 1.0),)
PAIRS = (
    ("b_vit_imagenet", "a_coco_detr_binary"),
    ("b_vit_joint", "a_coco_detr_binary"),
    ("b_vit_joint", "b_vit_imagenet"),
)
N_RESAMPLES, SEED = 1000, 42
CLS = "infected"
CACHE_DIR = "app/cache/robustness"
OUT = "reports/tables/robustness_val"
CLEAN = ("none", 0.0)


def conditions(kinds: list[str]) -> list[tuple[str, float]]:
    out = [CLEAN]
    for k in kinds:
        out += [c for c in CONTROL if c[0] == k] if k in C.CONTROL_KINDS else [(k, s) for s in GRID]
    return out


def cond_name(kind: str, s: float) -> str:
    return f"{kind}_{s:.2f}"


def cond_file(model_id: str, kind: str, s: float, ext: str) -> Path:
    """Cached prediction file of one condition (not Path.with_suffix: the name contains a dot, e.g. blur_0.25)."""
    return resolve_path(CACHE_DIR) / model_id / f"{cond_name(kind, s)}{ext}"


def val_split():
    dcfg = load_config(resolve_path("configs/data.yaml"))
    gt = load_split_coco("val", "binary", dcfg, resolve_path)
    return gt, resolve_path(dcfg["paths"]["raw_root"]) / "images"


def cmd_predict(args) -> None:
    if args.threads:
        torch.set_num_threads(args.threads)
    log.info("torch threads: %d", torch.get_num_threads())
    gt, raw = val_split()
    images = sorted(gt["images"], key=lambda im: im["id"])
    registry = load_registry()
    device = torch.device(args.device)
    pixels = {im["id"]: read_rgb(raw / im["file_name"]) for im in images}
    log.info("val images: %d (test is never read)", len(pixels))
    for model_id in args.models:
        out_dir = resolve_path(CACHE_DIR) / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [c for c in conditions(args.corruptions) if not (out_dir / f"{cond_name(*c)}.npz").exists()]
        if not todo:
            log.info("%s: all requested conditions cached", model_id)
            continue
        pred = DetrPredictor(registry[model_id], device)
        names = pred.class_names
        for kind, s in todo:
            t0 = time.perf_counter()
            ls = pred.eval_long_side if kind == "none" else C.long_side(kind, s, pred.eval_long_side)
            rows = []
            for i, im in enumerate(images):
                corrupt = None if kind == "none" else (lambda x, k=kind, st=s, iid=im["id"]: C.apply(x, k, st, iid))
                for d in pred.detections(pixels[im["id"]], ls, corrupt):
                    rows.append((im["id"], names.index(d["label"]), d["score"], *d["bbox"]))
                if (i + 1) % 30 == 0:
                    el = time.perf_counter() - t0
                    log.info(
                        "%s %s: %d/%d (%.2f s/image)",
                        model_id,
                        cond_name(kind, s),
                        i + 1,
                        len(images),
                        el / (i + 1),
                    )
            a = np.asarray(rows, dtype=np.float64)
            tmp = out_dir / f"{cond_name(kind, s)}.tmp.npz"
            np.savez_compressed(
                tmp,
                image_id=a[:, 0].astype(np.int64),
                label=a[:, 1].astype(np.int8),
                score=a[:, 2].astype(np.float32),
                bbox=a[:, 3:7].astype(np.float32),
            )
            tmp.replace(out_dir / f"{cond_name(kind, s)}.npz")
            meta = {
                "model_id": model_id,
                "corruption": kind,
                "strength": s,
                "long_side": ls,
                "params": None if kind == "none" else C.describe(kind, s, pred.eval_long_side),
                "class_names": names,
                "checkpoint": pred.version,
                "device": str(device),
                "seconds": round(time.perf_counter() - t0, 1),
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            (out_dir / f"{cond_name(kind, s)}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            log.info("%s %s done in %.0f s", model_id, cond_name(kind, s), meta["seconds"])


def load_dets(path: Path, names: list[str], cat_of: dict[str, int]) -> list[dict]:
    z = np.load(path)
    return [
        {
            "image_id": int(i),
            "category_id": cat_of[names[int(lab)]],
            "score": float(sc),
            "bbox": b.tolist(),
        }
        for i, lab, sc, b in zip(z["image_id"], z["label"], z["score"], z["bbox"], strict=True)
    ]


def cmd_aggregate(args) -> None:
    gt, _ = val_split()
    cat_of = {c["name"]: c["id"] for c in gt["categories"]}
    img_ids = [im["id"] for im in gt["images"]]
    samples = resamples(img_ids, args.n, args.seed)
    planned = [(m, c) for m in MODELS for c in conditions(list(C.KINDS + C.CONTROL_KINDS))]
    runs, metas, missing = {}, {}, []
    for model_id, (kind, s) in planned:
        npz = cond_file(model_id, kind, s, ".npz")
        if not npz.exists():
            missing.append(f"{model_id}/{cond_name(kind, s)}")
            continue
        meta = json.loads(cond_file(model_id, kind, s, ".json").read_text(encoding="utf-8"))
        thr = json.loads(resolve_path(f"runs/{model_id}/thresholds.json").read_text(encoding="utf-8"))[CLS]
        dets = load_dets(npz, meta["class_names"], cat_of)
        runs[(model_id, kind, s)] = prepare_run(gt, dets, cat_of[CLS], thr)
        metas[(model_id, kind, s)] = {**meta, "threshold": thr}
        log.info("prepared %s %s", model_id, cond_name(kind, s))
    if not runs:
        raise SystemExit("no cached predictions: run `predict` first")
    boots = bootstrap_table(runs, samples)
    points = {k: score(r, img_ids) for k, r in runs.items()}

    rows = []
    for key in runs:
        model_id, kind, s = key
        m = metas[key]
        row = {
            "model": model_id,
            "corruption": kind,
            "strength": s,
            "long_side": m["long_side"],
            "params": json.dumps(m["params"]),
            "threshold": m["threshold"],
            "n_images": len(img_ids),
        }
        for j, met in enumerate(METRICS):
            row[met] = points[key][met]
            row[f"{met}_lo"], row[f"{met}_hi"] = ci(boots[key][:, j])
        rows.append(row)
    table = pd.DataFrame(rows)

    diffs = []

    def add(a, b, comparison):
        for j, met in enumerate(METRICS):
            diffs.append(
                {
                    "comparison": comparison,
                    "a_model": a[0],
                    "b_model": b[0],
                    "corruption": a[1],
                    "strength": a[2],
                    "metric": f"val_{met}_{CLS}",
                    **summarize_diff(points[a][met], points[b][met], boots[a][:, j], boots[b][:, j]),
                }
            )

    for key in runs:  # (i) condition - clean, same model
        clean = (key[0], *CLEAN)
        if key[1:] != CLEAN and clean in runs:
            add(key, clean, "condition - clean (same model)")
    for a_m, b_m in PAIRS:  # (ii) model - model, same condition
        for _, kind, s in [k for k in runs if k[0] == a_m]:
            if (b_m, kind, s) in runs:
                add((a_m, kind, s), (b_m, kind, s), f"{a_m} - {b_m} (same condition)")
    diff = pd.DataFrame(diffs)

    reference = {}
    for model_id in MODELS:  # evaluate.py outputs (CUDA bf16): context only, never replaced by the CPU values
        ref = {}
        for name, f in (
            ("clean_gpu", "metrics_val.json"),
            ("stain_full_gpu", "metrics_val_stain.json"),
        ):
            p = resolve_path(f"runs/{model_id}/{f}")
            if p.exists():
                mm = json.loads(p.read_text(encoding="utf-8"))
                ref[name] = {
                    "AP50": mm["coco"]["per_class"][CLS]["AP50"],
                    "recall": mm["prf"]["per_class"][CLS]["recall"],
                }
        reference[model_id] = ref

    out = resolve_path(args.out)
    table.to_csv(f"{out}.csv", index=False, float_format="%.4f")
    diff.to_csv(f"{out}_diff.csv", index=False, float_format="%.4f")
    protocol = {
        "status": "EXPLORATORY",
        "split": "val",
        "n_images": len(img_ids),
        "class": CLS,
        "seeds": 1,
        "models": list(MODELS),
        "corruptions": list(C.KINDS),
        "control": [list(c) for c in CONTROL],
        "grid": [0.0, *GRID],
        "n_resamples": args.n,
        "bootstrap_seed": args.seed,
        "metrics": "infected AP50 (threshold-free); precision / recall / F1 at the clean-val thresholds",
        "inference": "app.inference.DetrPredictor, all queries, no NMS, fp32",
        "device": sorted({m["device"] for m in metas.values()}),
        "plan": "reports/robustness_plan.md",
        "missing_conditions": missing,
    }
    (out.parent / f"{out.name}.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "reference_evaluate_py": reference,
                "rows": table.to_dict("records"),
                "diffs": diffs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 300):
        print(
            table[
                [
                    "model",
                    "corruption",
                    "strength",
                    "AP50",
                    "AP50_lo",
                    "AP50_hi",
                    "recall",
                    "recall_lo",
                    "recall_hi",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )
    if missing:
        print(f"missing conditions ({len(missing)}): {missing}")
    print(f"wrote {out}.csv, {out}_diff.csv, {out}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument(
        "--corruptions",
        nargs="+",
        required=True,
        choices=list(C.KINDS + C.CONTROL_KINDS),
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU threads (default: torch's choice)",
    )
    a = sub.add_parser("aggregate")
    a.add_argument("--n", type=int, default=N_RESAMPLES)
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--out", default=OUT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    {"predict": cmd_predict, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
