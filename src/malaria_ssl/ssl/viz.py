"""Loading SSL run / probe artifacts and plotting for notebooks/03_ssl_analysis.ipynb (reads only)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from malaria_ssl.data.viz import CLASS_COLORS, SERIES, _recessive_axes
from malaria_ssl.ssl.probe import apply_stain_shift, build_stain_shift, select_run
from malaria_ssl.utils.config import load_experiment, resolve_path

HEADLINE = ("auroc_infected", "auprc_infected", "balanced_acc", "macro_f1")


# --------------------------------------------------------------------------- loading

def load_ssl_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    return {"name": run_dir.name, "dir": run_dir,
            "config": yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
            "history": pd.read_csv(run_dir / "metrics.csv"),
            # init.png first, then epoch_XXX.png in order (a plain sort puts init last)
            "recon": sorted((run_dir / "recon").glob("*.png"), key=lambda f: (f.stem != "init", f.stem))}


def find_ssl_runs(prefix: str = "ssl_") -> list[dict[str, Any]]:
    """All runs/<prefix>*/ with a metrics.csv, sorted by name."""
    return [load_ssl_run(d) for d in sorted(resolve_path("runs").glob(f"{prefix}*")) if (d / "metrics.csv").exists()]


def load_probes(tags: list[str] | None = None) -> list[dict[str, Any]]:
    """reports/tables/probe_<tag>.json (all present if tags is None)."""
    paths = (sorted(resolve_path("reports/tables").glob("probe_*.json")) if tags is None
             else [resolve_path("reports/tables") / f"probe_{t}.json" for t in tags])
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths if p.exists()]


# --------------------------------------------------------------------------- tables

def _fmt_ci(ci: dict[str, float] | None) -> str:
    return f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci else ""


def probe_table(probes: list[dict[str, Any]], methods: tuple[str, ...] = ("knn", "linear")) -> pd.DataFrame:
    """One row per (encoder, method): headline metrics with 95% bootstrap CIs (val crops, 100% labels)."""
    rows = []
    for p in probes:
        for m in methods:
            e = p["methods"][m]
            row = {"encoder": p["tag"], "method": m}
            for k in HEADLINE:
                row[k] = e["metrics"][k]
                row[f"{k}_CI95"] = _fmt_ci(e.get("ci95", {}).get(k))
            rows.append(row)
    return pd.DataFrame(rows)


def stain_table(probes: list[dict[str, Any]], methods: tuple[str, ...] = ("knn", "linear")) -> pd.DataFrame:
    """Clean vs stain-shifted val crops (same classifiers, fitted on clean train crops), with 95% CIs."""
    rows = []
    for p in probes:
        if "stain_shift" not in p:
            continue
        for m in methods:
            row = {"encoder": p["tag"], "method": m}
            for k in ("auprc_infected", "macro_f1"):
                clean, shifted = p["methods"][m], p["stain_shift"]["methods"][m]
                row[f"{k}_clean"] = clean["metrics"][k]
                row[f"{k}_clean_CI95"] = _fmt_ci(clean.get("ci95", {}).get(k))
                row[f"{k}_stain"] = shifted["metrics"][k]
                row[f"{k}_stain_CI95"] = _fmt_ci(shifted.get("ci95", {}).get(k))
                row[f"{k}_drop"] = clean["metrics"][k] - shifted["metrics"][k]
            rows.append(row)
    return pd.DataFrame(rows)


def selection_table(probes: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lambda-selection criteria per sweep run (priority order = column order) and `select_run` result."""
    sel = select_run({p["tag"]: p for p in probes})
    df = pd.DataFrame(sel["criteria"]).T
    df["winner"] = df.index == sel["winner"]
    return df, sel


def comparison_table(probes: list[dict[str, Any]]) -> pd.DataFrame:
    """Encoder comparison on val crops: the lambda-selection criteria plus kNN balanced accuracy,
    each with its 95% bootstrap CI (10%-label: mean ± std over seeds)."""
    rows = []
    for p in probes:
        lin, knn, stain = p["methods"]["linear"], p["methods"]["knn"], p["stain_shift"]["methods"]["linear"]
        ll = p["low_label"]["0.1"]["linear"]
        row = {"encoder": p["tag"]}
        for col, e, k in (("linear_auprc_clean", lin, "auprc_infected"),
                          ("linear_auprc_stain", stain, "auprc_infected")):
            row[col], row[f"{col}_CI95"] = e["metrics"][k], _fmt_ci(e["ci95"][k])
        row["linear_auprc_10pct_mean"] = ll["mean"]["auprc_infected"]
        row["linear_auprc_10pct_std"] = ll["std"]["auprc_infected"]
        for col, e, k in (("linear_macro_f1", lin, "macro_f1"), ("knn_balanced_acc", knn, "balanced_acc")):
            row[col], row[f"{col}_CI95"] = e["metrics"][k], _fmt_ci(e["ci95"][k])
        rows.append(row)
    return pd.DataFrame(rows)


def val_test_probe_table(tags: list[str], keys: tuple[str, ...] = ("auprc_infected", "auroc_infected",
                                                                     "macro_f1", "balanced_acc_binary")) -> pd.DataFrame:
    """Same classifiers (fit on train crops) on val crops (probe_<tag>.json) vs test crops (probe_test_<tag>.json),
    95% CIs over images of each split (independent splits: the CIs are not paired)."""
    rows = []
    for t in tags:
        val, test = load_probes([t, f"test_{t}"])
        for m in ("linear", "knn"):
            row = {"encoder": t, "method": m}
            for split, p in (("val", val), ("test", test)):
                for k in keys:
                    e = p["methods"][m]
                    row[f"{split}_{k}"] = e["metrics"][k]
                    row[f"{split}_{k}_lo"], row[f"{split}_{k}_hi"] = e["ci95"][k]["low"], e["ci95"][k]["high"]
            rows.append(row)
    return pd.DataFrame(rows)


def final_epoch_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Losses and collapse diagnostics (fixed val pairs) at the last logged epoch of each run."""
    cols = ["epoch", "loss_mae", "val_loss_mae", "loss_con", "val_loss_con", "rep/std_x_sqrt_d", "rep/erank",
            "proj/std_x_sqrt_d", "proj/erank"]
    return pd.DataFrame([{"run": r["name"], **{c: r["history"].iloc[-1].get(c, np.nan) for c in cols}}
                         for r in runs])


def per_class_recall_table(probes: list[dict[str, Any]], method: str = "linear") -> pd.DataFrame:
    rows = []
    for p in probes:
        e = p["methods"][method]
        for c in p["classes"]:
            k = f"recall/{c}"
            rows.append({"encoder": p["tag"], "class": c, "n_val": p["n_crops"]["val"][c],
                         "recall": e["metrics"][k], "CI95": _fmt_ci(e.get("ci95", {}).get(k))})
    return pd.DataFrame(rows)


def low_label_table(probes: list[dict[str, Any]], metric: str = "auprc_infected") -> pd.DataFrame:
    rows = []
    for p in probes:
        for m in ("knn", "linear"):
            rows.append({"encoder": p["tag"], "method": m, "fraction": 1.0, "mean": p["methods"][m]["metrics"][metric],
                         "std": 0.0})
            for frac, by_m in p.get("low_label", {}).items():
                rows.append({"encoder": p["tag"], "method": m, "fraction": float(frac),
                             "mean": by_m[m]["mean"][metric], "std": by_m[m]["std"][metric],
                             "n_train": by_m[m]["mean"]["n_train"]})
    return pd.DataFrame(rows).sort_values(["encoder", "method", "fraction"])


# --------------------------------------------------------------------------- curves

def _series(ax, runs, col, title, logy=False):
    for k, r in enumerate(runs):
        h = r["history"]
        if col in h and h[col].notna().any():
            ax.plot(h["epoch"], h[col], color=SERIES[k % len(SERIES)], lw=2, label=r["name"])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch")
    if logy:
        ax.set_yscale("log")
    _recessive_axes(ax)


def plot_losses(runs: list[dict[str, Any]]):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    for ax, (col, title) in zip(axes, [("loss_mae", "train L_mae"), ("val_loss_mae", "val L_mae (fixed masks)"),
                                       ("loss_con", "train L_con"), ("val_loss_con", "val L_con")]):
        _series(ax, runs, col, title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_weights_lr(runs: list[dict[str, Any]]):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.2))
    for ax, (col, title, logy) in zip(axes, [("w_mae", "effective weight MAE", False), ("w_con", "effective weight InfoNCE", False),
                                             ("lr", "learning rate (end of epoch)", True), ("grad_norm", "grad norm before clipping", False)]):
        _series(ax, runs, col, title, logy)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_collapse(runs: list[dict[str, Any]], space: str = "rep"):
    """Collapse diagnostics on fixed val pairs; space = rep (backbone mean-patch) or proj (projector)."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.2))
    for ax, (col, title) in zip(axes, [("std_x_sqrt_d", "std of L2-normed emb. x sqrt(d) (1 = spread)"),
                                       ("erank", "effective rank"), ("alignment", "alignment (lower = views closer)"),
                                       ("uniformity", "uniformity (lower = more uniform)")]):
        _series(ax, runs, f"{space}/{col}", f"{space}: {title}")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_light_probe(runs: list[dict[str, Any]], metric: str = "auprc_infected"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, m in zip(axes, ("knn", "linear")):
        for k, r in enumerate(runs):
            h = r["history"]
            col = f"probe/{m}/{metric}"
            if col in h:
                hh = h.dropna(subset=[col])
                ax.plot(hh["epoch"], hh[col], color=SERIES[k % len(SERIES)], lw=2, marker="o", ms=4, label=r["name"])
        ax.set_title(f"light probe ({m}): val {metric}", fontsize=10)
        ax.set_xlabel("epoch")
        _recessive_axes(ax)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def show_recon(run: dict[str, Any], which: tuple[int, ...] = (0, -1)):
    """Reconstruction grids (rows: original | masked | reconstruction) for selected files."""
    files = [run["recon"][i] for i in which if run["recon"]]
    fig, axes = plt.subplots(len(files), 1, figsize=(16, 2.4 * len(files)), squeeze=False)
    for ax, f in zip(axes[:, 0], files):
        ax.imshow(Image.open(f))
        ax.set_title(f"{run['name']}: {f.stem}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_low_label(probes: list[dict[str, Any]], metric: str = "auprc_infected"):
    df = low_label_table(probes, metric)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, m in zip(axes, ("knn", "linear")):
        for k, (enc, g) in enumerate(df[df["method"] == m].groupby("encoder", sort=False)):
            ax.errorbar(g["fraction"], g["mean"], yerr=g["std"], color=SERIES[k % len(SERIES)], lw=2, marker="o",
                        ms=4, capsize=3, label=enc)
        ax.set_xscale("log")
        ax.set_xlabel("fraction of labelled train crops")
        ax.set_title(f"{m}: val {metric} (mean ± std over seeds)", fontsize=10)
        _recessive_axes(ax)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def show_stain_shift(n: int = 6, config: str = "configs/ssl/probe.yaml", window: int = 224, seed: int = 0):
    """Clean (top) vs stain-shifted (bottom) 224 windows of random val images at the probe scale."""
    cfg = load_experiment(config)
    pcfg, sc = cfg["probe"], cfg["probe"]["stain_shift"]
    cache_dir = resolve_path(cfg["cache"]["dir"])
    arr = np.load(cache_dir / f"val_{cfg['cache']['long_side']}.npy", mmap_mode="r")
    index = json.loads((cache_dir / "val_index.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(arr), size=n, replace=False)
    t = build_stain_shift(sc)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for j, r in enumerate(rows):
        hc, wc = arr.shape[1:3]
        s_ = pcfg["long_side"] / max(hc, wc)
        img = cv2.resize(np.asarray(arr[r]), (round(wc * s_), round(hc * s_)), interpolation=cv2.INTER_AREA)
        shifted = apply_stain_shift(img, t, sc["seed"], index["images"][r]["id"])
        y0, x0 = rng.integers(0, img.shape[0] - window), rng.integers(0, img.shape[1] - window)
        for i, im in enumerate((img, shifted)):
            axes[i, j].imshow(im[y0:y0 + window, x0:x0 + window])
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
        axes[0, j].set_title(f"val image {index['images'][r]['id']}", fontsize=8)
    axes[0, 0].set_ylabel("clean"), axes[1, 0].set_ylabel("stain-shifted")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- PCA

def pca_2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Projection on the top-2 principal components (torch SVD) and their explained variance ratio."""
    xc = x.float() - x.float().mean(0)
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    return xc @ vh[:2].t(), var[:2] / var.sum()


def plot_pca(tags: list[str], split: str = "val", max_rbc: int = 600, seed: int = 0):
    """PCA scatter of probe features per encoder, coloured by class (RBC thinned for legibility)."""
    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4.4), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        d = torch.load(resolve_path("runs/probe") / tag / f"features_{split}.pt", weights_only=False)
        y = d["label"].numpy()
        xy, ratio = pca_2d(d["features"])
        xy = xy.numpy()
        rbc = d["classes"].index("red blood cell")
        rng = np.random.default_rng(seed)
        rbc_idx = np.nonzero(y == rbc)[0]
        keep = np.r_[rng.choice(rbc_idx, size=min(max_rbc, len(rbc_idx)), replace=False), np.nonzero(y != rbc)[0]]
        for c, name in enumerate(d["classes"]):
            sel = keep[y[keep] == c]
            ax.scatter(xy[sel, 0], xy[sel, 1], s=6 if c == rbc else 10, alpha=0.4 if c == rbc else 0.8,
                       color=CLASS_COLORS[name], label=name, linewidths=0)
        ax.set_title(f"{tag} ({split}); PC1+PC2 = {ratio.sum():.1%} var.", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        _recessive_axes(ax)
    axes[0, 0].legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    return fig
