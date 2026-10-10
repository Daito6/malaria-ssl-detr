"""Plotting helpers for box checks and EDA (used by scripts and notebooks)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from PIL import Image

# Fixed per-class colours (colour-blind friendly where possible); RBC is muted on purpose.
CLASS_COLORS = {
    "red blood cell": "#b0b0b0",
    "leukocyte": "#1f77b4",
    "gametocyte": "#9467bd",
    "ring": "#2ca02c",
    "trophozoite": "#ff7f0e",
    "schizont": "#d62728",
    "difficult": "#e7d40a",
    "uninfected": "#b0b0b0",
    "infected": "#d62728",
}


def draw_image_with_boxes(ax, image_path: Path, anns: list[dict[str, Any]], cat_names: dict[int, str],
                          title: str | None = None, linewidth: float = 1.0) -> None:
    with Image.open(image_path) as im:
        ax.imshow(np.asarray(im.convert("RGB")))
    for a in anns:
        x, y, w, h = a["bbox"]
        name = cat_names[a["category_id"]]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=linewidth if name != "red blood cell" else 0.6,
                               edgecolor=CLASS_COLORS.get(name, "black")))
    present = sorted({cat_names[a["category_id"]] for a in anns})
    ax.legend(handles=[Patch(edgecolor=CLASS_COLORS.get(n, "black"), facecolor="none", label=n)
                       for n in present], loc="upper right", fontsize=7, framealpha=0.8)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_box_check(coco: dict[str, Any], image_id: int, images_dir: Path, out_path: Path,
                   title: str | None = None) -> None:
    """Save a single image with all its boxes drawn at full resolution."""
    im = next(i for i in coco["images"] if i["id"] == image_id)
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    dpi = 100
    fig = plt.figure(figsize=(im["width"] / dpi, im["height"] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_image_with_boxes(ax, Path(images_dir) / im["file_name"], anns, cat_names, linewidth=2.0)
    if title:
        ax.text(10, 30, title, fontsize=14, color="white", backgroundcolor="black")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, pil_kwargs={"quality": 90} if out_path.suffix == ".jpg" else None)
    plt.close(fig)


def plot_class_distribution(class_counts: pd.DataFrame, splits: list[str], log: bool = True):
    """Grouped bar chart of boxes per class per split (log scale by default)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(splits)
    x = np.arange(len(class_counts))
    for k, s in enumerate(splits):
        vals = class_counts[f"{s}_boxes"].to_numpy()
        bars = ax.bar(x + k * width, vals, width, label=s)
        ax.bar_label(bars, labels=[str(v) for v in vals], fontsize=7, padding=1)
    ax.set_xticks(x + width * (len(splits) - 1) / 2, class_counts.index, rotation=20)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("boxes" + (" (log)" if log else ""))
    ax.set_title("Boxes per class per split")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_boxes_per_image(images: pd.DataFrame, queries: int = 100):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, images["n_boxes"].max() + 6, 5)
    for s, sub in images.groupby("split"):
        ax.hist(sub["n_boxes"], bins=bins, alpha=0.6, label=s)
    ax.axvline(queries, color="red", ls="--", label=f"DETR default queries = {queries}")
    ax.set_xlabel("boxes per image")
    ax.set_ylabel("images")
    ax.set_title("Boxes per image")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_image_sizes(images: pd.DataFrame):
    sizes = images.assign(size=images["width"].astype(str) + "x" + images["height"].astype(str))
    tbl = sizes.groupby(["size", "split"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    tbl.plot.barh(stacked=True, ax=ax)
    ax.set_xlabel("images")
    ax.set_title("Image size (W x H) per split")
    fig.tight_layout()
    return fig


def plot_box_sizes(boxes: pd.DataFrame, patch_sizes: tuple[int, ...] = (8, 16, 32)):
    """Per-class boxplot of the longer box side in pixels, with ViT patch sizes marked."""
    order = [c for c in CLASS_COLORS if c in set(boxes["class"])]
    side = boxes[["w", "h"]].max(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([side[boxes["class"] == c] for c in order], tick_labels=order, showfliers=False)
    for p in patch_sizes:
        ax.axhline(p, ls=":", color="gray")
        ax.text(len(order) + 0.45, p, f"patch {p}px", va="center", fontsize=7, color="gray")
    ax.set_ylabel("max(w, h), px (original resolution)")
    ax.set_title("Box size per class (whiskers 1.5 IQR, outliers hidden)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_class_crops(coco: dict[str, Any], images_dir: Path, per_class: int = 6, seed: int = 0,
                     pad: int = 10):
    """Grid of random box crops for each class (rows = classes)."""
    rng = np.random.default_rng(seed)
    cats = coco["categories"]
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    fig, axes = plt.subplots(len(cats), per_class, figsize=(per_class * 1.6, len(cats) * 1.7))
    for r, c in enumerate(cats):
        anns = [a for a in coco["annotations"] if a["category_id"] == c["id"]]
        pick = rng.choice(len(anns), size=min(per_class, len(anns)), replace=False) if anns else []
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k >= len(pick):
                continue
            a = anns[pick[k]]
            x, y, w, h = (int(round(v)) for v in a["bbox"])
            with Image.open(Path(images_dir) / id2file[a["image_id"]]) as im:
                crop = im.convert("RGB").crop((max(x - pad, 0), max(y - pad, 0), x + w + pad, y + h + pad))
            ax.imshow(crop)
            if k == 0:
                ax.set_title(c["name"], fontsize=9, loc="left")
    fig.tight_layout()
    return fig
