"""Plotting helpers for box checks and EDA (used by scripts and notebooks)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from PIL import Image

# Categorical slots in a fixed order (CVD-validated); RBC is a muted neutral on purpose.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLASS_COLORS = {
    "red blood cell": "#9a9a9a",
    "leukocyte": SERIES[0],
    "gametocyte": SERIES[1],
    "ring": SERIES[2],
    "trophozoite": SERIES[3],
    "schizont": SERIES[4],
    "difficult": SERIES[5],
    "uninfected": "#9a9a9a",
    "infected": SERIES[1],
}
SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}


def _recessive_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)


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
        bars = ax.bar(x + k * width, vals, width * 0.9, label=s, color=SPLIT_COLORS.get(s))
        ax.bar_label(bars, labels=[str(v) for v in vals], fontsize=7, padding=1)
    ax.set_xticks(x + width * (len(splits) - 1) / 2, class_counts.index, rotation=20)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("boxes" + (" (log)" if log else ""))
    ax.set_title("Boxes per class per split")
    _recessive_axes(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_boxes_per_image(images: pd.DataFrame, queries: int = 100):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, images["n_boxes"].max() + 6, 5)
    for s in [k for k in SPLIT_COLORS if k in set(images["split"])]:
        sub = images[images["split"] == s]
        ax.hist(sub["n_boxes"], bins=bins, histtype="step", lw=2, label=s, color=SPLIT_COLORS.get(s),
                density=True)
    ax.axvline(queries, color="#333333", ls="--", label=f"DETR default queries = {queries}")
    ax.set_xlabel("boxes per image")
    ax.set_ylabel("fraction of images (density)")
    ax.set_title("Boxes per image")
    _recessive_axes(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_image_sizes(images: pd.DataFrame):
    sizes = images.assign(size=images["width"].astype(str) + "x" + images["height"].astype(str))
    tbl = sizes.groupby(["size", "split"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    tbl.plot.barh(stacked=True, ax=ax, color=[SPLIT_COLORS.get(c) for c in tbl.columns])
    ax.set_xlabel("images")
    ax.set_title("Image size (W x H) per split")
    fig.tight_layout()
    return fig


def plot_box_sizes(boxes: pd.DataFrame, by: str = "class", patch_sizes: tuple[int, ...] = (8, 16, 32)):
    """Boxplot of the longer box side in pixels grouped by `by` ("class" or "split")."""
    present = set(boxes[by])
    order = [k for k in (CLASS_COLORS if by == "class" else SPLIT_COLORS) if k in present]
    side = boxes[["w", "h"]].max(axis=1)
    fig, ax = plt.subplots(figsize=(9 if by == "class" else 6, 4))
    ax.boxplot([side[boxes[by] == k] for k in order], tick_labels=order, showfliers=False)
    for p in patch_sizes:
        ax.axhline(p, ls=":", color="#9a9a9a")
        ax.text(len(order) + 0.45, p, f"patch {p}px", va="center", fontsize=7, color="#555555")
    ax.set_ylabel("max(w, h), px (original resolution)")
    ax.set_title(f"Box size per {by} (whiskers 1.5 IQR, outliers hidden)")
    ax.tick_params(axis="x", rotation=20)
    _recessive_axes(ax)
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
