"""Dataset statistics over COCO split files -> JSON + CSV tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
DETR_DEFAULT_QUERIES = 100


def load_coco(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def boxes_frame(coco: dict[str, Any], split: str) -> pd.DataFrame:
    """One row per box: split, image_id, class, x, y, w, h."""
    name = {c["id"]: c["name"] for c in coco["categories"]}
    rows = [(split, a["image_id"], name[a["category_id"]], *a["bbox"]) for a in coco["annotations"]]
    return pd.DataFrame(rows, columns=["split", "image_id", "class", "x", "y", "w", "h"])


def images_frame(coco: dict[str, Any], split: str) -> pd.DataFrame:
    """One row per image: split, image_id, file_name, width, height, n_boxes."""
    df = pd.DataFrame(coco["images"])[["id", "file_name", "width", "height"]]
    df = df.rename(columns={"id": "image_id"})
    counts = pd.Series([a["image_id"] for a in coco["annotations"]]).value_counts()
    df["n_boxes"] = df["image_id"].map(counts).fillna(0).astype(int)
    df.insert(0, "split", split)
    return df


def _quantiles(s: pd.Series) -> dict[str, float]:
    return {f"q{int(q * 100):02d}": float(s.quantile(q)) for q in QUANTILES}


def compute_stats(cocos: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Compute all statistics for the given {split: coco} dict.

    Returns (summary JSON dict, {table_name: DataFrame}).
    """
    class_names = [c["name"] for c in next(iter(cocos.values()))["categories"]]
    boxes = pd.concat([boxes_frame(c, s) for s, c in cocos.items()], ignore_index=True)
    images = pd.concat([images_frame(c, s) for s, c in cocos.items()], ignore_index=True)
    boxes["side_max"] = boxes[["w", "h"]].max(axis=1)
    boxes["side_min"] = boxes[["w", "h"]].min(axis=1)
    boxes["sqrt_area"] = np.sqrt(boxes["w"] * boxes["h"])
    boxes["aspect"] = boxes["w"] / boxes["h"]

    # per class per split: boxes and images containing the class
    n_boxes = boxes.pivot_table(index="class", columns="split", values="x", aggfunc="count", fill_value=0)
    n_imgs = (boxes.groupby(["class", "split"])["image_id"].nunique()
              .unstack(fill_value=0))
    class_tbl = (n_boxes.add_suffix("_boxes").join(n_imgs.add_suffix("_images"))
                 .reindex(class_names).fillna(0).astype(int))
    class_tbl["total_boxes"] = class_tbl[[c for c in class_tbl if c.endswith("_boxes")]].sum(axis=1)
    class_tbl["share_of_boxes"] = class_tbl["total_boxes"] / class_tbl["total_boxes"].sum()
    majority = class_tbl["total_boxes"].max()
    class_tbl["imbalance_vs_majority"] = majority / class_tbl["total_boxes"].replace(0, np.nan)
    class_tbl.index.name = "class"

    # boxes per image
    bpi = images.groupby("split")["n_boxes"].agg(["count", "mean", "median", "std", "min", "max"])
    bpi["n_images_over_100"] = images.groupby("split")["n_boxes"].apply(
        lambda s: int((s > DETR_DEFAULT_QUERIES).sum()))
    bpi.loc["all"] = [len(images), images["n_boxes"].mean(), images["n_boxes"].median(),
                      images["n_boxes"].std(), images["n_boxes"].min(), images["n_boxes"].max(),
                      int((images["n_boxes"] > DETR_DEFAULT_QUERIES).sum())]
    bpi.index.name = "split"

    # image sizes
    sizes = (images.assign(size=images["width"].astype(str) + "x" + images["height"].astype(str))
             .groupby(["size", "split"]).size().unstack(fill_value=0))
    sizes["total"] = sizes.sum(axis=1)
    sizes = sizes.sort_values("total", ascending=False)

    # box sizes in pixels, per class (all splits)
    box_size_rows = []
    for cls in class_names + ["ALL"]:
        sub = boxes if cls == "ALL" else boxes[boxes["class"] == cls]
        if sub.empty:
            continue
        for col in ("w", "h", "sqrt_area", "side_max", "side_min"):
            box_size_rows.append({"class": cls, "measure": col, "n": len(sub), "mean": float(sub[col].mean()),
                                  **_quantiles(sub[col])})
    box_sizes = pd.DataFrame(box_size_rows)

    worst = images.loc[images["n_boxes"].idxmax()]
    summary = {
        "splits": {s: {"images": int((images["split"] == s).sum()),
                       "boxes": int((boxes["split"] == s).sum())} for s in cocos},
        "boxes_per_image": {s: {k: float(v) for k, v in row.items()} for s, row in bpi.iterrows()},
        "max_boxes_image": {"split": worst["split"], "file_name": worst["file_name"],
                            "n_boxes": int(worst["n_boxes"])},
        "detr_default_queries": DETR_DEFAULT_QUERIES,
        "image_sizes": {sz: int(n) for sz, n in sizes["total"].items()},
        "class_share_of_boxes": {c: float(v) for c, v in class_tbl["share_of_boxes"].items()},
        "imbalance_vs_majority": {c: (None if pd.isna(v) else float(v))
                                  for c, v in class_tbl["imbalance_vs_majority"].items()},
        "box_side_px_all": _quantiles(boxes["side_max"]),
        "box_sqrt_area_px_all": _quantiles(boxes["sqrt_area"]),
    }
    tables = {"class_counts": class_tbl, "boxes_per_image": bpi, "image_sizes": sizes,
              "box_sizes": box_sizes}
    return summary, tables


def save_stats(summary: dict[str, Any], tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=name != "box_sizes", float_format="%.4f")
