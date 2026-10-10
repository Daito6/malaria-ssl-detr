"""Train/val split of the official train set: by image (or image group), multi-label stratified.

Iterative stratification (Sechidis et al., 2011) restricted to two subsets. The unit of
assignment is a group of images, so images from the same group never end up on both sides.
The filenames (UUIDs) and metadata carry no slide/source information, so groups are built only
from near-duplicate images detected with a difference hash.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def dhash(path: Path, size: int = 16) -> np.ndarray:
    """Difference hash: `size*size` booleans from horizontal gradients of a grayscale thumbnail."""
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    return (small[:, 1:] > small[:, :-1]).ravel()


def near_duplicate_pairs(images_dir: Path, file_names: list[str], max_hamming: int
                         ) -> list[tuple[str, str, int]]:
    """All pairs of images whose dHash Hamming distance is <= `max_hamming`."""
    hashes = np.stack([dhash(Path(images_dir) / n) for n in file_names])
    pairs = []
    for i in range(len(file_names) - 1):
        dist = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.nonzero(dist <= max_hamming)[0]:
            pairs.append((file_names[i], file_names[i + 1 + j], int(dist[j])))
    return pairs


def groups_from_pairs(items: list[Any], pairs: list[tuple[Any, Any]]) -> dict[Any, str]:
    """Union-find: item -> group key (connected components of the pair graph)."""
    parent = {x: x for x in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    return {x: str(find(x)) for x in items}


def image_label_sets(coco: dict[str, Any]) -> dict[int, set[int]]:
    """image_id -> set of category ids present in the image (empty set if no boxes)."""
    labels: dict[int, set[int]] = {im["id"]: set() for im in coco["images"]}
    for a in coco["annotations"]:
        labels[a["image_id"]].add(a["category_id"])
    return labels


def stratified_split(
    labels: dict[int, set[int]],
    val_fraction: float,
    seed: int,
    groups: dict[int, str] | None = None,
) -> tuple[list[int], list[int]]:
    """Split image ids into (train, val).

    Label counts are "number of images containing the class". Rarest classes are distributed
    first; each unit goes to the subset with the larger remaining demand for that class.
    """
    rng = random.Random(seed)
    groups = groups or {i: str(i) for i in labels}
    units: dict[str, list[int]] = defaultdict(list)
    for img_id in sorted(labels):
        units[groups[img_id]].append(img_id)
    unit_labels = {u: Counter(c for i in ids for c in labels[i]) for u, ids in units.items()}

    n_images = len(labels)
    totals = Counter(c for s in labels.values() for c in s)
    fractions = {"train": 1.0 - val_fraction, "val": val_fraction}
    demand = {s: {c: totals[c] * f for c in totals} for s, f in fractions.items()}
    capacity = {s: n_images * f for s, f in fractions.items()}
    assigned: dict[str, list[int]] = {"train": [], "val": []}

    remaining = list(units)
    rng.shuffle(remaining)

    def place(unit: str, subset: str) -> None:
        assigned[subset].extend(units[unit])
        capacity[subset] -= len(units[unit])
        for c, k in unit_labels[unit].items():
            demand[subset][c] -= k

    while remaining:
        # rarest class among still-unassigned units
        left = Counter()
        for u in remaining:
            left.update(unit_labels[u])
        if not left:
            break
        cls = min(left, key=lambda c: (left[c], c))
        with_cls = [u for u in remaining if cls in unit_labels[u]]
        for u in with_cls:
            best = max(("train", "val"),
                       key=lambda s: (demand[s][cls], capacity[s], rng.random()))
            place(u, best)
        taken = set(with_cls)
        remaining = [u for u in remaining if u not in taken]

    # units without any label: fill by capacity
    for u in remaining:
        place(u, max(("train", "val"), key=lambda s: (capacity[s], rng.random())))

    return sorted(assigned["train"]), sorted(assigned["val"])


def subset_coco(coco: dict[str, Any], image_ids: list[int]) -> dict[str, Any]:
    keep = set(image_ids)
    return {
        "images": [im for im in coco["images"] if im["id"] in keep],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        "categories": coco["categories"],
    }


def write_split_file(path: Path, file_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(file_names) + "\n", encoding="utf-8")


def read_split_file(path: Path) -> list[str]:
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln]


def split_class_counts(coco: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per class: number of images containing it and number of boxes."""
    name = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = Counter(name[c] for s in image_label_sets(coco).values() for c in s)
    boxes = Counter(name[a["category_id"]] for a in coco["annotations"])
    return {n: {"images": imgs[n], "boxes": boxes[n]} for n in name.values()}


def save_meta(path: Path, meta: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def make_splits(cfg: dict[str, Any], resolve) -> dict[str, Any]:
    """Carve val out of the official train set and write split files + train/val/test COCO.

    Near-duplicate images (dHash) are grouped so they never straddle train/val; near-duplicate
    pairs between the official train and test sets are only reported.
    """
    scfg = cfg["split"]
    coco_dir = resolve(cfg["paths"]["coco_dir"])
    splits_dir = resolve(cfg["paths"]["splits_dir"])
    images_dir = resolve(cfg["paths"]["raw_root"]) / "images"
    with (coco_dir / "train_full.json").open(encoding="utf-8") as f:
        train_full = json.load(f)
    with (coco_dir / "test.json").open(encoding="utf-8") as f:
        test = json.load(f)

    split_of = {im["file_name"]: "train" for im in train_full["images"]}
    split_of |= {im["file_name"]: "test" for im in test["images"]}
    names = sorted(split_of)
    pairs = near_duplicate_pairs(images_dir, names, scfg["near_duplicate_max_hamming"])
    train_pairs = [(a, b) for a, b, _ in pairs if split_of[a] == split_of[b] == "train"]
    cross_pairs = [p for p in pairs if split_of[p[0]] != split_of[p[1]]]

    name_to_id = {im["file_name"]: im["id"] for im in train_full["images"]}
    groups_by_name = groups_from_pairs(sorted(name_to_id), train_pairs)
    groups = {name_to_id[n]: g for n, g in groups_by_name.items()}
    train_ids, val_ids = stratified_split(image_label_sets(train_full), scfg["val_fraction"],
                                          cfg["seed"], groups)

    out = {"train": subset_coco(train_full, train_ids), "val": subset_coco(train_full, val_ids),
           "test": test}
    for s, coco in out.items():
        with (coco_dir / f"{s}.json").open("w", encoding="utf-8") as f:
            json.dump(coco, f)
        write_split_file(splits_dir / f"{s}.txt", sorted(im["file_name"] for im in coco["images"]))

    meta = {
        "seed": cfg["seed"],
        "val_fraction_target": scfg["val_fraction"],
        "n_images": {s: len(c["images"]) for s, c in out.items()},
        "val_fraction_actual": len(val_ids) / len(train_full["images"]),
        "near_duplicate_max_hamming": scfg["near_duplicate_max_hamming"],
        "near_duplicate_pairs": [{"a": a, "b": b, "hamming": d, "splits": [split_of[a], split_of[b]]}
                                 for a, b, d in pairs],
        "n_cross_train_test_pairs": len(cross_pairs),
        "n_multi_image_groups": sum(1 for g in set(groups.values())
                                    if sum(v == g for v in groups.values()) > 1),
        "class_counts": {s: split_class_counts(c) for s, c in out.items()},
    }
    save_meta(splits_dir / "split_meta.json", meta)
    return meta
