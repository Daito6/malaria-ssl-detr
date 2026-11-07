"""NIH-NLM thin blood smears (P. falciparum) as an external TARGET domain: zero-shot evaluation only.

Images: NIH LHNCBC `NIH-NLM-ThinBloodSmearsPf.zip` (Kassim et al., IEEE JBHI 2020; NLM data licence).
Boxes: COCO files of Wilm et al. (arXiv 2507.18483, Zenodo 17514694; Apache 2.0), polygon set (165 images) and
the revised point set (800 images, extra class `ambiguous`). Patients: the folder of each image, read from the NIH
`Dataset_statistics.xlsx` (193 patients x 5 images; image file names are capture timestamps, not patient ids).

Binary labels as BBBC041 (`configs/data.yaml` binary ids: 1 uninfected, 2 infected); white blood cells are dropped;
`ambiguous` cells become ignore regions: one `iscrowd=1` annotation per evaluated class, so COCOeval and
`match_per_class` count a detection overlapping them (IoA >= 0.5) neither as TP nor as FP.

Tiling: the frame is resized by `scale` (cells at the BBBC041 eval pixel size) and cut into model-input tiles with
overlap; each tile owns a core region (split at the middle of every overlap), a prediction is kept only by the tile
that owns its box centre. Tiles almost entirely outside the circular field of view are not evaluated.
"""
from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from malaria_ssl.eval.detection_metrics import iou_matrix

# subset -> (Zenodo COCO file, NIH folder, image id offset)
SUBSETS = {"polys": ("nih_polys_coco.json", "Polygon Set", 0), "points": ("nih_points_coco.json", "Point Set", 1000)}
CATEGORIES = [{"id": 1, "name": "uninfected"}, {"id": 2, "name": "infected"}]   # = BBBC041 binary ids
MAPPING = {"uninfected": "uninfected", "parasitized": "infected", "white blood cell": None, "ambiguous": "ignore"}

_XLSX_PATH = re.compile(r"'?NIH-NLM-ThinBloodSmearsPf\\(Polygon Set|Point Set)\\([^\\]+)\\GT\\(IMG_[0-9_]+)\.txt'?")


# --------------------------------------------------------------------------- patients

def read_patient_table(xlsx: Path) -> list[dict[str, Any]]:
    """One row per image of the NIH statistics workbook: subset, file name, patient id (folder), image path inside
    the zip and the ORIGINAL NIH cell counts. Plain zip/XML parsing (no spreadsheet dependency)."""
    folder_to_subset = {folder: s for s, (_, folder, _) in SUBSETS.items()}
    with zipfile.ZipFile(xlsx) as z:
        strings = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf8"))
        sheets = [z.read(n).decode("utf8") for n in sorted(z.namelist()) if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)]
    rows = []
    for xml in sheets:
        for r in re.findall(r"<row[^>]*>(.*?)</row>", xml):
            cells = {c: (strings[int(v)] if 't="s"' in attrs else v) for c, attrs, v in
                     re.findall(r'<c r="([A-Z]+)\d+"([^>]*)>(?:<f>[^<]*</f>)?<v>([^<]*)</v>', r)}
            m = _XLSX_PATH.fullmatch(cells.get("A", ""))
            if not m:
                continue
            folder, patient, stem = m.groups()
            rows.append({"subset": folder_to_subset[folder], "file_name": f"{stem}.jpg", "patient_id": patient,
                         "path": f"{folder}/{patient}/img/{stem}.jpg", "nih_wbc": int(cells["K"]),
                         "nih_infected": int(cells["L"]), "nih_uninfected": int(cells["M"])})
    return rows


def check_patients(rows: list[dict[str, Any]]) -> dict[str, int]:
    """193 patients, 5 images each, one subset per patient, unique (subset, file) keys; raises otherwise."""
    keys = Counter((r["subset"], r["file_name"]) for r in rows)
    if max(keys.values()) > 1:
        raise ValueError("duplicate (subset, file_name) in the patient table")
    per_patient = Counter(r["patient_id"] for r in rows)
    subsets = {}
    for r in rows:
        subsets.setdefault(r["patient_id"], set()).add(r["subset"])
    if len(per_patient) != 193 or set(per_patient.values()) != {5} or any(len(s) != 1 for s in subsets.values()):
        raise ValueError(f"expected 193 patients x 5 images in one subset each, got {len(per_patient)} patients, "
                         f"images/patient {sorted(set(per_patient.values()))}")
    return {s: sum(1 for v in subsets.values() if s in v) for s in SUBSETS}


# --------------------------------------------------------------------------- conversion

def convert(sources: dict[str, dict[str, Any]], patients: list[dict[str, Any]]) -> dict[str, Any]:
    """Zenodo COCO files {subset: coco} -> one binary COCO dict (ids remapped, WBC dropped, ambiguous -> ignore).

    Images keep subset, patient_id, the zip-relative `file_name` and their original Zenodo id / file name."""
    by_key = {(r["subset"], r["file_name"]): r for r in patients}
    cat_id = {c["name"]: c["id"] for c in CATEGORIES}
    images, anns = [], []
    for subset, coco in sources.items():
        offset = SUBSETS[subset][2]
        src_cat = {c["id"]: c["name"] for c in coco["categories"]}
        if not set(src_cat.values()) <= set(MAPPING):
            raise ValueError(f"{subset}: unknown categories {set(src_cat.values()) - set(MAPPING)}")
        id_map = {}
        for im in sorted(coco["images"], key=lambda i: i["id"]):
            p = by_key[(subset, im["file_name"])]
            id_map[im["id"]] = im["id"] + offset
            images.append({"id": im["id"] + offset, "file_name": p["path"], "width": im["width"],
                           "height": im["height"], "subset": subset, "patient_id": p["patient_id"],
                           "source_id": im["id"], "source_file_name": im["file_name"]})
        for a in sorted(coco["annotations"], key=lambda a: a["id"]):
            target = MAPPING[src_cat[a["category_id"]]]
            if target is None:
                continue
            x, y, w, h = (float(v) for v in a["bbox"])
            base = {"image_id": id_map[a["image_id"]], "bbox": [x, y, w, h], "area": w * h,
                    "source": subset, "source_ann_id": a["id"]}
            if target == "ignore":   # one ignore region per class
                anns.extend({**base, "category_id": c["id"], "iscrowd": 1, "ignore": 1} for c in CATEGORIES)
            else:
                anns.append({**base, "category_id": cat_id[target], "iscrowd": 0})
    if len({im["id"] for im in images}) != len(images):
        raise ValueError("image id collision after the subset offsets")
    for i, a in enumerate(anns, start=1):
        a["id"] = i
    return {"info": {"description": "NIH-NLM-ThinBloodSmearsPf, binary, ambiguous = iscrowd ignore regions"},
            "images": images, "annotations": anns, "categories": CATEGORIES}


def subset_coco(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    return {**coco, "images": [im for im in coco["images"] if im["id"] in image_ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids]}


def agnostic_coco(coco: dict[str, Any]) -> dict[str, Any]:
    """Single class `cell` (id 1): every labelled cell is GT; each ambiguous region once as an ignore region."""
    seen, anns = set(), []
    for a in coco["annotations"]:
        if a["iscrowd"]:
            key = (a["source"], a["source_ann_id"])
            if key in seen:
                continue
            seen.add(key)
        anns.append({**a, "category_id": 1})
    return {**coco, "annotations": anns, "categories": [{"id": 1, "name": "cell"}]}


def parasitemia(coco: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """True parasitemia (infected / (infected + uninfected) labelled RBCs; ambiguous and WBC excluded)
    per image and per patient."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    n = {im["id"]: Counter() for im in coco["images"]}
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            n[a["image_id"]][names[a["category_id"]]] += 1
    n_amb = Counter(a["image_id"] for a in coco["annotations"] if a["iscrowd"] and a["category_id"] == 1)
    per_image, per_patient = [], {}
    for im in coco["images"]:
        c = n[im["id"]]
        tot = c["infected"] + c["uninfected"]
        per_image.append({"image_id": im["id"], "subset": im["subset"], "patient_id": im["patient_id"],
                          "n_infected": c["infected"], "n_uninfected": c["uninfected"],
                          "n_ambiguous": n_amb[im["id"]], "parasitemia": c["infected"] / tot if tot else float("nan")})
        p = per_patient.setdefault(im["patient_id"], {"patient_id": im["patient_id"], "subset": im["subset"],
                                                      "n_images": 0, "n_infected": 0, "n_uninfected": 0,
                                                      "n_ambiguous": 0})
        p["n_images"] += 1
        p["n_infected"] += c["infected"]
        p["n_uninfected"] += c["uninfected"]
        p["n_ambiguous"] += n_amb[im["id"]]
    for p in per_patient.values():
        tot = p["n_infected"] + p["n_uninfected"]
        p["parasitemia"] = p["n_infected"] / tot if tot else float("nan")
    return per_image, list(per_patient.values())


# --------------------------------------------------------------------------- tiling

def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Tile origins along one axis: stride tile - overlap, the last tile flush with the end."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    n = int(np.ceil((length - tile) / stride)) + 1
    return sorted({min(i * stride, length - tile) for i in range(n)})


def core_bounds(starts: list[int], tile: int, length: int) -> list[tuple[float, float]]:
    """[lo, hi) region owned by each tile: split at the middle of each overlap, outer tiles reach the edges."""
    ends = [min(s + tile, length) for s in starts]
    cuts = [(ends[i] + starts[i + 1]) / 2 for i in range(len(starts) - 1)]
    return [(0.0 if i == 0 else cuts[i - 1], float(length) if i == len(starts) - 1 else cuts[i])
            for i in range(len(starts))]


def make_tiles(width: int, height: int, tile_wh: tuple[int, int], overlap: int) -> list[dict[str, Any]]:
    """Tiles of a (resized) frame: pixel window x0, y0, x1, y1 and owned core cx0, cy0, cx1, cy1."""
    tw, th = tile_wh
    xs, ys = tile_starts(width, tw, overlap), tile_starts(height, th, overlap)
    cx, cy = core_bounds(xs, tw, width), core_bounds(ys, th, height)
    return [{"x0": x, "y0": y, "x1": min(x + tw, width), "y1": min(y + th, height),
             "core": (cx[i][0], cy[j][0], cx[i][1], cy[j][1])}
            for j, y in enumerate(ys) for i, x in enumerate(xs)]


def fov_fraction(frame: np.ndarray, tile: dict[str, Any], threshold: float) -> float:
    """Share of tile pixels inside the circular field of view (grey level above `threshold`)."""
    patch = frame[tile["y0"]:tile["y1"], tile["x0"]:tile["x1"]]
    return float((patch.mean(axis=2) > threshold).mean())


def in_core(boxes_xywh: np.ndarray, core: tuple[float, float, float, float]) -> np.ndarray:
    """Mask of boxes (frame coordinates, same scale as `core`) whose centre lies in [lo, hi) of the core."""
    cx = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    cy = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    return (cx >= core[0]) & (cx < core[2]) & (cy >= core[1]) & (cy < core[3])


def seam_nms(boxes_xywh: np.ndarray, scores: np.ndarray, tiles: np.ndarray, iou: float) -> np.ndarray:
    """Keep mask after greedy NMS that only suppresses boxes from a DIFFERENT tile (duplicates across seams);
    boxes of the same tile are never suppressed (the model's own output is left as it is)."""
    keep = np.ones(len(scores), bool)
    order = np.argsort(-scores, kind="mergesort")
    ious = iou_matrix(boxes_xywh, boxes_xywh)
    for i in order:   # every still-kept box other than i comes later in the order (a higher one would have removed i)
        if keep[i]:
            keep[(ious[i] >= iou) & (tiles != tiles[i])] = False
    return keep
