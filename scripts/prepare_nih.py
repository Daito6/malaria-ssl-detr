"""Prepare the NIH-NLM thin-smear set (EXPLORATORY zero-shot target domain; see src/malaria_ssl/data/nih.py).

    python scripts/prepare_nih.py [--config configs/nih_zeroshot.yaml] [--measure-only]

Idempotent steps (raw data is only read; outputs in data/processed/nih_nlm/ and reports/tables/):
1. check the md5 of the Zenodo annotation files; extract the NIH zip (images, GT txt, licence) next to it;
   write the NLM Informational Notice as plain text next to the data;
2. patients.csv: file -> patient id from Dataset_statistics.xlsx (193 patients x 5 images, subsets disjoint);
3. nih_binary.json: polys + points in one binary COCO (points image ids + 1000, WBC dropped, ambiguous = iscrowd);
4. parasitemia_images.csv / parasitemia_patients.csv (true fraction infected, ambiguous / WBC excluded);
5. scale: median max(w, h) of the BBBC041 TRAIN boxes at the eval long side vs the NIH boxes ->
   reports/tables/nih_zeroshot_scale.csv; the factor in the config must equal the measured one (--measure-only
   prints it and stops);
6. cache of the frames resized by the factor (uint8 memmap) and the tile list with field-of-view fractions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import zipfile

import cv2
import numpy as np
import pandas as pd

from malaria_ssl.data.detection import load_split_coco, median_resized_box_side, read_rgb
from malaria_ssl.data.nih import (SUBSETS, check_patients, convert, fov_fraction, make_tiles, parasitemia,
                                  read_patient_table)
from malaria_ssl.utils.config import load_config, resolve_path

log = logging.getLogger("prepare_nih")


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(cfg) -> None:
    zpath, out = resolve_path(cfg["zip"]), resolve_path(cfg["images_dir"])
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.infolist() if not m.filename.lower().endswith("thumbs.db")]
        if any(m.filename.startswith(("/", "\\")) or ".." in m.filename for m in members):
            raise SystemExit("unsafe path in the NIH zip")
        todo = [m for m in members if not m.is_dir() and not (out / m.filename).exists()]
        for m in todo:
            z.extract(m, out)
        log.info("zip: %d members, %d extracted now", len(members), len(todo))
        doc = z.read("Data License Agreement.docx")
    with zipfile.ZipFile(io.BytesIO(doc)) as d:
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d.read("word/document.xml").decode("utf8"))).strip()
    (resolve_path(cfg["raw_dir"]) / "NLM_INFORMATIONAL_NOTICE.txt").write_text(
        "Text of 'Data License Agreement.docx' from NIH-NLM-ThinBloodSmearsPf.zip (whitespace normalized).\n"
        "Courtesy of the U.S. National Library of Medicine.\n\n" + text + "\n", encoding="utf-8")


def measure_scale(ncfg, scfg, coco) -> dict:
    dcfg = load_config(resolve_path("configs/data.yaml"))
    bbbc = median_resized_box_side(load_split_coco("train", "binary", dcfg, resolve_path), scfg["bbbc_long_side"])
    row = {"bbbc_train_median_side_at_long_side": bbbc, "bbbc_long_side": scfg["bbbc_long_side"]}
    for name, subset in (("polys", "polys"), ("points", "points"), ("all", None)):
        ids = {im["id"] for im in coco["images"] if subset in (None, im["subset"])}
        sides = [max(a["bbox"][2], a["bbox"][3]) for a in coco["annotations"]
                 if a["image_id"] in ids and not a["iscrowd"]]
        row[f"nih_{name}_median_side"] = float(np.median(sides))
        row[f"nih_{name}_n_boxes"] = len(sides)
    row["factor"] = bbbc / row["nih_all_median_side"]
    return row


def build_cache(ncfg, tcfg, coco, factor) -> None:
    out = resolve_path(ncfg["processed_dir"])
    images = sorted(coco["images"], key=lambda im: im["id"])
    sizes = {(im["width"], im["height"]) for im in images}
    (w0, h0), = sizes
    w, h = round(w0 * factor), round(h0 * factor)
    index = {"factor": factor, "shape": [len(images), h, w, 3], "ids": [im["id"] for im in images]}
    npy, idx = out / f"cache_{w}x{h}.npy", out / "cache_index.json"
    if not (npy.exists() and idx.exists() and json.loads(idx.read_text(encoding="utf-8")) == index):
        tmp = npy.with_name(npy.stem + ".tmp.npy")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=tuple(index["shape"]))
        root = resolve_path(ncfg["images_dir"])
        for i, im in enumerate(images):
            arr[i] = cv2.resize(read_rgb(root / im["file_name"]), (w, h), interpolation=cv2.INTER_AREA)
            if (i + 1) % 100 == 0:
                log.info("cache %d/%d", i + 1, len(images))
        arr.flush()
        del arr
        tmp.replace(npy)
        idx.write_text(json.dumps({**index, "file": npy.name}), encoding="utf-8")
    arr = np.load(npy, mmap_mode="r")
    tiles, kept = [], 0
    for row, im in enumerate(images):
        frame = np.asarray(arr[row])
        for t in make_tiles(w, h, tuple(tcfg["size"]), tcfg["overlap"]):
            f = fov_fraction(frame, t, tcfg["fov_threshold"])
            t.update(image_id=im["id"], row=row, fov=f, keep=f >= tcfg["min_fov_fraction"])
            kept += t["keep"]
            tiles.append(t)
    (out / "tiles.json").write_text(json.dumps({"tile": tcfg, "frame": [w, h], "tiles": tiles}), encoding="utf-8")
    log.info("frames %dx%d, %d tiles, %d kept (fov >= %.2f)", w, h, len(tiles), kept, tcfg["min_fov_fraction"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/nih_zeroshot.yaml")
    ap.add_argument("--measure-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(resolve_path(args.config))
    ncfg = cfg["nih"]
    ann_dir = resolve_path(ncfg["annotations_dir"])
    for name, digest in ncfg["md5"].items():
        if md5(ann_dir / name) != digest:
            raise SystemExit(f"md5 mismatch: {name}")
    extract(ncfg)
    out = resolve_path(ncfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    rows = read_patient_table(resolve_path(ncfg["xlsx"]))
    per_subset = check_patients(rows)
    sources = {s: json.loads((ann_dir / f).read_text(encoding="utf-8")) for s, (f, _, _) in SUBSETS.items()}
    coco = convert(sources, rows)
    missing = [im["file_name"] for im in coco["images"] if not (resolve_path(ncfg["images_dir"]) / im["file_name"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing after extraction, e.g. {missing[:3]}")
    id_of = {(im["subset"], im["source_file_name"]): im["id"] for im in coco["images"]}
    with (out / "patients.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "subset", "file_name", "patient_id", "path", "nih_wbc",
                                          "nih_infected", "nih_uninfected"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: id_of[(r["subset"], r["file_name"])]):
            w.writerow({"image_id": id_of[(r["subset"], r["file_name"])], **r})
    (out / "nih_binary.json").write_text(json.dumps(coco), encoding="utf-8")
    per_image, per_patient = parasitemia(coco)
    pd.DataFrame(per_image).to_csv(out / "parasitemia_images.csv", index=False)
    pd.DataFrame(per_patient).to_csv(out / "parasitemia_patients.csv", index=False)
    n = pd.DataFrame(per_image)[["n_infected", "n_uninfected", "n_ambiguous"]].sum()
    log.info("patients per subset %s; images %d; infected %d, uninfected %d, ambiguous %d, parasitemia %.4f",
             per_subset, len(coco["images"]), n["n_infected"], n["n_uninfected"], n["n_ambiguous"],
             n["n_infected"] / (n["n_infected"] + n["n_uninfected"]))

    scale = measure_scale(ncfg, cfg["scale"], coco)
    pd.DataFrame([scale]).to_csv(resolve_path("reports/tables/nih_zeroshot_scale.csv"), index=False,
                                 float_format="%.4f")
    log.info("scale: %s", scale)
    if args.measure_only:
        return
    if cfg["scale"]["factor"] is None or abs(cfg["scale"]["factor"] - scale["factor"]) > 5e-4:
        raise SystemExit(f"scale.factor in {args.config} must be the measured {scale['factor']:.4f}")
    build_cache(ncfg, cfg["tiles"], coco, cfg["scale"]["factor"])


if __name__ == "__main__":
    main()
