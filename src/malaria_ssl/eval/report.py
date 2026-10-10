"""Full evaluation of one split: COCO AP/AR, thresholded P/R/F1, confusion matrix, density split, CIs."""
from __future__ import annotations

from typing import Any

from malaria_ssl.eval.bootstrap import bootstrap_ci
from malaria_ssl.eval.coco_metrics import run_cocoeval, summarize
from malaria_ssl.eval.detection_metrics import choose_thresholds, confusion_matrix, match_per_class, prf_report

DENSE_CELLS = 100  # DETR's default number of queries


def evaluate_split(gt: dict[str, Any], dets: list[dict[str, Any]], cells_per_image: dict[int, int],
                   thresholds: dict[int, float] | None, iou_thr: float = 0.5,
                   n_bootstrap: int = 0, seed: int = 0) -> tuple[dict[str, Any], dict[int, float]]:
    """Evaluate detections against `gt`.

    `thresholds` (cat_id -> score) must come from val; pass None only for val itself, in which
    case they are chosen here (max F1 per class) and returned.
    `cells_per_image` counts ALL annotated cells per image (multiclass GT) for the density split.
    """
    names = {c["id"]: c["name"] for c in gt["categories"]}
    ev = run_cocoeval(gt, dets)
    metrics: dict[str, Any] = {"coco": summarize(ev, names)}

    matches = match_per_class(gt, dets, iou_thr)
    chosen_here = thresholds is None
    if chosen_here:
        thresholds = choose_thresholds(matches)
    metrics["thresholds"] = {names[c]: t for c, t in thresholds.items()}
    metrics["thresholds_source"] = "chosen on this split (max F1)" if chosen_here else "val"
    metrics["prf"] = prf_report(matches, thresholds, names)
    metrics["confusion_iou50"] = confusion_matrix(gt, dets, thresholds, iou_thr)

    img_ids = [im["id"] for im in gt["images"]]
    groups = {f"<={DENSE_CELLS}": [i for i in img_ids if cells_per_image[i] <= DENSE_CELLS],
              f">{DENSE_CELLS}": [i for i in img_ids if cells_per_image[i] > DENSE_CELLS]}
    metrics["by_density"] = {}
    for g, ids in groups.items():
        if not ids:
            metrics["by_density"][g] = {"n_images": 0}
            continue
        sub = summarize(run_cocoeval(gt, dets, ids), names)
        metrics["by_density"][g] = {"n_images": len(ids), "AP": sub["AP"], "AP50": sub["AP50"],
                                    "AP50_per_class": {k: v["AP50"] for k, v in sub["per_class"].items()},
                                    **{k: v for k, v in prf_report(matches, thresholds, names, ids).items()}}
    if n_bootstrap:
        metrics["bootstrap"] = bootstrap_ci(ev, matches, thresholds, names, n_bootstrap, seed)
    return metrics, thresholds
