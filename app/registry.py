"""Model registry (app/models.yaml): which models exist on disk and their default thresholds.

Nothing here loads weights; a model whose files are missing is listed as unavailable with a clear error
naming the missing path. Default thresholds are read from files written by the experiments (chosen on val).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD
from malaria_ssl.utils.config import PROJECT_ROOT

REGISTRY_PATH = Path(__file__).with_name("models.yaml")
KINDS = ("detr", "two_stage")


class ModelUnavailable(RuntimeError):
    """The model's files are missing or inconsistent; the message says which."""


@dataclass
class ModelEntry:
    id: str
    kind: str
    name: dict[str, str]
    description: dict[str, str]
    spec: dict[str, Any]
    normalization: dict[str, list[float]]
    files: dict[str, Path] = field(default_factory=dict)
    thresholds: dict[str, float | None] = field(default_factory=dict)
    threshold_source: str | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None

    def require(self) -> None:
        if self.error:
            raise ModelUnavailable(self.error)


def _norm(value: Any) -> dict[str, list[float]]:
    if value == "imagenet":
        return {"mean": [float(v) for v in IMAGENET_MEAN], "std": [float(v) for v in IMAGENET_STD]}
    if isinstance(value, dict) and {"mean", "std"} <= set(value):
        return {"mean": [float(v) for v in value["mean"]], "std": [float(v) for v in value["std"]]}
    raise ValueError(f"normalization must be 'imagenet' or {{mean, std}}, got {value!r}")


def _files(spec: dict[str, Any], root: Path) -> dict[str, Path]:
    """Every file the model needs, by role."""
    if spec["kind"] == "detr":
        run = root / spec["run"]
        return {
            "config": run / "config.yaml",
            "checkpoint": root / spec.get("checkpoint", f"{spec['run']}/checkpoints/best.pt"),
            "thresholds": root / spec.get("thresholds", f"{spec['run']}/thresholds.json"),
        }
    ref = spec["reference"]
    return {
        "train_features": root / spec["train_features"],
        "ssl_config": root / spec["encoder"]["ssl_config"],
        "reference_candidates": root / ref["candidates"],
        "reference_features": root / ref["features"],
        "selection": root / ref["selection"]["file"],
    }


def _two_stage_selection(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    """The row of the two-stage evaluation that fixed the score variant and the infected threshold on val."""
    sel = spec["reference"]["selection"]
    rows = json.loads(path.read_text(encoding="utf-8"))["selection"]
    for r in rows:
        if r["run"] == sel["run"] and r["encoder"] == sel["encoder"] and r["training"] == spec["training"]:
            return r
    raise ModelUnavailable(
        f"no selection row run={sel['run']} encoder={sel['encoder']} training={spec['training']} in {path}"
    )


def load_registry(path: Path = REGISTRY_PATH, root: Path = PROJECT_ROOT) -> dict[str, ModelEntry]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["models"]
    entries: dict[str, ModelEntry] = {}
    for spec in raw:
        if spec["kind"] not in KINDS:
            raise ValueError(f"model {spec['id']}: kind must be one of {KINDS}, got {spec['kind']!r}")
        if spec["id"] in entries:
            raise ValueError(f"duplicate model id {spec['id']!r}")
        e = ModelEntry(
            id=spec["id"],
            kind=spec["kind"],
            name=spec["name"],
            description=spec.get("description", {"uk": "", "en": ""}),
            spec=spec,
            normalization=_norm(spec.get("normalization", "imagenet")),
            files=_files(spec, Path(root)),
        )
        if e.kind == "detr" and e.files["config"].is_file():  # ViT-DETR built on a saved encoder (e.g. SSL joint)
            vit_init = (yaml.safe_load(e.files["config"].read_text(encoding="utf-8"))["model"].get("vit") or {}).get(
                "init"
            )
            if vit_init not in (None, "random", "imagenet", "dino"):
                e.files["vit_init"] = Path(root) / vit_init
        missing = [f"{role}: {p}" for role, p in e.files.items() if not p.is_file()]
        if spec["kind"] == "two_stage":
            boxes = entries.get(spec["boxes"])
            if boxes is None:
                missing.append(f"boxes model {spec['boxes']!r} (must be listed before {spec['id']!r})")
            elif not boxes.available:
                missing.append(f"boxes model {spec['boxes']!r} is unavailable")
        if missing:
            e.error = "missing files: " + "; ".join(missing)
        else:
            try:
                if e.kind == "detr":
                    e.thresholds = json.loads(e.files["thresholds"].read_text(encoding="utf-8"))
                    e.threshold_source = e.files["thresholds"].relative_to(root).as_posix()
                else:
                    row = _two_stage_selection(spec, e.files["selection"])
                    e.spec = {**spec, "variant": row["variant"]}
                    # only the infected threshold was chosen on val; uninfected has no val-selected value
                    e.thresholds = {"uninfected": None, "infected": row["threshold"]}
                    e.threshold_source = e.files["selection"].relative_to(root).as_posix()
            except (ModelUnavailable, KeyError, ValueError) as err:
                e.error = f"cannot read thresholds: {err}"
        entries[e.id] = e
    return entries
