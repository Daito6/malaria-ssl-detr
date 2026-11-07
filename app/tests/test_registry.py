"""Model registry: real entries resolve, missing files give a clear error naming the path."""

from __future__ import annotations

import pytest
import yaml

from app.registry import REGISTRY_PATH, ModelUnavailable, load_registry
from malaria_ssl.utils.config import PROJECT_ROOT


def _write(tmp_path, models):
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    return p


def test_registered_models_are_available(registry):
    assert set(registry) == {"a_coco_detr_binary", "b_vit_imagenet", "b_vit_joint", "two_stage_dino"}
    for e in registry.values():
        assert e.available, e.error
    assert registry["a_coco_detr_binary"].threshold_source == "runs/a_coco_detr_binary/thresholds.json"
    assert registry["two_stage_dino"].thresholds["uninfected"] is None  # never chosen on val
    assert registry["two_stage_dino"].spec["variant"] in ("det_x_p", "p")


def test_missing_checkpoint_is_reported(tmp_path):
    spec = {
        "id": "ghost",
        "kind": "detr",
        "run": "runs/a_coco_detr_binary",
        "checkpoint": "runs/a_coco_detr_binary/checkpoints/does_not_exist.pt",
        "name": {"uk": "x", "en": "x"},
    }
    e = load_registry(_write(tmp_path, [spec]), PROJECT_ROOT)["ghost"]
    assert not e.available
    assert "does_not_exist.pt" in e.error
    with pytest.raises(ModelUnavailable, match="does_not_exist.pt"):
        e.require()


def test_missing_run_and_dependent_two_stage(tmp_path):
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["models"]
    two_stage = next(m for m in raw if m["kind"] == "two_stage")
    boxes = {"id": two_stage["boxes"], "kind": "detr", "run": "runs/no_such_run", "name": {"uk": "x", "en": "x"}}
    reg = load_registry(_write(tmp_path, [boxes, two_stage]), PROJECT_ROOT)
    assert "runs/no_such_run" in reg[boxes["id"]].error.replace("\\", "/")
    assert not reg[two_stage["id"]].available and "unavailable" in reg[two_stage["id"]].error


def test_bad_kind_and_duplicates(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        load_registry(_write(tmp_path, [{"id": "x", "kind": "yolo", "run": "r", "name": {}}]), PROJECT_ROOT)
    m = {"id": "x", "kind": "detr", "run": "runs/a_coco_detr_binary", "name": {"uk": "x", "en": "x"}}
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(_write(tmp_path, [m, m]), PROJECT_ROOT)
