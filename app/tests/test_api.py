"""API contract: OpenAPI paths, schemas of every response, error codes, the job flow end to end (CPU)."""

from __future__ import annotations

import io

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from app import schemas
from app.inference import ModelPool
from app.main import create_app
from app.registry import load_registry
from app.tests.conftest import wait_job
from malaria_ssl.utils.config import PROJECT_ROOT


@pytest.fixture(scope="module")
def client(registry, pool):
    with TestClient(create_app(registry=registry, pool=pool, static_dir=None)) as c:
        yield c


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (210, 170, 190)).save(buf, format="PNG")
    return buf.getvalue()


def test_openapi_contract(client):
    spec = client.get("/openapi.json").json()
    assert {"/api/health", "/api/models", "/api/jobs", "/api/jobs/{job_id}"} <= set(spec["paths"])
    body = spec["paths"]["/api/jobs"]["post"]["requestBody"]["content"]
    assert {"image/jpeg", "image/png"} <= set(body)
    assert client.get("/docs").status_code == 200


def test_health(client):
    h = schemas.Health.model_validate(client.get("/api/health").json())
    assert h.device == "cpu" and h.limits.formats == ["jpeg", "png"]


def test_models(client, registry):
    models = [schemas.ModelInfo.model_validate(m) for m in client.get("/api/models").json()]
    assert [m.id for m in models] == list(registry)
    by_id = {m.id: m for m in models}
    assert by_id["a_coco_detr_binary"].default_thresholds == registry["a_coco_detr_binary"].thresholds
    assert by_id["b_vit_imagenet"].normalization.mean == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("model_id", ["a_coco_detr_binary", "two_stage_dino"])
def test_job_end_to_end(client, val_image_path, model_id):
    r = client.post(
        f"/api/jobs?model_id={model_id}", content=val_image_path.read_bytes(), headers={"Content-Type": "image/png"}
    )
    assert r.status_code == 202, r.text
    schemas.JobCreated.model_validate(r.json())
    body = wait_job(client, r.json()["job_id"])
    status = schemas.JobStatus.model_validate(body)
    assert status.status == "done", body.get("error")
    res = status.result
    assert (res.width, res.height) == (1600, 1200) and res.device == "cpu"
    assert res.detections
    assert "sha256" in res.checkpoint.get("boxes", res.checkpoint)  # two-stage: version of its box model
    scores = [d.score for d in res.detections]
    assert scores == sorted(scores, reverse=True)
    assert [d.id for d in res.detections] == list(range(len(scores)))


def test_large_image_job(client):
    r = client.post("/api/jobs?model_id=a_coco_detr_binary", content=_png(6000, 4500))
    assert r.status_code == 202, r.text
    body = wait_job(client, r.json()["job_id"])
    assert body["status"] == "done", body.get("error")
    res = body["result"]
    assert (res["width"], res["height"]) == (6000, 4500)
    assert all(0 <= d["bbox"][0] <= 6000 and 0 <= d["bbox"][1] <= 4500 for d in res["detections"])


def test_unknown_model(client, val_image_path):
    r = client.post("/api/jobs?model_id=nope", content=b"x")
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_model"


def test_unknown_job(client):
    r = client.get("/api/jobs/deadbeef")
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_job"


@pytest.mark.parametrize(
    "data,status,code",
    [
        (b"", 400, "empty_file"),
        (b"not an image at all", 415, "unsupported_format"),
        (b"\x89PNG\r\n\x1a\n" + b"\0" * 40, 400, "invalid_image"),
    ],
)
def test_bad_uploads(client, data, status, code):
    r = client.post("/api/jobs?model_id=a_coco_detr_binary", content=data)
    assert r.status_code == status, r.text
    assert schemas.ErrorResponse.model_validate(r.json()).error.code == code


def test_file_too_large(registry, pool):
    with TestClient(create_app(registry=registry, pool=pool, static_dir=None, max_bytes=1000)) as c:
        r = c.post("/api/jobs?model_id=a_coco_detr_binary", content=_png(400, 300) + b"\0" * 2000)
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_missing_checkpoint_model(tmp_path, val_image_path):
    spec = {
        "id": "ghost",
        "kind": "detr",
        "run": "runs/a_coco_detr_binary",
        "checkpoint": "runs/a_coco_detr_binary/checkpoints/missing.pt",
        "name": {"uk": "x", "en": "x"},
    }
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump({"models": [spec]}), encoding="utf-8")
    reg = load_registry(p, PROJECT_ROOT)
    with TestClient(create_app(registry=reg, pool=ModelPool(reg, "cpu"), static_dir=None)) as c:
        info = c.get("/api/models").json()[0]
        assert info["available"] is False and "missing.pt" in info["error"]
        r = c.post("/api/jobs?model_id=ghost", content=val_image_path.read_bytes())
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "model_unavailable" and "missing.pt" in err["message"]
