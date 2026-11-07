"""Robustness mode: parity at strength 0, reproducibility, val-only images, missing checkpoint, stacks, API smoke."""

from __future__ import annotations

import io

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from app import corruptions as C
from app import robustness, schemas
from app.inference import ModelPool, class_agnostic_nms
from app.main import create_app
from app.registry import load_registry
from app.tests.conftest import VAL_IMAGE_ID, wait_job
from malaria_ssl.data.detection import load_split_coco, read_rgb
from malaria_ssl.utils.config import PROJECT_ROOT, load_config, resolve_path

DETR = ["a_coco_detr_binary", "b_vit_imagenet", "b_vit_joint"]


@pytest.fixture(scope="module")
def client(registry, pool):
    with TestClient(create_app(registry=registry, pool=pool, static_dir=None)) as c:
        yield c


def _dets(r):
    return [(d["label"], round(d["score"], 6), tuple(round(v, 3) for v in d["bbox"])) for d in r.detections]


@pytest.mark.parametrize("model_id", DETR)
def test_strength_zero_equals_plain_inference(pool, val_image_path, model_id):
    img = read_rgb(val_image_path)
    plain = pool.run(model_id, img)
    for kind in C.KINDS:
        assert _dets(pool.run(model_id, img, corruption=(kind, 0.0, VAL_IMAGE_ID))) == _dets(plain), kind


@pytest.mark.parametrize("kind", ["stain", "jpeg", "scale_up"])
def test_reproducible(pool, val_image_path, kind):
    img = read_rgb(val_image_path)
    a = pool.run("a_coco_detr_binary", img, corruption=(kind, 0.5, VAL_IMAGE_ID))
    b = pool.run("a_coco_detr_binary", img.copy(), corruption=(kind, 0.5, VAL_IMAGE_ID))
    assert _dets(a) == _dets(b)
    assert a.corruption["params"] == C.describe(kind, 0.5)


def test_scale_maps_boxes_to_original_pixels(pool, val_image_path):
    img = read_rgb(val_image_path)
    r = pool.run("a_coco_detr_binary", img, corruption=("scale_up", 1.0, VAL_IMAGE_ID))
    assert (r.width, r.height) == (1600, 1200)
    boxes = np.array([d["bbox"] for d in r.detections])
    assert boxes[:, 0].max() <= 1600 and boxes[:, 1].max() <= 1200
    assert r.corruption["params"]["long_side"] == 1016


def test_val_images_only():
    dcfg = load_config(resolve_path("configs/data.yaml"))
    val_ids = {im["id"] for im in load_split_coco("val", "multiclass", dcfg, resolve_path)["images"]}
    test_ids = {im["id"] for im in load_split_coco("test", "multiclass", dcfg, resolve_path)["images"]}
    served = set(robustness.val_images())
    assert served == val_ids and not served & test_ids
    with pytest.raises(KeyError):
        robustness.load_val_image(next(iter(test_ids)))


def test_api_smoke(client):
    imgs = [schemas.ValImage.model_validate(x) for x in client.get("/api/val-images").json()]
    assert len(imgs) == 181 and VAL_IMAGE_ID in {i.id for i in imgs}
    kinds = [schemas.CorruptionKind.model_validate(x) for x in client.get("/api/robustness/corruptions").json()]
    assert [k.kind for k in kinds] == list(C.KINDS)
    d = client.get("/api/robustness/describe", params={"corruption": "jpeg", "strength": 0.5}).json()
    assert d["quality"] == 52

    r = client.get(
        "/api/robustness/preview", params={"image_id": VAL_IMAGE_ID, "corruption": "scale_down", "strength": 1.0}
    )
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (640, 480)

    body = {"model_id": "b_vit_joint", "image_id": VAL_IMAGE_ID, "corruption": "blur", "strength": 0.5}
    first = wait_job(client, client.post("/api/robustness/jobs", json=body).json()["job_id"])
    st = schemas.JobStatus.model_validate(first)
    assert st.status == "done", first.get("error")
    assert st.result.corruption["kind"] == "blur" and st.result.corruption["params"]["sigma_px"] == 1.0
    again = wait_job(client, client.post("/api/robustness/jobs", json=body).json()["job_id"])
    assert again["result"]["detections"] == first["result"]["detections"]  # served from the cache


@pytest.mark.parametrize(
    "params,code",
    [
        ({"image_id": 999999, "corruption": "blur", "strength": 0.5}, "unknown_val_image"),
        ({"image_id": VAL_IMAGE_ID, "corruption": "nope", "strength": 0.5}, "invalid_corruption"),
    ],
)
def test_preview_errors(client, params, code):
    r = client.get("/api/robustness/preview", params=params)
    assert r.json()["error"]["code"] == code


def test_job_errors(client):
    r = client.post(
        "/api/robustness/jobs",
        json={"model_id": "two_stage_dino", "image_id": VAL_IMAGE_ID, "corruption": "blur", "strength": 0.5},
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_model"
    r = client.post(
        "/api/robustness/jobs",
        json={"model_id": "a_coco_detr_binary", "image_id": VAL_IMAGE_ID, "corruption": "blur", "strength": 2},
    )
    assert r.status_code == 422


def test_curve_endpoint_matches_file(client):
    r = client.get("/api/robustness/curve")
    if robustness.read_curve() is None:
        assert r.status_code == 404 and r.json()["error"]["code"] == "curve_missing"
    else:
        assert schemas.Curve.model_validate(r.json()).rows


def test_missing_joint_checkpoint(tmp_path):
    """Joint is listed but unavailable when its checkpoint (or its SSL encoder) is missing; the API says which file."""
    raw = yaml.safe_load((PROJECT_ROOT / "app/models.yaml").read_text("utf-8"))["models"]
    joint = next(m for m in raw if m["id"] == "b_vit_joint")
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump({"models": [{**joint, "checkpoint": "runs/b_vit_joint/checkpoints/gone.pt"}]}), "utf-8")
    reg = load_registry(p, PROJECT_ROOT)
    assert not reg["b_vit_joint"].available and "gone.pt" in reg["b_vit_joint"].error
    with TestClient(create_app(registry=reg, pool=ModelPool(reg, "cpu"), static_dir=None)) as c:
        r = c.post(
            "/api/robustness/jobs",
            json={"model_id": "b_vit_joint", "image_id": VAL_IMAGE_ID, "corruption": "blur", "strength": 0.5},
        )
    assert r.status_code == 503 and "gone.pt" in r.json()["error"]["message"]


def test_joint_requires_its_ssl_encoder(registry):
    assert registry["b_vit_joint"].files["vit_init"].as_posix().endswith("runs/ssl_joint/checkpoints/encoder_final.pt")


def test_nms_still_applied(pool, val_image_path):
    r = pool.run("b_vit_joint", read_rgb(val_image_path), corruption=("jpeg", 1.0, VAL_IMAGE_ID))
    assert len(class_agnostic_nms(r.detections)) == len(r.detections)


def test_stack_inference_parity(pool, val_image_path):
    """A stack of one = the single corruption; an all-zero stack = plain inference."""
    img = read_rgb(val_image_path)
    single = pool.run("a_coco_detr_binary", img, corruption=("scale_up", 0.5, VAL_IMAGE_ID))
    one = pool.run("a_coco_detr_binary", img, stack=([("scale_up", 0.5)], VAL_IMAGE_ID))
    assert _dets(one) == _dets(single)
    zero = pool.run("a_coco_detr_binary", img, stack=([(k, 0.0) for k in C.STACK_ORDER], VAL_IMAGE_ID))
    assert _dets(zero) == _dets(pool.run("a_coco_detr_binary", img))
    assert zero.corruption["identity"] is True and zero.corruption["steps"] == []


def test_stack_api(client):
    stack = [
        {"kind": "jpeg", "strength": 0.5},
        {"kind": "blur", "strength": 0.0},
        {"kind": "scale_down", "strength": 1},
    ]
    body = {"model_id": "a_coco_detr_binary", "image_id": VAL_IMAGE_ID, "stack": stack}
    first = wait_job(client, client.post("/api/robustness/jobs", json=body).json()["job_id"])
    assert first["status"] == "done", first.get("error")
    c = first["result"]["corruption"]
    assert [s["kind"] for s in c["steps"]] == ["jpeg", "scale_down"] and c["long_side"] == 640
    reordered = {**body, "stack": [stack[0], stack[2]]}  # same canonical stack -> served from the cache
    again = wait_job(client, client.post("/api/robustness/jobs", json=reordered).json()["job_id"])
    assert again["result"]["detections"] == first["result"]["detections"]

    q = {"image_id": VAL_IMAGE_ID, "stack": "jpeg:0.5,scale_down:1"}
    r = client.get("/api/robustness/preview", params=q)
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (640, 480)
    d = client.get("/api/robustness/describe", params={"stack": "jpeg:0.5,scale_down:1"}).json()
    assert d["long_side"] == 640 and d["steps"][0]["params"]["quality"] == 52


def test_stack_api_errors(client):
    bad = {"model_id": "a_coco_detr_binary", "image_id": VAL_IMAGE_ID}
    r = client.post("/api/robustness/jobs", json={**bad, "stack": [{"kind": "blur", "strength": 0.5}] * 2})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_corruption"
    r = client.post("/api/robustness/jobs", json={**bad, "corruption": "blur", "strength": 0.5, "stack": []})
    assert r.status_code == 422
    r = client.get("/api/robustness/preview", params={"image_id": VAL_IMAGE_ID, "stack": "scale_up:1,scale_down:1"})
    assert r.json()["error"]["code"] == "invalid_corruption"
