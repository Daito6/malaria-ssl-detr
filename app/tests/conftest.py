"""Shared fixtures. Every test runs on CPU: the GPU is hidden before torch is imported."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # '' would unset it and leave the GPU visible

import time
from pathlib import Path

import pytest

from app.inference import ModelPool
from app.registry import load_registry
from malaria_ssl.utils.config import PROJECT_ROOT

# one val image (id 2), never a test image
VAL_IMAGE = PROJECT_ROOT / "data/raw/malaria/images/003a89b0-a095-417a-8dd6-f408339bbc68.png"
VAL_IMAGE_ID = 2


@pytest.fixture(scope="session")
def registry():
    return load_registry()


@pytest.fixture(scope="session")
def pool(registry):
    return ModelPool(registry, "cpu")


@pytest.fixture(scope="session")
def val_image_path() -> Path:
    if not VAL_IMAGE.is_file():
        pytest.fail(f"val image missing: {VAL_IMAGE}")
    return VAL_IMAGE


def wait_job(client, job_id: str, timeout: float = 300.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.2)
    raise TimeoutError(job_id)
