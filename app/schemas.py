"""Pydantic schemas of the SmearScan API (the OpenAPI contract at /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ClassName = Literal["uninfected", "infected"]


class LocalizedText(BaseModel):
    uk: str
    en: str


class Normalization(BaseModel):
    mean: list[float]
    std: list[float]


class ModelInfo(BaseModel):
    id: str
    kind: Literal["detr", "two_stage"]
    name: LocalizedText
    description: LocalizedText
    classes: list[ClassName]
    normalization: Normalization
    default_thresholds: dict[str, float | None] = Field(
        description="Per-class score thresholds chosen on val (max F1); null = no value was chosen on val"
    )
    threshold_source: str | None = Field(description="File the default thresholds were read from")
    available: bool
    error: str | None = Field(None, description="Why the model is unavailable (e.g. a missing file)")


class Detection(BaseModel):
    id: int = Field(description="Index in the result, by decreasing score")
    label: ClassName
    score: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x, y, width, height] in original pixels")


class InferenceResult(BaseModel):
    model_id: str
    width: int
    height: int
    device: str
    elapsed_ms: float
    nms_iou: float = Field(description="Class-agnostic NMS applied before the result is returned")
    checkpoint: dict = Field(description="Identity of the weights: file, short sha256, modification time, epoch")
    corruption: dict | None = Field(None, description="Robustness mode: kind, strength, val image id, real parameters")
    detections: list[Detection]


class ValImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int


class CorruptionKind(BaseModel):
    kind: str
    params_at_1: dict


class StackStep(BaseModel):
    kind: str
    strength: float = Field(ge=0, le=1)


class RobustnessRequest(BaseModel):
    """Either one corruption (`corruption` + `strength`) or an ordered `stack` of steps (a demonstration)."""

    model_id: str
    image_id: int = Field(description="A val image id from /api/val-images")
    corruption: str | None = None
    strength: float | None = Field(None, ge=0, le=1)
    stack: list[StackStep] | None = Field(None, description="Applied in this order; strength-0 steps are dropped")

    @model_validator(mode="after")
    def _one_mode(self):
        single = self.corruption is not None and self.strength is not None
        if single == (self.stack is not None) or (self.corruption is None) != (self.strength is None):
            raise ValueError("give either corruption + strength or stack")
        return self


class Curve(BaseModel):
    source: str
    protocol: dict
    rows: list[dict]


class JobCreated(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobStatus(BaseModel):
    job_id: str
    model_id: str
    status: Literal["queued", "running", "done", "error"]
    stage: str
    progress: float = Field(ge=0, le=1)
    queue_position: int
    error: ErrorBody | None = None
    result: InferenceResult | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class Limits(BaseModel):
    max_bytes: int
    max_pixels: int
    formats: list[str]


class Health(BaseModel):
    status: Literal["ok"]
    device: str
    limits: Limits


JobStatus.model_rebuild()
