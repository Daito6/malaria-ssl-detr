"""SmearScan FastAPI application: model registry, inference jobs, the built frontend.

Requests are logged (method, path, status, duration; for uploads the byte size and the model id). Image bytes are
never written to disk or logged: the upload is read into memory with a size limit, decoded, queued and dropped when
the job finishes.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import corruptions, robustness, schemas
from app.inference import InferenceError, InferenceResult, ModelPool
from app.jobs import Job, JobManager
from app.registry import REGISTRY_PATH, ModelEntry, ModelUnavailable, load_registry
from app.uploads import MAX_BYTES, MAX_PIXELS, UploadError, decode_image

log = logging.getLogger("smearscan")
FRONTEND_DIST = Path(__file__).with_name("frontend") / "dist"
IMAGE_BODY = {
    "requestBody": {
        "required": True,
        "description": "The raw JPEG or PNG file (not multipart).",
        "content": {t: {"schema": {"type": "string", "format": "binary"}} for t in ("image/jpeg", "image/png")},
    }
}
ERRORS = {s: {"model": schemas.ErrorResponse} for s in (400, 404, 413, 415, 503)}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_info(err: Exception) -> dict[str, str]:
    """User-facing error of a failed job."""
    if isinstance(err, ModelUnavailable):
        return {"code": "model_unavailable", "message": str(err)}
    if isinstance(err, InferenceError):
        return {"code": "inference_error", "message": str(err)}
    return {"code": "internal_error", "message": "internal error, see the server log"}


def model_info(e: ModelEntry) -> schemas.ModelInfo:
    return schemas.ModelInfo(
        id=e.id,
        kind=e.kind,
        name=e.name,
        description=e.description,
        classes=["uninfected", "infected"],
        normalization=e.normalization,
        default_thresholds=e.thresholds,
        threshold_source=e.threshold_source,
        available=e.available,
        error=e.error,
    )


def result_payload(r: InferenceResult) -> schemas.InferenceResult:
    dets = [
        schemas.Detection(
            id=i, label=d["label"], score=min(max(d["score"], 0.0), 1.0), bbox=[round(v, 2) for v in d["bbox"]]
        )
        for i, d in enumerate(r.detections)
    ]
    return schemas.InferenceResult(
        model_id=r.model_id,
        width=r.width,
        height=r.height,
        device=r.device,
        elapsed_ms=round(r.elapsed_ms, 1),
        nms_iou=r.nms_iou,
        checkpoint=r.version,
        corruption=r.corruption,
        detections=dets,
    )


def create_app(
    registry: dict[str, ModelEntry] | None = None,
    device: str = "cpu",
    max_bytes: int = MAX_BYTES,
    max_pixels: int = MAX_PIXELS,
    static_dir: Path | None = FRONTEND_DIST,
    pool: ModelPool | None = None,
) -> FastAPI:
    registry = load_registry(REGISTRY_PATH) if registry is None else registry
    pool = pool or ModelPool(registry, device)
    jobs = JobManager(error_info)
    cache = robustness.ResultCache()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(
        title="SmearScan API",
        version="0.1.0",
        lifespan=lifespan,
        description="Research tool for cell detection on blood-smear images. NOT a medical device.",
    )
    app.state.pool, app.state.jobs = pool, jobs

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        extra = getattr(request.state, "log_extra", "")
        log.info(
            "%s %s -> %d %.0f ms%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1e3,
            extra,
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, err: ApiError):
        return JSONResponse({"error": {"code": err.code, "message": err.message}}, status_code=err.status)

    @app.exception_handler(UploadError)
    async def upload_error(request: Request, err: UploadError):
        return JSONResponse({"error": {"code": err.code, "message": err.message}}, status_code=err.status)

    @app.get("/api/health", response_model=schemas.Health, tags=["meta"])
    def health():
        return {
            "status": "ok",
            "device": str(pool.device),
            "limits": {"max_bytes": max_bytes, "max_pixels": max_pixels, "formats": ["jpeg", "png"]},
        }

    @app.get("/api/models", response_model=list[schemas.ModelInfo], tags=["models"])
    def list_models():
        return [model_info(e) for e in registry.values()]

    def require_model(model_id: str) -> ModelEntry:
        if model_id not in registry:
            raise ApiError(404, "unknown_model", f"unknown model {model_id!r}")
        entry = registry[model_id]
        if not entry.available:
            raise ApiError(503, "model_unavailable", entry.error)
        return entry

    def check_image(image_id: int) -> None:
        if image_id not in robustness.val_images():
            raise ApiError(404, "unknown_val_image", f"{image_id} is not a val image id (only val images are served)")

    def steps_of(corruption: str | None, strength: float | None, stack: str | None) -> list[corruptions.Step]:
        """One corruption or a stack ('kind:s,kind:s') as a validated list of steps."""
        try:
            if stack is not None:
                steps = robustness.parse_stack(stack)
                corruptions.canonical(steps)
                return steps
            if corruption is None or strength is None:
                raise ValueError("give either corruption + strength or stack")
            corruptions.describe(corruption, strength)
            return [(corruption, strength)]
        except ValueError as err:
            raise ApiError(400, "invalid_corruption", str(err)) from err

    @app.get("/api/val-images", response_model=list[schemas.ValImage], tags=["robustness"])
    def list_val_images():
        """Reference images of the VAL split (never test)."""
        return list(robustness.val_images().values())

    @app.get("/api/robustness/corruptions", response_model=list[schemas.CorruptionKind], tags=["robustness"])
    def list_corruptions():
        """Corruption types and the parameters at strength 1 (the mapping is linear or geometric in strength)."""
        return [{"kind": k, "params_at_1": corruptions.describe(k, 1.0)} for k in corruptions.KINDS]

    @app.get("/api/robustness/describe", response_model=dict, responses=ERRORS, tags=["robustness"])
    def describe_corruption(
        corruption: str | None = None,
        strength: float | None = Query(None, ge=0, le=1),
        stack: str | None = Query(None, description="Ordered steps 'kind:strength,kind:strength'"),
    ):
        """Real parameters of one corruption, or of every step of a stack plus the final long side."""
        steps = steps_of(corruption, strength, stack)
        return corruptions.describe_stack(steps) if stack is not None else corruptions.describe(*steps[0])

    @app.get(
        "/api/robustness/preview",
        responses={200: {"content": {"image/png": {}}}, **ERRORS},
        response_class=Response,
        tags=["robustness"],
    )
    def preview(
        image_id: int,
        corruption: str | None = None,
        strength: float | None = Query(None, ge=0, le=1),
        stack: str | None = Query(None, description="Ordered steps 'kind:strength,kind:strength'"),
        max_side: int = 1200,
    ):
        """The model input of a val image under a corruption or a stack (eval resize + corruptions), PNG, not stored."""
        check_image(image_id)
        png = robustness.preview_png(image_id, steps_of(corruption, strength, stack), max_side)
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post(
        "/api/robustness/jobs",
        response_model=schemas.JobCreated,
        status_code=202,
        responses=ERRORS,
        tags=["robustness"],
    )
    def create_robustness_job(req: schemas.RobustnessRequest):
        """Queue one (model, val image, corruption or stack); results are cached by (model, image, canonical stack)."""
        entry = require_model(req.model_id)
        if entry.kind != "detr":
            raise ApiError(400, "unsupported_model", "the robustness mode supports end-to-end DETR models only")
        check_image(req.image_id)
        if req.stack is not None:
            steps = [(st.kind, robustness.strength_key(st.strength)) for st in req.stack]
            try:
                key = (req.model_id, req.image_id, "stack", corruptions.canonical(steps))
            except ValueError as err:
                raise ApiError(400, "invalid_corruption", str(err)) from err
            run_kw = {"stack": (list(key[3]), req.image_id)}
        else:
            s = robustness.strength_key(req.strength)
            steps_of(req.corruption, s, None)
            key = (req.model_id, req.image_id, req.corruption, s)
            run_kw = {"corruption": (req.corruption, s, req.image_id)}

        def work(progress):
            hit = cache.get(key)
            if hit is not None:
                return hit
            img = robustness.load_val_image(req.image_id)
            res = pool.run(req.model_id, img, progress, **run_kw)
            cache.put(key, res)
            return res

        job = jobs.submit(req.model_id, work)
        return {"job_id": job.id, "status": "queued"}

    @app.get("/api/robustness/curve", response_model=schemas.Curve, responses={404: ERRORS[404]}, tags=["robustness"])
    def curve():
        """The robustness curve tables written by scripts/robustness_curve.py (read from reports/tables)."""
        data = robustness.read_curve()
        if data is None:
            raise ApiError(
                404, "curve_missing", f"{robustness.CURVE_PREFIX}.csv not found: run scripts/robustness_curve.py"
            )
        return data

    @app.post(
        "/api/jobs",
        response_model=schemas.JobCreated,
        status_code=202,
        responses=ERRORS,
        openapi_extra=IMAGE_BODY,
        tags=["inference"],
    )
    async def create_job(request: Request, model_id: str = Query(..., description="Model id from /api/models")):
        """Queue one image for inference; poll GET /api/jobs/{job_id} for progress and the result."""
        require_model(model_id)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise UploadError("file_too_large", f"file is {declared} bytes, the limit is {max_bytes}", 413)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > max_bytes:
                raise UploadError("file_too_large", f"file exceeds the limit of {max_bytes} bytes", 413)
        request.state.log_extra = f" model={model_id} bytes={len(body)}"
        image = decode_image(bytes(body), max_bytes, max_pixels)
        job = jobs.submit(model_id, lambda progress: pool.run(model_id, image, progress))
        return {"job_id": job.id, "status": "queued"}

    @app.get("/api/jobs/{job_id}", response_model=schemas.JobStatus, responses={404: ERRORS[404]}, tags=["inference"])
    def get_job(job_id: str):
        job: Job | None = jobs.get(job_id)
        if job is None:
            raise ApiError(404, "unknown_job", f"unknown or expired job {job_id!r}")
        return schemas.JobStatus(
            job_id=job.id,
            model_id=job.model_id,
            status=job.status,
            stage=job.stage,
            progress=round(job.progress, 3),
            queue_position=jobs.queue_position(job),
            error=job.error,
            result=result_payload(job.result) if job.status == "done" else None,
        )

    if static_dir is not None and (Path(static_dir) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        def no_frontend():
            return JSONResponse(
                {
                    "message": "frontend is not built: run `python -m app` (builds it) or see "
                    "app/README.md; the API docs are at /docs"
                }
            )

    return app
