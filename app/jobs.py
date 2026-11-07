"""In-memory job queue: one worker thread runs inferences one at a time, clients poll the progress.

The HTTP request returns at once with a job id, so the UI never waits on a blocking call. Images live only in
memory (inside the job's work closure) while a job is queued or running and are dropped when it finishes; only
results are kept (the most recent `max_kept` finished jobs).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("smearscan.jobs")
Work = Callable[[Callable[[str, float], None]], Any]


@dataclass
class Job:
    id: str
    model_id: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    error: dict[str, str] | None = None
    result: Any = None
    created: float = field(default_factory=time.time)
    work: Work | None = None  # holds the image until the job finishes


class JobManager:
    def __init__(self, error_info: Callable[[Exception], dict[str, str]], max_kept: int = 100):
        self._error_info = error_info
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smearscan-infer")
        self.max_kept = max_kept

    def submit(self, model_id: str, work: Work) -> Job:
        """Queue `work(progress)`; its return value becomes the job result."""
        job = Job(id=uuid.uuid4().hex, model_id=model_id, work=work)
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        self._pool.submit(self._work, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "queued" and j.created < job.created)

    def _evict(self) -> None:
        finished = [k for k, j in self._jobs.items() if j.status in ("done", "error")]
        for k in finished[: max(0, len(finished) - self.max_kept)]:
            del self._jobs[k]

    def _work(self, job: Job) -> None:
        job.status, job.stage = "running", "start"

        def progress(stage: str, frac: float) -> None:
            job.stage, job.progress = stage, max(job.progress, min(frac, 0.99))

        t0 = time.perf_counter()
        try:
            job.result = job.work(progress)
            job.status, job.stage, job.progress = "done", "done", 1.0
        except Exception as err:  # noqa: BLE001 - reported to the client, logged here
            job.error = self._error_info(err)
            job.status, job.stage = "error", "error"
            if job.error.get("code") == "internal_error":
                log.exception("job %s (%s) failed", job.id, job.model_id)
        finally:
            job.work = None
            log.info(
                "job %s model=%s status=%s %.0f ms", job.id, job.model_id, job.status, (time.perf_counter() - t0) * 1e3
            )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
