import type { Step } from "./lib/stack";
import type {
  CorruptionKind,
  Curve,
  Health,
  InferenceResult,
  JobStatus,
  ModelInfo,
  StackDescription,
  ValImage,
} from "./types";

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function json<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T;
  let code = "http_error";
  let message = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body?.error) ({ code, message } = body.error);
  } catch {
    /* not JSON */
  }
  throw new ApiError(code, message, res.status);
}

export const getHealth = () => fetch("/api/health").then((r) => json<Health>(r));
export const getModels = () => fetch("/api/models").then((r) => json<ModelInfo[]>(r));

/** Uploads the raw file (not multipart) and returns the job id. */
export async function submitJob(file: File, modelId: string): Promise<string> {
  const res = await fetch(`/api/jobs?model_id=${encodeURIComponent(modelId)}`, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
  return (await json<{ job_id: string }>(res)).job_id;
}

export const getJob = (id: string) => fetch(`/api/jobs/${id}`).then((r) => json<JobStatus>(r));

/** Submit + poll until done; inference runs on the server, the UI only polls. */
export async function runJob(
  file: File,
  modelId: string,
  onProgress: (s: JobStatus) => void,
  signal?: AbortSignal,
): Promise<InferenceResult> {
  return pollJob(await submitJob(file, modelId), onProgress, signal);
}

export const getValImages = () => fetch("/api/val-images").then((r) => json<ValImage[]>(r));
export const getCorruptions = () =>
  fetch("/api/robustness/corruptions").then((r) => json<CorruptionKind[]>(r));
/** Real parameters of every step of a stack ("kind:s,kind:s"; "" = the original) and the final long side. */
export const describeStack = (stack: string) =>
  fetch(`/api/robustness/describe?stack=${encodeURIComponent(stack)}`).then((r) => json<StackDescription>(r));
export const previewUrl = (imageId: number, stack: string) =>
  `/api/robustness/preview?image_id=${imageId}&stack=${encodeURIComponent(stack)}`;
export const getCurve = () => fetch("/api/robustness/curve").then((r) => json<Curve>(r));

export async function runRobustnessJob(
  req: { model_id: string; image_id: number; stack: Step[] },
  onProgress: (s: JobStatus) => void,
): Promise<InferenceResult> {
  const res = await fetch("/api/robustness/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return pollJob((await json<{ job_id: string }>(res)).job_id, onProgress);
}

async function pollJob(
  id: string,
  onProgress: (s: JobStatus) => void,
  signal?: AbortSignal,
): Promise<InferenceResult> {
  for (;;) {
    if (signal?.aborted) throw new ApiError("aborted", "aborted", 0);
    const s = await getJob(id);
    onProgress(s);
    if (s.status === "done" && s.result) return s.result;
    if (s.status === "error")
      throw new ApiError(s.error?.code ?? "internal_error", s.error?.message ?? "", 500);
    await new Promise((r) => setTimeout(r, 250));
  }
}
