// Mirrors app/schemas.py (the OpenAPI contract at /docs).

export type ClassName = "uninfected" | "infected";
export const CLASSES: ClassName[] = ["uninfected", "infected"];

export interface LocalizedText {
  uk: string;
  en: string;
}

export interface ModelInfo {
  id: string;
  kind: "detr" | "two_stage";
  name: LocalizedText;
  description: LocalizedText;
  classes: ClassName[];
  normalization: { mean: number[]; std: number[] };
  default_thresholds: Record<ClassName, number | null>;
  threshold_source: string | null;
  available: boolean;
  error: string | null;
}

export interface Detection {
  id: number;
  label: ClassName;
  score: number;
  bbox: [number, number, number, number]; // x, y, width, height in original pixels
}

export interface InferenceResult {
  model_id: string;
  width: number;
  height: number;
  device: string;
  elapsed_ms: number;
  nms_iou: number;
  checkpoint: Record<string, unknown>;
  corruption?: (StackDescription & { image_id: number }) | null;
  detections: Detection[];
}

export type JobState = "queued" | "running" | "done" | "error";

export interface JobStatus {
  job_id: string;
  model_id: string;
  status: JobState;
  stage: string;
  progress: number;
  queue_position: number;
  error: { code: string; message: string } | null;
  result: InferenceResult | null;
}

export interface Health {
  status: "ok";
  device: string;
  limits: { max_bytes: number; max_pixels: number; formats: string[] };
}

export type Thresholds = Record<ClassName, number>;

export interface ValImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
}

export interface CorruptionKind {
  kind: string;
  params_at_1: Record<string, unknown>;
}

export interface StackDescription {
  steps: { kind: string; strength: number; params: Record<string, unknown> }[];
  long_side: number;
  identity: boolean;
}

export interface CurveRow {
  model: string;
  corruption: string;
  strength: number;
  long_side: number;
  params: string;
  threshold: number;
  n_images: number;
  AP50: number | null;
  AP50_lo: number | null;
  AP50_hi: number | null;
  precision: number | null;
  precision_lo: number | null;
  precision_hi: number | null;
  recall: number | null;
  recall_lo: number | null;
  recall_hi: number | null;
  f1: number | null;
  f1_lo: number | null;
  f1_hi: number | null;
}

export interface Curve {
  source: string;
  protocol: Record<string, unknown>;
  rows: CurveRow[];
}
