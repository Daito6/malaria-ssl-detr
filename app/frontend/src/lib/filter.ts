import type { ClassName, Detection, ModelInfo, Thresholds } from "../types";

/** Used only when the model has no val-selected threshold for a class (the UI says so next to the slider). */
export const UI_FALLBACK_THRESHOLD = 0.5;

export function defaultThresholds(model: ModelInfo): {
  thresholds: Thresholds;
  fromVal: Record<ClassName, boolean>;
} {
  const pick = (c: ClassName) => model.default_thresholds[c];
  return {
    thresholds: {
      uninfected: pick("uninfected") ?? UI_FALLBACK_THRESHOLD,
      infected: pick("infected") ?? UI_FALLBACK_THRESHOLD,
    },
    fromVal: { uninfected: pick("uninfected") != null, infected: pick("infected") != null },
  };
}

/**
 * Detections at or above their class threshold. The backend already ran class-agnostic NMS, so this filter only
 * removes boxes: it can never introduce a duplicate (same rule as app/tests/test_nms.py `_visible`).
 */
export function applyThresholds(dets: Detection[], thr: Thresholds): Detection[] {
  return dets.filter((d) => d.score >= thr[d.label]);
}

export function applyClassFilter(dets: Detection[], shown: ReadonlySet<ClassName>): Detection[] {
  return dets.filter((d) => shown.has(d.label));
}

export interface Summary {
  uninfected: number;
  infected: number;
  total: number;
  /** infected / (infected + uninfected); null when there are no cells */
  parasitemia: number | null;
}

export function summarize(dets: Detection[]): Summary {
  const infected = dets.filter((d) => d.label === "infected").length;
  const uninfected = dets.length - infected;
  return {
    uninfected,
    infected,
    total: dets.length,
    parasitemia: dets.length ? infected / dets.length : null,
  };
}

/** Score histogram of one class over [0, 1] (last bin includes 1.0). */
export function histogram(dets: Detection[], label: ClassName, bins = 20): number[] {
  const counts = new Array<number>(bins).fill(0);
  for (const d of dets) {
    if (d.label !== label) continue;
    counts[Math.min(bins - 1, Math.floor(d.score * bins))] += 1;
  }
  return counts;
}

export type SortKey = "id" | "label" | "score";

export function sortDetections(dets: Detection[], key: SortKey, desc: boolean): Detection[] {
  const sign = desc ? -1 : 1;
  return [...dets].sort((a, b) => {
    const v = key === "label" ? a.label.localeCompare(b.label) : (a[key] as number) - (b[key] as number);
    return sign * (v || a.id - b.id);
  });
}
