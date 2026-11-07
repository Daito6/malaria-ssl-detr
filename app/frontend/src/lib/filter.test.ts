import { describe, expect, it } from "vitest";
import type { Detection, ModelInfo } from "../types";
import {
  applyClassFilter,
  applyThresholds,
  defaultThresholds,
  histogram,
  sortDetections,
  summarize,
  UI_FALLBACK_THRESHOLD,
} from "./filter";

// An NMS output as the backend returns it: sorted by score, no two boxes overlapping with IoU > 0.5.
const dets: Detection[] = [
  { id: 0, label: "uninfected", score: 0.99, bbox: [0, 0, 50, 50] },
  { id: 1, label: "infected", score: 0.95, bbox: [100, 0, 50, 50] },
  { id: 2, label: "uninfected", score: 0.9, bbox: [200, 0, 50, 50] },
  { id: 3, label: "infected", score: 0.6, bbox: [300, 0, 50, 50] },
  { id: 4, label: "uninfected", score: 0.2, bbox: [400, 0, 50, 50] },
];

function iou(a: Detection["bbox"], b: Detection["bbox"]): number {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter);
}

const model = (u: number | null, i: number | null): ModelInfo => ({
  id: "m",
  kind: "detr",
  name: { uk: "m", en: "m" },
  description: { uk: "", en: "" },
  classes: ["uninfected", "infected"],
  normalization: { mean: [0, 0, 0], std: [1, 1, 1] },
  default_thresholds: { uninfected: u, infected: i },
  threshold_source: null,
  available: true,
  error: null,
});

describe("applyThresholds", () => {
  it("keeps boxes at or above their class threshold", () => {
    const out = applyThresholds(dets, { uninfected: 0.9, infected: 0.7 });
    expect(out.map((d) => d.id)).toEqual([0, 1, 2]);
  });

  it("never adds a box or a duplicate when the threshold moves (works on the NMS result)", () => {
    let prev = new Set(dets.map((d) => d.id));
    for (let t = 0; t <= 1.0001; t += 0.05) {
      const out = applyThresholds(dets, { uninfected: t, infected: t });
      const ids = new Set(out.map((d) => d.id));
      expect(ids.size).toBe(out.length); // no box twice
      for (const id of ids) expect(prev.has(id)).toBe(true); // a subset of the lower threshold's result
      for (let a = 0; a < out.length; a++)
        for (let b = a + 1; b < out.length; b++)
          expect(iou(out[a].bbox, out[b].bbox)).toBeLessThanOrEqual(0.5);
      prev = ids;
    }
  });

  it("does not mutate the input", () => {
    const copy = structuredClone(dets);
    applyThresholds(dets, { uninfected: 0.5, infected: 0.5 });
    expect(dets).toEqual(copy);
  });
});

describe("defaults", () => {
  it("uses val-selected thresholds and marks the missing ones", () => {
    const d = defaultThresholds(model(null, 0.937));
    expect(d.thresholds).toEqual({ uninfected: UI_FALLBACK_THRESHOLD, infected: 0.937 });
    expect(d.fromVal).toEqual({ uninfected: false, infected: true });
  });
});

describe("summary", () => {
  it("parasitemia = infected / (infected + uninfected)", () => {
    const s = summarize(applyThresholds(dets, { uninfected: 0.5, infected: 0.5 }));
    expect(s).toEqual({ uninfected: 2, infected: 2, total: 4, parasitemia: 0.5 });
  });

  it("is null without cells", () => {
    expect(summarize([]).parasitemia).toBeNull();
  });

  it("class filter does not change the summary input", () => {
    const above = applyThresholds(dets, { uninfected: 0, infected: 0 });
    expect(applyClassFilter(above, new Set(["infected"])).map((d) => d.id)).toEqual([1, 3]);
    expect(summarize(above).total).toBe(5);
  });
});

describe("histogram and sort", () => {
  it("bins scores per class, 1.0 goes to the last bin", () => {
    const h = histogram(
      [...dets, { id: 9, label: "infected", score: 1, bbox: [0, 0, 1, 1] }],
      "infected",
      20,
    );
    expect(h.reduce((a, b) => a + b, 0)).toBe(3);
    expect(h[19]).toBe(2);
    expect(h[12]).toBe(1);
  });

  it("sorts by score with id as tie-break", () => {
    expect(sortDetections(dets, "score", false).map((d) => d.id)).toEqual([4, 3, 2, 1, 0]);
    expect(sortDetections(dets, "label", false)[0].label).toBe("infected");
  });
});
