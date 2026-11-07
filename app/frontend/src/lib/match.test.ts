import { describe, expect, it } from "vitest";
import type { Detection } from "../types";
import { applyThresholds } from "./filter";
import { iou, matchDetections } from "./match";

const d = (id: number, label: Detection["label"], score: number, x: number, y = 0): Detection => ({
  id,
  label,
  score,
  bbox: [x, y, 50, 50],
});

describe("iou", () => {
  it("is 1 for identical boxes and 0 for disjoint ones", () => {
    expect(iou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
    expect(iou([0, 0, 10, 10], [20, 0, 10, 10])).toBe(0);
    expect(iou([0, 0, 10, 10], [5, 0, 10, 10])).toBeCloseTo(50 / 150);
  });
});

describe("matchDetections", () => {
  const a = [d(0, "uninfected", 0.99, 0), d(1, "infected", 0.95, 100), d(2, "uninfected", 0.9, 200)];
  const b = [d(0, "uninfected", 0.98, 2), d(1, "uninfected", 0.97, 102), d(2, "uninfected", 0.9, 400)];
  const m = matchDetections(a, b);

  it("pairs overlapping boxes regardless of class", () => {
    expect(m.pairs.map(([x, y]) => [x.id, y.id])).toEqual([
      [0, 0],
      [1, 1],
    ]);
  });

  it("reports cells found by one model only and class disagreements", () => {
    expect(m.disagreements.map((x) => [x.kind, x.a?.id ?? null, x.b?.id ?? null])).toEqual([
      ["class", 1, 1],
      ["only_a", 2, null],
      ["only_b", null, 2],
    ]);
    expect([...m.markedA].sort()).toEqual([1, 2]);
    expect([...m.markedB].sort()).toEqual([1, 2]);
  });

  it("uses each box at most once", () => {
    const one = [d(0, "uninfected", 0.9, 0)];
    const two = [d(0, "uninfected", 0.9, 0), d(1, "uninfected", 0.8, 1)];
    const r = matchDetections(two, one);
    expect(r.pairs).toHaveLength(1);
    expect(r.disagreements).toEqual([{ kind: "only_a", a: two[1], b: null }]);
  });

  it("identical outputs agree completely; respects thresholds applied before matching", () => {
    expect(matchDetections(a, a).disagreements).toHaveLength(0);
    const thrA = applyThresholds(a, { uninfected: 0.95, infected: 0.5 });
    expect(matchDetections(thrA, b).disagreements.filter((x) => x.kind === "only_a")).toHaveLength(0);
  });
});
