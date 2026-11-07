import type { Detection } from "../types";

export const MATCH_IOU = 0.5;

export function iou(a: Detection["bbox"], b: Detection["bbox"]): number {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

export type DisagreementKind = "only_a" | "only_b" | "class";

export interface Disagreement {
  kind: DisagreementKind;
  a: Detection | null;
  b: Detection | null;
}

export interface MatchResult {
  pairs: [Detection, Detection][];
  disagreements: Disagreement[];
  /** Ids in A / B that take part in a disagreement (for highlighting). */
  markedA: Set<number>;
  markedB: Set<number>;
}

/**
 * Class-agnostic greedy matching of two models' boxes (each already filtered by its own thresholds): A's boxes by
 * decreasing score take the unmatched B box with the highest IoU >= `thr`. Disagreements: a cell found by one model
 * only, or matched with a different class.
 */
export function matchDetections(a: Detection[], b: Detection[], thr = MATCH_IOU): MatchResult {
  const usedB = new Set<number>();
  const pairs: [Detection, Detection][] = [];
  const disagreements: Disagreement[] = [];
  for (const da of [...a].sort((x, y) => y.score - x.score || x.id - y.id)) {
    let best: Detection | null = null;
    let bestIou = 0;
    for (const db of b) {
      if (usedB.has(db.id)) continue;
      const v = iou(da.bbox, db.bbox);
      if (v < thr) continue;
      if (best === null || v > bestIou || (v === bestIou && db.score > best.score)) {
        best = db;
        bestIou = v;
      }
    }
    if (best) {
      usedB.add(best.id);
      pairs.push([da, best]);
      if (da.label !== best.label) disagreements.push({ kind: "class", a: da, b: best });
    } else disagreements.push({ kind: "only_a", a: da, b: null });
  }
  for (const db of b) if (!usedB.has(db.id)) disagreements.push({ kind: "only_b", a: null, b: db });
  return {
    pairs,
    disagreements,
    markedA: new Set(disagreements.flatMap((d) => (d.a ? [d.a.id] : []))),
    markedB: new Set(disagreements.flatMap((d) => (d.b ? [d.b.id] : []))),
  };
}
