/** UI logic of the before / after curtain and its strength animation (pure, unit-tested). */

/** Animation frames: strength (single corruption) or the factor applied to every stack strength. */
export const FRAMES: readonly number[] = Array.from({ length: 11 }, (_, i) => i / 10);
export const SPEEDS = [0.5, 1, 2] as const;
export type Speed = (typeof SPEEDS)[number];
export const BASE_INTERVAL_MS = 700;

export const frameInterval = (speed: number) => Math.round(BASE_INTERVAL_MS / speed);

export interface Playhead {
  i: number;
  dir: 1 | -1;
}

/** Ping-pong over the frames: 0 → 1 → 0 → … without repeating the end frames. */
export function nextFrame(p: Playhead, n: number = FRAMES.length): Playhead {
  if (n <= 1) return { i: 0, dir: 1 };
  const i = p.i + p.dir;
  if (i >= n) return { i: n - 2, dir: -1 };
  if (i < 0) return { i: 1, dir: 1 };
  return { i, dir: p.dir };
}

export const clampSplit = (v: number) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.5);

/** Split position (fraction of the image width) under a pointer, given the viewer's pan / zoom. */
export function splitFromPointer(
  clientX: number,
  rectLeft: number,
  viewX: number,
  scale: number,
  width: number,
) {
  return clampSplit((clientX - rectLeft - viewX) / scale / width);
}

/** Keyboard on the curtain handle: arrows move 2 % (10 % with Shift), Home / End go to the edges. */
export function splitFromKey(split: number, key: string, shift = false): number | null {
  const step = shift ? 0.1 : 0.02;
  if (key === "ArrowLeft") return clampSplit(split - step);
  if (key === "ArrowRight") return clampSplit(split + step);
  if (key === "Home") return 0;
  if (key === "End") return 1;
  return null;
}

/** Which half of the curtain shows (part of) a box: the left (original) or the right (corrupted) one. */
export function halfOf(bbox: readonly number[], splitPx: number): "left" | "right" | "both" {
  const [x, , w] = bbox;
  if (x + w <= splitPx) return "left";
  if (x >= splitPx) return "right";
  return "both";
}
