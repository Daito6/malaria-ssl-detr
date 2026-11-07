import { describe, expect, it } from "vitest";
import {
  BASE_INTERVAL_MS,
  FRAMES,
  clampSplit,
  frameInterval,
  halfOf,
  nextFrame,
  splitFromKey,
  splitFromPointer,
  type Playhead,
} from "./curtain";

describe("animation", () => {
  it("has 11 frames from 0 to 1", () => {
    expect(FRAMES).toHaveLength(11);
    expect(FRAMES[0]).toBe(0);
    expect(FRAMES[10]).toBe(1);
  });

  it("ping-pongs without repeating the end frames", () => {
    let p: Playhead = { i: 0, dir: 1 };
    const seen: number[] = [p.i];
    for (let k = 0; k < 24; k++) {
      p = nextFrame(p);
      seen.push(p.i);
    }
    expect(seen.slice(0, 13)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8]);
    expect(seen.slice(19, 22)).toEqual([1, 0, 1]);
    expect(nextFrame({ i: 0, dir: 1 }, 1)).toEqual({ i: 0, dir: 1 });
  });

  it("speed scales the frame interval", () => {
    expect(frameInterval(1)).toBe(BASE_INTERVAL_MS);
    expect(frameInterval(2)).toBe(BASE_INTERVAL_MS / 2);
    expect(frameInterval(0.5)).toBe(BASE_INTERVAL_MS * 2);
  });
});

describe("curtain", () => {
  it("clamps the split to [0, 1]", () => {
    expect(clampSplit(-1)).toBe(0);
    expect(clampSplit(2)).toBe(1);
    expect(clampSplit(Number.NaN)).toBe(0.5);
  });

  it("maps a pointer to the image under pan / zoom", () => {
    // image 1600 px wide drawn at scale 0.5 from x = 100 inside a container at left = 20
    expect(splitFromPointer(20 + 100 + 400, 20, 100, 0.5, 1600)).toBe(0.5);
    expect(splitFromPointer(0, 20, 100, 0.5, 1600)).toBe(0);
    expect(splitFromPointer(5000, 20, 100, 0.5, 1600)).toBe(1);
  });

  it("moves with the keyboard", () => {
    expect(splitFromKey(0.5, "ArrowRight")).toBeCloseTo(0.52);
    expect(splitFromKey(0.5, "ArrowLeft", true)).toBeCloseTo(0.4);
    expect(splitFromKey(0.99, "ArrowRight", true)).toBe(1);
    expect(splitFromKey(0.5, "Home")).toBe(0);
    expect(splitFromKey(0.5, "End")).toBe(1);
    expect(splitFromKey(0.5, "a")).toBeNull();
  });

  it("assigns boxes to the halves", () => {
    expect(halfOf([10, 0, 20, 20], 100)).toBe("left");
    expect(halfOf([100, 0, 20, 20], 100)).toBe("right");
    expect(halfOf([90, 0, 20, 20], 100)).toBe("both");
  });
});
