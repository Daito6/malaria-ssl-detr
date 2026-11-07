import { describe, expect, it } from "vitest";
import {
  BUILTIN_PRESETS,
  DEFAULT_ORDER,
  defaultStack,
  exportPresets,
  move,
  parsePresets,
  sanitizeItems,
  stepsKey,
  toSteps,
  update,
} from "./stack";

describe("canonical stack", () => {
  it("is empty (the original) when nothing is enabled or every strength is 0", () => {
    expect(toSteps(defaultStack())).toEqual([]);
    const allZero = defaultStack().map((it) => ({ ...it, enabled: true, strength: 0 }));
    expect(toSteps(allZero)).toEqual([]);
    expect(stepsKey(toSteps(allZero))).toBe("");
  });

  it("keeps the editor order and maps scale to its direction", () => {
    let items = update(defaultStack(), "jpeg", { enabled: true, strength: 0.5 });
    items = update(items, "scale", { enabled: true, strength: 1, direction: "up" });
    items = update(items, "blur", { enabled: true, strength: 0 });
    expect(toSteps(items)).toEqual([
      { kind: "scale_up", strength: 1 },
      { kind: "jpeg", strength: 0.5 },
    ]);
    const moved = move(items, 0, items.length - 1); // scale last
    expect(stepsKey(toSteps(moved))).toBe("jpeg:0.5,scale_up:1");
  });

  it("is deterministic and scales every strength by the animation factor", () => {
    const items = BUILTIN_PRESETS[0].items;
    expect(toSteps(items)).toEqual(toSteps(structuredClone(items)));
    expect(toSteps(items, 0)).toEqual([]);
    expect(toSteps(items, 0.5)).toEqual(toSteps(items).map((s) => ({ ...s, strength: s.strength / 2 })));
  });
});

describe("move", () => {
  it("reorders and ignores out-of-range indices", () => {
    expect(move(["a", "b", "c"], 2, 0)).toEqual(["c", "a", "b"]);
    expect(move(["a", "b", "c"], 0, 5)).toEqual(["a", "b", "c"]);
  });
});

describe("presets", () => {
  it("built-ins cover every kind once in the default order", () => {
    for (const p of BUILTIN_PRESETS) expect(p.items.map((i) => i.kind)).toEqual([...DEFAULT_ORDER]);
  });

  it("round-trip through export / import (built-ins are not exported)", () => {
    const mine = { id: "x", name: "мій", items: move(BUILTIN_PRESETS[1].items, 4, 0) };
    const text = exportPresets([...BUILTIN_PRESETS, mine]);
    expect(parsePresets(text)).toEqual([mine]);
  });

  it("rejects foreign or broken files", () => {
    expect(() => parsePresets("{")).toThrow("not_json");
    expect(() => parsePresets(JSON.stringify({ format: "x", version: 1, presets: [] }))).toThrow(
      "wrong_format",
    );
    const dup = {
      format: "smearscan-presets",
      version: 1,
      presets: [
        {
          name: "a",
          items: [
            { kind: "blur", strength: 1 },
            { kind: "blur", strength: 1 },
          ],
        },
      ],
    };
    expect(() => parsePresets(JSON.stringify(dup))).toThrow("bad_preset_0");
  });

  it("sanitizes strengths to the 0.05 grid and fills missing kinds", () => {
    const items = sanitizeItems([
      { kind: "jpeg", strength: 0.333, enabled: true },
      { kind: "blur", strength: 7 },
    ]);
    expect(items?.slice(0, 2)).toEqual([
      { kind: "jpeg", enabled: true, strength: 0.35, direction: "down" },
      { kind: "blur", enabled: false, strength: 1, direction: "down" },
    ]);
    expect(items?.map((i) => i.kind).sort()).toEqual([...DEFAULT_ORDER].sort());
    expect(sanitizeItems([{ kind: "nope", strength: 0 }])).toBeNull();
  });
});
