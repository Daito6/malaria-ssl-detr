/** Corruption stacks of the robustness mode: the editor state, its canonical form (the request and cache key) and
 * presets. A stack is a demonstration of combined corruptions; the robustness curve stays single-corruption. */

export type StackKind = "scale" | "stain" | "blur" | "brightness_contrast" | "jpeg";
export type ScaleDirection = "down" | "up";

export interface StackItem {
  kind: StackKind;
  enabled: boolean;
  strength: number; // 0..1
  direction: ScaleDirection; // used by "scale" only
}

/** A request step as the server takes it (scale is scale_down / scale_up). */
export interface Step {
  kind: string;
  strength: number;
}

export interface Preset {
  id: string;
  name: string | { uk: string; en: string };
  items: StackItem[];
  builtin?: boolean;
}

/** Default order: the resize first (as the single scale corruption), then stain -> optics -> illumination ->
 * compression last (as a scanned smear would get them). The server applies the steps in the order sent. */
export const DEFAULT_ORDER: readonly StackKind[] = ["scale", "stain", "blur", "brightness_contrast", "jpeg"];
export const STRENGTH_STEP = 0.05;
export const PRESET_FORMAT = "smearscan-presets";
export const PRESET_VERSION = 1;

const round3 = (v: number) => Math.round(v * 1000) / 1000;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export function defaultStack(): StackItem[] {
  return DEFAULT_ORDER.map((kind) => ({ kind, enabled: false, strength: 0.5, direction: "down" }));
}

/** Enabled steps with strength > 0, in order: the canonical stack (empty = the original input). */
export function toSteps(items: readonly StackItem[], factor = 1): Step[] {
  return items
    .filter((it) => it.enabled)
    .map((it) => ({
      kind: it.kind === "scale" ? `scale_${it.direction}` : it.kind,
      strength: round3(clamp01(it.strength * factor)),
    }))
    .filter((s) => s.strength > 0);
}

/** Query / cache form of a canonical stack, e.g. "blur:0.5,jpeg:0.25"; "" is the original. */
export const stepsKey = (steps: readonly Step[]) => steps.map((s) => `${s.kind}:${s.strength}`).join(",");

export function move<T>(xs: readonly T[], from: number, to: number): T[] {
  if (from === to || from < 0 || to < 0 || from >= xs.length || to >= xs.length) return [...xs];
  const out = [...xs];
  const [x] = out.splice(from, 1);
  out.splice(to, 0, x);
  return out;
}

export function update(items: readonly StackItem[], kind: StackKind, patch: Partial<StackItem>): StackItem[] {
  return items.map((it) => (it.kind === kind ? { ...it, ...patch, kind } : it));
}

/** Validates an untrusted list of items (preset import / storage): each known kind once, strengths on the
 * 0.05 grid in [0, 1]; missing kinds are appended disabled in the default order. Null if unusable. */
export function sanitizeItems(raw: unknown): StackItem[] | null {
  if (!Array.isArray(raw)) return null;
  const seen = new Set<StackKind>();
  const out: StackItem[] = [];
  for (const r of raw) {
    if (typeof r !== "object" || r === null) return null;
    const o = r as Record<string, unknown>;
    const kind = o.kind as StackKind;
    if (!DEFAULT_ORDER.includes(kind) || seen.has(kind)) return null;
    if (typeof o.strength !== "number" || !Number.isFinite(o.strength)) return null;
    seen.add(kind);
    out.push({
      kind,
      enabled: o.enabled === true,
      strength: round3(Math.round(clamp01(o.strength) / STRENGTH_STEP) * STRENGTH_STEP),
      direction: o.direction === "up" ? "up" : "down",
    });
  }
  for (const kind of DEFAULT_ORDER)
    if (!seen.has(kind)) out.push({ kind, enabled: false, strength: 0.5, direction: "down" });
  return out;
}

function preset(
  id: string,
  uk: string,
  en: string,
  on: Partial<Record<StackKind, Partial<StackItem>>>,
): Preset {
  const items = defaultStack().map((it) => (on[it.kind] ? { ...it, enabled: true, ...on[it.kind] } : it));
  return { id, name: { uk, en }, items, builtin: true };
}

/** Illustrative combinations (strengths are not estimated from data). */
export const BUILTIN_PRESETS: readonly Preset[] = [
  preset("old-microscope", "Старий мікроскоп", "Old microscope", {
    stain: { strength: 0.3 },
    blur: { strength: 0.5 },
    brightness_contrast: { strength: 0.6 },
  }),
  preset("compressed-scan", "Стиснутий скан", "Compressed scan", {
    scale: { strength: 0.5, direction: "down" },
    jpeg: { strength: 0.8 },
  }),
  preset("other-stain", "Інший барвник", "Different stain", { stain: { strength: 1 } }),
  preset("defocus", "Розфокус", "Defocus", { blur: { strength: 1 } }),
];

export function exportPresets(presets: readonly Preset[]): string {
  const user = presets.filter((p) => !p.builtin).map(({ id, name, items }) => ({ id, name, items }));
  return JSON.stringify({ format: PRESET_FORMAT, version: PRESET_VERSION, presets: user }, null, 2);
}

/** Parses an exported file (or the stored list); throws Error with a short reason. */
export function parsePresets(text: string): Preset[] {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("not_json");
  }
  const o = data as Record<string, unknown>;
  if (o?.format !== PRESET_FORMAT || o.version !== PRESET_VERSION || !Array.isArray(o.presets))
    throw new Error("wrong_format");
  return o.presets.map((p: unknown, i: number) => {
    const q = p as Record<string, unknown>;
    const items = sanitizeItems(q?.items);
    const name = typeof q?.name === "string" ? q.name.slice(0, 80) : "";
    if (!items || !name.trim()) throw new Error(`bad_preset_${i}`);
    return { id: typeof q.id === "string" ? q.id.slice(0, 40) : `p${i}`, name, items };
  });
}

const STORAGE_KEY = "smearscan.presets.v1";

export function loadStoredPresets(): Preset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? parsePresets(raw) : [];
  } catch {
    return []; // storage unavailable or corrupted: start empty
  }
}

export function storePresets(presets: readonly Preset[]): boolean {
  try {
    localStorage.setItem(STORAGE_KEY, exportPresets(presets));
    return true;
  } catch {
    return false;
  }
}
