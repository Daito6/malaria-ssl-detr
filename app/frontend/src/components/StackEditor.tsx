import { useRef, useState, type DragEvent } from "react";
import { useI18n } from "../i18n";
import {
  BUILTIN_PRESETS,
  STRENGTH_STEP,
  defaultStack,
  exportPresets,
  loadStoredPresets,
  move,
  parsePresets,
  storePresets,
  update,
  type Preset,
  type StackItem,
} from "../lib/stack";
import { Button } from "./ui";

interface Props {
  items: StackItem[];
  onChange: (items: StackItem[]) => void;
}

const kindLabel = (it: StackItem) => (it.kind === "scale" ? `scale_${it.direction}` : it.kind);

/** Ordered corruption steps: a switch and a strength per step, drag (or ↑ / ↓) to reorder; presets. */
export function StackEditor({ items, onChange }: Props) {
  const { t, lang, maybe } = useI18n();
  const [presets, setPresets] = useState<Preset[]>(loadStoredPresets);
  const [presetId, setPresetId] = useState("");
  const [name, setName] = useState("");
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);
  const [drag, setDrag] = useState<{ from: number; over: number } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const all = [...BUILTIN_PRESETS, ...presets];
  const nameOf = (p: Preset) => (typeof p.name === "string" ? p.name : p.name[lang]);
  const persist = (next: Preset[]) => {
    setPresets(next);
    if (!storePresets(next)) setStatus({ ok: false, text: t("stack.storageFailed") });
  };

  const choose = (id: string) => {
    setPresetId(id);
    const p = all.find((x) => x.id === id);
    if (p) onChange(p.items.map((it) => ({ ...it })));
  };
  const save = () => {
    const n = name.trim();
    if (!n) return;
    const id = `u${Date.now().toString(36)}`;
    persist([
      ...presets.filter((p) => nameOf(p) !== n),
      { id, name: n, items: items.map((it) => ({ ...it })) },
    ]);
    setPresetId(id);
    setName("");
    setStatus({ ok: true, text: t("stack.saved", { name: n }) });
  };
  const remove = () => {
    persist(presets.filter((p) => p.id !== presetId));
    setPresetId("");
  };
  const download = () => {
    const url = URL.createObjectURL(new Blob([exportPresets(presets)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "smearscan-presets.json";
    a.click();
    URL.revokeObjectURL(url);
  };
  const upload = async (f: File | undefined) => {
    if (!f) return;
    try {
      if (f.size > 256 * 1024) throw new Error("too_large");
      const imported = parsePresets(await f.text());
      const names = new Set(imported.map(nameOf));
      persist([...presets.filter((p) => !names.has(nameOf(p))), ...imported]);
      setStatus({ ok: true, text: t("stack.imported", { n: imported.length }) });
    } catch (e) {
      setStatus({ ok: false, text: t("stack.importFailed", { reason: (e as Error).message }) });
    }
  };

  const onDrop = (e: DragEvent, to: number) => {
    e.preventDefault();
    if (drag) onChange(move(items, drag.from, to));
    setDrag(null);
  };
  const isUser = presets.some((p) => p.id === presetId);
  // position in the applied order: only enabled steps with strength > 0 count (as toSteps)
  const rank = new Map(
    items.filter((it) => it.enabled && it.strength > 0).map((it, i) => [it.kind, i + 1] as const),
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex min-w-0 flex-1 flex-col gap-1 text-sm sm:max-w-64">
          <span className="font-semibold">{t("stack.preset")}</span>
          <select
            value={presetId}
            onChange={(e) => choose(e.target.value)}
            className="w-full min-w-0 rounded-lg border border-line bg-panel px-2 py-1.5"
          >
            <option value="">{t("stack.presetPick")}</option>
            <optgroup label={t("stack.builtin")}>
              {BUILTIN_PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {nameOf(p)}
                </option>
              ))}
            </optgroup>
            {presets.length > 0 && (
              <optgroup label={t("stack.mine")}>
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {nameOf(p)}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
        </label>
        {isUser && <Button onClick={remove}>{t("stack.delete")}</Button>}
        <Button
          onClick={() => {
            onChange(defaultStack());
            setPresetId("");
          }}
        >
          {t("stack.reset")}
        </Button>
      </div>

      <ol className="flex flex-col gap-1.5" aria-label={t("stack.order")}>
        {items.map((it, i) => {
          const id = `stack-${it.kind}`;
          const label = maybe(`stack.kind.${it.kind}`) ?? kindLabel(it);
          return (
            <li
              key={it.kind}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.effectAllowed = "move";
                setDrag({ from: i, over: i });
              }}
              onDragOver={(e) => {
                e.preventDefault();
                if (drag && drag.over !== i) setDrag({ ...drag, over: i });
              }}
              onDrop={(e) => onDrop(e, i)}
              onDragEnd={() => setDrag(null)}
              className={`grid grid-cols-[auto_auto_minmax(0,1fr)_auto] items-center gap-x-2 gap-y-1 rounded-lg border bg-panel px-2 py-1.5 text-sm ${
                drag?.over === i && drag.from !== i ? "border-accent border-dashed" : "border-line"
              } ${it.enabled ? "" : "opacity-70"}`}
            >
              <span className="cursor-grab text-muted select-none" aria-hidden="true" title={t("stack.drag")}>
                ⠿
              </span>
              <span className="tabular w-5 text-center text-xs font-semibold text-ink-2" aria-hidden="true">
                {rank.get(it.kind) ?? "–"}
              </span>
              <label className="flex min-w-0 items-center gap-2">
                <input
                  type="checkbox"
                  checked={it.enabled}
                  onChange={(e) => onChange(update(items, it.kind, { enabled: e.target.checked }))}
                />
                <span className="truncate font-medium">{label}</span>
              </label>
              <span className="flex gap-0.5">
                {(
                  [
                    ["↑", i - 1, t("stack.up", { name: label })],
                    ["↓", i + 1, t("stack.down", { name: label })],
                  ] as const
                ).map(([sym, to, title]) => (
                  <button
                    key={sym}
                    type="button"
                    title={title}
                    aria-label={title}
                    disabled={to < 0 || to >= items.length}
                    onClick={() => onChange(move(items, i, to))}
                    className="min-w-7 rounded-md border border-line px-1.5 py-0.5 hover:bg-panel-2 disabled:opacity-30"
                  >
                    {sym}
                  </button>
                ))}
              </span>
              <div className="col-span-4 flex min-w-0 items-center gap-2 pl-9">
                {it.kind === "scale" && (
                  <select
                    value={it.direction}
                    aria-label={t("stack.direction")}
                    disabled={!it.enabled}
                    onChange={(e) =>
                      onChange(
                        update(items, "scale", { direction: e.target.value as StackItem["direction"] }),
                      )
                    }
                    className="rounded-md border border-line bg-panel px-1 py-0.5 text-xs"
                  >
                    <option value="down">↓ ×0.8</option>
                    <option value="up">↑ ×1.27</option>
                  </select>
                )}
                <input
                  id={id}
                  type="range"
                  min={0}
                  max={1}
                  step={STRENGTH_STEP}
                  value={it.strength}
                  disabled={!it.enabled}
                  aria-label={`${t("rob.strength")}: ${label}`}
                  onChange={(e) => onChange(update(items, it.kind, { strength: +e.target.value }))}
                  className="min-w-0 flex-1"
                />
                <span className="tabular w-9 text-right text-xs">{it.strength.toFixed(2)}</span>
              </div>
            </li>
          );
        })}
      </ol>
      <p className="text-xs text-muted">{t("stack.orderHelp")}</p>

      <div className="flex flex-wrap items-center gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && save()}
          maxLength={80}
          placeholder={t("stack.namePlaceholder")}
          aria-label={t("stack.namePlaceholder")}
          className="min-w-0 flex-1 rounded-lg border border-line bg-panel px-2 py-1 text-sm"
        />
        <Button onClick={save} disabled={!name.trim()}>
          {t("stack.save")}
        </Button>
        <Button onClick={download} disabled={presets.length === 0}>
          {t("stack.export")}
        </Button>
        <Button onClick={() => fileRef.current?.click()}>{t("stack.import")}</Button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={(e) => {
            void upload(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      <p
        role="status"
        aria-live="polite"
        className={`text-xs ${status?.ok === false ? "text-danger-ink" : "text-ink-2"}`}
      >
        {status?.text}
      </p>
    </div>
  );
}
