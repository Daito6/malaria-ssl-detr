import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, describeStack, getCorruptions, getValImages, previewUrl, runRobustnessJob } from "../api";
import { useI18n } from "../i18n";
import {
  FRAMES,
  SPEEDS,
  frameInterval,
  halfOf,
  nextFrame,
  splitFromKey,
  type Playhead,
  type Speed,
} from "../lib/curtain";
import { applyThresholds, defaultThresholds, summarize, type Summary } from "../lib/filter";
import { defaultStack, stepsKey, toSteps, type StackItem, type Step } from "../lib/stack";
import type { CorruptionKind, InferenceResult, ModelInfo, StackDescription, ValImage } from "../types";
import { StackEditor } from "./StackEditor";
import { Button, Panel } from "./ui";
import { Viewer, type View, type ViewState } from "./Viewer";

const STEP = 0.05;
const DEBOUNCE_MS = 400;
const MAX_LONG_SIDE = 1016; // scale_up at strength 1: the widest model input, for the proportional preview

type Mode = "single" | "stack";
type Layout = "columns" | "curtain";

const scaleSteps = (steps: Step[], f: number): Step[] =>
  steps
    .map((s) => ({ ...s, strength: Math.round(s.strength * f * 1000) / 1000 }))
    .filter((s) => s.strength > 0);

/** Result-cache key on the client: (model, image, canonical stack); "" is the original. */
const rkey = (mid: string, img: number, sk: string) => `${mid}|${img}|${sk}`;

function Delta({ v, digits = 0, unit = "" }: { v: number | null; digits?: number; unit?: string }) {
  if (v == null) return <span className="text-muted">—</span>;
  const r = Number(v.toFixed(digits));
  if (r === 0) return <span className="text-muted">= 0{unit}</span>;
  return (
    <span className="rounded bg-warn-bg px-1 font-semibold text-warn-ink">
      {r > 0 ? "▲ +" : "▼ −"}
      {Math.abs(r).toFixed(digits)}
      {unit}
    </span>
  );
}

function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: [T, string][];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex gap-1 self-start rounded-lg bg-panel-2 p-1">
      {options.map(([v, text]) => (
        <button
          key={v}
          type="button"
          role="radio"
          aria-checked={value === v}
          onClick={() => onChange(v)}
          className={`rounded-md px-3 py-1 text-sm font-medium ${value === v ? "bg-panel text-ink shadow-sm" : "text-ink-2"}`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

function SummaryTable({ s, s0, show }: { s: Summary; s0: Summary | null; show: boolean }) {
  const { t } = useI18n();
  const d = (f: (x: Summary) => number | null) =>
    s0 && f(s) != null && f(s0) != null ? f(s)! - f(s0)! : null;
  const rows: [string, string | number, number | null, number, string][] = [
    [t("summary.infected"), s.infected, d((x) => x.infected), 0, ""],
    [t("summary.uninfected"), s.uninfected, d((x) => x.uninfected), 0, ""],
    [
      t("summary.parasitemia"),
      s.parasitemia == null ? "—" : `${(s.parasitemia * 100).toFixed(1)} %`,
      d((x) => (x.parasitemia == null ? null : x.parasitemia * 100)),
      1,
      ` ${t("compare.pp")}`,
    ],
  ];
  return (
    <table className="tabular w-full">
      <tbody>
        {rows.map(([label, v, dv, digits, unit]) => (
          <tr key={label}>
            <td className="py-0.5 text-ink-2">{label}</td>
            <td className="py-0.5 text-right font-semibold">{v}</td>
            <td className="py-0.5 pl-2 text-right text-xs">
              {show ? <Delta v={dv} digits={digits} unit={unit} /> : null}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Live({ models }: { models: ModelInfo[] }) {
  const { t, lang, maybe } = useI18n();
  const detr = useMemo(() => models.filter((m) => m.kind === "detr"), [models]);
  const [images, setImages] = useState<ValImage[]>([]);
  const [kinds, setKinds] = useState<CorruptionKind[]>([]);
  const [imageId, setImageId] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>("single");
  const [layout, setLayout] = useState<Layout>("columns");
  const [kind, setKind] = useState("stain");
  const [strength, setStrength] = useState(0.5);
  const [items, setItems] = useState<StackItem[]>(defaultStack);
  const [results, setResults] = useState<Record<string, InferenceResult>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [desc, setDesc] = useState<StackDescription | null>(null);
  const requested = useRef(new Set<string>());
  const view = useState<View | null>(null);

  // curtain + animation
  const [curtainModel, setCurtainModel] = useState<string | null>(null);
  const [split, setSplit] = useState(0.5);
  const [animOn, setAnimOn] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);
  const [playhead, setPlayhead] = useState<Playhead>({ i: FRAMES.length - 1, dir: -1 });

  useEffect(() => {
    getValImages()
      .then((xs) => {
        setImages(xs);
        setImageId((cur) => cur ?? xs[0]?.id ?? null);
      })
      .catch(() => setLoadError("network"));
    getCorruptions()
      .then(setKinds)
      .catch(() => setLoadError("network"));
  }, []);

  const liveSteps = useMemo<Step[]>(
    () => (mode === "single" ? (strength > 0 ? [{ kind, strength }] : []) : toSteps(items)),
    [mode, kind, strength, items],
  );
  const liveKey = stepsKey(liveSteps);
  const [committed, setCommitted] = useState(liveKey);
  useEffect(() => {
    const h = setTimeout(() => setCommitted(liveKey), DEBOUNCE_MS);
    return () => clearTimeout(h);
  }, [liveKey]);
  const committedSteps = useMemo<Step[]>(
    () =>
      committed
        ? committed.split(",").map((p) => {
            const [k, v] = p.split(":");
            return { kind: k, strength: +v };
          })
        : [],
    [committed],
  );

  useEffect(() => {
    let alive = true;
    describeStack(committed)
      .then((d) => alive && setDesc(d))
      .catch(() => alive && setDesc(null));
    return () => {
      alive = false;
    };
  }, [committed]);

  const request = useCallback((mid: string, img: number, steps: Step[]) => {
    const id = rkey(mid, img, stepsKey(steps));
    if (requested.current.has(id)) return;
    requested.current.add(id);
    runRobustnessJob({ model_id: mid, image_id: img, stack: steps }, () => {})
      .then((r) => setResults((all) => ({ ...all, [id]: r })))
      .catch((e) => {
        requested.current.delete(id);
        setErrors((all) => ({ ...all, [id]: e instanceof ApiError ? e.code : "network" }));
      });
  }, []);

  const available = useMemo(() => detr.filter((m) => m.available), [detr]);
  const cm = curtainModel ?? available[0]?.id ?? null;

  useEffect(() => {
    if (imageId == null) return;
    for (const m of available) {
      if (layout === "curtain" && m.id !== cm) continue;
      request(m.id, imageId, []);
      if (committedSteps.length) request(m.id, imageId, committedSteps);
    }
  }, [available, imageId, committedSteps, request, layout, cm]);

  // animation frames (the committed corruption with every strength x factor): computed once (queued on the server),
  // then played from the cache
  const frameSteps = useMemo(() => FRAMES.map((f) => scaleSteps(committedSteps, f)), [committedSteps]);
  useEffect(() => {
    if (!animOn || imageId == null || !cm) return;
    for (const st of frameSteps) {
      request(cm, imageId, st);
      new Image().src = previewUrl(imageId, stepsKey(st)); // warm the browser cache
    }
  }, [animOn, frameSteps, imageId, cm, request]);
  const ready =
    imageId != null && cm ? frameSteps.filter((st) => results[rkey(cm, imageId, stepsKey(st))]).length : 0;
  const allReady = ready === FRAMES.length;

  useEffect(() => {
    if (!playing || !allReady) return;
    const h = setInterval(() => setPlayhead((p) => nextFrame(p)), frameInterval(speed));
    return () => clearInterval(h);
  }, [playing, allReady, speed]);

  const img = images.find((i) => i.id === imageId) ?? null;
  const src = imageId != null ? previewUrl(imageId, committed) : null;
  const longSide = desc?.long_side ?? 800;
  const thrOf = (m: ModelInfo) => defaultThresholds(m).thresholds;

  const paramText = (params: Record<string, unknown>) =>
    Object.entries(params)
      .filter(([k]) => k !== "identity" && k !== "seed" && k !== "long_side")
      .map(([k, v]) => `${k} = ${typeof v === "number" ? +v.toFixed(4) : JSON.stringify(v)}`)
      .join(" · ");

  const kindName = (k: string) => maybe(`rob.kind.${k}`) ?? k;

  return (
    <>
      <Panel title={t("tab.live")}>
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-end gap-4">
            <label className="flex min-w-0 flex-1 flex-col gap-1 text-sm sm:max-w-96">
              <span className="font-semibold">{t("rob.image")}</span>
              <select
                value={imageId ?? ""}
                onChange={(e) => {
                  setImageId(+e.target.value);
                  view[1](null);
                }}
                className="w-full min-w-0 rounded-lg border border-line bg-panel px-2 py-1.5"
              >
                {images.map((i) => (
                  <option key={i.id} value={i.id}>
                    #{i.id} · {i.file_name}
                  </option>
                ))}
              </select>
              <span className="text-xs text-muted">{t("rob.valNote")}</span>
            </label>
            <Segmented
              label={t("stack.mode")}
              value={mode}
              onChange={setMode}
              options={[
                ["single", t("stack.single")],
                ["stack", t("stack.stack")],
              ]}
            />
          </div>

          {mode === "single" ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <label className="flex min-w-0 flex-col gap-1 text-sm">
                <span className="font-semibold">{t("rob.corruption")}</span>
                <select
                  value={kind}
                  onChange={(e) => setKind(e.target.value)}
                  className="w-full min-w-0 rounded-lg border border-line bg-panel px-2 py-1.5"
                >
                  {kinds.map((k) => (
                    <option key={k.kind} value={k.kind}>
                      {kindName(k.kind)}
                    </option>
                  ))}
                </select>
              </label>
              <div className="flex flex-col gap-1 text-sm">
                <label htmlFor="rob-strength" className="flex justify-between font-semibold">
                  <span>{t("rob.strength")}</span>
                  <span className="tabular">{strength.toFixed(2)}</span>
                </label>
                <input
                  id="rob-strength"
                  type="range"
                  min={0}
                  max={1}
                  step={STEP}
                  value={strength}
                  onChange={(e) => setStrength(+e.target.value)}
                />
                <div className="flex justify-between text-[11px] text-muted">
                  <span>0</span>
                  <span>0.5</span>
                  <span>1</span>
                </div>
              </div>
              <p className="rounded-lg bg-panel-2 p-2.5 text-xs text-ink-2 lg:col-span-2">
                {maybe(`rob.help.${kind}`)}
              </p>
            </div>
          ) : (
            <>
              <p className="rounded-lg border border-warn-ink/40 bg-warn-bg p-2.5 text-xs text-warn-ink">
                {t("stack.notProtocol")}
              </p>
              <StackEditor items={items} onChange={setItems} />
            </>
          )}

          {desc && !desc.identity && (
            <div className="font-mono text-xs break-all text-ink-2">
              <span className="font-sans font-semibold">{t("rob.params")}:</span>
              <ol className="mt-1 flex flex-col gap-0.5">
                {desc.steps.map((st, i) => (
                  <li key={st.kind}>
                    {desc.steps.length > 1 ? `${i + 1}. ` : ""}
                    <span className="font-sans">{kindName(st.kind)}</span> ({st.strength}):{" "}
                    {paramText(st.params)}
                  </li>
                ))}
              </ol>
              <p className="mt-1">long_side = {desc.long_side}</p>
            </div>
          )}
          {loadError && <p className="text-sm text-danger-ink">{maybe(`error.${loadError}`)}</p>}
        </div>
      </Panel>

      {img && src && (
        <Panel title={`${t("rob.input")} · ${longSide} px`}>
          <div className="mx-auto max-w-[600px]">
            <img
              src={src}
              alt={`${t("rob.input")}: ${img.file_name}`}
              className="rounded-lg"
              style={{ width: `${(longSide / MAX_LONG_SIDE) * 100}%` }}
            />
          </div>
          <p className="mt-2 text-xs text-muted">{t("rob.inputNote")}</p>
        </Panel>
      )}

      {img && src && (
        <Segmented
          label={t("curtain.layout")}
          value={layout}
          onChange={(v) => {
            setLayout(v);
            if (v === "columns") {
              setAnimOn(false);
              setPlaying(false);
            }
          }}
          options={[
            ["columns", t("curtain.columns")],
            ["curtain", t("curtain.curtain")],
          ]}
        />
      )}

      {img && src && layout === "columns" && (
        <div className="grid gap-4 md:grid-cols-3">
          {detr.map((m) => {
            const thr = thrOf(m);
            const kS = rkey(m.id, img.id, committed);
            const k0 = rkey(m.id, img.id, "");
            const r = results[kS];
            const r0 = results[k0];
            const err = errors[kS] ?? errors[k0];
            return (
              <section key={m.id} className="flex min-w-0 flex-col gap-2" aria-label={m.name[lang]}>
                <div className="h-[38vh] min-h-[260px]">
                  <Viewer
                    src={src}
                    width={img.width}
                    height={img.height}
                    dets={r ? applyThresholds(r.detections, thr) : []}
                    selectedId={null}
                    onSelect={() => {}}
                    focus={null}
                    view={view}
                    busy={m.available && !r && !err}
                    label={m.name[lang]}
                  />
                </div>
                <div className="rounded-xl border border-line bg-panel p-3 text-sm">
                  {!m.available ? (
                    <p className="text-danger-ink">
                      {t("rob.unavailable")}: <span className="font-mono text-xs break-all">{m.error}</span>
                    </p>
                  ) : err ? (
                    <p className="text-danger-ink">{maybe(`error.${err}`) ?? t("error.unknown")}</p>
                  ) : !r ? (
                    <p className="text-muted">{t("rob.pending")}</p>
                  ) : (
                    <SummaryTable
                      s={summarize(applyThresholds(r.detections, thr))}
                      s0={r0 ? summarize(applyThresholds(r0.detections, thr)) : null}
                      show={committed !== ""}
                    />
                  )}
                </div>
              </section>
            );
          })}
        </div>
      )}

      {img && layout === "curtain" && cm && (
        <CurtainPanel
          img={img}
          models={available}
          model={cm}
          onModel={(id) => {
            setCurtainModel(id);
            setPlaying(false);
          }}
          split={split}
          onSplit={setSplit}
          view={view}
          clean={results[rkey(cm, img.id, "")] ?? null}
          shownKey={animOn ? stepsKey(frameSteps[playhead.i]) : committed}
          results={results}
          errors={errors}
          anim={{
            on: animOn,
            setOn: (v) => {
              setAnimOn(v);
              setPlaying(false);
              setPlayhead({ i: FRAMES.length - 1, dir: -1 });
            },
            playing,
            setPlaying,
            speed,
            setSpeed,
            frame: FRAMES[playhead.i],
            setFrame: (i) => {
              setPlaying(false);
              setPlayhead({ i, dir: 1 });
            },
            ready,
            mode,
          }}
        />
      )}

      <p className="text-xs text-muted">
        {t("rob.deltaNote")} {t("summary.parasitemiaHelp")}
      </p>
    </>
  );
}

interface Anim {
  on: boolean;
  setOn: (v: boolean) => void;
  playing: boolean;
  setPlaying: (v: boolean) => void;
  speed: Speed;
  setSpeed: (v: Speed) => void;
  frame: number;
  setFrame: (i: number) => void;
  ready: number;
  mode: Mode;
}

function CurtainPanel(p: {
  img: ValImage;
  models: ModelInfo[];
  model: string;
  onModel: (id: string) => void;
  split: number;
  onSplit: (v: number) => void;
  view: ViewState;
  clean: InferenceResult | null;
  shownKey: string;
  results: Record<string, InferenceResult>;
  errors: Record<string, string>;
  anim: Anim;
}) {
  const { t, lang, maybe } = useI18n();
  const m = p.models.find((x) => x.id === p.model)!;
  const thr = defaultThresholds(m).thresholds;
  const after = p.results[rkey(p.model, p.img.id, p.shownKey)];
  const err = p.errors[rkey(p.model, p.img.id, p.shownKey)] ?? p.errors[rkey(p.model, p.img.id, "")];
  const before = p.clean ? applyThresholds(p.clean.detections, thr) : [];
  const afterDets = after ? applyThresholds(after.detections, thr) : [];
  const splitPx = p.split * p.img.width;
  const leftN = before.filter((d) => halfOf(d.bbox, splitPx) !== "right").length;
  const rightN = afterDets.filter((d) => halfOf(d.bbox, splitPx) !== "left").length;
  const a = p.anim;
  const reduced =
    typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  return (
    <Panel
      title={t("curtain.curtain")}
      actions={
        <select
          value={p.model}
          onChange={(e) => p.onModel(e.target.value)}
          aria-label={t("model.title")}
          className="min-w-0 rounded-lg border border-line bg-panel px-2 py-1 text-sm"
        >
          {p.models.map((x) => (
            <option key={x.id} value={x.id}>
              {x.name[lang]}
            </option>
          ))}
        </select>
      }
    >
      <div className="h-[58vh] min-h-[320px]">
        <Viewer
          src={previewUrl(p.img.id, "")}
          width={p.img.width}
          height={p.img.height}
          dets={before}
          selectedId={null}
          onSelect={() => {}}
          focus={null}
          view={p.view}
          busy={!after && !err}
          label={`← ${t("curtain.before")}`}
          curtain={{
            src: previewUrl(p.img.id, p.shownKey),
            dets: afterDets,
            split: p.split,
            onSplit: p.onSplit,
            labels: [t("curtain.before"), t("curtain.after")],
          }}
        />
      </div>
      <div className="mt-3 flex flex-col gap-3">
        <div className="flex items-center gap-3 text-sm">
          <span className="shrink-0 font-semibold" id="curtain-split-label">
            {t("curtain.split")}
          </span>
          <div
            role="slider"
            tabIndex={0}
            aria-labelledby="curtain-split-label"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(p.split * 100)}
            aria-valuetext={t("curtain.splitText", { l: leftN, r: rightN })}
            onKeyDown={(e) => {
              const v = splitFromKey(p.split, e.key, e.shiftKey);
              if (v != null) {
                e.preventDefault();
                p.onSplit(v);
              }
            }}
            onPointerDown={(e) => {
              const el = e.currentTarget;
              el.setPointerCapture(e.pointerId);
              const set = (x: number) => {
                const r = el.getBoundingClientRect();
                p.onSplit(Math.min(1, Math.max(0, (x - r.left) / r.width)));
              };
              set(e.clientX);
              el.onpointermove = (ev) => set(ev.clientX);
              el.onpointerup = () => (el.onpointermove = null);
            }}
            className="relative h-6 min-w-0 flex-1 cursor-ew-resize touch-none rounded-full bg-panel-2 focus-visible:outline-2 focus-visible:outline-accent"
          >
            <div
              className="absolute inset-y-0 left-0 rounded-l-full bg-accent/25"
              style={{ width: `${p.split * 100}%` }}
            />
            <div
              className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink bg-panel"
              style={{ left: `${p.split * 100}%` }}
            />
          </div>
          <span className="tabular w-10 text-right text-xs">{Math.round(p.split * 100)}%</span>
        </div>
        <p className="text-xs text-ink-2" aria-live="polite">
          {t("curtain.counts", { l: leftN, r: rightN })}
          {err ? ` · ${maybe(`error.${err}`) ?? t("error.unknown")}` : ""}
        </p>

        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line p-2.5 text-sm">
          <label className="flex items-center gap-2 font-semibold">
            <input type="checkbox" checked={a.on} onChange={(e) => a.setOn(e.target.checked)} />
            {t("anim.title")}
          </label>
          {a.on && (
            <>
              <Button
                variant="primary"
                disabled={a.ready < FRAMES.length}
                onClick={() => a.setPlaying(!a.playing)}
              >
                {a.playing ? `❚❚ ${t("anim.pause")}` : `▶ ${t("anim.play")}`}
              </Button>
              <label className="flex items-center gap-1.5">
                {t("anim.speed")}
                <select
                  value={a.speed}
                  onChange={(e) => a.setSpeed(+e.target.value as Speed)}
                  className="rounded-md border border-line bg-panel px-1 py-0.5"
                >
                  {SPEEDS.map((s) => (
                    <option key={s} value={s}>
                      {s}×
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex min-w-40 flex-1 items-center gap-2">
                <span className="shrink-0">{a.mode === "single" ? t("rob.strength") : t("anim.factor")}</span>
                <input
                  type="range"
                  min={0}
                  max={FRAMES.length - 1}
                  step={1}
                  value={Math.round(a.frame * (FRAMES.length - 1))}
                  onChange={(e) => a.setFrame(+e.target.value)}
                  className="min-w-0 flex-1"
                  aria-valuetext={a.frame.toFixed(1)}
                />
                <span className="tabular w-8 text-right">{a.frame.toFixed(1)}</span>
              </label>
              <span className="text-xs text-ink-2" aria-live="polite">
                {a.ready < FRAMES.length
                  ? t("anim.preparing", { n: a.ready, total: FRAMES.length })
                  : t("anim.ready", { total: FRAMES.length })}
              </span>
            </>
          )}
        </div>
        <p className="text-xs text-muted">
          {t("anim.help")}
          {reduced ? ` ${t("anim.reduced")}` : ""}
        </p>
      </div>
    </Panel>
  );
}
