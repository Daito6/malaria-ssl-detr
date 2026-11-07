import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getHealth, getModels, runJob } from "./api";
import { Compare } from "./components/Compare";
import { ClassFilter, ModelPicker, ProgressBar, ThresholdControls, Uploader } from "./components/Controls";
import { CellTable, ResultInfo, ScoreHistograms, SummaryTiles } from "./components/Results";
import { Robustness } from "./components/Robustness";
import { Button, Panel } from "./components/ui";
import { Viewer, type FocusRequest, type View } from "./components/Viewer";
import { useI18n, type Lang } from "./i18n";
import { baseName, download, drawAnnotated, toCsv } from "./lib/export";
import { applyClassFilter, applyThresholds, defaultThresholds, summarize } from "./lib/filter";
import { useTheme, type ThemePref } from "./theme";
import type { ClassName, Health, InferenceResult, ModelInfo, Thresholds } from "./types";
import { CLASSES } from "./types";

export interface LoadedImage {
  url: string;
  width: number;
  height: number;
  el: HTMLImageElement;
}

export interface AppError {
  code: string;
  detail?: string;
}

export interface RunState {
  stage: string;
  progress: number;
  queuePos: number;
}

type Tab = "analysis" | "compare" | "robustness";

function omit<T>(obj: Record<string, T>, k: string): Record<string, T> {
  const out = { ...obj };
  delete out[k];
  return out;
}
const TABS: Tab[] = ["analysis", "compare", "robustness"];

function storedTab(): Tab {
  try {
    const v = localStorage.getItem("smearscan.tab");
    if (TABS.includes(v as Tab)) return v as Tab;
  } catch {
    /* storage unavailable */
  }
  return "analysis";
}

export default function App() {
  const { t, lang, setLang, maybe } = useI18n();
  const [theme, setTheme] = useTheme();
  const [tab, setTabState] = useState<Tab>(storedTab);
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [image, setImage] = useState<LoadedImage | null>(null);
  const [results, setResults] = useState<Record<string, InferenceResult>>({});
  const [runs, setRuns] = useState<Record<string, RunState>>({});
  const [runErrors, setRunErrors] = useState<Record<string, AppError>>({});
  const [error, setError] = useState<AppError | null>(null);
  const [thresholds, setThresholds] = useState<Record<string, Thresholds>>({});
  const [shown, setShown] = useState<Set<ClassName>>(() => new Set(CLASSES));
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [focus, setFocus] = useState<FocusRequest | null>(null);
  const analysisView = useState<View | null>(null);
  const compareView = useState<View | null>(null);
  const fileToken = useRef(0);
  const inflight = useRef(new Set<string>());

  const setTab = (v: Tab) => {
    setTabState(v);
    try {
      localStorage.setItem("smearscan.tab", v);
    } catch {
      /* storage unavailable */
    }
  };

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setError({ code: "network" }));
    getModels()
      .then((ms) => {
        setModels(ms);
        setModelId(ms.find((m) => m.available)?.id ?? null);
        setThresholds(Object.fromEntries(ms.map((m) => [m.id, defaultThresholds(m).thresholds])));
      })
      .catch(() => setError({ code: "network" }));
  }, []);

  /** One inference of the current file per model; results of a replaced file are ignored. */
  const run = useCallback(async (f: File, mid: string) => {
    const token = fileToken.current;
    const key = `${token}:${mid}`;
    if (inflight.current.has(key)) return;
    inflight.current.add(key);
    const current = () => token === fileToken.current;
    setRunErrors((all) => omit(all, mid));
    setRuns((r) => ({ ...r, [mid]: { stage: "queued", progress: 0, queuePos: 0 } }));
    try {
      const r = await runJob(f, mid, (s) => {
        if (current())
          setRuns((all) => ({
            ...all,
            [mid]: { stage: s.stage, progress: s.progress, queuePos: s.queue_position },
          }));
      });
      if (current()) setResults((rs) => ({ ...rs, [mid]: r }));
    } catch (e) {
      if (current())
        setRunErrors((all) => ({
          ...all,
          [mid]: e instanceof ApiError ? { code: e.code, detail: e.message } : { code: "network" },
        }));
    } finally {
      inflight.current.delete(key);
      if (current()) setRuns((all) => omit(all, mid));
    }
  }, []);

  const ensure = useCallback(
    (mid: string) => {
      if (file && !results[mid] && !runErrors[mid]) run(file, mid);
    },
    [file, results, runErrors, run],
  );

  const onFile = (f: File) => {
    const url = URL.createObjectURL(f);
    const el = new Image();
    el.onload = () => {
      fileToken.current += 1;
      setImage((old) => {
        if (old) URL.revokeObjectURL(old.url);
        return { url, width: el.naturalWidth, height: el.naturalHeight, el };
      });
      setFile(f);
      setResults({});
      setRuns({});
      setRunErrors({});
      setError(null);
      setSelectedId(null);
      analysisView[1](null);
      compareView[1](null);
      if (modelId && tab === "analysis") run(f, modelId);
    };
    el.onerror = () => {
      URL.revokeObjectURL(url);
      setError({ code: "invalid_image" });
    };
    el.src = url;
  };

  const onModel = (id: string) => {
    setModelId(id);
    setSelectedId(null);
    if (file && !results[id]) run(file, id);
  };

  useEffect(() => {
    if (tab === "analysis" && modelId) ensure(modelId);
  }, [tab, modelId, ensure]);

  const model = models.find((m) => m.id === modelId) ?? null;
  const result = modelId ? results[modelId] : undefined;
  const thr = modelId ? thresholds[modelId] : undefined;
  const job = modelId ? runs[modelId] : undefined;
  const runError = (modelId && runErrors[modelId]) || null;
  const above = useMemo(() => (result && thr ? applyThresholds(result.detections, thr) : []), [result, thr]);
  const visible = useMemo(() => applyClassFilter(above, shown), [above, shown]);
  const summary = useMemo(() => summarize(above), [above]);
  const selected = selectedId != null && visible.some((d) => d.id === selectedId) ? selectedId : null;

  const setThreshold = (mid: string, c: ClassName, v: number) =>
    setThresholds((all) => ({ ...all, [mid]: { ...all[mid], [c]: v } }));
  const resetThreshold = (mid: string) => {
    const m = models.find((x) => x.id === mid);
    if (m) setThresholds((all) => ({ ...all, [mid]: defaultThresholds(m).thresholds }));
  };
  const toggleClass = (c: ClassName) =>
    setShown((s) => {
      const n = new Set(s);
      if (n.has(c)) n.delete(c);
      else n.add(c);
      return n;
    });

  const exportCsv = () => {
    if (!file || !modelId || !thr) return;
    const csv = toCsv(visible, { imageName: file.name, modelId, thresholds: thr });
    download(
      new Blob(["﻿", csv], { type: "text/csv;charset=utf-8" }),
      `${baseName(file.name)}_${modelId}.csv`,
    );
  };
  const exportPng = () => {
    if (!file || !image || !model || !result || !thr) return;
    const ck = (result.checkpoint.boxes ?? result.checkpoint) as Record<string, unknown>;
    const footer = [
      t("export.footer.model", {
        model: model.name[lang],
        ckpt: `${ck.file} (sha256 ${ck.sha256}, epoch ${ck.epoch})`,
      }),
      t("export.footer.thr", {
        u: thr.uninfected.toFixed(3),
        i: thr.infected.toFixed(3),
        date: new Date().toLocaleString(lang === "uk" ? "uk-UA" : "en-GB"),
      }),
      t("export.footer.disclaimer"),
    ];
    drawAnnotated(image.el, visible, footer).toBlob(
      (b) => b && download(b, `${baseName(file.name)}_${modelId}_annotated.png`),
      "image/png",
    );
  };

  const shownError = runError ?? error;
  const errorText = shownError ? (maybe(`error.${shownError.code}`) ?? t("error.unknown")) : null;
  const maxBytes = health?.limits.max_bytes ?? 25 * 1024 * 1024;

  return (
    <div className="min-h-screen">
      <div className="z-20 lg:sticky lg:top-0">
        <header className="border-b border-line bg-panel">
          <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-3 px-4 py-2.5">
            <div className="flex items-center gap-2.5">
              <img src="/favicon.svg" alt="" width={28} height={28} />
              <div>
                <h1 className="text-lg leading-tight font-bold">SmearScan</h1>
                <p className="text-xs text-ink-2">{t("app.tagline")}</p>
              </div>
            </div>
            <nav role="tablist" aria-label="SmearScan" className="flex gap-1 rounded-lg bg-panel-2 p-1">
              {TABS.map((v) => (
                <button
                  key={v}
                  type="button"
                  role="tab"
                  aria-selected={tab === v}
                  onClick={() => setTab(v)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium ${tab === v ? "bg-panel text-ink shadow-sm" : "text-ink-2 hover:text-ink"}`}
                >
                  {t(`tab.${v}` as const)}
                </button>
              ))}
            </nav>
            <div className="flex items-center gap-2">
              <div
                role="group"
                aria-label={t("lang.label")}
                className="flex rounded-lg border border-line p-0.5"
              >
                {(["uk", "en"] as Lang[]).map((l) => (
                  <button
                    key={l}
                    type="button"
                    aria-pressed={lang === l}
                    onClick={() => setLang(l)}
                    className={`rounded-md px-2.5 py-1 text-xs font-semibold uppercase ${lang === l ? "bg-accent text-accent-ink" : "text-ink-2"}`}
                  >
                    {l === "uk" ? "УКР" : "EN"}
                  </button>
                ))}
              </div>
              <label className="sr-only" htmlFor="theme">
                {t("theme.label")}
              </label>
              <select
                id="theme"
                value={theme}
                onChange={(e) => setTheme(e.target.value as ThemePref)}
                className="rounded-lg border border-line bg-panel px-2 py-1.5 text-xs"
              >
                {(["system", "light", "dark"] as ThemePref[]).map((v) => (
                  <option key={v} value={v}>
                    {t(`theme.${v}` as const)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </header>
        <div role="note" className="border-b border-warn-line bg-warn-bg text-warn-ink">
          <p className="mx-auto max-w-[1600px] px-4 py-1.5 text-xs">
            <strong className="font-semibold">⚠ {t("disclaimer.title")}</strong> {t("disclaimer.body")}
          </p>
        </div>
      </div>

      {tab === "analysis" && (
        <main className="mx-auto grid max-w-[1600px] gap-4 p-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="flex flex-col gap-4">
            <Panel title={t("upload.title")}>
              <Uploader
                fileName={file?.name ?? null}
                maxBytes={maxBytes}
                onFile={onFile}
                onError={(code) => setError({ code })}
              />
              {job && (
                <div className="mt-3">
                  <ProgressBar stage={job.stage} progress={job.progress} queuePos={job.queuePos} />
                  {model?.kind === "two_stage" && (
                    <p className="mt-1 text-xs text-muted">{t("run.firstTwoStage")}</p>
                  )}
                </div>
              )}
              {shownError && (
                <div role="alert" className="mt-3 rounded-lg bg-danger-bg p-2.5 text-sm text-danger-ink">
                  <p className="font-semibold">
                    {t("error.title")}: {errorText}
                  </p>
                  {shownError.detail && (
                    <p className="mt-1 font-mono text-xs break-words">{shownError.detail}</p>
                  )}
                  {file && modelId && runError && !job && (
                    <div className="mt-2">
                      <Button onClick={() => run(file, modelId)}>{t("run.rerun")}</Button>
                    </div>
                  )}
                </div>
              )}
            </Panel>
            <Panel title={t("model.title")}>
              <ModelPicker models={models} value={modelId} onChange={onModel} />
            </Panel>
            {model && thr && (
              <Panel title={t("thr.title")}>
                <ThresholdControls
                  model={model}
                  value={thr}
                  onChange={(c, v) => setThreshold(model.id, c, v)}
                  onReset={() => resetThreshold(model.id)}
                />
              </Panel>
            )}
            <Panel title={t("filter.title")}>
              <ClassFilter shown={shown} onToggle={toggleClass} />
            </Panel>
            <Panel title={t("export.title")}>
              <div className="flex flex-wrap gap-2">
                <Button onClick={exportCsv} disabled={!result}>
                  ⭳ {t("export.csv")}
                </Button>
                <Button onClick={exportPng} disabled={!result}>
                  ⭳ {t("export.png")}
                </Button>
              </div>
            </Panel>
          </aside>

          <div className="flex min-w-0 flex-col gap-4">
            <div className="h-[55vh] min-h-[340px] lg:h-[62vh]">
              <Viewer
                src={image?.url ?? null}
                width={result?.width ?? image?.width ?? 0}
                height={result?.height ?? image?.height ?? 0}
                dets={visible}
                selectedId={selected}
                onSelect={setSelectedId}
                focus={focus}
                view={analysisView}
                busy={!!job}
              />
            </div>
            <p className="-mt-2 text-xs text-muted">{t("viewer.help")}</p>
            {result && model && thr && (
              <>
                <div className="grid gap-4 2xl:grid-cols-2">
                  <Panel title={t("summary.title")}>
                    <SummaryTiles summary={summary} />
                    <div className="mt-4">
                      <h3 className="mb-2 text-xs font-semibold text-ink-2">{t("hist.title")}</h3>
                      <ScoreHistograms dets={result.detections} thresholds={thr} />
                    </div>
                  </Panel>
                  <Panel title={t("table.title")}>
                    <CellTable
                      dets={visible}
                      total={above.length}
                      selectedId={selected}
                      onSelect={(id) => {
                        setSelectedId(id);
                        const d = visible.find((x) => x.id === id);
                        if (d) setFocus({ bbox: d.bbox });
                      }}
                    />
                  </Panel>
                </div>
                <Panel title={t("info.title")}>
                  <ResultInfo result={result} model={model} />
                </Panel>
              </>
            )}
          </div>
        </main>
      )}

      {tab === "compare" && (
        <Compare
          models={models}
          image={image}
          fileName={file?.name ?? null}
          maxBytes={maxBytes}
          onFile={onFile}
          onUploadError={(code) => setError({ code })}
          uploadError={error}
          results={results}
          runs={runs}
          runErrors={runErrors}
          thresholds={thresholds}
          setThreshold={setThreshold}
          resetThreshold={resetThreshold}
          ensure={ensure}
          retry={(mid) => file && run(file, mid)}
          view={compareView}
        />
      )}

      {tab === "robustness" && <Robustness models={models} />}

      <footer className="mx-auto max-w-[1600px] px-4 pb-6 text-xs text-muted">
        SmearScan 0.2 ·{" "}
        <a className="underline" href="/docs">
          API /docs
        </a>
      </footer>
    </div>
  );
}
