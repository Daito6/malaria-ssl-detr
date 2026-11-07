import { useEffect, useMemo, useState } from "react";
import type { AppError, LoadedImage, RunState } from "../App";
import { useI18n } from "../i18n";
import { applyThresholds, summarize, type Summary } from "../lib/filter";
import { MATCH_IOU, matchDetections, type Disagreement } from "../lib/match";
import type { ClassName, InferenceResult, ModelInfo, Thresholds } from "../types";
import { ProgressBar, ThresholdControls, Uploader } from "./Controls";
import { Button, ClassSwatch, Panel } from "./ui";
import { Viewer, type FocusRequest, type ViewState } from "./Viewer";

interface Props {
  models: ModelInfo[];
  image: LoadedImage | null;
  fileName: string | null;
  maxBytes: number;
  onFile: (f: File) => void;
  onUploadError: (code: string) => void;
  uploadError: AppError | null;
  results: Record<string, InferenceResult>;
  runs: Record<string, RunState>;
  runErrors: Record<string, AppError>;
  thresholds: Record<string, Thresholds>;
  setThreshold: (mid: string, c: ClassName, v: number) => void;
  resetThreshold: (mid: string) => void;
  ensure: (mid: string) => void;
  retry: (mid: string) => void;
  view: ViewState;
}

function signed(v: number, digits = 0): string {
  const r = Number(v.toFixed(digits)); // so that -0.04 at 1 digit reads 0.0, not -0.0
  const s = Math.abs(r).toFixed(digits);
  return r > 0 ? `+${s}` : r < 0 ? `−${s}` : s;
}

export function Compare(p: Props) {
  const { t, lang, maybe } = useI18n();
  const available = p.models.filter((m) => m.available);
  const [ids, setIds] = useState<[string | null, string | null]>([null, null]);
  const a = ids[0] ?? available[0]?.id ?? null;
  const b = ids[1] ?? available.find((m) => m.id !== a)?.id ?? null;
  const [selected, setSelected] = useState<Disagreement | null>(null);
  const [focus, setFocus] = useState<FocusRequest | null>(null);
  const { ensure } = p;

  useEffect(() => {
    if (a) ensure(a);
    if (b) ensure(b);
  }, [a, b, ensure]);

  const side = (mid: string | null) => {
    const r = mid ? p.results[mid] : undefined;
    const thr = mid ? p.thresholds[mid] : undefined;
    const dets = r && thr ? applyThresholds(r.detections, thr) : [];
    return { mid, model: p.models.find((m) => m.id === mid) ?? null, r, thr, dets };
  };
  const A = side(a);
  const B = side(b);
  const match = useMemo(
    () => (A.r && B.r ? matchDetections(A.dets, B.dets) : null),
    [A.r, B.r, A.dets, B.dets],
  );
  const sumA = summarize(A.dets);
  const sumB = summarize(B.dets);

  const pick = (d: Disagreement) => {
    setSelected(d);
    setFocus({ bbox: (d.a ?? d.b)!.bbox });
  };

  const viewer = (s: typeof A, marked: Set<number> | undefined, selId: number | null, label: string) => (
    <div className="h-[48vh] min-h-[300px]">
      <Viewer
        src={p.image?.url ?? null}
        width={s.r?.width ?? p.image?.width ?? 0}
        height={s.r?.height ?? p.image?.height ?? 0}
        dets={s.dets}
        selectedId={selId}
        onSelect={() => setSelected(null)}
        focus={focus}
        view={p.view}
        marked={marked}
        busy={!!(s.mid && p.runs[s.mid])}
        label={label}
      />
    </div>
  );

  const picker = (which: 0 | 1, value: string | null) => (
    <label className="flex min-w-0 flex-1 flex-col gap-1 text-sm">
      <span className="font-semibold">{t(which === 0 ? "compare.modelA" : "compare.modelB")}</span>
      <select
        value={value ?? ""}
        onChange={(e) => {
          setSelected(null);
          setIds((cur) => (which === 0 ? [e.target.value, cur[1] ?? b] : [cur[0] ?? a, e.target.value]));
        }}
        className="w-full min-w-0 rounded-lg border border-line bg-panel px-2 py-1.5"
      >
        {p.models.map((m) => (
          <option key={m.id} value={m.id} disabled={!m.available}>
            {m.name[lang]}
            {m.available ? "" : ` (${t("model.unavailable")})`}
          </option>
        ))}
      </select>
    </label>
  );

  const status = (s: typeof A) => {
    if (!s.mid) return null;
    const job = p.runs[s.mid];
    const err = p.runErrors[s.mid];
    if (job) return <ProgressBar stage={job.stage} progress={job.progress} queuePos={job.queuePos} />;
    if (err)
      return (
        <div role="alert" className="rounded-lg bg-danger-bg p-2 text-xs text-danger-ink">
          {maybe(`error.${err.code}`) ?? t("error.unknown")} {err.detail}
          <div className="mt-1">
            <Button onClick={() => p.retry(s.mid!)}>{t("run.rerun")}</Button>
          </div>
        </div>
      );
    return null;
  };

  const rows: [string, (s: Summary) => number | null, number][] = [
    [t("summary.uninfected"), (s) => s.uninfected, 0],
    [t("summary.infected"), (s) => s.infected, 0],
    [t("summary.total"), (s) => s.total, 0],
  ];
  const pct = (s: Summary) => (s.parasitemia == null ? null : s.parasitemia * 100);

  return (
    <main className="mx-auto flex max-w-[1600px] flex-col gap-4 p-4">
      <Panel title={t("tab.compare")}>
        <div className="flex flex-col gap-3 md:flex-row md:items-end">
          {picker(0, a)}
          {picker(1, b)}
          <div className="md:w-72">
            <Uploader
              fileName={p.fileName}
              maxBytes={p.maxBytes}
              onFile={p.onFile}
              onError={p.onUploadError}
            />
          </div>
        </div>
        {p.uploadError && (
          <p role="alert" className="mt-2 text-sm text-danger-ink">
            {maybe(`error.${p.uploadError.code}`) ?? t("error.unknown")}
          </p>
        )}
      </Panel>

      {!p.image ? (
        <p className="text-sm text-muted">{t("compare.needImage")}</p>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="flex min-w-0 flex-col gap-2">
              {viewer(A, match?.markedA, selected?.a?.id ?? null, `A · ${A.model?.name[lang] ?? ""}`)}
              {status(A)}
            </div>
            <div className="flex min-w-0 flex-col gap-2">
              {viewer(B, match?.markedB, selected?.b?.id ?? null, `B · ${B.model?.name[lang] ?? ""}`)}
              {status(B)}
            </div>
          </div>
          <p className="-mt-2 text-xs text-muted">{t("viewer.help")}</p>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <Panel title={t("compare.diffTitle")}>
              <table className="tabular w-full text-sm">
                <thead className="text-xs text-ink-2">
                  <tr>
                    <th scope="col" className="py-1 text-left font-medium" />
                    <th scope="col" className="py-1 text-right font-medium">
                      A
                    </th>
                    <th scope="col" className="py-1 text-right font-medium">
                      B
                    </th>
                    <th scope="col" className="py-1 text-right font-medium">
                      B − A
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([label, f]) => {
                    const va = f(sumA) ?? 0;
                    const vb = f(sumB) ?? 0;
                    return (
                      <tr key={label} className="border-t border-line">
                        <td className="py-1.5">{label}</td>
                        <td className="py-1.5 text-right">{A.r ? va : "—"}</td>
                        <td className="py-1.5 text-right">{B.r ? vb : "—"}</td>
                        <td className="py-1.5 text-right font-semibold">
                          {A.r && B.r ? signed(vb - va) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                  <tr className="border-t border-line">
                    <td className="py-1.5">{t("summary.parasitemia")}</td>
                    <td className="py-1.5 text-right">
                      {A.r && pct(sumA) != null ? `${pct(sumA)!.toFixed(1)} %` : "—"}
                    </td>
                    <td className="py-1.5 text-right">
                      {B.r && pct(sumB) != null ? `${pct(sumB)!.toFixed(1)} %` : "—"}
                    </td>
                    <td className="py-1.5 text-right font-semibold">
                      {A.r && B.r && pct(sumA) != null && pct(sumB) != null
                        ? `${signed(pct(sumB)! - pct(sumA)!, 1)} ${t("compare.pp")}`
                        : "—"}
                    </td>
                  </tr>
                </tbody>
              </table>
              <p className="mt-2 text-xs text-muted">{t("compare.diffNote", { iou: MATCH_IOU })}</p>
              <p className="mt-1 text-xs text-muted">{t("summary.parasitemiaHelp")}</p>
              {a && b && A.model && B.model && A.thr && B.thr && (
                <details className="mt-3">
                  <summary className="cursor-pointer text-sm font-medium">{t("compare.thresholds")}</summary>
                  <div className="mt-3 grid gap-4 sm:grid-cols-2">
                    {[A, B].map((s, i) => (
                      <div key={i}>
                        <p className="mb-2 text-xs font-semibold">{i === 0 ? "A" : "B"}</p>
                        <ThresholdControls
                          model={s.model!}
                          value={s.thr!}
                          onChange={(c, v) => p.setThreshold(s.mid!, c, v)}
                          onReset={() => p.resetThreshold(s.mid!)}
                        />
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </Panel>

            <Panel title={`${t("compare.disagreements")}${match ? ` · ${match.disagreements.length}` : ""}`}>
              {!match ? null : match.disagreements.length === 0 ? (
                <p className="text-sm text-muted">{t("compare.none")}</p>
              ) : (
                <div className="max-h-80 overflow-auto rounded-lg border border-line">
                  <table className="tabular w-full text-sm">
                    <thead className="sticky top-0 bg-panel-2 text-xs text-ink-2">
                      <tr>
                        <th scope="col" className="px-2 py-1.5 text-left font-medium">
                          ≠
                        </th>
                        <th scope="col" className="px-2 py-1.5 text-left font-medium">
                          A
                        </th>
                        <th scope="col" className="px-2 py-1.5 text-left font-medium">
                          B
                        </th>
                        <th scope="col" className="px-2 py-1.5 text-right font-medium">
                          x, y
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {match.disagreements.map((d, i) => {
                        const box = (d.a ?? d.b)!.bbox;
                        const sel = selected === d;
                        const cell = (x: typeof d.a) =>
                          x ? (
                            <span className="inline-flex items-center gap-1">
                              <ClassSwatch label={x.label} size={14} />
                              {x.score.toFixed(2)}
                            </span>
                          ) : (
                            <span className="text-muted">—</span>
                          );
                        return (
                          <tr
                            key={i}
                            tabIndex={0}
                            aria-selected={sel}
                            onClick={() => pick(d)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                pick(d);
                              }
                            }}
                            className={`cursor-pointer border-t border-line ${sel ? "bg-accent/15 font-semibold" : "hover:bg-panel-2"}`}
                          >
                            <td className="px-2 py-1">{t(`compare.${d.kind}` as const)}</td>
                            <td className="px-2 py-1">{cell(d.a)}</td>
                            <td className="px-2 py-1">{cell(d.b)}</td>
                            <td className="px-2 py-1 text-right text-ink-2">
                              {Math.round(box[0])}, {Math.round(box[1])}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </Panel>
          </div>
        </>
      )}
    </main>
  );
}
