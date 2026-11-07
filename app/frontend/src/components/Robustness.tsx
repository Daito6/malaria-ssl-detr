import { useEffect, useState } from "react";
import { ApiError, getCurve } from "../api";
import { useI18n } from "../i18n";
import type { Curve, CurveRow, ModelInfo } from "../types";
import { CurveChart, type Series } from "./CurveChart";
import { Live } from "./RobustnessLive";
import { Panel } from "./ui";
const SERIES_COLORS = ["var(--c-m1)", "var(--c-m2)", "var(--c-m3)"];

export function Robustness({ models }: { models: ModelInfo[] }) {
  const { t } = useI18n();
  const [sub, setSub] = useState<"live" | "curve">("live");
  return (
    <main className="mx-auto flex max-w-[1600px] flex-col gap-4 p-4">
      <div
        role="tablist"
        aria-label={t("tab.robustness")}
        className="flex gap-1 self-start rounded-lg bg-panel-2 p-1"
      >
        {(["live", "curve"] as const).map((v) => (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={sub === v}
            onClick={() => setSub(v)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium ${sub === v ? "bg-panel text-ink shadow-sm" : "text-ink-2"}`}
          >
            {t(`tab.${v}` as const)}
          </button>
        ))}
      </div>
      {sub === "live" ? <Live models={models} /> : <CurveTab models={models} />}
    </main>
  );
}

function CurveTab({ models }: { models: ModelInfo[] }) {
  const { t, lang, maybe } = useI18n();
  const [curve, setCurve] = useState<Curve | null>(null);
  const [missing, setMissing] = useState(false);
  const [kind, setKind] = useState("stain");
  const [asTable, setAsTable] = useState(false);

  useEffect(() => {
    getCurve()
      .then(setCurve)
      .catch((e) => setMissing(e instanceof ApiError && e.code === "curve_missing"));
  }, []);

  if (missing || !curve)
    return (
      <Panel title={t("tab.curve")}>
        {missing ? (
          <>
            <p className="text-sm">{t("curve.missing")}</p>
            <p className="mt-2 text-xs text-muted">{t("curve.missingHow")}</p>
            <pre className="mt-1 overflow-x-auto rounded-lg bg-panel-2 p-2 text-xs">
              {`conda run -n medthesis python scripts/robustness_curve.py predict --corruptions stain blur jpeg
conda run -n medthesis python scripts/robustness_curve.py predict --corruptions brightness_contrast scale_down scale_up stain_full
conda run -n medthesis python scripts/robustness_curve.py aggregate`}
            </pre>
            <p className="mt-2 text-xs text-muted">reports/robustness_plan.md</p>
          </>
        ) : (
          <p className="text-sm text-muted">…</p>
        )}
      </Panel>
    );

  const proto = curve.protocol as Record<string, unknown>;
  const kinds = [...new Set(curve.rows.map((r) => r.corruption))].filter(
    (k) => k !== "none" && k !== "stain_full",
  );
  const modelIds = [...new Set(curve.rows.map((r) => r.model))];
  const nameOf = (id: string) => models.find((m) => m.id === id)?.name[lang] ?? id;
  const series: Series[] = modelIds.map((id, i) => ({
    id,
    name: nameOf(id),
    color: SERIES_COLORS[i % SERIES_COLORS.length],
    rows: curve.rows
      .filter((r) => r.model === id && (r.corruption === kind || r.corruption === "none"))
      .sort((a, b) => a.strength - b.strength),
  }));
  const control = curve.rows.filter((r) => r.corruption === "stain_full");
  const tableRows: CurveRow[] = [...series.flatMap((s) => s.rows), ...(kind === "stain" ? control : [])];
  const missingConditions = (proto.missing_conditions as string[] | undefined) ?? [];
  const fmt = (v: number | null) => (v == null ? "—" : v.toFixed(3));

  return (
    <Panel
      title={t("tab.curve")}
      actions={
        <div className="flex gap-2">
          <select
            value={kind}
            onChange={(e) => setKind(e.target.value)}
            aria-label={t("rob.corruption")}
            className="rounded-lg border border-line bg-panel px-2 py-1 text-sm"
          >
            {kinds.map((k) => (
              <option key={k} value={k}>
                {maybe(`rob.kind.${k}`) ?? k}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setAsTable((v) => !v)}
            className="rounded-lg border border-line bg-panel-2 px-2 py-1 text-sm"
          >
            {asTable ? t("curve.chart") : t("curve.table")}
          </button>
        </div>
      }
    >
      {asTable ? (
        <div className="overflow-auto rounded-lg border border-line">
          <table className="tabular w-full text-sm">
            <thead className="bg-panel-2 text-xs text-ink-2">
              <tr>
                <th className="px-2 py-1.5 text-left font-medium">{t("curve.model")}</th>
                <th className="px-2 py-1.5 text-left font-medium">{t("rob.corruption")}</th>
                <th className="px-2 py-1.5 text-right font-medium">{t("curve.strength")}</th>
                <th className="px-2 py-1.5 text-right font-medium">AP50 [95% CI]</th>
                <th className="px-2 py-1.5 text-right font-medium">recall [95% CI]</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map((r) => (
                <tr key={`${r.model}|${r.corruption}|${r.strength}`} className="border-t border-line">
                  <td className="px-2 py-1">{nameOf(r.model)}</td>
                  <td className="px-2 py-1">{r.corruption}</td>
                  <td className="px-2 py-1 text-right">{r.strength}</td>
                  <td className="px-2 py-1 text-right">
                    {fmt(r.AP50)} [{fmt(r.AP50_lo)}, {fmt(r.AP50_hi)}]
                  </td>
                  <td className="px-2 py-1 text-right">
                    {fmt(r.recall)} [{fmt(r.recall_lo)}, {fmt(r.recall_hi)}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-2">
          <CurveChart series={series} metric="AP50" title={t("curve.metric.AP50")} />
          <CurveChart series={series} metric="recall" title={t("curve.metric.recall")} />
        </div>
      )}
      {kind === "stain" && control.length > 0 && !asTable && (
        <p className="mt-2 text-xs text-ink-2">
          {t("curve.control")}:{" "}
          {control
            .map((r) => `${nameOf(r.model)} AP50 ${fmt(r.AP50)} [${fmt(r.AP50_lo)}, ${fmt(r.AP50_hi)}]`)
            .join(" · ")}
        </p>
      )}
      <p className="mt-3 text-xs text-muted">
        {t("curve.caption", {
          n: String(proto.n_images ?? "?"),
          r: String(proto.n_resamples ?? "?"),
          seed: String(proto.bootstrap_seed ?? "?"),
          src: curve.source,
        })}
      </p>
      {missingConditions.length > 0 && (
        <p className="mt-1 text-xs text-warn-ink">
          {t("curve.missingConditions", { n: missingConditions.length })}
        </p>
      )}
    </Panel>
  );
}
