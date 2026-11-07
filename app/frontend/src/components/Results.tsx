import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n";
import { histogram, sortDetections, type SortKey, type Summary } from "../lib/filter";
import type { ClassName, Detection, InferenceResult, ModelInfo, Thresholds } from "../types";
import { CLASSES } from "../types";
import { ClassSwatch } from "./ui";

export function SummaryTiles({ summary }: { summary: Summary }) {
  const { t } = useI18n();
  const [help, setHelp] = useState(false);
  const pct = summary.parasitemia == null ? "—" : `${(summary.parasitemia * 100).toFixed(1)} %`;
  const tiles: [string, string | number, ClassName | null][] = [
    [t("summary.uninfected"), summary.uninfected, "uninfected"],
    [t("summary.infected"), summary.infected, "infected"],
    [t("summary.total"), summary.total, null],
  ];
  return (
    <div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {tiles.map(([label, v, cls]) => (
          <div key={label} className="rounded-lg bg-panel-2 p-3">
            <div className="flex items-center gap-1.5 text-xs text-ink-2">
              {cls && <ClassSwatch label={cls} size={14} />}
              {label}
            </div>
            <div className="tabular mt-1 text-2xl font-semibold">{v}</div>
          </div>
        ))}
        <div className="rounded-lg bg-panel-2 p-3">
          <div className="flex items-center gap-1 text-xs text-ink-2">
            {t("summary.parasitemia")}
            <button
              type="button"
              aria-expanded={help}
              aria-label={t("summary.parasitemiaHelp")}
              title={t("summary.parasitemiaHelp")}
              onClick={() => setHelp((h) => !h)}
              className="grid h-4 w-4 place-items-center rounded-full border border-ink-2 text-[10px] leading-none font-bold"
            >
              i
            </button>
          </div>
          <div className="tabular mt-1 text-2xl font-semibold">{pct}</div>
        </div>
      </div>
      {help && (
        <p className="mt-2 rounded-lg bg-panel-2 p-2 text-xs text-ink-2">{t("summary.parasitemiaHelp")}</p>
      )}
      <p className="mt-2 text-xs text-muted">{t("summary.note")}</p>
    </div>
  );
}

const BINS = 20;

/** Bars anchored to the baseline with 4px rounded tops. */
function barPath(x: number, y: number, w: number, h: number): string {
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function ClassHistogram({
  dets,
  label,
  threshold,
}: {
  dets: Detection[];
  label: ClassName;
  threshold: number;
}) {
  const { t } = useI18n();
  const counts = useMemo(() => histogram(dets, label, BINS), [dets, label]);
  const [hover, setHover] = useState<number | null>(null);
  const W = 480;
  const H = 96;
  const max = Math.max(1, ...counts);
  const bw = W / BINS;
  const color = label === "infected" ? "var(--c-i)" : "var(--c-u)";
  const readout =
    hover == null
      ? null
      : t("hist.bar", {
          from: (hover / BINS).toFixed(2),
          to: ((hover + 1) / BINS).toFixed(2),
          n: counts[hover],
        });
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex items-center justify-between gap-2 text-xs">
        <span className="flex items-center gap-1.5 font-medium">
          <ClassSwatch label={label} size={14} />
          {t(`class.${label}` as const)}
        </span>
        <span className="tabular text-muted">{readout ?? `max ${max}`}</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H + 14}`}
        className="w-full"
        role="img"
        aria-label={`${t("hist.title")}: ${t(`class.${label}` as const)}`}
      >
        <line x1={0} x2={W} y1={H} y2={H} stroke="var(--c-border)" />
        {counts.map((n, i) => {
          const h = (n / max) * (H - 4);
          const above = (i + 1) / BINS > threshold;
          return (
            <g key={i} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={i * bw} y={0} width={bw} height={H} fill="transparent" />
              {n > 0 && (
                <path
                  d={barPath(i * bw + 1, H - h, bw - 2, h)}
                  fill={color}
                  opacity={above ? 1 : 0.32}
                  stroke={hover === i ? "var(--c-text)" : "none"}
                />
              )}
            </g>
          );
        })}
        <line
          x1={threshold * W}
          x2={threshold * W}
          y1={0}
          y2={H}
          stroke="var(--c-text)"
          strokeWidth={1.5}
          strokeDasharray="3 3"
        />
        <text x={0} y={H + 12} fontSize={12} fill="var(--c-muted)">
          0
        </text>
        <text x={W} y={H + 12} fontSize={12} fill="var(--c-muted)" textAnchor="end">
          1
        </text>
        <text
          x={Math.min(Math.max(threshold * W, 40), W - 40)}
          y={H + 12}
          fontSize={12}
          fill="var(--c-text-2)"
          textAnchor="middle"
        >
          {t("hist.threshold")} {threshold.toFixed(2)}
        </text>
      </svg>
    </figure>
  );
}

export function ScoreHistograms({ dets, thresholds }: { dets: Detection[]; thresholds: Thresholds }) {
  const { t } = useI18n();
  return (
    <div>
      <div className="grid gap-4 sm:grid-cols-2">
        {CLASSES.map((c) => (
          <ClassHistogram key={c} dets={dets} label={c} threshold={thresholds[c]} />
        ))}
      </div>
      <p className="mt-1 text-xs text-muted">{t("hist.caption")}</p>
    </div>
  );
}

export function CellTable({
  dets,
  total,
  selectedId,
  onSelect,
}: {
  dets: Detection[];
  total: number;
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const { t } = useI18n();
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "score", desc: true });
  const rows = useMemo(() => sortDetections(dets, sort.key, sort.desc), [dets, sort]);
  const rowRefs = useRef(new Map<number, HTMLTableRowElement>());

  useEffect(() => {
    if (selectedId != null) rowRefs.current.get(selectedId)?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);

  const header = (key: SortKey, label: string, align = "text-left") => (
    <th
      scope="col"
      className={`px-2 py-1.5 font-medium ${align}`}
      aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}
    >
      <button
        type="button"
        className="inline-flex items-center gap-1"
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key === "score" }))}
      >
        {label}
        <span aria-hidden="true" className="text-muted">
          {sort.key === key ? (sort.desc ? "▼" : "▲") : "↕"}
        </span>
      </button>
    </th>
  );

  return (
    <div>
      <p className="mb-2 text-xs text-muted">{t("table.count", { shown: dets.length, total })}</p>
      <div className="max-h-80 overflow-auto rounded-lg border border-line">
        <table className="tabular w-full text-sm">
          <thead className="sticky top-0 bg-panel-2 text-xs text-ink-2">
            <tr>
              {header("id", t("table.id"))}
              {header("label", t("table.class"))}
              {header("score", t("table.score"), "text-right")}
              <th scope="col" className="px-2 py-1.5 text-right font-medium">
                x
              </th>
              <th scope="col" className="px-2 py-1.5 text-right font-medium">
                y
              </th>
              <th scope="col" className="px-2 py-1.5 text-right font-medium">
                w
              </th>
              <th scope="col" className="px-2 py-1.5 text-right font-medium">
                h
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="px-2 py-4 text-center text-muted">
                  {t("table.empty")}
                </td>
              </tr>
            )}
            {rows.map((d) => {
              const sel = d.id === selectedId;
              return (
                <tr
                  key={d.id}
                  ref={(el) => {
                    if (el) rowRefs.current.set(d.id, el);
                    else rowRefs.current.delete(d.id);
                  }}
                  tabIndex={0}
                  aria-selected={sel}
                  onClick={() => onSelect(d.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(d.id);
                    }
                  }}
                  className={`cursor-pointer border-t border-line ${sel ? "bg-accent/15 font-semibold" : "hover:bg-panel-2"}`}
                >
                  <td className="px-2 py-1">{d.id}</td>
                  <td className="px-2 py-1">
                    <span className="inline-flex items-center gap-1.5">
                      <ClassSwatch label={d.label} size={14} />
                      {t(`class.${d.label}` as const)}
                    </span>
                  </td>
                  <td className="px-2 py-1 text-right">{d.score.toFixed(3)}</td>
                  {d.bbox.map((v, i) => (
                    <td key={i} className="px-2 py-1 text-right text-ink-2">
                      {Math.round(v)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ResultInfo({ result, model }: { result: InferenceResult; model: ModelInfo }) {
  const { t } = useI18n();
  const ck = (result.checkpoint.boxes ?? result.checkpoint) as Record<string, unknown>;
  const rows: [string, string][] = [
    [t("info.checkpoint"), `${ck.file} · sha256 ${ck.sha256} · epoch ${ck.epoch}`],
    [t("info.device"), result.device],
    [t("info.time"), `${(result.elapsed_ms / 1000).toFixed(2)} s`],
    [t("info.nms"), result.nms_iou.toFixed(2)],
    [t("info.thrSource"), model.threshold_source ?? "—"],
  ];
  if (result.checkpoint.encoder) rows.splice(1, 0, [t("info.encoder"), String(result.checkpoint.encoder)]);
  if (result.checkpoint.classifier)
    rows.splice(2, 0, [t("info.classifier"), String(result.checkpoint.classifier)]);
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="font-mono break-all text-ink-2">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
