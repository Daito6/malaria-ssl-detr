import { useMemo, useState, type KeyboardEvent } from "react";
import type { CurveRow } from "../types";

export interface Series {
  id: string;
  name: string;
  color: string; // CSS colour (token)
  rows: CurveRow[]; // sorted by strength
}

type Metric = "AP50" | "recall";

const W = 640;
const H = 280;
const M = { l: 44, r: 150, t: 14, b: 36 };
const LABEL_CHARS = 20; // direct labels are shortened; the legend above carries the full name
const short = (name: string) => (name.length > LABEL_CHARS ? `${name.slice(0, LABEL_CHARS - 1)}…` : name);
const PW = W - M.l - M.r;
const PH = H - M.t - M.b;

const val = (r: CurveRow, m: Metric, k: "" | "_lo" | "_hi" = "") =>
  r[`${m}${k}` as keyof CurveRow] as number | null;

/** Metric vs strength, one line per model with its 95% CI band; direct labels, crosshair tooltip, keyboard. */
export function CurveChart({ series, metric, title }: { series: Series[]; metric: Metric; title: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const strengths = useMemo(
    () => [...new Set(series.flatMap((s) => s.rows.map((r) => r.strength)))].sort((a, b) => a - b),
    [series],
  );
  const values = series.flatMap((s) =>
    s.rows.flatMap((r) => [val(r, metric, "_lo"), val(r, metric, "_hi"), val(r, metric)]),
  );
  const finite = values.filter((v): v is number => v != null && Number.isFinite(v));
  const lo = Math.max(0, Math.floor(((finite.length ? Math.min(...finite) : 0) - 0.02) * 20) / 20);
  const hi = Math.min(1, Math.ceil(((finite.length ? Math.max(...finite) : 1) + 0.02) * 20) / 20);
  const step = hi - lo > 0.5 ? 0.1 : 0.05;
  const ticks: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(Math.round(v * 100) / 100);
  const x = (s: number) => M.l + s * PW;
  const y = (v: number) => M.t + (1 - (v - lo) / (hi - lo || 1)) * PH;

  // direct labels at the last point, spread to avoid collisions
  const labels = series
    .map((s) => {
      const last = [...s.rows].reverse().find((r) => val(r, metric) != null);
      return last ? { s, y: y(val(last, metric)!) } : null;
    })
    .filter((v): v is { s: Series; y: number } => v !== null)
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);

  const onMove = (clientX: number, el: SVGSVGElement) => {
    const r = el.getBoundingClientRect();
    const sx = ((clientX - r.left) / r.width) * W;
    const s = (sx - M.l) / PW;
    let best = 0;
    strengths.forEach((v, i) => {
      if (Math.abs(v - s) < Math.abs(strengths[best] - s)) best = i;
    });
    setHover(strengths.length ? best : null);
  };
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (!strengths.length) return;
    if (e.key === "ArrowRight") setHover((h) => Math.min(strengths.length - 1, (h ?? -1) + 1));
    else if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? strengths.length) - 1));
    else return;
    e.preventDefault();
  };
  const hs = hover != null ? strengths[hover] : null;

  return (
    <figure className="relative min-w-0">
      <figcaption className="mb-1 text-sm font-semibold">{title}</figcaption>
      <div className="mb-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-hidden="true">
        {series.map((s) => (
          <span key={s.id} className="inline-flex items-center gap-1.5">
            <svg width="18" height="8">
              <line x1="0" x2="18" y1="4" y2="4" stroke={s.color} strokeWidth="2" />
              <circle cx="9" cy="4" r="3" fill={s.color} />
            </svg>
            {s.name}
          </span>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={title}
        tabIndex={0}
        onPointerMove={(e) => onMove(e.clientX, e.currentTarget)}
        onPointerLeave={() => setHover(null)}
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
      >
        {ticks.map((v) => (
          <g key={v}>
            <line x1={M.l} x2={M.l + PW} y1={y(v)} y2={y(v)} stroke="var(--c-border)" strokeWidth={1} />
            <text x={M.l - 6} y={y(v) + 4} fontSize={11} textAnchor="end" fill="var(--c-muted)">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((s) => (
          <text key={s} x={x(s)} y={H - M.b + 18} fontSize={11} textAnchor="middle" fill="var(--c-muted)">
            {s}
          </text>
        ))}
        {series.map((s) => {
          const pts = s.rows.filter((r) => val(r, metric) != null);
          const band = pts.filter((r) => val(r, metric, "_lo") != null && val(r, metric, "_hi") != null);
          const poly = [
            ...band.map((r) => `${x(r.strength)},${y(val(r, metric, "_hi")!)}`),
            ...[...band].reverse().map((r) => `${x(r.strength)},${y(val(r, metric, "_lo")!)}`),
          ].join(" ");
          return (
            <g key={s.id}>
              {band.length > 1 && <polygon points={poly} fill={s.color} opacity={0.14} />}
              <polyline
                points={pts.map((r) => `${x(r.strength)},${y(val(r, metric)!)}`).join(" ")}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
              />
              {pts.map((r) => (
                <circle
                  key={r.strength}
                  cx={x(r.strength)}
                  cy={y(val(r, metric)!)}
                  r={4}
                  fill={s.color}
                  stroke="var(--c-panel)"
                  strokeWidth={2}
                />
              ))}
            </g>
          );
        })}
        {labels.map(({ s, y: ly }) => (
          <text key={s.id} x={M.l + PW + 8} y={ly + 4} fontSize={11} fill="var(--c-text-2)">
            <tspan fill={s.color} fontWeight={700}>
              —{" "}
            </tspan>
            {short(s.name)}
            <title>{s.name}</title>
          </text>
        ))}
        {hs != null && (
          <line
            x1={x(hs)}
            x2={x(hs)}
            y1={M.t}
            y2={M.t + PH}
            stroke="var(--c-text)"
            strokeWidth={1}
            opacity={0.5}
          />
        )}
        <rect x={M.l} y={M.t} width={PW} height={PH} fill="transparent" />
      </svg>
      {hs != null && (
        <div
          className="pointer-events-none absolute top-8 rounded-lg border border-line bg-panel px-3 py-2 text-xs shadow-lg"
          style={{ left: `${Math.min(70, (x(hs) / W) * 100)}%` }}
          role="status"
        >
          <div className="mb-1 text-muted">s = {hs}</div>
          {series.map((s) => {
            const r = s.rows.find((q) => q.strength === hs);
            const v = r ? val(r, metric) : null;
            return (
              <div key={s.id} className="flex items-center gap-2">
                <svg width="12" height="4" aria-hidden="true">
                  <line x1="0" x2="12" y1="2" y2="2" stroke={s.color} strokeWidth="2" />
                </svg>
                <strong className="tabular">{v == null ? "—" : v.toFixed(3)}</strong>
                {r && val(r, metric, "_lo") != null && (
                  <span className="tabular text-muted">
                    [{val(r, metric, "_lo")!.toFixed(3)}, {val(r, metric, "_hi")!.toFixed(3)}]
                  </span>
                )}
                <span className="text-ink-2">{s.name}</span>
              </div>
            );
          })}
        </div>
      )}
    </figure>
  );
}
