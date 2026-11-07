import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type KeyboardEvent,
  type PointerEvent,
  type SetStateAction,
} from "react";
import { useI18n } from "../i18n";
import { BOX_STYLE, HALO, MARK, SELECTED } from "../lib/classes";
import { splitFromPointer } from "../lib/curtain";
import type { Detection } from "../types";

/** Centre the view on a box (original pixels); a new object = a new request. */
export interface FocusRequest {
  bbox: Detection["bbox"];
}

/** Zoom / pan in container pixels; null = fitted to the container. Owned by the parent so that several viewers of
 * the same image (comparison, robustness columns) stay synchronised by sharing one state. */
export interface View {
  s: number;
  x: number;
  y: number;
}

export type ViewState = [View | null, Dispatch<SetStateAction<View | null>>];

/** Before / after curtain: `src` + `dets` on the left of the split, these on the right (same coordinates). */
export interface Curtain {
  src: string;
  dets: Detection[];
  split: number; // fraction of the image width
  onSplit: (v: number) => void;
  labels: [string, string];
}

interface Props {
  src: string | null;
  /** Coordinate system of the boxes (original image pixels); the image is drawn stretched to it. */
  width: number;
  height: number;
  dets: Detection[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  focus: FocusRequest | null;
  view: ViewState;
  /** Boxes to mark as disagreements (double white outline + "≠" badge, not colour alone). */
  marked?: ReadonlySet<number>;
  busy?: boolean;
  label?: string;
  curtain?: Curtain;
}

const MAX_SCALE = 16;

/** Image with box overlay; wheel zoom around the cursor, drag to pan, click a box to select it. */
export function Viewer({
  src,
  width,
  height,
  dets,
  selectedId,
  onSelect,
  focus,
  view: [userView, setUserView],
  marked,
  busy,
  label,
  curtain,
}: Props) {
  const { t } = useI18n();
  const clipId = useId();
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const drag = useRef<{
    px: number;
    py: number;
    x: number;
    y: number;
    moved: boolean;
    id: number | null;
    split?: boolean;
  } | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fitted = useMemo<View>(() => {
    if (!size.w || !width) return { s: 1, x: 0, y: 0 };
    const s = Math.min(size.w / width, size.h / height);
    return { s, x: (size.w - width * s) / 2, y: (size.h - height * s) / 2 };
  }, [size.w, size.h, width, height]);

  const view = userView ?? fitted;
  const update = useCallback(
    (fn: (v: View) => View) => setUserView((v) => fn(v ?? fitted)),
    [setUserView, fitted],
  );
  const fit = () => setUserView(null);

  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) =>
      update((v) => {
        const s = Math.min(MAX_SCALE, Math.max(fitted.s * 0.5, v.s * factor));
        return { s, x: cx - ((cx - v.x) * s) / v.s, y: cy - ((cy - v.y) * s) / v.s };
      }),
    [update, fitted.s],
  );

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  // centre on a cell requested from a table / list (only a new request moves the view)
  const lastFocus = useRef<FocusRequest | null>(null);
  useEffect(() => {
    if (!focus || focus === lastFocus.current || !size.w) return;
    lastFocus.current = focus;
    const [bx, by, bw, bh] = focus.bbox;
    update((v) => {
      const s = Math.min(MAX_SCALE, Math.max(v.s, fitted.s * 3));
      return { s, x: size.w / 2 - (bx + bw / 2) * s, y: size.h / 2 - (by + bh / 2) * s };
    });
  }, [focus, size.w, size.h, fitted.s, update]);

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const idAttr = (e.target as Element).closest("[data-id]")?.getAttribute("data-id");
    const onHandle = !!(e.target as Element).closest("[data-split]");
    drag.current = {
      split: onHandle,
      px: e.clientX,
      py: e.clientY,
      x: view.x,
      y: view.y,
      moved: false,
      id: idAttr ? +idAttr : null,
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    if (d.split && curtain) {
      const r = e.currentTarget.getBoundingClientRect();
      curtain.onSplit(splitFromPointer(e.clientX, r.left, view.x, view.s, width));
      return;
    }
    const dx = e.clientX - d.px;
    const dy = e.clientY - d.py;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    if (d.moved) update((v) => ({ ...v, x: d.x + dx, y: d.y + dy }));
  };
  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved && !d.split) onSelect(d.id);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = 60;
    const keys: Record<string, () => void> = {
      "+": () => zoomAt(1.25, size.w / 2, size.h / 2),
      "=": () => zoomAt(1.25, size.w / 2, size.h / 2),
      "-": () => zoomAt(0.8, size.w / 2, size.h / 2),
      "0": fit,
      ArrowLeft: () => update((v) => ({ ...v, x: v.x + step })),
      ArrowRight: () => update((v) => ({ ...v, x: v.x - step })),
      ArrowUp: () => update((v) => ({ ...v, y: v.y + step })),
      ArrowDown: () => update((v) => ({ ...v, y: v.y - step })),
      Escape: () => onSelect(null),
    };
    if (keys[e.key]) {
      e.preventDefault();
      keys[e.key]();
    }
  };

  const s = view.s;
  const ordered =
    selectedId == null
      ? dets
      : [...dets.filter((d) => d.id !== selectedId), ...dets.filter((d) => d.id === selectedId)];

  const box = (d: Detection) => {
    const st = BOX_STYLE[d.label];
    const [x, y, w, h] = d.bbox;
    const sel = d.id === selectedId;
    const mark = marked?.has(d.id) ?? false;
    // infected labels whenever readable; uninfected only on large boxes, so dense fields stay legible
    const showLabel = sel || mark || Math.min(w, h) * s >= (d.label === "infected" ? 24 : 90);
    const fs = 11 / s;
    const text = `${mark ? "≠ " : ""}${st.letter} ${d.score.toFixed(2)}`;
    const pad = 4 / s;
    return (
      <g key={d.id} data-id={d.id} className="cursor-pointer">
        <title>{`#${d.id} ${d.label} ${d.score.toFixed(3)}${mark ? " ≠" : ""}`}</title>
        {mark && (
          <>
            <rect
              x={x - pad}
              y={y - pad}
              width={w + 2 * pad}
              height={h + 2 * pad}
              fill="none"
              stroke={HALO}
              strokeWidth={5 / s}
            />
            <rect
              x={x - pad}
              y={y - pad}
              width={w + 2 * pad}
              height={h + 2 * pad}
              fill="none"
              stroke={MARK}
              strokeWidth={2 / s}
              strokeDasharray={`${2 / s} ${2 / s}`}
            />
          </>
        )}
        <rect x={x} y={y} width={w} height={h} fill="none" stroke={HALO} strokeWidth={(st.width + 2) / s} />
        {sel && (
          <rect
            x={x}
            y={y}
            width={w}
            height={h}
            fill="none"
            stroke={SELECTED}
            strokeWidth={(st.width + 5) / s}
          />
        )}
        <rect
          x={x}
          y={y}
          width={w}
          height={h}
          fill={sel ? "rgba(255,255,255,0.12)" : "transparent"}
          pointerEvents="all"
          stroke={st.color}
          strokeWidth={st.width / s}
          strokeDasharray={st.dash.length ? st.dash.map((v) => v / s).join(" ") : undefined}
        />
        {showLabel && (
          <g pointerEvents="none">
            <rect
              x={x}
              y={y - fs * 1.45}
              width={(text.length * 6.6 + 6) / s}
              height={fs * 1.45}
              fill={mark ? "#111" : st.color}
            />
            <text x={x + 3 / s} y={y - fs * 0.38} fontSize={fs} fontWeight={700} fill="#fff">
              {text}
            </text>
          </g>
        )}
      </g>
    );
  };

  return (
    <div
      ref={ref}
      tabIndex={0}
      role="application"
      aria-label={label ? `${label}. ${t("viewer.help")}` : t("viewer.help")}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => (drag.current = null)}
      onKeyDown={onKeyDown}
      className="relative h-full w-full cursor-grab touch-none overflow-hidden rounded-xl bg-stage select-none active:cursor-grabbing"
    >
      {src ? (
        <div
          className="absolute top-0 left-0"
          style={{
            width,
            height,
            transform: `translate(${view.x}px, ${view.y}px) scale(${s})`,
            transformOrigin: "0 0",
          }}
        >
          <img
            src={src}
            width={width}
            height={height}
            alt=""
            draggable={false}
            className="absolute top-0 left-0 block max-w-none"
            style={{ width, height }}
          />
          {curtain && (
            <img
              src={curtain.src}
              width={width}
              height={height}
              alt=""
              draggable={false}
              className="absolute top-0 left-0 block max-w-none"
              style={{ width, height, clipPath: `inset(0 0 0 ${curtain.split * 100}%)` }}
            />
          )}
          <svg className="absolute inset-0" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
            {curtain ? (
              <>
                <defs>
                  <clipPath id={`${clipId}l`}>
                    <rect x={0} y={0} width={curtain.split * width} height={height} />
                  </clipPath>
                  <clipPath id={`${clipId}r`}>
                    <rect x={curtain.split * width} y={0} width={width} height={height} />
                  </clipPath>
                </defs>
                <g clipPath={`url(#${clipId}l)`}>{ordered.map(box)}</g>
                <g clipPath={`url(#${clipId}r)`}>{curtain.dets.map(box)}</g>
                <line
                  x1={curtain.split * width}
                  x2={curtain.split * width}
                  y1={0}
                  y2={height}
                  stroke="#111"
                  strokeWidth={6 / s}
                  pointerEvents="none"
                />
                <line
                  x1={curtain.split * width}
                  x2={curtain.split * width}
                  y1={0}
                  y2={height}
                  stroke="#fff"
                  strokeWidth={3 / s}
                  pointerEvents="none"
                />
                <g data-split="" className="cursor-ew-resize">
                  <rect
                    x={curtain.split * width - 14 / s}
                    y={0}
                    width={28 / s}
                    height={height}
                    fill="transparent"
                    pointerEvents="all"
                  />
                  <circle
                    cx={curtain.split * width}
                    cy={height / 2}
                    r={14 / s}
                    fill="#fff"
                    stroke="#111"
                    strokeWidth={1.5 / s}
                  />
                  <text
                    x={curtain.split * width}
                    y={height / 2 + 4.5 / s}
                    fontSize={13 / s}
                    textAnchor="middle"
                    fill="#111"
                    fontWeight={700}
                    pointerEvents="none"
                  >
                    ⇔
                  </text>
                </g>
              </>
            ) : (
              ordered.map(box)
            )}
          </svg>
        </div>
      ) : (
        <div className="absolute inset-0 grid place-items-center text-sm text-white/70">
          {t("viewer.empty")}
        </div>
      )}
      {label && (
        <div className="pointer-events-none absolute top-2 left-2 rounded-md bg-black/60 px-2 py-1 text-xs font-semibold text-white">
          {label}
        </div>
      )}
      {curtain && (
        <div className="pointer-events-none absolute top-2 right-2 rounded-md bg-black/60 px-2 py-1 text-xs font-semibold text-white">
          {curtain.labels[1]} →
        </div>
      )}
      {busy && (
        <div className="pointer-events-none absolute inset-0 animate-pulse bg-black/20" aria-hidden="true" />
      )}
      {src && (
        <div className="absolute right-2 bottom-2 flex items-center gap-1 rounded-lg bg-black/60 p-1 text-white">
          <span className="tabular px-1.5 text-xs" aria-live="polite">
            {Math.round(s * 100)}%
          </span>
          {[
            ["−", t("viewer.zoomOut"), () => zoomAt(0.8, size.w / 2, size.h / 2)],
            ["+", t("viewer.zoomIn"), () => zoomAt(1.25, size.w / 2, size.h / 2)],
            ["⤢", t("viewer.fit"), fit],
            ["1:1", t("viewer.actual"), () => zoomAt(1 / s, size.w / 2, size.h / 2)],
          ].map(([sym, title, fn]) => (
            <button
              key={title as string}
              type="button"
              title={title as string}
              aria-label={title as string}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={fn as () => void}
              className="min-w-7 rounded-md px-1.5 py-0.5 text-sm hover:bg-white/20"
            >
              {sym as string}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
