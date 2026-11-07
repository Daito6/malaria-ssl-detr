import type { ReactNode } from "react";
import { BOX_STYLE } from "../lib/classes";
import type { ClassName } from "../types";

export function Panel({
  title,
  children,
  actions,
}: {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="min-w-0 rounded-xl border border-line bg-panel p-4" aria-label={title}>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold tracking-wide text-ink-2 uppercase">{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}

/** Class marker: the same line style + letter as the boxes on the image, so it is readable without colour. */
export function ClassSwatch({ label, size = 18 }: { label: ClassName; size?: number }) {
  const st = BOX_STYLE[label];
  const color = label === "infected" ? "var(--c-i)" : "var(--c-u)";
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" aria-hidden="true" className="shrink-0">
      <rect
        x="2"
        y="2"
        width="14"
        height="14"
        rx="2"
        fill="none"
        stroke={color}
        strokeWidth={label === "infected" ? 2.75 : 1.75}
        strokeDasharray={st.dash.length ? "3.5 2.5" : undefined}
      />
      <text x="9" y="12.5" textAnchor="middle" fontSize="8.5" fontWeight="700" fill={color}>
        {st.letter}
      </text>
    </svg>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "default",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary";
  title?: string;
}) {
  const cls =
    variant === "primary"
      ? "bg-accent text-accent-ink border-transparent hover:opacity-90"
      : "bg-panel-2 text-ink border-line hover:border-ink-2";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${cls}`}
    >
      {children}
    </button>
  );
}
