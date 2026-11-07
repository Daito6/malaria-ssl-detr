import { useRef, useState, type DragEvent } from "react";
import { useI18n } from "../i18n";
import type { ClassName, ModelInfo, Thresholds } from "../types";
import { CLASSES } from "../types";
import { Button, ClassSwatch } from "./ui";

const ACCEPT = ["image/jpeg", "image/png"];

export function Uploader({
  fileName,
  maxBytes,
  onFile,
  onError,
}: {
  fileName: string | null;
  maxBytes: number;
  onFile: (f: File) => void;
  onError: (code: string) => void;
}) {
  const { t } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    const okType = ACCEPT.includes(f.type) || /\.(jpe?g|png)$/i.test(f.name);
    if (!okType) return onError("unsupported_format");
    if (f.size === 0) return onError("empty_file");
    if (f.size > maxBytes) return onError("file_too_large");
    onFile(f);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    take(e.dataTransfer.files);
  };

  return (
    <div>
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`flex w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed px-3 py-5 text-center transition ${
          over ? "border-accent bg-panel-2" : "border-line hover:border-ink-2"
        }`}
      >
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          aria-hidden="true"
          className="text-ink-2"
        >
          <path
            d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="text-sm font-medium">{fileName ? t("upload.replace") : t("upload.drop")}</span>
        <span className="text-xs text-muted">
          {t("upload.formats", { mb: Math.round(maxBytes / 1024 / 1024) })}
        </span>
      </button>
      {fileName && (
        <p className="mt-2 truncate text-xs text-ink-2" title={fileName}>
          {fileName}
        </p>
      )}
      <input
        ref={input}
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          take(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}

export function ModelPicker({
  models,
  value,
  onChange,
}: {
  models: ModelInfo[];
  value: string | null;
  onChange: (id: string) => void;
}) {
  const { t, lang } = useI18n();
  return (
    <div role="radiogroup" aria-label={t("model.title")} className="flex flex-col gap-2">
      {models.map((m) => {
        const checked = m.id === value;
        return (
          <label
            key={m.id}
            className={`flex cursor-pointer gap-2 rounded-lg border p-2.5 text-sm transition ${
              checked ? "border-accent bg-panel-2" : "border-line hover:border-ink-2"
            } ${m.available ? "" : "cursor-not-allowed opacity-60"}`}
          >
            <input
              type="radio"
              name="model"
              className="mt-1 accent-[var(--c-accent)]"
              checked={checked}
              disabled={!m.available}
              onChange={() => onChange(m.id)}
            />
            <span className="min-w-0">
              <span className="block font-medium">{m.name[lang]}</span>
              <span className="mt-0.5 inline-block rounded bg-panel-2 px-1.5 text-[11px] text-ink-2">
                {t(m.kind === "detr" ? "model.kind.detr" : "model.kind.two_stage")}
              </span>
              <span className="mt-1 block text-xs text-muted">{m.description[lang]}</span>
              {!m.available && (
                <span className="mt-1 block text-xs break-words text-danger-ink">
                  {t("model.unavailable")}: {m.error}
                </span>
              )}
            </span>
          </label>
        );
      })}
    </div>
  );
}

export function ThresholdControls({
  model,
  value,
  onChange,
  onReset,
}: {
  model: ModelInfo;
  value: Thresholds;
  onChange: (c: ClassName, v: number) => void;
  onReset: () => void;
}) {
  const { t } = useI18n();
  return (
    <div className="flex flex-col gap-4">
      {CLASSES.map((c) => {
        const def = model.default_thresholds[c];
        const id = `thr-${c}`;
        return (
          <div key={c}>
            <div className="flex items-center justify-between gap-2">
              <label htmlFor={id} className="flex items-center gap-2 text-sm font-medium">
                <ClassSwatch label={c} />
                {t(`class.${c}`)}
              </label>
              <span className="tabular text-sm font-semibold">{value[c].toFixed(3)}</span>
            </div>
            <input
              id={id}
              type="range"
              min={0}
              max={1}
              step={0.001}
              value={value[c]}
              onChange={(e) => onChange(c, +e.target.value)}
              className="mt-1 w-full"
            />
            <p className="text-xs text-muted">
              {def != null
                ? t("thr.fromVal", { v: def.toFixed(3) })
                : t("thr.noVal", { v: value[c].toFixed(1) })}
            </p>
          </div>
        );
      })}
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-muted">{t("thr.hint")}</p>
        <Button onClick={onReset}>{t("thr.reset")}</Button>
      </div>
    </div>
  );
}

export function ClassFilter({
  shown,
  onToggle,
}: {
  shown: ReadonlySet<ClassName>;
  onToggle: (c: ClassName) => void;
}) {
  const { t } = useI18n();
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="sr-only">{t("filter.title")}</legend>
      {CLASSES.map((c) => (
        <label key={c} className="flex cursor-pointer items-start gap-2 text-sm">
          <input
            type="checkbox"
            className="mt-1 accent-[var(--c-accent)]"
            checked={shown.has(c)}
            onChange={() => onToggle(c)}
          />
          <ClassSwatch label={c} />
          <span>
            <span className="font-medium">{t(`class.${c}`)}</span>
            <span className="block text-xs text-muted">{t(`class.${c}.short`)}</span>
          </span>
        </label>
      ))}
    </fieldset>
  );
}

export function ProgressBar({
  stage,
  progress,
  queuePos,
}: {
  stage: string;
  progress: number;
  queuePos: number;
}) {
  const { t, maybe } = useI18n();
  const label = maybe(`run.stage.${stage}`) ?? stage;
  return (
    <div className="flex flex-col gap-1" role="status" aria-live="polite">
      <div className="flex justify-between text-xs text-ink-2">
        <span>
          {label}
          {queuePos > 0 && ` · ${t("run.queuePos", { n: queuePos })}`}
        </span>
        <span className="tabular">{Math.round(progress * 100)}%</span>
      </div>
      <div
        className="h-1.5 overflow-hidden rounded-full bg-panel-2"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
        aria-label={label}
      >
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-300"
          style={{ width: `${Math.max(4, progress * 100)}%` }}
        />
      </div>
    </div>
  );
}
