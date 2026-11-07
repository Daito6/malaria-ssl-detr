import type { Detection, Thresholds } from "../types";
import { BOX_STYLE, HALO } from "./classes";

export interface ExportMeta {
  imageName: string;
  modelId: string;
  thresholds: Thresholds;
}

const CSV_HEADER = [
  "image",
  "model_id",
  "cell_id",
  "class",
  "confidence",
  "x",
  "y",
  "width",
  "height",
  "class_threshold",
];

/** RFC 4180 field: quoted when it contains a comma, quote or line break; a leading = + - @ is neutralised. */
export function csvField(v: string | number): string {
  let s = String(v);
  if (typeof v === "string" && /^[=+\-@]/.test(s)) s = `'${s}`;
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** One row per detection (pixel coordinates of the original image). */
export function toCsv(dets: Detection[], meta: ExportMeta): string {
  const rows = dets.map((d) =>
    [
      meta.imageName,
      meta.modelId,
      d.id,
      d.label,
      d.score.toFixed(6),
      d.bbox[0].toFixed(2),
      d.bbox[1].toFixed(2),
      d.bbox[2].toFixed(2),
      d.bbox[3].toFixed(2),
      meta.thresholds[d.label].toFixed(6),
    ]
      .map(csvField)
      .join(","),
  );
  return [CSV_HEADER.join(","), ...rows].join("\r\n") + "\r\n";
}

export function baseName(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_") || "image";
}

/** Full-resolution annotated image: the boxes as on screen plus a footer with model, thresholds and disclaimer. */
export function drawAnnotated(img: HTMLImageElement, dets: Detection[], footer: string[]): HTMLCanvasElement {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const unit = Math.max(1, Math.max(w, h) / 1000);
  const font = Math.round(13 * unit);
  const footerH = Math.round((footer.length * 1.5 + 1) * font);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h + footerH;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  ctx.drawImage(img, 0, 0);
  ctx.font = `600 ${font}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  for (const d of dets) {
    const st = BOX_STYLE[d.label];
    const [x, y, bw, bh] = d.bbox;
    ctx.setLineDash([]);
    ctx.strokeStyle = HALO;
    ctx.lineWidth = (st.width + 2) * unit;
    ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash(st.dash.map((v) => v * unit));
    ctx.strokeStyle = st.color;
    ctx.lineWidth = st.width * unit;
    ctx.strokeRect(x, y, bw, bh);
    const text = `${st.letter} ${d.score.toFixed(2)}`;
    const tw = ctx.measureText(text).width + 6 * unit;
    const ty = y > font + 4 * unit ? y : y + font + 4 * unit;
    ctx.fillStyle = st.color;
    ctx.fillRect(x, ty - font - 2 * unit, tw, font + 2 * unit);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, x + 3 * unit, ty);
  }
  ctx.setLineDash([]);
  ctx.fillStyle = "#111111";
  ctx.fillRect(0, h, w, footerH);
  ctx.fillStyle = "#ffffff";
  ctx.font = `${font}px system-ui, sans-serif`;
  footer.forEach((line, i) => ctx.fillText(line, font * 0.75, h + (i + 1) * font * 1.5 + font * 0.25));
  return canvas;
}

export function download(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
