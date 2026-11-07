import { describe, expect, it } from "vitest";
import type { Detection } from "../types";
import { baseName, csvField, toCsv } from "./export";

const dets: Detection[] = [
  { id: 0, label: "uninfected", score: 0.99123456, bbox: [10.5, 20.25, 100, 110.125] },
  { id: 3, label: "infected", score: 0.9, bbox: [0, 0, 50, 60] },
];

function parseCsv(text: string): string[][] {
  // minimal RFC 4180 parser for the test
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\r" && text[i + 1] === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      i++;
    } else field += c;
  }
  return rows;
}

describe("toCsv", () => {
  const csv = toCsv(dets, {
    imageName: 'smear, "A".png',
    modelId: "a_coco_detr_binary",
    thresholds: { uninfected: 0.94, infected: 0.87 },
  });
  const rows = parseCsv(csv);

  it("has a header and one row per detection", () => {
    expect(rows[0]).toEqual([
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
    ]);
    expect(rows).toHaveLength(3);
  });

  it("writes values that round-trip, with quoting", () => {
    expect(rows[1]).toEqual([
      'smear, "A".png',
      "a_coco_detr_binary",
      "0",
      "uninfected",
      "0.991235",
      "10.50",
      "20.25",
      "100.00",
      "110.13",
      "0.940000",
    ]);
    expect(rows[2][3]).toBe("infected");
    expect(rows[2][9]).toBe("0.870000");
  });

  it("ends lines with CRLF", () => {
    expect(csv.endsWith("\r\n")).toBe(true);
  });

  it("neutralises spreadsheet formulas in text fields", () => {
    expect(csvField("=HYPERLINK(1)")).toBe("'=HYPERLINK(1)");
    expect(csvField(-1)).toBe("-1");
  });

  it("empty result gives only the header", () => {
    expect(
      parseCsv(
        toCsv([], { imageName: "x.png", modelId: "m", thresholds: { uninfected: 0.5, infected: 0.5 } }),
      ),
    ).toHaveLength(1);
  });
});

describe("baseName", () => {
  it("strips the extension and unsafe characters", () => {
    expect(baseName("my smear (1).PNG")).toBe("my_smear_1_");
    expect(baseName(".png")).toBe("image");
  });
});
