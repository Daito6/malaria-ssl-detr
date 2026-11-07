import type { ClassName } from "../types";

/**
 * Box styles on the image. Classes differ by colour AND by line style and letter, so they stay distinguishable
 * without colour (dashed thin "U" vs solid thick "I"). Colours: slots 1-2 of the validated categorical palette
 * (blue / orange, colour-blind safe pair); a dark halo under every box keeps them visible on any stain.
 */
export const BOX_STYLE: Record<ClassName, { color: string; width: number; dash: number[]; letter: string }> =
  {
    uninfected: { color: "#2a78d6", width: 1.5, dash: [6, 4], letter: "U" },
    infected: { color: "#eb6834", width: 2.75, dash: [], letter: "I" },
  };

export const HALO = "rgba(0, 0, 0, 0.6)";
export const SELECTED = "#ffffff";
/** Disagreement marker (comparison mode): white dotted outer frame on a dark halo + "≠" in the label. */
export const MARK = "#ffffff";
