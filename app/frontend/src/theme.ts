import { useEffect, useState } from "react";

export type ThemePref = "light" | "dark" | "system";

function stored(): ThemePref {
  try {
    const v = localStorage.getItem("smearscan.theme");
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    /* storage unavailable */
  }
  return "system";
}

/** Theme preference; sets <html data-theme> to the resolved light / dark value. */
export function useTheme(): [ThemePref, (t: ThemePref) => void] {
  const [pref, setPref] = useState<ThemePref>(stored);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = pref === "system" ? (mq.matches ? "dark" : "light") : pref;
    };
    apply();
    try {
      localStorage.setItem("smearscan.theme", pref);
    } catch {
      /* storage unavailable */
    }
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [pref]);
  return [pref, setPref];
}
