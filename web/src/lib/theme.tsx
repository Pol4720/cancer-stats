import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type ThemeChoice = "system" | "light" | "dark";
type Resolved = "light" | "dark";

interface ThemeValue {
  choice: ThemeChoice;
  resolved: Resolved;
  setChoice: (c: ThemeChoice) => void;
}

const ThemeContext = createContext<ThemeValue>({ choice: "system", resolved: "light", setChoice: () => undefined });

function systemTheme(): Resolved {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function readChoice(): ThemeChoice {
  try {
    const t = localStorage.getItem("cs-theme");
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoiceState] = useState<ThemeChoice>(readChoice);
  const [system, setSystem] = useState<Resolved>(systemTheme);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const on = () => setSystem(mq.matches ? "dark" : "light");
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const setChoice = useCallback((c: ThemeChoice) => {
    setChoiceState(c);
    try {
      if (c === "system") localStorage.removeItem("cs-theme");
      else localStorage.setItem("cs-theme", c);
    } catch {
      // almacenamiento no disponible: el tema dura lo que la pestaña
    }
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (choice === "system") delete root.dataset.theme;
    else root.dataset.theme = choice;
  }, [choice]);

  const value = useMemo(
    () => ({ choice, resolved: choice === "system" ? system : choice, setChoice }),
    [choice, system, setChoice],
  );
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeValue {
  return useContext(ThemeContext);
}

/** Lee los tokens de color vigentes (cambian con el tema). */
export type ThemeTokens = {
  surface: string; ink: string; ink2: string; muted: string; grid: string; axis: string;
  s1: string; s2: string; s3: string; s4: string; s5: string; s6: string; s7: string; s8: string;
  critical: string; good: string;
};

export function tokens(): ThemeTokens {
  const cs = getComputedStyle(document.documentElement);
  const get = (n: string) => cs.getPropertyValue(n).trim();
  return {
    surface: get("--surface"),
    ink: get("--ink"),
    ink2: get("--ink-2"),
    muted: get("--muted"),
    grid: get("--grid"),
    axis: get("--axis"),
    s1: get("--s1"),
    s2: get("--s2"),
    s3: get("--s3"),
    s4: get("--s4"),
    s5: get("--s5"),
    s6: get("--s6"),
    s7: get("--s7"),
    s8: get("--s8"),
    critical: get("--critical"),
    good: get("--good"),
  };
}
