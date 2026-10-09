import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { Dataset, RunIndex, Source } from "../api/source";

export interface AppState {
  source: Source;
  index: RunIndex;
  runId: string;
  setRunId: (id: string) => void;
  res: any;
  labels: Record<string, string>;
  refreshRuns: () => Promise<void>;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ value, children }: { value: AppState; children: ReactNode }) {
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp fuera de AppProvider");
  return v;
}

/** Conjunto depurado de la corrida (se carga sólo en las secciones que lo usan). */
export function useDataset(): { data: Dataset | null; error: unknown } {
  const { source, runId } = useApp();
  const [state, setState] = useState<{ data: Dataset | null; error: unknown }>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    source
      .dataset(runId)
      .then((d) => alive && setState({ data: d, error: null }))
      .catch((e) => alive && setState({ data: null, error: e }));
    return () => {
      alive = false;
    };
  }, [source, runId]);
  return state;
}

export function labelsFrom(res: any): Record<string, string> {
  const out: Record<string, string> = {};
  for (const v of res?.dictionary ?? []) out[v.name] = v.label;
  out.logPop = out.logPop ?? "Población (log)";
  out.logIncome = out.logIncome ?? "Renta mediana (log)";
  out.logStudy = out.logStudy ?? "Ensayos clínicos (log)";
  return out;
}

export const REGION_ORDER = ["Sur", "Medio Oeste", "Oeste", "Noreste"] as const;
/** Color fijo por región (sigue a la entidad, nunca al rango) y forma como segunda codificación. */
export const REGION_STYLE: Record<string, { color: keyof import("./theme").ThemeTokens; symbol: string }> = {
  Sur: { color: "s1", symbol: "circle" },
  "Medio Oeste": { color: "s2", symbol: "square" },
  Oeste: { color: "s3", symbol: "diamond" },
  Noreste: { color: "s4", symbol: "triangle-up" },
};
