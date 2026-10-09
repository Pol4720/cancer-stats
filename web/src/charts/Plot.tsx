import { useEffect, useRef, useState } from "react";
import { tokens, useTheme } from "../lib/theme";

type PlotlyModule = typeof import("plotly.js-dist-min");
let plotly: Promise<PlotlyModule> | null = null;
function loadPlotly(): Promise<PlotlyModule> {
  plotly ??= import("plotly.js-dist-min").then((m) => ((m as any).default ?? m) as PlotlyModule);
  return plotly;
}

export type Tokens = ReturnType<typeof tokens>;

function wrap(label: string, width: number): string {
  if (label.length <= width) return label;
  const words = label.split(" ");
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    if ((cur + " " + w).trim().length > width && cur) {
      lines.push(cur);
      cur = w;
    } else cur = (cur + " " + w).trim();
  }
  if (cur) lines.push(cur);
  return lines.join("<br>");
}

export interface PlotProps {
  /** Construye las trazas con los colores del tema vigente. */
  data: (t: Tokens) => any[];
  layout?: (t: Tokens) => Record<string, any>;
  height?: number;
  name: string;
  ariaLabel: string;
  /** Valores de los que dependen las trazas: el gráfico se redibuja cuando cambian. */
  deps?: unknown[];
}

/** Gráfico Plotly cargado bajo demanda, con el tema (claro/oscuro) y el formato en castellano. */
export function Plot({ data, layout, height = 360, name, ariaLabel, deps = [] }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const build = useRef({ data, layout });
  build.current = { data, layout };
  const key = JSON.stringify(deps);
  const { resolved } = useTheme();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let alive = true;
    const el = ref.current;
    loadPlotly().then((P) => {
      if (!alive || !el) return;
      const t = tokens();
      const base: Record<string, any> = {
        height,
        margin: { l: 64, r: 18, t: 16, b: 52 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: getComputedStyle(document.body).fontFamily, size: 13, color: t.ink2 },
        separators: ", ",
        hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink, size: 13 } },
        legend: { orientation: "h", y: -0.2, x: 0, font: { color: t.ink2 } },
        xaxis: { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickcolor: t.axis, automargin: true },
        yaxis: { gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickcolor: t.axis, automargin: true },
        transition: { duration: 350, easing: "cubic-in-out" },
      };
      const { data: makeData, layout: makeLayout } = build.current;
      const extra = makeLayout ? makeLayout(t) : {};
      const merged: Record<string, any> = {
        ...base,
        ...extra,
        xaxis: { ...base.xaxis, ...(extra.xaxis ?? {}) },
        yaxis: { ...base.yaxis, ...(extra.yaxis ?? {}) },
      };
      for (const k of Object.keys(extra)) {
        if (/^[xy]axis\d+$/.test(k)) merged[k] = { ...base[k.slice(0, 5)], ...extra[k] };
      }
      const traces = makeData(t);
      // Pantallas estrechas: etiquetas de categoría partidas en líneas y margen izquierdo acotado.
      if (window.innerWidth < 640) {
        merged.margin = { ...(merged.margin ?? {}), l: Math.min(merged.margin?.l ?? 64, 132), r: 8 };
        merged.font = { ...merged.font, size: 11 };
        let cats = 0;
        let lines = 1;
        for (const tr of traces) {
          if (tr && Array.isArray(tr.y) && typeof tr.y[0] === "string" && tr.type !== "heatmap") {
            tr.y = tr.y.map((v: string) => wrap(v, 20));
            cats = Math.max(cats, new Set(tr.y).size);
            lines = Math.max(lines, ...tr.y.map((v: string) => v.split("<br>").length));
          }
        }
        if (cats) merged.height = Math.max(height, cats * (14 * lines + 14) + 90);
      }
      P.react(el, traces, merged, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
        toImageButtonOptions: { filename: name, format: "png", scale: 2 },
        locale: "es",
      });
      el.style.minHeight = `${merged.height}px`;
      setReady(true);
    });
    return () => {
      alive = false;
    };
  }, [key, height, name, resolved]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) loadPlotly().then((P) => P.purge(el));
    };
  }, []);

  return (
    <div className="plot-host" role="img" aria-label={ariaLabel}>
      {!ready && <div className="plot-skeleton" style={{ minHeight: height }}>Cargando gráfico…</div>}
      <div ref={ref} className="plot" style={{ minHeight: ready ? height : 0 }} />
    </div>
  );
}
