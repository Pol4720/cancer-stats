import { useMemo, useState } from "react";
import { Plot } from "../charts/Plot";
import { Card, DataTable, ErrorBox, Explain, Loading, Section, Segmented, Stat } from "../components/ui";
import { ci, int, num, pval } from "../lib/format";
import { finitePairs, lowess, simpleOls } from "../lib/stats";
import { REGION_ORDER, REGION_STYLE, useApp, useDataset } from "../lib/store";

const NUMERIC_EXCLUDE = new Set(["TARGET_deathRate", "avgDeathsPerYear", "Notificadomuerte", "avgAnnCount"]);

function Scatter() {
  const { labels, res } = useApp();
  const { data, error } = useDataset();
  const candidates = useMemo(
    () => (data ? data.columns.filter((c) => typeof data.data[c].find((v) => v !== null) === "number" && !NUMERIC_EXCLUDE.has(c)) : []),
    [data],
  );
  const [xvar, setX] = useState("incidenceRate");
  const [smooth, setSmooth] = useState<"lowess" | "recta">("lowess");
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading what="el conjunto depurado" />;
  const y = data.data.TARGET_deathRate;
  const [xs, ys, idx] = finitePairs(data.data[xvar], y);
  const fit = simpleOls(xs, ys);
  const sm = smooth === "lowess" ? lowess(xs, ys, 0.35) : null;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const region = data.data.region as string[];
  const county = data.data.county_id as string[];
  return (
    <Card
      title="Explorador: mortalidad frente a una explicativa"
      sub={`r de Pearson = ${num(fit.r, 3)} (n = ${int(xs.length)}). Color y forma identifican la región.`}
      actions={
        <>
          <select className="select" value={xvar} onChange={(e) => setX(e.target.value)} aria-label="Variable del eje horizontal">
            {candidates.map((c) => (
              <option key={c} value={c}>
                {labels[c] ?? c}
              </option>
            ))}
          </select>
          <Segmented label="Tendencia" value={smooth} onChange={setSmooth} options={[{ value: "lowess", label: "LOWESS" }, { value: "recta", label: "Recta MCO" }]} />
        </>
      }
    >
      <Plot
        name={`dispersion-${xvar}`}
        ariaLabel={`Mortalidad frente a ${labels[xvar] ?? xvar}`}
        height={440}
        deps={[res.run_id, xvar, smooth]}
        data={(t) => [
          ...REGION_ORDER.map((reg) => {
            const sel = idx.map((i, k) => [i, k] as const).filter(([i]) => region[i] === reg);
            const st = REGION_STYLE[reg];
            return {
              type: "scattergl",
              mode: "markers",
              name: reg,
              x: sel.map(([, k]) => xs[k]),
              y: sel.map(([, k]) => ys[k]),
              text: sel.map(([i]) => county[i]),
              marker: { color: (t as any)[st.color], symbol: st.symbol, size: 8, opacity: 0.6, line: { width: 1, color: t.surface } },
              hovertemplate: "%{text}<br>x = %{x}<br>Mortalidad = %{y:.1f}<extra>" + reg + "</extra>",
            };
          }),
          sm
            ? { type: "scatter", mode: "lines", name: "LOWESS", x: sm.x, y: sm.y, line: { color: t.ink, width: 2.5 }, hoverinfo: "skip" }
            : { type: "scatter", mode: "lines", name: "Recta MCO", x: [minX, maxX], y: [fit.a + fit.b * minX, fit.a + fit.b * maxX], line: { color: t.ink, width: 2.5 }, hoverinfo: "skip" },
        ]}
        layout={() => ({
          xaxis: { title: { text: labels[xvar] ?? xvar } },
          yaxis: { title: { text: "Mortalidad por cáncer (por 100 000)" } },
          legend: { orientation: "h", y: -0.18 },
        })}
      />
      <Explain title="Cómo leer este gráfico">
        <p>
          La curva LOWESS sigue la tendencia sin imponer una forma; si se separa claramente de una recta, la relación no es lineal y conviene
          transformar la variable (como se hizo con la población, la renta y los ensayos, en logaritmos). La correlación marginal no es el efecto
          del modelo: allí cada coeficiente se ajusta por las demás variables.
        </p>
      </Explain>
    </Card>
  );
}

export default function Explore() {
  const { res, labels } = useApp();
  const x = res.exploration;
  const r = x.response;
  const [method, setMethod] = useState<"pearson" | "spearman">("pearson");
  const corr = [...x.correlations].sort((a: any, b: any) => (method === "pearson" ? a.pearson - b.pearson : a.spearman - b.spearman));
  const cm = method === "pearson" ? x.corr_matrix : x.corr_matrix_spearman;
  const regions = [...x.region.groups].sort((a: any, b: any) => REGION_ORDER.indexOf(a.group) - REGION_ORDER.indexOf(b.group));

  return (
    <Section
      eyebrow="4 · Exploración"
      title="La respuesta y sus asociaciones"
      lead="Antes de modelar: forma de la distribución de la mortalidad, inferencia clásica sobre su centro y su dispersión, y relaciones marginales con cada explicativa."
    >
      <div className="grid grid-4">
        <Stat label="Media (IC 95 %)" value={num(r.mean_ci.estimate, 1)} note={ci(r.mean_ci.lower, r.mean_ci.upper, 1)} />
        <Stat label="Mediana (IC por orden)" value={num(r.median_ci_order.estimate, 1)} note={ci(r.median_ci_order.lower, r.median_ci_order.upper, 1)} />
        <Stat label="Desviación típica" value={num(r.variance_ci.sd, 1)} note={`IC σ: ${ci(r.variance_ci.sd_lower, r.variance_ci.sd_upper, 1)}`} />
        <Stat label="Asimetría · curtosis" value={`${num(r.shape.skewness, 2)} · ${num(r.shape.kurtosis_excess, 2)}`} note="curtosis en exceso sobre la normal" />
      </div>

      <div className="grid grid-2">
        <Card title="Distribución de la mortalidad" sub="Histograma con la densidad normal ajustada y la estimación núcleo.">
          <Plot
            name="histograma-mortalidad"
            ariaLabel="Histograma de la mortalidad por cáncer"
            deps={[res.run_id]}
            data={(t) => {
              const e = r.histogram.edges as number[];
              const mids = e.slice(0, -1).map((v, i) => (v + e[i + 1]) / 2);
              return [
                { type: "bar", x: mids, y: r.histogram.counts, width: e[1] - e[0] - 0.8, name: "Condados", marker: { color: t.s1, opacity: 0.75, cornerradius: 3 }, hovertemplate: "%{x:.0f}: %{y} condados<extra></extra>" },
                { type: "scatter", mode: "lines", x: r.histogram.grid, y: r.histogram.normal, name: "Normal", line: { color: t.s2, width: 2, dash: "dash" } },
                { type: "scatter", mode: "lines", x: r.histogram.grid, y: r.histogram.kde, name: "Núcleo", line: { color: t.ink, width: 2 } },
              ];
            }}
            layout={() => ({ xaxis: { title: { text: "Muertes por 100 000 hab." } }, yaxis: { title: { text: "Condados" } }, bargap: 0 })}
          />
        </Card>
        <Card title="Contrastes de normalidad" sub={`Box-Cox: λ = ${num(r.boxcox.lambda, 2)} ${ci(r.boxcox.lower, r.boxcox.upper, 2)}`}>
          <DataTable
            name="normalidad-respuesta"
            csv={false}
            rows={r.normality}
            columns={[
              { key: "test", header: "Contraste" },
              { key: "statistic", header: "Estadístico", num: true, render: (t: any) => num(t.statistic, 4) },
              { key: "p", header: "p", num: true, render: (t: any) => pval(t.p) },
            ]}
          />
          <Explain>
            <p>
              Con 3 047 condados cualquier desviación mínima de la normalidad es «significativa». Lo relevante es su tamaño: asimetría
              {" "}{num(r.shape.skewness, 2)} y colas algo más pesadas que la normal. Para el modelo, la hipótesis de normalidad recae en los
              residuos, no en la respuesta, y con este tamaño muestral la inferencia es robusta por el teorema central del límite.
            </p>
          </Explain>
        </Card>
      </div>

      <Scatter />

      <div className="stack">
        <Card
          title="Correlación con la mortalidad"
          sub="IC 95 % por la transformación z de Fisher; p corregidos por Holm."
          actions={<Segmented label="Método" value={method} onChange={setMethod} options={[{ value: "pearson", label: "Pearson" }, { value: "spearman", label: "Spearman" }]} />}
        >
          <Plot
            name={`correlaciones-${method}`}
            ariaLabel="Correlación de cada explicativa con la mortalidad"
            height={Math.max(360, corr.length * 22 + 60)}
            deps={[res.run_id, method]}
            data={(t) => [
              {
                type: "bar",
                orientation: "h",
                x: corr.map((c: any) => (method === "pearson" ? c.pearson : c.spearman)),
                y: corr.map((c: any) => c.etiqueta),
                marker: { color: corr.map((c: any) => ((method === "pearson" ? c.pearson : c.spearman) > 0 ? t.s8 : t.s1)), cornerradius: 4 },
                error_x:
                  method === "pearson"
                    ? { type: "data", symmetric: false, array: corr.map((c: any) => c.pearson_upper - c.pearson), arrayminus: corr.map((c: any) => c.pearson - c.pearson_lower), color: t.ink2, thickness: 1.2, width: 3 }
                    : undefined,
                hovertemplate: "%{y}: r = %{x:.3f}<extra></extra>",
              },
            ]}
            layout={() => ({ margin: { l: 210, r: 16, t: 8, b: 44 }, xaxis: { title: { text: method === "pearson" ? "r de Pearson" : "ρ de Spearman" }, range: [-0.8, 0.8] }, showlegend: false })}
          />
        </Card>
        <Card title="Matriz de correlaciones" sub="Escala divergente: azul negativa, gris nula, rojo positiva. Pase el cursor para ver el valor.">
          <Plot
            name={`matriz-correlaciones-${method}`}
            ariaLabel="Matriz de correlaciones entre variables"
            height={720}
            deps={[res.run_id, method]}
            data={(t) => [
              {
                type: "heatmap",
                z: cm.values,
                x: cm.labels,
                y: cm.labels,
                zmin: -1,
                zmax: 1,
                colorscale: [
                  [0, "#104281"],
                  [0.25, "#5598e7"],
                  [0.5, t.surface === "#1a1a19" ? "#383835" : "#f0efec"],
                  [0.75, "#e34948"],
                  [1, "#a61f1f"],
                ],
                xgap: 1,
                ygap: 1,
                hovertemplate: "%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
                colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: t.ink2 } },
              },
            ]}
            layout={() => ({ margin: { l: 210, r: 10, t: 8, b: 190 }, xaxis: { tickangle: -50, showgrid: false, dtick: 1 }, yaxis: { autorange: "reversed", showgrid: false, dtick: 1 } })}
          />
        </Card>
      </div>

      <div className="grid grid-2">
        <Card title="Mortalidad por región censal" sub={`ANOVA de Welch: F = ${num(x.region.welch.statistic, 1)}, p ${pval(x.region.welch.p)}; η² = ${num(x.region.anova.eta2, 3)}`}>
          <Plot
            name="regiones"
            ariaLabel="Media de mortalidad por región con su intervalo de confianza"
            height={300}
            deps={[res.run_id]}
            data={(t) => [
              {
                type: "scatter",
                mode: "markers",
                x: regions.map((g: any) => g.mean),
                y: regions.map((g: any) => g.group),
                error_x: { type: "data", symmetric: false, array: regions.map((g: any) => g.upper - g.mean), arrayminus: regions.map((g: any) => g.mean - g.lower), color: t.ink2, thickness: 2, width: 6 },
                marker: { size: 13, color: regions.map((g: any) => (t as any)[REGION_STYLE[g.group].color]), symbol: regions.map((g: any) => REGION_STYLE[g.group].symbol) },
                text: regions.map((g: any) => `n = ${g.n}`),
                hovertemplate: "%{y}: %{x:.1f} (%{text})<extra></extra>",
              },
            ]}
            layout={() => ({ margin: { l: 110, r: 16, t: 8, b: 44 }, xaxis: { title: { text: "Media (IC 95 %)" } }, showlegend: false })}
          />
          <Explain>
            <p>
              Las varianzas difieren entre regiones (Levene p {pval(x.region.levene.p)}), por eso se usa el ANOVA de Welch; las comparaciones por pares (Tukey) se confirman con
              Mann-Whitney y corrección de Holm. El Sur tiene la mortalidad más alta; buena parte de esa brecha se explica luego por la composición socioeconómica.
            </p>
          </Explain>
        </Card>
        <Card title="Gradiente por renta" sub="Mortalidad media por decil de renta mediana del condado.">
          <Plot
            name="gradiente-renta"
            ariaLabel="Mortalidad media por decil de renta"
            height={300}
            deps={[res.run_id]}
            data={(t) => [
              {
                type: "scatter",
                mode: "lines+markers",
                x: x.income_gradient.table.map((d: any) => d.decil),
                y: x.income_gradient.table.map((d: any) => d.mean),
                line: { color: t.s1, width: 2 },
                marker: { size: 9, color: t.s1, line: { color: t.surface, width: 2 } },
                hovertemplate: "Decil %{x}: %{y:.1f}<extra></extra>",
              },
            ]}
            layout={() => ({ xaxis: { title: { text: "Decil de renta (1 = más pobre)" }, dtick: 1 }, yaxis: { title: { text: "Mortalidad media" } }, showlegend: false })}
          />
        </Card>
      </div>

      <Card title="Estadística descriptiva" sub="Variables candidatas tras la depuración.">
        <DataTable
          name="descriptiva"
          rows={x.describe}
          columns={[
            { key: "etiqueta", header: "Variable" },
            { key: "n", header: "n", num: true, render: (d: any) => int(d.n) },
            { key: "media", header: "Media", num: true, render: (d: any) => num(d.media, 2) },
            { key: "dt", header: "D. t.", num: true, render: (d: any) => num(d.dt, 2) },
            { key: "min", header: "Mín.", num: true, render: (d: any) => num(d.min, 2) },
            { key: "q1", header: "Q1", num: true, render: (d: any) => num(d.q1, 2) },
            { key: "mediana", header: "Mediana", num: true, render: (d: any) => num(d.mediana, 2) },
            { key: "q3", header: "Q3", num: true, render: (d: any) => num(d.q3, 2) },
            { key: "max", header: "Máx.", num: true, render: (d: any) => num(d.max, 2) },
          ]}
        />
        <p className="small muted">Variables: {labels.TARGET_deathRate ?? "mortalidad"} y explicativas.</p>
      </Card>
    </Section>
  );
}
