import { useMemo, useState } from "react";
import { Plot } from "../charts/Plot";
import { Card, DataTable, Explain, Section, Segmented, StatusTag } from "../components/ui";
import { int, num, pval, termLabel } from "../lib/format";
import { lowess } from "../lib/stats";
import { REGION_ORDER, REGION_STYLE, useApp } from "../lib/store";

type Tab = "linealidad" | "independencia" | "homocedasticidad" | "normalidad" | "colinealidad" | "influencia";

function verdict(p: number | null | undefined) {
  const reject = p !== null && p !== undefined && p < 0.05;
  return <StatusTag status={reject ? "advertencia" : "ok"} label={reject ? "Se rechaza H₀ (5 %)" : "No se rechaza H₀"} />;
}

export default function Diagnostics() {
  const { res, labels } = useApp();
  const e = res.effects;
  const d = e.diagnostics;
  const resid = e.residuals as any[];
  const [tab, setTab] = useState<Tab>("linealidad");
  const [prVar, setPrVar] = useState<string>(d.linearity.partial_residuals[0]?.variable ?? "");
  const pr = d.linearity.partial_residuals.find((p: any) => p.variable === prVar);
  const L = (t: string) => termLabel(t, labels);
  const fittedSmooth = useMemo(() => lowess(resid.map((r) => r.fitted), resid.map((r) => r.std_resid), 0.3), [resid]);
  const norm = Object.fromEntries((d.normality.tests as any[]).map((t) => [t.test, t]));
  const h = d.homoscedasticity;
  const ind = d.independence;
  const inf = d.influence;

  const summary = [
    { s: "Linealidad", c: "RESET de Ramsey", v: num(d.linearity.reset.F, 3), p: d.linearity.reset.p },
    { s: "Independencia", c: "Durbin-Watson (orden del fichero)", v: num(ind.durbin_watson, 3), p: null },
    { s: "Independencia", c: "Correlación intraclase por estado", v: num(ind.icc_state.icc, 3), p: ind.icc_state.p },
    { s: "Homocedasticidad", c: "Breusch-Pagan (Koenker)", v: num(h.breusch_pagan.LM, 1), p: h.breusch_pagan.p },
    { s: "Homocedasticidad", c: "White (reducido)", v: num(h.white_reduced.LM, 1), p: h.white_reduced.p },
    { s: "Normalidad", c: "Jarque-Bera", v: num(norm["Jarque-Bera"]?.statistic, 1), p: norm["Jarque-Bera"]?.p },
    { s: "Colinealidad", c: "FIV máximo", v: num(Math.max(...d.collinearity.vif.map((x: any) => x.fiv)), 2), p: null },
  ];

  return (
    <Section
      eyebrow="8 · Diagnóstico"
      title="¿Se cumplen las hipótesis del modelo?"
      lead="Linealidad, independencia de los errores, homocedasticidad, normalidad de los residuos y ausencia de multicolinealidad, más observaciones influyentes. Cada violación detectada tiene una respuesta en la estrategia de inferencia."
    >
      <Card title="Resumen de contrastes">
        <DataTable
          name="diagnostico-resumen"
          csv={false}
          rows={summary}
          columns={[
            { key: "s", header: "Hipótesis" },
            { key: "c", header: "Contraste" },
            { key: "v", header: "Estadístico", num: true },
            { key: "p", header: "p", num: true, render: (r: any) => pval(r.p) },
            { key: "d", header: "Decisión", render: (r: any) => (r.p === null ? <span className="muted">descriptivo</span> : verdict(r.p)) },
          ]}
        />
        <Explain title="¿Qué se hace con cada violación?">
          <ul>
            <li><strong>Heterocedasticidad y dependencia por estado</strong>: errores típicos robustos por conglomerados (inferencia válida sin suponer varianza constante ni independencia dentro del estado).</li>
            <li><strong>Colas algo pesadas</strong>: con n ≈ 2 800 los estimadores MCO son aproximadamente normales (TCL); además, el bootstrap por conglomerados no supone normalidad y da intervalos muy parecidos.</li>
            <li><strong>Curvatura leve</strong>: transformaciones logarítmicas previas; el RESET global no detecta mala especificación grave.</li>
            <li><strong>Influencia</strong>: se reestima sin los condados influyentes (sensibilidad) y con regresión robusta de Huber.</li>
          </ul>
        </Explain>
      </Card>

      <div className="row" style={{ overflowX: "auto" }}>
        <Segmented
          label="Hipótesis"
          value={tab}
          onChange={setTab}
          options={[
            { value: "linealidad", label: "Linealidad" },
            { value: "independencia", label: "Independencia" },
            { value: "homocedasticidad", label: "Homocedasticidad" },
            { value: "normalidad", label: "Normalidad" },
            { value: "colinealidad", label: "Multicolinealidad" },
            { value: "influencia", label: "Influencia" },
          ]}
        />
      </div>

      {tab === "linealidad" && (
        <div className="grid grid-2">
          <Card
            title="Residuos parciales (componente + residuo)"
            sub="Si la relación es lineal, los puntos siguen la recta; la curva LOWESS delata curvatura."
            actions={
              <select className="select" value={prVar} onChange={(ev) => setPrVar(ev.target.value)} aria-label="Variable">
                {d.linearity.partial_residuals.map((p: any) => (
                  <option key={p.variable} value={p.variable}>{p.etiqueta}</option>
                ))}
              </select>
            }
          >
            {pr && (
              <Plot
                name={`residuos-parciales-${prVar}`}
                ariaLabel={`Residuos parciales de ${pr.etiqueta}`}
                height={360}
                deps={[res.run_id, prVar]}
                data={(t) => {
                  const xs = pr.x as number[];
                  const ys = pr.y as number[];
                  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
                  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
                  const lo = Math.min(...xs);
                  const hi = Math.max(...xs);
                  return [
                    { type: "scattergl", mode: "markers", x: xs, y: ys, name: "Residuos parciales", marker: { size: 5, color: t.s1, opacity: 0.35 } },
                    { type: "scatter", mode: "lines", x: [lo, hi], y: [my + pr.slope * (lo - mx), my + pr.slope * (hi - mx)], name: "Recta del modelo (β̂)", line: { color: t.ink2, width: 1.5, dash: "dash" } },
                    { type: "scatter", mode: "lines", x: pr.smooth_x, y: pr.smooth_y, name: "LOWESS", line: { color: t.s2, width: 2.5 } },
                  ];
                }}
                layout={() => ({ xaxis: { title: { text: pr.etiqueta } }, yaxis: { title: { text: "Componente + residuo" } } })}
              />
            )}
          </Card>
          <Card title="Contrastes de curvatura" sub={`RESET global: F = ${num(d.linearity.reset.F, 3)}, p ${pval(d.linearity.reset.p)}.`}>
            <DataTable
              name="curvatura"
              csv={false}
              rows={d.linearity.curvature}
              columns={[
                { key: "etiqueta", header: "Variable" },
                { key: "F", header: "F (término cuadrático)", num: true, render: (r: any) => num(r.F, 2) },
                { key: "p", header: "p", num: true, render: (r: any) => pval(r.p) },
                { key: "p_ajustado", header: "p (Holm)", num: true, render: (r: any) => pval(r.p_ajustado) },
              ]}
            />
            <Explain>
              <p>
                Para cada explicativa se añade su cuadrado y se contrasta. Con casi 3 000 condados se detectan curvaturas pequeñas; lo que importa es
                si cambian las conclusiones: la forma de los residuos parciales es casi lineal en el rango donde están la mayoría de los datos.
              </p>
            </Explain>
          </Card>
        </div>
      )}

      {tab === "independencia" && (
        <div className="grid grid-2">
          <Card title="Durbin-Watson y rachas" sub="Autocorrelación en el orden del fichero (los condados están ordenados por estado).">
            <div className="table-wrap">
              <table className="data">
                <tbody>
                  <tr><th scope="row">Durbin-Watson</th><td className="num">{num(ind.durbin_watson, 3)}</td><td>≈ 2 indica ausencia de autocorrelación de primer orden</td></tr>
                  <tr><th scope="row">Rachas de signos</th><td className="num">z = {num(ind.runs.z, 2)}</td><td>p {pval(ind.runs.p)} ({int(ind.runs.runs)} rachas; esperadas {num(ind.runs.expected, 0)})</td></tr>
                  <tr><th scope="row">Correlación intraclase (estado)</th><td className="num">{num(ind.icc_state.icc, 3)}</td><td>F = {num(ind.icc_state.F, 2)}, p {pval(ind.icc_state.p)}</td></tr>
                </tbody>
              </table>
            </div>
            <Explain>
              <p>
                En datos transversales el orden de las filas es arbitrario, así que Durbin-Watson sólo detecta dependencia si el orden la refleja.
                Aquí el fichero está ordenado por estado: las rachas y la correlación intraclase revelan que los residuos de un mismo estado se
                parecen. La respuesta es la inferencia por conglomerados de estado.
              </p>
            </Explain>
          </Card>
          <Card title="Residuo medio por estado" sub="Si los errores fueran independientes, oscilarían alrededor de cero sin patrón.">
            <Plot
              name="residuo-estado"
              ariaLabel="Residuo medio por estado"
              height={360}
              deps={[res.run_id]}
              data={(t) => {
                const byState = new Map<string, number[]>();
                resid.forEach((r) => {
                  const st = String(r.county_id).split(", ").pop() ?? "";
                  if (!byState.has(st)) byState.set(st, []);
                  byState.get(st)!.push(r.resid);
                });
                const rows = [...byState.entries()].map(([s, v]) => ({ s, m: v.reduce((a, b) => a + b, 0) / v.length, n: v.length })).sort((a, b) => a.m - b.m);
                return [
                  {
                    type: "bar",
                    x: rows.map((r) => r.s),
                    y: rows.map((r) => r.m),
                    marker: { color: rows.map((r) => (r.m > 0 ? t.s8 : t.s1)), cornerradius: 3 },
                    text: rows.map((r) => `n = ${r.n}`),
                    hovertemplate: "%{x}: %{y:.1f} (%{text})<extra></extra>",
                  },
                ];
              }}
              layout={() => ({ xaxis: { tickangle: -60, tickfont: { size: 10 } }, yaxis: { title: { text: "Residuo medio" } }, showlegend: false, margin: { b: 110 } })}
            />
          </Card>
        </div>
      )}

      {tab === "homocedasticidad" && (
        <div className="grid grid-2">
          <Card title="Residuos tipificados frente a valores ajustados" sub="Sin patrón ni embudo si la varianza es constante.">
            <Plot
              name="residuos-ajustados"
              ariaLabel="Residuos tipificados frente a valores ajustados"
              height={380}
              deps={[res.run_id]}
              data={(t) => [
                ...REGION_ORDER.map((reg) => {
                  const rows = resid.filter((r) => r.region === reg);
                  return {
                    type: "scattergl",
                    mode: "markers",
                    name: reg,
                    x: rows.map((r) => r.fitted),
                    y: rows.map((r) => r.std_resid),
                    text: rows.map((r) => r.county_id),
                    marker: { size: 6, opacity: 0.5, color: (t as any)[REGION_STYLE[reg].color], symbol: REGION_STYLE[reg].symbol },
                    hovertemplate: "%{text}<br>ŷ = %{x:.1f}, residuo tip. = %{y:.2f}<extra></extra>",
                  };
                }),
                { type: "scatter", mode: "lines", name: "LOWESS", x: fittedSmooth.x, y: fittedSmooth.y, line: { color: t.ink, width: 2.5 } },
              ]}
              layout={(t) => ({
                xaxis: { title: { text: "Valor ajustado" } },
                yaxis: { title: { text: "Residuo tipificado" } },
                shapes: [-3, 3].map((v) => ({ type: "line", x0: 0, x1: 1, xref: "paper", y0: v, y1: v, line: { color: t.critical, dash: "dot", width: 1 } })),
              })}
            />
          </Card>
          <Card title="Contrastes" sub="Sobre los residuos del modelo final.">
            <div className="table-wrap">
              <table className="data">
                <tbody>
                  <tr><th scope="row">Breusch-Pagan (Koenker)</th><td className="num">{num(h.breusch_pagan.LM, 1)}</td><td>p {pval(h.breusch_pagan.p)}</td></tr>
                  <tr><th scope="row">White (ŷ, ŷ²)</th><td className="num">{num(h.white_reduced.LM, 1)}</td><td>p {pval(h.white_reduced.p)}</td></tr>
                  <tr><th scope="row">Frente a 1/población</th><td className="num">{num(h.bp_population.LM, 1)}</td><td>p {pval(h.bp_population.p)}</td></tr>
                  <tr><th scope="row">Goldfeld-Quandt (pequeños/grandes)</th><td className="num">{num(h.goldfeld_quandt.F, 2)}</td><td>p {pval(h.goldfeld_quandt.p)}</td></tr>
                </tbody>
              </table>
            </div>
            <Explain>
              <p>
                La heterocedasticidad no sesga los coeficientes MCO, pero invalida sus errores típicos clásicos. Su fuente principal es el tamaño del
                condado (ver Construcción del modelo, paso 2). Los errores típicos robustos la corrigen; la estimación ponderada (MCPF) es la
                alternativa eficiente y se presenta como sensibilidad.
              </p>
            </Explain>
          </Card>
        </div>
      )}

      {tab === "normalidad" && (
        <div className="grid grid-2">
          <Card title="Gráfico cuantil-cuantil normal" sub="Si los residuos fueran normales, los puntos seguirían la diagonal.">
            <Plot
              name="qq-residuos"
              ariaLabel="Gráfico cuantil-cuantil de los residuos"
              height={360}
              deps={[res.run_id]}
              data={(t) => {
                const q = d.normality.qq;
                const lo = Math.min(...q.theoretical);
                const hi = Math.max(...q.theoretical);
                return [
                  { type: "scattergl", mode: "markers", x: q.theoretical, y: q.sample, name: "Residuos", marker: { size: 5, color: t.s1, opacity: 0.6 } },
                  { type: "scatter", mode: "lines", x: [lo, hi], y: [lo * (q.sd ?? 1) + (q.mean ?? 0), hi * (q.sd ?? 1) + (q.mean ?? 0)], name: "Normal", line: { color: t.ink, width: 1.5 } },
                ];
              }}
              layout={() => ({ xaxis: { title: { text: "Cuantil teórico N(0, 1)" } }, yaxis: { title: { text: "Cuantil muestral" } }, showlegend: false })}
            />
          </Card>
          <Card title="Contrastes de normalidad de los residuos" sub={`Asimetría ${num(d.normality.shape.skewness, 3)}; curtosis en exceso ${num(d.normality.shape.kurtosis_excess, 2)}.`}>
            <DataTable
              name="normalidad-residuos"
              csv={false}
              rows={d.normality.tests}
              columns={[
                { key: "test", header: "Contraste" },
                { key: "statistic", header: "Estadístico", num: true, render: (r: any) => num(r.statistic, 4) },
                { key: "p", header: "p", num: true, render: (r: any) => pval(r.p) },
              ]}
            />
            <Explain>
              <p>
                Se rechaza la normalidad exacta, como es esperable con n grande: las colas son algo más pesadas que las de la normal, pero la
                asimetría es casi nula. La normalidad sólo se necesita para la inferencia exacta en muestras pequeñas; aquí los contrastes t y F son
                válidos asintóticamente, y el bootstrap por conglomerados lo confirma (ver Sensibilidad).
              </p>
            </Explain>
          </Card>
        </div>
      )}

      {tab === "colinealidad" && (
        <Card title="Multicolinealidad en el modelo final" sub={`Número de condición: ${num(d.collinearity.condition_number, 1)} (problemático por encima de 30).`}>
          <DataTable
            name="fiv-final"
            rows={d.collinearity.vif}
            columns={[
              { key: "variable", header: "Término", value: (r: any) => L(r.variable) },
              { key: "fiv", header: "FIV", num: true, render: (r: any) => num(r.fiv, 2) },
              { key: "tolerancia", header: "Tolerancia", num: true, render: (r: any) => num(r.tolerancia, 3) },
              { key: "r2_aux", header: "R² auxiliar", num: true, render: (r: any) => num(r.r2_aux, 3) },
            ]}
          />
          <Explain>
            <p>
              Todos los FIV de los efectos principales quedan muy por debajo de 10. Los términos de interacción tienen FIV mayores por construcción
              (comparten información con sus efectos principales), lo que no afecta a la interpretación al estar las variables centradas.
            </p>
          </Explain>
        </Card>
      )}

      {tab === "influencia" && (
        <div className="grid grid-2">
          <Card title="Apalancamiento frente a residuo estudentizado" sub={`El tamaño del punto es la distancia de Cook. Umbrales: h > ${num(inf.thresholds.leverage, 4)}, |r*| > ${num(inf.thresholds.studentized, 1)}.`}>
            <Plot
              name="influencia"
              ariaLabel="Apalancamiento frente a residuo estudentizado"
              height={380}
              deps={[res.run_id]}
              data={(t) => {
                const maxCook = Math.max(...resid.map((r) => r.cook));
                return [
                  {
                    type: "scattergl",
                    mode: "markers",
                    x: resid.map((r) => r.leverage),
                    y: resid.map((r) => r.stud_resid),
                    text: resid.map((r) => `${r.county_id}<br>Cook = ${num(r.cook, 4)}`),
                    marker: {
                      size: resid.map((r) => 4 + 26 * Math.sqrt(r.cook / maxCook)),
                      color: resid.map((r) => (r.cook > inf.thresholds.cook ? t.s8 : t.s1)),
                      opacity: 0.55,
                      line: { width: 1, color: t.surface },
                    },
                    hovertemplate: "%{text}<br>h = %{x:.4f}, r* = %{y:.2f}<extra></extra>",
                  },
                ];
              }}
              layout={(t) => ({
                xaxis: { title: { text: "Apalancamiento h" } },
                yaxis: { title: { text: "Residuo estudentizado r*" } },
                showlegend: false,
                shapes: [
                  { type: "line", x0: inf.thresholds.leverage, x1: inf.thresholds.leverage, y0: 0, y1: 1, yref: "paper", line: { color: t.ink2, dash: "dot", width: 1 } },
                  ...[-inf.thresholds.studentized, inf.thresholds.studentized].map((v: number) => ({ type: "line", x0: 0, x1: 1, xref: "paper", y0: v, y1: v, line: { color: t.ink2, dash: "dot", width: 1 } })),
                ],
              })}
            />
            <p className="small muted">Rojo: distancia de Cook por encima de 4/n ({num(inf.thresholds.cook, 5)}).</p>
          </Card>
          <Card title="Condados más influyentes" sub={`${int(inf.n_cook)} superan 4/n; ${int(inf.n_cook_f50)} superan la mediana de F (influencia grave).`}>
            <DataTable
              name="influyentes"
              rows={inf.top}
              columns={[
                { key: "county_id", header: "Condado" },
                { key: "y", header: "y", num: true, render: (r: any) => num(r.y, 1) },
                { key: "fitted", header: "ŷ", num: true, render: (r: any) => num(r.fitted, 1) },
                { key: "stud_resid", header: "r*", num: true, render: (r: any) => num(r.stud_resid, 2) },
                { key: "leverage", header: "h", num: true, render: (r: any) => num(r.leverage, 4) },
                { key: "cook", header: "Cook", num: true, render: (r: any) => num(r.cook, 4) },
              ]}
            />
          </Card>
        </div>
      )}
    </Section>
  );
}
