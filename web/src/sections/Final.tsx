import { useState } from "react";
import { Plot } from "../charts/Plot";
import { Card, DataTable, Explain, Section, Segmented, Stat } from "../components/ui";
import { ci, int, num, pval, signed, termLabel } from "../lib/format";
import { REGION_ORDER, REGION_STYLE, useApp } from "../lib/store";

type SE = "cluster" | "hc3" | "classic";

export default function Final() {
  const { res, labels } = useApp();
  const e = res.effects;
  const fin = e.final;
  const sp = fin.spss;
  const ms = sp.model_summary;
  const [se, setSe] = useState<SE>("cluster");
  const coef = se === "cluster" ? fin.coef : se === "hc3" ? fin.coef_hc3 : fin.coef_classic;
  const L = (t: string) => termLabel(t, labels);
  const interacted = e.effects.filter((x: any) => x.interaccion_region);
  const adj = [...e.adjusted_region_means].sort((a: any, b: any) => REGION_ORDER.indexOf(a.region) - REGION_ORDER.indexOf(b.region));
  const dfe = sp.anova.n - sp.anova.k - 1;

  return (
    <Section
      eyebrow="7 · Modelo final"
      title="Modelo de regresión lineal múltiple por MCO"
      lead="Coeficientes, bondad de ajuste y salida tipo SPSS. Los coeficientes son los mismos con cualquier estimación de la covarianza; cambian sus errores típicos y, con ellos, los contrastes."
    >
      <div className="grid grid-4">
        <Stat label="R²" value={num(ms.R2, 3)} note={`R = ${num(ms.R, 3)}`} />
        <Stat label="R² ajustado" value={num(ms.R2_adj, 3)} note={`k = ${sp.anova.k}, n = ${int(ms.n)}`} />
        <Stat label="Error típico de la estimación" value={num(ms.se_estimate, 2)} note="√(SCE / (n − k − 1))" />
        <Stat label="RMSE" value={num(ms.rmse, 2)} note="√(SCE / n), dentro de la muestra" />
      </div>

      <Card
        title="Coeficientes"
        sub={`Inferencia con t de ${se === "cluster" ? `${int(fin.summary.df_inference)} g.l. (${int(fin.n_clusters)} estados − 1)` : `${int(dfe)} g.l.`}. Explicativas centradas en su media.`}
        actions={
          <Segmented
            label="Errores típicos"
            value={se}
            onChange={setSe}
            options={[
              { value: "cluster", label: "Cluster (estado)" },
              { value: "hc3", label: "HC3" },
              { value: "classic", label: "Clásicos (SPSS)" },
            ]}
          />
        }
      >
        <DataTable
          name={`coeficientes-${se}`}
          rows={coef}
          columns={[
            { key: "term", header: "Término", value: (c: any) => L(c.term) },
            { key: "coef", header: "Coeficiente", num: true, render: (c: any) => num(c.coef, 3) },
            { key: "se", header: "E. T.", num: true, render: (c: any) => num(c.se, 3) },
            { key: "t", header: "t", num: true, render: (c: any) => num(c.t, 2) },
            { key: "p", header: "p", num: true, render: (c: any) => <span className={c.p < 0.05 ? "sig" : "nosig"}>{pval(c.p)}</span> },
            { key: "lower", header: "IC 95 %", render: (c: any) => ci(c.lower, c.upper, 3) },
            { key: "beta", header: "Beta", num: true, render: (c: any) => (c.beta === null || c.beta === undefined ? "" : num(c.beta, 3)) },
          ]}
        />
        <Explain title="¿Por qué errores típicos por conglomerados?">
          <p>
            Los condados de un mismo estado comparten registro de cáncer, sistema sanitario y políticas: sus errores están correlacionados
            (correlación intraclase {num(e.diagnostics.independence.icc_state.icc, 3)}, p {pval(e.diagnostics.independence.icc_state.p)}). Los
            errores típicos clásicos suponen independencia y homocedasticidad y resultan demasiado pequeños; los robustos por conglomerados admiten
            ambas cosas. Compare las tres pestañas: los coeficientes no cambian, los errores típicos sí.
          </p>
        </Explain>
      </Card>

      <div className="stack">
        <Card title="Resumen del modelo (tipo SPSS)">
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr><th>R</th><th>R²</th><th>R² ajustado</th><th>E.T. estimación</th><th>Durbin-Watson</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td className="num">{num(ms.R, 3)}</td><td className="num">{num(ms.R2, 3)}</td><td className="num">{num(ms.R2_adj, 3)}</td>
                  <td className="num">{num(ms.se_estimate, 3)}</td><td className="num">{num(ms.durbin_watson, 3)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <h3 style={{ marginTop: 16 }}>ANOVA</h3>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr><th>Fuente</th><th className="num">Suma de cuadrados</th><th className="num">g.l.</th><th className="num">Media cuadrática</th><th className="num">F</th><th className="num">Sig.</th></tr>
              </thead>
              <tbody>
                <tr><td>Regresión</td><td className="num">{num(sp.anova.scr, 1)}</td><td className="num">{sp.anova.k}</td><td className="num">{num(sp.anova.scr / sp.anova.k, 1)}</td><td className="num">{num(sp.anova.F, 2)}</td><td className="num">{pval(sp.anova.p)}</td></tr>
                <tr><td>Residuo</td><td className="num">{num(sp.anova.sce, 1)}</td><td className="num">{int(dfe)}</td><td className="num">{num(sp.anova.sce / dfe, 2)}</td><td /><td /></tr>
                <tr><td>Total</td><td className="num">{num(sp.anova.sct, 1)}</td><td className="num">{int(sp.anova.n - 1)}</td><td /><td /><td /></tr>
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 8 }}>
            Contraste global robusto (Wald, cluster): F = {num(fin.summary.f_robust, 1)}, p {pval(fin.summary.p_f_robust)}.
          </p>
        </Card>
        <Card title="Coeficientes con tolerancia y FIV (tipo SPSS)">
          <DataTable
            name="spss-coeficientes"
            rows={sp.coefficients}
            columns={[
              { key: "term", header: "Término", value: (c: any) => L(c.term) },
              { key: "coef", header: "B", num: true, render: (c: any) => num(c.coef, 3) },
              { key: "se", header: "E.T.", num: true, render: (c: any) => num(c.se, 3) },
              { key: "beta", header: "Beta", num: true, render: (c: any) => (c.beta === null ? "" : num(c.beta, 3)) },
              { key: "t", header: "t", num: true, render: (c: any) => num(c.t, 2) },
              { key: "p", header: "Sig.", num: true, render: (c: any) => pval(c.p) },
              { key: "tolerancia", header: "Tol.", num: true, render: (c: any) => (c.tolerancia === null ? "" : num(c.tolerancia, 3)) },
              { key: "fiv", header: "FIV", num: true, render: (c: any) => (c.fiv === null ? "" : num(c.fiv, 2)) },
            ]}
          />
        </Card>
      </div>

      <Card title="Interpretación de cada efecto" sub="Asociaciones ajustadas por el resto de variables; no son efectos causales (diseño observacional y ecológico).">
        <ul>
          {e.effects.map((x: any) => (
            <li key={x.variable} style={{ marginBottom: 8 }}>
              {x.frase}
            </li>
          ))}
        </ul>
      </Card>

      {interacted.length > 0 && (
        <Card title="Pendientes por región (interacciones)" sub="Efecto de un punto porcentual más en cada región, con su IC 95 % robusto.">
          <div className={`grid grid-${Math.min(interacted.length, 2)}`}>
            {interacted.map((x: any) => {
              const rows = [...x.by_region].sort((a: any, b: any) => REGION_ORDER.indexOf(a.region) - REGION_ORDER.indexOf(b.region));
              return (
                <div key={x.variable}>
                  <h3>{x.etiqueta}</h3>
                  <Plot
                    name={`pendientes-${x.variable}`}
                    ariaLabel={`Pendiente de ${x.etiqueta} por región`}
                    height={260}
                    deps={[res.run_id]}
                    data={(t) => [
                      {
                        type: "scatter",
                        mode: "markers",
                        x: rows.map((r: any) => r.coef),
                        y: rows.map((r: any) => r.region),
                        error_x: { type: "data", symmetric: false, array: rows.map((r: any) => r.upper - r.coef), arrayminus: rows.map((r: any) => r.coef - r.lower), color: t.ink2, thickness: 2, width: 0 },
                        marker: {
                          size: 12,
                          symbol: rows.map((r: any) => REGION_STYLE[r.region].symbol),
                          color: rows.map((r: any) => (r.p < 0.05 ? (t as any)[REGION_STYLE[r.region].color] : t.surface)),
                          line: { width: 2, color: rows.map((r: any) => (t as any)[REGION_STYLE[r.region].color]) },
                        },
                        hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
                      },
                    ]}
                    layout={(t) => ({
                      margin: { l: 110, r: 16, t: 8, b: 44 },
                      xaxis: { title: { text: "Pendiente (IC 95 %)" }, zeroline: true, zerolinecolor: t.ink2 },
                      showlegend: false,
                    })}
                  />
                </div>
              );
            })}
          </div>
          <p className="small muted">Marcador relleno: pendiente significativa al 5 %; hueco: no significativa.</p>
        </Card>
      )}

      <div className="grid grid-2">
        <Card title="Mortalidad por región: bruta y ajustada" sub="Ajustada: predicción para un condado con las explicativas en su media.">
          <Plot
            name="regiones-ajustadas"
            ariaLabel="Mortalidad bruta y ajustada por región"
            height={300}
            deps={[res.run_id]}
            data={(t) => [
              { type: "scatter", mode: "markers", name: "Bruta", x: adj.map((a: any) => a.bruta), y: adj.map((a: any) => a.region), marker: { size: 11, color: t.surface, line: { color: t.ink2, width: 2 } }, hovertemplate: "%{y} · bruta: %{x:.1f}<extra></extra>" },
              {
                type: "scatter",
                mode: "markers",
                name: "Ajustada (IC 95 %)",
                x: adj.map((a: any) => a.ajustada),
                y: adj.map((a: any) => a.region),
                error_x: { type: "data", symmetric: false, array: adj.map((a: any) => a.upper - a.ajustada), arrayminus: adj.map((a: any) => a.ajustada - a.lower), color: t.s1, thickness: 2, width: 0 },
                marker: { size: 12, color: t.s1 },
                hovertemplate: "%{y} · ajustada: %{x:.1f}<extra></extra>",
              },
            ]}
            layout={() => ({ margin: { l: 110, r: 16, t: 8, b: 60 }, xaxis: { title: { text: "Muertes por 100 000" } } })}
          />
        </Card>
        {e.incidence_adjustment && (
          <Card title="¿Qué cambia al ajustar por la incidencia?" sub={`R² con incidencia ${num(e.incidence_adjustment.r2_with, 3)}; sin ella ${num(e.incidence_adjustment.r2_without, 3)}.`}>
            <DataTable
              name="ajuste-incidencia"
              csv={false}
              rows={e.incidence_adjustment.comparison}
              columns={[
                { key: "term", header: "Término", value: (c: any) => L(c.term) },
                { key: "con", header: "Con incidencia", num: true, render: (c: any) => num(c.con, 3) },
                { key: "sin", header: "Sin incidencia", num: true, render: (c: any) => num(c.sin, 3) },
                { key: "cambio_pct", header: "Cambio (%)", num: true, render: (c: any) => signed(c.cambio_pct, 1) },
              ]}
            />
            <Explain>
              <p>
                La incidencia está en el camino causal entre los factores socioeconómicos y la mortalidad (más casos, más muertes). Ajustar por ella
                responde a otra pregunta: la mortalidad a igual número de diagnósticos, que refleja sobre todo supervivencia y acceso al tratamiento.
              </p>
            </Explain>
          </Card>
        )}
      </div>
    </Section>
  );
}
