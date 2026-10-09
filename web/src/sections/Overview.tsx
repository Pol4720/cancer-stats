import { Plot } from "../charts/Plot";
import { Icon } from "../components/icons";
import { Card, Explain, Section, Stat } from "../components/ui";
import { ci, int, num, signed } from "../lib/format";
import { useApp } from "../lib/store";
import { go } from "../lib/router";

const DELIVERABLES = [
  { id: "a", text: "Modelo final y pasos seguidos para construirlo", to: "modelo" },
  { id: "b", text: "Salida del software con R², R² ajustado y RMSE", to: "final" },
  { id: "c", text: "Código (Python reproducible y sintaxis SPSS)", to: "corridas" },
  { id: "d", text: "Diagnóstico: linealidad, independencia, homocedasticidad, normalidad y multicolinealidad", to: "diagnostico" },
  { id: "e", text: "Interpretación del modelo", to: "final" },
  { id: "f", text: "Atípicos, valores perdidos y variables categóricas", to: "depuracion" },
];

export default function Overview() {
  const { res, source } = useApp();
  const eff = res.effects;
  const fin = eff.final;
  const ms = fin.spss.model_summary;
  const pred = res.predictive;
  const cvEff = pred?.cv.find((r: any) => r.model === "ols_effects");
  const testChosen = pred?.test.find((r: any) => r.model === pred.chosen);
  // Una fila por efecto; las variables con interacción aportan una fila por región.
  const effects = (eff.effects as any[])
    .flatMap((e) =>
      e.interaccion_region
        ? e.by_region.map((r: any) => ({ ...r, etiqueta: `${e.etiqueta} · ${r.region}`, variable: `${e.variable}-${r.region}` }))
        : [e],
    )
    .sort((a: any, b: any) => b.iqr_effect - a.iqr_effect);
  const mainEffects = (eff.effects as any[]).filter((e) => !e.interaccion_region);

  return (
    <Section
      eyebrow="Proyecto final · Regresión lineal múltiple"
      title="Mortalidad por cáncer en los condados de EE. UU."
      lead={
        <>
          ¿Qué características socioeconómicas, demográficas y sanitarias de un condado se asocian con su tasa de mortalidad por cáncer
          (muertes por 100 000 habitantes)? Modelo MCO con inferencia robusta sobre {int(res.ingest.n_rows)} condados del fichero
          oficial <code>practica.sav</code>.
        </>
      }
    >
      <div className="hero">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <div className="muted small">Corrida</div>
            <div className="mono">{res.run_id}</div>
          </div>
          <div className="row">
            <button className="btn primary" onClick={() => go("datos")}>
              <Icon name="compass" /> Recorrer el análisis
            </button>
            <a className="btn" href={source.fileUrl(res.run_id, "modelo_final.sps")} download>
              <Icon name="download" /> Sintaxis SPSS
            </a>
          </div>
        </div>
      </div>

      <div className="grid grid-4">
        <Stat label="R² (dentro de la muestra)" value={num(ms.R2, 3)} note={`${num(100 * ms.R2, 1)} % de la variabilidad explicada`} />
        <Stat label="R² ajustado" value={num(ms.R2_adj, 3)} note={`k = ${fin.summary.k} regresores, n = ${int(ms.n)}`} />
        <Stat label="RMSE (dentro de la muestra)" value={num(ms.rmse, 2)} note="muertes por 100 000 hab." />
        <Stat
          label="RMSE en prueba (modelo predictivo)"
          value={testChosen ? num(testChosen.rmse, 2) : "—"}
          note={pred ? `${pred.chosen_label}; VC del modelo de efectos: ${num(cvEff?.rmse_mean, 2)}` : "predicción desactivada"}
        />
      </div>

      <div className="grid grid-2">
        <Card title="Entregables de la orientación" sub="Cada punto lleva a la sección donde se responde.">
          <ul className="timeline">
            {DELIVERABLES.map((d) => (
              <li key={d.id}>
                <Icon name="ok" style={{ color: "var(--good)" }} />
                <span>
                  <strong>({d.id})</strong> {d.text} —{" "}
                  <a href={`#/${d.to}`} onClick={(e) => (e.preventDefault(), go(d.to))}>
                    ver
                  </a>
                </span>
              </li>
            ))}
          </ul>
        </Card>
        <Card title="Efectos al pasar del primer al tercer cuartil" sub="Cambio esperado en la mortalidad, manteniendo constante el resto (IC 95 %, robusto por estado).">
          <Plot
            name="efectos-iqr"
            ariaLabel="Efecto de cada explicativa al pasar de Q1 a Q3"
            height={Math.max(260, effects.length * 34 + 60)}
            deps={[res.run_id]}
            data={(t) => [
              {
                type: "scatter",
                mode: "markers",
                orientation: "h",
                x: effects.map((e: any) => e.iqr_effect),
                y: effects.map((e: any) => e.etiqueta),
                error_x: {
                  type: "data",
                  symmetric: false,
                  array: effects.map((e: any) => e.iqr_upper - e.iqr_effect),
                  arrayminus: effects.map((e: any) => e.iqr_effect - e.iqr_lower),
                  color: t.s1,
                  thickness: 2,
                  width: 0,
                },
                marker: { size: 10, color: effects.map((e: any) => (e.p < 0.05 ? t.s1 : t.surface)), line: { color: t.s1, width: 2 } },
                hovertemplate: "%{y}<br>Efecto Q1→Q3: %{x:.1f}<extra></extra>",
              },
            ]}
            layout={() => ({
              margin: { l: 250, r: 16, t: 8, b: 44 },
              xaxis: { title: { text: "Muertes por 100 000 (Q1 → Q3)" }, zeroline: true },
              yaxis: { autorange: "reversed" },
              showlegend: false,
            })}
          />
          <p className="small muted">Marcador relleno: significativo al 5 %; hueco: no significativo.</p>
        </Card>
      </div>

      <Card title="Ecuación del modelo final" sub="Explicativas continuas centradas en su media: la constante es la mortalidad esperada de un condado medio del Sur.">
        <div className="equation">{fin.equation}</div>
        <Explain title="Cómo leer los coeficientes">
          <p>
            Cada coeficiente es el cambio esperado en la mortalidad por una unidad más de su explicativa, manteniendo constantes las demás. No son
            efectos causales: el diseño es observacional y ecológico (condados, no personas).
          </p>
          <ul>
            {mainEffects.slice(0, 5).map((e: any) => (
              <li key={e.variable}>
                <strong>{e.etiqueta}</strong>: {e.per_unit_text} → {signed(e.per_unit, 3)} muertes/100 000 {ci(e.per_unit_lower, e.per_unit_upper, 3)}.
              </li>
            ))}
          </ul>
        </Explain>
      </Card>
    </Section>
  );
}
