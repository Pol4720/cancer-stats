import { useState } from "react";
import { Plot } from "../charts/Plot";
import { Card, DataTable, Explain, Section, Stat } from "../components/ui";
import { ci, int, num, pct } from "../lib/format";
import { REGION_ORDER, REGION_STYLE, useApp } from "../lib/store";

export default function Prediction() {
  const { res } = useApp();
  const p = res.predictive;
  const [pdVar, setPdVar] = useState<string>(p?.partial_dependence?.[0]?.variable ?? "");
  if (!p) {
    return (
      <Section eyebrow="10 · Predicción" title="Modelo predictivo desactivado" lead="Esta corrida no incluye la etapa predictiva.">
        <span />
      </Section>
    );
  }
  const group = Object.fromEntries(p.group_cv.map((r: any) => [r.model, r]));
  const test = Object.fromEntries(p.test.map((r: any) => [r.model, r]));
  const cv = [...p.cv].sort((a: any, b: any) => b.rmse_mean - a.rmse_mean);
  const chosen = test[p.chosen];
  const pd = p.partial_dependence.find((r: any) => r.variable === pdVar);
  const tp = p.test_predictions;
  const imp = [...p.importance].reverse();
  const rule = p.selection_rule === "min" ? "mínimo RMSE en validación cruzada" : "regla de un error típico";

  return (
    <Section
      eyebrow="10 · Predicción"
      title="Un modelo para predecir, evaluado con honestidad"
      lead="El modelo de efectos busca interpretar; el predictivo, acertar en condados nuevos. Se comparan ocho modelos con validación cruzada anidada en entrenamiento y el elegido se evalúa una sola vez en la partición de prueba."
    >
      <div className="grid grid-4">
        <Stat label="Modelo elegido" value={<span style={{ fontSize: "1.3rem" }}>{p.chosen_label}</span>} note={`regla: ${rule}`} />
        <Stat label="RMSE en prueba" value={num(chosen.rmse, 2)} note={p.test_ci ? `IC bootstrap ${ci(p.test_ci.rmse_lower, p.test_ci.rmse_upper, 2)}` : ""} />
        <Stat label="R² en prueba" value={num(chosen.r2, 3)} note={`MAE ${num(chosen.mae, 2)}`} />
        <Stat label="Entrenamiento / prueba" value={`${int(p.n_train)} / ${int(p.n_test)}`} note="partición estratificada fija" />
      </div>

      <Card title="Validación cruzada: error de predicción de cada modelo" sub="Barras: RMSE medio en 10 pliegues aleatorios (± d.t.). Rombos: validación agrupada por estado (predecir estados no vistos).">
        <Plot
          name="validacion-cruzada"
          ariaLabel="RMSE de validación cruzada por modelo"
          height={380}
          deps={[res.run_id]}
          data={(t) => [
            {
              type: "bar",
              orientation: "h",
              name: "VC aleatoria",
              x: cv.map((r: any) => r.rmse_mean),
              y: cv.map((r: any) => r.label),
              error_x: { type: "data", array: cv.map((r: any) => r.rmse_sd), color: t.ink2, thickness: 1.5, width: 4 },
              marker: { color: cv.map((r: any) => (r.model === p.chosen ? t.s2 : t.s1)), cornerradius: 4 },
              hovertemplate: "%{y}: RMSE %{x:.2f}<extra>VC aleatoria</extra>",
            },
            {
              type: "scatter",
              mode: "markers",
              name: "VC por estado",
              x: cv.map((r: any) => group[r.model]?.rmse_mean ?? null),
              y: cv.map((r: any) => r.label),
              marker: { symbol: "diamond", size: 11, color: t.ink, line: { color: t.surface, width: 2 } },
              hovertemplate: "%{y}: RMSE %{x:.2f}<extra>VC por estado</extra>",
            },
          ]}
          layout={() => ({ margin: { l: 280, r: 16, t: 8, b: 70 }, xaxis: { title: { text: "RMSE (muertes por 100 000)" } } })}
        />
        <DataTable
          name="modelos-predictivos"
          rows={p.cv}
          columns={[
            { key: "label", header: "Modelo" },
            { key: "rmse_mean", header: "RMSE VC", num: true, render: (r: any) => num(r.rmse_mean, 2) },
            { key: "r2_mean", header: "R² VC", num: true, render: (r: any) => num(r.r2_mean, 3) },
            { key: "g", header: "RMSE VC estado", num: true, value: (r: any) => group[r.model]?.rmse_mean, render: (r: any) => num(group[r.model]?.rmse_mean, 2) },
            { key: "tr", header: "RMSE prueba", num: true, value: (r: any) => test[r.model]?.rmse, render: (r: any) => num(test[r.model]?.rmse, 2) },
            { key: "t2", header: "R² prueba", num: true, value: (r: any) => test[r.model]?.r2, render: (r: any) => num(test[r.model]?.r2, 3) },
          ]}
        />
        <Explain>
          <p>
            La validación cruzada estima el error en condados nuevos sin tocar la prueba. La agrupada por estado es más pesimista porque el modelo
            no puede apoyarse en lo que sabe de ese estado: es la cifra relevante para predecir regiones sin datos. La prueba se usa una única vez,
            después de elegir, para que su RMSE no esté sesgado a la baja.
          </p>
        </Explain>
      </Card>

      <div className="grid grid-2">
        <Card title="Observado frente a predicho (prueba)" sub="La diagonal es la predicción perfecta.">
          <Plot
            name="prueba-observado-predicho"
            ariaLabel="Mortalidad observada frente a predicha en la partición de prueba"
            height={380}
            deps={[res.run_id]}
            data={(t) => {
              const lo = Math.min(...tp.y, ...tp.pred);
              const hi = Math.max(...tp.y, ...tp.pred);
              return [
                ...REGION_ORDER.map((reg) => {
                  const ix = tp.region.map((r: string, i: number) => (r === reg ? i : -1)).filter((i: number) => i >= 0);
                  return {
                    type: "scattergl",
                    mode: "markers",
                    name: reg,
                    x: ix.map((i: number) => tp.pred[i]),
                    y: ix.map((i: number) => tp.y[i]),
                    text: ix.map((i: number) => tp.county_id[i]),
                    marker: { size: 7, opacity: 0.6, color: (t as any)[REGION_STYLE[reg].color], symbol: REGION_STYLE[reg].symbol },
                    hovertemplate: "%{text}<br>predicho %{x:.1f} · observado %{y:.1f}<extra></extra>",
                  };
                }),
                { type: "scatter", mode: "lines", name: "y = ŷ", x: [lo, hi], y: [lo, hi], line: { color: t.ink, width: 1.5, dash: "dash" } },
              ];
            }}
            layout={() => ({ xaxis: { title: { text: "Predicho" } }, yaxis: { title: { text: "Observado" } } })}
          />
        </Card>
        <Card title="Importancia por permutación" sub="Cuánto empeora el RMSE en prueba al desordenar cada variable.">
          <Plot
            name="importancia"
            ariaLabel="Importancia por permutación de cada variable"
            height={380}
            deps={[res.run_id]}
            data={(t) => [
              {
                type: "bar",
                orientation: "h",
                x: imp.map((r: any) => r.importancia),
                y: imp.map((r: any) => r.etiqueta),
                error_x: { type: "data", array: imp.map((r: any) => r.dt), color: t.ink2, thickness: 1.2, width: 3 },
                marker: { color: t.s1, cornerradius: 4 },
                hovertemplate: "%{y}: +%{x:.2f}<extra></extra>",
              },
            ]}
            layout={() => ({ margin: { l: 200, r: 16, t: 8, b: 44 }, xaxis: { title: { text: "Aumento del RMSE" } }, showlegend: false })}
          />
        </Card>
      </div>

      <div className="grid grid-2">
        <Card
          title="Dependencia parcial"
          sub="Predicción media al fijar la variable en cada valor (del percentil 5 al 95)."
          actions={
            <select className="select" value={pdVar} onChange={(ev) => setPdVar(ev.target.value)} aria-label="Variable">
              {p.partial_dependence.map((r: any) => (
                <option key={r.variable} value={r.variable}>{r.etiqueta}</option>
              ))}
            </select>
          }
        >
          {pd && (
            <Plot
              name={`dependencia-parcial-${pdVar}`}
              ariaLabel={`Dependencia parcial de ${pd.etiqueta}`}
              height={300}
              deps={[res.run_id, pdVar]}
              data={(t) => [{ type: "scatter", mode: "lines", x: pd.grid, y: pd.average, line: { color: t.s1, width: 2.5 }, hovertemplate: "%{x:.2f} → %{y:.1f}<extra></extra>" }]}
              layout={() => ({ xaxis: { title: { text: pd.etiqueta } }, yaxis: { title: { text: "Mortalidad predicha media" } }, showlegend: false })}
            />
          )}
        </Card>
        <Card title="Intervalos de predicción conformales" sub={`Cobertura nominal ${pct(100 * p.conformal.nominal, 0)}; normalizados por σ(población).`}>
          <DataTable
            name="conformal"
            csv={false}
            rows={p.conformal.by_population}
            columns={[
              { key: "tercil", header: "Condados" },
              { key: "cobertura_estandar", header: "Cobertura estándar", num: true, render: (r: any) => pct(100 * r.cobertura_estandar) },
              { key: "anchura_estandar", header: "Anchura", num: true, render: (r: any) => num(r.anchura_estandar, 1) },
              { key: "cobertura_normalizada", header: "Cobertura normalizada", num: true, render: (r: any) => pct(100 * r.cobertura_normalizada) },
              { key: "anchura_normalizada", header: "Anchura", num: true, render: (r: any) => num(r.anchura_normalizada, 1) },
            ]}
          />
          <Explain>
            <p>
              El intervalo conformal garantiza la cobertura sin suponer normalidad. El estándar tiene la misma anchura para todos los condados y
              cubre de menos en los pequeños (más ruidosos) y de más en los grandes; el normalizado ensancha el intervalo donde la varianza es
              mayor y equilibra la cobertura entre tamaños.
            </p>
          </Explain>
        </Card>
      </div>
    </Section>
  );
}
