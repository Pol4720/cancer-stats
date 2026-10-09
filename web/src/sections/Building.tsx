import { Plot } from "../charts/Plot";
import { Card, DataTable, Explain, Section, Stat } from "../components/ui";
import { int, num, pval, termLabel } from "../lib/format";
import { useApp } from "../lib/store";

export default function Building() {
  const { res, labels } = useApp();
  const e = res.effects;
  const coll = e.collinearity;
  const vifBefore = [...coll.vif_before].sort((a: any, b: any) => b.fiv - a.fiv);
  const vifAfter = Object.fromEntries(coll.vif_after.map((v: any) => [v.variable, v.fiv]));
  const vf = e.variance_function_final;
  const strategies = Object.values(e.strategies) as any[];
  const allTerms = Array.from(new Set(strategies.flatMap((s) => s.selected)));
  const methodName: Record<string, string> = { backward: "Hacia atrás", forward: "Hacia delante", stepwise: "Por pasos" };
  const L = (t: string) => termLabel(t, labels);

  return (
    <Section
      eyebrow="6 · Construcción del modelo"
      title={`De ${e.candidates.length} candidatas al modelo final, paso a paso`}
      lead="La selección no es mecánica: primero se elimina la colinealidad severa, después se seleccionan variables con contrastes robustos, se vigila la confusión de las exposiciones de interés y sólo al final se contrastan interacciones."
    >
      <div className="grid grid-4">
        <Stat label="Términos del modelo máximo (tras la poda)" value={int(e.max_design.terms.length)} note={`n = ${int(e.max_design.n)} casos completos`} />
        <Stat label="Eliminadas por colinealidad" value={int(coll.trace.length)} note="FIV > 10, una a una" />
        <Stat label="Pasos de selección" value={int(e.selection.steps.length)} note={`${methodName[e.selection.method] ?? e.selection.method}, α = 0,05`} />
        <Stat label="Términos finales" value={int(e.final.terms.length)} note={`${e.confounding.length} por confusión, ${e.interactions.filter((i: any) => i.entra).length} interacciones`} />
      </div>

      <Card title="Paso 1 · Colinealidad: factores de inflación de la varianza" sub="Antes (barras) y después de la poda (puntos). La línea marca FIV = 10.">
        <Plot
          name="fiv"
          ariaLabel="Factor de inflación de la varianza antes y después de la poda"
          height={Math.max(360, vifBefore.length * 22 + 60)}
          deps={[res.run_id]}
          data={(t) => [
            { type: "bar", orientation: "h", name: "Antes", x: vifBefore.map((v: any) => v.fiv), y: vifBefore.map((v: any) => L(v.variable)), marker: { color: t.s1, opacity: 0.35, cornerradius: 4 }, hovertemplate: "%{y}: FIV %{x:.1f}<extra>antes</extra>" },
            {
              type: "scatter",
              mode: "markers",
              name: "Después",
              x: vifBefore.map((v: any) => vifAfter[v.variable] ?? null),
              y: vifBefore.map((v: any) => L(v.variable)),
              marker: { color: t.s1, size: 9, line: { color: t.surface, width: 2 } },
              hovertemplate: "%{y}: FIV %{x:.1f}<extra>después</extra>",
            },
          ]}
          layout={(t) => ({
            margin: { l: 210, r: 16, t: 8, b: 44 },
            xaxis: { type: "log", title: { text: "FIV (escala log)" } },
            yaxis: { autorange: "reversed" },
            shapes: [{ type: "line", x0: 10, x1: 10, y0: 0, y1: 1, yref: "paper", line: { color: t.critical, dash: "dash", width: 1.5 } }],
            barmode: "overlay",
          })}
        />
        <DataTable
          name="poda-colinealidad"
          csv={false}
          rows={coll.trace}
          columns={[
            { key: "paso", header: "Paso", num: true },
            { key: "etiqueta", header: "Eliminada" },
            { key: "fiv", header: "FIV", num: true, render: (r: any) => num(r.fiv, 1) },
            { key: "mas_correlada_con", header: "Más correlada con", value: (r: any) => L(r.mas_correlada_con) },
            { key: "r", header: "r", num: true, render: (r: any) => num(r.r, 3) },
          ]}
        />
        <Explain>
          <p>
            El FIV de una variable es 1/(1 − R²) de su regresión sobre las demás: cuánto se infla la varianza de su coeficiente por estar
            correlada con el resto. Se elimina de una en una la de mayor FIV (recalculando cada vez) hasta que todas quedan por debajo de 10. El
            índice de condición de Belsley pasa de {num(coll.belsley_before.numero_condicion, 1)} a {num(coll.belsley_after.numero_condicion, 1)}.
          </p>
        </Explain>
      </Card>

      {vf && (
        <Card title="Paso 2 · Estructura del error: varianza según la población" sub={`σ²(n) = a + b/n con a = ${num(vf.a, 1)} y b = ${num(vf.b / 1e6, 2)}·10⁶ (p de b ${pval(vf.p_b)}).`}>
          <Plot
            name="funcion-varianza"
            ariaLabel="Varianza residual por decil de población"
            height={300}
            deps={[res.run_id]}
            data={(t) => {
              const ed = e.initial.diagnostics.population_decile_edges as number[];
              const mids = ed.slice(0, -1).map((v, i) => Math.sqrt(v * ed[i + 1]));
              const bd = e.variance_by_decile ?? {};
              return [
                { type: "scatter", mode: "lines+markers", name: "MCO (e²)", x: mids, y: bd.mco ?? e.diagnostics.homoscedasticity.var_by_population_decile, line: { color: t.s2, width: 2 }, marker: { size: 8 } },
                ...(bd.mcpf ? [{ type: "scatter", mode: "lines+markers", name: "Ponderado MCPF (w·e²)", x: mids, y: bd.mcpf, line: { color: t.s1, width: 2 }, marker: { size: 8, symbol: "square" } }] : []),
                { type: "scatter", mode: "lines", name: "a + b/n", x: mids, y: mids.map((n) => vf.a + vf.b / n), line: { color: t.ink, width: 1.5, dash: "dot" } },
              ];
            }}
            layout={() => ({ xaxis: { type: "log", title: { text: "Población del condado (escala log)" } }, yaxis: { title: { text: "Varianza residual media" } } })}
          />
          <Explain>
            <p>
              La tasa de un condado pequeño se calcula con pocas muertes, así que es más ruidosa: su varianza crece como 1/n. Por eso el modelo
              final se estima por MCO (lo que pide la orientación) pero con errores típicos robustos, que no suponen varianza constante. La
              estimación ponderada por σ²(n) (MCPF) se usa como sensibilidad: estabiliza la varianza y da coeficientes muy parecidos.
            </p>
          </Explain>
        </Card>
      )}

      <Card title="Paso 3 · Selección de variables" sub="Contrastes de Wald con errores típicos por conglomerados de estado. Las tres estrategias del curso, comparadas.">
        <DataTable
          name="seleccion-pasos"
          csv={false}
          rows={e.selection.steps}
          columns={[
            { key: "paso", header: "Paso", num: true },
            { key: "accion", header: "Acción" },
            { key: "termino", header: "Término", value: (r: any) => L(r.termino) },
            { key: "F", header: "F", num: true, render: (r: any) => num(r.F, 3) },
            { key: "p", header: "p", num: true, render: (r: any) => pval(r.p) },
          ]}
        />
        <hr className="sep" />
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Término</th>
                {strategies.map((s) => (
                  <th key={s.method}>{methodName[s.method] ?? s.method}</th>
                ))}
                <th>Ingenua (MCO clásico)</th>
              </tr>
            </thead>
            <tbody>
              {Array.from(new Set([...allTerms, ...e.naive_selection.selected])).map((t) => (
                <tr key={t}>
                  <td>{L(t)}</td>
                  {strategies.map((s) => (
                    <td key={s.method}>{s.selected.includes(t) ? "✓" : ""}</td>
                  ))}
                  <td>{e.naive_selection.selected.includes(t) ? "✓" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Explain>
          <p>
            Las tres estrategias coinciden en lo esencial. La selección «ingenua», con errores típicos clásicos, retiene {e.naive_selection.selected.length}{" "}
            términos frente a {e.selection.selected.length}: al ignorar la dependencia dentro de cada estado subestima los errores típicos y declara
            significativas variables que no lo son.
          </p>
        </Explain>
      </Card>

      <Card title="Paso 4 · Confusión" sub={`Exposiciones vigiladas: ${(e.exposures as string[]).map(L).join(", ")}. Umbral: cambio relativo del 10 %.`}>
        {e.confounding.length ? (
          <DataTable
            name="confusion"
            csv={false}
            rows={e.confounding}
            columns={[
              { key: "reincorporada", header: "Variable reincorporada", value: (r: any) => L(r.reincorporada) },
              { key: "en", header: "Cambia el coeficiente de", value: (r: any) => L(r.en) },
              { key: "cambio_relativo", header: "Cambio relativo", num: true, render: (r: any) => `${num(100 * r.cambio_relativo, 1)} %` },
            ]}
          />
        ) : (
          <p>Ninguna variable eliminada altera más del 10 % el coeficiente de una exposición.</p>
        )}
        <Explain>
          <p>
            Una variable no significativa puede ser necesaria si, al quitarla, cambia el efecto estimado de un factor de interés: es una confusora.
            El criterio del cambio en la estimación (Greenland) la reincorpora. Se aplica sólo a las exposiciones socioeconómicas que motivan el
            estudio (educación, pobreza y renta) para no inflar el modelo con covariables irrelevantes.
          </p>
        </Explain>
      </Card>

      <Card title="Paso 5 · Interacciones" sub="Principio jerárquico (sólo con sus efectos principales en el modelo) y corrección de Holm en cada ronda.">
        <DataTable
          name="interacciones"
          csv={false}
          rows={e.interactions}
          columns={[
            { key: "ronda", header: "Ronda", num: true },
            { key: "termino", header: "Interacción", value: (r: any) => L(r.termino) },
            { key: "F", header: "F", num: true, render: (r: any) => num(r.F, 2) },
            { key: "gl", header: "g.l.", num: true },
            { key: "p", header: "p", num: true, render: (r: any) => pval(r.p) },
            { key: "p_ajustado", header: "p (Holm)", num: true, render: (r: any) => pval(r.p_ajustado) },
            { key: "entra", header: "Entra", render: (r: any) => (r.entra ? "✓ sí" : "no") },
          ]}
        />
      </Card>

      <Card title="Paso 6 · Comparación de modelos anidados" sub="Misma muestra; F parcial frente al modelo anterior.">
        <DataTable
          name="comparacion-modelos"
          csv={false}
          rows={e.model_comparison}
          columns={[
            { key: "modelo", header: "Modelo", wrap: true },
            { key: "k", header: "k", num: true },
            { key: "r2", header: "R²", num: true, render: (r: any) => num(r.r2, 3) },
            { key: "r2_adj", header: "R² aj.", num: true, render: (r: any) => num(r.r2_adj, 3) },
            { key: "aic", header: "AIC", num: true, render: (r: any) => num(r.aic, 1) },
            { key: "bic", header: "BIC", num: true, render: (r: any) => num(r.bic, 1) },
            { key: "F_parcial", header: "F parcial", num: true, render: (r: any) => num(r.F_parcial, 2) },
            { key: "p_parcial", header: "p", num: true, render: (r: any) => pval(r.p_parcial) },
          ]}
        />
      </Card>
    </Section>
  );
}
