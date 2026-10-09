import { Plot } from "../charts/Plot";
import { Card, DataTable, Explain, Section, Stat } from "../components/ui";
import { int, num } from "../lib/format";
import { useApp } from "../lib/store";

export default function Outliers() {
  const { res } = useApp();
  const o = res.outliers;
  const mv = o.multivariate;
  const d2 = Object.values(mv.d2 as Record<string, number>).filter((v) => Number.isFinite(v));
  const logd2 = d2.map((v) => Math.log10(Math.max(v, 1e-3)));
  const resp = o.univariate.find((u: any) => u.variable === "TARGET_deathRate");

  return (
    <Section
      eyebrow="5 · Atípicos"
      title="Valores atípicos: detectar no es eliminar"
      lead="Se buscan atípicos en cada variable (Tukey y z robusto) y en el espacio conjunto (distancia de Mahalanobis robusta, MCD). Un atípico verificado como real se conserva; su influencia sobre el modelo se evalúa después."
    >
      <div className="grid grid-4">
        <Stat label="Atípicos de Tukey en la respuesta" value={int(resp?.n_tukey)} note={`vallas ${num(resp?.valla_inferior, 1)} y ${num(resp?.valla_superior, 1)}`} />
        <Stat label="Extremos (3 IQR)" value={int(resp?.n_extremos)} note="en la mortalidad" />
        <Stat label="Multivariantes (MCD)" value={int(mv.n_flagged_robust)} note={`de ${int(mv.n)}; clásica: ${int(mv.n_flagged_classical)}`} />
        <Stat label="Umbral χ²" value={num(mv.cutoff, 1)} note={`cuantil ${num(mv.quantile, 3)} con ${mv.p} g.l.`} />
      </div>

      <Card title="Distancia de Mahalanobis robusta" sub="Escala logarítmica. La línea marca el umbral χ².">
        <Plot
          name="mahalanobis"
          ariaLabel="Histograma de la distancia de Mahalanobis robusta"
          height={300}
          deps={[res.run_id]}
          data={(t) => [{ type: "histogram", x: logd2, nbinsx: 60, marker: { color: t.s1, opacity: 0.8 }, hovertemplate: "log₁₀ d² ≈ %{x:.2f}: %{y} condados<extra></extra>" }]}
          layout={(t) => ({
            xaxis: { title: { text: "log₁₀ d² robusta" } },
            yaxis: { title: { text: "Condados" } },
            shapes: [{ type: "line", x0: Math.log10(mv.cutoff), x1: Math.log10(mv.cutoff), y0: 0, y1: 1, yref: "paper", line: { color: t.critical, width: 2, dash: "dash" } }],
            showlegend: false,
          })}
        />
        <Explain>
          <p>
            La versión robusta marca muchos más condados que la clásica porque la clásica se «contamina»: los atípicos inflan la covarianza y se
            ocultan a sí mismos (efecto máscara). Muchos condados extremos son reales (poco poblados, reservas indígenas, enclaves de altos
            ingresos), así que no se eliminan: el diagnóstico de influencia y el ajuste sin ellos (sensibilidad) miden si cambian las conclusiones.
          </p>
        </Explain>
      </Card>

      <div className="grid grid-2">
        <Card title="Condados más atípicos (multivariante)">
          <DataTable
            name="atipicos-multivariantes"
            rows={mv.top}
            columns={[
              { key: "county_id", header: "Condado" },
              { key: "d2_robusta", header: "d² robusta", num: true, render: (r: any) => num(r.d2_robusta, 0) },
              { key: "d2_clasica", header: "d² clásica", num: true, render: (r: any) => num(r.d2_clasica, 1) },
              { key: "variables", header: "Variables que más contribuyen", value: (r: any) => r.variables.join(", "), wrap: true },
            ]}
          />
        </Card>
        <Card title="Mortalidad extrema">
          <DataTable
            name="atipicos-respuesta"
            rows={o.response}
            columns={[
              { key: "county_id", header: "Condado" },
              { key: "valor", header: "Mortalidad", num: true, render: (r: any) => num(r.valor, 1) },
              { key: "incidencia", header: "Incidencia", num: true, render: (r: any) => num(r.incidencia, 1) },
              { key: "poblacion", header: "Población", num: true, render: (r: any) => int(r.poblacion) },
              { key: "z", header: "z robusto", num: true, render: (r: any) => num(r.z, 2) },
            ]}
          />
        </Card>
      </div>

      <Card title="Atípicos univariantes por variable">
        <DataTable
          name="atipicos-univariantes"
          rows={o.univariate}
          columns={[
            { key: "etiqueta", header: "Variable" },
            { key: "valla_inferior", header: "Valla inf.", num: true, render: (r: any) => num(r.valla_inferior, 2) },
            { key: "valla_superior", header: "Valla sup.", num: true, render: (r: any) => num(r.valla_superior, 2) },
            { key: "n_tukey", header: "Tukey (1,5 IQR)", num: true },
            { key: "n_extremos", header: "Extremos (3 IQR)", num: true },
            { key: "n_z_robusto", header: "|z robusto| > 3,5", num: true },
          ]}
        />
      </Card>
    </Section>
  );
}
