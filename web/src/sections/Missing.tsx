import { useState } from "react";
import { Plot } from "../charts/Plot";
import { Card, DataTable, Explain, Section, Segmented, Stat } from "../components/ui";
import { int, num, pct, pval } from "../lib/format";
import { useApp } from "../lib/store";

export default function Missing() {
  const { res, labels } = useApp();
  const m = res.missing;
  const cols = [...m.by_column].sort((a: any, b: any) => b.pct - a.pct);
  const [only, setOnly] = useState<"sig" | "all">("sig");
  const comps = (m.comparisons as any[]).filter((c) => only === "all" || c.p_welch_ajustado < 0.05);

  return (
    <Section
      eyebrow="3 · Datos ausentes"
      title="Cuánto falta, dónde y por qué"
      lead="El tratamiento correcto de los ausentes depende de su mecanismo. Se contrasta si son completamente aleatorios (MCAR) y, si no, de qué dependen."
    >
      <div className="grid grid-3">
        <Stat label="Variables con ausentes" value={int(m.by_column.length)} note={cols.map((c: any) => labels[c.variable] ?? c.variable).join(", ")} />
        <Stat label="Little: variables de ausencia aleatoria" value={`p = ${pval(m.little_random.p_value)}`} note={`χ² = ${num(m.little_random.statistic, 1)}, ${m.little_random.df} g.l.: ${m.little_random.p_value < 0.05 ? "se rechaza" : "no se rechaza"} MCAR`} />
        <Stat label="Little: todas las variables" value={`p = ${pval(m.little_all.p_value)}`} note={`χ² = ${num(m.little_all.statistic, 1)}, ${m.little_all.df} g.l.: ${m.little_all.p_value < 0.05 ? "se rechaza" : "no se rechaza"} MCAR`} />
      </div>

      <div className="grid grid-2">
        <Card title="Porcentaje de ausentes por variable">
          <Plot
            name="ausentes-por-variable"
            ariaLabel="Porcentaje de valores ausentes por variable"
            height={240}
            deps={[res.run_id]}
            data={(t) => [
              {
                type: "bar",
                orientation: "h",
                x: cols.map((c: any) => c.pct),
                y: cols.map((c: any) => labels[c.variable] ?? c.variable),
                marker: { color: t.s1, cornerradius: 4 },
                text: cols.map((c: any) => pct(c.pct)),
                textposition: "outside",
                textfont: { color: t.ink2 },
                hovertemplate: "%{y}: %{x:.1f} %<extra></extra>",
              },
            ]}
            layout={() => ({ margin: { l: 170, r: 50, t: 8, b: 40 }, xaxis: { title: { text: "% de condados" } }, showlegend: false })}
          />
        </Card>
        <Card title="Patrones de ausencia" sub="Qué variables faltan juntas.">
          <DataTable
            name="patrones-ausencia"
            rows={m.patterns}
            columns={[
              { key: "faltan", header: "Faltan", value: (r: any) => (r.faltan.length ? r.faltan.map((v: string) => labels[v] ?? v).join(" + ") : "nada"), wrap: true },
              { key: "n", header: "Condados", num: true, render: (r: any) => int(r.n) },
              { key: "pct", header: "%", num: true, render: (r: any) => num(r.pct, 1) },
            ]}
          />
        </Card>
      </div>

      <Card title="Mecanismo de cada variable">
        <ul>
          {Object.entries(m.mechanism as Record<string, string>).map(([k, v]) => (
            <li key={k} style={{ marginBottom: 6 }}>
              <strong>{labels[k] ?? k}</strong>: {v}
            </li>
          ))}
        </ul>
        <Explain title="¿Por qué importa el mecanismo?">
          <p>
            <strong>MCAR</strong> (completamente al azar): eliminar los casos incompletos no sesga, sólo pierde precisión. <strong>MAR</strong> (al azar
            dado lo observado): el análisis de casos completos puede sesgar, y la imputación múltiple con las variables que explican la ausencia lo
            corrige. <strong>MNAR</strong> (depende del propio valor no observado): ningún método lo resuelve sin supuestos externos.
          </p>
          <p>
            La incidencia falta por estados completos (registros que no la publican): es una ausencia estructural, MAR dado el estado. El modelo
            final usa casos completos y la imputación múltiple (reglas de Rubin) es un análisis de sensibilidad.
          </p>
        </Explain>
      </Card>

      <Card
        title="¿Se parecen los condados con y sin ausentes?"
        sub="Medias de cada variable en los condados con y sin el dato ausente (Welch, corrección de Holm)."
        actions={<Segmented label="Filtro" value={only} onChange={setOnly} options={[{ value: "sig", label: "Significativas" }, { value: "all", label: "Todas" }]} />}
      >
        <DataTable
          name="comparacion-ausentes"
          rows={comps}
          columns={[
            { key: "con_ausencia", header: "Ausente en", value: (r: any) => labels[r.con_ausencia] ?? r.con_ausencia },
            { key: "variable", header: "Variable comparada", value: (r: any) => labels[r.variable] ?? r.variable },
            { key: "media_ausentes", header: "Media (ausentes)", num: true, render: (r: any) => num(r.media_ausentes, 2) },
            { key: "media_observados", header: "Media (observados)", num: true, render: (r: any) => num(r.media_observados, 2) },
            { key: "dif_tipificada", header: "Dif. tipificada", num: true, render: (r: any) => num(r.dif_tipificada, 2) },
            { key: "p_welch_ajustado", header: "p (Holm)", num: true, render: (r: any) => pval(r.p_welch_ajustado) },
          ]}
        />
      </Card>
    </Section>
  );
}
