import { useState } from "react";
import { Plot } from "../charts/Plot";
import { Card, Explain, Section } from "../components/ui";
import { ci, num, termLabel } from "../lib/format";
import { useApp } from "../lib/store";

export default function Sensitivity() {
  const { res, labels } = useApp();
  const e = res.effects;
  const specs = e.sensitivity as any[];
  const terms = (e.final.coef as any[]).map((c) => c.term).filter((t) => t !== "const");
  const [term, setTerm] = useState<string>(terms.find((t) => t === "PctBachDeg25_Over") ?? terms[0]);
  const L = (t: string) => termLabel(t, labels);
  const boot = e.bootstrap;
  const bootMap = (k: string): Record<string, number> => {
    const v = boot?.[k];
    if (!v) return {};
    return Array.isArray(v) ? Object.fromEntries(v) : v;
  };
  const bc = bootMap("coef");
  const bl = bootMap("lower");
  const bu = bootMap("upper");
  const rows = [
    ...specs.filter((s) => s.coef[term] !== undefined).map((s) => ({ label: s.label, coef: s.coef[term], lower: s.lower?.[term], upper: s.upper?.[term], p: s.p?.[term], n: s.n, main: s.id === "main" })),
    ...(boot && bc[term] !== undefined ? [{ label: `Bootstrap por conglomerados (${boot.reps})`, coef: bc[term], lower: bl[term], upper: bu[term], p: null, n: null, main: false }] : []),
  ];

  return (
    <Section
      eyebrow="9 · Sensibilidad"
      title="¿Dependen las conclusiones de las decisiones tomadas?"
      lead="El mismo modelo se reestima bajo especificaciones alternativas: otra estimación del error, ponderación, efectos fijos de estado, modelo mixto, regresión robusta, sin observaciones influyentes, con imputación múltiple y con bootstrap por conglomerados."
    >
      <Card
        title="Coeficiente bajo cada especificación"
        sub="Punto: estimación; barra: IC 95 %. La especificación principal está resaltada."
        actions={
          <select className="select" value={term} onChange={(ev) => setTerm(ev.target.value)} aria-label="Término">
            {terms.map((t) => (
              <option key={t} value={t}>{L(t)}</option>
            ))}
          </select>
        }
      >
        <Plot
          name={`sensibilidad-${term}`}
          ariaLabel={`Coeficiente de ${L(term)} bajo especificaciones alternativas`}
          height={Math.max(300, rows.length * 38 + 70)}
          deps={[res.run_id, term]}
          data={(t) => [
            {
              type: "scatter",
              mode: "markers",
              x: rows.map((r) => r.coef),
              y: rows.map((r) => r.label),
              error_x: {
                type: "data",
                symmetric: false,
                array: rows.map((r) => (r.upper ?? r.coef) - r.coef),
                arrayminus: rows.map((r) => r.coef - (r.lower ?? r.coef)),
                color: t.ink2,
                thickness: 2,
                width: 0,
              },
              marker: { size: rows.map((r) => (r.main ? 15 : 11)), color: rows.map((r) => (r.main ? t.s2 : t.s1)), line: { color: t.surface, width: 2 } },
              hovertemplate: "%{y}<br>β̂ = %{x:.3f}<extra></extra>",
            },
          ]}
          layout={(t) => ({
            margin: { l: 330, r: 16, t: 8, b: 44 },
            xaxis: { title: { text: `Coeficiente de ${L(term)}` }, zeroline: true, zerolinecolor: t.ink2 },
            yaxis: { autorange: "reversed" },
            showlegend: false,
          })}
        />
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th>Especificación</th><th className="num">β̂</th><th>IC 95 %</th><th className="num">n</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label}>
                  <td style={{ fontWeight: r.main ? 700 : 400 }}>{r.label}</td>
                  <td className="num">{num(r.coef, 3)}</td>
                  <td>{r.lower !== undefined ? ci(r.lower, r.upper, 3) : "—"}</td>
                  <td className="num">{r.n ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Explain>
          <p>
            Si el signo y la magnitud se mantienen en todas las especificaciones, la conclusión no depende de una decisión discutible. Los efectos
            fijos de estado son la prueba más exigente: comparan condados dentro del mismo estado y eliminan cualquier factor estatal no medido.
          </p>
          <ul>
            {specs.filter((s) => s.note).map((s) => (
              <li key={s.id}><strong>{s.label}</strong>: {s.note}</li>
            ))}
          </ul>
        </Explain>
      </Card>

      {e.boxcox && (
        <Card title="¿Conviene transformar la respuesta? Perfil de Box-Cox" sub={`λ = ${num(e.boxcox.lambda, 3)} ${ci(e.boxcox.lower, e.boxcox.upper, 3)}`}>
          <Plot
            name="boxcox"
            ariaLabel="Perfil de verosimilitud de Box-Cox"
            height={280}
            deps={[res.run_id]}
            data={(t) => [{ type: "scatter", mode: "lines", x: e.boxcox.grid, y: e.boxcox.loglik, line: { color: t.s1, width: 2 }, hovertemplate: "λ = %{x:.2f}<extra></extra>" }]}
            layout={(t) => ({
              xaxis: { title: { text: "λ" } },
              yaxis: { title: { text: "Log-verosimilitud perfil" } },
              shapes: [e.boxcox.lower, e.boxcox.upper].map((v: number) => ({ type: "line", x0: v, x1: v, y0: 0, y1: 1, yref: "paper", line: { color: t.ink2, dash: "dot", width: 1 } })),
              showlegend: false,
            })}
          />
          <Explain>
            <p>
              El intervalo de λ {e.boxcox.includes_1 ? "incluye 1: no hace falta transformar." : "no incluye exactamente 1, pero la transformación óptima es suave y complicaría la interpretación en muertes por 100 000; se mantiene la escala original, con inferencia robusta."}
            </p>
          </Explain>
        </Card>
      )}
    </Section>
  );
}
