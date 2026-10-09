import { useState } from "react";
import { Card, DataTable, Explain, Section, Stat, StatusTag } from "../components/ui";
import { int } from "../lib/format";
import { useApp } from "../lib/store";

function Evidence({ ev }: { ev: Record<string, unknown> }) {
  const entries = Object.entries(ev ?? {});
  if (!entries.length) return null;
  return (
    <div className="table-wrap" style={{ marginTop: 8 }}>
      <table className="data">
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <th scope="row" style={{ cursor: "default" }}>{k.replace(/_/g, " ")}</th>
              <td className="wrap">{typeof v === "object" ? JSON.stringify(v) : String(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Cleaning() {
  const { res } = useApp();
  const v = res.validation;
  const clean = Object.fromEntries((v.clean as any[]).map((r) => [r.id, r]));
  const rows = (v.raw as any[]).map((r) => ({ ...r, after: clean[r.id] }));
  const [open, setOpen] = useState<string | null>(res.cleaning.decisions[0]?.id ?? null);

  return (
    <Section
      eyebrow="2 · Validación y depuración"
      title="Diecinueve reglas, once decisiones documentadas"
      lead="Cada incidencia se detecta con una regla explícita y se trata con una decisión justificada con evidencia de los propios datos. Ninguna corrección se aplica «porque sí»."
    >
      <div className="grid grid-4">
        <Stat label="Reglas de validación" value={int(v.summary_raw.total)} note="dominio, coherencia, identidades y plausibilidad" />
        <Stat label="Con errores en el original" value={int(v.summary_raw.error)} note={`${int(v.summary_raw.advertencia)} con advertencia`} />
        <Stat label="Con errores tras depurar" value={int(v.summary_clean.error)} note={`${int(v.summary_clean.advertencia)} con advertencia`} />
        <Stat label="Decisiones de depuración" value={int(res.cleaning.decisions.length)} note="D01 a D11" />
      </div>

      <Card title="Catálogo de reglas: antes y después de depurar">
        <DataTable
          name="reglas-validacion"
          rows={rows}
          columns={[
            { key: "id", header: "Regla" },
            { key: "title", header: "Qué comprueba", wrap: true },
            { key: "category", header: "Tipo" },
            { key: "n_violations", header: "Incidencias (original)", num: true, render: (r: any) => int(r.n_violations) },
            { key: "status", header: "Original", render: (r: any) => <StatusTag status={r.status} /> },
            { key: "after", header: "Depurado", value: (r: any) => r.after?.status, render: (r: any) => <StatusTag status={r.after?.status} /> },
            { key: "treatment", header: "Tratamiento", wrap: true },
          ]}
        />
        <Explain>
          <p>
            «Información» no es un fallo: describe rasgos de los datos que hay que tener en cuenta (por ejemplo, los valores perdidos declarados o
            la abundancia de ceros en los ensayos clínicos). Tras la depuración no queda ninguna regla en estado de error.
          </p>
        </Explain>
      </Card>

      <Card title="Decisiones de depuración" sub="Pulse una decisión para ver el problema, la acción, la justificación y la evidencia.">
        <div className="stack" style={{ gap: 8 }}>
          {(res.cleaning.decisions as any[]).map((d) => (
            <details key={d.id} className="explain" open={open === d.id} onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && setOpen(d.id)}>
              <summary>
                <span className="badge">{d.id}</span> {d.title}
                <span className="muted small" style={{ marginLeft: "auto" }}>{int(d.n_affected)} afectados</span>
              </summary>
              <p><strong>Problema.</strong> {d.problem}</p>
              <p><strong>Acción.</strong> {d.action}</p>
              <p><strong>Justificación.</strong> {d.rationale}</p>
              <p className="small muted">Clave de configuración: <code>{d.config_key}</code></p>
              <Evidence ev={d.evidence} />
              {d.affected?.length > 0 && (
                <p className="small" style={{ marginTop: 8 }}>
                  Ejemplos: {d.affected.slice(0, 8).join("; ")}
                  {d.affected.length > 8 ? "…" : ""}
                </p>
              )}
            </details>
          ))}
        </div>
      </Card>
    </Section>
  );
}
