import { useMemo, useState } from "react";
import { Card, DataTable, Explain, Section, Stat } from "../components/ui";
import { Icon } from "../components/icons";
import { int } from "../lib/format";
import { useApp } from "../lib/store";

export default function Data() {
  const { res } = useApp();
  const ing = res.ingest;
  const prov = res.provenance;
  const [q, setQ] = useState("");
  const dict = useMemo(() => {
    const s = q.trim().toLowerCase();
    return (res.dictionary as any[]).filter((v) => !s || `${v.name} ${v.label} ${v.description}`.toLowerCase().includes(s));
  }, [res.dictionary, q]);
  const excluded = Object.entries(res.excluded as Record<string, string>).map(([variable, motivo]) => ({ variable, motivo }));

  return (
    <Section
      eyebrow="1 · Datos"
      title="Fuente oficial y procedencia"
      lead="El profesor entregó practica.sav. Antes de analizar se comprueba que es el conjunto público de Kaggle, celda a celda, y se documenta cada variable."
    >
      <div className="grid grid-4">
        <Stat label="Condados" value={int(ing.n_rows)} note="una fila por condado" />
        <Stat label="Columnas originales" value={int(ing.n_columns_raw)} note={`formato ${ing.format === "spss" ? "SPSS (.sav)" : "CSV"}`} />
        <Stat label="Variables comparadas" value={`${prov?.n_identical ?? "—"} / ${prov?.n_columns_compared ?? "—"}`} note="idénticas al CSV de Kaggle" />
        <Stat
          label="Equivalencia de fuentes"
          value={
            <span className="status" style={{ fontSize: "1.3rem" }}>
              <Icon name={prov?.equivalent ? "ok" : "warn"} size={26} style={{ color: prov?.equivalent ? "var(--good)" : "var(--serious)" }} />
              {prov?.equivalent ? "Sí" : "No"}
            </span>
          }
          note="mismas filas, mismo orden"
        />
      </div>

      <Card title="Huella del fichero" sub="El SHA-256 identifica exactamente el fichero analizado: cualquier cambio de un solo byte lo altera.">
        <div className="table-wrap">
          <table className="data">
            <tbody>
              <tr><th scope="row">Fichero</th><td className="mono">{String(ing.path).split("/").slice(-3).join("/")}</td></tr>
              <tr><th scope="row">SHA-256</th><td className="mono" style={{ whiteSpace: "normal", wordBreak: "break-all" }}>{ing.sha256}</td></tr>
              <tr><th scope="row">Codificación</th><td>{ing.encoding}</td></tr>
              <tr><th scope="row">Columnas no esperadas</th><td>{(ing.unexpected_columns ?? []).join(", ") || "ninguna"}</td></tr>
            </tbody>
          </table>
        </div>
      </Card>

      {prov && (
        <Card title="Diferencias con el CSV público" sub="Las únicas diferencias son valores que el fichero oficial ya trae como perdidos del sistema.">
          <DataTable
            name="procedencia-diferencias"
            rows={prov.differing}
            columns={[
              { key: "variable", header: "Variable" },
              { key: "iguales", header: "Celdas iguales", num: true, render: (r: any) => int(r.iguales) },
              { key: "distintas", header: "Distintas", num: true, render: (r: any) => int(r.distintas) },
              { key: "ausente_solo_oficial", header: "Perdidas sólo en el oficial", num: true },
              { key: "valores_referencia_sustituidos", header: "Valor en Kaggle", render: (r: any) => (r.valores_referencia_sustituidos ?? []).join(", ") },
            ]}
          />
          <Explain>
            <p>
              El fichero oficial añade <code>{(prov.only_official ?? []).join(", ")}</code>, que resulta ser exactamente
              <code> avgAnnCount − avgDeathsPerYear</code> (regla R19). Como contiene las muertes, que son el numerador de la respuesta, se excluye
              de los modelos: usarla sería una fuga de información.
            </p>
          </Explain>
        </Card>
      )}

      <Card title="Diccionario de variables" sub="Etiqueta, unidad y descripción de cada variable (fuente: documentación pública del conjunto).">
        <div className="row" style={{ marginBottom: 10 }}>
          <input className="input" placeholder="Buscar variable…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Buscar en el diccionario" />
          <span className="muted small">{dict.length} variables</span>
        </div>
        <DataTable
          name="diccionario"
          rows={dict}
          columns={[
            { key: "name", header: "Variable", render: (r: any) => <code>{r.name}</code> },
            { key: "label", header: "Etiqueta" },
            { key: "unit", header: "Unidad" },
            { key: "group", header: "Grupo" },
            { key: "description", header: "Descripción", wrap: true },
          ]}
        />
      </Card>

      <Card title="Variables que no entran como explicativas" sub="Cada exclusión tiene un motivo estadístico.">
        <DataTable
          name="variables-excluidas"
          rows={excluded}
          columns={[
            { key: "variable", header: "Variable", render: (r: any) => <code>{r.variable}</code> },
            { key: "motivo", header: "Motivo", wrap: true },
          ]}
        />
      </Card>
    </Section>
  );
}
