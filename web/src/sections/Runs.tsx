import { Plot } from "../charts/Plot";
import { Icon } from "../components/icons";
import { Card, DataTable, Explain, Section } from "../components/ui";
import { download, num } from "../lib/format";
import { useApp } from "../lib/store";

const FILES = [
  { name: "results.json", label: "Resultados completos (JSON)" },
  { name: "config.yaml", label: "Configuración de la corrida (YAML)" },
  { name: "manifest.json", label: "Manifiesto: entorno, versiones y huellas" },
  { name: "log.txt", label: "Registro de la ejecución" },
  { name: "modelo_final.sps", label: "Sintaxis SPSS que reproduce el modelo" },
];

export default function Runs() {
  const { res, source, index, runId, setRunId } = useApp();
  const man = res.manifest;
  const timings = Object.entries(man.timings_s as Record<string, number>);

  return (
    <Section
      eyebrow="11 · Corridas y exportación"
      title="Cada corrida queda registrada y es reproducible"
      lead="Una corrida guarda su configuración, el entorno, las huellas del código y de los datos, el registro y todos los resultados. El informe y esta interfaz muestran siempre la última."
    >
      <Card title="Corridas disponibles" sub={source.mode === "static" ? "En la versión publicada sólo viaja la última corrida; el repositorio las conserva todas en runs/." : "Seleccione una corrida para explorarla."}>
        <DataTable
          name="corridas"
          rows={[...index.runs].reverse()}
          columns={[
            { key: "id", header: "Corrida", render: (r: any) => <code>{r.id}</code> },
            { key: "name", header: "Configuración", value: (r: any) => r.name ?? r.config_name ?? "" },
            { key: "r2_adj", header: "R² aj.", num: true, render: (r: any) => num(r.r2_adj as number, 3) },
            { key: "test_rmse", header: "RMSE prueba", num: true, render: (r: any) => num(r.test_rmse as number, 2) },
            {
              key: "sel",
              header: "",
              render: (r: any) =>
                r.id === runId ? (
                  <span className="badge">en uso</span>
                ) : (
                  <button className="btn btn-sm" onClick={() => setRunId(r.id)}>Ver</button>
                ),
            },
          ]}
        />
      </Card>

      <div className="grid grid-2">
        <Card title="Descargas de la corrida" sub={<code>{runId}</code>}>
          <div className="stack" style={{ gap: 8 }}>
            {FILES.map((f) => (
              <a key={f.name} className="btn" href={source.fileUrl(runId, f.name)} download={f.name}>
                <Icon name="file" /> {f.label}
              </a>
            ))}
            <button className="btn" onClick={() => download(`residuos-${runId}.json`, JSON.stringify(res.effects.residuals), "application/json")}>
              <Icon name="download" /> Residuos y medidas de influencia (JSON)
            </button>
          </div>
          <Explain title="Cómo reproducir esta corrida">
            <p>
              Con el repositorio clonado: <code>uv sync</code> y después <code>uv run cancerstats run -c config.yaml</code> con el fichero de
              configuración descargado. La huella de la configuración ({man.config_fingerprint}) y la del fichero de datos permiten comprobar que
              se analiza exactamente lo mismo. En SPSS, ejecute la sintaxis junto a <code>practica.sav</code>.
            </p>
          </Explain>
        </Card>
        <Card title="Procedencia" sub={`Creada ${new Date(man.created_at).toLocaleString("es-ES")} · ${num(man.duration_s, 0)} s`}>
          <div className="table-wrap">
            <table className="data">
              <tbody>
                <tr><th scope="row">Commit</th><td className="mono">{man.git?.commit?.slice(0, 12) ?? "—"} {man.git?.dirty ? "(con cambios sin confirmar)" : "(árbol limpio)"}</td></tr>
                <tr><th scope="row">Semilla</th><td className="mono">{man.seed}</td></tr>
                <tr><th scope="row">Datos (SHA-256)</th><td className="mono" style={{ wordBreak: "break-all", whiteSpace: "normal" }}>{man.data_sha256}</td></tr>
                {Object.entries(man.environment ?? {}).map(([k, v]) => (
                  <tr key={k}><th scope="row">{k}</th><td className="mono">{String(v)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <Card title="Tiempo por etapa">
        <Plot
          name="tiempos"
          ariaLabel="Tiempo de cada etapa de la corrida"
          height={280}
          deps={[runId]}
          data={(t) => [
            {
              type: "bar",
              orientation: "h",
              x: timings.map(([, v]) => v),
              y: timings.map(([k]) => k),
              marker: { color: t.s1, cornerradius: 4 },
              hovertemplate: "%{y}: %{x:.1f} s<extra></extra>",
            },
          ]}
          layout={() => ({ margin: { l: 110, r: 16, t: 8, b: 44 }, xaxis: { title: { text: "Segundos" } }, yaxis: { autorange: "reversed" }, showlegend: false })}
        />
      </Card>
    </Section>
  );
}
