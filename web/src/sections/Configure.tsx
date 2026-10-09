import { useEffect, useMemo, useState } from "react";
import type { JobEvent } from "../api/source";
import { Icon } from "../components/icons";
import { Card, ErrorBox, Explain, Loading, Section } from "../components/ui";
import { download } from "../lib/format";
import { useApp } from "../lib/store";
import { diff, toYaml } from "../lib/yaml";

type Schema = Record<string, any>;

function resolve(schema: Schema, node: Schema): Schema {
  if (node?.$ref) return resolve(schema, schema.$defs[node.$ref.split("/").pop()]);
  if (node?.allOf?.length === 1) return resolve(schema, node.allOf[0]);
  if (node?.anyOf) {
    const nonNull = node.anyOf.filter((n: Schema) => n.type !== "null");
    if (nonNull.length === 1) return { ...resolve(schema, nonNull[0]), nullable: true, title: node.title, description: node.description };
  }
  return node;
}

function Field({ name, node, value, onChange }: { name: string; node: Schema; value: any; onChange: (v: any) => void }) {
  const id = `cfg-${name}`;
  const title = node.title ?? name;
  const help = node.description;
  let input;
  if (node.type === "boolean") {
    input = (
      <label className="row" style={{ gap: 8 }}>
        <input id={id} type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /> {value ? "Sí" : "No"}
      </label>
    );
  } else if (node.enum) {
    input = (
      <select id={id} className="select" value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {node.enum.map((o: string) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    );
  } else if (node.type === "integer" || node.type === "number") {
    input = (
      <input
        id={id}
        className="input"
        type="number"
        step={node.type === "integer" ? 1 : "any"}
        min={node.minimum ?? node.exclusiveMinimum}
        max={node.maximum ?? node.exclusiveMaximum}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value === "" ? (node.nullable ? null : value) : Number(e.target.value))}
      />
    );
  } else if (node.type === "array") {
    const nested = node.items?.type === "array";
    const text = (value ?? []).map((v: any) => (Array.isArray(v) ? v.join(", ") : v)).join("\n");
    input = (
      <textarea
        id={id}
        className="input mono"
        rows={Math.min(8, Math.max(3, (value ?? []).length + 1))}
        defaultValue={text}
        onBlur={(e) => {
          const lines = e.target.value.split("\n").map((l) => l.trim()).filter(Boolean);
          onChange(nested ? lines.map((l) => l.split(",").map((s) => s.trim())) : lines);
        }}
      />
    );
  } else if (node.type === "object" || node.type === undefined) {
    return null;
  } else {
    input = <input id={id} className="input" value={value ?? ""} onChange={(e) => onChange(e.target.value)} />;
  }
  return (
    <div className="field">
      <label htmlFor={id}>{title}</label>
      {input}
      {help && <span className="help">{help}</span>}
      {node.type === "array" && <span className="help">Un elemento por línea{node.items?.type === "array" ? " (pares separados por comas)" : ""}.</span>}
    </div>
  );
}

const SECTION_TITLES: Record<string, string> = {
  meta: "Identificación de la corrida",
  data: "Datos de entrada",
  cleaning: "Validación y depuración",
  transforms: "Transformaciones",
  missing: "Datos ausentes",
  variables: "Variables candidatas",
  outliers: "Atípicos",
  collinearity: "Colinealidad",
  inference: "Inferencia clásica",
  effects: "Modelo de efectos",
  predictive: "Modelo predictivo",
};

const STAGE_TITLES: Record<string, string> = {
  ingesta: "Ingesta",
  validacion: "Validación",
  depuracion: "Depuración",
  ausentes: "Ausentes",
  exploracion: "Exploración",
  atipicos: "Atípicos",
  efectos: "Modelo de efectos",
  prediccion: "Modelo predictivo",
  persistencia: "Persistencia",
  exportacion: "Exportación del informe",
};

export default function Configure() {
  const { source, refreshRuns, setRunId } = useApp();
  const [schema, setSchema] = useState<Schema | null>(null);
  const [base, setBase] = useState<any>(null);
  const [cfg, setCfg] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [check, setCheck] = useState<string | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [job, setJob] = useState<{ status: string; run_id?: string | null; error?: string | null } | null>(null);
  const [open, setOpen] = useState<string>("effects");

  useEffect(() => {
    Promise.all([source.schema(), source.defaultConfig()])
      .then(([s, d]) => {
        setSchema(s);
        setBase(d);
        setCfg(structuredClone(d));
      })
      .catch(setError);
  }, [source]);

  const overrides = useMemo(() => (base && cfg ? diff(base, cfg) ?? {} : {}), [base, cfg]);
  const countLeaves = (o: any): number =>
    o && typeof o === "object" && !Array.isArray(o) ? Object.values(o).reduce((n: number, v) => n + countLeaves(v), 0) : 1;
  const nChanges = Object.keys(overrides).length ? countLeaves(overrides) : 0;
  if (error) return <ErrorBox error={error} />;
  if (!schema || !cfg) return <Loading what="la configuración" />;

  const sections = Object.entries(schema.properties as Record<string, Schema>).map(([k, node]) => [k, resolve(schema, node)] as const);
  const set = (sec: string, key: string, v: any) => {
    setCheck(null);
    setCfg((c: any) => ({ ...c, [sec]: { ...c[sec], [key]: v } }));
  };
  const live = source.mode === "live";
  const running = job?.status === "en curso" || job?.status === "en cola";

  const validate = async () => {
    const r = await source.validate(overrides);
    setCheck(r.ok ? `Configuración válida · huella ${r.fingerprint}` : `No válida: ${JSON.stringify(r.detail)}`);
  };
  const run = async () => {
    setEvents([]);
    setJob({ status: "en cola" });
    try {
      const j = await source.submit(overrides);
      setJob(j);
      source.follow(
        j.id,
        (ev) => setEvents((xs) => [...xs, ev]),
        async (end) => {
          setJob(end);
          if (end.status === "terminada" && end.run_id) {
            await refreshRuns();
            setRunId(end.run_id);
          }
        },
      );
    } catch (e) {
      setJob({ status: "error", error: String(e) });
    }
  };
  const progress = job?.status === "terminada" ? 1 : events.length ? events[events.length - 1].progress : 0;
  const stageState = new Map<string, string>();
  events.forEach((e) => stageState.set(e.stage, e.status));

  return (
    <Section
      eyebrow="12 · Configurar y ejecutar"
      title="Cambie la metodología y vuelva a correr el análisis"
      lead={
        live
          ? "Modo en vivo: los cambios se validan en el servidor y la corrida se ejecuta con seguimiento etapa a etapa. Al terminar, toda la interfaz pasa a mostrar la nueva corrida."
          : "Versión publicada (estática): puede explorar y descargar la configuración, pero para ejecutar hace falta el modo en vivo (uv run cancerstats serve)."
      }
    >
      <Card
        title="Configuración"
        sub={`${nChanges} cambio${nChanges === 1 ? "" : "s"} respecto de la metodología por defecto.`}
        actions={
          <>
            <button className="btn btn-sm" onClick={() => setCfg(structuredClone(base))} disabled={!nChanges}>Restablecer</button>
            <button className="btn btn-sm" onClick={() => download("config.yaml", toYaml(cfg) + "\n", "text/yaml;charset=utf-8")}>
              <Icon name="download" size={16} /> YAML completo
            </button>
          </>
        }
      >
        <div className="stack" style={{ gap: 8 }}>
          {sections.map(([sec, node]) => (
            <details key={sec} className="explain" open={open === sec} onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && setOpen(sec)} style={{ borderLeftColor: overrides[sec] ? "var(--s2)" : undefined }}>
              <summary>
                {SECTION_TITLES[sec] ?? node.title ?? sec}
                {overrides[sec] && <span className="badge" style={{ marginLeft: 8 }}>modificada</span>}
              </summary>
              {node.description && <p className="small ink2">{node.description}</p>}
              <div className="grid grid-2">
                {Object.entries((node.properties ?? {}) as Record<string, Schema>).map(([key, f]) => (
                  <Field key={key} name={`${sec}.${key}`} node={resolve(schema, f)} value={cfg[sec]?.[key]} onChange={(v) => set(sec, key, v)} />
                ))}
              </div>
            </details>
          ))}
        </div>
      </Card>

      <div className="grid grid-2">
        <Card title="Cambios (YAML)" sub="Lo que se enviará: sólo las claves que difieren de la configuración por defecto.">
          <pre className="equation">{nChanges ? toYaml(overrides) : "# sin cambios"}</pre>
          <div className="row">
            <button className="btn" onClick={validate} disabled={!live}>Validar</button>
            <button className="btn primary" onClick={run} disabled={!live || running}>
              <Icon name="play" /> {running ? "Ejecutando…" : "Ejecutar corrida"}
            </button>
          </div>
          {check && <p className="small" style={{ marginTop: 8 }}>{check}</p>}
          {!live && (
            <Explain title="Ejecutar en local" open>
              <p>
                Descargue el YAML y ejecute <code>uv run cancerstats run -c config.yaml</code>, o arranque la interfaz en vivo con{" "}
                <code>uv run cancerstats serve</code> y abra <code>http://127.0.0.1:8000</code>.
              </p>
            </Explain>
          )}
        </Card>
        <Card title="Progreso" sub={job ? `Estado: ${job.status}` : "Sin corridas lanzadas en esta sesión."}>
          <div className="progress" aria-label="Progreso de la corrida" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <ul className="timeline" style={{ marginTop: 12 }}>
            {Object.entries(STAGE_TITLES).map(([k, title]) => {
              const st = stageState.get(k);
              return (
                <li key={k} style={{ opacity: st ? 1 : 0.5 }}>
                  <Icon name={st === "fin" ? "ok" : st === "error" ? "error" : st ? "history" : "info"} style={{ color: st === "fin" ? "var(--good)" : st === "error" ? "var(--critical)" : "var(--muted)" }} />
                  {title}
                  {st === "inicio" && <span className="muted small">en curso…</span>}
                </li>
              );
            })}
          </ul>
          {job?.error && <div className="error-box" style={{ marginTop: 10 }}>{job.error}</div>}
          {events.length > 0 && (
            <details className="explain">
              <summary>Registro ({events.length} eventos)</summary>
              <pre className="equation" style={{ maxHeight: 240, overflow: "auto" }}>
                {events.map((e) => `[${e.t ?? ""}s] ${e.stage} ${e.status} ${e.message}`).join("\n")}
              </pre>
            </details>
          )}
        </Card>
      </div>
    </Section>
  );
}
