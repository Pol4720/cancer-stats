import { motion } from "motion/react";
import { Component, useMemo, useState, type ErrorInfo, type ReactNode } from "react";
import { download, toCsv } from "../lib/format";
import { Icon, type IconName } from "./icons";

export function Section({ eyebrow, title, lead, children }: { eyebrow: string; title: string; lead?: ReactNode; children: ReactNode }) {
  return (
    <motion.section initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, ease: "easeOut" }}>
      <header className="section-head">
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        {lead && <p className="lead">{lead}</p>}
      </header>
      <div className="stack">{children}</div>
    </motion.section>
  );
}

export function Card({ title, sub, children, actions, className = "" }: { title?: ReactNode; sub?: ReactNode; children: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <motion.div
      className={`card ${className}`}
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      {(title || actions) && (
        <div className="card-head">
          {title && <h3>{title}</h3>}
          {actions && <div className="card-actions">{actions}</div>}
        </div>
      )}
      {sub && <div className="sub">{sub}</div>}
      {children}
    </motion.div>
  );
}

/** Explicación plegable para la defensa: qué se hace, por qué y cómo leer el resultado. */
export function Explain({ title = "¿Qué significa?", children, open = false }: { title?: string; children: ReactNode; open?: boolean }) {
  return (
    <details className="explain" open={open}>
      <summary>
        <Icon name="info" /> {title}
      </summary>
      {children}
    </details>
  );
}

export function Stat({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="card stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

const statusMap: Record<string, { icon: IconName; label: string; color: string }> = {
  ok: { icon: "ok", label: "Correcto", color: "var(--good)" },
  advertencia: { icon: "warn", label: "Advertencia", color: "var(--serious)" },
  error: { icon: "error", label: "Error", color: "var(--critical)" },
  información: { icon: "info", label: "Información", color: "var(--accent)" },
};

/** Estado con icono y etiqueta (nunca sólo color). */
export function StatusTag({ status, label }: { status: string; label?: string }) {
  const s = statusMap[status] ?? statusMap["información"];
  return (
    <span className="status">
      <Icon name={s.icon} style={{ color: s.color }} />
      {label ?? s.label}
    </span>
  );
}

export interface Column<R> {
  key: string;
  header: ReactNode;
  value?: (r: R) => unknown;
  render?: (r: R) => ReactNode;
  num?: boolean;
  wrap?: boolean;
}

/** Tabla ordenable con descarga en CSV. */
export function DataTable<R extends Record<string, any>>({ rows, columns, name, maxHeight = 520, csv = true }: { rows: R[]; columns: Column<R>[]; name: string; maxHeight?: number; csv?: boolean }) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    const get = col?.value ?? ((r: R) => r[sort.key]);
    return [...rows].sort((a, b) => {
      const va = get(a) as any;
      const vb = get(b) as any;
      if (va === vb) return 0;
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      return (va > vb ? 1 : -1) * sort.dir;
    });
  }, [rows, columns, sort]);
  const exportCsv = () => {
    const plain = rows.map((r) => Object.fromEntries(columns.map((c) => [typeof c.header === "string" ? c.header : c.key, (c.value ?? ((x: R) => x[c.key]))(r)])));
    download(`${name}.csv`, toCsv(plain), "text/csv;charset=utf-8");
  };
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="table-wrap" style={{ maxHeight }}>
        <table className="data">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c.key}
                  className={c.num ? "num" : ""}
                  onClick={() => setSort((s) => ({ key: c.key, dir: s?.key === c.key && s.dir === 1 ? -1 : 1 }))}
                  aria-sort={sort?.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
                  scope="col"
                >
                  {c.header}
                  {sort?.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c.key} className={`${c.num ? "num" : ""} ${c.wrap ? "wrap" : ""}`}>
                    {c.render ? c.render(r) : String((c.value ?? ((x: R) => x[c.key]))(r) ?? "—")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {csv && (
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button type="button" className="btn btn-sm" onClick={exportCsv}>
            <Icon name="download" size={16} /> CSV
          </button>
        </div>
      )}
    </div>
  );
}

export function Loading({ what = "resultados" }: { what?: string }) {
  return <div className="plot-skeleton" style={{ minHeight: 200 }}>Cargando {what}…</div>;
}

export function ErrorBox({ error }: { error: unknown }) {
  return (
    <div className="error-box" role="alert">
      <strong>No se pudo cargar.</strong> <span className="small">{String(error)}</span>
    </div>
  );
}

/** Aísla los fallos de una sección: el resto de la interfaz sigue funcionando. */
export class SectionBoundary extends Component<{ children: ReactNode; name: string }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`Sección ${this.props.name}:`, error, info.componentStack);
  }
  componentDidUpdate(prev: { name: string }) {
    if (prev.name !== this.props.name && this.state.error) this.setState({ error: null });
  }
  render() {
    if (this.state.error) {
      return (
        <div className="error-box" role="alert">
          <strong>Esta sección no pudo mostrarse.</strong>
          <p className="small">{this.state.error.message}</p>
          <p className="small">Los resultados de la corrida no se ven afectados; el resto de secciones sigue disponible.</p>
        </div>
      );
    }
    return this.props.children;
  }
}
