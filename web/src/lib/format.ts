// Formato numérico en castellano: coma decimal y espacio fino como separador de miles.
const cache = new Map<number, Intl.NumberFormat>();

function nf(digits: number): Intl.NumberFormat {
  let f = cache.get(digits);
  if (!f) {
    f = new Intl.NumberFormat("es-ES", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      // Norma RAE y siunitx: sin separador en números de cuatro cifras.
      useGrouping: "min2" as unknown as boolean,
    });
    cache.set(digits, f);
  }
  return f;
}

/** Número con `digits` decimales; «—» si no es finito. */
export function num(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const s = nf(digits).format(x).replace(/[\u00a0.]/g, "\u202f");
  return s === `-${nf(digits).format(0)}` ? nf(digits).format(0) : s;
}

/** Entero con separador de miles. */
export function int(x: number | null | undefined): string {
  return num(x, 0);
}

/** Porcentaje (el valor ya está en %). */
export function pct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "—" : `${num(x, digits)} %`;
}

/** p-valor legible: nunca «0,000». */
export function pval(p: number | null | undefined, digits = 3): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "—";
  const t = 10 ** -digits;
  return p < t ? `< ${num(t, digits)}` : num(p, digits);
}

/** Intervalo [a; b]. */
export function ci(lo: number | null | undefined, hi: number | null | undefined, digits = 2): string {
  return `[${num(lo, digits)}; ${num(hi, digits)}]`;
}

/** Número con signo explícito. */
export function signed(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return (x > 0 ? "+" : "") + num(x, digits);
}

export function isSig(p: number | null | undefined, alpha = 0.05): boolean {
  return p !== null && p !== undefined && p < alpha;
}

/** Nombre legible de un término del modelo usando el diccionario de la corrida. */
export function termLabel(term: string, labels: Record<string, string>): string {
  if (term === "const") return "Constante";
  if (term.startsWith("region[")) return `Región: ${term.slice(7, -1)}`;
  if (term.includes(":")) return term.split(":").map((t) => termLabel(t, labels)).join(" × ");
  if (term === "region") return "Región censal";
  return labels[term] ?? term;
}

/** CSV (separador «;» y coma decimal, como espera una hoja de cálculo en castellano). */
export function toCsv(rows: Record<string, unknown>[], columns?: string[]): string {
  if (!rows.length) return "";
  const cols = columns ?? Object.keys(rows[0]);
  const cell = (v: unknown) => {
    if (v === null || v === undefined) return "";
    if (typeof v === "number") return String(v).replace(".", ",");
    const s = typeof v === "string" ? v : JSON.stringify(v);
    return /[;"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(";"), ...rows.map((r) => cols.map((c) => cell(r[c])).join(";"))].join("\n");
}

export function download(name: string, content: string | Blob, type = "text/plain;charset=utf-8"): void {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
