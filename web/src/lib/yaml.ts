// Serializador YAML mínimo (dicts, listas y escalares): basta para la configuración.
function scalar(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean" || typeof v === "number") return String(v);
  const s = String(v);
  return /^[\w.\-/ áéíóúñÁÉÍÓÚÑ]+$/.test(s) && !/^(true|false|null|yes|no|[\d.-]+)$/i.test(s) && s.trim() === s ? s : JSON.stringify(s);
}

export function toYaml(obj: unknown, indent = 0): string {
  const pad = " ".repeat(indent);
  if (Array.isArray(obj)) {
    if (!obj.length) return "[]";
    if (obj.every((x) => !x || typeof x !== "object")) return `[${obj.map(scalar).join(", ")}]`;
    return obj.map((x) => `\n${pad}- ${Array.isArray(x) ? `[${x.map(scalar).join(", ")}]` : toYaml(x, indent + 2).trimStart()}`).join("");
  }
  if (obj && typeof obj === "object") {
    return Object.entries(obj as Record<string, unknown>)
      .map(([k, v]) => {
        if (v && typeof v === "object" && !(Array.isArray(v) && v.every((x) => !x || typeof x !== "object"))) {
          const inner = toYaml(v, indent + 2);
          return `${pad}${k}:${inner.startsWith("\n") ? inner : `\n${inner}`}`;
        }
        return `${pad}${k}: ${toYaml(v, indent + 2)}`;
      })
      .join("\n");
  }
  return scalar(obj);
}

/** Diferencias entre dos configuraciones (sólo las claves que cambian). */
export function diff(base: any, edited: any): any {
  if (JSON.stringify(base) === JSON.stringify(edited)) return undefined;
  if (base && edited && typeof base === "object" && typeof edited === "object" && !Array.isArray(base) && !Array.isArray(edited)) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(edited)) {
      const d = diff(base[k], edited[k]);
      if (d !== undefined) out[k] = d;
    }
    return Object.keys(out).length ? out : undefined;
  }
  return edited;
}
