// Origen de los datos: el mismo código sirve al modo en vivo (FastAPI) y al estático (Pages).
export type Mode = "live" | "static";

export interface RunSummary {
  id: string;
  [key: string]: unknown;
}

export interface RunIndex {
  latest: string | null;
  runs: RunSummary[];
}

export interface Dataset {
  columns: string[];
  data: Record<string, (number | string | null)[]>;
}

export interface JobEvent {
  stage: string;
  status: string;
  message: string;
  progress: number;
  t?: number;
  detail?: string;
}

export interface JobState {
  id: string;
  status: string;
  run_id: string | null;
  error: string | null;
  progress: number;
}

export interface Source {
  mode: Mode;
  runs(): Promise<RunIndex>;
  results(id: string): Promise<any>;
  dataset(id: string): Promise<Dataset>;
  runConfig(id: string): Promise<any>;
  fileUrl(id: string, name: string): string;
  defaultConfig(): Promise<any>;
  schema(): Promise<any>;
  validate(config: unknown): Promise<{ ok: boolean; fingerprint?: string; detail?: unknown }>;
  submit(config: unknown): Promise<JobState>;
  follow(jobId: string, onEvent: (e: JobEvent) => void, onEnd: (s: JobState) => void): () => void;
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`);
  return (await r.json()) as T;
}

const resultsCache = new Map<string, Promise<any>>();
const datasetCache = new Map<string, Promise<Dataset>>();

function cached<T>(map: Map<string, Promise<T>>, key: string, load: () => Promise<T>): Promise<T> {
  let p = map.get(key);
  if (!p) {
    p = load();
    p.catch(() => map.delete(key));
    map.set(key, p);
  }
  return p;
}

export function liveSource(base = "api"): Source {
  return {
    mode: "live",
    runs: () => getJson(`${base}/runs`),
    results: (id) => cached(resultsCache, `L${id}`, () => getJson(`${base}/runs/${id}/results`)),
    dataset: (id) => cached(datasetCache, `L${id}`, () => getJson(`${base}/runs/${id}/dataset`)),
    runConfig: (id) => getJson(`${base}/runs/${id}/config`),
    fileUrl: (id, name) => `${base}/runs/${id}/files/${name}`,
    defaultConfig: () => getJson(`${base}/config/default`),
    schema: () => getJson(`${base}/config/schema`),
    async validate(config) {
      const r = await fetch(`${base}/config/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const body = await r.json();
      return r.ok ? body : { ok: false, detail: body.detail };
    },
    async submit(config) {
      const r = await fetch(`${base}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
      return body;
    },
    follow(jobId, onEvent, onEnd) {
      const es = new EventSource(`${base}/jobs/${jobId}/events`);
      es.onmessage = (m) => onEvent(JSON.parse(m.data));
      es.addEventListener("end", (m) => {
        onEnd(JSON.parse((m as MessageEvent).data));
        es.close();
      });
      es.onerror = () => {
        // El navegador reintenta solo; si el trabajo terminó, el evento «end» ya llegó.
      };
      return () => es.close();
    },
  };
}

export function staticSource(base = "data"): Source {
  const unsupported = () => Promise.reject(new Error("Disponible sólo en el modo en vivo."));
  return {
    mode: "static",
    runs: () => getJson(`${base}/runs.json`),
    results: (id) => cached(resultsCache, `S${id}`, () => getJson(`${base}/runs/${id}/results.json`)),
    dataset: (id) => cached(datasetCache, `S${id}`, () => getJson(`${base}/runs/${id}/dataset.json`)),
    runConfig: (id) => getJson(`${base}/runs/${id}/config.json`),
    fileUrl: (id, name) => `${base}/runs/${id}/${name}`,
    defaultConfig: () => getJson(`${base}/config-default.json`),
    schema: () => getJson(`${base}/config-schema.json`),
    validate: () => Promise.resolve({ ok: false, detail: "Disponible sólo en el modo en vivo." }),
    submit: unsupported,
    follow: () => () => undefined,
  };
}

/** Modo en vivo si responde la API; si no, los ficheros empaquetados con el sitio. */
export async function detectSource(): Promise<Source> {
  const forced = import.meta.env.VITE_CS_MODE as string | undefined;
  if (forced === "static") return staticSource();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 2500);
    const r = await fetch("api/health", { signal: ctrl.signal });
    clearTimeout(timer);
    if (r.ok && (await r.json()).mode === "live") return liveSource();
  } catch {
    // sin servidor: modo estático
  }
  return staticSource();
}
