import { useCallback, useEffect, useState } from "react";
import { detectSource, type RunIndex, type Source } from "./api/source";
import { Shell } from "./components/Shell";
import { ErrorBox, Loading } from "./components/ui";
import { AppProvider, labelsFrom } from "./lib/store";
import { ThemeProvider } from "./lib/theme";

function readRunParam(): string | null {
  return new URLSearchParams(window.location.search).get("corrida");
}

export default function App() {
  const [source, setSource] = useState<Source | null>(null);
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [runId, setRunIdState] = useState<string | null>(null);
  const [res, setRes] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    detectSource()
      .then(async (s) => {
        setSource(s);
        const idx = await s.runs();
        setIndex(idx);
        const wanted = readRunParam();
        setRunIdState(wanted && idx.runs.some((r) => r.id === wanted) ? wanted : idx.latest);
      })
      .catch(setError);
  }, []);

  useEffect(() => {
    if (!source || !runId) return;
    let alive = true;
    // Se mantiene la corrida anterior en pantalla hasta que llega la nueva: así no se
    // desmonta la interfaz (ni se pierde el progreso de una corrida recién terminada).
    source
      .results(runId)
      .then((r) => alive && setRes(r))
      .catch((e) => alive && setError(e));
    return () => {
      alive = false;
    };
  }, [source, runId]);

  const setRunId = useCallback((id: string) => {
    setRunIdState(id);
    const url = new URL(window.location.href);
    url.searchParams.set("corrida", id);
    window.history.replaceState(null, "", url);
  }, []);

  const refreshRuns = useCallback(async () => {
    if (source) setIndex(await source.runs());
  }, [source]);

  return (
    <ThemeProvider>
      {error ? (
        <div className="content"><ErrorBox error={error} /></div>
      ) : !source || !index || !runId || !res ? (
        <div className="content"><Loading /></div>
      ) : (
        <AppProvider value={{ source, index, runId, setRunId, res, labels: labelsFrom(res), refreshRuns }}>
          <Shell />
        </AppProvider>
      )}
    </ThemeProvider>
  );
}
