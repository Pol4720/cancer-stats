import { AnimatePresence, motion } from "motion/react";
import { Suspense, useEffect, useState } from "react";
import { go, useRoute } from "../lib/router";
import { useApp } from "../lib/store";
import { useTheme, type ThemeChoice } from "../lib/theme";
import { SECTIONS } from "../sections";
import { Icon } from "./icons";
import { Loading, SectionBoundary } from "./ui";

function ThemeSwitch() {
  const { choice, setChoice } = useTheme();
  const opts: { v: ThemeChoice; icon: "sun" | "moon" | "monitor"; label: string }[] = [
    { v: "light", icon: "sun", label: "Claro" },
    { v: "dark", icon: "moon", label: "Oscuro" },
    { v: "system", icon: "monitor", label: "Sistema" },
  ];
  return (
    <div className="segmented" role="group" aria-label="Tema">
      {opts.map((o) => (
        <button key={o.v} type="button" aria-pressed={choice === o.v} onClick={() => setChoice(o.v)} title={o.label} aria-label={`Tema ${o.label.toLowerCase()}`}>
          <Icon name={o.icon} size={16} />
        </button>
      ))}
    </div>
  );
}

function Guide({ onClose }: { onClose: () => void }) {
  const route = useRoute();
  const i = Math.max(0, SECTIONS.findIndex((s) => s.id === route));
  const s = SECTIONS[i];
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest("input, textarea, select")) return;
      if (e.key === "ArrowRight" && i < SECTIONS.length - 1) go(SECTIONS[i + 1].id);
      if (e.key === "ArrowLeft" && i > 0) go(SECTIONS[i - 1].id);
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [i, onClose]);
  return (
    <motion.aside className="guide" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} aria-label="Modo guía">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="steps">Paso {i + 1} de {SECTIONS.length}</span>
        <button className="btn ghost btn-sm" onClick={onClose} aria-label="Cerrar la guía"><Icon name="close" size={16} /></button>
      </div>
      <AnimatePresence mode="wait">
        <motion.div key={s.id} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: 0.2 }}>
          <h3 style={{ marginTop: 6 }}>{s.title}</h3>
          <p className="small">{s.guide}</p>
        </motion.div>
      </AnimatePresence>
      <div className="progress" style={{ margin: "8px 0 12px" }}><div style={{ width: `${((i + 1) / SECTIONS.length) * 100}%` }} /></div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <button className="btn btn-sm" disabled={i === 0} onClick={() => go(SECTIONS[i - 1].id)}><Icon name="prev" size={16} /> Anterior</button>
        <span className="muted small"><span className="kbd">←</span> <span className="kbd">→</span></span>
        <button className="btn btn-sm primary" disabled={i === SECTIONS.length - 1} onClick={() => go(SECTIONS[i + 1].id)}>Siguiente <Icon name="next" size={16} /></button>
      </div>
    </motion.aside>
  );
}

export function Shell() {
  const route = useRoute();
  const { source, runId, index } = useApp();
  const [menu, setMenu] = useState(false);
  const [guide, setGuide] = useState(false);
  const current = SECTIONS.find((s) => s.id === route) ?? SECTIONS[0];
  const Comp = current.component;
  useEffect(() => setMenu(false), [route]);
  useEffect(() => {
    document.title = `${current.title} · Mortalidad por cáncer`;
  }, [current]);
  let lastGroup = "";

  return (
    <div className="app">
      {menu && <div className="scrim" onClick={() => setMenu(false)} />}
      <nav className={`sidebar ${menu ? "open" : ""}`} aria-label="Secciones">
        <div className="brand">
          <div className="brand-mark"><Icon name="chart" /></div>
          <div>
            <div className="brand-title">Mortalidad por cáncer</div>
            <div className="brand-sub">Condados de EE. UU. · regresión múltiple</div>
          </div>
        </div>
        {SECTIONS.map((s, i) => {
          const header = s.group !== lastGroup ? <div className="nav-group" key={`g-${s.group}`}>{s.group}</div> : null;
          lastGroup = s.group;
          return [
            header,
            <a key={s.id} href={`#/${s.id}`} className="nav-link" aria-current={s.id === current.id ? "page" : undefined}>
              <span className="nav-num">{i}</span>
              {s.title}
            </a>,
          ];
        })}
        <div className="sidebar-foot">
          <span className={`badge ${source.mode}`}>
            <span className="dot" /> {source.mode === "live" ? "Modo en vivo" : "Versión publicada"}
          </span>
          <span className="muted small">
            Corrida <code>{runId}</code>
            {index.latest === runId ? " (última)" : ""}
          </span>
          <span className="muted small">Lic. Richard Alejandro Matos Arderí · Lic. Abel Ponce González · Lic. Lidier Robayna</span>
        </div>
      </nav>
      <div className="main">
        <header className="topbar">
          <button className="btn ghost menu-btn" onClick={() => setMenu(true)} aria-label="Abrir el menú"><Icon name="menu" /></button>
          <strong className="small title">{current.title}</strong>
          <span className="spacer" />
          <button className={`btn btn-sm ${guide ? "primary" : ""}`} onClick={() => setGuide((g) => !g)} aria-pressed={guide}>
            <Icon name="compass" size={16} /> <span className="btn-label">Modo guía</span>
          </button>
          <ThemeSwitch />
        </header>
        <main className="content" id="contenido">
          <Suspense fallback={<Loading what="la sección" />}>
            <AnimatePresence mode="wait">
              <motion.div key={`${current.id}-${runId}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
                <SectionBoundary name={current.id}>
                  <Comp />
                </SectionBoundary>
              </motion.div>
            </AnimatePresence>
          </Suspense>
        </main>
      </div>
      <AnimatePresence>{guide && <Guide onClose={() => setGuide(false)} />}</AnimatePresence>
    </div>
  );
}
