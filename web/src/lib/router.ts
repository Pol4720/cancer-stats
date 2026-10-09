import { useEffect, useState } from "react";

/** Enrutado por fragmento (#/seccion): funciona en cualquier subruta de GitHub Pages. */
export function currentRoute(): string {
  const h = window.location.hash.replace(/^#\/?/, "");
  return h.split("?")[0] || "inicio";
}

export function go(route: string): void {
  if (currentRoute() !== route) window.location.hash = `#/${route}`;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

export function useRoute(): string {
  const [route, setRoute] = useState(currentRoute);
  useEffect(() => {
    const on = () => setRoute(currentRoute());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
