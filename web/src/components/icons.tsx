// Iconos de trazo (24 × 24) dibujados a mano para no depender de una librería.
import type { SVGProps } from "react";

const paths: Record<string, string> = {
  home: "M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
  database: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  check: "M5 12.5l4.5 4.5L19 7.5",
  shield: "M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z M8.5 12l2.5 2.5 4.5-5",
  hole: "M4 4h16v16H4z M8 8h3v3H8z M13 13h3v3h-3z",
  chart: "M4 20V4 M4 20h16 M8 16l4-5 3 3 5-7",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z M12 12h.01",
  layers: "M12 3l9 5-9 5-9-5z M3 13l9 5 9-5",
  formula: "M5 5h14 M7 5l6 7-6 7h12 M16 12h3",
  stethoscope: "M6 3v6a4 4 0 0 0 8 0V3 M10 13v3a5 5 0 0 0 10 0v-2 M20 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
  scale: "M12 4v16 M5 20h14 M4 8h16 M6 8l-3 6a3 3 0 0 0 6 0z M18 8l-3 6a3 3 0 0 0 6 0z",
  crystal: "M12 3l7 6-7 12-7-12z M5 9h14",
  history: "M3 12a9 9 0 1 0 3-6.7L3 8 M3 3v5h5 M12 7v5l3 2",
  play: "M7 5l12 7-12 7z",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z M12 1v2 M12 21v2 M4.2 4.2l1.4 1.4 M18.4 18.4l1.4 1.4 M1 12h2 M21 12h2 M4.2 19.8l1.4-1.4 M18.4 5.6l1.4-1.4",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  monitor: "M3 4h18v12H3z M8 20h8 M12 16v4",
  menu: "M4 6h16 M4 12h16 M4 18h16",
  close: "M6 6l12 12 M18 6L6 18",
  download: "M12 4v12 M7 11l5 5 5-5 M5 20h14",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 11v5 M12 8h.01",
  warn: "M12 3l10 18H2z M12 10v4 M12 17h.01",
  error: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M9 9l6 6 M15 9l-6 6",
  ok: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M8 12.5l2.7 2.7L16 10",
  compass: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M15.5 8.5l-2 5-5 2 2-5z",
  next: "M9 6l6 6-6 6",
  prev: "M15 6l-6 6 6 6",
  file: "M6 3h8l4 4v14H6z M14 3v4h4",
  outlier: "M5 19l4-6 4 3 6-9 M19 4h.01 M4 6h.01",
};

export type IconName = keyof typeof paths;

export function Icon({ name, size = 18, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...rest}>
      <path d={paths[name]} />
    </svg>
  );
}
