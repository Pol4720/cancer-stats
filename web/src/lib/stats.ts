// Pequeñas utilidades numéricas para los gráficos interactivos (no sustituyen al pipeline).

export function finitePairs(x: (number | string | null)[], y: (number | string | null)[]): [number[], number[], number[]] {
  const xs: number[] = [];
  const ys: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i < x.length; i++) {
    const a = x[i];
    const b = y[i];
    if (typeof a === "number" && typeof b === "number" && Number.isFinite(a) && Number.isFinite(b)) {
      xs.push(a);
      ys.push(b);
      idx.push(i);
    }
  }
  return [xs, ys, idx];
}

/** Recta MCO simple y correlación de Pearson. */
export function simpleOls(x: number[], y: number[]): { a: number; b: number; r: number } {
  const n = x.length;
  const mx = x.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) ** 2;
    syy += (y[i] - my) ** 2;
  }
  const b = sxy / sxx;
  return { a: my - b * mx, b, r: sxy / Math.sqrt(sxx * syy) };
}

/** Suavizado LOWESS (vecinos más cercanos, pesos tricúbicos, ajuste lineal local). */
export function lowess(x: number[], y: number[], frac = 0.4, points = 60): { x: number[]; y: number[] } {
  const order = x.map((_, i) => i).sort((i, j) => x[i] - x[j]);
  const xs = order.map((i) => x[i]);
  const ys = order.map((i) => y[i]);
  const n = xs.length;
  const k = Math.max(5, Math.floor(frac * n));
  const lo = xs[0];
  const hi = xs[n - 1];
  const gx: number[] = [];
  const gy: number[] = [];
  for (let p = 0; p < points; p++) {
    const x0 = lo + ((hi - lo) * p) / (points - 1);
    const d = xs.map((v) => Math.abs(v - x0));
    const h = [...d].sort((a, b) => a - b)[k - 1] || 1e-9;
    let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
    for (let i = 0; i < n; i++) {
      const u = d[i] / h;
      if (u >= 1) continue;
      const w = (1 - u ** 3) ** 3;
      sw += w;
      swx += w * xs[i];
      swy += w * ys[i];
      swxx += w * xs[i] * xs[i];
      swxy += w * xs[i] * ys[i];
    }
    const den = sw * swxx - swx * swx;
    const b = Math.abs(den) > 1e-12 ? (sw * swxy - swx * swy) / den : 0;
    const a = (swy - b * swx) / sw;
    gx.push(x0);
    gy.push(a + b * x0);
  }
  return { x: gx, y: gy };
}

export function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
