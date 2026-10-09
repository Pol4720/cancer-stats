import { describe, expect, it } from "vitest";
import { ci, int, num, pval, termLabel, toCsv } from "../../src/lib/format";
import { lowess, simpleOls } from "../../src/lib/stats";
import { diff, toYaml } from "../../src/lib/yaml";

describe("formato en castellano", () => {
  it("usa coma decimal y espacio fino para los miles", () => {
    expect(num(12345.678, 2)).toBe("12\u202f345,68");
    expect(num(1234.567, 2)).toBe("1234,57");
    expect(int(3047)).toBe("3047");
    expect(num(-0.0001, 2)).toBe("0,00");
    expect(num(NaN)).toBe("—");
  });
  it("nunca imprime p = 0", () => {
    expect(pval(0)).toBe("< 0,001");
    expect(pval(0.0456)).toBe("0,046");
  });
  it("intervalos con punto y coma", () => {
    expect(ci(1, 2.5, 1)).toBe("[1,0; 2,5]");
  });
  it("etiqueta los términos del modelo", () => {
    const labels = { PctBachDeg25_Over: "Grado universitario (25+)" };
    expect(termLabel("const", labels)).toBe("Constante");
    expect(termLabel("region[Oeste]", labels)).toBe("Región: Oeste");
    expect(termLabel("PctBachDeg25_Over:region", labels)).toBe("Grado universitario (25+) × Región censal");
  });
  it("exporta CSV con punto y coma y coma decimal", () => {
    expect(toCsv([{ a: 1.5, b: "x;y" }])).toBe('a;b\n1,5;"x;y"');
  });
});

describe("estadística de apoyo", () => {
  it("recupera la recta MCO", () => {
    const x = [1, 2, 3, 4, 5];
    const y = x.map((v) => 2 + 3 * v);
    const f = simpleOls(x, y);
    expect(f.a).toBeCloseTo(2);
    expect(f.b).toBeCloseTo(3);
    expect(f.r).toBeCloseTo(1);
  });
  it("LOWESS reproduce una relación lineal", () => {
    const x = Array.from({ length: 100 }, (_, i) => i);
    const s = lowess(x, x.map((v) => 5 - 0.5 * v), 0.3, 10);
    s.x.forEach((v, i) => expect(s.y[i]).toBeCloseTo(5 - 0.5 * v, 6));
  });
});

describe("configuración", () => {
  it("diff devuelve sólo lo que cambia", () => {
    const base = { a: { x: 1, y: [1, 2] }, b: { z: true } };
    const edited = { a: { x: 2, y: [1, 2] }, b: { z: true } };
    expect(diff(base, edited)).toEqual({ a: { x: 2 } });
    expect(diff(base, base)).toBeUndefined();
  });
  it("YAML legible", () => {
    expect(toYaml({ effects: { alpha_remove: 0.1, exposures: ["a", "b"] } })).toBe("effects:\n  alpha_remove: 0.1\n  exposures: [a, b]");
    expect(toYaml({ e: { interactions: [["a", "region"]] } })).toBe("e:\n  interactions:\n    - [a, region]");
  });
});
