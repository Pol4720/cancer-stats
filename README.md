# Mortalidad por cáncer en los condados de EE. UU.

[![CI](https://github.com/Pol4720/cancer-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/Pol4720/cancer-stats/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/Pol4720/cancer-stats/actions/workflows/pages.yml/badge.svg)](https://github.com/Pol4720/cancer-stats/actions/workflows/pages.yml)

Proyecto final del Curso Propedéutico de Estadística (Universidad Complutense de Madrid, prof.
Juan Tejada): *Multiple Linear Regression Challenge*. Un modelo de regresión lineal múltiple para
la tasa de mortalidad por cáncer (`TARGET_deathRate`) de 3047 condados, que va desde los datos en
bruto (`practica.sav`) hasta la inferencia robusta. Todo es reproducible.

**Autores:** Lic. Richard Alejandro Matos Arderí, Lic. Abel Ponce González y Lic. Lidier Robayna.

| | |
|---|---|
| Interfaz interactiva | <https://pol4720.github.io/cancer-stats/> |
| Presentación | <https://pol4720.github.io/cancer-stats/presentacion/> |
| Informe (PDF) | [`report/dist/informe.pdf`](report/dist/informe.pdf) · [en Pages](https://pol4720.github.io/cancer-stats/informe.pdf) |
| Sintaxis SPSS del modelo final | [`spss/modelo_final.sps`](spss/modelo_final.sps) |

## Resultado

El modelo final se estima por mínimos cuadrados ordinarios (MCO) con 2840 condados y 19
regresores: 10 términos elegidos por eliminación hacia atrás, 1 confusora reincorporada y 2
interacciones con la región censal.

- Explica el 55,8 % de la variabilidad (R² = 0,558; R² ajustado = 0,555).
- RMSE = 18,44 muertes por 100 000 habitantes.
- La inferencia usa errores típicos robustos por conglomerados de estado (48 estados), porque la
  varianza depende del tamaño del condado y los condados de un mismo estado se parecen.

Las conclusiones se mantienen bajo 10 especificaciones de sensibilidad y un bootstrap por
conglomerados. Como modelo predictivo se eligió ridge, con un RMSE de 17,36 en la partición de
prueba.

## Entregables de la orientación

| | Entregable | Dónde |
|---|---|---|
| (a) | Modelo final y pasos seguidos | Informe, capítulos 5 y 6 · interfaz, «Construcción» y «Modelo final» |
| (b) | Salida del software con R², R² ajustado y RMSE | Informe, apartado 6.2 (tablas tipo SPSS) · `spss/modelo_final.sps` |
| (c) | Código | Este repositorio (`src/cancerstats`) · informe, apéndice A |
| (d) | Diagnóstico: linealidad, independencia, homocedasticidad, normalidad, multicolinealidad | Informe, capítulo 7 · interfaz, «Diagnóstico» |
| (e) | Interpretación | Informe, apartado 6.3 · interfaz, «Interpretación» |
| (f) | Atípicos, valores perdidos y variables categóricas | Informe, capítulos 2 y 3 |

## Uso

Requisitos: [uv](https://docs.astral.sh/uv/) (Python 3.12). Para la interfaz hace falta además
Node 22, y para el informe TeX Live con LuaLaTeX y biber.

```sh
uv sync                        # entorno exacto (uv.lock)
uv run cancerstats run         # ejecuta el pipeline y guarda la corrida en runs/
uv run cancerstats serve       # interfaz en vivo en http://127.0.0.1:8000
```

`cancerstats run` lee `config/default.yaml`. Cada opción metodológica está validada y
documentada, y se puede cambiar sin tocar el código:

```sh
uv run cancerstats run -c mi-configuracion.yaml
uv run cancerstats run -s effects.covariance=HC3
uv run cancerstats config schema   # todas las opciones con su descripción
```

Otras órdenes:

| Orden | Qué hace |
|---|---|
| `cancerstats runs list` / `runs show <id>` | Corridas guardadas y su manifiesto |
| `cancerstats export [--run <id>]` | Regenera `report/generado/` (cifras, tablas, figuras) y la sintaxis SPSS |
| `cancerstats presentation [--run <id>]` | Regenera `presentation/index.html` |
| `cancerstats web-data [--run <id>]` | Prepara los datos de la interfaz en modo estático |

### Informe

```sh
cd report && latexmk informe.tex                      # PDF en report/dist/informe.pdf
python3 ../scripts/check_latex_log.py build/informe.log   # falla ante cualquier advertencia
```

El texto está en `report/capitulos/` y `report/apendices/`. Las cifras, tablas y figuras no se
escriben a mano: se leen de `report/generado/` mediante `\res{clave}`. Por eso el informe recoge
siempre la última corrida, y una cifra que falta aparece en rojo y hace fallar la compilación.

### Interfaz web

```sh
cd web && npm ci
npm run dev          # desarrollo; las llamadas a /api van a «cancerstats serve»
npm run build        # compilación estática (antes: uv run cancerstats web-data)
```

La interfaz funciona de dos modos:

- **En vivo:** con `cancerstats serve`, permite editar la configuración, lanzar corridas y seguir
  su progreso.
- **Estático:** en GitHub Pages, muestra la última corrida versionada.

En los dos modos ofrece tema claro y oscuro, un modo guía que explica cada sección y un diseño
adaptado al móvil.

## Reproducibilidad

- **Corridas:** cada corrida se guarda en `runs/<fecha>-<hora>-<huella>/`. Contiene la
  configuración, los resultados, el registro y un manifiesto con el SHA-256 de los datos, el
  commit y si el árbol estaba limpio. `runs/LATEST` apunta a la última.
- **Datos:** `practica.sav` (orientación oficial). Se comprobó que coincide celda a celda con el
  conjunto público de Kaggle; véase `data/README.md`.
- **Pruebas:** pruebas unitarias y de integración. Entre ellas, una reestimación independiente del
  modelo final con `statsmodels` y una prueba de que la sintaxis SPSS marca exactamente los 206
  centinelas.
- **CI:** estilo (ruff), tipos (mypy), pruebas, pipeline completo, interfaz (ESLint, Vitest,
  Playwright en escritorio y móvil) e informe LaTeX sin advertencias.

```sh
uv run ruff check && uv run mypy && uv run pytest
```

## Estructura

```
config/          configuración de la metodología (YAML)
data/            datos de la orientación y su documentación
src/cancerstats/ ingesta, validación, depuración, ausentes, atípicos, exploración,
                 modelo de efectos (effects/), modelo predictivo, exportación (export/), API
runs/            corridas persistidas
report/          informe LaTeX: capítulos, material generado y PDF final
presentation/    presentación HTML autónoma
spss/            sintaxis SPSS del modelo final
web/             interfaz React + Vite
tests/           pruebas unitarias y de integración
```

Referencia principal: Wasserman, L. (2004). *All of Statistics*. Springer.

Licencia MIT.
