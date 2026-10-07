# Datos

## `raw/CANCER.csv` — datos originales

Conjunto *Cancer Death Rates* publicado en Kaggle
(<https://www.kaggle.com/datasets/gurtegsawhney/cancer-death-rates-data>), derivado del
reto «OLS Regression Challenge» de data.world. Cada fila es un condado (o entidad
equivalente) de EE. UU.: **3047 condados de los 50 estados y el Distrito de Columbia**.

| Propiedad | Valor |
|---|---|
| Huella SHA-256 | `35061442acc5acc028db0d3ec27d299d2a29c452b16f305253ea20ba98bdf682` |
| Tamaño | 728 286 bytes |
| Codificación | Mac Roman (el byte `0x96` es la «ñ» de *Doña Ana County*) |
| Finales de línea | CR (`\r`), formato del Mac OS clásico |
| Filas × columnas | 3047 × 36 (incluida una columna vacía sin nombre) |

El fichero se versiona **byte a byte** (ver `.gitattributes`): el pipeline comprueba su
huella en cada corrida y la registra en el manifiesto, de modo que cualquier alteración
queda a la vista.

Fuentes de las variables, según la descripción original:

- **(a)** registros de cáncer, 2010–2016 (cancer.gov, *State Cancer Profiles*);
- **(b)** estimaciones censales de 2013 (census.gov, *American Community Survey*).

## `raw/descripcion_del_dataset.txt`

Diccionario de datos tal como lo publica Kaggle. El diccionario comentado en castellano,
con unidades, constructos y papel de cada variable, está en
[`src/cancerstats/dictionary.py`](../src/cancerstats/dictionary.py).

## Incidencias conocidas

El fichero contiene problemas deliberados o heredados que el pipeline detecta con reglas
de validación (`src/cancerstats/validation.py`) y trata con decisiones documentadas
(`src/cancerstats/cleaning.py`): valores centinela en la incidencia (Kansas, Minnesota y
Nevada), edades medianas en meses, tamaños de hogar divididos por 100, una variable
educativa con un 75 % de ausentes recuperable por identidad contable y una categoría
racial residual que el diccionario no menciona. El informe y la interfaz explican cada una.
