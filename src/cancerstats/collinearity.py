"""Diagnóstico y tratamiento de la colinealidad.

Se aplican las tres herramientas del curso: correlaciones simples entre explicativas,
factor de inflación de la varianza (FIV) con su tolerancia, y el análisis de componentes
principales de Belsley, Kuh y Welsch (índices de condición y proporciones de
descomposición de la varianza). La poda elimina iterativamente la variable de mayor FIV
—salvo las protegidas— hasta que todas quedan por debajo del umbral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cancerstats.dictionary import label

Record = dict[str, object]


def vif_table(x: pd.DataFrame) -> pd.DataFrame:
    """FIV_j = 1 / (1 − R²_j), con R²_j de la regresión de X_j sobre las demás.

    Se obtiene de la diagonal de la inversa de la matriz de correlaciones, que es
    equivalente y numéricamente más estable que ajustar p regresiones.
    """
    corr = np.corrcoef(x.to_numpy(dtype=float), rowvar=False)
    inv = np.linalg.pinv(corr)
    vif = np.diag(inv)
    return pd.DataFrame(
        {"variable": x.columns, "fiv": vif, "tolerancia": 1 / vif, "r2_aux": 1 - 1 / vif}
    )


def high_correlations(x: pd.DataFrame, threshold: float) -> list[Record]:
    """Pares de explicativas con |r| por encima del umbral."""
    corr = x.corr()
    cols = list(x.columns)
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = float(corr.loc[a, b])
            if abs(r) >= threshold:
                out.append({"a": a, "b": b, "etiqueta_a": label(a), "etiqueta_b": label(b), "r": r})
    return sorted(out, key=lambda d: -abs(float(d["r"])))  # type: ignore[arg-type]


def belsley(x: pd.DataFrame, threshold: float) -> Record:
    """Índices de condición y proporciones de descomposición de la varianza (BKW, 1980).

    Las columnas de [1, X] se escalan a longitud unidad (sin centrar, para incluir la
    constante). Para cada componente con índice de condición alto, las variables con
    proporción > 0,5 son las implicadas en esa dependencia casi lineal.
    """
    mat = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    mat = mat / np.linalg.norm(mat, axis=0)
    _, s, vt = np.linalg.svd(mat, full_matrices=False)
    ci = s.max() / s
    phi = (vt.T**2) / s**2
    props = phi / phi.sum(axis=1, keepdims=True)
    names = ["(Constante)", *x.columns]
    components = []
    for k in np.argsort(-ci):
        involved = [names[j] for j in range(len(names)) if props[j, k] > 0.5]
        components.append(
            {
                "indice_condicion": float(ci[k]),
                "autovalor": float(s[k] ** 2),
                "proporciones": {names[j]: float(props[j, k]) for j in range(len(names))},
                "implicadas": involved,
                "problematica": bool(ci[k] > threshold and len(involved) >= 2),
            }
        )
    return {"numero_condicion": float(ci.max()), "componentes": components}


def prune(
    x: pd.DataFrame, threshold: float, protected: list[str]
) -> tuple[list[str], list[Record]]:
    """Poda iterativa por FIV respetando las variables protegidas.

    Returns:
        Variables retenidas y la traza de la poda (qué se eliminó, con qué FIV y con qué
        variable estaba más correlacionada).
    """
    keep = list(x.columns)
    trace: list[Record] = []
    while True:
        table = vif_table(x[keep]).set_index("variable")
        free = table.drop(index=[p for p in protected if p in table.index])
        if free.empty or free["fiv"].max() <= threshold:
            break
        worst = str(free["fiv"].idxmax())
        corr = x[keep].corr()[worst].drop(worst)
        partner = str(corr.abs().idxmax())
        trace.append(
            {
                "paso": len(trace) + 1,
                "eliminada": worst,
                "etiqueta": label(worst),
                "fiv": float(free.loc[worst, "fiv"]),
                "mas_correlada_con": partner,
                "r": float(corr[partner]),
            }
        )
        keep.remove(worst)
    return keep, trace


def analyze(
    x: pd.DataFrame,
    corr_threshold: float,
    vif_threshold: float,
    ci_threshold: float,
    protected: list[str],
) -> Record:
    """Etapa completa de colinealidad sobre el conjunto de explicativas candidatas."""
    before = vif_table(x)
    kept, trace = prune(x, vif_threshold, protected)
    after = vif_table(x[kept])
    return {
        "n": len(x),
        "pairs": high_correlations(x, corr_threshold),
        "vif_before": before.to_dict(orient="records"),
        "belsley_before": belsley(x, ci_threshold),
        "trace": trace,
        "kept": kept,
        "vif_after": after.to_dict(orient="records"),
        "belsley_after": belsley(x[kept], ci_threshold),
    }
