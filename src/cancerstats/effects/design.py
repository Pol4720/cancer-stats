"""Construcción explícita de la matriz de diseño.

Se construye a mano (en lugar de con fórmulas) para controlar con exactitud el centrado de
las variables, la codificación de la región con su categoría de referencia y los nombres de
los términos. Un *término* agrupa las columnas que se contrastan en bloque: las tres
indicadoras de la región, o las tres columnas de una interacción variable × región.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cancerstats.dictionary import REGIONS

REGION = "region"


def region_levels(reference: str) -> list[str]:
    """Niveles de la región distintos de la referencia, en orden fijo."""
    return [r for r in REGIONS if r != reference]


def dummy_name(level: str) -> str:
    """Nombre de la indicadora de un nivel de región."""
    return f"region[{level}]"


def term_name(pair: tuple[str, str] | list[str]) -> str:
    """Nombre canónico de una interacción."""
    a, b = pair
    if a == REGION:
        a, b = b, a
    return f"{a}:{b}"


@dataclass
class Design:
    """Matriz de diseño con metadatos de términos."""

    X: pd.DataFrame
    y: pd.Series
    terms: dict[str, list[str]]
    centers: dict[str, float]
    groups: np.ndarray
    group_labels: list[str]
    population: np.ndarray
    county: list[str]
    region: np.ndarray
    reference: str
    numeric: list[str]
    interactions: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Número de observaciones."""
        return len(self.y)

    def columns_for(self, terms: list[str]) -> list[str]:
        """Columnas de un conjunto de términos (más la constante)."""
        cols = ["const"]
        for t in terms:
            cols.extend(self.terms[t])
        return cols


def build_design(
    df: pd.DataFrame,
    response: str,
    numeric: list[str],
    use_region: bool,
    reference: str,
    interactions: list[tuple[str, str]] | list[list[str]],
    center: bool,
) -> Design:
    """Construye la matriz de diseño sobre los casos completos de las variables implicadas.

    Args:
        df: datos depurados (con ``region``, ``state``, ``popEst2015``, ``county_id``).
        response: columna respuesta.
        numeric: explicativas continuas.
        use_region: si se incluye la región (3 indicadoras).
        reference: región de referencia.
        interactions: pares ``(a, b)``; ``b`` puede ser ``"region"``.
        center: si se centran las continuas en su media muestral.
    """
    data = df.dropna(subset=[response, *numeric]).copy()
    centers = {c: float(data[c].mean()) if center else 0.0 for c in numeric}
    cols: dict[str, np.ndarray] = {"const": np.ones(len(data))}
    terms: dict[str, list[str]] = {}
    for c in numeric:
        cols[c] = data[c].to_numpy(dtype=float) - centers[c]
        terms[c] = [c]
    levels = region_levels(reference)
    if use_region:
        names = []
        for lev in levels:
            nm = dummy_name(lev)
            cols[nm] = (data[REGION] == lev).to_numpy(dtype=float)
            names.append(nm)
        terms[REGION] = names
    inter_names: list[str] = []
    for pair in interactions:
        a, b = pair if pair[0] != REGION else (pair[1], pair[0])
        name = term_name((a, b))
        if a not in cols or (b != REGION and b not in cols) or (b == REGION and not use_region):
            continue
        if b == REGION:
            names = []
            for lev in levels:
                nm = f"{a}:{dummy_name(lev)}"
                cols[nm] = cols[a] * cols[dummy_name(lev)]
                names.append(nm)
            terms[name] = names
        else:
            cols[name] = cols[a] * cols[b]
            terms[name] = [name]
        inter_names.append(name)
    X = pd.DataFrame(cols, index=data.index)
    codes, uniques = pd.factorize(data["state"])
    return Design(
        X=X,
        y=data[response].astype(float),
        terms=terms,
        centers=centers,
        groups=np.asarray(codes),
        group_labels=[str(u) for u in uniques],
        population=data["popEst2015"].to_numpy(dtype=float),
        county=data["county_id"].astype(str).tolist(),
        region=data[REGION].astype(str).to_numpy(),
        reference=reference,
        numeric=list(numeric),
        interactions=inter_names,
    )


def subset(design: Design, mask: np.ndarray) -> Design:
    """Subconjunto de filas de un diseño (mismas columnas y centrado)."""
    idx = np.flatnonzero(mask)
    return Design(
        X=design.X.iloc[idx],
        y=design.y.iloc[idx],
        terms=design.terms,
        centers=design.centers,
        groups=design.groups[idx],
        group_labels=design.group_labels,
        population=design.population[idx],
        county=[design.county[i] for i in idx],
        region=design.region[idx],
        reference=design.reference,
        numeric=design.numeric,
        interactions=design.interactions,
    )
