"""Procedencia: equivalencia entre el fichero oficial (SPSS) y el CSV público de Kaggle.

El profesor entregó ``practica.sav``; el conjunto público del que procede es el CSV de
Kaggle. Comprobar celda a celda que contienen los mismos datos documenta la procedencia y
permite reutilizar, con garantías, la descripción pública de las variables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cancerstats.io import EXPECTED_COLUMNS


def compare_sources(official: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Compara dos lecturas del conjunto (mismas filas en el mismo orden).

    Returns:
        Resumen con las columnas idénticas, las que difieren (y cómo) y las columnas que sólo
        existen en una de las fuentes.
    """
    same_rows = len(official) == len(reference)
    same_order = bool(
        same_rows and (official["county_id"].to_numpy() == reference["county_id"].to_numpy()).all()
    )
    columns: list[dict[str, Any]] = []
    for col in EXPECTED_COLUMNS:
        if col in ("Geography", "state"):
            continue
        a = official[col]
        b = reference[col]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            av = a.to_numpy(dtype=float)
            bv = b.to_numpy(dtype=float)
            both_na = np.isnan(av) & np.isnan(bv)
            equal = np.isclose(av, bv, rtol=1e-9, atol=1e-9) | both_na
            na_only_official = int((np.isnan(av) & ~np.isnan(bv)).sum())
            na_only_reference = int((~np.isnan(av) & np.isnan(bv)).sum())
            replaced = sorted({float(v) for v in bv[np.isnan(av) & ~np.isnan(bv)]})
        else:
            equal = (a.astype("string").str.strip() == b.astype("string").str.strip()).to_numpy()
            na_only_official = na_only_reference = 0
            replaced = []
        columns.append(
            {
                "variable": col,
                "iguales": int(equal.sum()),
                "distintas": int((~equal).sum()),
                "ausente_solo_oficial": na_only_official,
                "ausente_solo_referencia": na_only_reference,
                "valores_referencia_sustituidos": replaced[:5],
            }
        )
    differing = [c for c in columns if c["distintas"] > 0]
    only_official = [c for c in official.columns if c not in reference.columns]
    only_reference = [c for c in reference.columns if c not in official.columns]
    return {
        "n_official": len(official),
        "n_reference": len(reference),
        "same_order": same_order,
        "n_columns_compared": len(columns),
        "n_identical": len(columns) - len(differing),
        "differing": differing,
        "only_official": only_official,
        "only_reference": only_reference,
        "equivalent": same_order
        and all(c["distintas"] == c["ausente_solo_oficial"] for c in differing),
    }
