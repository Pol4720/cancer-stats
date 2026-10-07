"""Detección de valores atípicos univariantes y multivariantes.

Un atípico no se elimina por serlo: sólo se corrige si es un error documentado (y eso ya
lo hizo la depuración). El resto se identifica, se intenta explicar y se evalúa su
influencia en el modelo, ajustando con y sin él.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet

from cancerstats.dictionary import label

Record = dict[str, object]


def univariate(
    df: pd.DataFrame, columns: list[str], iqr_factor: float, z_cut: float
) -> list[Record]:
    """Atípicos por variable: vallas de Tukey y puntuación z robusta (mediana y MEDA)."""
    rows: list[Record] = []
    for col in columns:
        x = df[col].astype(float)
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        lo3, hi3 = q1 - 3 * iqr, q3 + 3 * iqr
        med = x.median()
        mad = (x - med).abs().median()
        robust_z = 0.6745 * (x - med) / mad if mad > 0 else pd.Series(0.0, index=x.index)
        tukey = (x < lo) | (x > hi)
        extreme = (x < lo3) | (x > hi3)
        rz = robust_z.abs() > z_cut
        top = df.loc[rz.fillna(False), ["county_id", col]].assign(z=robust_z[rz.fillna(False)])
        top = top.reindex(top["z"].abs().sort_values(ascending=False).index).head(5)
        rows.append(
            {
                "variable": col,
                "etiqueta": label(col),
                "valla_inferior": float(lo),
                "valla_superior": float(hi),
                "n_tukey": int(tukey.sum()),
                "n_extremos": int(extreme.sum()),
                "n_z_robusto": int(rz.sum()),
                "ejemplos": [
                    {"county_id": r["county_id"], "valor": float(r[col]), "z": float(r["z"])}
                    for _, r in top.iterrows()
                ],
            }
        )
    return rows


def multivariate(df: pd.DataFrame, columns: list[str], quantile: float, seed: int) -> Record:
    """Distancia de Mahalanobis robusta (determinante de covarianza mínimo, MCD).

    La distancia clásica usa la media y la covarianza muestrales, que los propios atípicos
    distorsionan (efecto máscara). El MCD estima ambas con el subconjunto más concentrado
    de los datos y es resistente a hasta casi la mitad de contaminación.
    """
    data = df[["county_id", *columns]].dropna()
    x = data[columns].to_numpy(dtype=float)
    x = (x - np.median(x, axis=0)) / x.std(axis=0, ddof=1)
    mcd = MinCovDet(random_state=seed).fit(x)
    d2 = mcd.mahalanobis(x)
    classical = np.einsum(
        "ij,jk,ik->i",
        x - x.mean(axis=0),
        np.linalg.inv(np.cov(x, rowvar=False)),
        x - x.mean(axis=0),
    )
    cut = float(stats.chi2.ppf(quantile, len(columns)))
    flagged = d2 > cut
    order = np.argsort(-d2)[:15]
    top = []
    for i in order:
        zrow = x[i]
        worst = np.argsort(-np.abs(zrow))[:3]
        top.append(
            {
                "county_id": str(data["county_id"].iloc[i]),
                "d2_robusta": float(d2[i]),
                "d2_clasica": float(classical[i]),
                "variables": [label(columns[j]) for j in worst],
            }
        )
    return {
        "n": len(data),
        "p": len(columns),
        "cutoff": cut,
        "quantile": quantile,
        "n_flagged_robust": int(flagged.sum()),
        "n_flagged_classical": int((classical > cut).sum()),
        "top": top,
        "d2": dict(zip(data["county_id"].astype(str), np.round(d2, 3).tolist(), strict=True)),
    }


def response_outliers(df: pd.DataFrame, response: str, z_cut: float) -> list[Record]:
    """Condados con mortalidad atípica según la puntuación z robusta."""
    y = df[response]
    med = y.median()
    mad = (y - med).abs().median()
    z = 0.6745 * (y - med) / mad
    sel = df.loc[z.abs() > z_cut, ["county_id", response, "incidenceRate", "popEst2015"]]
    sel = sel.assign(z=z[z.abs() > z_cut]).sort_values("z", key=np.abs, ascending=False)
    return [
        {
            "county_id": r["county_id"],
            "valor": float(r[response]),
            "incidencia": None if pd.isna(r["incidenceRate"]) else float(r["incidenceRate"]),
            "poblacion": int(r["popEst2015"]),
            "z": float(r["z"]),
        }
        for _, r in sel.iterrows()
    ]


def detect(
    df: pd.DataFrame,
    response: str,
    numeric: list[str],
    iqr_factor: float,
    z_cut: float,
    quantile: float,
    seed: int,
) -> Record:
    """Etapa completa de atípicos."""
    return {
        "univariate": univariate(df, [response, *numeric], iqr_factor, z_cut),
        "multivariate": multivariate(df, numeric, quantile, seed),
        "response": response_outliers(df, response, z_cut),
    }
