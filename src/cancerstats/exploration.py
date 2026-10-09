"""Análisis exploratorio e inferencia descriptiva sobre los datos depurados."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from cancerstats import inference as inf
from cancerstats.dictionary import label

Record = dict[str, object]


def describe(df: pd.DataFrame, columns: list[str]) -> list[Record]:
    """Estadísticos univariantes de cada variable numérica."""
    rows: list[Record] = []
    for col in columns:
        x = df[col].astype(float)
        a = x.dropna().to_numpy()
        q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
        rows.append(
            {
                "variable": col,
                "etiqueta": label(col),
                "n": int(a.size),
                "ausentes": int(x.isna().sum()),
                "media": float(a.mean()),
                "dt": float(a.std(ddof=1)),
                "min": float(a.min()),
                "q1": float(q1),
                "mediana": float(med),
                "q3": float(q3),
                "max": float(a.max()),
                "iqr": float(q3 - q1),
                "asimetria": float(stats.skew(a)),
                "curtosis": float(stats.kurtosis(a)),
                "cv": float(a.std(ddof=1) / a.mean()) if a.mean() != 0 else None,
                "ceros": int((a == 0).sum()),
            }
        )
    return rows


def histogram(x: pd.Series, bins: int = 40) -> Record:
    """Histograma con la densidad normal ajustada y una estimación núcleo (KDE)."""
    a = x.dropna().to_numpy(dtype=float)
    counts, edges = np.histogram(a, bins=bins)
    grid = np.linspace(edges[0], edges[-1], 200)
    kde = stats.gaussian_kde(a)(grid)
    normal = stats.norm.pdf(grid, a.mean(), a.std(ddof=1))
    return {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "grid": grid.tolist(),
        "kde": kde.tolist(),
        "normal": normal.tolist(),
        "n": int(a.size),
    }


def response_analysis(
    df: pd.DataFrame, response: str, alpha: float, reps: int, seed: int
) -> Record:
    """Estudio completo de la variable respuesta: estimación, normalidad y forma."""
    y = df[response].astype(float)
    a = y.dropna().to_numpy()
    return {
        "variable": response,
        "mean_ci": inf.mean_ci(a, alpha),
        "variance_ci": inf.variance_ci(a, alpha),
        "median_ci_order": inf.median_ci_order(a, alpha),
        "median_ci_boot": inf.bootstrap_ci(a, "median", reps, alpha, seed),
        "trimmed_ci_boot": inf.bootstrap_ci(a, "trimmed_mean", reps, alpha, seed + 1),
        "robust": inf.robust_location(a),
        "shape": inf.shape(a),
        "normality": inf.normality_battery(a),
        "boxcox": inf.boxcox_lambda(a, alpha),
        "histogram": histogram(y),
        "qq": inf.qq_points(a),
        "runs_file_order": inf.runs_test(a),
        "extremes": {
            "lowest": df.nsmallest(5, response)[["county_id", response]].to_dict(orient="records"),
            "highest": df.nlargest(5, response)[["county_id", response]].to_dict(orient="records"),
        },
    }


def correlations(
    df: pd.DataFrame, response: str, columns: list[str], alpha: float, correction: str
) -> list[Record]:
    """Correlación de cada explicativa con la respuesta, con p-valores corregidos."""
    rows = []
    for col in columns:
        res = inf.correlation_test(df[col].to_numpy(), df[response].to_numpy(), alpha)
        res.update({"variable": col, "etiqueta": label(col)})
        rows.append(res)
    adj_p = inf.adjust([float(r["p_pearson"]) for r in rows], correction)  # type: ignore[arg-type]
    adj_s = inf.adjust([float(r["p_spearman"]) for r in rows], correction)  # type: ignore[arg-type]
    for r, p, q in zip(rows, adj_p, adj_s, strict=True):
        r["p_pearson_adj"] = p
        r["p_spearman_adj"] = q
    return sorted(rows, key=lambda r: -abs(float(r["pearson"])))  # type: ignore[arg-type]


def correlation_matrix(df: pd.DataFrame, columns: list[str], method: str = "pearson") -> Record:
    """Matriz de correlaciones (casos disponibles por pares)."""
    mat = df[columns].astype(float).corr(method=method)  # type: ignore[arg-type]
    return {
        "variables": columns,
        "labels": [label(c) for c in columns],
        "values": np.round(mat.to_numpy(), 4).tolist(),
        "method": method,
    }


def income_gradient(df: pd.DataFrame, response: str) -> Record:
    """Mortalidad por decil de renta: tendencia monótona (Spearman) y ANOVA."""
    bounds = df["binnedInc"].str.extract(r"[\(\[]\s*([\d.]+)")[0].astype(float)
    order = bounds.rank(method="dense").astype(int)
    table = (
        df.assign(decil=order)
        .groupby("decil")[response]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    rho, p = stats.spearmanr(order, df[response])
    groups = [g[response].to_numpy(dtype=float) for _, g in df.assign(decil=order).groupby("decil")]
    anova = stats.f_oneway(*groups)
    return {
        "table": table.to_dict(orient="records"),
        "spearman": float(rho),
        "p_spearman": float(p),
        "anova_f": float(anova.statistic),
        "anova_p": float(anova.pvalue),
    }


def explore(
    df: pd.DataFrame,
    response: str,
    numeric: list[str],
    alpha: float,
    reps: int,
    seed: int,
    correction: str,
) -> Record:
    """Ejecuta toda la etapa exploratoria."""
    region = inf.k_sample(df, response, "region", alpha, correction)
    poverty = df["povertyPercent"] >= 20
    high_pov = inf.two_sample(
        df.loc[poverty, response].to_numpy(),
        df.loc[~poverty, response].to_numpy(),
        ("Pobreza ≥ 20 %", "Pobreza < 20 %"),
    )
    trials = df["studyPerCap"] > 0
    study = inf.two_sample(
        df.loc[trials, response].to_numpy(),
        df.loc[~trials, response].to_numpy(),
        ("Con ensayos", "Sin ensayos"),
    )
    state_means = (
        df.groupby("state")[response]
        .agg(["count", "mean", "std"])
        .sort_values("mean")
        .reset_index()
    )
    state_anova = stats.f_oneway(
        *[g[response].to_numpy(dtype=float) for _, g in df.groupby("state") if len(g) > 1]
    )
    ss_tot = float(((df[response] - df[response].mean()) ** 2).sum())
    ss_state = float(
        (df.groupby("state")[response].transform("mean") - df[response].mean()).pow(2).sum()
    )
    return {
        "describe": describe(df, [response, *numeric]),
        "response": response_analysis(df, response, alpha, reps, seed),
        "correlations": correlations(df, response, numeric, alpha, correction),
        "corr_matrix": correlation_matrix(df, [response, *numeric], "pearson"),
        "corr_matrix_spearman": correlation_matrix(df, [response, *numeric], "spearman"),
        "region": region,
        "high_poverty": high_pov,
        "clinical_trials": study,
        "income_gradient": income_gradient(df, response),
        "state": {
            "table": state_means.to_dict(orient="records"),
            "anova_f": float(state_anova.statistic),
            "anova_p": float(state_anova.pvalue),
            "eta2": ss_state / ss_tot,
        },
    }
