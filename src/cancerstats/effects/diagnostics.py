"""Diagnóstico del modelo: las cinco hipótesis, colinealidad y observaciones influyentes.

Con mínimos cuadrados ponderados, el diagnóstico se hace sobre el modelo *blanqueado*
y* = √w·y, X* = √w·X, que es un modelo de MCO con los mismos coeficientes: sus residuos
tipificados, su matriz sombrero y sus medidas de influencia (Cook, DFBETAS, DFFITS) son las
medidas correctas del ajuste ponderado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson

from cancerstats import inference as inf
from cancerstats.collinearity import belsley, vif_table
from cancerstats.dictionary import label
from cancerstats.effects.design import Design
from cancerstats.effects.fit import fit, wald

Record = dict[str, object]


def whitened(design: Design, terms: list[str], weights: np.ndarray | None):  # type: ignore[no-untyped-def]
    """Modelo de MCO equivalente al ponderado (y*, X*) y su ajuste."""
    cols = design.columns_for(terms)
    w = np.ones(design.n) if weights is None else weights
    sw = np.sqrt(w)
    xs = design.X[cols].to_numpy() * sw[:, None]
    ys = design.y.to_numpy() * sw
    res = sm.OLS(ys, pd.DataFrame(xs, columns=cols, index=design.X.index)).fit()
    return res, xs, ys


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> Record:
    """Correlación intraclase por ANOVA de un factor (estimador de momentos).

    ICC = (CMB − CMD) / (CMB + (n₀ − 1)·CMD), con n₀ el tamaño de grupo ajustado. Mide qué
    fracción de la variabilidad residual es común a los condados de un mismo estado.
    """
    df = pd.DataFrame({"v": values, "g": groups})
    k = df["g"].nunique()
    n = len(df)
    sizes = df.groupby("g")["v"].size().to_numpy()
    grand = df["v"].mean()
    means = df.groupby("g")["v"].transform("mean")
    ssb = float(((means - grand) ** 2).sum())
    ssw = float(((df["v"] - means) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - (sizes**2).sum() / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    f = msb / msw
    return {
        "icc": float(icc),
        "F": float(f),
        "df1": k - 1,
        "df2": n - k,
        "p": float(stats.f.sf(f, k - 1, n - k)),
        "n0": float(n0),
    }


def _component_residual(
    design: Design,
    res: sm.regression.linear_model.RegressionResultsWrapper,
    var: str,
    max_points: int = 1500,
    seed: int = 0,
) -> Record:
    x = design.X[var].to_numpy() + design.centers.get(var, 0.0)
    resid = design.y.to_numpy() - np.asarray(res.fittedvalues)
    partial = float(res.params[var]) * design.X[var].to_numpy() + resid
    smooth = lowess(partial, x, frac=0.4, it=1, return_sorted=True)
    idx = np.arange(x.size)
    if x.size > max_points:
        idx = np.random.default_rng(seed).choice(x.size, max_points, replace=False)
    step = max(1, smooth.shape[0] // 120)
    return {
        "variable": var,
        "etiqueta": label(var),
        "x": x[idx].round(4).tolist(),
        "y": partial[idx].round(4).tolist(),
        "smooth_x": smooth[::step, 0].round(4).tolist(),
        "smooth_y": smooth[::step, 1].round(4).tolist(),
        "slope": float(res.params[var]),
    }


def diagnose(
    design: Design,
    terms: list[str],
    weights: np.ndarray | None,
    cov: str,
    alpha: float,
    correction: str,
    seed: int,
) -> Record:
    """Diagnóstico completo del modelo final."""
    res = fit(design, terms, weights, cov)
    res_w, xs, _ys = whitened(design, terms, weights)
    infl = res_w.get_influence()
    n = design.n
    p = xs.shape[1]
    resid_w = np.asarray(res_w.resid, dtype=float)
    std_res = np.asarray(infl.resid_studentized_internal, dtype=float)
    stud = infl.resid_studentized_external
    lev = infl.hat_matrix_diag
    cook = infl.cooks_distance[0]
    dffits = infl.dffits[0]
    dfbetas = infl.dfbetas
    cols = design.columns_for(terms)

    # 1. Linealidad ---------------------------------------------------------------------
    reset = linear_reset(res_w, power=3, use_f=True)
    curvature = []
    numeric_terms = [t for t in terms if t in design.numeric]
    for var in numeric_terms:
        sq = design.X[var].to_numpy() ** 2
        aug = design.X[cols].assign(**{f"{var}^2": sq - sq.mean()})
        w = np.ones(n) if weights is None else weights
        m = sm.WLS(design.y, aug, weights=w)
        r = (
            m.fit(cov_type="cluster", cov_kwds={"groups": design.groups}, use_t=True)
            if cov == "cluster"
            else m.fit(cov_type="HC3", use_t=True)
            if cov == "HC3"
            else m.fit()
        )
        f_stat, p_val, _ = wald(r, [f"{var}^2"])
        curvature.append({"variable": var, "etiqueta": label(var), "F": f_stat, "p": p_val})
    if curvature:
        adj = multipletests([float(c["p"]) for c in curvature], method=correction)[1]  # type: ignore[arg-type]
        for c, q in zip(curvature, adj, strict=True):
            c["p_ajustado"] = float(q)
    top_vars = sorted(numeric_terms, key=lambda v: -abs(float(res.tvalues[v])))[:6]
    partial_plots = [_component_residual(design, res, v, seed=seed) for v in top_vars]

    # 2. Normalidad ---------------------------------------------------------------------
    normality = inf.normality_battery(std_res)
    shape = inf.shape(std_res)

    # 3. Homocedasticidad ---------------------------------------------------------------
    # Contraste de Breusch-Pagan (versión de Koenker) de los residuos blanqueados frente a
    # las explicativas originales: si la ponderación es correcta, su varianza no depende de X.
    bp = het_breuschpagan(resid_w, design.X[cols].to_numpy(), robust=True)
    fitted_w = np.asarray(res_w.fittedvalues)
    white_x = sm.add_constant(np.column_stack([fitted_w, fitted_w**2]))
    white = het_breuschpagan(resid_w, white_x, robust=True)
    inv_pop = sm.add_constant(1 / design.population)
    bp_pop = het_breuschpagan(resid_w, inv_pop, robust=True)
    order = np.argsort(design.population)
    third = n // 3
    e2 = resid_w**2
    gq = float(e2[order[:third]].mean() / e2[order[-third:]].mean())
    gq_p = float(stats.f.sf(gq, third - p, third - p))
    pop_bins = pd.qcut(design.population, 10, labels=False, duplicates="drop")
    var_by_decile = pd.Series(e2).groupby(pop_bins).mean().round(4).tolist()

    # 4. Independencia ------------------------------------------------------------------
    dw = float(durbin_watson(resid_w))
    runs = inf.runs_test(std_res, cutoff=0.0)
    icc = icc_oneway(std_res, design.groups)

    # 5. Media cero ---------------------------------------------------------------------
    w_all = np.ones(n) if weights is None else weights
    raw_resid = design.y.to_numpy() - np.asarray(res.fittedvalues)
    mean_resid = float((w_all * raw_resid).sum() / w_all.sum())

    # Colinealidad del modelo final ---------------------------------------------------
    xv = design.X[[c for c in cols if c != "const"]]
    vif = vif_table(xv).to_dict(orient="records") if xv.shape[1] > 1 else []
    bk = belsley(xv, 30.0) if xv.shape[1] > 1 else {"numero_condicion": 1.0, "componentes": []}

    # Influencia ------------------------------------------------------------------------
    thr = {
        "leverage": 2 * p / n,
        "cook": 4 / n,
        "cook_f50": float(stats.f.ppf(0.5, p, n - p)),
        "dfbetas": 2 / np.sqrt(n),
        "dffits": 2 * np.sqrt(p / n),
        "studentized": 3.0,
    }
    max_dfb = np.abs(dfbetas).max(axis=1)
    which_dfb = [cols[i] for i in np.abs(dfbetas).argmax(axis=1)]
    table = pd.DataFrame(
        {
            "county_id": design.county,
            "region": design.region,
            "y": design.y.to_numpy(),
            "fitted": np.asarray(res.fittedvalues),
            "resid": raw_resid,
            "std_resid": std_res,
            "stud_resid": stud,
            "leverage": lev,
            "cook": cook,
            "dffits": dffits,
            "max_dfbetas": max_dfb,
            "dfbetas_coef": which_dfb,
            "weight": w_all,
            "population": design.population,
        }
    )
    flagged = table["cook"] > thr["cook"]
    top = table.sort_values("cook", ascending=False).head(15)
    dfbetas_counts = {
        cols[j]: int((np.abs(dfbetas[:, j]) > thr["dfbetas"]).sum()) for j in range(len(cols))
    }
    return {
        "linearity": {
            "reset": {
                "F": float(reset.fvalue),
                "p": float(reset.pvalue),
                "df": [int(reset.df_num), int(reset.df_denom)],
            },
            "curvature": curvature,
            "partial_residuals": partial_plots,
        },
        "normality": {
            "tests": normality,
            "shape": shape,
            "qq": inf.qq_points(std_res),
            "histogram": np.histogram(std_res, bins=50)[0].tolist(),
            "histogram_edges": np.histogram(std_res, bins=50)[1].round(4).tolist(),
        },
        "homoscedasticity": {
            "breusch_pagan": {"LM": float(bp[0]), "p": float(bp[1])},
            "white_reduced": {"LM": float(white[0]), "p": float(white[1])},
            "bp_population": {"LM": float(bp_pop[0]), "p": float(bp_pop[1])},
            "goldfeld_quandt": {"F": gq, "p": gq_p},
            "var_by_population_decile": var_by_decile,
        },
        "independence": {"durbin_watson": dw, "runs": runs, "icc_state": icc},
        "mean_zero": {"weighted_mean_residual": mean_resid},
        "collinearity": {"vif": vif, "condition_number": bk["numero_condicion"]},
        "influence": {
            "thresholds": thr,
            "n_leverage": int((table["leverage"] > thr["leverage"]).sum()),
            "n_cook": int(flagged.sum()),
            "n_cook_f50": int((table["cook"] > thr["cook_f50"]).sum()),
            "n_studentized": int((table["stud_resid"].abs() > thr["studentized"]).sum()),
            "n_dffits": int((table["dffits"].abs() > thr["dffits"]).sum()),
            "dfbetas_counts": dfbetas_counts,
            "top": top.to_dict(orient="records"),
            "flagged_ids": table.loc[flagged, "county_id"].tolist(),
        },
        "table": table,
    }


def residual_check_before(design: Design, terms: list[str]) -> Record:
    """Diagnóstico mínimo del ajuste inicial por MCO (motiva la ponderación y el cluster)."""
    res = fit(design, terms, None, "nonrobust")
    cols = design.columns_for(terms)
    xs = design.X[cols].to_numpy()
    bp = het_breuschpagan(res.resid, xs, robust=True)
    bp_pop = het_breuschpagan(res.resid, sm.add_constant(1 / design.population), robust=True)
    std = res.get_influence().resid_studentized_internal
    e2 = np.asarray(res.resid) ** 2
    pop_bins = pd.qcut(design.population, 10, labels=False, duplicates="drop")
    deciles = pd.Series(e2).groupby(pop_bins).mean()
    edges = np.quantile(design.population, np.linspace(0, 1, 11))
    return {
        "breusch_pagan": {"LM": float(bp[0]), "p": float(bp[1])},
        "bp_population": {"LM": float(bp_pop[0]), "p": float(bp_pop[1])},
        "icc_state": icc_oneway(std, design.groups),
        "durbin_watson": float(durbin_watson(res.resid)),
        "shape": inf.shape(std),
        "var_by_population_decile": deciles.round(3).tolist(),
        "population_decile_edges": edges.round(0).tolist(),
        "r2": float(res.rsquared),
        "n": int(res.nobs),
    }
