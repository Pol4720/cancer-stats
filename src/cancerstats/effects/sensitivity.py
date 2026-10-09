"""Análisis de sensibilidad: ¿dependen las conclusiones de las decisiones tomadas?

Se reajusta el modelo final bajo especificaciones alternativas razonables. Si los
coeficientes y su significación se mantienen, las conclusiones no son un artefacto de una
decisión concreta (ponderar, el tipo de error típico, los condados influyentes, el
tratamiento de los ausentes, el efecto de estado...).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.robust.norms import HuberT
from statsmodels.robust.robust_linear_model import RLM

from cancerstats.effects.design import Design, build_design
from cancerstats.effects.fit import estimate_variance_function, fit
from cancerstats.missing import multiple_imputation, rubin_pool

Record = dict[str, object]


def _table(
    name: str,
    label: str,
    params: pd.Series,
    se: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    pvalues: pd.Series,
    n: int,
    note: str = "",
) -> Record:
    return {
        "id": name,
        "label": label,
        "n": n,
        "note": note,
        "coef": {k: float(v) for k, v in params.items()},
        "se": {k: float(v) for k, v in se.items()},
        "lower": {k: float(v) for k, v in lower.items()},
        "upper": {k: float(v) for k, v in upper.items()},
        "p": {k: float(v) for k, v in pvalues.items()},
    }


def _from_res(name: str, label: str, res, note: str = "") -> Record:  # type: ignore[no-untyped-def]
    ci = res.conf_int()
    return _table(
        name,
        label,
        res.params,
        res.bse,
        ci.iloc[:, 0],
        ci.iloc[:, 1],
        res.pvalues,
        int(res.nobs),
        note,
    )


def cluster_bootstrap(
    design: Design, terms: list[str], weights: np.ndarray, reps: int, seed: int, alpha: float
) -> Record:
    """Bootstrap por conglomerados: se remuestrean estados completos con reemplazamiento.

    Respeta la dependencia dentro de cada estado (remuestrear condados sueltos la ignoraría)
    y no supone normalidad. Los pesos de cada condado se mantienen fijos.
    """
    cols = design.columns_for(terms)
    X = design.X[cols].to_numpy()
    y = design.y.to_numpy()
    sw = np.sqrt(weights)
    groups = design.groups
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(seed)
    draws = []
    failures = 0
    for _ in range(reps):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([members[i] for i in pick])
        xs = X[idx] * sw[idx, None]
        ys = y[idx] * sw[idx]
        if np.linalg.matrix_rank(xs) < xs.shape[1]:
            failures += 1
            continue
        beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
        draws.append(beta)
    arr = np.asarray(draws)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2], axis=0)
    return {
        "reps": reps,
        "failures": failures,
        "coef": dict(zip(cols, arr.mean(axis=0).tolist(), strict=True)),
        "se": dict(zip(cols, arr.std(axis=0, ddof=1).tolist(), strict=True)),
        "lower": dict(zip(cols, lo.tolist(), strict=True)),
        "upper": dict(zip(cols, hi.tolist(), strict=True)),
        "p_sign": dict(
            zip(
                cols,
                (2 * np.minimum((arr > 0).mean(axis=0), (arr < 0).mean(axis=0))).tolist(),
                strict=True,
            )
        ),
    }


def mixed_model(design: Design, terms: list[str]) -> Record:
    """Modelo lineal mixto con intercepto aleatorio por estado (sin ponderar)."""
    cols = design.columns_for(terms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedLM(design.y.to_numpy(), design.X[cols], groups=design.groups).fit(reml=True)
    fe = res.fe_params
    bse = res.bse_fe
    z = 1.959963984540054
    s2u = float(res.cov_re.iloc[0, 0])
    out = _table(
        "mixed",
        "Modelo mixto (intercepto aleatorio por estado)",
        fe,
        bse,
        fe - z * bse,
        fe + z * bse,
        res.pvalues[fe.index],
        int(res.nobs),
        (
            f"σ²_estado = {s2u:.2f}; σ²_residual = {res.scale:.2f}; "
            f"ICC = {s2u / (s2u + res.scale):.3f}"
        ).replace(".", ","),
    )
    out["icc"] = s2u / (s2u + float(res.scale))
    return out


def huber(design: Design, terms: list[str], weights: np.ndarray) -> Record:
    """Regresión robusta M de Huber sobre el modelo blanqueado (resistente a atípicos en y)."""
    cols = design.columns_for(terms)
    sw = np.sqrt(weights)
    xs = design.X[cols].mul(sw, axis=0)
    ys = design.y * sw
    res = RLM(ys, xs, M=HuberT()).fit()
    out = _from_res(
        "huber",
        "Regresión robusta de Huber (M-estimador)",
        res,
        "Pondera a la baja los residuos grandes en lugar de eliminarlos.",
    )
    w_rob = np.asarray(res.weights)
    out["downweighted"] = int((w_rob < 1).sum())
    return out


def state_fixed_effects(
    df: pd.DataFrame, design: Design, terms: list[str], weights: np.ndarray
) -> Record:
    """Efectos fijos de estado: la asociación se estima sólo con variación dentro de estado."""
    keep = [t for t in terms if t != "region" and not t.endswith(":region")]
    cols = design.columns_for(keep)
    states = pd.get_dummies(
        pd.Categorical(design.groups), prefix="st", drop_first=True, dtype=float
    )
    states.index = design.X.index
    X = pd.concat([design.X[cols], states], axis=1)
    res = sm.WLS(design.y, X, weights=weights).fit(
        cov_type="cluster", cov_kwds={"groups": design.groups}, use_t=True
    )
    shown = list(cols)
    ci = res.conf_int().loc[shown]
    weighted = not np.allclose(weights, weights[0])
    return _table(
        "state_fe",
        f"Efectos fijos de estado ({'MCPF' if weighted else 'MCO'} + cluster)",
        res.params[shown],
        res.bse[shown],
        ci.iloc[:, 0],
        ci.iloc[:, 1],
        res.pvalues[shown],
        int(res.nobs),
        f"La región desaparece: queda absorbida por los {states.shape[1] + 1} efectos de estado.",
    )


def without_rows(
    design: Design, terms: list[str], weights: np.ndarray, drop_ids: list[str], cov: str
) -> Record:
    """Reajuste sin las observaciones influyentes."""
    mask = ~np.isin(np.asarray(design.county), drop_ids)
    res = fit(design, terms, weights, cov, rows=mask)
    return _from_res(
        "no_influential",
        f"Sin los {len(drop_ids)} condados influyentes (Cook > 4/n)",
        res,
        "Si los coeficientes apenas cambian, ningún condado concreto dirige la conclusión.",
    )


def with_multiple_imputation(
    df: pd.DataFrame,
    response: str,
    numeric: list[str],
    all_numeric: list[str],
    use_region: bool,
    reference: str,
    interactions: list[list[str]],
    terms: list[str],
    cov: str,
    weighting: str,
    fgls_iterations: int,
    m: int,
    max_iter: int,
    seed: int,
) -> Record:
    """Ajuste del modelo final sobre todos los condados con imputación múltiple (Rubin)."""
    to_impute = [c for c in all_numeric if df[c].isna().any()]
    aux = [response, *[c for c in all_numeric if c not in to_impute]]
    region_dummies = pd.get_dummies(df["region"], prefix="reg", dtype=float)
    data = pd.concat([df, region_dummies], axis=1)
    imputed = multiple_imputation(
        data, to_impute, aux + list(region_dummies.columns), m, max_iter, seed
    )
    est, var = [], []
    n = 0
    dfc = 0.0
    for d in imputed:
        des = build_design(d, response, numeric, use_region, reference, interactions, True)
        w = (
            estimate_variance_function(des, terms, fgls_iterations)[1]
            if weighting == "fgls"
            else None
        )
        res = fit(des, terms, w, cov)
        est.append(res.params)
        var.append(res.bse**2)
        n = int(res.nobs)
        dfc = float(res.df_resid_inference if cov == "cluster" else res.df_resid)
    pooled = rubin_pool(est, var, dfc)
    from scipy import stats

    q = stats.t.ppf(0.975, pooled.df)
    lower = pooled.params - q * pooled.se
    upper = pooled.params + q * pooled.se
    pvals = pd.Series(
        2 * stats.t.sf(np.abs(pooled.params / pooled.se), pooled.df), index=pooled.params.index
    )
    out = _table(
        "mi",
        f"Imputación múltiple ({m} conjuntos, todos los condados)",
        pooled.params,
        pooled.se,
        lower,
        upper,
        pvals,
        n,
        "Reglas de Rubin; incluye Kansas, Minnesota y Nevada con incidencia imputada.",
    )
    out["fmi"] = {k: float(v) for k, v in pooled.fmi.items()}
    out["m"] = m
    return out
