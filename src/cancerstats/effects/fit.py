"""Ajuste por mínimos cuadrados (ordinarios o ponderados) y contrastes de Wald.

La ponderación sigue el modelo de varianza de una tasa estimada:

    Var(ε_i) = a + b / n_i,

donde *a* es la varianza «real» entre condados y *b / n_i* el ruido de muestreo de una
tasa calculada sobre una población n_i (para un proceso de Poisson, la varianza de la tasa
es proporcional a la tasa dividida por la población expuesta). Los parámetros se estiman
regresando los residuos al cuadrado sobre 1/n_i y se itera (mínimos cuadrados ponderados
factibles, FGLS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from cancerstats.effects.design import Design


@dataclass
class VarianceFunction:
    """Modelo de varianza σ²_i = a + b / población_i."""

    a: float
    b: float
    se_a: float
    se_b: float
    p_b: float
    r2: float
    iterations: int
    clamped: bool

    def variance(self, population: np.ndarray) -> np.ndarray:
        """Varianza implicada para cada población."""
        return self.a + self.b / population

    def weights(self, population: np.ndarray) -> np.ndarray:
        """Pesos 1/σ²_i normalizados a media 1."""
        w = 1.0 / self.variance(population)
        return w / w.mean()

    def sampling_share(self, population: float) -> float:
        """Fracción de la varianza atribuible al ruido de muestreo para una población dada."""
        return (self.b / population) / (self.a + self.b / population)

    def to_dict(self) -> dict[str, float | int | bool]:
        """Representación serializable."""
        return dict(self.__dict__)


def fit(
    design: Design,
    terms: list[str],
    weights: np.ndarray | None = None,
    cov: str = "cluster",
    rows: np.ndarray | None = None,
) -> RegressionResultsWrapper:
    """Ajusta y = Xβ + ε con los términos indicados.

    Args:
        design: matriz de diseño completa.
        terms: términos incluidos (la constante siempre lo está).
        weights: pesos de MCP; ``None`` para MCO.
        cov: ``"nonrobust"``, ``"HC3"`` o ``"cluster"`` (por estado, con t de G − 1 g.l.).
        rows: máscara booleana opcional de filas.
    """
    cols = design.columns_for(terms)
    X = design.X[cols]
    y = design.y
    groups = design.groups
    w = weights
    if rows is not None:
        X, y, groups = X[rows], y[rows], groups[rows]
        w = None if weights is None else weights[rows]
    model = sm.WLS(y, X, weights=w) if w is not None else sm.OLS(y, X)
    if cov == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    if cov == "HC3":
        return model.fit(cov_type="HC3", use_t=True)
    return model.fit()


def df_denominator(res: RegressionResultsWrapper) -> float:
    """Grados de libertad del denominador para t y F según la covarianza usada."""
    if res.cov_type == "cluster":
        return float(res.df_resid_inference)
    return float(res.df_resid)


def wald(res: RegressionResultsWrapper, columns: list[str]) -> tuple[float, float, int]:
    """Contraste de Wald H0: β_j = 0 para todas las columnas indicadas.

    F = (Rβ̂)ᵀ (R V̂ Rᵀ)⁻¹ (Rβ̂) / q, comparado con F(q, g.l.), donde V̂ es la covarianza
    elegida (clásica, HC3 o por conglomerados). Con una sola columna, F = t².
    """
    names = list(res.params.index)
    idx = [names.index(c) for c in columns]
    beta = res.params.to_numpy()[idx]
    cov = np.asarray(res.cov_params())[np.ix_(idx, idx)]
    q = len(idx)
    stat = float(beta @ np.linalg.solve(cov, beta)) / q
    dfd = df_denominator(res)
    return stat, float(stats.f.sf(stat, q, dfd)), q


def estimate_variance_function(
    design: Design,
    terms: list[str],
    iterations: int,
) -> tuple[VarianceFunction, np.ndarray]:
    """Estima σ²_i = a + b/población_i por FGLS iterativo.

    Returns:
        La función de varianza y los pesos normalizados resultantes.
    """
    inv_pop = 1.0 / design.population
    z = sm.add_constant(inv_pop)
    weights = np.ones(design.n)
    res_aux = None
    clamped = False
    a = b = 0.0
    it = 0
    for it in range(1, iterations + 1):
        res = fit(design, terms, weights if it > 1 else None, cov="nonrobust")
        e2 = (design.y.to_numpy() - np.asarray(res.fittedvalues)) ** 2
        res_aux = sm.OLS(e2, z).fit(cov_type="HC3")
        a, b = (float(v) for v in res_aux.params)
        floor = 0.01 * float(e2.mean())
        if a < floor:
            a, clamped = floor, True
        if b < 0:
            b, clamped = 0.0, True
        new = 1.0 / (a + b * inv_pop)
        new = new / new.mean()
        if np.max(np.abs(new - weights)) < 1e-6:
            weights = new
            break
        weights = new
    assert res_aux is not None
    vf = VarianceFunction(
        a=a,
        b=b,
        se_a=float(res_aux.bse[0]),
        se_b=float(res_aux.bse[1]),
        p_b=float(res_aux.pvalues[1]),
        r2=float(res_aux.rsquared),
        iterations=it,
        clamped=clamped,
    )
    return vf, weights


def coef_table(res: RegressionResultsWrapper, design: Design, alpha: float = 0.05) -> pd.DataFrame:
    """Tabla de coeficientes con IC, t, p y coeficiente tipificado (Beta del SPSS)."""
    ci = res.conf_int(alpha)
    y_sd = float(design.y.std(ddof=1))
    beta_std = []
    for name in res.params.index:
        if name in design.numeric:
            beta_std.append(float(res.params[name]) * float(design.X[name].std(ddof=1)) / y_sd)
        else:
            beta_std.append(np.nan)
    return pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.to_numpy(),
            "se": res.bse.to_numpy(),
            "t": res.tvalues.to_numpy(),
            "p": res.pvalues.to_numpy(),
            "lower": ci.iloc[:, 0].to_numpy(),
            "upper": ci.iloc[:, 1].to_numpy(),
            "beta": beta_std,
        }
    )


def summary_stats(res: RegressionResultsWrapper, n_clusters: int) -> dict[str, float]:
    """Bondad de ajuste: R², R² ajustado, F global (clásico y robusto), AIC, BIC, σ.

    Con ponderación, R² y la tabla ANOVA se calculan en la métrica ponderada (la que
    minimiza el estimador); ``r2_unweighted`` mide el ajuste en la escala original.
    """
    k = int(res.df_model)
    n = int(res.nobs)
    yv = np.asarray(res.model.endog, dtype=float)
    fitted = np.asarray(res.fittedvalues, dtype=float)
    w = np.asarray(getattr(res.model, "weights", np.ones(n)), dtype=float) * np.ones(n)
    sse_raw = float(((yv - fitted) ** 2).sum())
    sst_raw = float(((yv - yv.mean()) ** 2).sum())
    ybar_w = float((w * yv).sum() / w.sum())
    sct = float((w * (yv - ybar_w) ** 2).sum())
    sce = float((w * (yv - fitted) ** 2).sum())
    scr = sct - sce
    cols = [c for c in res.params.index if c != "const"]
    f_rob, p_rob, _ = wald(res, cols) if cols else (float("nan"), float("nan"), 0)
    dfe = n - k - 1
    f_classic = (scr / k) / (sce / dfe) if k > 0 else float("nan")
    return {
        "n": n,
        "k": k,
        "r2": float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "r2_unweighted": 1 - sse_raw / sst_raw,
        "sigma": float(np.sqrt(res.scale)),
        "rmse_unweighted": float(np.sqrt(sse_raw / n)),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "loglik": float(res.llf),
        "f_classic": float(f_classic),
        "p_f_classic": float(stats.f.sf(f_classic, k, dfe)) if k > 0 else float("nan"),
        "f_robust": f_rob,
        "p_f_robust": p_rob,
        "scr": scr,
        "sce": sce,
        "sct": sct,
        "df_resid": float(res.df_resid),
        "df_inference": df_denominator(res),
        "n_clusters": float(n_clusters),
    }
