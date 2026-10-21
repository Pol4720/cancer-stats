"""Variables indicadoras de atípicos: el modelo de cambio en la media.

Añadir al modelo una indicadora D_i (1 en el condado i, 0 en el resto) equivale a estimar
los demás coeficientes sin ese condado: su coeficiente δ_i = y_i − ŷ_(i) es el error de
predicción del condado cuando no interviene en el ajuste, y el estadístico t de δ_i es su
residuo estudentizado externamente. Con varias indicadoras a la vez:

* el t de cada una contrasta si ese condado es atípico respecto al modelo; como se mira el
  mayor de n residuos, el contraste se corrige por Bonferroni (p·n);
* el F parcial del bloque contrasta si el grupo, en conjunto, es atípico, es decir, si
  conviene excluirlo del ajuste. Equivale al F que compara el ajuste con y sin esos condados;
* los demás coeficientes son exactamente los del ajuste sin esos condados, así que comparar
  con el modelo final mide cuánto cambian las conclusiones.

Si los condados se eligen con los propios residuos (por ejemplo, Cook > 4/n), el F del bloque
no sigue una F: se seleccionan precisamente los residuos grandes. Su distribución nula se
calibra repitiendo la elección y el contraste sobre residuos simulados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from cancerstats.dictionary import label
from cancerstats.effects.design import Design
from cancerstats.effects.fit import wald

Record = dict[str, object]


def indicator_matrix(design: Design, rows: list[int]) -> pd.DataFrame:
    """Una columna indicadora por fila señalada (``D1``, ``D2``…, en el orden dado)."""
    D = np.zeros((design.n, len(rows)))
    D[rows, np.arange(len(rows))] = 1.0
    return pd.DataFrame(D, index=design.X.index, columns=[f"D{k + 1}" for k in range(len(rows))])


def fit_with_indicators(  # type: ignore[no-untyped-def]
    design: Design,
    terms: list[str],
    rows: list[int],
    weights: np.ndarray | None,
    cov: str,
):
    """Ajusta el modelo con una indicadora por cada fila de ``rows``."""
    D = indicator_matrix(design, rows)
    X = pd.concat([design.X[design.columns_for(terms)], D], axis=1)
    model = sm.WLS(design.y, X, weights=weights) if weights is not None else sm.OLS(design.y, X)
    if cov == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": design.groups}, use_t=True)
    if cov == "HC3":
        return model.fit(cov_type="HC3", use_t=True)
    return model.fit()


def _changes(base, alt, cols: list[str], alpha: float) -> tuple[list[Record], float, list[str]]:  # type: ignore[no-untyped-def]
    """Cambio de cada coeficiente (en errores típicos del modelo final) y de su conclusión."""
    out: list[Record] = []
    flips: list[str] = []
    for c in cols:
        if c == "const":
            continue
        b0, b1 = float(base.params[c]), float(alt.params[c])
        se = float(base.bse[c])
        sig0, sig1 = float(base.pvalues[c]) < alpha, float(alt.pvalues[c]) < alpha
        flip = sig0 != sig1 or (sig0 and sig1 and np.sign(b0) != np.sign(b1))
        if flip:
            flips.append(c)
        out.append(
            {
                "term": c,
                "etiqueta": label(c),
                "coef_final": b0,
                "coef": b1,
                "cambio_se": (b1 - b0) / se if se > 0 else 0.0,
                "p_final": float(base.pvalues[c]),
                "p": float(alt.pvalues[c]),
                "cambia_conclusion": bool(flip),
            }
        )
    max_change = max((abs(float(r["cambio_se"])) for r in out), default=0.0)  # type: ignore[arg-type]
    return out, max_change, flips


def mean_shift_model(
    design: Design,
    terms: list[str],
    rows: list[int],
    weights: np.ndarray | None,
    cov: str,
    alpha: float,
    model_id: str,
    title: str,
    criterion: str,
    cook: np.ndarray,
) -> Record:
    """Modelo final + indicadoras de ``rows``: contrastes, ajuste y cambio de los coeficientes."""
    cols = design.columns_for(terms)
    n = design.n
    base_cl = _fit_cov(design, cols, weights, "nonrobust")
    alt_cl = fit_with_indicators(design, terms, rows, weights, "nonrobust")
    base = _fit_cov(design, cols, weights, cov)
    alt = fit_with_indicators(design, terms, rows, weights, cov)
    dcols = [f"D{k + 1}" for k in range(len(rows))]
    q = len(rows)

    # F parcial clásico: H0 δ_1 = … = δ_q = 0 (equivale a comparar con y sin esos condados).
    f_stat = ((base_cl.ssr - alt_cl.ssr) / q) / (alt_cl.ssr / alt_cl.df_resid)
    p_f = float(stats.f.sf(f_stat, q, alt_cl.df_resid))
    # Wald con la covarianza por conglomerados: sólo es estimable con menos indicadoras que
    # estados (la matriz de covarianza robusta tiene rango G − 1).
    n_groups = len(np.unique(design.groups))
    f_rob: float | None = None
    p_rob: float | None = None
    if cov != "cluster" or q < n_groups - 1:
        f_rob, p_rob, _ = wald(alt, dcols)

    indicators: list[Record] = []
    for k, i in enumerate(rows):
        d = dcols[k]
        p_t = float(alt_cl.pvalues[d])
        indicators.append(
            {
                "indicadora": d,
                "county_id": design.county[i],
                "region": str(design.region[i]),
                "poblacion": float(design.population[i]),
                "y": float(design.y.iloc[i]),
                "cook": float(cook[i]),
                "delta": float(alt_cl.params[d]),
                "se": float(alt_cl.bse[d]),
                "t": float(alt_cl.tvalues[d]),
                "p": p_t,
                "p_bonferroni": min(1.0, p_t * n),
                "p_robusto": float(alt.pvalues[d]),
                "atipico": bool(p_t * n < alpha),
            }
        )
    changes, max_change, flips = _changes(base, alt, cols, alpha)
    sigma0 = float(np.sqrt(base_cl.scale))
    ci = alt.conf_int()
    return {
        "id": model_id,
        "label": title,
        "criterio": criterion,
        "q": q,
        "n": n,
        "k": int(alt_cl.df_model),
        "r2": float(alt_cl.rsquared),
        "r2_adj": float(alt_cl.rsquared_adj),
        "aic": float(alt_cl.aic),
        "bic": float(alt_cl.bic),
        "sigma": float(np.sqrt(alt_cl.scale)),
        "sigma_final": sigma0,
        "r2_final": float(base_cl.rsquared),
        "r2_adj_final": float(base_cl.rsquared_adj),
        "aic_final": float(base_cl.aic),
        "bic_final": float(base_cl.bic),
        "F": float(f_stat),
        "gl": [q, int(alt_cl.df_resid)],
        "p": p_f,
        "F_robusto": f_rob,
        "p_robusto": p_rob,
        "n_bonferroni": int(sum(bool(r["atipico"]) for r in indicators)),
        "n_significativas": int(sum(float(r["p"]) < alpha for r in indicators)),  # type: ignore[arg-type]
        "indicators": indicators,
        "changes": changes,
        "max_cambio_se": max_change,
        "cambian": flips,
        "aic_mejora": float(alt_cl.aic) < float(base_cl.aic),
        "bic_mejora": float(alt_cl.bic) < float(base_cl.bic),
        "coef": {c: float(alt.params[c]) for c in cols},
        "se": {c: float(alt.bse[c]) for c in cols},
        "lower": {c: float(ci.loc[c, 0]) for c in cols},
        "upper": {c: float(ci.loc[c, 1]) for c in cols},
        "pvalues": {c: float(alt.pvalues[c]) for c in cols},
    }


def _fit_cov(design: Design, cols: list[str], weights: np.ndarray | None, cov: str):  # type: ignore[no-untyped-def]
    model = (
        sm.WLS(design.y, design.X[cols], weights=weights)
        if weights is not None
        else sm.OLS(design.y, design.X[cols])
    )
    if cov == "cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": design.groups}, use_t=True)
    if cov == "HC3":
        return model.fit(cov_type="HC3", use_t=True)
    return model.fit()


def _cook_blocks(
    e: np.ndarray, Q: np.ndarray, h: np.ndarray, top: int
) -> tuple[tuple[int, float], float]:
    """Repite la elección por Cook y el F de las indicadoras sobre unos residuos MCO.

    Devuelve (número de condados con Cook > 4/n, F de sus indicadoras) y el F de las
    indicadoras de los ``top`` condados de mayor Cook. Usa las fórmulas de eliminación: la
    suma de cuadrados sin el conjunto S es SCE − e_Sᵀ (I − H_SS)⁻¹ e_S, sin reajustar.
    """
    n, p = Q.shape
    sse = float(e @ e)
    s2 = sse / (n - p)
    cook = e**2 * h / (p * s2 * (1 - h) ** 2)

    def block(S: np.ndarray) -> float:
        k = S.size
        if k == 0:
            return 0.0
        QS = Q[S]
        eS = e[S]
        red = float(eS @ np.linalg.solve(np.eye(k) - QS @ QS.T, eS))
        return (red / k) / ((sse - red) / (n - p - k))

    flagged = np.flatnonzero(cook > 4 / n)
    f_top = block(np.argsort(-cook)[:top]) if top > 0 else 0.0
    return (flagged.size, block(flagged)), f_top


def calibrate_cook_blocks(
    design: Design,
    terms: list[str],
    weights: np.ndarray | None,
    top: int,
    reps: int,
    seed: int,
) -> Record:
    """Distribución nula de los contrastes de indicadoras elegidas por la distancia de Cook.

    Se simulan respuestas sin ningún atípico añadido (y* = Xβ̂ + ε*) con dos leyes del error:
    normal con la σ̂ del modelo y bootstrap de los residuos (conserva sus colas pesadas). En
    cada réplica se recalcula Cook, se eligen los condados igual que con los datos (los
    ``top`` de mayor Cook, y los que superan 4/n) y se calcula el F de sus indicadoras. La
    proporción de réplicas que iguala o supera lo observado es el valor p calibrado.
    """
    cols = design.columns_for(terms)
    sw = np.ones(design.n) if weights is None else np.sqrt(weights)
    xs = design.X[cols].to_numpy() * sw[:, None]
    ys = design.y.to_numpy() * sw
    Q, _ = np.linalg.qr(xs)
    h = (Q**2).sum(axis=1)
    e0 = ys - Q @ (Q.T @ ys)
    n, p = Q.shape
    (k_obs, f_obs), ftop_obs = _cook_blocks(e0, Q, h, top)
    sigma = float(np.sqrt(e0 @ e0 / (n - p)))
    rng = np.random.default_rng(seed)
    sims: dict[str, list[tuple[int, float, float]]] = {"normal": [], "bootstrap": []}
    for _ in range(reps):
        for name in sims:
            eps = rng.normal(0.0, sigma, n) if name == "normal" else rng.choice(e0, n)
            (k, f), ftop = _cook_blocks(eps - Q @ (Q.T @ eps), Q, h, top)
            sims[name].append((k, f, ftop))
    out: Record = {
        "reps": reps,
        "top": top,
        "k_obs": k_obs,
        "F_obs": f_obs,
        "F_top_obs": ftop_obs,
        "cut": 4 / n,
    }
    for name, rows in sims.items():
        arr = np.asarray(rows, dtype=float)

        def p_ge(col: int, obs: float, a: np.ndarray = arr) -> float:
            return float((1 + (a[:, col] >= obs).sum()) / (reps + 1))

        out[name] = {
            "k_media": float(arr[:, 0].mean()),
            "k_lower": float(np.quantile(arr[:, 0], 0.025)),
            "k_upper": float(np.quantile(arr[:, 0], 0.975)),
            "F_media": float(arr[:, 1].mean()),
            "F_lower": float(np.quantile(arr[:, 1], 0.025)),
            "F_upper": float(np.quantile(arr[:, 1], 0.975)),
            "F_top_media": float(arr[:, 2].mean()),
            "F_top_lower": float(np.quantile(arr[:, 2], 0.025)),
            "F_top_upper": float(np.quantile(arr[:, 2], 0.975)),
            "p_k": p_ge(0, k_obs),
            "p_F": p_ge(1, f_obs),
            "p_F_top": p_ge(2, ftop_obs),
        }
    return out


def bonferroni_outliers(
    design: Design, stud: np.ndarray, p: int, alpha: float, cook: np.ndarray
) -> Record:
    """Contraste de atípicos de Bonferroni sobre los residuos estudentizados externamente.

    El residuo estudentizado externamente r*_i es el t de la indicadora del condado i cuando
    se añade sola; sigue una t de Student con n − p − 1 g.l. Como se examinan los n condados,
    se rechaza si |r*_i| supera el cuantil 1 − α/(2n).
    """
    n = design.n
    df = n - p - 1
    crit = float(stats.t.ppf(1 - alpha / (2 * n), df))
    pv = 2 * stats.t.sf(np.abs(stud), df)
    idx = np.flatnonzero(np.abs(stud) > crit)
    idx = idx[np.argsort(-np.abs(stud[idx]))]
    return {
        "critico": crit,
        "gl": df,
        "rows": idx.tolist(),
        "outliers": [
            {
                "county_id": design.county[i],
                "region": str(design.region[i]),
                "y": float(design.y.iloc[i]),
                "stud_resid": float(stud[i]),
                "p": float(pv[i]),
                "p_bonferroni": float(min(1.0, pv[i] * n)),
                "cook": float(cook[i]),
            }
            for i in idx
        ],
    }


def verdict(model: Record, alpha: float, p_joint: float | None = None) -> Record:
    """¿Conviene excluir del ajuste a esos condados (o, lo que es igual, mantener sus indicadoras)?

    Sólo si el grupo es atípico (F conjunto significativo, calibrado si la selección lo exige)
    *y* su exclusión cambia alguna conclusión (signo o significación de algún coeficiente).
    Si son atípicos pero las conclusiones no cambian, son condados reales que el modelo
    explica peor, y se mantienen: excluirlos sólo maquillaría el ajuste.
    """
    p = float(model["p"]) if p_joint is None else p_joint  # type: ignore[arg-type]
    atypical = p < alpha
    changes = bool(model["cambian"])
    return {
        "atipicos": atypical,
        "p_conjunto": p,
        "cambian_conclusiones": changes,
        "vale_la_pena": bool(atypical and changes),
    }
