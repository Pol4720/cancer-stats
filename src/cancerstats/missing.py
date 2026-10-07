"""Datos ausentes: mecanismo, contraste MCAR de Little e imputación múltiple.

La pregunta que decide el tratamiento no es *cuántos* datos faltan sino *por qué*:

* MCAR (completamente al azar): la ausencia no depende de nada; los casos completos son
  una submuestra aleatoria y el análisis de casos completos es insesgado.
* MAR (al azar dadas las observadas): la ausencia depende de variables observadas; la
  imputación múltiple que las incluya corrige el sesgo.
* MNAR: depende del propio valor ausente; ningún método lo resuelve sin supuestos externos.

En regresión, el análisis de casos completos sigue siendo insesgado aunque la ausencia no
sea MCAR, siempre que no dependa de la respuesta una vez fijadas las explicativas (Little,
1992); ese es el caso de una supresión administrativa por estado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------------------------------- Little (1988)
@dataclass
class LittleResult:
    """Resultado del contraste MCAR de Little."""

    statistic: float
    df: int
    p_value: float
    n_patterns: int
    n_rows: int
    variables: list[str]
    em_iterations: int
    converged: bool

    def to_dict(self) -> dict[str, object]:
        """Representación serializable."""
        return dict(self.__dict__)


def em_mvn(
    x: np.ndarray, max_iter: int = 500, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Estimación de máxima verosimilitud de (μ, Σ) normal multivariante con ausentes (EM).

    Args:
        x: matriz n × p con ``nan`` en las celdas ausentes (sin filas totalmente vacías).

    Returns:
        μ, Σ (de máxima verosimilitud, divisor n), iteraciones y si convergió.
    """
    n, p = x.shape
    obs = ~np.isnan(x)
    mu = np.nanmean(x, axis=0)
    sigma = np.diag(np.nanvar(x, axis=0))
    patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 - se devuelve el número de iteraciones
        t1 = np.zeros(p)
        t2 = np.zeros((p, p))
        for k, pat in enumerate(patterns):
            rows = x[inverse == k]
            o = np.flatnonzero(pat)
            m = np.flatnonzero(~pat)
            yo = rows[:, o]
            if m.size == 0:
                t1 += yo.sum(axis=0)
                t2 += yo.T @ yo
                continue
            s_oo = sigma[np.ix_(o, o)]
            s_mo = sigma[np.ix_(m, o)]
            beta = np.linalg.solve(s_oo, s_mo.T).T  # Σ_mo Σ_oo⁻¹
            ym = mu[m] + (yo - mu[o]) @ beta.T
            cond = sigma[np.ix_(m, m)] - beta @ s_mo.T
            full = np.empty((rows.shape[0], p))
            full[:, o] = yo
            full[:, m] = ym
            t1 += full.sum(axis=0)
            t2 += full.T @ full
            t2[np.ix_(m, m)] += rows.shape[0] * cond
        mu_new = t1 / n
        sigma_new = t2 / n - np.outer(mu_new, mu_new)
        delta = max(np.max(np.abs(mu_new - mu)), np.max(np.abs(sigma_new - sigma)))
        mu, sigma = mu_new, sigma_new
        if delta < tol:
            converged = True
            break
    return mu, sigma, it, converged


def little_mcar_test(df: pd.DataFrame, columns: list[str]) -> LittleResult:
    """Contraste MCAR de Little (1988).

    Estadístico: d² = Σ_j n_j (ȳ_j − μ̂_j)ᵀ Σ̂_j⁻¹ (ȳ_j − μ̂_j), sumando sobre los patrones de
    ausencia j, con (μ̂, Σ̂) estimados por EM bajo normalidad. Bajo MCAR, d² ~ χ² con
    Σ_j p_j − p grados de libertad, siendo p_j el número de variables observadas en j.
    """
    data = df[columns].astype(float)
    data = data.loc[data.notna().any(axis=1)]
    # Tipificar no cambia el estadístico (es invariante afín) y mejora el condicionamiento.
    z = ((data - data.mean()) / data.std(ddof=0)).to_numpy()
    mu, sigma, iters, converged = em_mvn(z)
    obs = ~np.isnan(z)
    patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    d2 = 0.0
    dof = 0
    for k, pat in enumerate(patterns):
        o = np.flatnonzero(pat)
        rows = z[inverse == k][:, o]
        diff = rows.mean(axis=0) - mu[o]
        d2 += rows.shape[0] * float(diff @ np.linalg.solve(sigma[np.ix_(o, o)], diff))
        dof += o.size
    dof -= z.shape[1]
    p_value = float(stats.chi2.sf(d2, dof)) if dof > 0 else float("nan")
    return LittleResult(
        statistic=float(d2),
        df=int(dof),
        p_value=p_value,
        n_patterns=len(patterns),
        n_rows=int(z.shape[0]),
        variables=list(columns),
        em_iterations=iters,
        converged=converged,
    )


# --------------------------------------------------------------------------- descripción
@dataclass
class MissingReport:
    """Diagnóstico completo de los datos ausentes."""

    by_column: list[dict[str, object]]
    patterns: list[dict[str, object]]
    comparisons: list[dict[str, object]]
    by_state: list[dict[str, object]]
    little_random: LittleResult | None
    little_all: LittleResult | None
    mechanism: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Representación serializable."""
        return {
            "by_column": self.by_column,
            "patterns": self.patterns,
            "comparisons": self.comparisons,
            "by_state": self.by_state,
            "little_random": self.little_random.to_dict() if self.little_random else None,
            "little_all": self.little_all.to_dict() if self.little_all else None,
            "mechanism": self.mechanism,
        }


def missing_by_column(df: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    """Recuento y porcentaje de ausentes por columna."""
    out = []
    for col in columns:
        n = int(df[col].isna().sum())
        if n:
            out.append({"variable": col, "n": n, "pct": round(100 * n / len(df), 2)})
    return sorted(out, key=lambda r: -int(r["n"]))  # type: ignore[call-overload]


def missing_patterns(
    df: pd.DataFrame, columns: list[str], top: int = 10
) -> list[dict[str, object]]:
    """Patrones de ausencia más frecuentes (qué variables faltan juntas)."""
    cols = [c for c in columns if df[c].isna().any()]
    if not cols:
        return []
    keys = df[cols].isna().apply(lambda r: tuple(c for c in cols if r[c]), axis=1)
    counts = keys.value_counts().head(top)
    return [
        {"faltan": list(k) if k else [], "n": int(n), "pct": round(100 * n / len(df), 2)}
        for k, n in counts.items()
    ]


def compare_missing(
    df: pd.DataFrame, target_cols: list[str], other_cols: list[str], method: str = "holm"
) -> list[dict[str, object]]:
    """Compara las demás variables entre filas con y sin ausencia (Welch y Mann-Whitney).

    Si la ausencia fuera MCAR, las distribuciones de las demás variables no deberían diferir
    entre ambos grupos. Se informa la diferencia de medias tipificada (tamaño del efecto) y
    los p-valores corregidos por comparaciones múltiples.
    """
    rows: list[dict[str, object]] = []
    for tcol in target_cols:
        miss = df[tcol].isna()
        if miss.sum() < 3 or (~miss).sum() < 3:
            continue
        for ocol in other_cols:
            if ocol == tcol:
                continue
            a = df.loc[miss, ocol].dropna()
            b = df.loc[~miss, ocol].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            smd = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
            rows.append(
                {
                    "con_ausencia": tcol,
                    "variable": ocol,
                    "media_ausentes": float(a.mean()),
                    "media_observados": float(b.mean()),
                    "dif_tipificada": smd,
                    "p_welch": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
                    "p_mann_whitney": float(stats.mannwhitneyu(a, b).pvalue),
                }
            )
    if rows:
        adj = multipletests([float(r["p_welch"]) for r in rows], method=method)[1]  # type: ignore[arg-type]
        for r, p in zip(rows, adj, strict=True):
            r["p_welch_ajustado"] = float(p)
    return rows


def missing_by_state(df: pd.DataFrame, target_cols: list[str]) -> list[dict[str, object]]:
    """Contraste χ² de independencia entre la ausencia y el estado (y la región)."""
    out = []
    for tcol in target_cols:
        miss = df[tcol].isna()
        if miss.sum() == 0:
            continue
        for group in ("state", "region"):
            table = pd.crosstab(df[group], miss)
            if table.shape[1] < 2:
                continue
            chi2, p, dof, expected = stats.chi2_contingency(table)
            share_small = float((expected < 5).mean())
            concentration = (
                df.loc[miss, group].value_counts().head(3).to_dict() if group == "state" else {}
            )
            out.append(
                {
                    "variable": tcol,
                    "agrupacion": group,
                    "chi2": float(chi2),
                    "gl": int(dof),
                    "p": float(p),
                    "pct_esperadas_menores_5": round(100 * share_small, 1),
                    "principales": {str(k): int(v) for k, v in concentration.items()},
                }
            )
    return out


def analyze_missing(
    df: pd.DataFrame,
    numeric: list[str],
    structural: list[str],
    alpha: float = 0.05,
    method: str = "holm",
) -> MissingReport:
    """Diagnóstico de ausentes.

    Args:
        df: datos depurados.
        numeric: variables numéricas que intervienen en el análisis.
        structural: variables cuya ausencia es estructural (supresión por estado); se excluyen
            del contraste de Little «aleatorio» y se analizan aparte.
    """
    with_missing = [c for c in numeric if df[c].isna().any()]
    random_missing = [c for c in with_missing if c not in structural]
    by_col = missing_by_column(df, numeric)
    patterns = missing_patterns(df, numeric)
    comparisons = compare_missing(
        df, with_missing, [c for c in numeric if c not in with_missing], method
    )
    by_state = missing_by_state(df, with_missing)

    complete_vars = [c for c in numeric if c not in with_missing]
    little_random = little_mcar_test(df, complete_vars + random_missing) if random_missing else None
    little_all = little_mcar_test(df, complete_vars + with_missing) if with_missing else None

    mechanism: dict[str, str] = {}
    for col in with_missing:
        if col in structural:
            mechanism[col] = (
                "Ausencia estructural: suprimida por el registro de estados completos. No "
                "depende del valor (MNAR descartado por diseño) pero sí del estado: MAR dado el "
                "estado."
            )
            continue
        rows = [r for r in comparisons if r["con_ausencia"] == col]
        significant = [r for r in rows if float(r["p_welch_ajustado"]) < alpha]  # type: ignore[arg-type]
        state_p = next(
            (
                float(r["p"])
                for r in by_state
                if r["variable"] == col and r["agrupacion"] == "state"
            ),  # type: ignore[arg-type]
            float("nan"),
        )
        verdict = "compatible con MCAR" if not significant and state_p >= alpha else "no MCAR"
        mechanism[col] = (
            f"{verdict}: {len(significant)} de {len(rows)} comparaciones significativas tras "
            f"corregir; χ² por estado p = {state_p:.3f}."
        )
    return MissingReport(
        by_column=by_col,
        patterns=patterns,
        comparisons=comparisons,
        by_state=by_state,
        little_random=little_random,
        little_all=little_all,
        mechanism=mechanism,
    )


# --------------------------------------------------------------------------- imputación
def multiple_imputation(
    df: pd.DataFrame,
    columns: list[str],
    auxiliary: list[str],
    m: int,
    max_iter: int,
    seed: int,
) -> list[pd.DataFrame]:
    """Imputación múltiple por ecuaciones encadenadas con extracciones de la posterior.

    Cada imputación usa ``sample_posterior=True`` (regresión bayesiana), de modo que la
    variabilidad entre imputaciones refleja la incertidumbre del valor ausente; sin ella,
    las reglas de Rubin subestimarían la varianza. La respuesta se incluye entre las
    variables auxiliares, como recomienda van Buuren (2018): omitirla atenuaría las
    asociaciones de las variables imputadas con ella.
    """
    cols = list(dict.fromkeys(columns + auxiliary))
    base = df[cols].astype(float)
    out = []
    for i in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=max_iter,
            random_state=seed + i,
            skip_complete=True,
        )
        filled = pd.DataFrame(imputer.fit_transform(base), columns=cols, index=df.index)
        res = df.copy()
        res[columns] = filled[columns]
        out.append(res)
    return out


@dataclass
class RubinPooled:
    """Estimaciones combinadas por las reglas de Rubin."""

    params: pd.Series
    se: pd.Series
    df: pd.Series
    fmi: pd.Series
    within: pd.Series
    between: pd.Series


def rubin_pool(
    estimates: list[pd.Series], variances: list[pd.Series], df_complete: float
) -> RubinPooled:
    """Combina m estimaciones con las reglas de Rubin (1987).

    Q̄ = media de las estimaciones; Ū = media de las varianzas; B = varianza entre
    imputaciones; T = Ū + (1 + 1/m) B. Los grados de libertad siguen la corrección para
    muestras finitas de Barnard y Rubin (1999).
    """
    m = len(estimates)
    q = pd.concat(estimates, axis=1)
    u = pd.concat(variances, axis=1)
    qbar = q.mean(axis=1)
    ubar = u.mean(axis=1)
    b = q.var(axis=1, ddof=1)
    t = ubar + (1 + 1 / m) * b
    lam = ((1 + 1 / m) * b / t).clip(lower=1e-12, upper=1 - 1e-12)
    nu_old = (m - 1) / lam**2
    nu_obs = (df_complete + 1) / (df_complete + 3) * df_complete * (1 - lam)
    nu = 1 / (1 / nu_old + 1 / nu_obs)
    r = (1 + 1 / m) * b / ubar
    fmi = (r + 2 / (nu + 3)) / (r + 1)
    return RubinPooled(params=qbar, se=np.sqrt(t), df=nu, fmi=fmi, within=ubar, between=b)
