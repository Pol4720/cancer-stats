"""Herramientas de inferencia clásica del curso, implementadas como funciones puras.

Estimación puntual y por intervalos (cantidad pivotal y bootstrap), contrastes de
normalidad y bondad de ajuste, contrastes paramétricos y no paramétricos de dos o más
muestras, contraste de rachas y correlaciones con su intervalo de confianza. Cada función
devuelve diccionarios serializables para alimentar la interfaz y el informe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

Record = dict[str, object]


# ----------------------------------------------------------------------- estimación
def mean_ci(x: Sequence[float] | np.ndarray, alpha: float = 0.05) -> Record:
    """Intervalo para μ con σ desconocida: x̄ ± t_{n−1, α/2} S/√n (cantidad pivotal t)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    m = float(a.mean())
    s = float(a.std(ddof=1))
    q = float(stats.t.ppf(1 - alpha / 2, n - 1))
    half = q * s / math.sqrt(n)
    return {"estimate": m, "lower": m - half, "upper": m + half, "se": s / math.sqrt(n), "n": n}


def variance_ci(x: Sequence[float] | np.ndarray, alpha: float = 0.05) -> Record:
    """Intervalo para σ² con el pivote (n − 1) S²/σ² ~ χ²_{n−1} (exige normalidad)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    s2 = float(a.var(ddof=1))
    lo = (n - 1) * s2 / float(stats.chi2.ppf(1 - alpha / 2, n - 1))
    hi = (n - 1) * s2 / float(stats.chi2.ppf(alpha / 2, n - 1))
    return {
        "estimate": s2,
        "lower": lo,
        "upper": hi,
        "sd": math.sqrt(s2),
        "sd_lower": math.sqrt(lo),
        "sd_upper": math.sqrt(hi),
        "n": n,
    }


def median_ci_order(x: Sequence[float] | np.ndarray, alpha: float = 0.05) -> Record:
    """Intervalo de libre distribución para la mediana basado en estadísticos de orden.

    Usa que el número de observaciones por debajo de la mediana es B(n, 1/2): el intervalo
    [x_(j), x_(k)] tiene cobertura ≥ 1 − α sin suponer ninguna distribución.
    """
    a = np.sort(np.asarray(x, dtype=float)[~np.isnan(np.asarray(x, dtype=float))])
    n = a.size
    j = max(int(stats.binom.ppf(alpha / 2, n, 0.5)), 1)
    k = min(n - j + 1, n)
    coverage = float(stats.binom.cdf(k - 1, n, 0.5) - stats.binom.cdf(j - 1, n, 0.5))
    return {
        "estimate": float(np.median(a)),
        "lower": float(a[j - 1]),
        "upper": float(a[k - 1]),
        "coverage": coverage,
        "n": n,
    }


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    statistic: str = "median",
    reps: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Record:
    """Intervalo bootstrap percentil para la media, la mediana o la media recortada."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(reps, a.size))
    samples = a[idx]
    if statistic == "median":
        boot = np.median(samples, axis=1)
        est = float(np.median(a))
    elif statistic == "mean":
        boot = samples.mean(axis=1)
        est = float(a.mean())
    elif statistic == "trimmed_mean":
        boot = stats.trim_mean(samples, 0.1, axis=1)
        est = float(stats.trim_mean(a, 0.1))
    else:  # pragma: no cover - protegido por los llamadores
        raise ValueError(statistic)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": est,
        "lower": float(lo),
        "upper": float(hi),
        "se": float(boot.std(ddof=1)),
        "reps": reps,
        "statistic": statistic,
    }


def robust_location(x: Sequence[float] | np.ndarray) -> Record:
    """Estimadores robustos del curso: mediana, media recortada, MEDA y M-estimador de Huber."""
    from statsmodels.robust.norms import HuberT
    from statsmodels.robust.robust_linear_model import RLM
    from statsmodels.robust.scale import mad

    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    huber = RLM(a, np.ones_like(a), M=HuberT()).fit()
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "trimmed_mean_10": float(stats.trim_mean(a, 0.1)),
        "meda": float(np.median(np.abs(a - np.median(a)))),
        "mad_normalized": float(mad(a)),
        "huber": float(huber.params[0]),
    }


# ----------------------------------------------------------------------- normalidad
def chi2_gof_normal(x: Sequence[float] | np.ndarray, k: int | None = None) -> Record:
    """Contraste χ² de Pearson de normalidad con k clases equiprobables.

    Se estiman μ y σ (r = 2 parámetros), de modo que los grados de libertad son k − 1 − 2.
    Por defecto k = ⌈2 n^{2/5}⌉ (regla de Mann y Wald moderada), lo que garantiza E_i ≥ 5.
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if k is None:
        k = max(4, math.ceil(2 * n**0.4))
    mu, sd = a.mean(), a.std(ddof=1)
    edges = stats.norm.ppf(np.linspace(0, 1, k + 1), loc=mu, scale=sd)
    observed = np.histogram(a, bins=edges)[0]
    expected = np.full(k, n / k)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = k - 1 - 2
    return {
        "statistic": chi2,
        "df": dof,
        "p": float(stats.chi2.sf(chi2, dof)),
        "k": k,
        "min_expected": float(expected.min()),
    }


def normality_battery(x: Sequence[float] | np.ndarray) -> list[Record]:
    """Batería de contrastes de normalidad (los de la diapositiva del curso y alguno más)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    out: list[Record] = []
    if a.size <= 5000:
        sw = stats.shapiro(a)
        out.append(
            {"test": "Shapiro-Wilk", "statistic": float(sw.statistic), "p": float(sw.pvalue)}
        )
    lil = lilliefors(a, dist="norm")
    out.append(
        {"test": "Kolmogorov-Smirnov-Lilliefors", "statistic": float(lil[0]), "p": float(lil[1])}
    )
    k2 = stats.normaltest(a)
    out.append(
        {"test": "D'Agostino-Pearson K²", "statistic": float(k2.statistic), "p": float(k2.pvalue)}
    )
    jb = stats.jarque_bera(a)
    out.append({"test": "Jarque-Bera", "statistic": float(jb.statistic), "p": float(jb.pvalue)})
    try:
        ad = stats.anderson(a, dist="norm", method="interpolate")
        out.append(
            {"test": "Anderson-Darling", "statistic": float(ad.statistic), "p": float(ad.pvalue)}
        )
    except TypeError:  # pragma: no cover - SciPy < 1.17 no admite «method»
        ad_old = stats.anderson(a, dist="norm")
        out.append({"test": "Anderson-Darling", "statistic": float(ad_old.statistic), "p": None})
    gof = chi2_gof_normal(a)
    out.append(
        {
            "test": "χ² de Pearson (clases equiprobables)",
            "statistic": gof["statistic"],
            "p": gof["p"],
            "df": gof["df"],
            "k": gof["k"],
        }
    )
    return out


def shape(x: Sequence[float] | np.ndarray) -> Record:
    """Asimetría y curtosis (de exceso) con sus contrastes."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    sk = stats.skewtest(a)
    ku = stats.kurtosistest(a)
    return {
        "skewness": float(stats.skew(a)),
        "kurtosis_excess": float(stats.kurtosis(a)),
        "p_skew": float(sk.pvalue),
        "p_kurtosis": float(ku.pvalue),
    }


def boxcox_lambda(x: Sequence[float] | np.ndarray, alpha: float = 0.05) -> Record:
    """λ de Box-Cox por máxima verosimilitud con su intervalo de confianza (perfil)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    a = a[a > 0]
    _, lam, ci = stats.boxcox(a, alpha=alpha)
    return {
        "lambda": float(lam),
        "lower": float(ci[0]),
        "upper": float(ci[1]),
        "includes_1": bool(ci[0] <= 1 <= ci[1]),
        "includes_0": bool(ci[0] <= 0 <= ci[1]),
    }


def qq_points(x: Sequence[float] | np.ndarray, max_points: int = 600) -> Record:
    """Puntos del gráfico Q-Q normal (submuestreados uniformemente en los cuantiles)."""
    a = np.sort(np.asarray(x, dtype=float)[~np.isnan(np.asarray(x, dtype=float))])
    n = a.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.norm.ppf(probs)
    if n > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
        a, theo = a[idx], theo[idx]
    mu, sd = float(np.mean(a)), float(np.std(a, ddof=1))
    return {"theoretical": theo.tolist(), "sample": a.tolist(), "mean": mu, "sd": sd}


# ----------------------------------------------------------------------- independencia
def runs_test(x: Sequence[float] | np.ndarray, cutoff: float | None = None) -> Record:
    """Contraste de rachas de Wald-Wolfowitz respecto de la mediana.

    Un número de rachas demasiado pequeño indica que valores parecidos aparecen juntos en
    el orden de registro (dependencia); demasiado grande, alternancia sistemática.
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    c = float(np.median(a)) if cutoff is None else cutoff
    signs = a[a != c] > c
    n1 = int(signs.sum())
    n2 = int(signs.size - n1)
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / math.sqrt(var)
    return {
        "runs": runs,
        "expected": mu,
        "z": z,
        "p": float(2 * stats.norm.sf(abs(z))),
        "n_above": n1,
        "n_below": n2,
    }


# ----------------------------------------------------------------------- dos muestras
def two_sample(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    labels: tuple[str, str] = ("A", "B"),
) -> Record:
    """Comparación completa de dos muestras independientes.

    Contraste F de igualdad de varianzas, t de Student (varianzas iguales), t de Welch,
    U de Mann-Whitney y Kolmogorov-Smirnov de dos muestras, más la d de Cohen.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    f = v1 / v2
    pf = 2 * min(stats.f.cdf(f, x.size - 1, y.size - 1), stats.f.sf(f, x.size - 1, y.size - 1))
    student = stats.ttest_ind(x, y, equal_var=True)
    welch = stats.ttest_ind(x, y, equal_var=False)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided")
    ks = stats.ks_2samp(x, y)
    sp = math.sqrt(((x.size - 1) * v1 + (y.size - 1) * v2) / (x.size + y.size - 2))
    diff = float(x.mean() - y.mean())
    se_w = math.sqrt(v1 / x.size + v2 / y.size)
    df_w = (v1 / x.size + v2 / y.size) ** 2 / (
        (v1 / x.size) ** 2 / (x.size - 1) + (v2 / y.size) ** 2 / (y.size - 1)
    )
    q = float(stats.t.ppf(0.975, df_w))
    return {
        "labels": list(labels),
        "n": [int(x.size), int(y.size)],
        "mean": [float(x.mean()), float(y.mean())],
        "median": [float(np.median(x)), float(np.median(y))],
        "sd": [float(math.sqrt(v1)), float(math.sqrt(v2))],
        "diff": diff,
        "diff_ci_welch": [diff - q * se_w, diff + q * se_w],
        "f_test": {"statistic": float(f), "p": float(pf)},
        "student": {"statistic": float(student.statistic), "p": float(student.pvalue)},
        "welch": {"statistic": float(welch.statistic), "p": float(welch.pvalue), "df": float(df_w)},
        "mann_whitney": {"statistic": float(mw.statistic), "p": float(mw.pvalue)},
        "ks": {"statistic": float(ks.statistic), "p": float(ks.pvalue)},
        "cohen_d": diff / sp if sp > 0 else 0.0,
    }


# ----------------------------------------------------------------------- k muestras
def welch_anova(groups: list[np.ndarray]) -> Record:
    """ANOVA de Welch (no supone varianzas iguales)."""
    k = len(groups)
    n = np.array([g.size for g in groups], dtype=float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    mw = float((w * m).sum() / w.sum())
    a = float((w * (m - mw) ** 2).sum() / (k - 1))
    tmp = float(((1 - w / w.sum()) ** 2 / (n - 1)).sum())
    b = 1 + 2 * (k - 2) / (k**2 - 1) * tmp
    f = a / b
    df2 = (k**2 - 1) / (3 * tmp)
    return {"statistic": f, "df1": k - 1, "df2": float(df2), "p": float(stats.f.sf(f, k - 1, df2))}


def k_sample(
    df: pd.DataFrame, value: str, group: str, alpha: float = 0.05, correction: str = "holm"
) -> Record:
    """Comparación de una variable entre k grupos.

    ANOVA de un factor (con η²), Levene/Brown-Forsythe para la homogeneidad de varianzas,
    ANOVA de Welch, Kruskal-Wallis, Tukey HSD y Mann-Whitney por pares con corrección.
    """
    data = df[[value, group]].dropna()
    names = list(data.groupby(group, observed=True).groups)
    groups = [data.loc[data[group] == g, value].to_numpy(dtype=float) for g in names]
    anova = stats.f_oneway(*groups)
    grand = data[value].mean()
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((data[value] - grand) ** 2).sum())
    levene = stats.levene(*groups, center="median")
    kw = stats.kruskal(*groups)
    tukey = pairwise_tukeyhsd(
        data[value].to_numpy(dtype=float), data[group].astype(str).to_numpy(), alpha=alpha
    )
    tk = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    pairs: list[Record] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mw = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            pairs.append(
                {
                    "a": str(names[i]),
                    "b": str(names[j]),
                    "diff": float(groups[i].mean() - groups[j].mean()),
                    "p_mann_whitney": float(mw.pvalue),
                }
            )
    adj = multipletests([float(p["p_mann_whitney"]) for p in pairs], method=correction)[1]  # type: ignore[arg-type]
    for p, q in zip(pairs, adj, strict=True):
        p["p_mann_whitney_adj"] = float(q)
        row = tk[
            ((tk.group1 == p["a"]) & (tk.group2 == p["b"]))
            | ((tk.group1 == p["b"]) & (tk.group2 == p["a"]))
        ]
        if len(row):
            p["p_tukey"] = float(row["p-adj"].iloc[0])
            p["tukey_lower"] = float(row["lower"].iloc[0])
            p["tukey_upper"] = float(row["upper"].iloc[0])
            # Tukey informa media(group2) − media(group1); se expresa como media(a) − media(b).
            if str(row["group1"].iloc[0]) == p["a"]:
                p["tukey_lower"], p["tukey_upper"] = -p["tukey_upper"], -p["tukey_lower"]  # type: ignore[operator]
    descr = []
    for name, g in zip(names, groups, strict=True):
        ci = mean_ci(g, alpha)
        descr.append(
            {
                "group": str(name),
                "n": int(g.size),
                "mean": float(g.mean()),
                "sd": float(g.std(ddof=1)),
                "median": float(np.median(g)),
                "lower": ci["lower"],
                "upper": ci["upper"],
            }
        )
    return {
        "value": value,
        "group": group,
        "groups": descr,
        "anova": {
            "statistic": float(anova.statistic),
            "p": float(anova.pvalue),
            "df1": len(groups) - 1,
            "df2": int(data.shape[0] - len(groups)),
            "eta2": float(ss_between / ss_total),
        },
        "levene": {"statistic": float(levene.statistic), "p": float(levene.pvalue)},
        "welch": welch_anova(groups),
        "kruskal": {"statistic": float(kw.statistic), "p": float(kw.pvalue)},
        "pairs": pairs,
    }


# ----------------------------------------------------------------------- correlación
def correlation_test(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, alpha: float = 0.05
) -> Record:
    """Correlación de Pearson con IC por la transformación z de Fisher, y de Spearman."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = a.size
    r, p = stats.pearsonr(a, b)
    z = math.atanh(float(r))
    half = float(stats.norm.ppf(1 - alpha / 2)) / math.sqrt(n - 3)
    rho, p_s = stats.spearmanr(a, b)
    return {
        "n": n,
        "pearson": float(r),
        "pearson_lower": math.tanh(z - half),
        "pearson_upper": math.tanh(z + half),
        "p_pearson": float(p),
        "spearman": float(rho),
        "p_spearman": float(p_s),
        "r2": float(r) ** 2,
    }


def adjust(pvalues: Sequence[float], method: str) -> list[float]:
    """Corrección por comparaciones múltiples."""
    return [float(v) for v in multipletests(list(pvalues), method=method)[1]]
