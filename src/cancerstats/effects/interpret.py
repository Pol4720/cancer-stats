"""Interpretación de los coeficientes en el lenguaje del problema.

Un coeficiente de regresión múltiple es el cambio esperado en la mortalidad por unidad de
la explicativa *manteniendo constantes las demás variables del modelo*. Para comparar
variables con escalas distintas se informa también el efecto de pasar del primer al tercer
cuartil (rango intercuartílico) de cada una. Cuando una variable interacciona con la región,
su efecto es distinto en cada región y se calcula como combinación lineal de coeficientes,
con su varianza Var(β₁ + β₂) = Var(β₁) + Var(β₂) + 2·Cov(β₁, β₂).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cancerstats.dictionary import label, unit
from cancerstats.effects.design import REGION, Design, dummy_name, region_levels
from cancerstats.effects.fit import df_denominator

Record = dict[str, object]


def _lincomb(res, weights: dict[str, float], q: float) -> tuple[float, float, float, float, float]:  # type: ignore[no-untyped-def]
    names = list(res.params.index)
    vec = np.zeros(len(names))
    for k, v in weights.items():
        vec[names.index(k)] = v
    est = float(vec @ res.params.to_numpy())
    se = float(np.sqrt(vec @ np.asarray(res.cov_params()) @ vec))
    dfd = df_denominator(res)
    t = est / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), dfd))
    return est, se, est - q * se, est + q * se, p


def _per_unit_text(var: str) -> str:
    u = unit(var)
    if var.startswith("log"):
        return "un aumento del 10 %"
    if u == "%":
        return "un punto porcentual más"
    return f"una unidad más ({u})" if u else "una unidad más"


def effects_table(  # type: ignore[no-untyped-def]
    res,
    design: Design,
    df: pd.DataFrame,
    terms: list[str],
    alpha: float,
) -> list[Record]:
    """Efecto de cada explicativa continua: por unidad, por 10 % (logaritmos) y por IQR."""
    q = float(stats.t.ppf(1 - alpha / 2, df_denominator(res)))
    out: list[Record] = []
    region_inter = {t.split(":")[0] for t in terms if t.endswith(f":{REGION}")}
    for var in [t for t in terms if t in design.numeric]:
        x = df[var].dropna() if var in df else design.X[var] + design.centers[var]
        iqr = float(x.quantile(0.75) - x.quantile(0.25))
        scale = np.log(1.1) if var.startswith("log") else 1.0
        if var in region_inter:
            slopes = []
            levels = [design.reference, *region_levels(design.reference)]
            for lev in levels:
                wts = {var: 1.0}
                if lev != design.reference:
                    wts[f"{var}:{dummy_name(lev)}"] = 1.0
                est, se, lo, hi, p = _lincomb(res, wts, q)
                slopes.append(
                    {
                        "region": lev,
                        "coef": est,
                        "se": se,
                        "lower": lo,
                        "upper": hi,
                        "p": p,
                        "iqr_effect": est * iqr,
                        "iqr_lower": lo * iqr,
                        "iqr_upper": hi * iqr,
                    }
                )
            out.append(
                {
                    "variable": var,
                    "etiqueta": label(var),
                    "iqr": iqr,
                    "per_unit_text": _per_unit_text(var),
                    "by_region": slopes,
                    "coef": float(res.params[var]),
                    "interaccion_region": True,
                }
            )
            continue
        est, se, lo, hi, p = _lincomb(res, {var: scale}, q)
        out.append(
            {
                "variable": var,
                "etiqueta": label(var),
                "coef": float(res.params[var]),
                "per_unit": est,
                "per_unit_lower": lo,
                "per_unit_upper": hi,
                "p": p,
                "per_unit_text": _per_unit_text(var),
                "iqr": iqr,
                "iqr_effect": float(res.params[var]) * iqr,
                "iqr_lower": float(res.params[var] - q * res.bse[var]) * iqr,
                "iqr_upper": float(res.params[var] + q * res.bse[var]) * iqr,
                "interaccion_region": False,
            }
        )
    return out


def numeric_interactions(res, design: Design, terms: list[str], alpha: float) -> list[Record]:  # type: ignore[no-untyped-def]
    """Pendiente de a en los cuartiles de b para las interacciones continua × continua."""
    q = float(stats.t.ppf(1 - alpha / 2, df_denominator(res)))
    out: list[Record] = []
    for term in terms:
        if ":" not in term or term.endswith(f":{REGION}"):
            continue
        a, b = term.split(":")
        xb = design.X[b]
        rows = []
        for prob in (0.25, 0.5, 0.75):
            at = float(xb.quantile(prob))
            est, _se, lo, hi, p = _lincomb(res, {a: 1.0, term: at}, q)
            rows.append(
                {
                    "cuantil": prob,
                    "valor_b": at + design.centers[b],
                    "pendiente": est,
                    "lower": lo,
                    "upper": hi,
                    "p": p,
                }
            )
        out.append({"term": term, "a": a, "b": b, "etiqueta": label(term), "slopes": rows})
    return out


def adjusted_region_means(res, design: Design, alpha: float) -> list[Record]:  # type: ignore[no-untyped-def]
    """Mortalidad esperada en cada región para un condado con las explicativas en su media.

    Como las continuas están centradas, la media ajustada de la región de referencia es la
    constante y la de otra región es la constante más su indicadora. Comparadas con las
    medias brutas, indican cuánta diferencia regional explican las demás variables.
    """
    if REGION not in [t.split(":")[0] for t in design.terms] or "region[" not in "".join(
        res.params.index
    ):
        return []
    q = float(stats.t.ppf(1 - alpha / 2, df_denominator(res)))
    out: list[Record] = []
    for lev in [design.reference, *region_levels(design.reference)]:
        wts = {"const": 1.0}
        if lev != design.reference:
            wts[dummy_name(lev)] = 1.0
        est, se, lo, hi, _ = _lincomb(res, wts, q)
        raw = float(design.y[design.region == lev].mean())
        out.append(
            {
                "region": lev,
                "ajustada": est,
                "se": se,
                "lower": lo,
                "upper": hi,
                "bruta": raw,
                "n": int((design.region == lev).sum()),
            }
        )
    return out


def sentence(effect: Record, alpha: float) -> str:
    """Frase interpretativa de un efecto (sin lenguaje causal)."""
    var = str(effect["etiqueta"])
    if effect.get("interaccion_region"):
        parts = []
        by_region: list[dict[str, Any]] = effect["by_region"]  # type: ignore[assignment]
        for s in by_region:
            parts.append(f"{s['region']}: {_es(float(s['coef']), 3)}")
        return (
            f"El efecto de {var.lower()} difiere según la región (interacción): "
            + "; ".join(parts)
            + " muertes por 100 000 por unidad."
        )
    est = float(effect["per_unit"])  # type: ignore[arg-type]
    lo = float(effect["per_unit_lower"])  # type: ignore[arg-type]
    hi = float(effect["per_unit_upper"])  # type: ignore[arg-type]
    direction = "más" if est > 0 else "menos"
    sig = "" if float(effect["p"]) < alpha else " (no significativo)"  # type: ignore[arg-type]
    return (
        f"Con {effect['per_unit_text']} de {var.lower()}, la mortalidad esperada es "
        f"{_es(abs(est))} muertes por 100 000 {direction} (IC 95 %: {_es(lo)}; {_es(hi)}), "
        f"a igualdad de las demás variables{sig}."
    )


def _es(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def compare_coefficients(base: pd.Series, alt: pd.Series, keys: list[str]) -> list[Record]:
    """Cambio relativo de los coeficientes entre dos especificaciones (criterio de confusión)."""
    out = []
    for k in keys:
        if k in base.index and k in alt.index:
            b0, b1 = float(base[k]), float(alt[k])
            out.append(
                {
                    "term": k,
                    "etiqueta": label(k),
                    "con": b0,
                    "sin": b1,
                    "cambio_pct": 100 * (b1 - b0) / abs(b0) if b0 != 0 else None,
                }
            )
    return out
