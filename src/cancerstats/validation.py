"""Catálogo de reglas de validación.

Cada regla expresa una propiedad que los datos *deben* cumplir por su propia definición
(dominio, coherencia interna, identidades contables) y cuenta cuántos condados la violan.
Las reglas se ejecutan dos veces: sobre los datos originales, para documentar las
incidencias, y sobre los datos depurados, para demostrar que el tratamiento las resolvió.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from cancerstats.config import CleaningConfig
from cancerstats.dictionary import STATE_TO_REGION

Severity = Literal["error", "advertencia", "información"]

PERCENT_COLUMNS: tuple[str, ...] = (
    "povertyPercent",
    "PercentMarried",
    "PctNoHS18_24",
    "PctHS18_24",
    "PctSomeCol18_24",
    "PctBachDeg18_24",
    "PctHS25_Over",
    "PctBachDeg25_Over",
    "PctEmployed16_Over",
    "PctUnemployed16_Over",
    "PctPrivateCoverage",
    "PctPrivateCoverageAlone",
    "PctEmpPrivCoverage",
    "PctPublicCoverage",
    "PctPublicCoverageAlone",
    "PctWhite",
    "PctBlack",
    "PctAsian",
    "PctOtherRace",
    "PctMarriedHouseholds",
)

EDU_18_24: tuple[str, ...] = ("PctNoHS18_24", "PctHS18_24", "PctSomeCol18_24", "PctBachDeg18_24")
RACES: tuple[str, ...] = ("PctWhite", "PctBlack", "PctAsian", "PctOtherRace")


@dataclass
class RuleResult:
    """Resultado de aplicar una regla."""

    id: str
    title: str
    description: str
    category: str
    severity: Severity
    columns: list[str]
    n_checked: int
    n_violations: int
    treatment: str
    examples: list[dict[str, object]] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``"ok"`` si no hay violaciones; si no, la severidad."""
        return "ok" if self.n_violations == 0 else self.severity

    def to_dict(self) -> dict[str, object]:
        """Representación serializable."""
        out = asdict(self)
        out["status"] = self.status
        return out


def _examples(
    df: pd.DataFrame, mask: pd.Series, cols: list[str], k: int = 8
) -> list[dict[str, object]]:
    rows = df.loc[mask.fillna(False).astype(bool), ["county_id", *cols]].head(k)
    out: list[dict[str, object]] = []
    for rec in rows.to_dict(orient="records"):
        out.append(
            {
                str(key): (
                    None
                    if isinstance(val, float) and np.isnan(val)
                    else (round(val, 4) if isinstance(val, float) else val)
                )
                for key, val in rec.items()
            }
        )
    return out


def _decimals(value: float) -> int:
    text = f"{value:.12g}"
    if "e" in text or "." not in text:
        return 0
    return len(text.split(".")[1])


def find_sentinels(df: pd.DataFrame, min_repeats: int, min_decimals: int) -> dict[str, float]:
    """Detecta valores centinela: un mismo valor no redondo repetido muchas veces.

    Una medición continua con seis o siete decimales no se repite exactamente en decenas de
    condados por azar: es la huella de una imputación previa (típicamente por la media).
    """
    found: dict[str, float] = {}
    for col in df.select_dtypes(include="number").columns:
        counts = df[col].value_counts()
        for value, n in counts.items():
            v = float(value)  # type: ignore[arg-type]
            if n >= min_repeats and _decimals(v) >= min_decimals:
                found[str(col)] = v
                break
    return found


def sentinel_mask(df: pd.DataFrame, sentinels: dict[str, float]) -> pd.Series:
    """Filas en las que alguna columna toma su valor centinela."""
    mask = pd.Series(False, index=df.index)
    for col, value in sentinels.items():
        if col in df.columns:
            mask |= np.isclose(df[col].to_numpy(dtype=float), value, rtol=0, atol=1e-6)
    return mask


Rule = Callable[[pd.DataFrame, CleaningConfig, dict[str, float]], RuleResult]


def r01_unique_id(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    dup = df.duplicated(["Geography", "state"], keep=False)
    return RuleResult(
        "R01",
        "Identificador único",
        "Cada par (condado, estado) aparece una sola vez.",
        "estructura",
        "error",
        ["Geography", "state"],
        len(df),
        int(dup.sum()),
        "Sin incidencias: no hay condados duplicados.",
        _examples(df, dup, ["Geography"]),
    )


def r02_percent_domain(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    cols = [c for c in PERCENT_COLUMNS if c in df.columns]
    block = df[cols]
    bad = ((block < 0) | (block > 100)).any(axis=1)
    return RuleResult(
        "R02",
        "Porcentajes en [0, 100]",
        "Toda variable expresada en % está entre 0 y 100.",
        "dominio",
        "error",
        cols,
        int(block.notna().any(axis=1).sum()),
        int(bad.sum()),
        "Sin incidencias.",
        _examples(df, bad, cols[:4]),
    )


def r03_positive(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    cols = ["TARGET_deathRate", "incidenceRate", "medIncome", "popEst2015"]
    bad = (df[cols] <= 0).any(axis=1) | (
        df[["avgAnnCount", "avgDeathsPerYear", "studyPerCap", "BirthRate"]] < 0
    ).any(axis=1)
    return RuleResult(
        "R03",
        "Tasas, renta y población positivas",
        "Las tasas, la renta y la población son estrictamente positivas; los conteos, no "
        "negativos.",
        "dominio",
        "error",
        [*cols, "avgAnnCount", "avgDeathsPerYear"],
        len(df),
        int(bad.sum()),
        "Sin incidencias.",
        _examples(df, bad, cols),
    )


def r04_sentinels(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    found = find_sentinels(df, cfg.sentinel_min_repeats, cfg.sentinel_min_decimals)
    mask = sentinel_mask(df, found)
    desc = ", ".join(f"{c} = {v:.10g}".replace(".", ",") for c, v in found.items()) or "ninguno"
    return RuleResult(
        "R04",
        "Valores centinela",
        "Un valor con muchos decimales no puede repetirse exactamente en muchos condados; si "
        f"ocurre, es una imputación previa disfrazada de dato. Detectados: {desc}.",
        "plausibilidad",
        "error",
        list(found),
        len(df),
        int(mask.sum()),
        "D03: se sustituyen por ausente (dato suprimido en origen, no medido).",
        _examples(df, mask, [*found, "popEst2015", "avgDeathsPerYear", "state"]),
    )


def r05_age_domain(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    bad = df["MedianAge"] > 100
    return RuleResult(
        "R05",
        "Edad mediana plausible (≤ 100 años)",
        "Ningún condado puede tener una edad mediana superior a 100 años.",
        "dominio",
        "error",
        ["MedianAge"],
        len(df),
        int(bad.sum()),
        "D04: la razón con la media de las medianas por sexo es 12 ⇒ el valor está en meses.",
        _examples(df, bad, ["MedianAge", "MedianAgeMale", "MedianAgeFemale"]),
    )


def r06_age_coherence(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    lo = np.minimum(df["MedianAgeMale"], df["MedianAgeFemale"]) - cfg.median_age_tolerance
    hi = np.maximum(df["MedianAgeMale"], df["MedianAgeFemale"]) + cfg.median_age_tolerance
    bad = (df["MedianAge"] < lo) | (df["MedianAge"] > hi)
    return RuleResult(
        "R06",
        "Coherencia de las edades medianas",
        "La mediana conjunta debe estar entre las medianas de hombres y mujeres (con una "
        f"tolerancia de {cfg.median_age_tolerance:g} años).".replace(".", ",", 1),
        "consistencia",
        "error",
        ["MedianAge", "MedianAgeMale", "MedianAgeFemale"],
        len(df),
        int(bad.sum()),
        "D04: tras dividir entre 12 todas las corregidas cumplen la regla.",
        _examples(df, bad, ["MedianAge", "MedianAgeMale", "MedianAgeFemale"]),
    )


def r07_household(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    bad = df["AvgHouseholdSize"] < 1
    return RuleResult(
        "R07",
        "Tamaño del hogar ≥ 1 persona",
        "Un hogar tiene, por definición, al menos una persona.",
        "dominio",
        "error",
        ["AvgHouseholdSize"],
        len(df),
        int(bad.sum()),
        "D05: los valores están en [0,022; 0,032] ⇒ error de escala ×1/100.",
        _examples(df, bad, ["AvgHouseholdSize", "PctMarriedHouseholds"]),
    )


def r08_edu_identity(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    block = df[list(EDU_18_24)]
    complete = block.notna().all(axis=1)
    total = block.sum(axis=1, min_count=4)
    bad_complete = complete & ((total - 100).abs() > cfg.identity_tolerance)
    partial = block[["PctNoHS18_24", "PctHS18_24", "PctBachDeg18_24"]].sum(axis=1)
    bad_partial = ~complete & (partial > 100 + cfg.identity_tolerance)
    bad = bad_complete | bad_partial
    return RuleResult(
        "R08",
        "Identidad educativa de 18–24 años",
        "Las cuatro categorías (sin bachillerato, bachillerato, superiores incompletos, grado) "
        "agotan la población de 18–24 años: deben sumar 100 %.",
        "identidad",
        "error",
        list(EDU_18_24),
        int(complete.sum()),
        int(bad.sum()),
        "D06: se cumple en todos los casos completos, lo que permite recuperar el valor "
        "ausente de PctSomeCol18_24 por diferencia.",
        _examples(df, bad, list(EDU_18_24)),
    )


def r09_race_sum(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    total = df[list(RACES)].sum(axis=1)
    bad = total > 100 + cfg.identity_tolerance
    return RuleResult(
        "R09",
        "Composición racial ≤ 100 %",
        "Las categorías raciales son excluyentes: su suma no puede superar el 100 %. El "
        "residuo hasta 100 corresponde a la población nativa o multirracial.",
        "identidad",
        "error",
        list(RACES),
        len(df),
        int(bad.sum()),
        "D07: se deriva PctNativeMulti = 100 − suma.",
        _examples(df, bad, list(RACES)),
    )


def r10_labour(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    total = df["PctEmployed16_Over"] + df["PctUnemployed16_Over"]
    bad = total > 100 + cfg.identity_tolerance
    return RuleResult(
        "R10",
        "Empleo + desempleo ≤ 100 %",
        "Ocupados y desempleados son subconjuntos disjuntos de la población de 16+ años.",
        "identidad",
        "error",
        ["PctEmployed16_Over", "PctUnemployed16_Over"],
        int(total.notna().sum()),
        int(bad.sum()),
        "Sin incidencias.",
        _examples(df, bad, ["PctEmployed16_Over", "PctUnemployed16_Over"]),
    )


def r11_edu25(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    total = df["PctHS25_Over"] + df["PctBachDeg25_Over"]
    bad = total > 100 + cfg.identity_tolerance
    return RuleResult(
        "R11",
        "Educación de 25+ años ≤ 100 %",
        "Bachillerato como máximo y grado universitario son categorías disjuntas.",
        "identidad",
        "error",
        ["PctHS25_Over", "PctBachDeg25_Over"],
        len(df),
        int(bad.sum()),
        "Sin incidencias.",
        _examples(df, bad, ["PctHS25_Over", "PctBachDeg25_Over"]),
    )


def r12_coverage(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    tol = cfg.identity_tolerance
    bad = (
        (df["PctEmpPrivCoverage"] > df["PctPrivateCoverage"] + tol)
        | (df["PctPrivateCoverageAlone"] > df["PctPrivateCoverage"] + tol)
        | (df["PctPublicCoverageAlone"] > df["PctPublicCoverage"] + tol)
        | (df["PctPrivateCoverage"] + df["PctPublicCoverageAlone"] > 100 + tol)
    )
    cols = [
        "PctPrivateCoverage",
        "PctPrivateCoverageAlone",
        "PctEmpPrivCoverage",
        "PctPublicCoverage",
        "PctPublicCoverageAlone",
    ]
    return RuleResult(
        "R12",
        "Jerarquía de coberturas sanitarias",
        "«Solo privada» ⊆ privada, «vía empleador» ⊆ privada, «solo pública» ⊆ pública, y "
        "privada + solo pública ≤ 100 %.",
        "consistencia",
        "error",
        cols,
        len(df),
        int(bad.sum()),
        "Sin incidencias.",
        _examples(df, bad, cols),
    )


def r13_income_bin(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    bounds = df["binnedInc"].str.extract(r"[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*\]").astype(float)
    closed_left = df["binnedInc"].str.startswith("[")
    lower_ok = np.where(closed_left, df["medIncome"] >= bounds[0], df["medIncome"] > bounds[0])
    bad = ~(pd.Series(lower_ok, index=df.index) & (df["medIncome"] <= bounds[1]))
    return RuleResult(
        "R13",
        "Renta mediana dentro de su decil",
        "binnedInc es el decil de medIncome: cada renta debe caer en su intervalo.",
        "consistencia",
        "error",
        ["medIncome", "binnedInc"],
        len(df),
        int(bad.sum()),
        "Se confirma que binnedInc es redundante; no entra en los modelos.",
        _examples(df, bad, ["medIncome", "binnedInc"]),
    )


def r14_cases_deaths(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    valid = ~sentinel_mask(df, s) & df["avgAnnCount"].notna()
    bad = valid & (df["avgAnnCount"] < df["avgDeathsPerYear"])
    return RuleResult(
        "R14",
        "Casos anuales ≥ muertes anuales",
        "En régimen estacionario no puede morir de cáncer más gente de la que se diagnostica "
        "(se excluyen los centinelas).",
        "plausibilidad",
        "advertencia",
        ["avgAnnCount", "avgDeathsPerYear"],
        int(valid.sum()),
        int(bad.sum()),
        "Sin incidencias fuera de los centinelas.",
        _examples(df, bad, ["avgAnnCount", "avgDeathsPerYear"]),
    )


def r15_mir(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    valid = ~sentinel_mask(df, s) & df["incidenceRate"].notna()
    mir = df["TARGET_deathRate"] / df["incidenceRate"]
    limit = cfg.mir_max if cfg.mir_max is not None else np.inf
    bad = valid & (mir > limit)
    view = df.assign(MIR=mir)
    return RuleResult(
        "R15",
        "Razón mortalidad/incidencia plausible",
        "La razón mortalidad/incidencia (MIR) ronda 0,40; una tasa de mortalidad superior a "
        "la de incidencia sugiere un registro de diagnósticos incompleto (p. ej., pacientes "
        "diagnosticados en otro estado).",
        "plausibilidad",
        "advertencia",
        ["TARGET_deathRate", "incidenceRate"],
        int(valid.sum()),
        int(bad.sum()),
        "D08: la incidencia de esos condados se marca como ausente.",
        _examples(view, bad, ["TARGET_deathRate", "incidenceRate", "MIR", "popEst2015"]),
    )


def r16_region(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    bad = ~df["state"].isin(list(STATE_TO_REGION))
    return RuleResult(
        "R16",
        "Estado reconocido",
        "Cada estado pertenece a una de las cuatro regiones censales.",
        "estructura",
        "error",
        ["state"],
        len(df),
        int(bad.sum()),
        "Sin incidencias.",
        _examples(df, bad, ["state"]),
    )


def r17_declared_missing(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    na = df.isna().sum()
    na = na[na > 0]
    detail = ", ".join(f"{c}: {n}" for c, n in na.items()) or "ninguno"
    rows = df.isna().any(axis=1)
    return RuleResult(
        "R17",
        "Ausentes declarados",
        f"Celdas vacías en el fichero. Por columna: {detail}.",
        "completitud",
        "información",
        list(na.index),
        len(df),
        int(rows.sum()),
        "Se analiza su mecanismo en la etapa de datos ausentes.",
        _examples(df, rows, list(na.index)[:4]),
    )


def r18_zero_inflation(df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]) -> RuleResult:
    zeros = df["studyPerCap"] == 0
    share = float(zeros.mean())
    share_text = f"{100 * share:.1f}".replace(".", ",")
    return RuleResult(
        "R18",
        "Inflación de ceros en ensayos clínicos",
        f"El {share_text} % de los condados no tiene ningún ensayo clínico: la variable es "
        "una mezcla de ceros estructurales y una cola muy asimétrica.",
        "distribución",
        "información",
        ["studyPerCap"],
        len(df),
        int(zeros.sum()),
        "Se transforma con log(1 + x) (o indicador binario, según configuración).",
        _examples(df, zeros, ["studyPerCap", "popEst2015"], k=3),
    )


def r19_cases_minus_deaths(
    df: pd.DataFrame, cfg: CleaningConfig, s: dict[str, float]
) -> RuleResult:
    if "Notificadomuerte" not in df.columns:
        return RuleResult(
            "R19",
            "Identidad de Notificadomuerte",
            "La fuente no contiene la variable Notificadomuerte.",
            "identidad",
            "información",
            [],
            0,
            0,
            "No aplica.",
        )
    valid = df[["Notificadomuerte", "avgAnnCount", "avgDeathsPerYear"]].notna().all(axis=1)
    diff = df["Notificadomuerte"] - (df["avgAnnCount"] - df["avgDeathsPerYear"])
    bad = valid & (diff.abs() > 1e-6)
    same_missing = bool((df["Notificadomuerte"].isna() == df["avgAnnCount"].isna()).all())
    return RuleResult(
        "R19",
        "Identidad de Notificadomuerte",
        "La variable que añade el fichero oficial debe ser avgAnnCount − avgDeathsPerYear "
        f"(se comprueba en {int(valid.sum())} condados); "
        + (
            "falta exactamente donde falta avgAnnCount."
            if same_missing
            else "su patrón de ausencia difiere del de avgAnnCount."
        ),
        "identidad",
        "error",
        ["Notificadomuerte", "avgAnnCount", "avgDeathsPerYear"],
        int(valid.sum()),
        int(bad.sum()),
        "D02: al contener las muertes (numerador de la respuesta) se excluye de los modelos.",
        _examples(df, bad, ["Notificadomuerte", "avgAnnCount", "avgDeathsPerYear"]),
    )


RULES: tuple[Rule, ...] = (
    r01_unique_id,
    r02_percent_domain,
    r03_positive,
    r04_sentinels,
    r05_age_domain,
    r06_age_coherence,
    r07_household,
    r08_edu_identity,
    r09_race_sum,
    r10_labour,
    r11_edu25,
    r12_coverage,
    r13_income_bin,
    r14_cases_deaths,
    r15_mir,
    r16_region,
    r17_declared_missing,
    r18_zero_inflation,
    r19_cases_minus_deaths,
)


def validate(
    df: pd.DataFrame, cfg: CleaningConfig, sentinels: dict[str, float] | None = None
) -> list[RuleResult]:
    """Aplica todas las reglas del catálogo."""
    if sentinels is None:
        sentinels = find_sentinels(df, cfg.sentinel_min_repeats, cfg.sentinel_min_decimals)
    return [rule(df, cfg, sentinels) for rule in RULES]


def summarize(results: list[RuleResult]) -> dict[str, int]:
    """Recuento de reglas por estado."""
    out = {"total": len(results), "ok": 0, "error": 0, "advertencia": 0, "información": 0}
    for res in results:
        out[res.status] += 1
    return out


_ID = re.compile(r"^R\d{2}$")


def by_id(results: list[RuleResult]) -> dict[str, RuleResult]:
    """Índice de resultados por identificador de regla."""
    return {r.id: r for r in results if _ID.match(r.id)}
