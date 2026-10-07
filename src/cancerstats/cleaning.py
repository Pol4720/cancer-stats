"""Depuración: tratamiento razonado de cada incidencia detectada.

Cada corrección se registra como una *decisión* con su justificación, la evidencia que la
respalda (calculada sobre los propios datos) y los condados afectados. Ninguna corrección
se aplica «porque sí»: cada una se apoya en una regla de validación que la detecta y en una
comprobación que verifica que el valor corregido es coherente con el resto del registro.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from cancerstats.config import FORBIDDEN_PREDICTORS, AnalysisConfig
from cancerstats.dictionary import STATE_TO_REGION
from cancerstats.validation import EDU_18_24, RACES, find_sentinels, sentinel_mask


@dataclass
class Decision:
    """Una decisión de depuración documentada."""

    id: str
    title: str
    problem: str
    action: str
    rationale: str
    config_key: str
    n_affected: int
    applied: bool = True
    evidence: dict[str, object] = field(default_factory=dict)
    affected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Representación serializable."""
        return asdict(self)


@dataclass
class CleaningResult:
    """Datos depurados y registro de decisiones."""

    data: pd.DataFrame
    decisions: list[Decision]
    sentinels: dict[str, float]
    excluded_variables: dict[str, str]


EXCLUDED_REASONS: dict[str, str] = {
    **FORBIDDEN_PREDICTORS,
    "MedianAgeMale": "redundante con MedianAge (r > 0,9); se usa sólo para validar y reparar.",
    "MedianAgeFemale": "redundante con MedianAge (r > 0,9); se usa sólo para validar y reparar.",
    "PctPrivateCoverageAlone": (
        "combinación casi lineal de las demás coberturas (R² ≈ 0,99) y 20 % de ausentes."
    ),
    "PctSomeCol18_24": (
        "categoría de referencia de la composición educativa de 18–24 años: las cuatro suman "
        "100 % e incluirlas todas haría singular X'X (como las tres indicadoras de una "
        "variable con tres categorías)."
    ),
    "PctWhite": (
        "categoría de referencia de la composición racial: blanca + negra + asiática + otra + "
        "nativa/multirracial = 100 %."
    ),
}


def clean(raw: pd.DataFrame, config: AnalysisConfig) -> CleaningResult:
    """Aplica las decisiones de depuración según la configuración."""
    cfg = config.cleaning
    df = raw.copy()
    decisions: list[Decision] = []

    # D01 y D02 se aplican en la ingesta (codificación y columna vacía); se registran aquí
    # para que el registro de decisiones sea completo.
    decisions.append(
        Decision(
            "D01",
            "Codificación y finales de línea",
            "El fichero está en Mac Roman con finales de línea CR; una lectura en UTF-8 falla y "
            "en Windows-1252 convierte «Doña Ana» en «Do–a Ana».",
            "Se decodifica en Mac Roman y se normalizan los finales de línea.",
            "La huella del Mac clásico (CR) y el byte 0x96 ⇒ «ñ» identifican la codificación "
            "sin ambigüedad.",
            "data.encoding",
            1,
            evidence={"condado": "Doña Ana County, New Mexico"},
            affected=["Doña Ana County, New Mexico"],
        )
    )
    decisions.append(
        Decision(
            "D02",
            "Columna vacía sin nombre",
            "Hay una columna sin cabecera y sin ningún valor entre MedianAgeFemale y "
            "AvgHouseholdSize, justo donde la documentación sitúa Geography.",
            "Se elimina.",
            "Geography se movió al final y su coma interna la dividió en «Geography» y "
            "«state»; la columna vacía es el hueco que dejó.",
            "—",
            len(df),
        )
    )

    # D03: centinelas -------------------------------------------------------------------
    sentinels = find_sentinels(df, cfg.sentinel_min_repeats, cfg.sentinel_min_decimals)
    smask = sentinel_mask(df, sentinels)
    states = df.loc[smask, "state"].value_counts()
    state_totals = df["state"].value_counts()
    coverage = {s: f"{int(n)}/{int(state_totals[s])}" for s, n in states.items()}
    if cfg.sentinel_to_missing:
        for col, value in sentinels.items():
            hit = np.isclose(df[col].to_numpy(dtype=float), value, rtol=0, atol=1e-6)
            df.loc[hit, col] = np.nan
    big = raw.loc[smask].sort_values("popEst2015", ascending=False).head(1)
    example = ""
    if len(big):
        row = big.iloc[0]
        example = (
            f" Además produce incoherencias evidentes ({row['county_id']}: "
            f"{_fmt_int(row['popEst2015'])} habitantes y {_fmt_int(row['avgDeathsPerYear'])} "
            f"muertes anuales, pero «{_fmt_int(row['avgAnnCount'])} casos»)."
        )
    values = " e ".join(f"{c} = {_fmt_dec(v)}" for c, v in sentinels.items())
    state_list = _join_es(list(states.index))
    whole_states = all(states[s] == state_totals[s] for s in states.index)
    decisions.append(
        Decision(
            "D03",
            "Valores centinela en incidencia y casos anuales",
            f"{values} se repiten exactamente en los mismos {int(smask.sum())} condados: "
            f"{'todos' if whole_states else 'casi todos'} "
            f"los de {state_list}, cuyos registros no publican la incidencia por condado.",
            "Se sustituyen por valores ausentes."
            if cfg.sentinel_to_missing
            else "No se aplica (desactivado en la configuración).",
            "Un valor con siete decimales repetido en cientos de condados no es una medición: "
            "es una imputación por la media que finge ser dato. Conservarla introduciría "
            f"{int(smask.sum())} observaciones con varianza nula en la incidencia y sesgaría su "
            "coeficiente hacia cero." + example,
            "cleaning.sentinel_to_missing",
            int(smask.sum()),
            cfg.sentinel_to_missing,
            evidence={"centinelas": sentinels, "estados": coverage},
            affected=df.loc[smask, "county_id"].tolist(),
        )
    )

    # D04: edad mediana en meses ---------------------------------------------------------
    months = df["MedianAge"] > 100
    lo = np.minimum(df["MedianAgeMale"], df["MedianAgeFemale"]) - cfg.median_age_tolerance
    hi = np.maximum(df["MedianAgeMale"], df["MedianAgeFemale"]) + cfg.median_age_tolerance
    corrected = (df["MedianAge"] / 12).round(1)
    verified = months & (corrected >= lo) & (corrected <= hi)
    inside = ~months & (df["MedianAge"] >= lo) & (df["MedianAge"] <= hi)
    share_inside = inside.sum() / max(int((~months).sum()), 1)
    coherent_text = "siempre cae" if share_inside == 1 else f"cae en el {100 * share_inside:.1f} %"
    ratio = df.loc[months, "MedianAge"] / (
        (df.loc[months, "MedianAgeMale"] + df.loc[months, "MedianAgeFemale"]) / 2
    )
    if cfg.repair_median_age:
        df.loc[verified, "MedianAge"] = corrected[verified]
        df.loc[months & ~verified, "MedianAge"] = np.nan
    decisions.append(
        Decision(
            "D04",
            "Edad mediana registrada en meses",
            f"{int(months.sum())} condados tienen edades medianas entre "
            f"{_fmt_dec(raw.loc[months, 'MedianAge'].min())} y "
            f"{_fmt_dec(raw.loc[months, 'MedianAge'].max())} años.",
            "Se divide entre 12 y se verifica que el resultado quede entre las medianas "
            "masculina y femenina."
            if cfg.repair_median_age
            else "No se aplica.",
            "El cociente entre el valor registrado y la media de las medianas por sexo es 12 "
            f"(media {_fmt_dec(ratio.mean(), 3)}, desviación típica {_fmt_dec(ratio.std(), 3)}): "
            f"el dato está en meses. En los {_fmt_int((~months).sum())} condados válidos la "
            f"mediana conjunta {coherent_text} entre las dos medianas por sexo, de modo que esa "
            "misma regla verifica cada corrección.",
            "cleaning.repair_median_age",
            int(months.sum()),
            cfg.repair_median_age,
            evidence={
                "razon_media": round(float(ratio.mean()), 4),
                "razon_dt": round(float(ratio.std()), 4),
                "verificadas": int(verified.sum()),
            },
            affected=df.loc[months, "county_id"].tolist(),
        )
    )

    # D05: escala del tamaño del hogar ---------------------------------------------------
    small = df["AvgHouseholdSize"] < 1
    scaled = df.loc[small, "AvgHouseholdSize"] * 100
    valid_range = df.loc[~small, "AvgHouseholdSize"]
    if cfg.repair_household_size:
        df.loc[small, "AvgHouseholdSize"] = scaled.round(2)
    decisions.append(
        Decision(
            "D05",
            "Tamaño medio del hogar dividido por 100",
            f"{int(small.sum())} condados declaran entre {_fmt_dec(df_min(raw, small), 4)} y "
            f"{_fmt_dec(df_max(raw, small), 4)} personas por hogar.",
            "Se multiplica por 100." if cfg.repair_household_size else "No se aplica.",
            "Un hogar tiene al menos una persona. Los valores erróneos tienen cuatro decimales "
            "frente a los dos del resto, y multiplicados por 100 quedan en "
            f"[{_fmt_dec(scaled.min(), 2)}; {_fmt_dec(scaled.max(), 2)}], dentro del rango de "
            f"los válidos [{_fmt_dec(valid_range.min(), 2)}; {_fmt_dec(valid_range.max(), 2)}]: "
            "es un error de escala.",
            "cleaning.repair_household_size",
            int(small.sum()),
            cfg.repair_household_size,
            evidence={
                "min_corregido": round(float(scaled.min()), 3),
                "max_corregido": round(float(scaled.max()), 3),
            },
            affected=df.loc[small, "county_id"].tolist(),
        )
    )

    # D06: recuperación de PctSomeCol18_24 por identidad --------------------------------
    block = df[list(EDU_18_24)]
    complete = block.notna().all(axis=1)
    residual = (block.loc[complete].sum(axis=1) - 100).abs()
    gap = df["PctSomeCol18_24"].isna() & df[
        ["PctNoHS18_24", "PctHS18_24", "PctBachDeg18_24"]
    ].notna().all(axis=1)
    recovered = (100 - df["PctNoHS18_24"] - df["PctHS18_24"] - df["PctBachDeg18_24"]).clip(lower=0)
    if cfg.recover_some_college:
        df.loc[gap, "PctSomeCol18_24"] = recovered[gap].round(1)
    decisions.append(
        Decision(
            "D06",
            "PctSomeCol18_24 recuperada por identidad contable",
            f"Falta el {_fmt_dec(100 * gap.mean(), 1)} % de los valores de PctSomeCol18_24.",
            "Se calcula como 100 − (sin bachillerato + bachillerato + grado)."
            if cfg.recover_some_college
            else "No se aplica.",
            "En los casos completos las cuatro categorías suman 100 % con un error máximo de "
            f"{_fmt_dec(residual.max(), 2)} puntos (redondeo a un decimal): la identidad es "
            "exacta y el valor ausente no se imputa, se deduce. Una imputación estadística "
            "sería aquí un error metodológico, porque desecharía información determinista.",
            "cleaning.recover_some_college",
            int(gap.sum()),
            cfg.recover_some_college,
            evidence={
                "casos_completos": int(complete.sum()),
                "error_maximo_identidad": round(float(residual.max()), 3),
            },
            affected=[],
        )
    )

    # D07: residuo racial ----------------------------------------------------------------
    race_sum = df[list(RACES)].sum(axis=1)
    if cfg.derive_native_multi:
        df["PctNativeMulti"] = (100 - race_sum).clip(lower=0)
    low = race_sum < 80
    min_county = str(df.loc[race_sum.idxmin(), "county_id"])
    decisions.append(
        Decision(
            "D07",
            "Población nativa o multirracial (variable derivada)",
            "Según el diccionario, PctOtherRace recoge todo lo que no es blanco, negro ni "
            f"asiático; sin embargo, en {int(low.sum())} condados las cuatro categorías suman "
            f"menos del 80 % ({min_county}: {_fmt_dec(race_sum.min(), 1)} %).",
            "Se crea PctNativeMulti = 100 − (blanca + negra + asiática + otra)."
            if cfg.derive_native_multi
            else "No se aplica.",
            "Los condados con mayor residuo son reservas indias y áreas censales de Alaska: el "
            "residuo es la población nativa americana o de Alaska y la multirracial, que el "
            "diccionario no menciona. Es un grupo con tasas de cáncer documentadamente "
            "distintas, de modo que omitirlo sería perder información relevante.",
            "cleaning.derive_native_multi",
            int(low.sum()),
            cfg.derive_native_multi,
            evidence={"suma_minima": round(float(race_sum.min()), 3)},
            affected=df.loc[low, "county_id"].tolist(),
        )
    )

    # D08: incidencia inverosímil (MIR > umbral) -----------------------------------------
    mir = df["TARGET_deathRate"] / df["incidenceRate"]
    implausible = (
        (mir > cfg.mir_max) if cfg.mir_max is not None else pd.Series(False, index=df.index)
    )
    implausible = implausible.fillna(False)
    if cfg.mir_max is not None:
        df.loc[implausible, "incidenceRate"] = np.nan
    decisions.append(
        Decision(
            "D08",
            "Incidencia inverosímil (mortalidad ≥ incidencia)",
            "En algún condado la tasa de mortalidad supera a la de incidencia (MIR > "
            f"{_fmt_dec(cfg.mir_max or 0)}); la MIR mediana es {_fmt_dec(mir.median(), 2)}.",
            "Se marca la incidencia como ausente." if cfg.mir_max is not None else "No se aplica.",
            "Que mueran más personas de las que se diagnostican no es sostenible: indica que "
            "parte de los diagnósticos se registra fuera del condado de residencia (pacientes "
            "diagnosticados en otra jurisdicción, poblaciones flotantes). Condados afectados: "
            + (_join_es(df.loc[implausible, "county_id"].tolist()) or "ninguno")
            + ".",
            "cleaning.mir_max",
            int(implausible.sum()),
            cfg.mir_max is not None,
            evidence={"mir_mediana": round(float(mir.median()), 4)},
            affected=df.loc[implausible, "county_id"].tolist(),
        )
    )

    # D09: región censal ------------------------------------------------------------------
    df["region"] = df["state"].map(STATE_TO_REGION)
    decisions.append(
        Decision(
            "D09",
            "Región censal",
            "El estado tiene 51 categorías y algunas con muy pocos condados (Distrito de "
            "Columbia: 1; Delaware: 3).",
            "Se agrupan en las cuatro regiones oficiales de la Oficina del Censo.",
            "La región permite ilustrar variables indicadoras e interacciones con 3 grados de "
            "libertad en lugar de 50; el estado se reserva como conglomerado para los errores "
            "típicos robustos.",
            "variables.use_region",
            len(df),
            evidence={"n_por_region": df["region"].value_counts().to_dict()},
        )
    )

    # D10: transformaciones ---------------------------------------------------------------
    t = config.transforms
    df["logPop"] = np.log(df["popEst2015"])
    df["logIncome"] = np.log(df["medIncome"])
    df["logStudy"] = np.log1p(df["studyPerCap"])
    df["anyStudy"] = (df["studyPerCap"] > 0).astype(int)
    decisions.append(
        Decision(
            "D10",
            "Transformaciones de variables muy asimétricas",
            "Población (asimetría ≈ 14), renta (≈ 1,4) y ensayos per cápita (63 % de ceros y "
            "asimetría ≈ 5,7) violan la linealidad que exige el modelo.",
            f"Población: {'logaritmo' if t.log_population else 'sin transformar'}; renta: "
            f"{'logaritmo' if t.log_income else 'sin transformar'}; ensayos: {t.study}.",
            "El parámetro λ de Box-Cox estimado es ≈ 0 para la población y los ensayos, que "
            "corresponde al logaritmo; además los efectos de escala (un 10 % más de renta) "
            "son más interpretables que los absolutos.",
            "transforms",
            len(df),
        )
    )

    # D11: variables excluidas ------------------------------------------------------------
    decisions.append(
        Decision(
            "D11",
            "Variables excluidas de los modelos",
            "Algunas columnas no pueden o no deben usarse como explicativas.",
            "Se excluyen: " + ", ".join(EXCLUDED_REASONS) + ".",
            "Ver el motivo de cada una en la tabla de variables excluidas.",
            "variables",
            len(EXCLUDED_REASONS),
            evidence={"motivos": EXCLUDED_REASONS},
        )
    )

    return CleaningResult(df, decisions, sentinels, dict(EXCLUDED_REASONS))


def _fmt_int(x: float) -> str:
    """Entero con separador de miles español (espacio fino)."""
    return f"{round(float(x)):,}".replace(",", "\u202f")


def _fmt_dec(x: float, digits: int | None = None) -> str:
    """Decimal con coma; sin ``digits`` conserva la representación más corta."""
    text = f"{float(x):.{digits}f}" if digits is not None else f"{float(x):.10g}"
    return text.replace(".", ",")


def _join_es(items: list[str]) -> str:
    """Une una lista con comas y «y» final."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def df_min(df: pd.DataFrame, mask: pd.Series) -> float:
    """Mínimo de AvgHouseholdSize en las filas marcadas."""
    return float(df.loc[mask, "AvgHouseholdSize"].min()) if mask.any() else float("nan")


def df_max(df: pd.DataFrame, mask: pd.Series) -> float:
    """Máximo de AvgHouseholdSize en las filas marcadas."""
    return float(df.loc[mask, "AvgHouseholdSize"].max()) if mask.any() else float("nan")


def model_column(name: str, config: AnalysisConfig) -> str:
    """Nombre de la columna que representa a una variable original en los modelos."""
    t = config.transforms
    if name == "popEst2015" and t.log_population:
        return "logPop"
    if name == "medIncome" and t.log_income:
        return "logIncome"
    if name == "studyPerCap":
        return {"log1p": "logStudy", "binary": "anyStudy", "none": "studyPerCap"}[t.study]
    return name


def response_column(config: AnalysisConfig) -> str:
    """Columna de respuesta (transformada si así se configura)."""
    return "logDeathRate" if config.transforms.response == "log" else "TARGET_deathRate"


def add_response(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Añade la respuesta transformada si procede."""
    if config.transforms.response == "log":
        df = df.assign(logDeathRate=np.log(df["TARGET_deathRate"]))
    return df


__all__ = [
    "EXCLUDED_REASONS",
    "CleaningResult",
    "Decision",
    "add_response",
    "clean",
    "model_column",
    "response_column",
]
