"""Configuración tipada de la metodología.

Cada decisión del análisis que admite criterio del investigador es un parámetro de esta
configuración. Las descripciones de los campos están en castellano porque la interfaz de
usuario genera sus formularios a partir del esquema JSON de estos modelos.

La configuración se guarda en YAML y se identifica por una huella SHA-256 de su forma
canónica: dos corridas con la misma huella y los mismos datos producen los mismos
resultados.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cancerstats.dictionary import REGIONS, VARIABLES

# Variables cuya inclusión como predictor invalidaría el análisis. Se rechazan en la
# validación de la configuración con un mensaje que explica el motivo.
FORBIDDEN_PREDICTORS: dict[str, str] = {
    "TARGET_deathRate": "es la propia variable respuesta.",
    "avgDeathsPerYear": (
        "fuga de información: es el numerador de la tasa de mortalidad que se quiere explicar."
    ),
    "avgAnnCount": (
        "es un conteo que escala con la población (numerador de la incidencia) y falta, o es "
        "un valor centinela, en los condados de Kansas, Minnesota y Nevada."
    ),
    "Notificadomuerte": (
        "fuga de información: es exactamente avgAnnCount − avgDeathsPerYear, de modo que "
        "contiene el numerador de la respuesta."
    ),
    "binnedInc": "es una función determinista de medIncome (su decil), información redundante.",
    "Geography": "es un identificador, no una característica del condado.",
    "state": "se usa como unidad de conglomerado; la región recoge el efecto geográfico.",
}

DEFAULT_CANDIDATES: list[str] = [
    "incidenceRate",
    "povertyPercent",
    "medIncome",
    "MedianAge",
    "AvgHouseholdSize",
    "PercentMarried",
    "PctNoHS18_24",
    "PctHS18_24",
    "PctBachDeg18_24",
    "PctHS25_Over",
    "PctBachDeg25_Over",
    "PctEmployed16_Over",
    "PctUnemployed16_Over",
    "PctPrivateCoverage",
    "PctEmpPrivCoverage",
    "PctPublicCoverage",
    "PctPublicCoverageAlone",
    "PctBlack",
    "PctAsian",
    "PctOtherRace",
    "PctNativeMulti",
    "PctMarriedHouseholds",
    "BirthRate",
    "studyPerCap",
    "popEst2015",
]

DEFAULT_INTERACTIONS: list[list[str]] = [
    ["incidenceRate", "region"],
    ["PctBachDeg25_Over", "region"],
    ["PctPublicCoverageAlone", "region"],
    ["PctUnemployed16_Over", "region"],
    ["povertyPercent", "region"],
    ["incidenceRate", "PctBachDeg25_Over"],
]

PredictiveModelName = Literal[
    "baseline",
    "ols_effects",
    "ols_full",
    "ridge",
    "lasso",
    "elasticnet",
    "random_forest",
    "gradient_boosting",
]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetaConfig(_Section):
    """Identificación de la corrida."""

    name: str = Field("Metodología por defecto", title="Nombre de la configuración")
    description: str = Field(
        "Configuración con la que se elaboró el informe final.",
        title="Descripción",
        description="Texto libre que explica qué se cambia respecto de la metodología base.",
    )
    seed: int = Field(
        20261007,
        title="Semilla aleatoria",
        description="Fija el remuestreo, la partición entrenamiento/prueba y la validación "
        "cruzada. Misma semilla y misma configuración ⇒ mismos resultados.",
        ge=0,
    )


class DataConfig(_Section):
    """Origen de los datos."""

    path: str = Field(
        "data/raw/practica.sav",
        title="Fichero de datos",
        description="Fuente oficial (fichero de SPSS entregado por el profesor). Ruta relativa "
        "a la raíz del repositorio; también admite el CSV de Kaggle.",
    )
    reference_path: str | None = Field(
        "data/raw/CANCER.csv",
        title="Fuente de referencia",
        description="CSV público de Kaggle con el que se verifica celda a celda la procedencia "
        "de la fuente oficial. Vacío para omitir la comprobación.",
    )
    encoding: Literal["auto", "mac_roman", "utf-8", "cp1252"] = Field(
        "auto",
        title="Codificación del CSV",
        description="«auto» detecta la codificación. El CSV de Kaggle está en Mac Roman con "
        "finales de línea CR (formato del Mac clásico); el .sav declara la suya.",
    )


class CleaningConfig(_Section):
    """Reglas de depuración: cada una corrige una incidencia documentada."""

    sentinel_to_missing: bool = Field(
        True,
        title="Centinelas → ausente",
        description="Sustituye por ausente los valores constantes repetidos con muchos decimales "
        "(imputaciones previas por la media que fingen ser datos).",
    )
    sentinel_min_repeats: int = Field(
        10,
        title="Repeticiones mínimas del centinela",
        description="Número de veces que debe repetirse un valor no redondo para considerarse "
        "centinela.",
        ge=2,
    )
    sentinel_min_decimals: int = Field(
        4,
        title="Decimales mínimos del centinela",
        description="Un valor con al menos estos decimales repetido muchas veces no es una "
        "medición plausible.",
        ge=1,
    )
    repair_median_age: bool = Field(
        True,
        title="Corregir edad mediana en meses",
        description="Divide entre 12 las edades medianas superiores a 100 si el resultado cae "
        "entre las medianas masculina y femenina.",
    )
    median_age_tolerance: float = Field(
        0.5,
        title="Tolerancia de la edad corregida (años)",
        description="Margen admitido respecto del intervalo [mín(H, M), máx(H, M)].",
        ge=0,
    )
    repair_household_size: bool = Field(
        True,
        title="Corregir escala del tamaño del hogar",
        description="Multiplica por 100 los tamaños medios de hogar inferiores a 1 persona "
        "(error de escala).",
    )
    recover_some_college: bool = Field(
        True,
        title="Recuperar PctSomeCol18_24 por identidad",
        description="Las cuatro categorías educativas de 18–24 años suman 100 %: el valor "
        "ausente se deduce de las otras tres.",
    )
    identity_tolerance: float = Field(
        0.2,
        title="Tolerancia de las identidades contables (puntos %)",
        description="Error de redondeo admitido al comprobar que una composición suma 100 %.",
        ge=0,
    )
    derive_native_multi: bool = Field(
        True,
        title="Derivar población nativa o multirracial",
        description="Calcula el residuo racial 100 − (blanca + negra + asiática + otra).",
    )
    mir_max: float | None = Field(
        1.0,
        title="Razón mortalidad/incidencia máxima",
        description="Si la mortalidad supera a la incidencia (MIR > umbral), la incidencia es "
        "inverosímil (registro incompleto) y se marca como ausente. Vacío para desactivar.",
        gt=0,
    )


class TransformConfig(_Section):
    """Transformaciones de variables."""

    log_population: bool = Field(
        True,
        title="Logaritmo de la población",
        description="La población es extremadamente asimétrica (asimetría ≈ 14); Box-Cox "
        "sugiere λ ≈ 0.",
    )
    log_income: bool = Field(
        True,
        title="Logaritmo de la renta",
        description="La renta es asimétrica a la derecha; el logaritmo lineariza su efecto.",
    )
    study: Literal["log1p", "binary", "none"] = Field(
        "log1p",
        title="Transformación de ensayos clínicos",
        description="El 63 % de los condados no tiene ensayos: log(1 + x) o indicador binario.",
    )
    response: Literal["none", "log"] = Field(
        "none",
        title="Transformación de la respuesta",
        description="La respuesta es casi simétrica; transformarla complica la interpretación "
        "sin mejora sustancial (se documenta con Box-Cox).",
    )


class MissingConfig(_Section):
    """Tratamiento de los datos ausentes."""

    strategy: Literal["complete_case", "multiple_imputation"] = Field(
        "complete_case",
        title="Estrategia principal",
        description="Casos completos (válido si la ausencia no depende de la respuesta dadas "
        "las explicativas) o imputación múltiple con reglas de Rubin.",
    )
    n_imputations: int = Field(
        20,
        title="Número de imputaciones",
        description="Número de conjuntos imputados para la imputación múltiple.",
        ge=2,
        le=200,
    )
    max_iter: int = Field(
        15,
        title="Iteraciones de las ecuaciones encadenadas",
        ge=1,
        le=100,
    )
    sensitivity: bool = Field(
        True,
        title="Análisis de sensibilidad con la estrategia alternativa",
        description="Ajusta también el modelo final con la otra estrategia y compara.",
    )


class VariablesConfig(_Section):
    """Conjunto de variables candidatas del modelo de efectos."""

    candidates: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CANDIDATES),
        title="Explicativas candidatas",
        description="Variables (nombre original) que entran en el modelo máximo antes de la "
        "depuración de colinealidad y la selección.",
    )
    protected: list[str] = Field(
        default_factory=list,
        title="Variables protegidas",
        description="Nunca se eliminan en la poda de colinealidad ni en la selección (p. ej., "
        "la exposición de interés).",
    )
    use_region: bool = Field(
        True,
        title="Incluir la región censal",
        description="Variable categórica con 4 niveles, codificada con 3 variables indicadoras.",
    )
    region_reference: str = Field(
        "Sur",
        title="Región de referencia",
        description="Categoría de referencia de las variables indicadoras de región.",
    )

    @field_validator("candidates", "protected")
    @classmethod
    def _known(cls, value: list[str]) -> list[str]:
        unknown = [v for v in value if v not in VARIABLES]
        if unknown:
            raise ValueError(f"Variables desconocidas: {unknown}")
        forbidden = [v for v in value if v in FORBIDDEN_PREDICTORS]
        if forbidden:
            reasons = "; ".join(f"{v}: {FORBIDDEN_PREDICTORS[v]}" for v in forbidden)
            raise ValueError(f"No pueden usarse como explicativas — {reasons}")
        if len(set(value)) != len(value):
            raise ValueError("Hay variables repetidas.")
        return value

    @field_validator("region_reference")
    @classmethod
    def _region(cls, value: str) -> str:
        if value not in REGIONS:
            raise ValueError(f"Región desconocida «{value}». Opciones: {list(REGIONS)}")
        return value


class OutlierConfig(_Section):
    """Criterios de detección de atípicos."""

    iqr_factor: float = Field(1.5, title="Factor de Tukey (IQR)", gt=0)
    robust_z: float = Field(
        3.5,
        title="Umbral del z robusto (MAD)",
        description="Iglewicz y Hoaglin recomiendan 3,5 para la puntuación z modificada.",
        gt=0,
    )
    mahalanobis_quantile: float = Field(
        0.975,
        title="Cuantil χ² para Mahalanobis robusta",
        gt=0.5,
        lt=1,
    )


class CollinearityConfig(_Section):
    """Umbrales de colinealidad (los del curso)."""

    corr_threshold: float = Field(
        0.8,
        title="Correlación simple preocupante",
        description="Pares de explicativas con |r| por encima de este valor.",
        gt=0,
        lt=1,
    )
    vif_threshold: float = Field(
        10.0,
        title="FIV máximo admitido",
        description="Regla del curso: FIV > 10 (tolerancia < 0,1) indica colinealidad.",
        gt=1,
    )
    condition_index_threshold: float = Field(
        30.0,
        title="Índice de condición preocupante",
        description="A partir de 30 se considera colinealidad fuerte (Belsley).",
        gt=1,
    )


class InferenceConfig(_Section):
    """Parámetros generales de la inferencia."""

    alpha: float = Field(0.05, title="Nivel de significación α", gt=0, lt=0.5)
    bootstrap_reps: int = Field(
        2000,
        title="Réplicas bootstrap (exploración)",
        description="Para intervalos bootstrap de la mediana y de las medias por región.",
        ge=100,
        le=20000,
    )
    multiple_testing: Literal["holm", "fdr_bh", "bonferroni"] = Field(
        "holm",
        title="Corrección por comparaciones múltiples",
    )


class EffectsConfig(_Section):
    """Modelo de estimación de efectos (regresión lineal múltiple)."""

    include_incidence: bool = Field(
        True,
        title="Ajustar por la incidencia",
        description="Si se incluye, los efectos socioeconómicos se interpretan a igualdad de "
        "incidencia (letalidad); si no, como asociación total.",
    )
    center_predictors: bool = Field(
        True,
        title="Centrar las explicativas continuas",
        description="Resta la media: la constante pasa a ser el condado medio y se elimina la "
        "colinealidad no esencial de las interacciones.",
    )
    weighting: Literal["none", "fgls"] = Field(
        "none",
        title="Estimador",
        description="«none»: mínimos cuadrados ordinarios (MCO), el estimador que pide la "
        "orientación oficial. «fgls»: mínimos cuadrados ponderados factibles con varianza "
        "σ²(n) = a + b/población; con MCO se ajusta igualmente como análisis de sensibilidad.",
    )
    fgls_iterations: int = Field(5, title="Iteraciones FGLS", ge=1, le=50)
    covariance: Literal["nonrobust", "HC3", "cluster"] = Field(
        "cluster",
        title="Estimación de la matriz de covarianzas",
        description="«cluster»: errores típicos robustos por conglomerados (estado), válidos "
        "con dependencia dentro del estado y heterocedasticidad.",
    )
    selection: Literal["backward", "forward", "stepwise", "none"] = Field(
        "backward",
        title="Selección de variables",
        description="Estrategias del curso; los contrastes usan la covarianza elegida.",
    )
    alpha_enter: float = Field(0.05, title="α de entrada", gt=0, lt=1)
    alpha_remove: float = Field(0.05, title="α de salida", gt=0, lt=1)
    exposures: list[str] = Field(
        default_factory=lambda: [
            "PctBachDeg25_Over",
            "povertyPercent",
            "medIncome",
        ],
        title="Exposiciones de interés",
        description="Indicadores del nivel socioeconómico (educación, pobreza y renta): "
        "factores de estudio cuyo efecto se quiere estimar sin confusión. El criterio del "
        "cambio en la estimación sólo vigila sus coeficientes.",
    )
    confounding_threshold: float = Field(
        0.10,
        title="Cambio relativo que indica confusión",
        description="Una variable eliminada se reincorpora si su retirada cambia algún "
        "coeficiente retenido más de este porcentaje.",
        ge=0,
        le=1,
    )
    interactions: list[list[str]] = Field(
        default_factory=lambda: [list(p) for p in DEFAULT_INTERACTIONS],
        title="Interacciones candidatas",
        description="Pares de variables (o variable × región) cuya interacción se contrasta "
        "tras la selección, respetando el principio jerárquico.",
    )
    interaction_correction: Literal["holm", "bonferroni", "none"] = Field(
        "holm",
        title="Corrección de los contrastes de interacción",
    )
    bootstrap_reps: int = Field(
        999,
        title="Réplicas del bootstrap por conglomerados",
        description="Remuestreo de estados completos para los intervalos de los coeficientes.",
        ge=0,
        le=10000,
    )
    influence_sensitivity: bool = Field(
        True,
        title="Reajustar sin observaciones influyentes",
    )

    @field_validator("interactions")
    @classmethod
    def _pairs(cls, value: list[list[str]]) -> list[list[str]]:
        for pair in value:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(f"Interacción mal formada: {pair}")
            for v in pair:
                if v != "region" and v not in VARIABLES:
                    raise ValueError(f"Variable desconocida en interacción: {v}")
        return value


class PredictiveConfig(_Section):
    """Modelo predictivo."""

    enabled: bool = Field(True, title="Ajustar modelos predictivos")
    test_size: float = Field(
        0.2,
        title="Proporción de prueba",
        description="Fracción reservada que no se toca hasta la evaluación final.",
        gt=0.05,
        lt=0.5,
    )
    cv_folds: int = Field(10, title="Pliegues de validación cruzada", ge=3, le=20)
    group_cv: bool = Field(
        True,
        title="Validación cruzada agrupada por estado",
        description="Mide la generalización a estados no vistos, más exigente que la aleatoria.",
    )
    use_state: bool = Field(
        True,
        title="Usar el estado como predictor",
        description="Indicadoras de estado (sólo ayudan a predecir condados de estados vistos).",
    )
    models: list[PredictiveModelName] = Field(
        default_factory=lambda: [  # type: ignore[arg-type]
            "baseline",
            "ols_effects",
            "ols_full",
            "ridge",
            "lasso",
            "elasticnet",
            "random_forest",
            "gradient_boosting",
        ],
        title="Modelos a comparar",
    )
    selection_rule: Literal["min", "one_se"] = Field(
        "min",
        title="Regla de elección del modelo",
        description="«min»: menor RMSE medio en validación cruzada; «one_se»: el modelo más "
        "sencillo cuyo RMSE está a menos de un error típico del mejor.",
    )
    conformal_alpha: float = Field(
        0.1,
        title="α de los intervalos de predicción conformales",
        description="Intervalos con cobertura garantizada 1 − α sin suponer normalidad.",
        gt=0,
        lt=0.5,
    )
    n_jobs: int = Field(-1, title="Procesos en paralelo (−1 = todos)")


class AnalysisConfig(_Section):
    """Configuración completa de una corrida."""

    meta: MetaConfig = Field(default_factory=MetaConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    missing: MissingConfig = Field(default_factory=MissingConfig)
    variables: VariablesConfig = Field(default_factory=VariablesConfig)
    outliers: OutlierConfig = Field(default_factory=OutlierConfig)
    collinearity: CollinearityConfig = Field(default_factory=CollinearityConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    effects: EffectsConfig = Field(default_factory=EffectsConfig)
    predictive: PredictiveConfig = Field(default_factory=PredictiveConfig)

    @model_validator(mode="after")
    def _coherent(self) -> AnalysisConfig:
        missing = [p for p in self.variables.protected if p not in self.variables.candidates]
        if missing:
            raise ValueError(f"Variables protegidas que no son candidatas: {missing}")
        if not self.effects.include_incidence and "incidenceRate" in self.variables.protected:
            raise ValueError("No se puede proteger incidenceRate si se excluye la incidencia.")
        for pair in self.effects.interactions:
            if "region" in pair and not self.variables.use_region:
                raise ValueError(f"La interacción {pair} requiere incluir la región.")
        return self

    # ------------------------------------------------------------------ serialización
    def canonical_json(self) -> str:
        """JSON canónico (claves ordenadas) sobre el que se calcula la huella."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)

    def fingerprint(self) -> str:
        """Huella SHA-256 (12 primeros caracteres) de la configuración."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:12]

    def to_yaml(self) -> str:
        """Serializa a YAML conservando el orden de las secciones."""
        return yaml.safe_dump(
            self.model_dump(mode="json"), allow_unicode=True, sort_keys=False, width=100
        )

    def save(self, path: Path) -> None:
        """Guarda la configuración en YAML."""
        path.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> AnalysisConfig:
        """Construye la configuración desde un diccionario (claves ausentes = por defecto)."""
        return cls.model_validate(data or {})

    @classmethod
    def load(cls, path: Path) -> AnalysisConfig:
        """Lee una configuración YAML."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_mapping(data)


def merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Fusión recursiva de diccionarios: ``overrides`` prevalece sobre ``base``."""
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_overrides(out[key], value)
        else:
            out[key] = value
    return out


def parse_dotted_override(expr: str) -> dict[str, Any]:
    """Convierte ``"effects.alpha_remove=0.1"`` en ``{"effects": {"alpha_remove": 0.1}}``.

    El valor se interpreta como YAML, de modo que admite números, booleanos y listas.
    """
    if "=" not in expr:
        raise ValueError(f"Se esperaba clave=valor, se recibió «{expr}»")
    key, raw = expr.split("=", 1)
    value = yaml.safe_load(raw)
    node: dict[str, Any] = {}
    cursor = node
    parts = key.strip().split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return node
