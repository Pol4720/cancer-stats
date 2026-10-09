"""Diccionario de variables: metadatos en castellano de cada columna.

Este módulo es la única fuente de verdad sobre qué significa cada variable, en qué
unidades se mide, a qué constructo pertenece y qué papel desempeña en el análisis. La
interfaz, el informe y la guía consumen estas etiquetas, de modo que un cambio aquí se
propaga a todos los entregables.

Fuentes (según la descripción de Kaggle):

* (a) Registros de cáncer, años 2010–2016 (State Cancer Profiles, cancer.gov).
* (b) Estimaciones censales de 2013 (American Community Survey, census.gov).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Kind = Literal["continua", "porcentaje", "conteo", "categórica", "ordinal", "identificador"]
Source = Literal["a", "b", "derivada"]


@dataclass(frozen=True)
class Variable:
    """Metadatos de una variable."""

    name: str
    label: str
    description: str
    unit: str
    kind: Kind
    source: Source
    group: str

    def to_dict(self) -> dict[str, str]:
        """Representación serializable."""
        return asdict(self)


RESPONSE = "TARGET_deathRate"

_VARS: tuple[Variable, ...] = (
    Variable(
        "TARGET_deathRate",
        "Mortalidad por cáncer",
        "Tasa media anual de mortalidad por cáncer ajustada por edad, por 100 000 habitantes.",
        "muertes/100 000 hab.",
        "continua",
        "a",
        "Respuesta",
    ),
    Variable(
        "avgAnnCount",
        "Casos anuales",
        "Número medio de casos de cáncer diagnosticados al año.",
        "casos/año",
        "conteo",
        "a",
        "Carga de cáncer",
    ),
    Variable(
        "avgDeathsPerYear",
        "Muertes anuales",
        "Número medio de muertes por cáncer al año (numerador de la respuesta).",
        "muertes/año",
        "conteo",
        "a",
        "Carga de cáncer",
    ),
    Variable(
        "incidenceRate",
        "Incidencia de cáncer",
        "Tasa media anual de diagnósticos de cáncer ajustada por edad, por 100 000 habitantes.",
        "casos/100 000 hab.",
        "continua",
        "a",
        "Carga de cáncer",
    ),
    Variable(
        "medIncome",
        "Renta mediana",
        "Renta mediana de los hogares del condado.",
        "USD",
        "continua",
        "b",
        "Socioeconómico",
    ),
    Variable(
        "popEst2015",
        "Población",
        "Población estimada del condado en 2015.",
        "habitantes",
        "conteo",
        "b",
        "Tamaño",
    ),
    Variable(
        "povertyPercent",
        "Pobreza",
        "Porcentaje de la población por debajo del umbral de pobreza.",
        "%",
        "porcentaje",
        "b",
        "Socioeconómico",
    ),
    Variable(
        "studyPerCap",
        "Ensayos clínicos per cápita",
        "Número de ensayos clínicos relacionados con el cáncer per cápita en el condado.",
        "ensayos/100 000 hab.",
        "continua",
        "a",
        "Investigación clínica",
    ),
    Variable(
        "binnedInc",
        "Decil de renta",
        "Intervalo (decil) al que pertenece la renta mediana del condado.",
        "intervalo de USD",
        "ordinal",
        "b",
        "Socioeconómico",
    ),
    Variable(
        "MedianAge",
        "Edad mediana",
        "Edad mediana de los residentes del condado.",
        "años",
        "continua",
        "b",
        "Demografía",
    ),
    Variable(
        "MedianAgeMale",
        "Edad mediana (hombres)",
        "Edad mediana de los residentes varones.",
        "años",
        "continua",
        "b",
        "Demografía",
    ),
    Variable(
        "MedianAgeFemale",
        "Edad mediana (mujeres)",
        "Edad mediana de las residentes mujeres.",
        "años",
        "continua",
        "b",
        "Demografía",
    ),
    Variable(
        "AvgHouseholdSize",
        "Tamaño medio del hogar",
        "Número medio de personas por hogar.",
        "personas",
        "continua",
        "b",
        "Hogar",
    ),
    Variable(
        "PercentMarried",
        "Población casada",
        "Porcentaje de residentes casados.",
        "%",
        "porcentaje",
        "b",
        "Hogar",
    ),
    Variable(
        "PctNoHS18_24",
        "Sin bachillerato (18–24)",
        "Porcentaje de residentes de 18 a 24 años cuyo nivel máximo es inferior al bachillerato.",
        "%",
        "porcentaje",
        "b",
        "Educación",
    ),
    Variable(
        "PctHS18_24",
        "Bachillerato (18–24)",
        "Porcentaje de residentes de 18 a 24 años con bachillerato (high school) como máximo.",
        "%",
        "porcentaje",
        "b",
        "Educación",
    ),
    Variable(
        "PctSomeCol18_24",
        "Estudios superiores incompletos (18–24)",
        "Porcentaje de residentes de 18 a 24 años con estudios universitarios sin terminar.",
        "%",
        "porcentaje",
        "b",
        "Educación",
    ),
    Variable(
        "PctBachDeg18_24",
        "Grado universitario (18–24)",
        "Porcentaje de residentes de 18 a 24 años con grado universitario (bachelor).",
        "%",
        "porcentaje",
        "b",
        "Educación",
    ),
    Variable(
        "PctHS25_Over",
        "Bachillerato (25+)",
        "Porcentaje de residentes de 25 años o más con bachillerato como nivel máximo.",
        "%",
        "porcentaje",
        "b",
        "Educación",
    ),
    Variable(
        "PctBachDeg25_Over",
        "Grado universitario (25+)",
        "Porcentaje de residentes de 25 años o más con grado universitario (bachelor).",
        "%",
        "porcentaje",
        "b",
        "Educación",
    ),
    Variable(
        "PctEmployed16_Over",
        "Empleo (16+)",
        "Porcentaje de residentes de 16 años o más con empleo.",
        "%",
        "porcentaje",
        "b",
        "Empleo",
    ),
    Variable(
        "PctUnemployed16_Over",
        "Desempleo (16+)",
        "Porcentaje de residentes de 16 años o más desempleados.",
        "%",
        "porcentaje",
        "b",
        "Empleo",
    ),
    Variable(
        "PctPrivateCoverage",
        "Seguro privado",
        "Porcentaje de residentes con cobertura sanitaria privada.",
        "%",
        "porcentaje",
        "b",
        "Cobertura sanitaria",
    ),
    Variable(
        "PctPrivateCoverageAlone",
        "Solo seguro privado",
        "Porcentaje de residentes con cobertura privada exclusivamente (sin ayuda pública).",
        "%",
        "porcentaje",
        "b",
        "Cobertura sanitaria",
    ),
    Variable(
        "PctEmpPrivCoverage",
        "Seguro privado vía empleador",
        "Porcentaje de residentes con cobertura privada proporcionada por el empleador.",
        "%",
        "porcentaje",
        "b",
        "Cobertura sanitaria",
    ),
    Variable(
        "PctPublicCoverage",
        "Seguro público",
        "Porcentaje de residentes con cobertura sanitaria pública.",
        "%",
        "porcentaje",
        "b",
        "Cobertura sanitaria",
    ),
    Variable(
        "PctPublicCoverageAlone",
        "Solo seguro público",
        "Porcentaje de residentes con cobertura pública exclusivamente.",
        "%",
        "porcentaje",
        "b",
        "Cobertura sanitaria",
    ),
    Variable(
        "PctWhite",
        "Población blanca",
        "Porcentaje de residentes que se identifican como blancos.",
        "%",
        "porcentaje",
        "b",
        "Composición racial",
    ),
    Variable(
        "PctBlack",
        "Población negra",
        "Porcentaje de residentes que se identifican como negros.",
        "%",
        "porcentaje",
        "b",
        "Composición racial",
    ),
    Variable(
        "PctAsian",
        "Población asiática",
        "Porcentaje de residentes que se identifican como asiáticos.",
        "%",
        "porcentaje",
        "b",
        "Composición racial",
    ),
    Variable(
        "PctOtherRace",
        "Otra raza",
        "Porcentaje de residentes que se identifican con otra categoría racial.",
        "%",
        "porcentaje",
        "b",
        "Composición racial",
    ),
    Variable(
        "PctMarriedHouseholds",
        "Hogares de matrimonios",
        "Porcentaje de hogares formados por un matrimonio.",
        "%",
        "porcentaje",
        "b",
        "Hogar",
    ),
    Variable(
        "BirthRate",
        "Natalidad",
        "Nacimientos vivos en relación con el número de mujeres del condado.",
        "%",
        "continua",
        "b",
        "Demografía",
    ),
    Variable(
        "Geography",
        "Condado",
        "Nombre del condado (o entidad equivalente: parroquia, ciudad independiente…).",
        "—",
        "identificador",
        "b",
        "Geografía",
    ),
    Variable(
        "state",
        "Estado",
        "Estado al que pertenece el condado (51 entidades, incluido el Distrito de Columbia).",
        "—",
        "categórica",
        "b",
        "Geografía",
    ),
    Variable(
        "Notificadomuerte",
        "Casos menos muertes",
        "Variable añadida en el fichero oficial de SPSS; la validación demuestra que es "
        "exactamente avgAnnCount − avgDeathsPerYear (casos anuales menos muertes anuales).",
        "casos/año",
        "conteo",
        "derivada",
        "Carga de cáncer",
    ),
    # --- Variables derivadas en la depuración -------------------------------------------
    Variable(
        "PctNativeMulti",
        "Población nativa o multirracial",
        "Residuo racial 100 − (blanca + negra + asiática + otra): población nativa americana, "
        "nativa de Alaska o de dos o más razas.",
        "%",
        "porcentaje",
        "derivada",
        "Composición racial",
    ),
    Variable(
        "region",
        "Región censal",
        "Región censal de EE. UU. (Noreste, Medio Oeste, Sur, Oeste) asignada a partir del estado.",
        "—",
        "categórica",
        "derivada",
        "Geografía",
    ),
    Variable(
        "logPop",
        "log(Población)",
        "Logaritmo natural de la población de 2015.",
        "log(hab.)",
        "continua",
        "derivada",
        "Tamaño",
    ),
    Variable(
        "logIncome",
        "log(Renta mediana)",
        "Logaritmo natural de la renta mediana de los hogares.",
        "log(USD)",
        "continua",
        "derivada",
        "Socioeconómico",
    ),
    Variable(
        "logStudy",
        "log(1 + ensayos per cápita)",
        "Transformación log(1 + x) de los ensayos clínicos per cápita (63 % de ceros).",
        "log(ensayos)",
        "continua",
        "derivada",
        "Investigación clínica",
    ),
    Variable(
        "anyStudy",
        "Algún ensayo clínico",
        "Indicador de que el condado tiene al menos un ensayo clínico (1) o ninguno (0).",
        "0/1",
        "categórica",
        "derivada",
        "Investigación clínica",
    ),
)

VARIABLES: dict[str, Variable] = {v.name: v for v in _VARS}

REGIONS: dict[str, tuple[str, ...]] = {
    "Noreste": (
        "Connecticut",
        "Maine",
        "Massachusetts",
        "New Hampshire",
        "Rhode Island",
        "Vermont",
        "New Jersey",
        "New York",
        "Pennsylvania",
    ),
    "Medio Oeste": (
        "Illinois",
        "Indiana",
        "Michigan",
        "Ohio",
        "Wisconsin",
        "Iowa",
        "Kansas",
        "Minnesota",
        "Missouri",
        "Nebraska",
        "North Dakota",
        "South Dakota",
    ),
    "Sur": (
        "Delaware",
        "District of Columbia",
        "Florida",
        "Georgia",
        "Maryland",
        "North Carolina",
        "South Carolina",
        "Virginia",
        "West Virginia",
        "Alabama",
        "Kentucky",
        "Mississippi",
        "Tennessee",
        "Arkansas",
        "Louisiana",
        "Oklahoma",
        "Texas",
    ),
    "Oeste": (
        "Arizona",
        "Colorado",
        "Idaho",
        "Montana",
        "Nevada",
        "New Mexico",
        "Utah",
        "Wyoming",
        "Alaska",
        "California",
        "Hawaii",
        "Oregon",
        "Washington",
    ),
}
"""Regiones censales oficiales de la Oficina del Censo de EE. UU."""

STATE_TO_REGION: dict[str, str] = {s: r for r, states in REGIONS.items() for s in states}


def label(name: str) -> str:
    """Etiqueta legible de una variable, o el propio nombre si no está catalogada.

    Los términos de interacción ``a:b`` se etiquetan como ``a × b``.
    """
    if ":" in name:
        return " × ".join(label(p) for p in name.split(":"))
    var = VARIABLES.get(name)
    return var.label if var else name


def unit(name: str) -> str:
    """Unidad de medida de una variable."""
    var = VARIABLES.get(name)
    return var.unit if var else ""


def as_records() -> list[dict[str, str]]:
    """Diccionario completo como lista de registros (para la interfaz y el informe)."""
    return [v.to_dict() for v in _VARS]
