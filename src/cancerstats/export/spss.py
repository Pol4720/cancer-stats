"""Sintaxis de SPSS que reproduce la depuración y el modelo final MCO.

El curso trabaja con SPSS y el profesor entregó los datos en ``practica.sav``. Esta sintaxis,
generada a partir de la corrida, aplica sobre ese fichero las mismas decisiones de
depuración, construye las mismas variables (región, transformaciones, centrado e
interacciones) y ajusta el mismo modelo con el procedimiento REGRESSION, de modo que los
coeficientes y errores típicos clásicos coinciden con los de la tabla «salida tipo SPSS»
del informe.
"""

from __future__ import annotations

from typing import Any

from cancerstats.dictionary import REGIONS

Results = dict[str, Any]

_SHORT = {"Noreste": "NE", "Medio Oeste": "MO", "Sur": "SU", "Oeste": "OE"}


def _name(term: str) -> str:
    """Nombre de variable SPSS (≤ 64 caracteres, sin corchetes ni dos puntos)."""
    out = term
    for lev, short in _SHORT.items():
        out = out.replace(f"region[{lev}]", f"reg_{short}")
    return out.replace(":", "_x_")


def _wrap(items: list[str], indent: str = "    ", width: int = 78) -> str:
    lines: list[str] = []
    current = indent
    for it in items:
        if len(current) + len(it) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = indent
        current += it + " "
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def syntax(res: Results, data_file: str = "practica.sav") -> str:
    """Genera el texto de la sintaxis."""
    eff = res["effects"]
    fin = eff["final"]
    terms: list[str] = list(fin["terms"])
    centers: dict[str, float] = fin["centers"]
    exact = res["cleaning"].get("sentinels_exact") or {}
    sentinels = {
        col: float(exact.get(col, value)) for col, value in res["cleaning"]["sentinels"].items()
    }
    reference = next(
        lev for lev in REGIONS if not any(c["term"] == f"region[{lev}]" for c in fin["coef"])
    )
    mir = next((d for d in res["cleaning"]["decisions"] if d["id"] == "D08"), None)

    out: list[str] = [
        "* " + "=" * 74 + ".",
        "* Reproducción en SPSS del modelo final MCO (proyecto cancer-stats).",
        f"* Generado automáticamente a partir de la corrida {res['run_id']}.",
        "* Los coeficientes y errores típicos clásicos deben coincidir con la tabla",
        "* «salida tipo SPSS» del informe. Ejecutar con el fichero oficial en la",
        "* misma carpeta que esta sintaxis.",
        "* " + "=" * 74 + ".",
        "",
        "SET DECIMAL=DOT.",
        f"GET FILE='{data_file}'.",
        "DATASET NAME practica WINDOW=FRONT.",
        "",
        "* D01: Geography separada en condado y estado (por la última coma).",
        "STRING state (A40).",
        "COMPUTE #coma = CHAR.RINDEX(Geography, ',').",
        "COMPUTE state = LTRIM(CHAR.SUBSTR(Geography, #coma + 1)).",
        "EXECUTE.",
        "",
        "* D02: Notificadomuerte = avgAnnCount - avgDeathsPerYear (fuga): no entra en el modelo.",
        "",
    ]
    for col, value in sentinels.items():
        out += [
            f"* D03: valor centinela de {col}.",
            # Tolerancia holgada: las mediciones reales tienen uno o dos decimales.
            f"IF (ABS({col} - {value!r}) < 0.0001) {col} = $SYSMIS.",
        ]
    out += [
        "",
        "* D04: edad mediana registrada en meses.",
        "IF (MedianAge > 100) MedianAge = RND(MedianAge / 12 * 10) / 10.",
        "* D05: tamaño medio del hogar dividido por 100.",
        "IF (AvgHouseholdSize < 1) AvgHouseholdSize = AvgHouseholdSize * 100.",
        "* D06: PctSomeCol18_24 recuperada por identidad contable.",
        "IF (MISSING(PctSomeCol18_24)) PctSomeCol18_24 = MAX(0, 100 - PctNoHS18_24 - PctHS18_24 - "
        "PctBachDeg18_24).",
        "* D07: población nativa o multirracial (residuo racial).",
        "COMPUTE PctNativeMulti = MAX(0, 100 - (PctWhite + PctBlack + PctAsian + PctOtherRace)).",
    ]
    if mir and mir.get("applied"):
        out += [
            "* D08: incidencia inverosímil (mortalidad mayor que incidencia).",
            "IF (TARGET_deathRate / incidenceRate > 1) incidenceRate = $SYSMIS.",
        ]
    out += [
        "* D09: región censal.",
        "STRING region (A12).",
    ]
    for lev, states in REGIONS.items():
        quoted = [f"'{s}'" for s in states]
        out.append(f"IF (ANY(RTRIM(state), {', '.join(quoted)})) region = '{lev}'.")
    out += [
        "* D10: transformaciones.",
        "COMPUTE logPop = LN(popEst2015).",
        "COMPUTE logIncome = LN(medIncome).",
        "COMPUTE logStudy = LN(1 + studyPerCap).",
        "EXECUTE.",
        "",
        "* Indicadoras de región (referencia: " + reference + ").",
    ]
    for lev in REGIONS:
        if lev != reference:
            out.append(f"COMPUTE reg_{_SHORT[lev]} = (RTRIM(region) = '{lev}').")
    out += ["", "* Explicativas continuas centradas en la media de la muestra del modelo final."]
    numeric = [t for t in terms if t in centers]
    for v in numeric:
        out.append(f"COMPUTE c_{v} = {v} - {centers[v]:.10f}.")
    predictors: list[str] = [f"c_{v}" for v in numeric]
    if "region" in terms:
        predictors += [f"reg_{_SHORT[lev]}" for lev in REGIONS if lev != reference]
    inter = [t for t in terms if ":" in t]
    if inter:
        out += ["", "* Interacciones (productos de variables centradas)."]
    for t in inter:
        a, b = t.split(":")
        if b == "region":
            for lev in REGIONS:
                if lev == reference:
                    continue
                name = f"c_{a}_x_{_SHORT[lev]}"
                out.append(f"COMPUTE {name} = c_{a} * reg_{_SHORT[lev]}.")
                predictors.append(name)
        else:
            name = f"c_{a}_x_c_{b}"
            out.append(f"COMPUTE {name} = c_{a} * c_{b}.")
            predictors.append(name)
    out += [
        "EXECUTE.",
        "",
        "* Modelo final por mínimos cuadrados ordinarios, con los diagnósticos de la",
        "* orientación: linealidad, independencia (Durbin-Watson), homocedasticidad,",
        "* normalidad de los residuos, colinealidad (tolerancia, FIV e índices de condición)",
        "* y casos atípicos e influyentes.",
        "REGRESSION",
        "  /MISSING LISTWISE",
        "  /STATISTICS COEFF OUTS CI(95) R ANOVA COLLIN TOL",
        "  /CRITERIA=PIN(.05) POUT(.10)",
        "  /NOORIGIN",
        "  /DEPENDENT TARGET_deathRate",
        "  /METHOD=ENTER",
        _wrap(predictors),
        "  /SCATTERPLOT=(*ZRESID, *ZPRED) (*SRESID, *ZPRED)",
        "  /RESIDUALS DURBIN HISTOGRAM(ZRESID) NORMPROB(ZRESID)",
        "  /CASEWISE PLOT(ZRESID) OUTLIERS(3)",
        "  /SAVE PRED ZRESID SRESID COOK LEVER SDBETA.",
        "",
        "* Contraste de Breusch-Pagan (versión de Koenker) a mano: regresión de los",
        "* residuos al cuadrado sobre las explicativas; LM = n·R².",
        "COMPUTE res2 = (ZRE_1) ** 2.",
        "REGRESSION /DEPENDENT res2 /METHOD=ENTER",
        _wrap(predictors) + ".",
        "",
    ]
    return "\n".join(out) + "\n"
