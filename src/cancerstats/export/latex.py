"""Generación del material LaTeX del informe a partir de una corrida.

Produce en ``report/generado/``:

* ``resultados.tex``: una macro por cifra citada en el texto, accesible como ``\\res{clave}``.
  Si una corrida posterior deja de producir una cifra (p. ej., porque una variable sale del
  modelo), el informe sigue compilando: la clave ausente se imprime en rojo y se emite una
  advertencia que señala el párrafo que hay que revisar.
* ``tablas/*.tex``: tablas completas (con título y etiqueta) listas para ``\\input``.

El texto interpretativo de las secciones lo escribe el investigador; las cifras, no.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cancerstats.dictionary import label, unit
from cancerstats.export.fmt import ci, integer, num, pcell, pvalue, tex_escape

Results = dict[str, Any]

ALIAS: dict[str, str] = {
    "incidenceRate": "incidencia",
    "PctBachDeg25_Over": "grado25",
    "PctHS25_Over": "bach25",
    "PctHS18_24": "bach1824",
    "PctNoHS18_24": "sinbach1824",
    "PctBachDeg18_24": "grado1824",
    "PctSomeCol18_24": "supinc1824",
    "PctUnemployed16_Over": "desempleo",
    "PctEmployed16_Over": "empleo",
    "PctPublicCoverageAlone": "solopublico",
    "PctPublicCoverage": "publico",
    "PctPrivateCoverage": "privado",
    "PctPrivateCoverageAlone": "soloprivado",
    "PctEmpPrivCoverage": "privempleador",
    "PctOtherRace": "otraraza",
    "PctNativeMulti": "nativa",
    "PctBlack": "negra",
    "PctAsian": "asiatica",
    "PctWhite": "blanca",
    "MedianAge": "edad",
    "AvgHouseholdSize": "hogar",
    "PercentMarried": "casados",
    "PctMarriedHouseholds": "hogarescasados",
    "BirthRate": "natalidad",
    "logStudy": "ensayos",
    "anyStudy": "ensayos",
    "studyPerCap": "ensayos",
    "logPop": "poblacion",
    "popEst2015": "poblacion",
    "logIncome": "renta",
    "medIncome": "renta",
    "povertyPercent": "pobreza",
    "TARGET_deathRate": "mortalidad",
}
REGION_ALIAS = {"Sur": "sur", "Noreste": "noreste", "Medio Oeste": "medio-oeste", "Oeste": "oeste"}
MODEL_ALIAS = {
    "baseline": "referencia",
    "ols_effects": "mco-efectos",
    "ols_full": "mco-completo",
    "ridge": "ridge",
    "lasso": "lasso",
    "elasticnet": "red-elastica",
    "random_forest": "bosque",
    "gradient_boosting": "potenciacion",
}


def alias(term: str) -> str:
    """Clave corta de un término para las macros."""
    if term.startswith("region["):
        return REGION_ALIAS.get(term[7:-1], term[7:-1].lower())
    if ":" in term:
        return "x".join(alias(p) for p in term.split(":"))
    return ALIAS.get(term, term.lower().replace("_", ""))


def join_es(items: list[str]) -> str:
    """Lista en castellano: «a, b y c»."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def _get(seq: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    for item in seq:
        if all(item.get(k) == v for k, v in kw.items()):
            return item
    return None


# ============================================================================ macros
def macros(res: Results) -> dict[str, str]:
    """Todas las cifras citadas en el informe, ya formateadas para LaTeX."""
    m: dict[str, str] = {}
    ing = res["ingest"]
    man = res["manifest"]
    m["n-condados"] = integer(ing["n_rows"])
    m["n-columnas"] = integer(ing["n_columns_raw"])
    m["sha-corto"] = r"\texttt{" + ing["sha256"][:16] + "}"
    m["codificacion"] = (
        "Mac Roman" if ing["encoding"] == "mac_roman" else tex_escape(ing["encoding"])
    )

    # Validación --------------------------------------------------------------------
    v = res["validation"]
    m["n-reglas"] = integer(v["summary_raw"]["total"])
    m["n-reglas-error"] = integer(v["summary_raw"]["error"])
    m["n-reglas-advertencia"] = integer(v["summary_raw"]["advertencia"])
    m["n-reglas-ok"] = integer(v["summary_raw"]["ok"])
    m["n-reglas-error-dep"] = integer(v["summary_clean"]["error"])
    for r_raw, r_clean in zip(v["raw"], v["clean"], strict=True):
        m[f"viol-{r_raw['id'].lower()}"] = integer(r_raw["n_violations"])
        m[f"viol-{r_raw['id'].lower()}-dep"] = integer(r_clean["n_violations"])

    # Depuración ------------------------------------------------------------------
    dec = {d["id"]: d for d in res["cleaning"]["decisions"]}
    sent = res["cleaning"]["sentinels"]
    m["n-centinela"] = integer(dec["D03"]["n_affected"])
    if "avgAnnCount" in sent:
        m["centinela-casos"] = num(sent["avgAnnCount"], 6)
    if "incidenceRate" in sent:
        m["centinela-incidencia"] = num(sent["incidenceRate"], 7)
    m["estados-centinela"] = join_es(list(dec["D03"]["evidence"].get("estados", {})))
    m["n-edad-meses"] = integer(dec["D04"]["n_affected"])
    m["razon-edad"] = num(dec["D04"]["evidence"].get("razon_media", float("nan")), 3)
    m["dt-razon-edad"] = num(dec["D04"]["evidence"].get("razon_dt", float("nan")), 3)
    m["n-hogar"] = integer(dec["D05"]["n_affected"])
    m["n-somecol"] = integer(dec["D06"]["n_affected"])
    m["pct-somecol"] = num(100 * dec["D06"]["n_affected"] / ing["n_rows"], 1)
    m["error-identidad"] = num(dec["D06"]["evidence"].get("error_maximo_identidad", 0.0), 2)
    m["n-identidad-completos"] = integer(dec["D06"]["evidence"].get("casos_completos", 0))
    m["n-nativa"] = integer(dec["D07"]["n_affected"])
    m["n-mir"] = integer(dec["D08"]["n_affected"])
    m["condados-mir"] = tex_escape(join_es(dec["D08"]["affected"])) or "ninguno"
    m["mir-mediana"] = num(dec["D08"]["evidence"].get("mir_mediana", float("nan")), 2)

    # Ausentes --------------------------------------------------------------------
    miss = res["missing"]
    for key, lt in (("aleatorio", miss.get("little_random")), ("todo", miss.get("little_all"))):
        if lt:
            m[f"little-{key}-est"] = num(lt["statistic"], 1)
            m[f"little-{key}-gl"] = integer(lt["df"])
            m[f"little-{key}-p"] = pvalue(lt["p_value"])
            m[f"little-{key}-patrones"] = integer(lt["n_patterns"])
    for row in miss["by_column"]:
        a = alias(row["variable"])
        m[f"n-ausente-{a}"] = integer(row["n"])
        m[f"pct-ausente-{a}"] = num(row["pct"], 1)
    for row in miss["by_state"]:
        a = alias(row["variable"])
        m[f"chi-estado-{a}-p"] = (
            pvalue(row["p"]) if row["agrupacion"] == "state" else m.get(f"chi-estado-{a}-p", "")
        )
    complete = next((p for p in miss["patterns"] if not p["faltan"]), None)
    if complete:
        m["n-completos-todo"] = integer(complete["n"])
        m["pct-completos-todo"] = num(complete["pct"], 1)

    # Exploración -----------------------------------------------------------------
    ex = res["exploration"]
    r = ex["response"]
    m["media"] = num(r["mean_ci"]["estimate"], 2)
    m["media-ic"] = ci(r["mean_ci"]["lower"], r["mean_ci"]["upper"], 2)
    m["mediana"] = num(r["median_ci_order"]["estimate"], 1)
    m["mediana-ic"] = ci(r["median_ci_order"]["lower"], r["median_ci_order"]["upper"], 1)
    m["mediana-ic-boot"] = ci(r["median_ci_boot"]["lower"], r["median_ci_boot"]["upper"], 1)
    m["cobertura-mediana"] = num(100 * r["median_ci_order"]["coverage"], 1)
    m["dt"] = num(r["variance_ci"]["sd"], 2)
    m["dt-ic"] = ci(r["variance_ci"]["sd_lower"], r["variance_ci"]["sd_upper"], 2)
    m["varianza"] = num(r["variance_ci"]["estimate"], 1)
    m["varianza-ic"] = ci(r["variance_ci"]["lower"], r["variance_ci"]["upper"], 1)
    m["recortada"] = num(r["robust"]["trimmed_mean_10"], 2)
    m["huber"] = num(r["robust"]["huber"], 2)
    m["meda"] = num(r["robust"]["meda"], 1)
    m["asimetria"] = num(r["shape"]["skewness"], 3)
    m["curtosis"] = num(r["shape"]["kurtosis_excess"], 3)
    m["boxcox-lambda"] = num(r["boxcox"]["lambda"], 3)
    m["boxcox-ic"] = ci(r["boxcox"]["lower"], r["boxcox"]["upper"], 3)
    for t in r["normality"]:
        key = {
            "Shapiro-Wilk": "sw",
            "Kolmogorov-Smirnov-Lilliefors": "lilliefors",
            "D'Agostino-Pearson K²": "dagostino",
            "Jarque-Bera": "jb",
            "Anderson-Darling": "ad",
        }.get(t["test"], "chi")
        m[f"norm-{key}-est"] = num(t["statistic"], 3)
        m[f"norm-{key}-p"] = pvalue(t["p"]) if t.get("p") is not None else "---"
    m["rachas"] = integer(r["runs_file_order"]["runs"])
    m["rachas-esperadas"] = num(r["runs_file_order"]["expected"], 1)
    m["rachas-z"] = num(r["runs_file_order"]["z"], 2)
    m["rachas-p"] = pvalue(r["runs_file_order"]["p"])
    lo = r["extremes"]["lowest"][0]
    hi = r["extremes"]["highest"][0]
    m["min-condado"] = tex_escape(lo["county_id"])
    m["min-valor"] = num(lo["TARGET_deathRate"], 1)
    m["max-condado"] = tex_escape(hi["county_id"])
    m["max-valor"] = num(hi["TARGET_deathRate"], 1)

    reg = ex["region"]
    for g in reg["groups"]:
        a = REGION_ALIAS[g["group"]]
        m[f"media-{a}"] = num(g["mean"], 1)
        m[f"media-{a}-ic"] = ci(g["lower"], g["upper"], 1)
        m[f"n-{a}"] = integer(g["n"])
    m["anova-f"] = num(reg["anova"]["statistic"], 1)
    m["anova-p"] = pvalue(reg["anova"]["p"])
    m["eta2-region"] = num(reg["anova"]["eta2"], 3)
    m["pct-region"] = num(100 * reg["anova"]["eta2"], 1)
    m["levene-est"] = num(reg["levene"]["statistic"], 2)
    m["levene-p"] = pvalue(reg["levene"]["p"])
    m["welch-f"] = num(reg["welch"]["statistic"], 1)
    m["welch-gl2"] = num(reg["welch"]["df2"], 1)
    m["welch-p"] = pvalue(reg["welch"]["p"])
    m["kruskal-h"] = num(reg["kruskal"]["statistic"], 1)
    m["kruskal-p"] = pvalue(reg["kruskal"]["p"])
    mw_no_mo = _get(reg["pairs"], a="Medio Oeste", b="Noreste")
    if mw_no_mo:
        m["p-tukey-mo-no"] = pvalue(mw_no_mo.get("p_tukey"))
        m["p-mw-mo-no"] = pvalue(mw_no_mo["p_mann_whitney_adj"])
    m["eta2-estado"] = num(ex["state"]["eta2"], 3)
    m["pct-estado"] = num(100 * ex["state"]["eta2"], 1)
    m["anova-estado-f"] = num(ex["state"]["anova_f"], 1)
    hp = ex["high_poverty"]
    m["pobreza-n-alta"] = integer(hp["n"][0])
    m["pobreza-media-alta"] = num(hp["mean"][0], 1)
    m["pobreza-media-baja"] = num(hp["mean"][1], 1)
    m["pobreza-dif"] = num(hp["diff"], 1)
    m["pobreza-dif-ic"] = ci(hp["diff_ci_welch"][0], hp["diff_ci_welch"][1], 1)
    m["pobreza-d"] = num(hp["cohen_d"], 2)
    m["pobreza-f-p"] = pvalue(hp["f_test"]["p"])
    m["pobreza-welch-t"] = num(hp["welch"]["statistic"], 2)
    m["pobreza-ks"] = num(hp["ks"]["statistic"], 3)
    ct = ex["clinical_trials"]
    m["ensayos-dif"] = num(ct["diff"], 1)
    m["ensayos-mw-p"] = pvalue(ct["mann_whitney"]["p"])
    m["ensayos-welch-p"] = pvalue(ct["welch"]["p"])
    m["ensayos-n-con"] = integer(ct["n"][0])
    ig = ex["income_gradient"]
    m["renta-spearman"] = num(ig["spearman"], 3)
    m["renta-decil1"] = num(ig["table"][0]["mean"], 1)
    m["renta-decil10"] = num(ig["table"][-1]["mean"], 1)
    for row in ex["correlations"]:
        a = alias(row["variable"])
        m[f"r-{a}"] = num(row["pearson"], 3)
        m[f"rho-{a}"] = num(row["spearman"], 3)
        m[f"r-{a}-ic"] = ci(row["pearson_lower"], row["pearson_upper"], 3)
    n_sig = sum(1 for row in ex["correlations"] if row["p_pearson_adj"] < 0.05)
    m["n-corr-sig"] = integer(n_sig)
    m["n-corr"] = integer(len(ex["correlations"]))

    # Atípicos --------------------------------------------------------------------
    out = res["outliers"]
    mv = out["multivariate"]
    m["n-mcd"] = integer(mv["n_flagged_robust"])
    m["n-mahal-clasica"] = integer(mv["n_flagged_classical"])
    m["mcd-p"] = integer(mv["p"])
    m["mcd-corte"] = num(mv["cutoff"], 1)
    m["n-atipicos-respuesta"] = integer(len(out["response"]))
    resp_uni = _get(out["univariate"], variable="TARGET_deathRate")
    if resp_uni:
        m["n-tukey-respuesta"] = integer(resp_uni["n_tukey"])

    # Efectos ---------------------------------------------------------------------
    eff = res["effects"]
    coll = eff["collinearity"]
    m["n-candidatas"] = integer(len(eff["candidates"]))
    m["n-podadas"] = integer(len(coll["trace"]))
    m["podadas"] = join_es([tex_escape(label(t["eliminada"])).lower() for t in coll["trace"]])
    m["fiv-max-antes"] = num(max(x["fiv"] for x in coll["vif_before"]), 1)
    m["fiv-max-despues"] = num(max(x["fiv"] for x in coll["vif_after"]), 2)
    m["ncond-antes"] = num(coll["belsley_before"]["numero_condicion"], 1)
    m["ncond-despues"] = num(coll["belsley_after"]["numero_condicion"], 1)
    m["n-pares-corr"] = integer(len(coll["pairs"]))
    for t in coll["trace"]:
        m[f"fiv-poda-{alias(t['eliminada'])}"] = num(t["fiv"], 1)
    m["n-maximo"] = integer(eff["max_design"]["n"])
    m["k-maximo"] = integer(len(eff["max_design"]["terms"]))
    ini = eff["initial"]
    m["r2-inicial"] = num(ini["summary"]["r2"], 3)
    bd = ini["diagnostics"]
    m["bp-antes"] = num(bd["breusch_pagan"]["LM"], 1)
    m["bp-antes-p"] = pvalue(bd["breusch_pagan"]["p"])
    m["bp-pob-antes"] = num(bd["bp_population"]["LM"], 1)
    m["bp-pob-antes-p"] = pvalue(bd["bp_population"]["p"])
    m["icc-antes"] = num(bd["icc_state"]["icc"], 3)
    m["icc-antes-f"] = num(bd["icc_state"]["F"], 2)
    m["icc-antes-p"] = pvalue(bd["icc_state"]["p"])
    m["dw-antes"] = num(bd["durbin_watson"], 3)
    m["curtosis-antes"] = num(bd["shape"]["kurtosis_excess"], 2)
    dec_v = bd["var_by_population_decile"]
    m["var-decil1"] = num(dec_v[0], 0)
    m["var-decil10"] = num(dec_v[-1], 0)
    m["razon-var-deciles"] = num(dec_v[0] / dec_v[-1], 1)
    vf = eff.get("variance_function_final")
    if vf:
        m["vf-a"] = num(vf["a"], 1)
        m["vf-b"] = num(vf["b"] / 1e6, 3)
        m["vf-b-p"] = pvalue(vf["p_b"])
        m["vf-iter"] = integer(vf["iterations"])
        for ex_row in eff.get("variance_examples") or []:
            m[f"frac-muestreo-{ex_row['poblacion']}"] = num(100 * ex_row["fraccion_muestreo"], 0)
    sel = eff["selection"]
    m["seleccion-metodo"] = {
        "backward": "hacia atrás",
        "forward": "hacia delante",
        "stepwise": "por pasos",
        "none": "sin selección",
    }[sel["method"]]
    m["n-pasos-seleccion"] = integer(len(sel["steps"]))
    m["eliminadas"] = join_es(
        [tex_escape(label(s["termino"])).lower() for s in sel["steps"] if s["accion"] == "sale"]
    )
    m["n-seleccionadas"] = integer(len(sel["selected"]))
    m["n-ingenua"] = integer(len(eff["naive_selection"]["selected"]))
    extra = [t for t in eff["naive_selection"]["selected"] if t not in sel["selected"]]
    m["ingenua-extra"] = join_es([tex_escape(label(t)).lower() for t in extra])
    m["n-ingenua-extra"] = integer(len(extra))
    common = set(eff["strategies"]["backward"]["selected"])
    for s in eff["strategies"].values():
        common &= set(s["selected"])
    m["n-comunes-estrategias"] = integer(len(common))
    m["comunes-estrategias"] = join_es(sorted(tex_escape(label(t)).lower() for t in common))
    m["confusoras"] = (
        join_es([tex_escape(label(c["reincorporada"])).lower() for c in eff["confounding"]])
        or "ninguna"
    )
    m["n-confusoras"] = integer(len(eff["confounding"]))
    for c in eff["confounding"]:
        m[f"confusion-{alias(c['reincorporada'])}"] = num(100 * c["cambio_relativo"], 1)
        m[f"confusion-{alias(c['reincorporada'])}-en"] = tex_escape(label(c["en"])).lower()
    entered = [t for t in eff["interactions"] if t["entra"]]
    m["n-interacciones"] = integer(len(entered))
    m["interacciones"] = (
        join_es(
            [tex_escape(label(t["termino"])).replace("Región censal", "región") for t in entered]
        )
        or "ninguna"
    )
    m["n-interacciones-candidatas"] = integer(len(eff["max_design"]["interaction_candidates"]))
    for t in eff["interactions"]:
        if t["entra"]:
            a = alias(t["termino"])
            m[f"int-{a}-f"] = num(t["F"], 2)
            m[f"int-{a}-p"] = pvalue(t["p"])
            m[f"int-{a}-padj"] = pvalue(t["p_ajustado"])
    fin = eff["final"]
    s = fin["summary"]
    m["n-final"] = integer(fin["n"])
    m["n-estados-final"] = integer(fin["n_clusters"])
    m["k-final"] = integer(s["k"])
    m["n-terminos-final"] = integer(len(fin["terms"]))
    m["r2-final"] = num(s["r2"], 3)
    m["r2aj-final"] = num(s["r2_adj"], 3)
    m["pct-r2-final"] = num(100 * s["r2"], 1)
    m["r2-noponderado"] = num(s["r2_unweighted"], 3)
    m["sigma-final"] = num(s["sigma"], 2)
    m["rmse-final"] = num(s["rmse_unweighted"], 2)
    m["f-robusto"] = num(s["f_robust"], 1)
    m["f-robusto-p"] = pvalue(s["p_f_robust"])
    m["f-clasico"] = num(s["f_classic"], 1)
    m["gl-inferencia"] = integer(s["df_inference"])
    m["aic-final"] = num(s["aic"], 1)
    for c in fin["coef"]:
        a = alias(c["term"])
        m[f"b-{a}"] = num(c["coef"], 3)
        m[f"ee-{a}"] = num(c["se"], 3)
        m[f"t-{a}"] = num(c["t"], 2)
        m[f"p-{a}"] = pvalue(c["p"])
        m[f"ic-{a}"] = ci(c["lower"], c["upper"], 3)
        if c.get("beta") is not None:
            m[f"beta-{a}"] = num(c["beta"], 3)
    for e in eff["effects"]:
        a = alias(e["variable"])
        m[f"iqr-{a}"] = num(e["iqr"], 2)
        if e.get("interaccion_region"):
            for sl in e["by_region"]:
                ra = REGION_ALIAS[sl["region"]]
                m[f"pend-{a}-{ra}"] = num(sl["coef"], 3)
                m[f"pend-{a}-{ra}-ic"] = ci(sl["lower"], sl["upper"], 3)
                m[f"pend-{a}-{ra}-p"] = pvalue(sl["p"])
                m[f"efiqr-{a}-{ra}"] = num(sl["iqr_effect"], 1)
        else:
            m[f"efiqr-{a}"] = num(e["iqr_effect"], 1)
            m[f"efiqr-{a}-ic"] = ci(e["iqr_lower"], e["iqr_upper"], 1)
    for it in eff.get("numeric_interactions") or []:
        a = alias(it["a"])
        for sl, q in zip(it["slopes"], ("q1", "q2", "q3"), strict=True):
            m[f"pend-{a}-{alias(it['b'])}-{q}"] = num(sl["pendiente"], 3)
            m[f"valor-{alias(it['b'])}-{q}"] = num(sl["valor_b"], 1)
    adj = {x["region"]: x for x in eff["adjusted_region_means"]}
    for reg_name, x in adj.items():
        ra = REGION_ALIAS[reg_name]
        m[f"aj-{ra}"] = num(x["ajustada"], 1)
        m[f"aj-{ra}-ic"] = ci(x["lower"], x["upper"], 1)
        m[f"bruta-{ra}"] = num(x["bruta"], 1)
    if "Sur" in adj and "Oeste" in adj:
        raw_gap = adj["Sur"]["bruta"] - adj["Oeste"]["bruta"]
        adj_gap = adj["Sur"]["ajustada"] - adj["Oeste"]["ajustada"]
        m["brecha-bruta"] = num(raw_gap, 1)
        m["brecha-ajustada"] = num(adj_gap, 1)
        m["pct-brecha-explicada"] = num(100 * (1 - adj_gap / raw_gap), 0)
    inc = eff.get("incidence_adjustment")
    if inc:
        m["r2-sin-incidencia"] = num(inc["r2_without"], 3)
        m["r2-con-incidencia"] = num(inc["r2_with"], 3)
        for c in inc["comparison"]:
            if c["cambio_pct"] is not None:
                m[f"cambio-{alias(c['term'])}"] = num(c["cambio_pct"], 1)
                m[f"sin-inc-{alias(c['term'])}"] = num(c["sin"], 3)

    d = eff["diagnostics"]
    lin = d["linearity"]
    m["reset-f"] = num(lin["reset"]["F"], 3)
    m["reset-p"] = pvalue(lin["reset"]["p"])
    curv = [c for c in lin["curvature"] if c.get("p_ajustado", 1) < 0.05]
    m["n-curvatura"] = integer(len(curv))
    m["curvatura"] = join_es([tex_escape(label(c["variable"])).lower() for c in curv]) or "ninguna"
    norm = {t["test"]: t for t in d["normality"]["tests"]}
    m["res-sw"] = num(norm["Shapiro-Wilk"]["statistic"], 4) if "Shapiro-Wilk" in norm else "---"
    m["res-sw-p"] = pvalue(norm["Shapiro-Wilk"]["p"]) if "Shapiro-Wilk" in norm else "---"
    m["res-jb"] = num(norm["Jarque-Bera"]["statistic"], 1)
    m["res-jb-p"] = pvalue(norm["Jarque-Bera"]["p"])
    m["res-asimetria"] = num(d["normality"]["shape"]["skewness"], 3)
    m["res-curtosis"] = num(d["normality"]["shape"]["kurtosis_excess"], 2)
    h = d["homoscedasticity"]
    m["bp-despues"] = num(h["breusch_pagan"]["LM"], 1)
    m["bp-despues-p"] = pvalue(h["breusch_pagan"]["p"])
    m["bp-pob-despues"] = num(h["bp_population"]["LM"], 2)
    m["bp-pob-despues-p"] = pvalue(h["bp_population"]["p"])
    m["white-despues"] = num(h["white_reduced"]["LM"], 2)
    m["white-despues-p"] = pvalue(h["white_reduced"]["p"])
    m["gq-despues"] = num(h["goldfeld_quandt"]["F"], 2)
    ind = d["independence"]
    m["dw-final"] = num(ind["durbin_watson"], 3)
    m["rachas-res-z"] = num(ind["runs"]["z"], 2)
    m["rachas-res-p"] = pvalue(ind["runs"]["p"])
    m["icc-final"] = num(ind["icc_state"]["icc"], 3)
    m["icc-final-f"] = num(ind["icc_state"]["F"], 2)
    m["icc-final-p"] = pvalue(ind["icc_state"]["p"])
    m["media-residuos"] = r"\num{" + f"{d['mean_zero']['weighted_mean_residual']:.1e}" + "}"
    m["fiv-max-final"] = num(max((x["fiv"] for x in d["collinearity"]["vif"]), default=1.0), 2)
    m["ncond-final"] = num(d["collinearity"]["condition_number"], 1)
    inf_ = d["influence"]
    thr = inf_["thresholds"]
    m["n-cook"] = integer(inf_["n_cook"])
    m["n-cook-f50"] = integer(inf_["n_cook_f50"])
    m["umbral-cook"] = num(thr["cook"], 5)
    m["n-palanca"] = integer(inf_["n_leverage"])
    m["umbral-palanca"] = num(thr["leverage"], 4)
    m["n-estudentizados"] = integer(inf_["n_studentized"])
    m["umbral-dfbetas"] = num(thr["dfbetas"], 4)
    top = inf_["top"]
    if top:
        m["influyente1"] = tex_escape(top[0]["county_id"])
        m["influyente1-cook"] = num(top[0]["cook"], 3)
        m["influyente1-palanca"] = num(top[0]["leverage"], 3)
    big = max(top, key=lambda t: abs(t["stud_resid"])) if top else None
    if big:
        m["mayor-residuo"] = tex_escape(big["county_id"])
        m["mayor-residuo-valor"] = num(big["stud_resid"], 2)
        m["mayor-residuo-y"] = num(big["y"], 1)
        m["mayor-residuo-ajuste"] = num(big["fitted"], 1)

    specs = {sp["id"]: sp for sp in eff["sensitivity"]}
    m["n-especificaciones"] = integer(len(specs))
    main_terms = [c["term"] for c in fin["coef"] if c["term"] != "const"]
    for t in main_terms:
        vals = [sp["coef"][t] for sp in specs.values() if t in sp["coef"]]
        if vals:
            a = alias(t)
            m[f"rango-{a}"] = ci(min(vals), max(vals), 3)
    if "mixed" in specs:
        m["icc-mixto"] = num(specs["mixed"].get("icc", float("nan")), 3)
    if "huber" in specs:
        m["n-huber"] = integer(specs["huber"].get("downweighted", 0))
    if "no_influential" in specs:
        m["n-sin-influyentes"] = integer(specs["no_influential"]["n"])
    if "mi" in specs:
        m["n-imputacion"] = integer(specs["mi"]["n"])
    boot = eff.get("bootstrap")
    if boot:
        m["boot-reps"] = integer(boot["reps"])
        for t in main_terms:
            if t in boot["lower"]:
                m[f"boot-{alias(t)}"] = ci(boot["lower"][t], boot["upper"][t], 3)
    bc = eff.get("boxcox")
    if bc:
        m["boxcox-modelo"] = num(bc["lambda"], 3)
        m["boxcox-modelo-ic"] = ci(bc["lower"], bc["upper"], 3)

    # Predicción ------------------------------------------------------------------
    pred = res.get("predictive")
    if pred:
        m["n-entrenamiento"] = integer(pred["n_train"])
        m["n-prueba"] = integer(pred["n_test"])
        m["modelo-elegido"] = tex_escape(pred["chosen_label"]).lower()
        m["modelo-mejor-cv"] = tex_escape(pred["best_label"]).lower()
        m["modelo-1ee"] = tex_escape(
            next(r["label"] for r in pred["cv"] if r["model"] == pred["one_se_choice"])
        ).lower()
        for row in pred["cv"]:
            a = MODEL_ALIAS[row["model"]]
            m[f"cv-rmse-{a}"] = num(row["rmse_mean"], 2)
            m[f"cv-rmse-{a}-dt"] = num(row["rmse_sd"], 2)
            m[f"cv-r2-{a}"] = num(row["r2_mean"], 3)
        for row in pred["group_cv"]:
            a = MODEL_ALIAS[row["model"]]
            m[f"gcv-rmse-{a}"] = num(row["rmse_mean"], 2)
            m[f"gcv-r2-{a}"] = num(row["r2_mean"], 3)
        if pred["group_cv"]:
            gbest = min(pred["group_cv"], key=lambda r: r["rmse_mean"])
            m["modelo-mejor-gcv"] = tex_escape(gbest["label"]).lower()
        for row in pred["test"]:
            a = MODEL_ALIAS[row["model"]]
            m[f"test-rmse-{a}"] = num(row["rmse"], 2)
            m[f"test-r2-{a}"] = num(row["r2"], 3)
            m[f"test-mae-{a}"] = num(row["mae"], 2)
        chosen = next(r for r in pred["test"] if r["model"] == pred["chosen"])
        m["test-rmse-elegido"] = num(chosen["rmse"], 2)
        m["test-r2-elegido"] = num(chosen["r2"], 3)
        m["test-mae-elegido"] = num(chosen["mae"], 2)
        tci = pred["test_ci"]
        m["test-rmse-ic"] = ci(tci["rmse_lower"], tci["rmse_upper"], 2)
        m["test-dif-ic"] = ci(tci["diff_lower"], tci["diff_upper"], 2)
        m["test-dif-frente"] = tex_escape(MODEL_ALIAS.get(tci["diff_vs"], tci["diff_vs"]))
        c = pred["conformal"]
        m["conf-nominal"] = num(100 * c["nominal"], 0)
        m["conf-std"] = num(100 * c["coverage_standard"], 1)
        m["conf-norm"] = num(100 * c["coverage_normalized"], 1)
        m["conf-anchura-std"] = num(c["width_standard"], 1)
        for b in c["by_population"]:
            k = {"pequeños": "peq", "medianos": "med", "grandes": "gra"}[b["tercil"]]
            m[f"conf-std-{k}"] = num(100 * b["cobertura_estandar"], 1)
            m[f"conf-norm-{k}"] = num(100 * b["cobertura_normalizada"], 1)
            m[f"conf-anchura-norm-{k}"] = num(b["anchura_normalizada"], 1)
        if pred.get("lasso"):
            m["lasso-nocero"] = integer(pred["lasso"]["n_nonzero"])
            m["lasso-total"] = integer(pred["lasso"]["n_total"])
        imp = pred["importance"]
        if imp:
            m["importancia1"] = tex_escape(imp[0]["etiqueta"]).lower()
            m["importancia2"] = tex_escape(imp[1]["etiqueta"]).lower()
            m["importancia3"] = tex_escape(imp[2]["etiqueta"]).lower()

    # Corrida ---------------------------------------------------------------------
    m["corrida-id"] = r"\texttt{" + tex_escape(res["run_id"]) + "}"
    m["corrida-fecha"] = tex_escape(man["created_at"].replace("T", " ").replace("+00:00", " UTC"))
    m["corrida-huella"] = r"\texttt{" + man["config_fingerprint"] + "}"
    commit = (man.get("git") or {}).get("commit") or "sin control de versiones"
    m["corrida-commit"] = r"\texttt{" + tex_escape(commit[:12]) + "}"
    m["corrida-duracion"] = num(man["duration_s"], 0)
    m["corrida-semilla"] = integer(man["seed"])
    env = man["environment"]
    for k in ("python", "numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib"):
        m[f"version-{k.replace('-', '')}"] = tex_escape(env.get(k, "?"))
    return m


def write_macros(res: Results, path: Path) -> int:
    """Escribe ``resultados.tex`` y devuelve el número de macros."""
    values = macros(res)
    lines = [
        "% =====================================================================",
        "% Generado automáticamente por cancerstats a partir de la corrida",
        f"% {res['run_id']}. No editar: se sobrescribe en cada exportación.",
        "% =====================================================================",
    ]
    for key in sorted(values):
        lines.append(rf"\resdef{{{key}}}{{{values[key]}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(values)


# ============================================================================ tablas
def _weighted(res: Results) -> bool:
    return res["effects"].get("weighting", "fgls") == "fgls"


def _estimator(res: Results) -> str:
    return "MCPF" if _weighted(res) else "MCO"


def _metric(res: Results) -> str:
    return " (métrica ponderada)" if _weighted(res) else ""


def _cov_text(res: Results) -> str:
    return {
        "cluster": "errores típicos robustos por conglomerados de estado",
        "HC3": "errores típicos robustos HC3",
        "nonrobust": "errores típicos clásicos",
    }[res["effects"].get("covariance", "cluster")]


def _ragged(colspec: str) -> str:
    """Columnas de texto (X y p{…}) en bandera: evita cajas mal llenas en celdas estrechas."""
    rag = r">{\raggedright\arraybackslash}"
    out = re.sub(r"(?<![>{])\bX\b", lambda _m: rag + "X", colspec)
    return re.sub(r"(?<![}])p\{", lambda _m: rag + "p{", out)


def _table(
    name: str,
    caption: str,
    colspec: str,
    header: str,
    rows: list[str],
    note: str = "",
    size: str = r"\small",
    placement: str = "htbp",
    env: str = "tabular",
) -> str:
    body = "\n".join(rows)
    colspec = _ragged(colspec)
    note_tex = rf"\par\smallskip{{\footnotesize\raggedright {note}\par}}" if note else ""
    width = r"{\linewidth}" if env == "tabularx" else ""
    return (
        f"\\begin{{table}}[{placement}]\n\\centering\n{size}\n"
        f"\\caption{{{caption}}}\\label{{tab:{name}}}\n"
        f"\\begin{{{env}}}{width}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{{env}}}\n{note_tex}\n\\end{{table}}\n"
    )


def _long(
    name: str,
    caption: str,
    colspec: str,
    header: str,
    rows: list[str],
    note: str = "",
    size: str = r"\small",
) -> str:
    """Tabla larga que puede partirse entre páginas (xltabular)."""
    body = "\n".join(rows)
    colspec = _ragged(colspec)
    ncols = header.split(r"\\")[-1].count("&") + 1
    cont = rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{(continuación)}}}} \\"
    follows = rf"\multicolumn{{{ncols}}}{{r}}{{\emph{{(continúa)}}}} \\"
    note_row = (
        rf"\multicolumn{{{ncols}}}{{p{{0.97\linewidth}}}}{{\footnotesize {note}}} \\"
        if note
        else ""
    )
    return (
        f"{{{size}\n\\begin{{xltabular}}{{\\linewidth}}{{{colspec}}}\n"
        f"\\caption{{{caption}}}\\label{{tab:{name}}} \\\\\n\\toprule\n{header} \\\\\n"
        f"\\midrule\n\\endfirsthead\n{cont}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"\\endhead\n\\midrule\n{follows}\n\\endfoot\n\\bottomrule\n{note_row}\n"
        f"\\endlastfoot\n{body}\n\\end{{xltabular}}\n}}\n"
    )


def t_variables(res: Results) -> str:
    excluded = res["excluded"]
    cands = set(res["effects"]["candidates"])
    rows = []
    kinds = {
        "continua": "cont.",
        "porcentaje": "\\%",
        "conteo": "conteo",
        "categórica": "cat.",
        "ordinal": "ord.",
        "identificador": "id.",
    }
    for v in res["dictionary"]:
        name = v["name"]
        if name in ("logPop", "logIncome", "logStudy", "anyStudy"):
            continue
        model_name = {
            "popEst2015": "logPop",
            "medIncome": "logIncome",
            "studyPerCap": "logStudy",
        }.get(name, name)
        if name == "TARGET_deathRate":
            role = "respuesta"
        elif model_name in cands:
            role = "candidata"
        elif name in excluded:
            role = "excluida"
        elif name in ("region",):
            role = "factor"
        else:
            role = "auxiliar"
        rows.append(
            rf"\texttt{{{tex_escape(name)}}} & {tex_escape(v['label'])} & "
            rf"{tex_escape(v['unit'])} & {kinds[v['kind']]} & ({tex_escape(v['source'])}) & "
            rf"{role} \\"
        )
    return _long(
        "variables",
        "Diccionario de variables y papel de cada una en el análisis",
        r"l X l l c l",
        r"Variable & Descripción & Unidad & Tipo & Fuente & Papel",
        rows,
        note="Fuente (a): registros de cáncer 2010--2016; (b): estimaciones "
        "censales de 2013; derivada: calculada en la depuración. Las variables "
        "excluidas y su motivo se detallan en la tabla~\\ref{tab:excluidas}.",
        size=r"\footnotesize",
    )


def t_excluded(res: Results) -> str:
    rows = [
        rf"\texttt{{{tex_escape(k)}}} & {tex_escape(v[0].upper() + v[1:])} \\"
        for k, v in res["excluded"].items()
        if k != "TARGET_deathRate"
    ]
    return _table(
        "excluidas",
        "Variables que no entran como explicativas y motivo",
        r"l X",
        r"Variable & Motivo",
        rows,
        size=r"\footnotesize",
        env="tabularx",
    )


def t_validation(res: Results) -> str:
    raw = res["validation"]["raw"]
    clean = {r["id"]: r for r in res["validation"]["clean"]}
    rows = []
    sev = {"error": "error", "advertencia": "advert.", "información": "info."}
    for r in raw:
        c = clean[r["id"]]
        rows.append(
            rf"{r['id']} & {tex_escape(r['title'])} & {sev[r['severity']]} & "
            rf"{integer(r['n_violations'])} & {integer(c['n_violations'])} \\"
        )
    return _table(
        "validacion",
        "Catálogo de reglas de validación: casos que las incumplen "
        "antes y después de la depuración",
        r"l X l r r",
        r"Id. & Regla & Severidad & Original & Depurado",
        rows,
        note="R17 y R18 son informativas: describen ausencias declaradas e "
        "inflación de ceros, que se tratan en etapas posteriores.",
        size=r"\small",
        env="tabularx",
    )


def t_decisions(res: Results) -> str:
    rows = []
    for d in res["cleaning"]["decisions"]:
        rows.append(
            rf"{d['id']} & {tex_escape(d['title'])} & {integer(d['n_affected'])} & "
            rf"{tex_escape(d['action'])} \\"
        )
    return _long(
        "decisiones",
        "Registro de decisiones de depuración",
        r"l p{0.28\linewidth} r X",
        r"Id. & Decisión & Casos & Acción",
        rows,
        size=r"\footnotesize",
    )


def t_missing(res: Results) -> str:
    miss = res["missing"]
    rows = []
    for r in miss["by_column"]:
        mech = miss["mechanism"].get(r["variable"], "")
        short = mech.split(":")[0]
        state = _get(miss["by_state"], variable=r["variable"], agrupacion="state")
        rows.append(
            rf"{tex_escape(label(r['variable']))} & {integer(r['n'])} & {num(r['pct'], 1)} & "
            rf"{pcell(state['p']) if state else '---'} & {tex_escape(short)} \\"
        )
    lt_r = miss.get("little_random")
    lt_a = miss.get("little_all")
    note = ""
    if lt_r and lt_a:
        note = (
            rf"Contraste MCAR de Little sin la incidencia: $d^2 = {num(lt_r['statistic'], 1)}$, "
            rf"{integer(lt_r['df'])} g.\,l., {pvalue(lt_r['p_value'])}. Con la incidencia: "
            rf"$d^2 = {num(lt_a['statistic'], 1)}$, {integer(lt_a['df'])} g.\,l., "
            rf"{pvalue(lt_a['p_value'])}."
        )
    return _table(
        "ausentes",
        "Datos ausentes tras la depuración y diagnóstico de su mecanismo",
        r"X r r r l",
        r"Variable & $n$ & \% & $p$ ($\chi^2$ estado) & Mecanismo",
        rows,
        note=note,
        env="tabularx",
    )


def t_response(res: Results) -> str:
    r = res["exploration"]["response"]
    rows = [
        rf"Media & {num(r['mean_ci']['estimate'], 2)} & "
        rf"{ci(r['mean_ci']['lower'], r['mean_ci']['upper'])} & pivote $t$ de Student \\",
        rf"Mediana & {num(r['median_ci_order']['estimate'], 1)} & "
        rf"{ci(r['median_ci_order']['lower'], r['median_ci_order']['upper'], 1)} & "
        rf"estadísticos de orden (libre distribución) \\",
        rf"Mediana & {num(r['median_ci_boot']['estimate'], 1)} & "
        rf"{ci(r['median_ci_boot']['lower'], r['median_ci_boot']['upper'], 1)} & "
        rf"bootstrap percentil ({integer(r['median_ci_boot']['reps'])} réplicas) \\",
        rf"Media recortada al 10\,\% & {num(r['trimmed_ci_boot']['estimate'], 2)} & "
        rf"{ci(r['trimmed_ci_boot']['lower'], r['trimmed_ci_boot']['upper'])} & bootstrap percentil \\",
        rf"M-estimador de Huber & {num(r['robust']['huber'], 2)} & --- & estimador robusto \\",
        rf"Varianza & {num(r['variance_ci']['estimate'], 1)} & "
        rf"{ci(r['variance_ci']['lower'], r['variance_ci']['upper'], 1)} & pivote $\chi^2_{{n-1}}$ \\",
        rf"Desviación típica & {num(r['variance_ci']['sd'], 2)} & "
        rf"{ci(r['variance_ci']['sd_lower'], r['variance_ci']['sd_upper'])} & pivote $\chi^2_{{n-1}}$ \\",
        rf"MEDA & {num(r['robust']['meda'], 1)} & --- & mediana de las desviaciones absolutas \\",
    ]
    return _table(
        "respuesta",
        "Estimación puntual y por intervalos (95\\,\\%) de la mortalidad por cáncer",
        r"l r l X",
        r"Parámetro & Estimación & IC 95\,\% & Método",
        rows,
        env="tabularx",
    )


def t_normality(res: Results) -> str:
    rows = []
    for t in res["exploration"]["response"]["normality"]:
        stat = num(t["statistic"], 4)
        p = pcell(t.get("p")) if t.get("p") is not None else "---"
        extra = rf" ($k = {t['k']}$, {t['df']} g.\,l.)" if "k" in t else ""
        rows.append(rf"{tex_escape(t['test'])}{extra} & {stat} & {p} \\")
    shape = res["exploration"]["response"]["shape"]
    rows.append(
        rf"Asimetría (contraste de D'Agostino) & {num(shape['skewness'], 3)} & "
        rf"{pcell(shape['p_skew'])} \\"
    )
    rows.append(
        rf"Curtosis de exceso (contraste de Anscombe-Glynn) & "
        rf"{num(shape['kurtosis_excess'], 3)} & {pcell(shape['p_kurtosis'])} \\"
    )
    return _table(
        "normalidad",
        "Contrastes de normalidad de la mortalidad",
        r"X r r",
        r"Contraste & Estadístico & $p$",
        rows,
        env="tabularx",
        note="Anderson-Darling informa un $p$ interpolado en tablas (acotado entre "
        "0,01 y 0,15). Con $n$ grande, cualquier desviación mínima resulta "
        "significativa: los contrastes se leen junto al gráfico Q-Q.",
    )


def t_regions(res: Results) -> str:
    reg = res["exploration"]["region"]
    rows = []
    for g in sorted(reg["groups"], key=lambda g: -g["mean"]):
        rows.append(
            rf"{g['group']} & {integer(g['n'])} & {num(g['mean'], 1)} & "
            rf"{ci(g['lower'], g['upper'], 1)} & {num(g['sd'], 1)} & {num(g['median'], 1)} \\"
        )
    note = (
        rf"ANOVA: $F = {num(reg['anova']['statistic'], 1)}$ ({reg['anova']['df1']} y "
        rf"{integer(reg['anova']['df2'])} g.\,l.), {pvalue(reg['anova']['p'])}, "
        rf"$\eta^2 = {num(reg['anova']['eta2'], 3)}$. Levene (mediana): "
        rf"{pvalue(reg['levene']['p'])}. Welch: $F = {num(reg['welch']['statistic'], 1)}$, "
        rf"{pvalue(reg['welch']['p'])}. Kruskal-Wallis: $H = {num(reg['kruskal']['statistic'], 1)}$, "
        rf"{pvalue(reg['kruskal']['p'])}."
    )
    return _table(
        "regiones",
        "Mortalidad por región censal",
        r"l r r l r r",
        r"Región & $n$ & Media & IC 95\,\% & Desv. típ. & Mediana",
        rows,
        note=note,
    )


def t_pairs(res: Results) -> str:
    rows = []
    for p in res["exploration"]["region"]["pairs"]:
        rows.append(
            rf"{p['a']} -- {p['b']} & {num(p['diff'], 1)} & "
            rf"{ci(p.get('tukey_lower'), p.get('tukey_upper'), 1)} & {pcell(p.get('p_tukey'))} & "
            rf"{pcell(p['p_mann_whitney_adj'])} \\"
        )
    return _table(
        "pares",
        "Comparaciones por pares entre regiones",
        r"l r l r r",
        r"Par & Diferencia & IC 95\,\% (Tukey) & $p$ Tukey & $p$ Mann-Whitney (Holm)",
        rows,
    )


def t_correlations(res: Results) -> str:
    rows = []
    for r in res["exploration"]["correlations"]:
        rows.append(
            rf"{tex_escape(label(r['variable']))} & {num(r['pearson'], 3)} & "
            rf"{ci(r['pearson_lower'], r['pearson_upper'], 3)} & {num(r['spearman'], 3)} & "
            rf"{pcell(r['p_pearson_adj'])} \\"
        )
    return _long(
        "correlaciones",
        "Correlación de cada candidata con la mortalidad",
        r"X r l r r",
        r"Variable & $r$ & IC 95\,\% (Fisher) & $\rho$ & $p$ (Holm)",
        rows,
        note="$r$: Pearson; $\rho$: Spearman. Los $p$ de Pearson se corrigen por "
        "comparaciones múltiples con el método de Holm.",
        size=r"\footnotesize",
    )


def t_collinearity(res: Results) -> str:
    coll = res["effects"]["collinearity"]
    rows = []
    for t in coll["trace"]:
        rows.append(
            rf"{t['paso']} & {tex_escape(label(t['eliminada']))} & {num(t['fiv'], 1)} & "
            rf"{tex_escape(label(t['mas_correlada_con']))} & {num(t['r'], 3)} \\"
        )
    return _table(
        "poda",
        "Poda iterativa por factor de inflación de la varianza (umbral 10)",
        r"r X r X r",
        r"Paso & Eliminada & FIV & Más correlada con & $r$",
        rows,
        env="tabularx",
        note=rf"Número de condición: {num(coll['belsley_before']['numero_condicion'], 1)} "
        rf"antes y {num(coll['belsley_after']['numero_condicion'], 1)} después de la poda.",
    )


def t_selection(res: Results) -> str:
    sel = res["effects"]["selection"]
    rows = []
    for s in sel["steps"]:
        rows.append(
            rf"{s['paso']} & {'sale' if s['accion'] == 'sale' else 'entra'} & "
            rf"{tex_escape(label(s['termino']))} & {num(s['F'], 3)} & {pcell(s['p'])} \\"
        )
    if not rows:
        rows.append(r"\multicolumn{5}{l}{Ningún término cumple el criterio de salida.} \\")
    return _table(
        "seleccion",
        "Traza de la selección de variables (contrastes de Wald con "
        "covarianza robusta por conglomerados)",
        r"r l X r r",
        r"Paso & Acción & Término & $F$ & $p$",
        rows,
        env="tabularx",
    )


def t_interactions(res: Results) -> str:
    rows = []
    for t in res["effects"]["interactions"]:
        mark = r"\textbf{entra}" if t["entra"] else ""
        rows.append(
            rf"{t['ronda']} & {tex_escape(label(t['termino']))} & {t['gl']} & "
            rf"{num(t['F'], 2)} & {pcell(t['p'])} & {pcell(t['p_ajustado'])} & {mark} \\"
        )
    return _table(
        "interacciones",
        "Contraste de interacciones candidatas por rondas (corrección "
        "de Holm sobre las candidatas de cada ronda)",
        r"r X r r r r l",
        r"Ronda & Interacción & g.\,l. & $F$ & $p$ & $p$ Holm & ",
        rows,
        size=r"\footnotesize",
        env="tabularx",
    )


def t_comparison(res: Results) -> str:
    rows = []
    for c in res["effects"]["model_comparison"]:
        fpart = (
            rf"{num(c['F_parcial'], 2)} & {pcell(c['p_parcial'])}"
            if "F_parcial" in c
            else r"--- & ---"
        )
        rows.append(
            rf"{tex_escape(c['modelo'])} & {c['k']} & {num(c['r2_adj'], 3)} & "
            rf"{num(c['aic'], 1)} & {num(c['bic'], 1)} & {fpart} \\"
        )
    return _table(
        "comparacion",
        "Comparación de modelos anidados (misma muestra y mismos pesos)",
        r"X r r r r r r",
        r"Modelo & $k$ & $\bar R^2$ & AIC & BIC & $F$ parcial & $p$",
        rows,
        env="tabularx",
        size=r"\footnotesize",
        note="El $F$ parcial compara cada modelo con el de la fila anterior.",
    )


def t_anova(res: Results) -> str:
    s = res["effects"]["final"]["summary"]
    k = int(s["k"])
    n = int(s["n"])
    dfe = n - k - 1
    rows = [
        rf"Regresión (SCR) & {num(s['scr'], 1)} & {k} & {num(s['scr'] / k, 1)} & "
        rf"{num(s['f_classic'], 2)} & {pcell(s['p_f_classic'])} \\",
        rf"Residual (SCE) & {num(s['sce'], 1)} & {integer(dfe)} & {num(s['sce'] / dfe, 2)} & & \\",
        rf"Total (SCT) & {num(s['sct'], 1)} & {integer(n - 1)} & & & \\",
    ]
    return _table(
        "anova",
        "Tabla ANOVA del modelo final" + _metric(res),
        r"l r r r r r",
        r"Fuente & Suma de cuadrados & g.\,l. & Cuadrado medio & $F$ & $p$",
        rows,
        note=rf"$R^2 = {num(s['r2'], 3)}$, $\bar R^2 = {num(s['r2_adj'], 3)}$. "
        rf"Contraste global robusto (Wald, cluster): $F = {num(s['f_robust'], 1)}$, "
        rf"{pvalue(s['p_f_robust'])}.",
    )


def t_coefficients(res: Results) -> str:
    rows = []
    for c in res["effects"]["final"]["coef"]:
        name = "Constante" if c["term"] == "const" else label(c["term"])
        beta = num(c["beta"], 3) if c.get("beta") is not None else ""
        rows.append(
            rf"{tex_escape(name)} & {num(c['coef'], 3)} & {num(c['se'], 3)} & "
            rf"{num(c['t'], 2)} & {pcell(c['p'])} & {ci(c['lower'], c['upper'], 3)} & {beta} \\"
        )
    eff = res["effects"]["final"]
    return _long(
        "coeficientes",
        "Coeficientes del modelo final de efectos ("
        + _estimator(res)
        + " con "
        + _cov_text(res)
        + ")",
        r"X r r r r l r",
        r"Término & $\hat\beta$ & E.\,T. & $t$ & $p$ & IC 95\,\% & Beta",
        rows,
        note=rf"$n = {integer(eff['n'])}$ condados de {integer(eff['n_clusters'])} estados. "
        rf"Explicativas continuas centradas en su media: la constante es la mortalidad "
        rf"esperada de un condado medio del Sur. Inferencia con $t$ de "
        rf"{integer(eff['summary']['df_inference'])} g.\,l. Beta: coeficiente tipificado.",
        size=r"\footnotesize",
    )


def t_spss_summary(res: Results) -> str:
    sp = res["effects"]["final"]["spss"]
    ms = sp["model_summary"]
    an = sp["anova"]
    dfe = int(an["n"] - an["k"] - 1)
    rows = [
        rf"{num(ms['R'], 3)} & {num(ms['R2'], 3)} & {num(ms['R2_adj'], 3)} & "
        rf"{num(ms['se_estimate'], 2)} & {num(ms['rmse'], 2)} & {num(ms['durbin_watson'], 3)} & "
        rf"{num(an['F'], 2)} & {integer(an['k'])} & {integer(dfe)} & {pcell(an['p'])} \\",
    ]
    return _table(
        "spss-resumen",
        "Resumen del modelo final (salida tipo SPSS)",
        r"r r r r r r r r r r",
        r"$R$ & $R^2$ & $\bar R^2$ & E.\,T. estimación & RMSE & D-W & $F$ & g.\,l.$_1$ & "
        r"g.\,l.$_2$ & $p$",
        rows,
        size=r"\footnotesize",
        note=r"E.\,T. de la estimación: $\sqrt{\mathrm{SCE}/(n-k-1)}$; RMSE: "
        r"$\sqrt{\mathrm{SCE}/n}$, ambos dentro de la muestra. D-W: Durbin-Watson. "
        r"El $F$ y su $p$ son los clásicos que imprime SPSS.",
    )


def t_spss_coefficients(res: Results) -> str:
    rows = []
    for c in res["effects"]["final"]["spss"]["coefficients"]:
        name = "Constante" if c["term"] == "const" else label(c["term"])
        beta = num(c["beta"], 3) if c.get("beta") is not None else ""
        tol = num(c["tolerancia"], 3) if c.get("tolerancia") is not None else ""
        fiv = num(c["fiv"], 2) if c.get("fiv") is not None else ""
        rows.append(
            rf"{tex_escape(name)} & {num(c['coef'], 3)} & {num(c['se'], 3)} & {beta} & "
            rf"{num(c['t'], 2)} & {pcell(c['p'])} & {ci(c['lower'], c['upper'], 3)} & "
            rf"{tol} & {fiv} \\"
        )
    return _long(
        "spss-coeficientes",
        "Coeficientes del modelo final con errores típicos clásicos (salida tipo SPSS)",
        r"X r r r r r l r r",
        r"Término & $B$ & E.\,T. & Beta & $t$ & Sig. & IC 95\,\% & Tol. & FIV",
        rows,
        note=r"Errores típicos clásicos $\hat\sigma^2(X^\top X)^{-1}$, como en el "
        r"procedimiento REGRESSION; la sintaxis \texttt{spss/modelo\_final.sps} los "
        r"reproduce. La inferencia del informe usa los robustos (tabla~\ref{tab:errores}).",
        size=r"\footnotesize",
    )


def t_standard_errors(res: Results) -> str:
    fin = res["effects"]["final"]
    classic = {c["term"]: c for c in fin["coef_classic"]}
    hc3 = {c["term"]: c for c in fin["coef_hc3"]}
    rows = []
    for c in fin["coef"]:
        if c["term"] == "const":
            continue
        t = c["term"]
        cl, h = classic[t], hc3[t]
        rows.append(
            rf"{tex_escape(label(t))} & {num(c['coef'], 3)} & {num(cl['se'], 3)} & "
            rf"{num(h['se'], 3)} & {num(c['se'], 3)} & {num(c['se'] / cl['se'], 2)} & "
            rf"{pcell(cl['p'])} & {pcell(c['p'])} \\"
        )
    return _long(
        "errores",
        "Errores típicos del modelo final: clásicos, HC3 y por conglomerados de estado",
        r"X r r r r r r r",
        r"Término & $\hat\beta$ & Clásico & HC3 & Cluster & Razón & $p$ clásico & "
        r"$p$ cluster",
        rows,
        note=r"Razón: E.\,T. por conglomerados entre E.\,T. clásico. Una razón mayor que "
        r"1 indica que la inferencia clásica es optimista, por la dependencia dentro de "
        r"cada estado y la heterocedasticidad.",
        size=r"\footnotesize",
    )


def t_slopes(res: Results) -> str:
    rows = []
    for e in res["effects"]["effects"]:
        if not e.get("interaccion_region"):
            continue
        first = True
        for s in e["by_region"]:
            var = tex_escape(label(e["variable"])) if first else ""
            first = False
            rows.append(
                rf"{var} & {s['region']} & {num(s['coef'], 3)} & {ci(s['lower'], s['upper'], 3)} & "
                rf"{pcell(s['p'])} & {num(s['iqr_effect'], 1)} \\"
            )
        rows.append(r"\addlinespace")
    if rows:
        rows.pop()
    return _table(
        "pendientes",
        "Efecto de las variables que interaccionan con la región: "
        "pendiente específica de cada región",
        r"l l r l r r",
        r"Variable & Región & Pendiente & IC 95\,\% & $p$ & Efecto Q1$\to$Q3",
        rows,
        note=r"Pendiente en la región $R$: $\hat\beta_{x} + \hat\beta_{x\times R}$, con "
        r"$\widehat{\mathrm{Var}} = \widehat{\mathrm{Var}}(\hat\beta_x) + "
        r"\widehat{\mathrm{Var}}(\hat\beta_{x\times R}) + 2\,\widehat{\mathrm{Cov}}$.",
        size=r"\footnotesize",
    )


def t_effects(res: Results) -> str:
    rows = []
    for e in res["effects"]["effects"]:
        if e.get("interaccion_region"):
            continue
        u = unit(e["variable"])
        rows.append(
            rf"{tex_escape(label(e['variable']))} & {tex_escape(u)} & "
            rf"{num(e['per_unit'], 3)} & {num(e['iqr'], 2)} & {num(e['iqr_effect'], 1)} & "
            rf"{ci(e['iqr_lower'], e['iqr_upper'], 1)} \\"
        )
    return _table(
        "efectos",
        "Magnitud de los efectos: por unidad y al pasar del primer al tercer cuartil",
        r"X l r r r l",
        r"Variable & Unidad & Por unidad & IQR & Efecto Q1$\to$Q3 & IC 95\,\%",
        rows,
        env="tabularx",
        size=r"\footnotesize",
        note="Muertes por 100\\,000 habitantes, manteniendo constantes las demás "
        "variables del modelo.",
    )


def _verdict(p: float, alpha: float = 0.05) -> str:
    return "se rechaza $H_0$" if p < alpha else "no se rechaza $H_0$"


def t_diagnostics(res: Results) -> str:
    d = res["effects"]["diagnostics"]
    ini = res["effects"]["initial"]["diagnostics"]
    lin = d["linearity"]
    norm = {t["test"]: t for t in d["normality"]["tests"]}
    h = d["homoscedasticity"]
    ind = d["independence"]
    rows = [
        rf"Linealidad & RESET de Ramsey (potencias 2 y 3) & {num(lin['reset']['F'], 3)} & "
        rf"{pcell(lin['reset']['p'])} & {_verdict(lin['reset']['p'])} \\",
        rf"Normalidad & Shapiro-Wilk (residuos tipificados) & "
        rf"{num(norm['Shapiro-Wilk']['statistic'], 4) if 'Shapiro-Wilk' in norm else '---'} & "
        rf"{pcell(norm['Shapiro-Wilk']['p']) if 'Shapiro-Wilk' in norm else '---'} & "
        rf"{_verdict(norm['Shapiro-Wilk']['p']) if 'Shapiro-Wilk' in norm else ''} \\",
        rf" & Jarque-Bera & {num(norm['Jarque-Bera']['statistic'], 1)} & "
        rf"{pcell(norm['Jarque-Bera']['p'])} & {_verdict(norm['Jarque-Bera']['p'])} \\",
        rf"Homocedasticidad & Breusch-Pagan, MCO inicial & {num(ini['breusch_pagan']['LM'], 1)} & "
        rf"{pcell(ini['breusch_pagan']['p'])} & {_verdict(ini['breusch_pagan']['p'])} \\",
        rf" & Breusch-Pagan, modelo final & {num(h['breusch_pagan']['LM'], 1)} & "
        rf"{pcell(h['breusch_pagan']['p'])} & {_verdict(h['breusch_pagan']['p'])} \\",
        rf" & Frente a 1/población, MCO inicial & {num(ini['bp_population']['LM'], 1)} & "
        rf"{pcell(ini['bp_population']['p'])} & {_verdict(ini['bp_population']['p'])} \\",
        rf" & Frente a 1/población, modelo final & {num(h['bp_population']['LM'], 2)} & "
        rf"{pcell(h['bp_population']['p'])} & {_verdict(h['bp_population']['p'])} \\",
        rf"Independencia & Durbin-Watson (orden del fichero) & {num(ind['durbin_watson'], 3)} & --- & \\",
        rf" & Rachas de los signos & {num(ind['runs']['z'], 2)} & {pcell(ind['runs']['p'])} & "
        rf"{_verdict(ind['runs']['p'])} \\",
        rf" & Correlación intraclase por estado & {num(ind['icc_state']['icc'], 3)} & "
        rf"{pcell(ind['icc_state']['p'])} & {_verdict(ind['icc_state']['p'])} \\",
        rf"Media cero & Media de los residuos & "
        rf"\num{{{d['mean_zero']['weighted_mean_residual']:.1e}}} & --- & por construcción \\",
    ]
    return _table(
        "diagnostico",
        "Diagnóstico de las hipótesis del modelo",
        r"l X r r l",
        r"Hipótesis & Contraste & Estadístico & $p$ & Decisión al 5\,\%",
        rows,
        env="tabularx",
        size=r"\footnotesize",
        note="La dependencia dentro de cada estado y la heterocedasticidad "
        "se absorben con errores típicos robustos por conglomerados; la falta de "
        "normalidad, con el bootstrap por conglomerados, que no la supone.",
    )


def t_influence(res: Results) -> str:
    top = res["effects"]["diagnostics"]["influence"]["top"][:10]
    rows = []
    for t in top:
        rows.append(
            rf"{tex_escape(t['county_id'])} & {num(t['y'], 1)} & {num(t['fitted'], 1)} & "
            rf"{num(t['stud_resid'], 2)} & {num(t['leverage'], 3)} & {num(t['cook'], 4)} \\"
        )
    return _table(
        "influyentes",
        "Los diez condados más influyentes (distancia de Cook)",
        r"X r r r r r",
        r"Condado & $y$ & $\hat y$ & $r^*$ & $h_{ii}$ & Cook",
        rows,
        env="tabularx",
        size=r"\footnotesize",
        note=r"$r^*$: residuo estudentizado externamente; $h_{ii}$: apalancamiento.",
    )


def t_sensitivity(res: Results) -> str:
    specs = res["effects"]["sensitivity"]
    coef = [
        c
        for c in res["effects"]["final"]["coef"]
        if c["term"] != "const" and ":" not in c["term"] and not c["term"].startswith("region")
    ]
    keys = [c["term"] for c in sorted(coef, key=lambda c: -abs(c["t"]))][:6]
    header = "Especificación & $n$ & " + " & ".join(
        rf"\rotatebox{{70}}{{{tex_escape(label(k))}}}" for k in keys
    )
    rows = []
    for s in specs:
        cells = []
        for k in keys:
            if k in s["coef"]:
                p = s["p"].get(k)
                mark = "" if p is None or p < 0.05 else r"$^{\dagger}$"
                cells.append(num(s["coef"][k], 3) + mark)
            else:
                cells.append("---")
        rows.append(
            rf"{tex_escape(s['label'])} & {integer(s['n'])} & " + " & ".join(cells) + r" \\"
        )
    boot = res["effects"].get("bootstrap")
    if boot:
        cells = [num(boot["coef"].get(k), 3) for k in keys]
        rows.append(
            rf"Bootstrap por conglomerados ({integer(boot['reps'])}) & --- & "
            + " & ".join(cells)
            + r" \\"
        )
    return _table(
        "sensibilidad",
        "Análisis de sensibilidad: coeficientes de las seis "
        "explicativas con mayor $|t|$ bajo especificaciones alternativas",
        "X r " + "r " * len(keys),
        header,
        rows,
        env="tabularx",
        size=r"\footnotesize",
        note=r"$^{\dagger}$: no significativo al 5\,\% en esa especificación.",
    )


def t_incidence(res: Results) -> str:
    inc = res["effects"].get("incidence_adjustment")
    if not inc:
        return "% El modelo final no incluye la incidencia.\n"
    rows = []
    for c in inc["comparison"]:
        name = "Región: " + c["term"][7:-1] if c["term"].startswith("region") else label(c["term"])
        rows.append(
            rf"{tex_escape(name)} & {num(c['con'], 3)} & {num(c['sin'], 3)} & "
            rf"{num(c['cambio_pct'], 1)} \\"
        )
    return _table(
        "incidencia",
        "Coeficientes con y sin ajustar por la incidencia (misma muestra y mismos pesos)",
        r"X r r r",
        r"Término & Con incidencia & Sin incidencia & Cambio (\%)",
        rows,
        env="tabularx",
        size=r"\footnotesize",
        note=rf"$R^2$ con incidencia: {num(inc['r2_with'], 3)}; sin ella: "
        rf"{num(inc['r2_without'], 3)}. Un cambio grande indica que la incidencia es "
        "confusora (o mediadora) de esa asociación.",
    )


def t_cv(res: Results) -> str:
    pred = res["predictive"]
    group = {r["model"]: r for r in pred["group_cv"]}
    test = {r["model"]: r for r in pred["test"]}
    rows = []
    for r in pred["cv"]:
        g = group.get(r["model"])
        t = test[r["model"]]
        mark = r"$^\star$" if r["model"] == pred["chosen"] else ""
        rows.append(
            rf"{tex_escape(r['label'])}{mark} & {num(r['rmse_mean'], 2)} & {num(r['rmse_sd'], 2)} & "
            rf"{num(r['r2_mean'], 3)} & {num(g['rmse_mean'], 2) if g else '---'} & "
            rf"{num(t['rmse'], 2)} & {num(t['r2'], 3)} & {num(t['mae'], 2)} \\"
        )
    return _table(
        "prediccion",
        "Comparación de modelos predictivos: validación cruzada en "
        "entrenamiento y evaluación en prueba",
        r"X r r r r r r r",
        r"Modelo & \multicolumn{3}{c}{VC aleatoria (10)} & VC estado & "
        r"\multicolumn{3}{c}{Prueba} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-5}"
        r"\cmidrule(lr){6-8} & RMSE & D.\,T. & $R^2$ & RMSE & RMSE & $R^2$ & MAE",
        rows,
        env="tabularx",
        size=r"\footnotesize",
        note=r"$^\star$: modelo elegido por la regla configurada. La prueba se usa una "
        "sola vez, tras elegir.",
    )


def t_conformal(res: Results) -> str:
    c = res["predictive"]["conformal"]
    rows = []
    for b in c["by_population"]:
        rows.append(
            rf"{b['tercil'].capitalize()} & {num(100 * b['cobertura_estandar'], 1)} & "
            rf"{num(b['anchura_estandar'], 1)} & {num(100 * b['cobertura_normalizada'], 1)} & "
            rf"{num(b['anchura_normalizada'], 1)} \\"
        )
    rows.append(r"\midrule")
    rows.append(
        rf"Total & {num(100 * c['coverage_standard'], 1)} & {num(c['width_standard'], 1)} & "
        rf"{num(100 * c['coverage_normalized'], 1)} & {num(c['width_normalized_mean'], 1)} \\"
    )
    return _table(
        "conformal",
        "Cobertura empírica en prueba de los intervalos de predicción "
        rf"conformales al {num(100 * c['nominal'], 0)}\,\%",
        r"l r r r r",
        r"Condados & \multicolumn{2}{c}{Estándar} & \multicolumn{2}{c}{Normalizado} \\ "
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5} & Cobertura (\%) & Anchura & "
        r"Cobertura (\%) & Anchura",
        rows,
    )


def t_environment(res: Results) -> str:
    man = res["manifest"]
    env = man["environment"]
    rows = [
        rf"{tex_escape(k)} & \texttt{{{tex_escape(v)}}} \\"
        for k, v in env.items()
        if k != "platform"
    ]
    rows.append(rf"Plataforma & \texttt{{{tex_escape(env.get('platform', ''))}}} \\")
    return _table(
        "entorno",
        "Entorno de ejecución de la corrida del informe",
        r"l X",
        r"Componente & Versión",
        rows,
        env="tabularx",
        size=r"\footnotesize",
    )


TABLES: dict[str, Callable[[Results], str]] = {
    "variables": t_variables,
    "excluidas": t_excluded,
    "validacion": t_validation,
    "decisiones": t_decisions,
    "ausentes": t_missing,
    "respuesta": t_response,
    "normalidad": t_normality,
    "regiones": t_regions,
    "pares": t_pairs,
    "correlaciones": t_correlations,
    "poda": t_collinearity,
    "seleccion": t_selection,
    "interacciones": t_interactions,
    "comparacion": t_comparison,
    "anova": t_anova,
    "coeficientes": t_coefficients,
    "spss-resumen": t_spss_summary,
    "spss-coeficientes": t_spss_coefficients,
    "errores": t_standard_errors,
    "pendientes": t_slopes,
    "efectos": t_effects,
    "diagnostico": t_diagnostics,
    "influyentes": t_influence,
    "sensibilidad": t_sensitivity,
    "incidencia": t_incidence,
    "prediccion": t_cv,
    "conformal": t_conformal,
    "entorno": t_environment,
}


def write_tables(res: Results, out: Path) -> list[str]:
    """Escribe todas las tablas en ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in TABLES.items():
        if name in ("prediccion", "conformal") and not res.get("predictive"):
            (out / f"{name}.tex").write_text("% Modelo predictivo desactivado.\n", encoding="utf-8")
            continue
        (out / f"{name}.tex").write_text(fn(res), encoding="utf-8")
        made.append(name)
    return made
