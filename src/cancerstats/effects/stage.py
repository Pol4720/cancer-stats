"""Orquestación de la etapa de efectos: del modelo máximo al modelo final validado.

Sigue el diagrama del proceso de inferencia del curso:

    modelo máximo → depuración de colinealidad → ajuste inicial (MCO) → diagnóstico →
    modelo de la estructura del error (FGLS + cluster) → contrastes de simplificación
    (selección, confusión, interacciones) → modelo final → diagnóstico → interpretación y
    sensibilidad.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from cancerstats import collinearity
from cancerstats.cleaning import response_column
from cancerstats.config import AnalysisConfig
from cancerstats.dictionary import VARIABLES, label
from cancerstats.effects import interpret, sensitivity
from cancerstats.effects.design import REGION, Design, build_design, term_name
from cancerstats.effects.diagnostics import diagnose, residual_check_before
from cancerstats.effects.fit import (
    coef_table,
    estimate_variance_function,
    fit,
    summary_stats,
    wald,
)
from cancerstats.effects.selection import (
    backward,
    confounding_check,
    interaction_search,
    select,
)

Record = dict[str, object]
Logger = Callable[[str], None]


def _records(df: pd.DataFrame) -> list[Record]:
    return [
        {str(k): v for k, v in row.items()}
        for row in df.replace({np.nan: None}).to_dict(orient="records")
    ]


def _term_tests(res, design: Design, terms: list[str]) -> list[Record]:  # type: ignore[no-untyped-def]
    out = []
    for t in terms:
        f_stat, p, q = wald(res, design.terms[t])
        out.append({"term": t, "etiqueta": label(t), "F": f_stat, "gl": q, "p": p})
    return out


def _construct_blocks(res, design: Design, terms: list[str]) -> list[Record]:  # type: ignore[no-untyped-def]
    """Contrastes F parciales por constructo (todas las variables de educación, de cobertura...)."""
    groups: dict[str, list[str]] = {}
    for t in terms:
        if t in design.numeric:
            groups.setdefault(VARIABLES[t].group if t in VARIABLES else t, []).append(t)
    out = []
    for g, ts in groups.items():
        if len(ts) < 2:
            continue
        cols = [c for t in ts for c in design.terms[t]]
        f_stat, p, q = wald(res, cols)
        out.append({"constructo": g, "terminos": ts, "F": f_stat, "gl": q, "p": p})
    return out


def _nested_comparison(
    design: Design, models: dict[str, list[str]], weights: np.ndarray | None
) -> list[Record]:
    """Comparación de modelos anidados sobre la misma muestra (AIC, BIC, R² ajustado, F parcial)."""
    rows: list[Record] = []
    prev_name: str | None = None
    prev_res = None
    for name, terms in models.items():
        res = fit(design, terms, weights, "nonrobust")
        row: Record = {
            "modelo": name,
            "k": int(res.df_model),
            "r2": float(res.rsquared),
            "r2_adj": float(res.rsquared_adj),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "sce": float(res.ssr),
        }
        if prev_res is not None and res.df_model != prev_res.df_model:
            big, small = (res, prev_res) if res.df_model > prev_res.df_model else (prev_res, res)
            dq = big.df_model - small.df_model
            f_stat = ((small.ssr - big.ssr) / dq) / (big.ssr / big.df_resid)
            row["vs"] = prev_name
            row["F_parcial"] = float(f_stat)
            row["gl"] = [int(dq), int(big.df_resid)]
            row["p_parcial"] = float(stats.f.sf(f_stat, dq, big.df_resid))
        rows.append(row)
        prev_name, prev_res = name, res
    return rows


def boxcox_profile(design: Design, terms: list[str], alpha: float) -> Record:
    """λ de Box-Cox de la respuesta condicionada a las explicativas (verosimilitud perfil)."""
    y = design.y.to_numpy()
    X = design.X[design.columns_for(terms)].to_numpy()
    n = y.size
    log_gm = np.log(y).sum()
    grid = np.linspace(-1, 2, 121)
    ll = []
    for lam in grid:
        z = np.log(y) if abs(lam) < 1e-9 else (y**lam - 1) / lam
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        sse = float(((z - X @ beta) ** 2).sum())
        ll.append(-n / 2 * np.log(sse / n) + (lam - 1) * log_gm)
    ll_arr = np.asarray(ll)
    best = int(ll_arr.argmax())
    cut = ll_arr[best] - stats.chi2.ppf(1 - alpha, 1) / 2
    inside = grid[ll_arr >= cut]
    return {
        "lambda": float(grid[best]),
        "lower": float(inside.min()),
        "upper": float(inside.max()),
        "grid": grid.round(3).tolist(),
        "loglik": (ll_arr - ll_arr[best]).round(3).tolist(),
        "includes_1": bool(inside.min() <= 1 <= inside.max()),
    }


def run_effects(
    df: pd.DataFrame, config: AnalysisConfig, numeric: list[str], log: Logger | None = None
) -> Record:
    """Ejecuta la etapa completa del modelo de efectos."""
    say = log or (lambda _msg: None)
    eff = config.effects
    var = config.variables
    alpha = config.inference.alpha
    seed = config.meta.seed
    response = response_column(config)
    if not eff.include_incidence:
        numeric = [c for c in numeric if c != "incidenceRate"]
    protected = [p for p in (_model_names(var.protected, numeric)) if p in numeric]

    # 1. Colinealidad sobre el modelo máximo ----------------------------------------------
    say("Colinealidad: FIV, índices de condición y poda")
    sample = df.dropna(subset=[response, *numeric])
    coll = collinearity.analyze(
        sample[numeric],
        config.collinearity.corr_threshold,
        config.collinearity.vif_threshold,
        config.collinearity.condition_index_threshold,
        protected,
    )
    kept = list(coll["kept"])  # type: ignore[call-overload]

    pairs: list[list[str]] = [_model_names(p, [*numeric, REGION]) for p in eff.interactions]
    pairs = [
        p
        for p in pairs
        if len(p) == 2
        and all(c in kept or c == REGION for c in p)
        and (REGION not in p or var.use_region)
    ]
    design = build_design(
        df,
        response,
        kept,
        var.use_region,
        var.region_reference,
        [list(p) for p in pairs],
        eff.center_predictors,
    )
    main_terms = kept + ([REGION] if var.use_region else [])
    say(f"Modelo máximo: {len(kept)} continuas + región, n = {design.n}")

    # 2. Ajuste inicial por MCO y diagnóstico que motiva la estructura del error ---------
    initial = fit(design, main_terms, None, "nonrobust")
    before = residual_check_before(design, main_terms)

    # 3. Función de varianza: se estima siempre (diagnóstico y sensibilidad); sólo pondera el
    #    ajuste si el estimador elegido es MCPF.
    say("Estimando la función de varianza σ²(n) = a + b/n")
    vf0, w_fgls0 = estimate_variance_function(design, main_terms, eff.fgls_iterations)
    w0 = w_fgls0 if eff.weighting == "fgls" else None

    # 4. Selección --------------------------------------------------------------------------
    say(f"Selección de variables ({eff.selection}) con covarianza {eff.covariance}")
    sel = select(
        eff.selection,
        design,
        main_terms,
        protected,
        w0,
        eff.covariance,
        eff.alpha_enter,
        eff.alpha_remove,
    )
    strategies = {}
    for method in ("backward", "forward", "stepwise"):
        r = (
            sel
            if method == eff.selection
            else select(
                method,
                design,
                main_terms,
                protected,
                w0,
                eff.covariance,
                eff.alpha_enter,
                eff.alpha_remove,
            )
        )
        strategies[method] = r.to_dict()
    naive = backward(design, main_terms, protected, None, "nonrobust", eff.alpha_remove)

    # 5. Confusión ---------------------------------------------------------------------------
    exposures = _model_names(eff.exposures, numeric)
    terms, confounding = confounding_check(
        design,
        sel.selected,
        main_terms,
        w0,
        eff.covariance,
        eff.confounding_threshold,
        alpha,
        exposures,
    )

    # 6. Interacciones ---------------------------------------------------------------------
    say("Contrastando interacciones (principio jerárquico, corrección de Holm)")
    candidates = [term_name(p) for p in pairs]
    final_terms, inter_trace = interaction_search(
        design, terms, candidates, w0, eff.covariance, alpha, eff.interaction_correction
    )
    main_selected = [t for t in final_terms if ":" not in t]
    comparison = _nested_comparison(
        design,
        {
            "Nulo (región)": [REGION] if var.use_region else [],
            "Máximo (tras la poda)": main_terms,
            "Seleccionado (efectos principales)": terms,
            "Final (con interacciones)": final_terms,
        }
        if var.use_region
        else {
            "Máximo (tras la poda)": main_terms,
            "Seleccionado (efectos principales)": terms,
            "Final (con interacciones)": final_terms,
        },
        w0,
    )

    # 7. Modelo final sobre su propia muestra de casos completos -------------------------
    final_numeric = [t for t in final_terms if t in kept]
    final_pairs = [list(p) for p in pairs if term_name(p) in final_terms]
    fdesign = build_design(
        df,
        response,
        final_numeric,
        REGION in final_terms,
        var.region_reference,
        final_pairs,
        eff.center_predictors,
    )
    vf, w_fgls = estimate_variance_function(fdesign, final_terms, eff.fgls_iterations)
    wf = w_fgls if eff.weighting == "fgls" else None
    say(f"Modelo final: {len(final_terms)} términos, n = {fdesign.n}")
    res = fit(fdesign, final_terms, wf, eff.covariance)
    n_clusters = len(np.unique(fdesign.groups))
    coef = coef_table(res, fdesign, alpha)
    summary = summary_stats(res, n_clusters)
    res_classic = fit(fdesign, final_terms, wf, "nonrobust")
    res_hc3 = fit(fdesign, final_terms, wf, "HC3")
    coef_classic = coef_table(res_classic, fdesign, alpha)
    coef_hc3 = coef_table(res_hc3, fdesign, alpha)

    # 8. Diagnóstico -----------------------------------------------------------------------
    say("Diagnóstico de las hipótesis e influencia")
    diag = diagnose(
        fdesign, final_terms, wf, eff.covariance, alpha, config.inference.multiple_testing, seed
    )
    residual_table: pd.DataFrame = diag.pop("table")  # type: ignore[assignment]
    # Varianza de los residuos por decil de población en la escala original (MCO) y tras
    # ponderar por la función de varianza estimada (MCPF, especificación de sensibilidad).
    pop_bins = pd.qcut(fdesign.population, 10, labels=False, duplicates="drop")
    res_wls = fit(fdesign, final_terms, w_fgls, "nonrobust")
    e2_wls = w_fgls * np.asarray(res_wls.resid) ** 2
    variance_by_decile = {
        "mco": pd.Series(np.asarray(res_classic.resid) ** 2).groupby(pop_bins).mean().tolist(),
        "mcpf": pd.Series(e2_wls).groupby(pop_bins).mean().tolist(),
    }
    vif_by = {v["variable"]: v for v in diag["collinearity"]["vif"]}  # type: ignore[index]
    spss_coef = []
    for row in _records(coef_classic):
        extra = vif_by.get(str(row["term"]), {})
        spss_coef.append({**row, "tolerancia": extra.get("tolerancia"), "fiv": extra.get("fiv")})
    spss = {
        "model_summary": {
            "R": float(np.sqrt(max(summary["r2"], 0.0))),
            "R2": summary["r2"],
            "R2_adj": summary["r2_adj"],
            "se_estimate": summary["sigma"],
            "rmse": summary["rmse_unweighted"],
            "durbin_watson": diag["independence"]["durbin_watson"],  # type: ignore[index]
            "n": summary["n"],
        },
        "anova": {
            "scr": summary["scr"],
            "sce": summary["sce"],
            "sct": summary["sct"],
            "k": summary["k"],
            "n": summary["n"],
            "F": summary["f_classic"],
            "p": summary["p_f_classic"],
        },
        "coefficients": spss_coef,
    }

    # 9. Interpretación -------------------------------------------------------------------
    effects = interpret.effects_table(res, fdesign, df, final_terms, alpha)
    for e in effects:
        e["frase"] = interpret.sentence(e, alpha)
    num_inter = interpret.numeric_interactions(res, fdesign, final_terms, alpha)
    adj_means = interpret.adjusted_region_means(res, fdesign, alpha)

    # 10. Pregunta de confusión/mediación: sin la incidencia -----------------------------
    incidence: Record | None = None
    if "incidenceRate" in final_terms:
        noinc = [t for t in final_terms if "incidenceRate" not in t.split(":")]
        res_no = fit(fdesign, noinc, wf, eff.covariance)
        keys = [t for t in noinc if t in fdesign.numeric or t == REGION]
        cols = [c for t in keys for c in fdesign.terms[t]]
        incidence = {
            "terms_without": noinc,
            "comparison": interpret.compare_coefficients(res.params, res_no.params, cols),
            "r2_with": float(res.rsquared),
            "r2_without": float(res_no.rsquared),
            "coef_without": _records(coef_table(res_no, fdesign, alpha)),
        }

    # 11. Sensibilidad --------------------------------------------------------------------
    say("Análisis de sensibilidad (especificaciones alternativas)")
    w_or_one = wf if wf is not None else np.ones(fdesign.n)
    principal = (
        "Principal: MCPF + errores cluster por estado"
        if eff.weighting == "fgls"
        else "Principal: MCO + errores cluster por estado"
    )
    alternative = (
        sensitivity._from_res(
            "wls",
            "MCPF con σ²(n) = a + b/n y errores cluster",
            fit(fdesign, final_terms, w_fgls, eff.covariance),
        )
        if eff.weighting != "fgls"
        else sensitivity._from_res(
            "ols_cluster",
            "MCO con errores cluster",
            fit(fdesign, final_terms, None, eff.covariance),
        )
    )
    specs: list[Record] = [
        sensitivity._from_res(
            "ols",
            "MCO con errores típicos clásicos (salida tipo SPSS)",
            fit(fdesign, final_terms, None, "nonrobust"),
        ),
        sensitivity._from_res(
            "ols_hc3", "MCO con errores típicos HC3", fit(fdesign, final_terms, None, "HC3")
        ),
        sensitivity._from_res("main", principal, res),
        alternative,
        sensitivity.state_fixed_effects(df, fdesign, final_terms, w_or_one),
        sensitivity.mixed_model(fdesign, final_terms),
        sensitivity.huber(fdesign, final_terms, w_or_one),
    ]
    flagged = diag["influence"]["flagged_ids"]  # type: ignore[index]
    if eff.influence_sensitivity and flagged:
        specs.append(
            sensitivity.without_rows(
                fdesign,
                final_terms,
                w_or_one,
                flagged,
                eff.covariance,
            )
        )
    if config.missing.sensitivity:
        say("Imputación múltiple (sensibilidad al tratamiento de ausentes)")
        specs.append(
            sensitivity.with_multiple_imputation(
                df,
                response,
                final_numeric,
                numeric,
                REGION in final_terms,
                var.region_reference,
                final_pairs,
                final_terms,
                eff.covariance,
                eff.weighting,
                eff.fgls_iterations,
                config.missing.n_imputations,
                config.missing.max_iter,
                seed,
            )
        )
    boot = None
    if eff.bootstrap_reps > 0:
        say(f"Bootstrap por conglomerados ({eff.bootstrap_reps} réplicas)")
        boot = sensitivity.cluster_bootstrap(
            fdesign, final_terms, w_or_one, eff.bootstrap_reps, seed, alpha
        )

    return {
        "response": response,
        "candidates": numeric,
        "protected": protected,
        "collinearity": coll,
        "max_design": {"n": design.n, "terms": main_terms, "interaction_candidates": candidates},
        "initial": {
            "coef": _records(coef_table(initial, design, alpha)),
            "summary": summary_stats(initial, len(np.unique(design.groups))),
            "diagnostics": before,
        },
        "variance_function": vf0.to_dict() if vf0 else None,
        "variance_function_final": vf.to_dict() if vf else None,
        "variance_examples": _variance_examples(vf) if vf else None,
        "variance_by_decile": variance_by_decile,
        "weighting": eff.weighting,
        "covariance": eff.covariance,
        "selection": sel.to_dict(),
        "strategies": strategies,
        "naive_selection": naive.to_dict(),
        "confounding": confounding,
        "exposures": exposures,
        "interactions": inter_trace,
        "model_comparison": comparison,
        "final": {
            "terms": final_terms,
            "main_terms": main_selected,
            "n": fdesign.n,
            "n_clusters": n_clusters,
            "centers": fdesign.centers,
            "coef": _records(coef),
            "coef_classic": _records(coef_classic),
            "coef_hc3": _records(coef_hc3),
            "spss": spss,
            "summary": summary,
            "term_tests": _term_tests(res, fdesign, final_terms),
            "construct_tests": _construct_blocks(res, fdesign, final_terms),
            "cov_type": eff.covariance,
            "weighting": eff.weighting,
            "equation": _equation(coef),
        },
        "diagnostics": diag,
        "residuals": _records(residual_table.round(5)),
        "effects": effects,
        "numeric_interactions": num_inter,
        "adjusted_region_means": adj_means,
        "incidence_adjustment": incidence,
        "sensitivity": specs,
        "bootstrap": boot,
        "boxcox": boxcox_profile(fdesign, final_terms, alpha),
    }


def _variance_examples(vf) -> list[Record]:  # type: ignore[no-untyped-def]
    out: list[Record] = []
    for pop in (2_000, 10_000, 25_000, 100_000, 1_000_000):
        out.append(
            {
                "poblacion": pop,
                "varianza": float(vf.variance(np.array([pop]))[0]),
                "fraccion_muestreo": float(vf.sampling_share(pop)),
            }
        )
    return out


def _equation(coef: pd.DataFrame) -> str:
    parts = []
    for _, r in coef.iterrows():
        c = float(r["coef"])
        name = "" if r["term"] == "const" else f"·{r['term']}"
        parts.append(f"{c:+.3f}{name}")
    return "ŷ = " + " ".join(parts)


def _model_names(names: list[str], available: list[str]) -> list[str]:
    """Traduce nombres originales (popEst2015, medIncome, studyPerCap) a columnas de modelo."""
    mapping = {"popEst2015": "logPop", "medIncome": "logIncome", "studyPerCap": "logStudy"}
    out = []
    for n in names:
        if n in available:
            out.append(n)
            continue
        alt = mapping.get(n)
        if alt and alt in available:
            out.append(alt)
            continue
        alt_any = (
            [a for a in available if a in ("anyStudy", "studyPerCap")] if n == "studyPerCap" else []
        )
        out.extend(alt_any[:1])
    return out
