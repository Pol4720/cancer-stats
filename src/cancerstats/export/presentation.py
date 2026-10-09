"""Presentación HTML autónoma para la defensa, generada a partir de una corrida.

La plantilla (``templates/presentacion.html``) contiene el diseño, la navegación y los gráficos
SVG animados; este módulo extrae de la corrida las cifras que se muestran y las incrusta como
JSON. El resultado es un único fichero sin dependencias externas: funciona sin conexión,
desde el disco o publicado en GitHub Pages.
"""

from __future__ import annotations

import json
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

from cancerstats import jsonutil
from cancerstats.dictionary import label
from cancerstats.paths import project_root
from cancerstats.runs import RunRegistry

REGION_ORDER = ["Sur", "Medio Oeste", "Oeste", "Noreste"]


def _term(term: str) -> str:
    if term.startswith("region["):
        return "Región: " + term[7:-1]
    if ":" in term:
        a, b = term.split(":", 1)
        return f"{label(a)} × {b[7:-1] if b.startswith('region[') else label(b)}"
    return label(term)


def slide_data(res: dict[str, Any]) -> dict[str, Any]:
    """Cifras que necesita la presentación (todas salen de la corrida)."""
    eff = res["effects"]
    fin = eff["final"]
    ms = fin["spss"]["model_summary"]
    dec = {d["id"]: d for d in res["cleaning"]["decisions"]}
    val = res["validation"]
    expl = res["exploration"]
    pred = res.get("predictive") or {}
    effects = []
    for e in eff["effects"]:
        if e.get("interaccion_region"):
            for r in e["by_region"]:
                effects.append(
                    {
                        "label": f"{e['etiqueta']} · {r['region']}",
                        "region": r["region"],
                        "v": r["iqr_effect"],
                        "lo": r["iqr_lower"],
                        "hi": r["iqr_upper"],
                        "sig": r["p"] < 0.05,
                    }
                )
        else:
            effects.append(
                {
                    "label": e["etiqueta"],
                    "v": e["iqr_effect"],
                    "lo": e["iqr_lower"],
                    "hi": e["iqr_upper"],
                    "sig": e["p"] < 0.05,
                }
            )
    effects.sort(key=lambda x: -x["v"])
    slopes = []
    for e in eff["effects"]:
        if e.get("interaccion_region"):
            rows = {r["region"]: r for r in e["by_region"]}
            slopes.append(
                {
                    "label": e["etiqueta"],
                    "rows": [
                        {
                            "region": reg,
                            "v": rows[reg]["coef"],
                            "lo": rows[reg]["lower"],
                            "hi": rows[reg]["upper"],
                            "sig": rows[reg]["p"] < 0.05,
                        }
                        for reg in REGION_ORDER
                        if reg in rows
                    ],
                }
            )
    key = "PctBachDeg25_Over"
    sens = [
        {
            "label": s["label"],
            "v": s["coef"][key],
            "lo": s["lower"][key],
            "hi": s["upper"][key],
            "main": s["id"] == "main",
        }
        for s in eff["sensitivity"]
        if key in s["coef"]
    ]
    boot = eff.get("bootstrap")
    if boot and key in boot.get("coef", {}):
        sens.append(
            {
                "label": f"Bootstrap por conglomerados ({boot['reps']})",
                "v": boot["coef"][key],
                "lo": boot["lower"][key],
                "hi": boot["upper"][key],
                "main": False,
            }
        )
    diag = eff["diagnostics"]
    norm = {t["test"]: t for t in diag["normality"]["tests"]}
    by_dec = eff.get("variance_by_decile") or {}
    edges = eff["initial"]["diagnostics"]["population_decile_edges"]
    vf = eff.get("variance_function_final") or {}
    regions = {g["group"]: g for g in expl["region"]["groups"]}
    adj = {a["region"]: a for a in eff["adjusted_region_means"]}
    return {
        "run_id": res["run_id"],
        "n_counties": res["ingest"]["n_rows"],
        "n_columns": res["ingest"]["n_columns_raw"],
        "response": {
            "min": expl["response"]["extremes"]["lowest"][0],
            "max": expl["response"]["extremes"]["highest"][0],
            "mean": expl["response"]["mean_ci"]["estimate"],
        },
        "rules": {
            "total": val["summary_raw"]["total"],
            "errors_raw": val["summary_raw"]["error"],
            "errors_clean": val["summary_clean"]["error"],
        },
        "decisions": len(res["cleaning"]["decisions"]),
        "problems": [
            {"title": "Valor centinela en la incidencia", "n": dec["D03"]["n_affected"]},
            {"title": "Edad mediana en meses", "n": dec["D04"]["n_affected"]},
            {"title": "Tamaño del hogar ÷ 100", "n": dec["D05"]["n_affected"]},
            {"title": "Dato recuperado por identidad", "n": dec["D06"]["n_affected"]},
        ],
        "little": {
            "random_p": res["missing"]["little_random"]["p_value"],
            "all_p": res["missing"]["little_all"]["p_value"],
        },
        "missing_incidence": next(
            (c["pct"] for c in res["missing"]["by_column"] if c["variable"] == "incidenceRate"),
            None,
        ),
        "steps": [
            {"label": "Candidatas", "n": len(eff["candidates"])},
            {
                "label": "Tras la poda (FIV ≤ 10)",
                "n": len(eff["candidates"]) - len(eff["collinearity"]["trace"]),
            },
            {"label": "Selección hacia atrás", "n": len(eff["selection"]["selected"])},
            {
                "label": "+ confusoras",
                "n": len(eff["selection"]["selected"]) + len(eff["confounding"]),
            },
            {"label": "+ interacciones", "n": len(fin["terms"])},
        ],
        "variance": {
            "mids": [(a * b) ** 0.5 for a, b in pairwise(edges)],
            "ols": by_dec.get("mco") or diag["homoscedasticity"]["var_by_population_decile"],
            "wls": by_dec.get("mcpf"),
            "a": vf.get("a"),
            "b": vf.get("b"),
        },
        "icc": diag["independence"]["icc_state"]["icc"],
        "fit": {
            "r2": ms["R2"],
            "r2_adj": ms["R2_adj"],
            "rmse": ms["rmse"],
            "se": ms["se_estimate"],
            "dw": ms["durbin_watson"],
            "n": ms["n"],
            "k": fin["summary"]["k"],
            "states": fin["n_clusters"],
        },
        "effects": effects,
        "slopes": slopes,
        "regions": [
            {
                "region": r,
                "raw": regions[r]["mean"],
                "adj": adj[r]["ajustada"] if r in adj else None,
            }
            for r in REGION_ORDER
            if r in regions
        ],
        "diagnostics": [
            {
                "name": "Linealidad",
                "test": "RESET",
                "p": diag["linearity"]["reset"]["p"],
                "action": "Se cumple",
            },
            {
                "name": "Independencia",
                "test": "Correlación intraclase por estado",
                "p": diag["independence"]["icc_state"]["p"],
                "action": "Errores típicos por estado",
            },
            {
                "name": "Homocedasticidad",
                "test": "Breusch-Pagan",
                "p": diag["homoscedasticity"]["breusch_pagan"]["p"],
                "action": "Errores típicos robustos",
            },
            {
                "name": "Normalidad",
                "test": "Jarque-Bera",
                "p": norm["Jarque-Bera"]["p"],
                "action": "TCL y bootstrap",
            },
            {
                "name": "Multicolinealidad",
                "test": "FIV máximo",
                "value": max(v["fiv"] for v in diag["collinearity"]["vif"]),
                "action": "FIV < 10",
            },
        ],
        "sensitivity": {"term": label(key), "rows": sens},
        "prediction": {
            "chosen": pred.get("chosen_label"),
            "cv": [
                {
                    "label": r["label"],
                    "rmse": r["rmse_mean"],
                    "chosen": r["model"] == pred.get("chosen"),
                }
                for r in sorted(pred.get("cv", []), key=lambda r: r["rmse_mean"])
                if r["model"] != "baseline"
            ],
            "baseline": next(
                (r["rmse"] for r in pred.get("test", []) if r["model"] == "baseline"), None
            ),
            "test_rmse": next(
                (r["rmse"] for r in pred.get("test", []) if r["model"] == pred.get("chosen")), None
            ),
            "test_r2": next(
                (r["r2"] for r in pred.get("test", []) if r["model"] == pred.get("chosen")), None
            ),
        },
    }


def build_presentation(
    run_id: str | None = None, registry: RunRegistry | None = None, out: Path | None = None
) -> Path:
    """Escribe ``presentation/index.html`` con las cifras de la corrida indicada (o la última)."""
    reg = registry or RunRegistry()
    rid = run_id or reg.latest()
    if rid is None:
        raise FileNotFoundError("No hay corridas: ejecute antes «cancerstats run».")
    data = jsonutil.to_jsonable(slide_data(reg.results(rid)))
    template = resources.files("cancerstats.export").joinpath("templates/presentacion.html")
    html = template.read_text(encoding="utf-8").replace(
        "/*__DATOS__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    )
    target = out or project_root() / "presentation" / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target
