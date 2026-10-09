"""Orquestador del pipeline: ejecuta las etapas en orden y persiste la corrida."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cancerstats import __version__, cleaning, dictionary, exploration, outliers, validation
from cancerstats.config import AnalysisConfig
from cancerstats.effects.stage import run_effects
from cancerstats.io import read_raw
from cancerstats.missing import analyze_missing
from cancerstats.paths import project_root
from cancerstats.predictive import run_predictive
from cancerstats.provenance import compare_sources
from cancerstats.runs import (
    RunRegistry,
    created_now,
    dataset_fingerprint,
    environment,
    git_state,
    new_run_id,
)

STAGES: tuple[tuple[str, str], ...] = (
    ("ingesta", "Ingesta del fichero original"),
    ("validacion", "Validación de reglas"),
    ("depuracion", "Depuración razonada"),
    ("ausentes", "Mecanismo de los datos ausentes"),
    ("exploracion", "Análisis exploratorio e inferencia"),
    ("atipicos", "Atípicos univariantes y multivariantes"),
    ("efectos", "Modelo de estimación de efectos"),
    ("prediccion", "Modelo predictivo"),
    ("persistencia", "Persistencia de la corrida"),
)

DATASET_COLUMNS: tuple[str, ...] = (
    "county_id",
    "Geography",
    "state",
    "region",
    "TARGET_deathRate",
    "incidenceRate",
    "avgAnnCount",
    "avgDeathsPerYear",
    "medIncome",
    "logIncome",
    "popEst2015",
    "logPop",
    "povertyPercent",
    "studyPerCap",
    "logStudy",
    "binnedInc",
    "MedianAge",
    "MedianAgeMale",
    "MedianAgeFemale",
    "AvgHouseholdSize",
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
    "PctNativeMulti",
    "PctMarriedHouseholds",
    "BirthRate",
    "Notificadomuerte",
)


@dataclass
class Event:
    """Evento de progreso que emite el pipeline (lo consume la interfaz en vivo)."""

    stage: str
    status: str
    message: str
    progress: float

    def to_dict(self) -> dict[str, Any]:
        """Representación serializable."""
        return dict(self.__dict__)


Listener = Callable[[Event], None]


def numeric_candidates(config: AnalysisConfig) -> list[str]:
    """Columnas de modelo de las explicativas candidatas (con las transformaciones)."""
    return [cleaning.model_column(c, config) for c in config.variables.candidates]


def run(
    config: AnalysisConfig,
    registry: RunRegistry | None = None,
    listener: Listener | None = None,
    persist: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Ejecuta el pipeline completo.

    Returns:
        El identificador de la corrida y el diccionario de resultados.
    """
    registry = registry or RunRegistry()
    run_id = new_run_id(config)
    # El estado del código se registra al empezar: es el que realmente se ejecuta.
    code_state = git_state(project_root())
    log_lines: list[str] = []
    timings: dict[str, float] = {}
    started = time.perf_counter()
    total = len(STAGES)

    def emit(stage: str, status: str, message: str, k: int) -> None:
        stamp = time.strftime("%H:%M:%S")
        log_lines.append(f"[{stamp}] {stage:<12} {status:<6} {message}")
        if listener:
            listener(Event(stage, status, message, round(k / total, 4)))

    def stage(k: int, key: str, fn: Callable[[], Any]) -> Any:
        title = dict(STAGES)[key]
        emit(key, "inicio", title, k)
        t0 = time.perf_counter()
        out = fn()
        timings[key] = round(time.perf_counter() - t0, 3)
        emit(key, "fin", f"{title} ({timings[key]:.1f} s)", k + 1)
        return out

    def sub(key: str) -> Callable[[str], None]:
        def _log(msg: str) -> None:
            emit(key, "info", msg, list(dict(STAGES)).index(key))

        return _log

    root = project_root()
    data_path = Path(config.data.path)
    if not data_path.is_absolute():
        data_path = root / data_path

    raw, ingest = stage(0, "ingesta", lambda: read_raw(data_path, config.data.encoding))
    rules_raw = stage(1, "validacion", lambda: validation.validate(raw, config.cleaning))
    provenance: dict[str, Any] | None = None
    if config.data.reference_path:
        ref_path = Path(config.data.reference_path)
        if not ref_path.is_absolute():
            ref_path = root / ref_path
        if ref_path.exists() and ref_path.resolve() != data_path.resolve():
            ref, ref_ingest = read_raw(ref_path, config.data.encoding)
            provenance = {**compare_sources(raw, ref), "reference": ref_ingest.to_dict()}
    cleaned = stage(2, "depuracion", lambda: cleaning.clean(raw, config, ingest.format))
    df = cleaning.add_response(cleaned.data, config)
    rules_clean = validation.validate(df, config.cleaning, cleaned.sentinels)

    numeric = numeric_candidates(config)
    response = cleaning.response_column(config)
    missing_cols = [response, *numeric, "PctPrivateCoverageAlone"]
    miss = stage(
        3,
        "ausentes",
        lambda: analyze_missing(
            df,
            missing_cols,
            ["incidenceRate"],
            config.inference.alpha,
            config.inference.multiple_testing,
        ),
    )
    explore = stage(
        4,
        "exploracion",
        lambda: exploration.explore(
            df,
            response,
            numeric,
            config.inference.alpha,
            config.inference.bootstrap_reps,
            config.meta.seed,
            config.inference.multiple_testing,
        ),
    )
    atyp = stage(
        5,
        "atipicos",
        lambda: outliers.detect(
            df,
            response,
            numeric,
            config.outliers.iqr_factor,
            config.outliers.robust_z,
            config.outliers.mahalanobis_quantile,
            config.meta.seed,
        ),
    )
    effects = stage(6, "efectos", lambda: run_effects(df, config, numeric, sub("efectos")))

    predictive: dict[str, Any] | None = None
    if config.predictive.enabled:
        final_terms = list(effects["final"]["terms"])
        eff_numeric = [t for t in final_terms if ":" not in t and t != "region"]
        vf = effects.get("variance_function_final")
        ab = (vf["a"], vf["b"]) if vf else None
        predictive = stage(
            7,
            "prediccion",
            lambda: run_predictive(
                df, config, numeric, eff_numeric, final_terms, ab, sub("prediccion")
            ),
        )

    results: dict[str, Any] = {
        "run_id": run_id,
        "version": __version__,
        "stages": [{"id": k, "title": t} for k, t in STAGES],
        "dictionary": dictionary.as_records(),
        "excluded": cleaned.excluded_variables,
        "ingest": ingest.to_dict(),
        "provenance": provenance,
        "validation": {
            "raw": [r.to_dict() for r in rules_raw],
            "clean": [r.to_dict() for r in rules_clean],
            "summary_raw": validation.summarize(rules_raw),
            "summary_clean": validation.summarize(rules_clean),
        },
        "cleaning": {
            "decisions": [d.to_dict() for d in cleaned.decisions],
            "sentinels": cleaned.sentinels,
            # El JSON redondea los reales a 7 cifras: el valor exacto viaja como texto para que
            # la sintaxis de SPSS lo reconozca sin ambigüedad.
            "sentinels_exact": {k: repr(v) for k, v in cleaned.sentinels.items()},
            "n_rows": len(df),
        },
        "missing": miss.to_dict(),
        "exploration": explore,
        "outliers": atyp,
        "effects": effects,
        "predictive": predictive,
    }

    duration = round(time.perf_counter() - started, 2)
    manifest = {
        "run_id": run_id,
        "created_at": created_now(),
        "duration_s": duration,
        "timings_s": timings,
        "package_version": __version__,
        "config_name": config.meta.name,
        "config_description": config.meta.description,
        "config_fingerprint": config.fingerprint(),
        "seed": config.meta.seed,
        "data_path": config.data.path,
        "data_sha256": ingest.sha256,
        "git": code_state,
        "environment": environment(),
    }
    results["manifest"] = manifest
    if persist:
        emit("persistencia", "inicio", "Guardando la corrida", total - 1)
        dataset = df[[c for c in DATASET_COLUMNS if c in df.columns]].copy()
        tables = _tables(results)
        registry.save(run_id, config, manifest, results, dataset, log_lines, tables)
        results["dataset"] = dataset_fingerprint(dataset)
        emit("persistencia", "fin", f"Corrida {run_id} guardada", total)
    return run_id, results


def _tables(results: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Tablas CSV de conveniencia que acompañan a cada corrida."""
    eff = results["effects"]
    out = {
        "validacion_original": pd.DataFrame(results["validation"]["raw"]).drop(
            columns=["examples"]
        ),
        "decisiones": pd.DataFrame(results["cleaning"]["decisions"]).drop(
            columns=["affected", "evidence"]
        ),
        "coeficientes": pd.DataFrame(eff["final"]["coef"]),
        "sensibilidad": pd.DataFrame(
            [
                {
                    "especificacion": s["label"],
                    "termino": k,
                    "coef": v,
                    "lower": s["lower"].get(k),
                    "upper": s["upper"].get(k),
                    "p": s["p"].get(k),
                }
                for s in eff["sensitivity"]
                for k, v in s["coef"].items()
            ]
        ),
        "descriptivos": pd.DataFrame(results["exploration"]["describe"]),
        "residuos": pd.DataFrame(eff["residuals"]),
    }
    if results.get("predictive"):
        pred = results["predictive"]
        out["validacion_cruzada"] = pd.DataFrame(pred["cv"]).drop(columns=["folds", "oof"])
        out["prueba"] = pd.DataFrame(pred["test"])
    return out
