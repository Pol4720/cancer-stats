"""Pipeline completo con una configuración reducida: coherencia, verificación y exportación.

Las cifras del modelo final se vuelven a estimar de forma independiente (fórmulas de
statsmodels sobre el conjunto persistido) para comprobar que lo que llega al informe es
exactamente lo que produce un ajuste MCO estándar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from cancerstats.cli import load_config
from cancerstats.export import export_run
from cancerstats.pipeline import STAGES, run
from cancerstats.runs import RunRegistry

pytestmark = pytest.mark.slow

FAST = [
    "missing.n_imputations=3",
    "effects.bootstrap_reps=0",
    "inference.bootstrap_reps=200",
    "predictive.models=[baseline, ols_effects, ridge, lasso]",
    "predictive.cv_folds=3",
    "predictive.n_jobs=1",
]


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> RunRegistry:
    return RunRegistry(tmp_path_factory.mktemp("runs"))


@pytest.fixture(scope="module")
def outcome(registry: RunRegistry) -> tuple[str, dict[str, Any]]:
    events: list[Any] = []
    rid, res = run(load_config(None, FAST), registry, events.append)
    res["_events"] = events
    return rid, res


@pytest.fixture(scope="module")
def dataset(outcome, registry: RunRegistry) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    return registry.dataset(outcome[1]["dataset"])


def test_all_stages_run(outcome) -> None:  # type: ignore[no-untyped-def]
    _, res = outcome
    assert [s["id"] for s in res["stages"]] == [k for k, _ in STAGES]
    progress = [e.progress for e in res["_events"]]
    assert progress == sorted(progress)
    assert progress[-1] == pytest.approx(1.0)


def test_official_source_and_provenance(outcome) -> None:  # type: ignore[no-untyped-def]
    _, res = outcome
    assert res["ingest"]["format"] == "spss"
    assert res["provenance"]["equivalent"]


def test_summary_statistics_are_consistent(outcome) -> None:  # type: ignore[no-untyped-def]
    s = outcome[1]["effects"]["final"]["summary"]
    n, k = s["n"], s["k"]
    assert s["r2"] == pytest.approx(s["scr"] / s["sct"], rel=1e-5)
    assert s["r2_adj"] == pytest.approx(1 - (1 - s["r2"]) * (n - 1) / (n - k - 1), rel=1e-5)
    assert s["rmse_unweighted"] == pytest.approx(np.sqrt(s["sce"] / n), rel=1e-5)
    assert s["sigma"] == pytest.approx(np.sqrt(s["sce"] / (n - k - 1)), rel=1e-5)


def _formula(final: dict[str, Any]) -> str:
    centers = final["centers"]
    parts = []
    for t in final["terms"]:
        if t == "region":
            parts.append("C(region, Treatment('Sur'))")
        elif ":" in t:
            a, b = t.split(":")
            right = "C(region, Treatment('Sur'))" if b == "region" else f"I({b} - {centers[b]!r})"
            parts.append(f"I({a} - {centers[a]!r}):{right}")
        else:
            parts.append(f"I({t} - {centers[t]!r})")
    return "y ~ " + " + ".join(parts)


def test_final_model_is_reproduced_independently(outcome, dataset: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    eff = outcome[1]["effects"]
    final = eff["final"]
    data = dataset.copy()
    data["y"] = data[eff["response"]]
    used = [t for t in final["terms"] if t in final["centers"]]
    data = data.dropna(subset=["y", *used])
    ref = smf.ols(_formula(final), data=data).fit()
    assert int(ref.nobs) == final["n"]
    assert ref.rsquared == pytest.approx(final["summary"]["r2"], rel=1e-5)
    classic = {c["term"]: c for c in final["coef_classic"]}
    for t in used:
        name = f"I({t} - {final['centers'][t]!r})"
        assert ref.params[name] == pytest.approx(classic[t]["coef"], rel=1e-4)
        assert ref.bse[name] == pytest.approx(classic[t]["se"], rel=1e-4)


def test_nested_comparison_has_robust_tests(outcome) -> None:  # type: ignore[no-untyped-def]
    rows = outcome[1]["effects"]["model_comparison"]
    for r in rows[1:]:
        assert 0 <= r["p_robusto"] <= 1
        assert r["F_robusto"] > 0
    ids = [s["id"] for s in outcome[1]["effects"]["sensitivity"]]
    assert "maximo" in ids


def test_robust_and_classic_share_estimates(outcome) -> None:  # type: ignore[no-untyped-def]
    final = outcome[1]["effects"]["final"]
    for table in ("coef_classic", "coef_hc3"):
        other = {c["term"]: c["coef"] for c in final[table]}
        for c in final["coef"]:
            assert other[c["term"]] == pytest.approx(c["coef"], rel=1e-6)


def test_spss_block_matches_summary(outcome) -> None:  # type: ignore[no-untyped-def]
    final = outcome[1]["effects"]["final"]
    ms = final["spss"]["model_summary"]
    assert ms["R2_adj"] == pytest.approx(final["summary"]["r2_adj"])
    assert ms["rmse"] == pytest.approx(final["summary"]["rmse_unweighted"])
    assert 0 < ms["durbin_watson"] < 4


def test_predictive_split_is_honest(outcome) -> None:  # type: ignore[no-untyped-def]
    pred = outcome[1]["predictive"]
    assert pred["n_train"] + pred["n_test"] == outcome[1]["ingest"]["n_rows"]
    base = next(r for r in pred["test"] if r["model"] == "baseline")
    chosen = next(r for r in pred["test"] if r["model"] == pred["chosen"])
    assert chosen["rmse"] < base["rmse"]


def test_partial_dependence_is_informative(outcome) -> None:  # type: ignore[no-untyped-def]
    for row in outcome[1]["predictive"]["partial_dependence"]:
        assert all(g is not None for g in row["grid"]), row["variable"]
        assert max(row["average"]) > min(row["average"]), row["variable"]


def test_registry_roundtrip(outcome, registry: RunRegistry) -> None:  # type: ignore[no-untyped-def]
    rid, _ = outcome
    assert registry.latest() == rid
    assert [r["id"] for r in registry.index()] == [rid]
    assert registry.config(rid) == load_config(None, FAST)
    assert registry.results(rid)["run_id"] == rid
    assert (registry.paths(rid).root / "log.txt").read_text(encoding="utf-8").strip()


def test_export(outcome, registry: RunRegistry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    rid, res = outcome
    assert export_run(rid, registry, tmp_path, with_figures=True) == rid
    macros = (tmp_path / "resultados.tex").read_text(encoding="utf-8")
    keys = re.findall(r"\\resdef\{([^}]+)\}", macros)
    assert len(keys) == len(set(keys))
    assert len(keys) > 300
    assert {"r2aj-final", "rmse-final", "r2-final"} <= set(keys)
    for tex in (tmp_path / "tablas").glob("*.tex"):
        body = tex.read_text(encoding="utf-8")
        assert body.count("{") == body.count("}"), tex.name
    assert len(list((tmp_path / "figuras").glob("*.pdf"))) >= 15
    sps = (tmp_path / "spss" / "modelo_final.sps").read_text(encoding="utf-8")
    assert "REGRESSION" in sps
    assert "/RESIDUALS DURBIN" in sps
    for c in res["effects"]["final"]["centers"]:
        assert f"c_{c}" in sps


def test_web_export(outcome, registry: RunRegistry, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from cancerstats.export.web import export_web

    rid, _ = outcome
    assert export_web(rid, registry, tmp_path) == rid
    run_dir = tmp_path / "runs" / rid
    for name in ("results.json", "dataset.json", "config.json", "modelo_final.sps", "log.txt"):
        assert (run_dir / name).is_file(), name
    index = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    assert index["latest"] == rid
    assert index["mode"] == "static"
    schema = json.loads((tmp_path / "config-schema.json").read_text(encoding="utf-8"))
    assert "effects" in schema["properties"]


def test_spss_syntax_flags_the_sentinel_exactly(outcome, registry: RunRegistry) -> None:  # type: ignore[no-untyped-def]
    """La sintaxis de SPSS debe reconocer el centinela tal como está en el fichero oficial."""
    from cancerstats.export.spss import syntax
    from cancerstats.io import read_raw
    from tests.conftest import SAV

    _, res = outcome
    sps = syntax(res, "practica.sav")
    raw, _ = read_raw(SAV)
    for col in res["cleaning"]["sentinels"]:
        line = next(ln for ln in sps.splitlines() if ln.startswith(f"IF (ABS({col} - "))
        value = float(line.split(" - ")[1].split(")")[0])
        tol = float(line.split("< ")[1].split(")")[0])
        n_flagged = int(((raw[col] - value).abs() < tol).sum())
        expected = next(d["n_affected"] for d in res["cleaning"]["decisions"] if d["id"] == "D03")
        assert n_flagged == expected
