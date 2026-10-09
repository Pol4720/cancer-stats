"""Catálogo de reglas de validación y decisiones de depuración sobre los datos oficiales."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancerstats.cleaning import clean
from cancerstats.config import AnalysisConfig
from cancerstats.validation import (
    EDU_18_24,
    RACES,
    by_id,
    find_sentinels,
    r19_cases_minus_deaths,
    summarize,
    validate,
)


@pytest.fixture(scope="module")
def cleaned(raw_official: pd.DataFrame, config: AnalysisConfig):  # type: ignore[no-untyped-def]
    return clean(raw_official, config)


def test_catalogue_is_complete(raw_official: pd.DataFrame, config: AnalysisConfig) -> None:
    results = validate(raw_official, config.cleaning)
    ids = [r.id for r in results]
    assert ids == [f"R{i:02d}" for i in range(1, 20)]
    assert summarize(results)["total"] == 19


def test_raw_problems_are_detected(raw_official: pd.DataFrame, config: AnalysisConfig) -> None:
    res = by_id(validate(raw_official, config.cleaning))
    assert res["R04"].n_violations > 0  # centinelas de una imputación previa
    assert res["R05"].n_violations > 0  # edad mediana registrada en meses
    assert res["R07"].n_violations > 0  # tamaño del hogar dividido por 100
    assert res["R19"].n_violations == 0  # Notificadomuerte = casos − muertes


def test_cleaning_resolves_every_error(cleaned, config: AnalysisConfig) -> None:  # type: ignore[no-untyped-def]
    after = validate(cleaned.data, config.cleaning, cleaned.sentinels)
    assert summarize(after)["error"] == 0
    assert summarize(after)["advertencia"] == 0


def test_decisions_are_documented(cleaned) -> None:  # type: ignore[no-untyped-def]
    ids = [d.id for d in cleaned.decisions]
    assert ids == [f"D{i:02d}" for i in range(1, 12)]
    for d in cleaned.decisions:
        assert all((d.problem, d.action, d.rationale, d.config_key)), d.id


def test_cleaned_values_are_plausible(cleaned) -> None:  # type: ignore[no-untyped-def]
    d = cleaned.data
    assert d["MedianAge"].between(15, 100).all()
    assert (d["AvgHouseholdSize"] >= 1).all()
    assert d["PctSomeCol18_24"].notna().all()
    np.testing.assert_allclose(d[list(EDU_18_24)].sum(axis=1), 100, atol=0.2)
    races = d[[*RACES, "PctNativeMulti"]].sum(axis=1)
    assert (races >= 100 - 1e-6).all()
    assert set(d["region"]) == {"Sur", "Noreste", "Medio Oeste", "Oeste"}
    for col in cleaned.sentinels:
        mask = np.isclose(d[col].astype(float), cleaned.sentinels[col], atol=1e-6)
        assert not mask.any()


def test_leakage_variables_are_excluded(cleaned) -> None:  # type: ignore[no-untyped-def]
    for v in ("avgDeathsPerYear", "Notificadomuerte", "TARGET_deathRate"):
        assert v in cleaned.excluded_variables


def test_r19_detects_a_broken_identity(raw_official: pd.DataFrame, config: AnalysisConfig) -> None:
    broken = raw_official.copy()
    i = broken["Notificadomuerte"].first_valid_index()
    broken.loc[i, "Notificadomuerte"] += 5
    assert r19_cases_minus_deaths(broken, config.cleaning, {}).n_violations == 1


def test_sentinel_detector() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(100, 10, 500).round(1)
    x[:40] = 453.5494221
    found = find_sentinels(pd.DataFrame({"x": x}), min_repeats=10, min_decimals=4)
    assert found == {"x": pytest.approx(453.5494221)}
    assert find_sentinels(pd.DataFrame({"x": x[40:]}), 10, 4) == {}
