"""Configuración tipada: valores por defecto, validación y huella."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cancerstats.cli import load_config
from cancerstats.config import AnalysisConfig, merge_overrides, parse_dotted_override
from tests.conftest import ROOT


def test_yaml_equals_code_defaults() -> None:
    assert AnalysisConfig.load(ROOT / "config" / "default.yaml") == AnalysisConfig()


def test_fingerprint_is_stable_and_sensitive() -> None:
    a = AnalysisConfig()
    assert a.fingerprint() == AnalysisConfig().fingerprint()
    b = load_config(None, ["effects.alpha_remove=0.1"])
    assert b.effects.alpha_remove == 0.1
    assert a.fingerprint() != b.fingerprint()


def test_yaml_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = load_config(None, ["predictive.cv_folds=5"])
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert AnalysisConfig.load(path) == cfg


@pytest.mark.parametrize("leak", ["avgDeathsPerYear", "Notificadomuerte", "TARGET_deathRate"])
def test_leakage_predictors_are_rejected(leak: str) -> None:
    with pytest.raises(ValidationError, match="No pueden usarse"):
        AnalysisConfig.from_mapping({"variables": {"candidates": [leak]}})


def test_out_of_range_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisConfig.from_mapping({"effects": {"alpha_remove": 1.5}})
    with pytest.raises(ValidationError):
        AnalysisConfig.from_mapping({"effects": {"interactions": [["x", "x"]]}})


def test_dotted_overrides() -> None:
    assert parse_dotted_override("a.b.c=[1, 2]") == {"a": {"b": {"c": [1, 2]}}}
    merged = merge_overrides({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}})
    assert merged == {"a": {"b": 5, "c": 2}}
    with pytest.raises(ValueError, match="clave=valor"):
        parse_dotted_override("sin-igual")


def test_schema_has_titles() -> None:
    schema = AnalysisConfig.model_json_schema()
    assert {"data", "cleaning", "effects", "predictive"} <= set(schema["properties"])
