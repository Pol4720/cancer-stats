"""Lectura de las dos fuentes y comprobación de su equivalencia."""

from __future__ import annotations

import pandas as pd
import pytest

from cancerstats.io import (
    EXPECTED_COLUMNS,
    decode,
    detect_encoding,
    detect_line_ending,
    read_raw,
    sha256_bytes,
)
from cancerstats.provenance import compare_sources
from tests.conftest import CSV, SAV


def test_official_source_shape(raw_official: pd.DataFrame) -> None:
    assert len(raw_official) == 3047
    for col in EXPECTED_COLUMNS:
        assert col in raw_official.columns
    assert raw_official["county_id"].is_unique
    assert "Notificadomuerte" in raw_official.columns


def test_official_report() -> None:
    _, report = read_raw(SAV)
    assert report.format == "spss"
    assert report.sha256 == sha256_bytes(SAV.read_bytes())


def test_reference_encoding_and_line_endings() -> None:
    data = CSV.read_bytes()
    assert detect_line_ending(data) == "CR"
    assert detect_encoding(data) == "mac_roman"
    text, enc = decode(data)
    assert enc == "mac_roman"
    assert "Doña Ana County" in text


def test_geography_split(raw_official: pd.DataFrame) -> None:
    row = raw_official.loc[raw_official["Geography"].str.startswith("Kitsap County")].iloc[0]
    assert row["state"] == "Washington"


def test_sources_are_equivalent(raw_official: pd.DataFrame, raw_reference: pd.DataFrame) -> None:
    cmp = compare_sources(raw_official, raw_reference)
    assert cmp["same_order"]
    assert cmp["equivalent"]
    assert "Notificadomuerte" in cmp["only_official"]


def test_tampered_source_is_detected(
    raw_official: pd.DataFrame, raw_reference: pd.DataFrame
) -> None:
    tampered = raw_reference.copy()
    tampered.loc[10, "TARGET_deathRate"] += 1.0
    cmp = compare_sources(raw_official, tampered)
    assert not cmp["equivalent"]
    bad = {c["variable"]: c for c in cmp["differing"]}
    assert bad["TARGET_deathRate"]["distintas"] == 1


def test_unknown_extension(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "datos.xlsx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="formato"):
        read_raw(path)
