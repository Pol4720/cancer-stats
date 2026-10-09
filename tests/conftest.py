"""Datos y configuraciones compartidas por las pruebas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cancerstats.config import AnalysisConfig
from cancerstats.dictionary import STATE_TO_REGION
from cancerstats.io import read_raw
from cancerstats.paths import project_root

ROOT = project_root()
SAV = ROOT / "data" / "raw" / "practica.sav"
CSV = ROOT / "data" / "raw" / "CANCER.csv"


@pytest.fixture(scope="session")
def config() -> AnalysisConfig:
    """Configuración por defecto del proyecto."""
    return AnalysisConfig.load(ROOT / "config" / "default.yaml")


@pytest.fixture(scope="session")
def raw_official() -> pd.DataFrame:
    """Fichero oficial del profesor (SPSS)."""
    df, _ = read_raw(SAV)
    return df


@pytest.fixture(scope="session")
def raw_reference() -> pd.DataFrame:
    """CSV público de Kaggle."""
    df, _ = read_raw(CSV)
    return df


def synthetic_counties(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Condados artificiales con la estructura que espera la matriz de diseño.

    La respuesta sigue un modelo lineal conocido: ``y = 180 + 2·x1 − 1,5·x2 + 0,8·z + ε``,
    con ``z`` correlacionada con ``x1`` (confusora) y un efecto regional.
    """
    rng = np.random.default_rng(seed)
    states = sorted(STATE_TO_REGION)[:30]
    state = rng.choice(states, n)
    region = np.array([STATE_TO_REGION[s] for s in state])
    x1 = rng.normal(10, 3, n)
    z = 0.8 * x1 + rng.normal(0, 1.5, n)
    x2 = rng.normal(5, 2, n)
    noise = rng.normal(0, 4, n)
    shift = pd.Series(region).map({"Sur": 6.0, "Noreste": -4.0, "Medio Oeste": 0.0, "Oeste": -6.0})
    y = 180 + 2 * x1 - 1.5 * x2 + 0.8 * z + shift.to_numpy() + noise
    return pd.DataFrame(
        {
            "county_id": [f"C{i:04d}" for i in range(n)],
            "state": state,
            "region": region,
            "popEst2015": rng.integers(2_000, 500_000, n),
            "x1": x1,
            "x2": x2,
            "z": z,
            "noise": rng.normal(size=n),
            "y": y,
        }
    )


@pytest.fixture
def counties() -> pd.DataFrame:
    """Condados artificiales (ver :func:`synthetic_counties`)."""
    return synthetic_counties()


@pytest.fixture
def tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Raíz de proyecto temporal (corridas e informe se escriben ahí)."""
    monkeypatch.setenv("CANCERSTATS_ROOT", str(tmp_path))
    return tmp_path
