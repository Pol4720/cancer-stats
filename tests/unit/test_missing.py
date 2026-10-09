"""Contraste MCAR de Little y combinación de imputaciones (reglas de Rubin)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancerstats.missing import little_mcar_test, rubin_pool


def _mvn(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    return pd.DataFrame(rng.multivariate_normal([0, 0, 0], cov, n), columns=["a", "b", "c"])


def test_little_does_not_reject_mcar() -> None:
    df = _mvn(1500, 0)
    rng = np.random.default_rng(1)
    df.loc[rng.random(1500) < 0.2, "b"] = np.nan
    df.loc[rng.random(1500) < 0.15, "c"] = np.nan
    res = little_mcar_test(df, ["a", "b", "c"])
    assert res.converged
    assert res.p_value > 0.01
    assert res.df > 0


def test_little_rejects_mar() -> None:
    df = _mvn(1500, 2)
    df.loc[df["a"] > 0.3, "b"] = np.nan  # la ausencia de b depende de a
    res = little_mcar_test(df, ["a", "b", "c"])
    assert res.p_value < 1e-6


def test_rubin_without_between_variance_reduces_to_complete_data() -> None:
    est = [pd.Series({"x": 2.0}) for _ in range(5)]
    var = [pd.Series({"x": 0.25}) for _ in range(5)]
    pooled = rubin_pool(est, var, df_complete=100)
    assert pooled.params["x"] == pytest.approx(2.0)
    assert pooled.se["x"] == pytest.approx(0.5)
    assert pooled.between["x"] == pytest.approx(0.0)


def test_rubin_total_variance() -> None:
    est = [pd.Series({"x": v}) for v in (1.0, 2.0, 3.0)]
    var = [pd.Series({"x": 0.5}) for _ in range(3)]
    pooled = rubin_pool(est, var, df_complete=50)
    # T = Ū + (1 + 1/m) B = 0,5 + (4/3)·1
    assert pooled.se["x"] ** 2 == pytest.approx(0.5 + 4 / 3)
    assert 0 < pooled.fmi["x"] < 1
