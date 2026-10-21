"""Variables indicadoras de atípicos: equivalencias del modelo de cambio en la media."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cancerstats.effects import indicators
from cancerstats.effects.design import build_design
from cancerstats.effects.fit import fit
from cancerstats.predictive import IndicatorOLS
from tests.conftest import synthetic_counties

TERMS = ["x1", "x2", "region"]


def _contaminated() -> pd.DataFrame:
    df = synthetic_counties(n=500, seed=7)
    df.loc[[3, 40], "y"] += 60.0  # dos atípicos verdaderos
    return df


def _design(df: pd.DataFrame):  # type: ignore[no-untyped-def]
    return build_design(df, "y", ["x1", "x2"], True, "Sur", [], True)


def test_indicator_t_is_the_externally_studentized_residual() -> None:
    d = _design(_contaminated())
    res = fit(d, TERMS, None, "nonrobust")
    stud = res.get_influence().resid_studentized_external
    alt = indicators.fit_with_indicators(d, TERMS, [3], None, "nonrobust")
    assert alt.tvalues["D1"] == pytest.approx(stud[3])


def test_indicators_equal_dropping_the_rows() -> None:
    d = _design(_contaminated())
    rows = [3, 40, 99]
    alt = indicators.fit_with_indicators(d, TERMS, rows, None, "nonrobust")
    mask = np.ones(d.n, dtype=bool)
    mask[rows] = False
    dropped = fit(d, TERMS, None, "nonrobust", rows=mask)
    for c in d.columns_for(TERMS):
        assert alt.params[c] == pytest.approx(dropped.params[c])
    # δ_i es el error de predicción del condado con el ajuste que no lo usa.
    x = d.X[d.columns_for(TERMS)].iloc[rows[0]]
    assert alt.params["D1"] == pytest.approx(d.y.iloc[rows[0]] - float(x @ dropped.params))


def test_mean_shift_model_finds_planted_outliers() -> None:
    d = _design(_contaminated())
    res = fit(d, TERMS, None, "nonrobust")
    infl = res.get_influence()
    cook = infl.cooks_distance[0]
    bonf = indicators.bonferroni_outliers(
        d, infl.resid_studentized_external, len(res.params), 0.05, cook
    )
    assert set(bonf["rows"]) == {3, 40}
    m = indicators.mean_shift_model(d, TERMS, [3, 40], None, "cluster", 0.05, "x", "x", "x", cook)
    # El F parcial coincide con el que compara la SCE con y sin esos condados.
    mask = np.ones(d.n, dtype=bool)
    mask[[3, 40]] = False
    sse_all = fit(d, TERMS, None, "nonrobust").ssr
    sse_drop = fit(d, TERMS, None, "nonrobust", rows=mask).ssr
    df_resid = d.n - len(res.params) - 2
    assert m["F"] == pytest.approx(((sse_all - sse_drop) / 2) / (sse_drop / df_resid))
    assert m["n_bonferroni"] == 2
    assert m["p"] < 1e-6
    assert indicators.verdict(m, 0.05)["atipicos"]


def test_calibration_matches_observed_selection() -> None:
    d = _design(_contaminated())
    res = fit(d, TERMS, None, "nonrobust")
    cook = res.get_influence().cooks_distance[0]
    flagged = np.flatnonzero(cook > 4 / d.n).tolist()
    cal = indicators.calibrate_cook_blocks(d, TERMS, None, 2, 49, 0)
    m = indicators.mean_shift_model(d, TERMS, flagged, None, "nonrobust", 0.05, "x", "x", "x", cook)
    assert cal["k_obs"] == len(flagged)
    assert cal["F_obs"] == pytest.approx(m["F"])
    # Con dos atípicos plantados, el F de los dos de mayor Cook supera a la nula normal.
    assert cal["normal"]["p_F_top"] < 0.05


def test_indicator_ols_drops_the_most_influential_rows() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.3, 200)
    y[[5, 17]] += 25.0
    model = IndicatorOLS(k=2).fit(X, y)
    assert set(model.indicated_) == {5, 17}
    keep = np.setdiff1d(np.arange(200), [5, 17])
    ref = LinearRegression().fit(X[keep], y[keep])
    np.testing.assert_allclose(model.predict(X[:5]), ref.predict(X[:5]))
