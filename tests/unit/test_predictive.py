"""Dependencia parcial con ausentes en las explicativas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from cancerstats.predictive import partial_dependence_curve


def test_partial_dependence_ignores_missing_values() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = 3 * X["a"] - X["b"]
    X.loc[::7, "a"] = np.nan
    model = make_pipeline(SimpleImputer(), LinearRegression()).fit(X, y)
    grid, avg = partial_dependence_curve(model, X, "a", points=10)
    assert np.isfinite(grid).all()
    assert grid[0] < grid[-1]
    slope = np.polyfit(grid, avg, 1)[0]
    assert abs(slope - 3) < 0.05
