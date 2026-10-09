"""Diseño, ajuste, selección, confusión e interacciones sobre datos con verdad conocida."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from cancerstats.effects.design import build_design, term_name
from cancerstats.effects.fit import estimate_variance_function, fit, wald
from cancerstats.effects.selection import backward, confounding_check, interaction_search
from tests.conftest import synthetic_counties


def _design(df: pd.DataFrame, numeric: list[str], pairs=(), center: bool = True):  # type: ignore[no-untyped-def]
    return build_design(df, "y", numeric, True, "Sur", list(pairs), center)


def test_design_centering_and_dummies(counties: pd.DataFrame) -> None:
    d = _design(counties, ["x1", "x2"])
    assert d.X["x1"].mean() == pytest.approx(0, abs=1e-10)
    assert d.terms["region"] == ["region[Noreste]", "region[Medio Oeste]", "region[Oeste]"]
    assert d.n == len(counties)


def test_ols_matches_formula_api(counties: pd.DataFrame) -> None:
    d = _design(counties, ["x1", "x2"], center=False)
    res = fit(d, ["x1", "x2", "region"], cov="nonrobust")
    ref = smf.ols("y ~ x1 + x2 + C(region, Treatment('Sur'))", data=counties).fit()
    assert res.params["x1"] == pytest.approx(ref.params["x1"])
    assert res.bse["x2"] == pytest.approx(ref.bse["x2"])
    assert res.rsquared == pytest.approx(ref.rsquared)


def test_cluster_inference_uses_g_minus_1(counties: pd.DataFrame) -> None:
    d = _design(counties, ["x1", "x2"])
    res = fit(d, ["x1", "x2"], cov="cluster")
    g = len(np.unique(d.groups))
    _, p, q = wald(res, ["x1"])
    assert q == 1
    assert p < 1e-6
    assert res.df_resid_inference == g - 1


def test_backward_drops_pure_noise(counties: pd.DataFrame) -> None:
    d = _design(counties, ["x1", "x2", "z", "noise"])
    sel = backward(d, ["x1", "x2", "z", "noise", "region"], [], None, "cluster", 0.05)
    assert "noise" not in sel.selected
    assert {"x1", "x2", "z"} <= set(sel.selected)


def test_confounder_is_restored() -> None:
    df = synthetic_counties(n=800, seed=4)
    d = _design(df, ["x1", "x2", "z"])
    # Se simula que la selección descartó z, que confunde la asociación de x1.
    kept, trace = confounding_check(
        d, ["x1", "x2", "region"], ["x1", "x2", "z", "region"], None, "cluster", 0.1, 0.05, ["x1"]
    )
    assert "z" in kept
    assert trace[0]["en"] == "x1"
    # Si no se vigila x1, nada se reincorpora.
    kept2, _ = confounding_check(
        d, ["x1", "x2", "region"], ["x1", "x2", "z", "region"], None, "cluster", 0.1, 0.05, ["x2"]
    )
    assert "z" not in kept2


def test_interactions_follow_hierarchy() -> None:
    df = synthetic_counties(n=900, seed=7)
    df["y"] += 1.5 * (df["x1"] - df["x1"].mean()) * (df["region"] == "Oeste")
    pairs = [["x1", "region"], ["x2", "noise"]]
    d = _design(df, ["x1", "x2", "noise"], pairs)
    terms, trace = interaction_search(
        d, ["x1", "x2", "region"], [term_name(p) for p in pairs], None, "cluster", 0.05, "holm"
    )
    assert "x1:region" in terms
    # x2:noise no se contrasta porque noise no está en el modelo.
    assert all(t["termino"] != "x2:noise" for t in trace)


def test_variance_function_recovers_population_pattern() -> None:
    rng = np.random.default_rng(11)
    df = synthetic_counties(n=1500, seed=11)
    pop = df["popEst2015"].to_numpy(dtype=float)
    df["y"] = 100 + 2 * df["x1"] + rng.normal(0, np.sqrt(4 + 2e6 / pop))
    d = _design(df, ["x1"])
    vf, w = estimate_variance_function(d, ["x1"], 5)
    assert 1e6 < vf.b < 3e6  # verdadero: 2·10⁶
    assert vf.p_b < 0.05
    assert w.mean() == pytest.approx(1.0)
    assert np.corrcoef(w, pop)[0, 1] > 0
