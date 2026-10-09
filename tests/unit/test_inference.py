"""Inferencia clásica: intervalos, contrastes y correcciones por multiplicidad."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from cancerstats import inference as inf


@pytest.fixture
def sample() -> np.ndarray:
    return np.random.default_rng(3).normal(50, 8, 200)


def test_mean_ci_matches_t_interval(sample: np.ndarray) -> None:
    out = inf.mean_ci(sample)
    lo, hi = stats.t.interval(0.95, len(sample) - 1, sample.mean(), stats.sem(sample))
    assert out["lower"] == pytest.approx(lo)
    assert out["upper"] == pytest.approx(hi)


def test_variance_ci_contains_estimate(sample: np.ndarray) -> None:
    out = inf.variance_ci(sample)
    assert out["lower"] < np.var(sample, ddof=1) < out["upper"]


def test_median_ci_coverage() -> None:
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(400):
        x = rng.exponential(1.0, 60)
        out = inf.median_ci_order(x)
        hits += out["lower"] <= np.log(2) <= out["upper"]
    assert 0.92 <= hits / 400 <= 0.99


@pytest.mark.parametrize("method", ["holm", "bonferroni", "fdr_bh"])
def test_adjust_matches_statsmodels(method: str) -> None:
    p = [0.001, 0.01, 0.02, 0.04, 0.3]
    expected = multipletests(p, method=method)[1]
    np.testing.assert_allclose(inf.adjust(p, method), expected)


def test_runs_test_detects_alternation() -> None:
    x = np.tile([1.0, -1.0], 50)
    assert inf.runs_test(x)["p"] < 0.001
    assert inf.runs_test(np.random.default_rng(1).normal(size=200))["p"] > 0.01


def test_normality_battery_on_normal_and_skewed() -> None:
    rng = np.random.default_rng(5)
    normal = {t["test"]: t["p"] for t in inf.normality_battery(rng.normal(size=500))}
    skewed = {t["test"]: t["p"] for t in inf.normality_battery(rng.exponential(size=500))}
    assert normal["Jarque-Bera"] > 0.01
    assert skewed["Jarque-Bera"] < 1e-6


def test_boxcox_recovers_log() -> None:
    x = np.exp(np.random.default_rng(2).normal(3, 0.5, 2000))
    out = inf.boxcox_lambda(x)
    assert out["lower"] <= 0 <= out["upper"]


def test_two_sample_and_correlation() -> None:
    rng = np.random.default_rng(9)
    a, b = rng.normal(0, 1, 300), rng.normal(0.5, 1, 300)
    assert inf.two_sample(a, b)["welch"]["p"] < 0.001
    x = rng.normal(size=300)
    y = 0.6 * x + rng.normal(size=300)
    out = inf.correlation_test(x, y)
    assert out["pearson"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert out["pearson_lower"] < out["pearson"] < out["pearson_upper"]
