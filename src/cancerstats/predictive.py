"""Modelo predictivo: comparación honesta de modelos fuera de muestra.

Protocolo:

1. Se reserva una partición de prueba (estratificada por región) que no se usa hasta el
   final.
2. En entrenamiento se comparan los modelos por validación cruzada de K pliegues. Los
   hiperparámetros se eligen *dentro* de cada pliegue (validación cruzada anidada) y la
   imputación y el escalado también se ajustan dentro del pliegue, para no filtrar
   información de validación.
3. Se repite con validación cruzada agrupada por estado, que mide la capacidad de predecir
   condados de estados no vistos (los condados vecinos se parecen; una partición aleatoria
   es optimista).
4. El modelo elegido se evalúa una única vez en prueba, con intervalos de predicción
   conformales (cobertura garantizada sin suponer normalidad).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cancerstats.config import AnalysisConfig
from cancerstats.dictionary import REGIONS, label

Record = dict[str, object]
Logger = Callable[[str], None]

MODEL_LABELS: dict[str, str] = {
    "baseline": "Referencia (media)",
    "ols_effects": "MCO con la especificación del modelo de efectos",
    "ols_full": "MCO con todas las variables",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "Red elástica",
    "random_forest": "Bosque aleatorio",
    "gradient_boosting": "Potenciación del gradiente (histogramas)",
}


class EffectsSpec(BaseEstimator, TransformerMixin):
    """Reproduce la matriz de diseño del modelo de efectos (centrado, región e interacciones).

    Las medianas para imputar y las medias para centrar se aprenden en ``fit`` con los
    datos de entrenamiento del pliegue.
    """

    def __init__(self, numeric: list[str], terms: list[str], reference: str) -> None:
        self.numeric = numeric
        self.terms = terms
        self.reference = reference

    def fit(self, X: pd.DataFrame, y: Any = None) -> EffectsSpec:
        self.medians_ = X[self.numeric].median()
        filled = X[self.numeric].fillna(self.medians_)
        self.means_ = filled.mean()
        self.missing_cols_ = [c for c in self.numeric if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols: dict[str, np.ndarray] = {}
        filled = X[self.numeric].fillna(self.medians_)
        for c in self.numeric:
            cols[c] = (filled[c] - self.means_[c]).to_numpy(dtype=float)
        for c in self.missing_cols_:
            cols[f"{c}_ausente"] = X[c].isna().to_numpy(dtype=float)
        levels = [r for r in REGIONS if r != self.reference]
        dummies = {lev: (X["region"] == lev).to_numpy(dtype=float) for lev in levels}
        if "region" in self.terms:
            for lev in levels:
                cols[f"region[{lev}]"] = dummies[lev]
        for t in self.terms:
            if ":" not in t:
                continue
            a, b = t.split(":")
            if b == "region":
                for lev in levels:
                    cols[f"{a}:{lev}"] = cols[a] * dummies[lev]
            else:
                cols[t] = cols[a] * cols[b]
        return np.column_stack(list(cols.values()))


def _preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def build_models(
    names: list[str],
    numeric: list[str],
    categorical: list[str],
    effects_numeric: list[str],
    effects_terms: list[str],
    reference: str,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Construye los estimadores (cada uno con su propio preprocesado)."""
    alphas = np.logspace(-3, 3, 40)
    inner = KFold(5, shuffle=True, random_state=seed)
    models: dict[str, Any] = {}
    for name in names:
        if name == "baseline":
            models[name] = DummyRegressor(strategy="mean")
        elif name == "ols_effects":
            models[name] = Pipeline(
                [
                    ("spec", EffectsSpec(effects_numeric, effects_terms, reference)),
                    ("ols", LinearRegression()),
                ]
            )
        elif name == "ols_full":
            models[name] = Pipeline(
                [("prep", _preprocessor(numeric, categorical, True)), ("ols", LinearRegression())]
            )
        elif name == "ridge":
            models[name] = Pipeline(
                [
                    ("prep", _preprocessor(numeric, categorical, True)),
                    ("model", RidgeCV(alphas=alphas)),
                ]
            )
        elif name == "lasso":
            models[name] = Pipeline(
                [
                    ("prep", _preprocessor(numeric, categorical, True)),
                    (
                        "model",
                        LassoCV(
                            alphas=np.logspace(-3, 1, 40), cv=inner, max_iter=20000, n_jobs=n_jobs
                        ),
                    ),
                ]
            )
        elif name == "elasticnet":
            models[name] = Pipeline(
                [
                    ("prep", _preprocessor(numeric, categorical, True)),
                    (
                        "model",
                        ElasticNetCV(
                            l1_ratio=[0.1, 0.5, 0.9],
                            alphas=np.logspace(-3, 1, 30),
                            cv=inner,
                            max_iter=20000,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            )
        elif name == "random_forest":
            models[name] = Pipeline(
                [
                    ("prep", _preprocessor(numeric, categorical, False)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_features=0.33,
                            min_samples_leaf=2,
                            random_state=seed,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            )
        elif name == "gradient_boosting":
            base = Pipeline(
                [
                    ("prep", _preprocessor(numeric, categorical, False)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=600,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=30,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            grid = {
                "model__learning_rate": [0.03, 0.08],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            }
            models[name] = GridSearchCV(
                base,
                grid,
                cv=KFold(3, shuffle=True, random_state=seed),
                scoring="neg_root_mean_squared_error",
                n_jobs=n_jobs,
            )
    return models


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def cross_validate(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    splitter: Any,
    groups: np.ndarray | None,
    log: Logger,
) -> tuple[list[Record], dict[str, np.ndarray]]:
    """Validación cruzada: métricas por pliegue y predicciones fuera de pliegue."""
    rows: list[Record] = []
    oof: dict[str, np.ndarray] = {}
    splits = list(splitter.split(X, y, groups))
    for name, model in models.items():
        log(f"  · {MODEL_LABELS[name]}")
        pred = np.empty_like(y, dtype=float)
        folds = []
        for k, (tr, va) in enumerate(splits):
            m = clone(model).fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[va])
            pred[va] = p
            folds.append({"fold": k + 1, **_metrics(y[va], p)})
        f = pd.DataFrame(folds)
        rows.append(
            {
                "model": name,
                "label": MODEL_LABELS[name],
                "rmse_mean": float(f["rmse"].mean()),
                "rmse_sd": float(f["rmse"].std(ddof=1)),
                "mae_mean": float(f["mae"].mean()),
                "r2_mean": float(f["r2"].mean()),
                "r2_sd": float(f["r2"].std(ddof=1)),
                "folds": f.round(4).to_dict(orient="records"),
                "oof": _metrics(y, pred),
            }
        )
        oof[name] = pred
    return rows, oof


def conformal(
    oof_resid: np.ndarray,
    scale_train: np.ndarray,
    pred_test: np.ndarray,
    scale_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    pop_test: np.ndarray,
) -> Record:
    """Intervalos de predicción conformales: estándar y normalizados por σ(población).

    Estándar: ŷ ± q, con q el cuantil (1 − α)(1 + 1/n) de los |residuos| fuera de pliegue.
    Normalizado: ŷ ± q·σ_i con q el cuantil de |residuo|/σ_i: intervalos más anchos en
    condados pequeños, donde la tasa es más ruidosa.
    """
    n = oof_resid.size
    level = min(1.0, (1 - alpha) * (1 + 1 / n))
    q_std = float(np.quantile(np.abs(oof_resid), level))
    q_norm = float(np.quantile(np.abs(oof_resid) / scale_train, level))
    lo_s, hi_s = pred_test - q_std, pred_test + q_std
    lo_n, hi_n = pred_test - q_norm * scale_test, pred_test + q_norm * scale_test
    cover_s = (y_test >= lo_s) & (y_test <= hi_s)
    cover_n = (y_test >= lo_n) & (y_test <= hi_n)
    terc = pd.qcut(pop_test, 3, labels=["pequeños", "medianos", "grandes"])
    by = []
    for t in ["pequeños", "medianos", "grandes"]:
        m = np.asarray(terc == t)
        by.append(
            {
                "tercil": t,
                "cobertura_estandar": float(cover_s[m].mean()),
                "cobertura_normalizada": float(cover_n[m].mean()),
                "anchura_estandar": float(2 * q_std),
                "anchura_normalizada": float(np.mean(hi_n[m] - lo_n[m])),
            }
        )
    return {
        "alpha": alpha,
        "nominal": 1 - alpha,
        "q_standard": q_std,
        "q_normalized": q_norm,
        "coverage_standard": float(cover_s.mean()),
        "coverage_normalized": float(cover_n.mean()),
        "width_standard": float(2 * q_std),
        "width_normalized_mean": float(np.mean(hi_n - lo_n)),
        "by_population": by,
    }


def _feature_names(model: Any) -> list[str] | None:
    est = model.best_estimator_ if isinstance(model, GridSearchCV) else model
    if isinstance(est, Pipeline) and "prep" in est.named_steps:
        return list(est.named_steps["prep"].get_feature_names_out())
    return None


def partial_dependence_curve(
    model: Any, X: pd.DataFrame, var: str, points: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Dependencia parcial media de ``var``: predicción media al fijar ``var`` en cada valor.

    La rejilla va del percentil 5 al 95 de los valores observados (los ausentes se ignoran
    al calcularla; el modelo los imputa como en el ajuste).
    """
    values = X[var].to_numpy(dtype=float)
    lo, hi = np.nanquantile(values, [0.05, 0.95])
    grid = np.linspace(lo, hi, points)
    average = np.empty(points)
    work = X.copy()
    for i, g in enumerate(grid):
        work[var] = g
        average[i] = float(np.mean(model.predict(work)))
    return grid, average


def run_predictive(
    df: pd.DataFrame,
    config: AnalysisConfig,
    numeric: list[str],
    effects_numeric: list[str],
    effects_terms: list[str],
    variance_ab: tuple[float, float] | None,
    log: Logger | None = None,
) -> Record:
    """Ejecuta la etapa predictiva completa."""
    say = log or (lambda _m: None)
    pc = config.predictive
    seed = config.meta.seed
    response = "TARGET_deathRate"
    data = df.dropna(subset=[response]).reset_index(drop=True)
    categorical = ["region", *(["state"] if pc.use_state else [])]
    features = [*numeric, *categorical]
    X = data[features]
    y = data[response].to_numpy(dtype=float)
    idx_train, idx_test = train_test_split(
        np.arange(len(data)), test_size=pc.test_size, random_state=seed, stratify=data["region"]
    )
    Xtr, Xte = X.iloc[idx_train], X.iloc[idx_test]
    ytr, yte = y[idx_train], y[idx_test]
    groups_tr = data["state"].to_numpy()[idx_train]
    models = build_models(
        list(pc.models),
        numeric,
        categorical,
        effects_numeric,
        effects_terms,
        config.variables.region_reference,
        seed,
        pc.n_jobs,
    )

    say(f"Validación cruzada aleatoria de {pc.cv_folds} pliegues")
    cv_rows, oof = cross_validate(
        models, Xtr, ytr, KFold(pc.cv_folds, shuffle=True, random_state=seed), None, say
    )
    group_rows: list[Record] = []
    if pc.group_cv:
        say(f"Validación cruzada agrupada por estado ({pc.cv_folds} grupos de estados)")
        group_rows, _ = cross_validate(models, Xtr, ytr, GroupKFold(pc.cv_folds), groups_tr, say)

    ranking = sorted(cv_rows, key=lambda r: float(r["rmse_mean"]))  # type: ignore[arg-type]
    best = str(ranking[0]["model"])
    best_rmse = float(ranking[0]["rmse_mean"])  # type: ignore[arg-type]
    best_se = float(ranking[0]["rmse_sd"]) / np.sqrt(pc.cv_folds)  # type: ignore[arg-type]
    complexity = [
        "baseline",
        "ols_effects",
        "lasso",
        "elasticnet",
        "ridge",
        "ols_full",
        "random_forest",
        "gradient_boosting",
    ]
    within = [str(r["model"]) for r in cv_rows if float(r["rmse_mean"]) <= best_rmse + best_se]  # type: ignore[arg-type]
    one_se = min(within, key=complexity.index)
    chosen = best if pc.selection_rule == "min" else one_se

    say("Evaluación final en la partición de prueba")
    test_rows: list[Record] = []
    fitted: dict[str, Any] = {}
    test_pred: dict[str, np.ndarray] = {}
    for name, model in models.items():
        m = clone(model).fit(Xtr, ytr)
        fitted[name] = m
        p = m.predict(Xte)
        test_pred[name] = p
        test_rows.append({"model": name, "label": MODEL_LABELS[name], **_metrics(yte, p)})

    rng = np.random.default_rng(seed)
    boot = []
    base_name = (
        "ols_effects"
        if "ols_effects" in test_pred and chosen != "ols_effects"
        else ("baseline" if "baseline" in test_pred else chosen)
    )
    for _ in range(2000):
        b = rng.integers(0, yte.size, yte.size)
        r_best = np.sqrt(np.mean((yte[b] - test_pred[chosen][b]) ** 2))
        r_base = np.sqrt(np.mean((yte[b] - test_pred[base_name][b]) ** 2))
        boot.append((r_best, r_best - r_base))
    barr = np.asarray(boot)
    test_ci = {
        "rmse_lower": float(np.quantile(barr[:, 0], 0.025)),
        "rmse_upper": float(np.quantile(barr[:, 0], 0.975)),
        "diff_vs": base_name,
        "diff_lower": float(np.quantile(barr[:, 1], 0.025)),
        "diff_upper": float(np.quantile(barr[:, 1], 0.975)),
    }

    pop_tr = data["popEst2015"].to_numpy(dtype=float)[idx_train]
    pop_te = data["popEst2015"].to_numpy(dtype=float)[idx_test]
    if variance_ab is not None:
        a, b_ = variance_ab
        s_tr, s_te = np.sqrt(a + b_ / pop_tr), np.sqrt(a + b_ / pop_te)
    else:
        s_tr, s_te = np.ones_like(pop_tr), np.ones_like(pop_te)
    conf = conformal(
        ytr - oof[chosen], s_tr, test_pred[chosen], s_te, yte, pc.conformal_alpha, pop_te
    )

    say("Importancia por permutación y dependencia parcial")
    perm = permutation_importance(
        fitted[chosen],
        Xte,
        yte,
        n_repeats=15,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=pc.n_jobs,
    )
    importance = sorted(
        [
            {"variable": f, "etiqueta": label(f), "importancia": float(m), "dt": float(s)}
            for f, m, s in zip(features, perm.importances_mean, perm.importances_std, strict=True)
        ],
        key=lambda r: -float(r["importancia"]),  # type: ignore[arg-type]
    )
    pd_rows = []
    top_numeric = [r["variable"] for r in importance if r["variable"] in numeric][:4]
    for var in top_numeric:
        grid, average = partial_dependence_curve(fitted[chosen], Xtr, str(var))
        pd_rows.append(
            {
                "variable": var,
                "etiqueta": label(str(var)),
                "grid": grid.round(4).tolist(),
                "average": average.round(4).tolist(),
            }
        )

    lasso_info: Record | None = None
    if "lasso" in fitted:
        las = fitted["lasso"]
        coef = las.named_steps["model"].coef_
        names = _feature_names(las) or []
        nz = [(n, float(c)) for n, c in zip(names, coef, strict=False) if abs(c) > 1e-10]
        lasso_info = {
            "alpha": float(las.named_steps["model"].alpha_),
            "n_nonzero": len(nz),
            "n_total": len(names),
            "top": sorted(nz, key=lambda t: -abs(t[1]))[:15],
        }
    gb_params = None
    if "gradient_boosting" in fitted:
        gb_params = {
            k.replace("model__", ""): v for k, v in fitted["gradient_boosting"].best_params_.items()
        }

    return {
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "features": features,
        "cv": cv_rows,
        "group_cv": group_rows,
        "best": best,
        "best_label": MODEL_LABELS[best],
        "one_se_choice": one_se,
        "chosen": chosen,
        "chosen_label": MODEL_LABELS[chosen],
        "selection_rule": pc.selection_rule,
        "test": test_rows,
        "test_ci": test_ci,
        "conformal": conf,
        "importance": importance[:20],
        "partial_dependence": pd_rows,
        "lasso": lasso_info,
        "gradient_boosting_params": gb_params,
        "test_predictions": {
            "county_id": data["county_id"].iloc[idx_test].astype(str).tolist(),
            "y": yte.round(3).tolist(),
            "pred": test_pred[chosen].round(3).tolist(),
            "pred_effects": test_pred.get("ols_effects", test_pred[chosen]).round(3).tolist(),
            "region": data["region"].iloc[idx_test].astype(str).tolist(),
        },
    }
