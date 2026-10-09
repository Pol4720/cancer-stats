"""Selección de variables: hacia atrás, hacia delante y por pasos (como en el curso).

La diferencia con la receta clásica es que los contrastes de entrada y salida usan la
matriz de covarianzas que el diagnóstico ha mostrado adecuada (robusta por conglomerados y
con ponderación FGLS). Seleccionar con las t clásicas, que suponen errores independientes y
homocedásticos, retendría variables cuya «significación» es un artefacto de esos supuestos.

Además de la selección, el módulo aplica el criterio de confusión del curso (una variable
eliminada vuelve si su retirada cambia los coeficientes retenidos más de un umbral) y la
búsqueda de interacciones con el principio jerárquico.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from statsmodels.stats.multitest import multipletests

from cancerstats.effects.design import REGION, Design
from cancerstats.effects.fit import fit, wald

Record = dict[str, object]


@dataclass
class SelectionResult:
    """Términos seleccionados y traza de la selección."""

    method: str
    selected: list[str]
    steps: list[Record] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Representación serializable."""
        return {"method": self.method, "selected": self.selected, "steps": self.steps}


def _term_test(
    design: Design, terms: list[str], term: str, weights: np.ndarray | None, cov: str
) -> tuple[float, float, int]:
    res = fit(design, terms, weights, cov)
    return wald(res, design.terms[term])


def backward(
    design: Design,
    candidates: list[str],
    protected: list[str],
    weights: np.ndarray | None,
    cov: str,
    alpha_remove: float,
) -> SelectionResult:
    """Eliminación hacia atrás: se retira el término menos significativo mientras p > α."""
    current = list(candidates)
    steps: list[Record] = []
    while True:
        res = fit(design, current, weights, cov)
        tests = {t: wald(res, design.terms[t]) for t in current if t not in protected}
        if not tests:
            break
        worst = max(tests, key=lambda t: tests[t][1])
        f_stat, p, q = tests[worst]
        if p <= alpha_remove:
            break
        current.remove(worst)
        steps.append(
            {
                "paso": len(steps) + 1,
                "accion": "sale",
                "termino": worst,
                "F": f_stat,
                "p": p,
                "gl": q,
                "n_terminos": len(current),
            }
        )
    return SelectionResult("backward", current, steps)


def forward(
    design: Design,
    candidates: list[str],
    protected: list[str],
    weights: np.ndarray | None,
    cov: str,
    alpha_enter: float,
) -> SelectionResult:
    """Selección hacia delante: entra el término más significativo mientras p < α."""
    current = [t for t in candidates if t in protected]
    steps: list[Record] = []
    while True:
        pool = [t for t in candidates if t not in current]
        if not pool:
            break
        tests = {t: _term_test(design, [*current, t], t, weights, cov) for t in pool}
        best = min(tests, key=lambda t: tests[t][1])
        f_stat, p, q = tests[best]
        if p >= alpha_enter:
            break
        current.append(best)
        steps.append(
            {
                "paso": len(steps) + 1,
                "accion": "entra",
                "termino": best,
                "F": f_stat,
                "p": p,
                "gl": q,
                "n_terminos": len(current),
            }
        )
    return SelectionResult("forward", current, steps)


def stepwise(
    design: Design,
    candidates: list[str],
    protected: list[str],
    weights: np.ndarray | None,
    cov: str,
    alpha_enter: float,
    alpha_remove: float,
    max_steps: int = 200,
) -> SelectionResult:
    """Por pasos: tras cada entrada se revisa si algún término ya incluido debe salir."""
    current = [t for t in candidates if t in protected]
    steps: list[Record] = []
    seen: set[tuple[str, ...]] = set()
    for _ in range(max_steps):
        changed = False
        pool = [t for t in candidates if t not in current]
        if pool:
            tests = {t: _term_test(design, [*current, t], t, weights, cov) for t in pool}
            best = min(tests, key=lambda t: tests[t][1])
            if tests[best][1] < alpha_enter:
                current.append(best)
                steps.append(
                    {
                        "paso": len(steps) + 1,
                        "accion": "entra",
                        "termino": best,
                        "F": tests[best][0],
                        "p": tests[best][1],
                        "gl": tests[best][2],
                        "n_terminos": len(current),
                    }
                )
                changed = True
        res = fit(design, current, weights, cov) if current else None
        if res is not None:
            removable = {t: wald(res, design.terms[t]) for t in current if t not in protected}
            if removable:
                worst = max(removable, key=lambda t: removable[t][1])
                if removable[worst][1] > alpha_remove:
                    current.remove(worst)
                    steps.append(
                        {
                            "paso": len(steps) + 1,
                            "accion": "sale",
                            "termino": worst,
                            "F": removable[worst][0],
                            "p": removable[worst][1],
                            "gl": removable[worst][2],
                            "n_terminos": len(current),
                        }
                    )
                    changed = True
        state = tuple(sorted(current))
        if not changed or state in seen:
            break
        seen.add(state)
    return SelectionResult("stepwise", current, steps)


def select(
    method: str,
    design: Design,
    candidates: list[str],
    protected: list[str],
    weights: np.ndarray | None,
    cov: str,
    alpha_enter: float,
    alpha_remove: float,
) -> SelectionResult:
    """Despacha la estrategia de selección configurada."""
    if method == "backward":
        return backward(design, candidates, protected, weights, cov, alpha_remove)
    if method == "forward":
        return forward(design, candidates, protected, weights, cov, alpha_enter)
    if method == "stepwise":
        return stepwise(design, candidates, protected, weights, cov, alpha_enter, alpha_remove)
    return SelectionResult("none", list(candidates), [])


def confounding_check(
    design: Design,
    selected: list[str],
    candidates: list[str],
    weights: np.ndarray | None,
    cov: str,
    threshold: float,
    alpha: float,
    exposures: list[str] | None = None,
) -> tuple[list[str], list[Record]]:
    """Reincorpora las variables de confusión eliminadas (criterio del cambio en la estimación).

    La confusión se define respecto de un factor de estudio: para cada término eliminado se
    compara el modelo seleccionado con el que lo añade, y si el coeficiente de alguna
    *exposición de interés* retenida y significativa cambia más que ``threshold`` (en términos
    relativos), el término confunde esa asociación y vuelve al modelo. Se reincorpora primero
    el de mayor cambio y se repite hasta que ninguno lo supera. Sin ``exposures`` se vigilan
    todas las explicativas continuas significativas.
    """
    current = list(selected)
    trace: list[Record] = []
    while True:
        base = fit(design, current, weights, cov)
        watch = [
            t for t in current if t in design.numeric and (exposures is None or t in exposures)
        ]
        monitored = [t for t in watch if float(base.pvalues[t]) < alpha]
        removed = [t for t in candidates if t not in current]
        best: tuple[str, float, str] | None = None
        for term in removed:
            alt = fit(design, [*current, term], weights, cov)
            for m in monitored:
                b0 = float(base.params[m])
                change = abs(float(alt.params[m]) - b0) / abs(b0) if b0 != 0 else 0.0
                if change > threshold and (best is None or change > best[1]):
                    best = (term, change, m)
        if best is None:
            break
        current.append(best[0])
        trace.append({"reincorporada": best[0], "cambio_relativo": best[1], "en": best[2]})
    return current, trace


def interaction_search(
    design: Design,
    base_terms: list[str],
    candidates: list[str],
    weights: np.ndarray | None,
    cov: str,
    alpha: float,
    correction: str,
) -> tuple[list[str], list[Record]]:
    """Búsqueda hacia delante de interacciones con corrección de Holm/Bonferroni.

    Sólo se contrastan interacciones cuyos efectos principales están en el modelo (principio
    jerárquico). En cada paso se contrastan todas las candidatas restantes, se corrigen los
    p-valores por su número y entra la más significativa si su p corregido es menor que α.
    """

    def components(term: str) -> list[str]:
        return term.split(":")

    current = list(base_terms)
    pool = [
        t
        for t in candidates
        if all(c in current or c == REGION for c in components(t))
        and (REGION not in components(t) or REGION in current)
    ]
    trace: list[Record] = []
    round_no = 0
    while pool:
        round_no += 1
        tests = {t: _term_test(design, [*current, t], t, weights, cov) for t in pool}
        names = list(tests)
        raw = [tests[t][1] for t in names]
        adj = raw if correction == "none" else list(multipletests(raw, method=correction)[1])
        for t, p_raw, p_adj in zip(names, raw, adj, strict=True):
            trace.append(
                {
                    "ronda": round_no,
                    "termino": t,
                    "F": tests[t][0],
                    "gl": tests[t][2],
                    "p": p_raw,
                    "p_ajustado": float(p_adj),
                    "n_candidatas": len(names),
                    "entra": False,
                }
            )
        best_i = int(np.argmin(adj))
        if adj[best_i] >= alpha:
            break
        best = names[best_i]
        trace[-len(names) + best_i]["entra"] = True
        current.append(best)
        pool.remove(best)
    return current, trace
