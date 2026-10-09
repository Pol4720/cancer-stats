"""Figuras del informe (PDF vectorial), generadas a partir de una corrida."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cancerstats.dictionary import label
from cancerstats.export import style
from cancerstats.export.style import (
    AXIS,
    INK,
    INK_2,
    MUTED,
    REGION_COLORS,
    REGION_ORDER,
    SERIES,
    WIDTH,
    comma,
    es,
    save,
)

Results = dict[str, Any]
FigureFn = Callable[[Results, pd.DataFrame, pd.DataFrame, Path], None]


def _short(name: str, n: int = 34) -> str:
    text = label(name).replace("region[", "").replace("]", "")
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------------------- depuración
def fig_repairs(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Tres incidencias de depuración: edad en meses, hogar ÷ 100 e identidad educativa."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2))
    ax = axes[0]
    mf = (raw["MedianAgeMale"] + raw["MedianAgeFemale"]) / 2
    bad = raw["MedianAge"] > 100
    ax.scatter(mf[~bad], raw.loc[~bad, "MedianAge"], s=4, color=SERIES[0], alpha=0.35, lw=0)
    ax.scatter(
        mf[bad],
        raw.loc[bad, "MedianAge"] / 12,
        s=12,
        color=SERIES[1],
        lw=0,
        label="Corregidas (÷ 12)",
    )
    lim = [20, 66]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="-")
    ax.set(xlim=lim, ylim=lim, xlabel="Media de medianas H y M (años)", ylabel="Edad mediana")
    ax.set_title("Edad mediana en meses")
    ax.legend(loc="upper left", handletextpad=0.2)
    ax = axes[1]
    hh = raw["AvgHouseholdSize"]
    small = hh < 1
    bins = np.linspace(1.8, 4.0, 34)
    ax.hist(hh[~small], bins=bins, color=SERIES[0], alpha=0.85, label="Válidos")
    ymax = ax.get_ylim()[1]
    ax.vlines(hh[small] * 100, 0, ymax * 0.08, color=SERIES[1], lw=1.1, label="Corregidos (× 100)")
    ax.set(xlabel="Personas por hogar", ylabel="Condados")
    ax.set_title("Tamaño del hogar ÷ 100")
    ax.legend(loc="upper right", fontsize=6.5)
    ax = axes[2]
    edu = raw[["PctNoHS18_24", "PctHS18_24", "PctSomeCol18_24", "PctBachDeg18_24"]]
    complete = edu.notna().all(axis=1)
    total = edu[complete].sum(axis=1)
    ax.hist(total, bins=np.arange(99.85, 100.2, 0.05) - 0.025, color=SERIES[2], rwidth=0.8)
    ax.set(xlabel="Suma de las 4 categorías (%)", ylabel="Condados completos")
    ax.set_title("Identidad educativa 18–24")
    ax.set_xlim(99.75, 100.25)
    for a in axes:
        comma(a)
    fig.tight_layout(w_pad=1.2)
    save(fig, path)


def fig_missing(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Ausencia por estado: estructural (incidencia) frente a aleatoria (empleo, seguro)."""
    cols = ["incidenceRate", "PctEmployed16_Over", "PctPrivateCoverageAlone"]
    share = clean[cols].isna().groupby(clean["state"]).mean() * 100
    share = share.sort_values("incidenceRate", ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    x = np.arange(len(share))
    width = 0.27
    names = {
        "incidenceRate": "Incidencia (estructural)",
        "PctEmployed16_Over": "Empleo 16+",
        "PctPrivateCoverageAlone": "Solo seguro privado",
    }
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1) * width, share[c], width * 0.92, color=SERIES[i], label=names[c])
    ax.set_xticks(x, share.index, rotation=35, ha="right")
    ax.set_ylabel("Condados con el dato ausente (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    comma(ax, "y")
    ax.grid(axis="x", visible=False)
    save(fig, path)


# ----------------------------------------------------------------------------- exploración
def fig_response(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Distribución de la respuesta: histograma con KDE y normal, y Q-Q normal."""
    r = res["exploration"]["response"]
    h = r["histogram"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    edges = np.asarray(h["edges"])
    counts = np.asarray(h["counts"], dtype=float)
    dens = counts / (counts.sum() * np.diff(edges))
    ax.bar(
        edges[:-1],
        dens,
        width=np.diff(edges) * 0.92,
        align="edge",
        color=SERIES[0],
        alpha=0.55,
        label="Histograma",
    )
    ax.plot(h["grid"], h["kde"], color=SERIES[0], lw=1.8, label="Densidad núcleo")
    ax.plot(h["grid"], h["normal"], color=SERIES[1], lw=1.4, ls="--", label="Normal ajustada")
    m = r["mean_ci"]["estimate"]
    ax.axvline(m, color=INK_2, lw=0.8)
    ax.annotate(
        f"media {es(m, 1)}",
        (m, ax.get_ylim()[1] * 0.45),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        color=INK_2,
        fontsize=7,
    )
    ax.set(xlabel="Muertes por cáncer por 100 000 hab.", ylabel="Densidad")
    ax.set_title(f"Mortalidad por cáncer en {h['n']} condados")
    ax.legend(loc="upper right", fontsize=6.5)
    ax = axes[1]
    q = r["qq"]
    ax.scatter(q["theoretical"], q["sample"], s=5, color=SERIES[0], lw=0, alpha=0.7)
    t = np.array([min(q["theoretical"]), max(q["theoretical"])])
    ax.plot(t, q["mean"] + q["sd"] * t, color=SERIES[1], lw=1.2)
    ax.set(xlabel="Cuantiles teóricos N(0, 1)", ylabel="Cuantiles muestrales")
    ax.set_title("Gráfico Q-Q normal")
    for a in axes:
        comma(a)
    fig.tight_layout(w_pad=1.5)
    save(fig, path)


def fig_regions(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Mortalidad por región: diagramas de caja con la media y su IC 95 %."""
    reg = {g["group"]: g for g in res["exploration"]["region"]["groups"]}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    data = [clean.loc[clean["region"] == r, "TARGET_deathRate"].dropna() for r in REGION_ORDER]
    bp = ax.boxplot(
        data,
        orientation="horizontal",
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": INK, "lw": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": MUTED,
            "markeredgewidth": 0,
            "alpha": 0.6,
        },
        whiskerprops={"color": AXIS},
        capprops={"color": AXIS},
    )
    for patch, r in zip(bp["boxes"], REGION_ORDER, strict=True):
        patch.set_facecolor(REGION_COLORS[r])
        patch.set_alpha(0.35)
        patch.set_edgecolor(REGION_COLORS[r])
    for i, r in enumerate(REGION_ORDER, start=1):
        g = reg[r]
        ax.errorbar(
            g["mean"],
            i,
            xerr=[[g["mean"] - g["lower"]], [g["upper"] - g["mean"]]],
            fmt="D",
            color=INK,
            ms=3.5,
            capsize=2,
            lw=1,
        )
        ax.text(
            312, i, f"n = {g['n']} · media {es(g['mean'], 1)}", va="center", fontsize=7, color=INK_2
        )
    ax.set_yticks(range(1, 5), REGION_ORDER)
    ax.set_xlim(50, 370)
    ax.set_xlabel("Muertes por cáncer por 100 000 hab.")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    comma(ax, "x")
    save(fig, path)


def fig_correlations(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Correlación de Pearson (IC 95 %) y de Spearman de cada explicativa con la respuesta."""
    rows = sorted(res["exploration"]["correlations"], key=lambda r: r["pearson"])
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    y = np.arange(len(rows))
    r = np.array([x["pearson"] for x in rows])
    lo = np.array([x["pearson_lower"] for x in rows])
    hi = np.array([x["pearson_upper"] for x in rows])
    colors = [SERIES[0] if v < 0 else SERIES[1] for v in r]
    ax.barh(y, r, height=0.62, color=colors, alpha=0.85)
    ax.errorbar(r, y, xerr=[r - lo, hi - r], fmt="none", ecolor=INK, lw=0.8, capsize=1.5)
    ax.scatter(
        [x["spearman"] for x in rows], y, marker="|", s=40, color=INK, zorder=3, label="Spearman"
    )
    ax.set_yticks(y, [_short(x["variable"]) for x in rows])
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Coeficiente de correlación con la mortalidad")
    ax.set_xlim(-0.6, 0.6)
    ax.legend(
        handles=[
            Patch(color=SERIES[1], label="Pearson positiva"),
            Patch(color=SERIES[0], label="Pearson negativa"),
            Line2D([], [], marker="|", ls="", color=INK, label="Spearman"),
        ],
        loc="lower right",
    )
    ax.grid(axis="y", visible=False)
    comma(ax, "x")
    save(fig, path)


def fig_corr_matrix(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Matriz de correlaciones de Pearson entre la respuesta y las candidatas."""
    cm = res["exploration"]["corr_matrix"]
    vals = np.asarray(cm["values"], dtype=float)
    labels = [_short(v, 26) for v in cm["variables"]]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.6))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "div", [style.DIVERGING[0], style.DIVERGING[1], style.DIVERGING[2]]
    )
    im = ax.imshow(vals, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and abs(vals[i, j]) >= 0.8:
                ax.text(
                    j, i, es(vals[i, j], 1), ha="center", va="center", fontsize=4.5, color="white"
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    comma(cb.ax, "y")
    save(fig, path)


# ----------------------------------------------------------------------------- colinealidad
def fig_vif(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """FIV antes y después de la poda (escala logarítmica, umbral del curso)."""
    coll = res["effects"]["collinearity"]
    before = {r["variable"]: r["fiv"] for r in coll["vif_before"]}
    after = {r["variable"]: r["fiv"] for r in coll["vif_after"]}
    order = sorted(before, key=lambda v: before[v])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    y = np.arange(len(order))
    ax.barh(
        y + 0.2, [before[v] for v in order], height=0.38, color=SERIES[1], label="Modelo máximo"
    )
    ax.barh(
        y - 0.2,
        [after.get(v, np.nan) for v in order],
        height=0.38,
        color=SERIES[0],
        label="Tras la poda",
    )
    ax.set_xscale("log")
    thr = 10.0
    ax.axvline(thr, color=style.CRITICAL, lw=1)
    ax.text(thr * 1.05, 0.2, "FIV = 10", color=style.CRITICAL, fontsize=7)
    ax.set_yticks(y, [_short(v) for v in order])
    ax.set_xlabel("Factor de inflación de la varianza (escala log)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: es(x, 0) if x >= 1 else es(x, 1)))
    save(fig, path)


# ----------------------------------------------------------------------------- efectos
def fig_variance(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Varianza residual por decil de población: el patrón a + b/n y qué hace la ponderación.

    El modelo final se estima por MCO con errores típicos robustos; la ponderación por la
    función de varianza (MCPF) es una especificación de sensibilidad. El panel derecho
    muestra por qué la inferencia robusta es necesaria (la varianza de los residuos MCO cae
    con la población) y que la ponderación la estabiliza.
    """
    eff = res["effects"]
    before = eff["initial"]["diagnostics"]
    edges = np.asarray(before["population_decile_edges"])
    mids = np.sqrt(edges[:-1] * edges[1:])
    vf = eff["variance_function_final"]
    by_dec = eff.get("variance_by_decile") or {
        "mco": eff["diagnostics"]["homoscedasticity"]["var_by_population_decile"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
    ax = axes[0]
    ax.plot(
        mids,
        before["var_by_population_decile"],
        "o-",
        color=SERIES[1],
        ms=4,
        label="Residuos MCO (e²)",
    )
    grid = np.geomspace(mids[0], mids[-1], 200)
    if vf:
        ax.plot(
            grid,
            vf["a"] + vf["b"] / grid,
            color=INK,
            lw=1.2,
            label=f"a + b/n  (a = {es(vf['a'], 0)}, b = {es(vf['b'] / 1e6, 2)}·10⁶)",
        )
    ax.set_xscale("log")
    ax.set(xlabel="Población del condado (escala log)", ylabel="Varianza residual media")
    ax.set_title("Modelo inicial: la varianza cae con la población")
    ax.legend(loc="upper right")
    ax = axes[1]
    mco = by_dec["mco"]
    ax.plot(mids[: len(mco)], mco, "o-", color=SERIES[1], ms=4, label="MCO, final (e²)")
    if "mcpf" in by_dec:
        mcpf = by_dec["mcpf"]
        ax.plot(mids[: len(mcpf)], mcpf, "s-", color=SERIES[0], ms=4, label="MCPF (w·e²)")
    ax.set_xscale("log")
    ax.set(xlabel="Población del condado (escala log)", ylabel="Varianza residual media")
    ax.set_title("Modelo final: MCO y ponderación (sensibilidad)")
    ax.legend(loc="upper right")
    for a in axes:
        comma(a, "y")
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x:,.0f}".replace(",", " ")))
    fig.tight_layout(w_pad=1.5)
    save(fig, path)


def _coef_rows(res: Results) -> list[dict[str, Any]]:
    return [c for c in res["effects"]["final"]["coef"] if c["term"] != "const"]


def fig_coefficients(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Efecto de pasar del primer al tercer cuartil de cada explicativa (IC cluster)."""
    effects = res["effects"]["effects"]
    rows = []
    for e in effects:
        if e.get("interaccion_region"):
            for s in e["by_region"]:
                rows.append(
                    (
                        f"{_short(e['variable'], 30)} · {s['region']}",
                        s["iqr_effect"],
                        s["iqr_lower"],
                        s["iqr_upper"],
                        s["p"],
                        REGION_COLORS[s["region"]],
                    )
                )
        else:
            rows.append(
                (
                    _short(e["variable"], 30),
                    e["iqr_effect"],
                    e["iqr_lower"],
                    e["iqr_upper"],
                    e["p"],
                    SERIES[0],
                )
            )
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(WIDTH, 0.2 * len(rows) + 0.8))
    y = np.arange(len(rows))
    for i, (_, est, lo, hi, p, col) in enumerate(rows):
        filled = p < 0.05
        ax.plot([lo, hi], [i, i], color=col, lw=1.6)
        ax.scatter(est, i, s=26, color=col if filled else "white", edgecolor=col, lw=1.2, zorder=3)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xlabel("Cambio en la mortalidad (muertes/100 000) al pasar de Q1 a Q3")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=INK, label="p < 0,05"),
            Line2D([], [], marker="o", ls="", mfc="white", color=INK, label="no significativo"),
        ],
        loc="lower right",
    )
    comma(ax, "x")
    save(fig, path)


def fig_interactions(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Pendientes específicas de cada región para las variables que interaccionan."""
    effects = [e for e in res["effects"]["effects"] if e.get("interaccion_region")]
    num = res["effects"].get("numeric_interactions") or []
    n = len(effects) + (1 if num else 0)
    if n == 0:
        fig, ax = plt.subplots(figsize=(WIDTH, 1))
        ax.axis("off")
        ax.text(0.5, 0.5, "El modelo final no contiene interacciones.", ha="center")
        save(fig, path)
        return
    fig, axes = plt.subplots(1, n, figsize=(WIDTH, 2.3), squeeze=False)
    for ax, e in zip(axes[0], effects, strict=False):
        for i, s in enumerate(
            sorted(e["by_region"], key=lambda s: REGION_ORDER.index(s["region"]))
        ):
            col = REGION_COLORS[s["region"]]
            ax.plot([s["lower"], s["upper"]], [i, i], color=col, lw=2)
            ax.scatter(
                s["coef"],
                i,
                s=30,
                color=col if s["p"] < 0.05 else "white",
                edgecolor=col,
                lw=1.2,
                zorder=3,
            )
        ax.axvline(0, color=AXIS, lw=0.9)
        ax.set_yticks(range(4), REGION_ORDER, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(_short(e["variable"], 26), fontsize=8)
        ax.set_xlabel("Pendiente (IC 95 %)", fontsize=7)
        ax.grid(axis="y", visible=False)
        comma(ax, "x")
    if num:
        ax = axes[0][-1]
        it = num[0]
        xs = [s["valor_b"] for s in it["slopes"]]
        ys = [s["pendiente"] for s in it["slopes"]]
        lo = [s["lower"] for s in it["slopes"]]
        hi = [s["upper"] for s in it["slopes"]]
        ax.fill_between(xs, lo, hi, color=SERIES[0], alpha=0.15, lw=0)
        ax.plot(xs, ys, "o-", color=SERIES[0], ms=4)
        ax.set_title(f"{_short(it['a'], 22)}\nsegún {_short(it['b'], 22).lower()}", fontsize=8)
        ax.set_xlabel(
            "% con grado (Q1, Q2, Q3)"
            if it["b"] == "PctBachDeg25_Over"
            else f"{_short(it['b'], 18)} (Q1–Q3)",
            fontsize=7,
        )
        ax.set_ylabel("Pendiente de la incidencia", fontsize=7)
        comma(ax)
    fig.tight_layout(w_pad=0.8)
    save(fig, path)


def fig_adjusted_regions(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Medias regionales brutas frente a ajustadas por las demás variables."""
    rows = {r["region"]: r for r in res["effects"]["adjusted_region_means"]}
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    for i, reg in enumerate(REGION_ORDER):
        r = rows[reg]
        col = REGION_COLORS[reg]
        ax.plot([r["bruta"], r["ajustada"]], [i, i], color=AXIS, lw=1.2, zorder=1)
        ax.scatter(r["bruta"], i, s=30, color="white", edgecolor=col, lw=1.4, zorder=3)
        ax.errorbar(
            r["ajustada"],
            i,
            xerr=[[r["ajustada"] - r["lower"]], [r["upper"] - r["ajustada"]]],
            fmt="o",
            color=col,
            ms=5.5,
            capsize=2,
            lw=1.2,
            zorder=3,
        )
    ax.set_yticks(range(4), REGION_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Mortalidad media (muertes/100 000)")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", mfc="white", color=INK, label="Bruta"),
            Line2D([], [], marker="o", ls="", color=INK, label="Ajustada (IC 95 %)"),
        ],
        loc="lower right",
        ncol=2,
    )
    ax.grid(axis="y", visible=False)
    comma(ax, "x")
    save(fig, path)


def fig_diagnostics(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Los cuatro gráficos clásicos del diagnóstico del modelo final."""
    t = pd.DataFrame(res["effects"]["residuals"])
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6))
    sc = {"s": 5, "lw": 0, "alpha": 0.45, "color": SERIES[0]}
    ax = axes[0, 0]
    ax.scatter(t["fitted"], t["std_resid"], **sc)
    ax.axhline(0, color=INK_2, lw=0.8)
    for v in (-3, 3):
        ax.axhline(v, color=style.CRITICAL, lw=0.6, ls="--")
    ax.set(xlabel="Valores ajustados", ylabel="Residuo tipificado")
    ax.set_title("Residuos frente a ajustados")
    ax = axes[0, 1]
    q = res["effects"]["diagnostics"]["normality"]["qq"]
    ax.scatter(q["theoretical"], q["sample"], **sc)
    lim = [min(q["theoretical"]), max(q["theoretical"])]
    ax.plot(lim, lim, color=SERIES[1], lw=1.2)
    ax.set(xlabel="Cuantiles teóricos N(0, 1)", ylabel="Residuo tipificado")
    ax.set_title("Q-Q normal de los residuos")
    ax = axes[1, 0]
    ax.scatter(t["fitted"], np.sqrt(np.abs(t["std_resid"])), **sc)
    ax.set(xlabel="Valores ajustados", ylabel="√|residuo tipificado|")
    ax.set_title("Escala-localización")
    ax = axes[1, 1]
    ax.scatter(
        t["leverage"],
        t["stud_resid"],
        s=np.clip(t["cook"] * 900, 3, 120),
        lw=0,
        alpha=0.5,
        color=SERIES[0],
    )
    thr = res["effects"]["diagnostics"]["influence"]["thresholds"]
    ax.axvline(thr["leverage"], color=MUTED, lw=0.8, ls="--")
    ax.set(xlabel="Apalancamiento $h_{ii}$", ylabel="Residuo estudentizado")
    ax.set_xscale("log")
    ax.set_title("Influencia (área según la distancia de Cook)")
    top = t.sort_values("cook", ascending=False).head(3)
    for _, r in top.iterrows():
        ax.annotate(
            str(r["county_id"]).split(",")[0],
            (r["leverage"], r["stud_resid"]),
            fontsize=6,
            color=INK_2,
            xytext=(3, 3),
            textcoords="offset points",
        )
    for a in axes.ravel():
        comma(a, "y")
    for a in (axes[0, 0], axes[1, 0], axes[0, 1]):
        comma(a, "x")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, path)


def fig_partial(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Gráficos de componente más residuo con suavizado LOWESS (linealidad)."""
    plots = res["effects"]["diagnostics"]["linearity"]["partial_residuals"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.9))
    for ax, p in zip(axes.ravel(), plots, strict=False):
        x = np.asarray(p["x"])
        y = np.asarray(p["y"])
        ax.scatter(x, y, s=3, lw=0, alpha=0.3, color=SERIES[0])
        lo_x, hi_x = np.quantile(x, [0.005, 0.995])
        xs = np.linspace(lo_x, hi_x, 50)
        center = np.mean(x)
        ax.set_xlim(lo_x - 0.05 * (hi_x - lo_x), hi_x + 0.05 * (hi_x - lo_x))
        ax.plot(xs, p["slope"] * (xs - center), color=SERIES[1], lw=1.3)
        ax.plot(p["smooth_x"], p["smooth_y"], color=INK, lw=1.2)
        ax.set_title(_short(p["variable"], 30), fontsize=8)
        comma(ax)
    for ax in axes.ravel()[len(plots) :]:
        ax.axis("off")
    fig.legend(
        handles=[
            Line2D([], [], color=SERIES[1], label="Recta del modelo"),
            Line2D([], [], color=INK, label="LOWESS"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(h_pad=1.0, w_pad=0.8, rect=(0, 0.04, 1, 1))
    save(fig, path)


def fig_sensitivity(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Estabilidad de los coeficientes clave entre especificaciones alternativas."""
    specs = res["effects"]["sensitivity"]
    final = [
        c["term"]
        for c in _coef_rows(res)
        if ":" not in c["term"] and not c["term"].startswith("region")
    ]
    keys = sorted(
        final, key=lambda k: -abs(next(c["t"] for c in _coef_rows(res) if c["term"] == k))
    )[:6]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.2), sharey=True)
    ids = [s["id"] for s in specs]
    labels_short = {
        "ols": "MCO",
        "ols_hc3": "MCO + HC3",
        "wls": "MCPF",
        "main": "Principal",
        "state_fe": "EF de estado",
        "mixed": "Mixto",
        "huber": "Huber",
        "no_influential": "Sin influyentes",
        "mi": "Imput. múltiple",
    }
    boot = res["effects"].get("bootstrap")
    for ax, k in zip(axes.ravel(), keys, strict=False):
        for i, s in enumerate(specs):
            if k not in s["coef"]:
                continue
            col = SERIES[1] if s["id"] == "main" else SERIES[0]
            ax.plot([s["lower"][k], s["upper"][k]], [i, i], color=col, lw=1.6)
            ax.scatter(s["coef"][k], i, s=18, color=col, zorder=3)
        if boot and k in boot["lower"]:
            j = len(specs)
            ax.plot([boot["lower"][k], boot["upper"][k]], [j, j], color=SERIES[2], lw=1.6)
            ax.scatter(boot["coef"][k], j, s=18, color=SERIES[2], zorder=3)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(_short(k, 28), fontsize=8)
        ax.grid(axis="y", visible=False)
        comma(ax, "x")
    names = [labels_short.get(i, i) for i in ids] + (["Bootstrap cluster"] if boot else [])
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(names)), names, fontsize=6.5)
    axes[0, 0].invert_yaxis()
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    save(fig, path)


def fig_incidence(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Cambio relativo de cada coeficiente al retirar la incidencia (criterio de confusión)."""
    inc = res["effects"].get("incidence_adjustment")
    if not inc:
        return
    rows = [r for r in inc["comparison"] if r["cambio_pct"] is not None]
    rows.sort(key=lambda r: r["cambio_pct"])
    fig, ax = plt.subplots(figsize=(WIDTH, 0.24 * len(rows) + 0.8))
    y = np.arange(len(rows))
    vals = np.array([r["cambio_pct"] for r in rows])
    colors = [SERIES[1] if abs(v) > 10 else AXIS for v in vals]
    ax.barh(y, vals, height=0.6, color=colors)
    for i, v in enumerate(vals):
        ax.text(
            v + (3 if v >= 0 else -3),
            i,
            f"{es(v, 0)} %",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=6.5,
            color=INK_2,
        )
    for v in (-10, 10):
        ax.axvline(v, color=MUTED, lw=0.8, ls="--")
    ax.axvline(0, color=AXIS, lw=0.9)
    names = [
        ("Región: " + r["term"][7:-1]) if r["term"].startswith("region[") else _short(r["term"], 34)
        for r in rows
    ]
    ax.set_yticks(y, names)
    lim = float(max(np.abs(np.asarray(vals, dtype=float)).max() * 1.25, 20))
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Cambio del coeficiente al retirar la incidencia (%)")
    ax.grid(axis="y", visible=False)
    comma(ax, "x")
    save(fig, path)


# ----------------------------------------------------------------------------- predicción
def fig_cv(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """RMSE de validación cruzada por modelo: aleatoria frente a agrupada por estado."""
    pred = res["predictive"]
    rows = [r for r in pred["cv"] if r["model"] != "baseline"]
    group = {r["model"]: r for r in pred["group_cv"]}
    rows.sort(key=lambda r: r["rmse_mean"])
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    y = np.arange(len(rows))
    ax.errorbar(
        [r["rmse_mean"] for r in rows],
        y + 0.15,
        xerr=[r["rmse_sd"] / np.sqrt(len(r["folds"])) for r in rows],
        fmt="o",
        color=SERIES[0],
        ms=5,
        capsize=2,
        label="Aleatoria (10 pliegues)",
    )
    if group:
        ax.errorbar(
            [group[r["model"]]["rmse_mean"] for r in rows],
            y - 0.15,
            xerr=[
                group[r["model"]]["rmse_sd"] / np.sqrt(len(group[r["model"]]["folds"]))
                for r in rows
            ],
            fmt="s",
            color=SERIES[1],
            ms=4.5,
            capsize=2,
            label="Agrupada por estado",
        )
    ax.set_yticks(
        y,
        [
            r["label"].replace(
                " con la especificación del modelo de efectos", " (modelo de efectos)"
            )
            for r in rows
        ],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("RMSE fuera de pliegue (± error típico)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    comma(ax, "x")
    save(fig, path)


def fig_test(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Observado frente a predicho en prueba y cobertura conformal por tamaño de condado."""
    pred = res["predictive"]
    tp = pred["test_predictions"]
    conf = pred["conformal"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for reg in REGION_ORDER:
        m = np.asarray(tp["region"]) == reg
        ax.scatter(
            np.asarray(tp["pred"])[m],
            np.asarray(tp["y"])[m],
            s=8,
            lw=0,
            alpha=0.7,
            color=REGION_COLORS[reg],
            label=reg,
        )
    lim = [80, 300]
    ax.plot(lim, lim, color=INK_2, lw=0.8)
    ax.set(xlim=lim, ylim=[50, 370], xlabel="Predicción", ylabel="Mortalidad observada")
    ax.set_title(f"Prueba: {pred['chosen_label']}")
    ax.legend(loc="upper left", markerscale=1.5, handletextpad=0.1)
    comma(ax)
    ax = axes[1]
    by = conf["by_population"]
    x = np.arange(len(by))
    ax.bar(
        x - 0.18,
        [b["cobertura_estandar"] * 100 for b in by],
        0.34,
        color=SERIES[0],
        label="Conformal estándar",
    )
    ax.bar(
        x + 0.18,
        [b["cobertura_normalizada"] * 100 for b in by],
        0.34,
        color=SERIES[2],
        label="Normalizado por σ(n)",
    )
    ax.axhline(conf["nominal"] * 100, color=INK, lw=1, ls="--")
    ax.set_xticks(x, [b["tercil"].capitalize() for b in by])
    ax.set_ylim(70, 100)
    ax.set_ylabel("Cobertura en prueba (%)")
    ax.set_title(f"Intervalos de predicción al {es(100 * conf['nominal'], 0)} %")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.5)
    ax.grid(axis="x", visible=False)
    comma(ax, "y")
    fig.tight_layout(w_pad=1.5)
    save(fig, path)


def fig_importance(res: Results, clean: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Importancia por permutación del modelo predictivo elegido."""
    imp = res["predictive"]["importance"][:12][::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    y = np.arange(len(imp))
    ax.barh(
        y,
        [r["importancia"] for r in imp],
        xerr=[r["dt"] for r in imp],
        height=0.6,
        color=SERIES[0],
        error_kw={"ecolor": INK_2, "lw": 0.8, "capsize": 1.5},
    )
    ax.set_yticks(y, [_short(r["variable"]) for r in imp])
    ax.set_xlabel("Aumento del RMSE al permutar la variable")
    ax.grid(axis="y", visible=False)
    comma(ax, "x")
    save(fig, path)


FIGURES: dict[str, FigureFn] = {
    "depuracion": fig_repairs,
    "ausentes": fig_missing,
    "respuesta": fig_response,
    "regiones": fig_regions,
    "correlaciones": fig_correlations,
    "matriz-correlaciones": fig_corr_matrix,
    "fiv": fig_vif,
    "varianza": fig_variance,
    "coeficientes": fig_coefficients,
    "interacciones": fig_interactions,
    "regiones-ajustadas": fig_adjusted_regions,
    "diagnostico": fig_diagnostics,
    "residuos-parciales": fig_partial,
    "sensibilidad": fig_sensitivity,
    "incidencia": fig_incidence,
    "validacion-cruzada": fig_cv,
    "prueba": fig_test,
    "importancia": fig_importance,
}


def render_all(results: Results, clean: pd.DataFrame, raw: pd.DataFrame, out: Path) -> list[str]:
    """Genera todas las figuras; devuelve las que se produjeron."""
    style.setup()
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in FIGURES.items():
        if name in ("validacion-cruzada", "prueba", "importancia") and not results.get(
            "predictive"
        ):
            continue
        fn(results, clean, raw, out / f"{name}.pdf")
        if (out / f"{name}.pdf").exists():
            made.append(name)
    return made
