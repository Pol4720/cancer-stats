"""Exportación de una corrida al material del informe (macros, tablas y figuras)."""

from __future__ import annotations

from pathlib import Path

from cancerstats import jsonutil
from cancerstats.config import AnalysisConfig
from cancerstats.export import figures, latex, spss
from cancerstats.io import read_raw
from cancerstats.paths import project_root, report_generated_dir
from cancerstats.runs import RunRegistry


def export_run(
    run_id: str | None = None,
    registry: RunRegistry | None = None,
    out: Path | None = None,
    with_figures: bool = True,
) -> str:
    """Regenera ``report/generado`` a partir de una corrida (por defecto, la última).

    Returns:
        El identificador de la corrida exportada.
    """
    reg = registry or RunRegistry()
    rid = run_id or reg.latest()
    if rid is None:
        raise FileNotFoundError("No hay corridas: ejecute antes «cancerstats run».")
    results = reg.results(rid)
    config: AnalysisConfig = reg.config(rid)
    target = out or report_generated_dir()
    (target / "tablas").mkdir(parents=True, exist_ok=True)
    (target / "figuras").mkdir(parents=True, exist_ok=True)

    n_macros = latex.write_macros(results, target / "resultados.tex")
    tables = latex.write_tables(results, target / "tablas")
    spss_dir = project_root() / "spss" if out is None else target / "spss"
    spss_dir.mkdir(parents=True, exist_ok=True)
    (spss_dir / "modelo_final.sps").write_text(
        spss.syntax(results, "../data/raw/practica.sav"), encoding="utf-8"
    )
    made: list[str] = []
    if with_figures:
        clean = reg.dataset(results["dataset"])
        data_path = Path(config.data.path)
        if not data_path.is_absolute():
            data_path = project_root() / data_path
        raw, _ = read_raw(data_path, config.data.encoding)
        made = figures.render_all(results, clean, raw, target / "figuras")
    jsonutil.dump(
        {"run_id": rid, "macros": n_macros, "tables": tables, "figures": made},
        target / "manifiesto.json",
        indent=2,
    )
    return rid


__all__ = ["export_run"]
