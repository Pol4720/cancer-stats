"""Datos de la interfaz en modo estático (GitHub Pages).

Escribe en ``web/public/data/`` los mismos documentos que sirve la API en vivo, para la
última corrida (u otra indicada): resultados, conjunto depurado, índice de corridas,
configuración (por defecto, esquema y la de la corrida) y la sintaxis SPSS. La interfaz
compilada los lee con rutas relativas, de modo que funciona bajo cualquier subruta.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from cancerstats import jsonutil
from cancerstats.config import AnalysisConfig
from cancerstats.export.spss import syntax
from cancerstats.paths import project_root
from cancerstats.runs import RunRegistry


def web_data_dir() -> Path:
    """Directorio de datos de la interfaz estática."""
    return project_root() / "web" / "public" / "data"


def export_web(
    run_id: str | None = None, registry: RunRegistry | None = None, out: Path | None = None
) -> str:
    """Copia la corrida indicada (por defecto, la última) al directorio de la interfaz.

    Returns:
        El identificador de la corrida exportada.
    """
    reg = registry or RunRegistry()
    rid = run_id or reg.latest()
    if rid is None:
        raise FileNotFoundError("No hay corridas: ejecute antes «cancerstats run».")
    target = out or web_data_dir()
    if target.exists():
        shutil.rmtree(target)
    run_dir = target / "runs" / rid
    run_dir.mkdir(parents=True)
    paths = reg.paths(rid)
    results = reg.results(rid)
    for name in ("results.json", "manifest.json", "config.yaml", "log.txt"):
        shutil.copyfile(paths.root / name, run_dir / name)
    shutil.copyfile(reg.base / "_datasets" / f"{results['dataset']}.json", run_dir / "dataset.json")
    (run_dir / "modelo_final.sps").write_text(syntax(results, "practica.sav"), encoding="utf-8")
    jsonutil.dump(reg.config(rid).model_dump(mode="json"), run_dir / "config.json")
    tables = run_dir / "tables"
    tables.mkdir()
    for csv in paths.tables.glob("*.csv"):
        shutil.copyfile(csv, tables / csv.name)
    # Sólo la corrida exportada viaja con el sitio; el índice la marca como la última.
    index = [r for r in reg.index() if r["id"] == rid]
    jsonutil.dump({"latest": rid, "runs": index, "mode": "static"}, target / "runs.json")
    jsonutil.dump(AnalysisConfig().model_dump(mode="json"), target / "config-default.json")
    (target / "config-schema.json").write_text(
        json.dumps(AnalysisConfig.model_json_schema(), ensure_ascii=False), encoding="utf-8"
    )
    return rid
