"""Rutas canónicas del repositorio.

Todas las rutas se resuelven a partir de la raíz del proyecto, que se localiza buscando
``pyproject.toml`` hacia arriba desde este fichero. Puede forzarse con la variable de
entorno ``CANCERSTATS_ROOT`` (útil en pruebas que trabajan en un directorio temporal).
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Devuelve la raíz del repositorio."""
    env = os.environ.get("CANCERSTATS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "cancerstats").exists():
            return parent
    return Path.cwd().resolve()


def runs_dir() -> Path:
    """Directorio donde se persisten todas las corridas."""
    return project_root() / "runs"


def report_dir() -> Path:
    """Directorio del informe LaTeX."""
    return project_root() / "report"


def report_generated_dir() -> Path:
    """Directorio de resultados generados que consume el informe (siempre la última corrida)."""
    return report_dir() / "generado"


def config_dir() -> Path:
    """Directorio de configuraciones de la metodología."""
    return project_root() / "config"


def default_config_path() -> Path:
    """Configuración por defecto de la metodología."""
    return config_dir() / "default.yaml"


def raw_data_path() -> Path:
    """Fichero de datos original, tal como se recibió."""
    return project_root() / "data" / "raw" / "CANCER.csv"
