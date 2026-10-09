"""Registro persistente de corridas.

Cada ejecución del pipeline se guarda en ``runs/<id>/`` con su configuración, su
manifiesto de reproducibilidad (huellas de datos y configuración, versiones, commit de git,
tiempos), sus resultados y su registro. El conjunto depurado se guarda una sola vez por
huella en ``runs/_datasets/`` porque muchas corridas comparten la misma depuración.

``runs/index.json`` resume todas las corridas y ``runs/LATEST`` apunta a la última, que es
la que alimentan el informe, la interfaz publicada y la presentación.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from cancerstats import jsonutil
from cancerstats.config import AnalysisConfig
from cancerstats.paths import project_root, runs_dir

PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "pydantic")


def new_run_id(config: AnalysisConfig, now: datetime | None = None) -> str:
    """Identificador ordenable: fecha y hora UTC más la huella de la configuración."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{config.fingerprint()[:8]}"


def git_state(root: Path) -> dict[str, Any]:
    """Commit actual y si el árbol de trabajo tiene cambios sin confirmar."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src", "config"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def environment() -> dict[str, str]:
    """Versiones de Python y de las bibliotecas que determinan los resultados."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except Exception:  # pragma: no cover - paquete opcional ausente
            out[pkg] = "no instalado"
    return out


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Huella del conjunto depurado (contenido, no representación)."""
    payload = df.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class RunPaths:
    """Rutas de los ficheros de una corrida."""

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def results(self) -> Path:
        return self.root / "results.json"

    @property
    def log(self) -> Path:
        return self.root / "log.txt"

    @property
    def tables(self) -> Path:
        return self.root / "tablas"


class RunRegistry:
    """Acceso al directorio de corridas."""

    def __init__(self, base: Path | None = None) -> None:
        self.base = base or runs_dir()
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "_datasets").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------- escritura
    def save(
        self,
        run_id: str,
        config: AnalysisConfig,
        manifest: dict[str, Any],
        results: dict[str, Any],
        dataset: pd.DataFrame,
        log_lines: list[str],
        tables: dict[str, pd.DataFrame],
    ) -> Path:
        """Guarda una corrida completa de forma atómica (directorio temporal + renombrado)."""
        tmp = self.base / f".tmp-{run_id}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        paths = RunPaths(tmp)
        config.save(paths.config)
        ds_id = dataset_fingerprint(dataset)
        ds_path = self.base / "_datasets" / f"{ds_id}.json"
        if not ds_path.exists():
            jsonutil.dump(
                {
                    "columns": list(dataset.columns),
                    "data": {c: dataset[c].tolist() for c in dataset.columns},
                },
                ds_path,
            )
        manifest = {**manifest, "dataset": ds_id}
        results = {**results, "dataset": ds_id}
        jsonutil.dump(manifest, paths.manifest, indent=2)
        jsonutil.dump(results, paths.results)
        paths.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        paths.tables.mkdir()
        for name, table in tables.items():
            table.to_csv(paths.tables / f"{name}.csv", index=False, float_format="%.6g")
        final = self.base / run_id
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        self._update_index(run_id, manifest, results)
        (self.base / "LATEST").write_text(run_id + "\n", encoding="utf-8")
        return final

    def _update_index(self, run_id: str, manifest: dict[str, Any], results: dict[str, Any]) -> None:
        index = self.index()
        index = [r for r in index if r["id"] != run_id]
        index.append({"id": run_id, **_summary(manifest, results)})
        index.sort(key=lambda r: r["id"])
        jsonutil.dump(index, self.base / "index.json", indent=2)

    # -------------------------------------------------------------------------- lectura
    def index(self) -> list[dict[str, Any]]:
        """Resumen de todas las corridas, de la más antigua a la más reciente."""
        path = self.base / "index.json"
        return list(jsonutil.load(path)) if path.exists() else []

    def latest(self) -> str | None:
        """Identificador de la última corrida."""
        path = self.base / "LATEST"
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    def paths(self, run_id: str) -> RunPaths:
        """Rutas de una corrida existente."""
        root = self.base / run_id
        if not root.is_dir():
            raise FileNotFoundError(f"No existe la corrida {run_id}")
        return RunPaths(root)

    def results(self, run_id: str) -> dict[str, Any]:
        """Resultados de una corrida."""
        return dict(jsonutil.load(self.paths(run_id).results))

    def manifest(self, run_id: str) -> dict[str, Any]:
        """Manifiesto de una corrida."""
        return dict(jsonutil.load(self.paths(run_id).manifest))

    def config(self, run_id: str) -> AnalysisConfig:
        """Configuración de una corrida."""
        return AnalysisConfig.load(self.paths(run_id).config)

    def dataset(self, dataset_id: str) -> pd.DataFrame:
        """Conjunto depurado por su huella."""
        raw = jsonutil.load(self.base / "_datasets" / f"{dataset_id}.json")
        return pd.DataFrame(raw["data"], columns=raw["columns"])

    def rebuild_index(self) -> list[dict[str, Any]]:
        """Reconstruye ``index.json`` a partir de los directorios (útil tras borrar corridas)."""
        rows = []
        for d in sorted(
            p for p in self.base.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
        ):
            try:
                rows.append(
                    {
                        "id": d.name,
                        **_summary(
                            jsonutil.load(d / "manifest.json"), jsonutil.load(d / "results.json")
                        ),
                    }
                )
            except (FileNotFoundError, KeyError):
                continue
        jsonutil.dump(rows, self.base / "index.json", indent=2)
        return rows


def _summary(manifest: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    eff = results.get("effects", {}).get("final", {})
    pred = results.get("predictive") or {}
    test = {r["model"]: r for r in pred.get("test", [])}
    chosen = pred.get("chosen")
    return {
        "created_at": manifest.get("created_at"),
        "name": manifest.get("config_name"),
        "description": manifest.get("config_description"),
        "fingerprint": manifest.get("config_fingerprint"),
        "data_sha256": manifest.get("data_sha256"),
        "git_commit": manifest.get("git", {}).get("commit"),
        "duration_s": manifest.get("duration_s"),
        "n_effects": eff.get("n"),
        "terms": eff.get("terms"),
        "r2_adj": eff.get("summary", {}).get("r2_adj"),
        "predictive_model": chosen,
        "test_rmse": test.get(chosen, {}).get("rmse") if chosen else None,
        "test_r2": test.get(chosen, {}).get("r2") if chosen else None,
    }


def created_now() -> str:
    """Marca temporal ISO 8601 en UTC."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    """Ruta relativa a la raíz del proyecto (para los manifiestos)."""
    try:
        return str(path.resolve().relative_to(project_root()))
    except ValueError:
        return str(path)
