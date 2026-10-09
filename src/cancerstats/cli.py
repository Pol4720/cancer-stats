"""Interfaz de línea de órdenes: ``cancerstats <orden>`` (o ``uv run cancerstats``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from cancerstats.config import AnalysisConfig, merge_overrides, parse_dotted_override
from cancerstats.paths import default_config_path, project_root

app = typer.Typer(
    help="Análisis estadístico reproducible de la mortalidad por cáncer en EE. UU.",
    no_args_is_help=True,
    add_completion=False,
)
runs_app = typer.Typer(help="Consulta las corridas persistidas.", no_args_is_help=True)
config_app = typer.Typer(help="Configuración de la metodología.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")
app.add_typer(config_app, name="config")
console = Console()


def load_config(path: Path | None, overrides: list[str] | None) -> AnalysisConfig:
    """Carga la configuración y aplica las sobrescrituras ``clave.anidada=valor``."""
    cfg_path = path or default_config_path()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    for expr in overrides or []:
        data = merge_overrides(data or {}, parse_dotted_override(expr))
    return AnalysisConfig.from_mapping(data)


@app.command()
def run(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Fichero YAML.")] = None,
    set_: Annotated[
        list[str] | None, typer.Option("--set", "-s", help="Sobrescribe una clave: a.b=valor.")
    ] = None,
    export: Annotated[bool, typer.Option(help="Regenera el material del informe y la web.")] = True,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Ejecuta el pipeline completo y persiste la corrida en runs/."""
    from cancerstats.pipeline import Event
    from cancerstats.pipeline import run as run_pipeline

    cfg = load_config(config, set_)

    def show(ev: Event) -> None:
        if not quiet:
            mark = {"inicio": "▶", "fin": "✔", "info": "·"}.get(ev.status, " ")
            console.print(f"[dim]{int(ev.progress * 100):3d}%[/] {mark} {ev.message}")

    run_id, _ = run_pipeline(cfg, listener=show)
    console.print(f"[bold green]Corrida guardada:[/] runs/{run_id}")
    if export:
        from cancerstats.export import export_run

        export_run(run_id)
        console.print("[bold green]Material del informe y de la web regenerado.[/]")


@app.command("export")
def export_cmd(
    run_id: Annotated[
        str | None, typer.Option("--run", help="Corrida (por defecto, la última).")
    ] = None,
) -> None:
    """Regenera report/generado (macros, tablas y figuras) a partir de una corrida."""
    from cancerstats.export import export_run

    rid = export_run(run_id)
    console.print(f"Exportada la corrida {rid}")


@app.command("web-data")
def web_data(
    run_id: Annotated[
        str | None, typer.Option("--run", help="Corrida (por defecto, la última).")
    ] = None,
) -> None:
    """Prepara web/public/data para compilar la interfaz en modo estático."""
    from cancerstats.export.web import export_web

    rid = export_web(run_id)
    console.print(f"Datos de la interfaz preparados con la corrida {rid}")


@app.command()
def presentation(
    run_id: Annotated[
        str | None, typer.Option("--run", help="Corrida (por defecto, la última).")
    ] = None,
) -> None:
    """Genera presentation/index.html (presentación autónoma) con las cifras de una corrida."""
    from cancerstats.export.presentation import build_presentation

    console.print(f"Presentación escrita en {build_presentation(run_id)}")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    export: Annotated[
        bool, typer.Option(help="Regenerar report/generado al terminar cada corrida.")
    ] = True,
) -> None:
    """Arranca la interfaz interactiva en modo en vivo (configurar y ejecutar corridas)."""
    import uvicorn

    from cancerstats.api.server import create_app

    console.print(f"Interfaz en http://{host}:{port}")
    uvicorn.run(create_app(export=export), host=host, port=port, log_level="info")


@runs_app.command("list")
def runs_list() -> None:
    """Lista las corridas persistidas."""
    from cancerstats.runs import RunRegistry

    reg = RunRegistry()
    latest = reg.latest()
    table = Table(title="Corridas")
    for col in ("id", "nombre", "n", "R² aj.", "modelo predictivo", "RMSE prueba", "s"):
        table.add_column(col)
    for r in reg.index():
        table.add_row(
            ("★ " if r["id"] == latest else "") + r["id"],
            str(r.get("name") or ""),
            str(r.get("n_effects") or ""),
            f"{r.get('r2_adj') or 0:.3f}",
            str(r.get("predictive_model") or "—"),
            f"{r['test_rmse']:.2f}" if r.get("test_rmse") else "—",
            f"{r.get('duration_s') or 0:.0f}",
        )
    console.print(table)


@runs_app.command("show")
def runs_show(run_id: str) -> None:
    """Muestra el manifiesto de una corrida."""
    from cancerstats.runs import RunRegistry

    console.print_json(json.dumps(RunRegistry().manifest(run_id), ensure_ascii=False))


@runs_app.command("rebuild-index")
def runs_rebuild() -> None:
    """Reconstruye runs/index.json a partir de los directorios."""
    from cancerstats.runs import RunRegistry

    rows = RunRegistry().rebuild_index()
    console.print(f"{len(rows)} corridas indexadas")


@config_app.command("schema")
def config_schema() -> None:
    """Imprime el esquema JSON de la configuración."""
    console.print_json(json.dumps(AnalysisConfig.model_json_schema(), ensure_ascii=False))


@config_app.command("default")
def config_default() -> None:
    """Imprime la configuración por defecto en YAML."""
    console.print(AnalysisConfig().to_yaml())


@config_app.command("validate")
def config_validate(path: Path) -> None:
    """Valida un fichero de configuración."""
    cfg = load_config(path, None)
    console.print(f"Configuración válida. Huella: {cfg.fingerprint()}")


@app.command()
def root() -> None:
    """Muestra la raíz del proyecto detectada."""
    console.print(str(project_root()))
