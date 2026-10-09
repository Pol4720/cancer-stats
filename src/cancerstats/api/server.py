"""API de la interfaz en modo en vivo.

La interfaz web funciona de dos maneras con el mismo código:

* **Estática** (GitHub Pages): lee los ficheros JSON de la última corrida, empaquetados al
  construir el sitio; permite explorar y exportar, pero no ejecutar.
* **En vivo** (este servidor): además permite editar la configuración, lanzar corridas y
  seguir su progreso etapa a etapa. Las rutas de lectura devuelven exactamente los mismos
  documentos que la versión estática, de modo que la interfaz no distingue el origen.

Las corridas se ejecutan de una en una en un hilo aparte (el pipeline usa todos los núcleos
en la etapa predictiva) y su progreso se publica como *Server-Sent Events*.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from cancerstats import jsonutil
from cancerstats.config import AnalysisConfig, merge_overrides
from cancerstats.export.spss import syntax
from cancerstats.paths import default_config_path, project_root
from cancerstats.pipeline import STAGES, Event
from cancerstats.runs import RunRegistry

JsonBody = Annotated[dict[str, Any] | None, Body()]

RUN_FILES = {
    "config.yaml": "text/yaml; charset=utf-8",
    "results.json": "application/json",
    "manifest.json": "application/json",
    "log.txt": "text/plain; charset=utf-8",
}


@dataclass
class Job:
    """Corrida lanzada desde la interfaz."""

    id: str
    status: str = "en cola"
    events: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Estado serializable (sin la traza completa de eventos)."""
        last = self.events[-1] if self.events else None
        return {
            "id": self.id,
            "status": self.status,
            "run_id": self.run_id,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
            "progress": last["progress"] if last else 0.0,
            "last": last,
            "n_events": len(self.events),
        }


class JobManager:
    """Cola de corridas: una a la vez, con su traza de eventos."""

    def __init__(self, registry: RunRegistry, export: bool) -> None:
        self.registry = registry
        self.export = export
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._running = threading.Lock()

    def submit(self, config: AnalysisConfig) -> Job:
        """Encola una corrida y la lanza en segundo plano."""
        job = Job(id=uuid.uuid4().hex[:12])
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._work, args=(job, config), daemon=True).start()
        return job

    def busy(self) -> bool:
        """Si hay una corrida en curso o en cola."""
        return any(j.status in ("en cola", "en curso") for j in self.jobs.values())

    def _work(self, job: Job, config: AnalysisConfig) -> None:
        from cancerstats.pipeline import run

        with self._running:
            job.status = "en curso"

            def listener(ev: Event) -> None:
                job.events.append({**ev.to_dict(), "t": round(time.time() - job.started, 2)})

            try:
                run_id, _ = run(config, self.registry, listener, persist=True)
                job.run_id = run_id
                if self.export:
                    from cancerstats.export import export_run

                    job.events.append(
                        {
                            "stage": "exportacion",
                            "status": "inicio",
                            "message": "Regenerando el material del informe",
                            "progress": 1.0,
                            "t": round(time.time() - job.started, 2),
                        }
                    )
                    export_run(run_id, self.registry)
                job.status = "terminada"
            except Exception as exc:  # la interfaz muestra el error tal cual
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.events.append(
                    {
                        "stage": "error",
                        "status": "error",
                        "message": job.error,
                        "detail": traceback.format_exc(limit=5),
                        "progress": 1.0,
                        "t": round(time.time() - job.started, 2),
                    }
                )
            finally:
                job.finished = time.time()


def _config_from_body(body: dict[str, Any] | None) -> AnalysisConfig:
    """Configuración por defecto con las claves recibidas superpuestas."""
    import yaml

    base = yaml.safe_load(default_config_path().read_text(encoding="utf-8")) or {}
    try:
        return AnalysisConfig.from_mapping(merge_overrides(base, body or {}))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def create_app(
    registry: RunRegistry | None = None,
    static_dir: Path | None = None,
    export: bool = False,
) -> FastAPI:
    """Construye la aplicación.

    Args:
        registry: registro de corridas (por defecto, ``runs/`` del proyecto).
        static_dir: interfaz compilada que se sirve en ``/`` (por defecto ``web/dist`` si
            existe).
        export: si al terminar cada corrida se regenera ``report/generado``.
    """
    reg = registry or RunRegistry()
    jobs = JobManager(reg, export)
    app = FastAPI(
        title="cancer-stats",
        description="Mortalidad por cáncer en los condados de EE. UU.: interfaz en vivo.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.jobs = jobs
    app.state.registry = reg

    def _check(run_id: str) -> str:
        rid = reg.latest() if run_id == "latest" else run_id
        if rid is None:
            raise HTTPException(404, "No hay corridas: ejecute antes una.")
        try:
            reg.paths(rid)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return rid

    # ------------------------------------------------------------------ información
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "mode": "live", "busy": jobs.busy(), "latest": reg.latest()}

    @app.get("/api/stages")
    def stages() -> list[dict[str, str]]:
        return [{"id": k, "title": v} for k, v in STAGES]

    # --------------------------------------------------------------- configuración
    @app.get("/api/config/schema")
    def config_schema() -> dict[str, Any]:
        return AnalysisConfig.model_json_schema()

    @app.get("/api/config/default")
    def config_default() -> dict[str, Any]:
        return _config_from_body(None).model_dump(mode="json")

    @app.post("/api/config/validate")
    def config_validate(body: JsonBody = None) -> dict[str, Any]:
        cfg = _config_from_body(body)
        return {"ok": True, "fingerprint": cfg.fingerprint(), "config": cfg.model_dump(mode="json")}

    @app.post("/api/config/yaml", response_class=PlainTextResponse)
    def config_yaml(body: JsonBody = None) -> str:
        return _config_from_body(body).to_yaml()

    # --------------------------------------------------------------------- corridas
    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        return {"latest": reg.latest(), "runs": reg.index()}

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: str) -> Response:
        rid = _check(run_id)
        return FileResponse(reg.paths(rid).results, media_type="application/json")

    @app.get("/api/runs/{run_id}/config")
    def run_config(run_id: str) -> dict[str, Any]:
        return reg.config(_check(run_id)).model_dump(mode="json")

    @app.get("/api/runs/{run_id}/dataset")
    def dataset(run_id: str) -> Response:
        rid = _check(run_id)
        res = reg.results(rid)
        path = reg.base / "_datasets" / f"{res['dataset']}.json"
        return FileResponse(path, media_type="application/json")

    @app.get("/api/runs/{run_id}/files/{name}")
    def run_file(run_id: str, name: str) -> Response:
        rid = _check(run_id)
        if name == "modelo_final.sps":
            text = syntax(reg.results(rid), "practica.sav")
            return Response(
                text,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{name}"'},
            )
        if name not in RUN_FILES:
            raise HTTPException(404, f"Fichero desconocido: {name}")
        return FileResponse(reg.paths(rid).root / name, media_type=RUN_FILES[name], filename=name)

    @app.get("/api/runs/{run_id}/tables/{name}")
    def run_table(run_id: str, name: str) -> Response:
        rid = _check(run_id)
        path = (reg.paths(rid).tables / name).resolve()
        if path.parent != reg.paths(rid).tables.resolve() or not path.is_file():
            raise HTTPException(404, f"Tabla desconocida: {name}")
        return FileResponse(path, filename=name)

    # ----------------------------------------------------------------------- trabajos
    @app.post("/api/jobs", status_code=202)
    def submit(body: JsonBody = None) -> dict[str, Any]:
        if jobs.busy():
            raise HTTPException(409, "Ya hay una corrida en curso.")
        cfg = _config_from_body(body)
        return jobs.submit(cfg).to_dict()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [j.to_dict() for j in jobs.jobs.values()]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        if job_id not in jobs.jobs:
            raise HTTPException(404, "Trabajo desconocido")
        return {**jobs.jobs[job_id].to_dict(), "events": jobs.jobs[job_id].events}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        if job_id not in jobs.jobs:
            raise HTTPException(404, "Trabajo desconocido")
        j = jobs.jobs[job_id]

        async def stream():  # type: ignore[no-untyped-def]
            sent = 0
            while True:
                while sent < len(j.events):
                    yield f"data: {json.dumps(jsonutil.to_jsonable(j.events[sent]))}\n\n"
                    sent += 1
                if j.finished is not None and sent >= len(j.events):
                    yield f"event: end\ndata: {json.dumps(j.to_dict())}\n\n"
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    # --------------------------------------------------------------------- interfaz
    web = static_dir or project_root() / "web" / "dist"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
