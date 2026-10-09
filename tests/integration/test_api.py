"""API del modo en vivo (con un pipeline simulado para que las pruebas sean rápidas)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cancerstats import pipeline
from cancerstats.api.server import create_app
from cancerstats.config import AnalysisConfig
from cancerstats.runs import RunRegistry


def _fake_results(run_id: str) -> dict[str, Any]:
    return {"run_id": run_id, "ingest": {"n_rows": 2}, "effects": {}, "predictive": None}


@pytest.fixture
def registry(tmp_path: Path) -> RunRegistry:
    reg = RunRegistry(tmp_path / "runs")
    reg.save(
        "20260101-000000-aaaaaaaa",
        AnalysisConfig(),
        {"environment": {}},
        _fake_results("20260101-000000-aaaaaaaa"),
        pd.DataFrame({"a": [1.0, None], "b": ["x", "y"]}),
        ["línea de registro"],
        {"t": pd.DataFrame({"x": [1]})},
    )
    return reg


@pytest.fixture
def client(registry: RunRegistry, tmp_path: Path) -> TestClient:
    return TestClient(create_app(registry, static_dir=tmp_path / "no-existe"))


def test_read_endpoints(client: TestClient) -> None:
    assert client.get("/api/health").json()["mode"] == "live"
    runs = client.get("/api/runs").json()
    assert runs["latest"] == "20260101-000000-aaaaaaaa"
    assert client.get("/api/runs/latest/results").json()["run_id"] == runs["latest"]
    ds = client.get("/api/runs/latest/dataset").json()
    assert ds["columns"] == ["a", "b"]
    assert ds["data"]["a"] == [1.0, None]
    assert client.get("/api/runs/latest/files/log.txt").text.strip() == "línea de registro"
    assert client.get("/api/runs/latest/tables/t.csv").status_code == 200
    assert len(client.get("/api/stages").json()) == len(pipeline.STAGES)


def test_unknown_resources_are_404(client: TestClient) -> None:
    assert client.get("/api/runs/no-existe/results").status_code == 404
    assert client.get("/api/runs/latest/files/secreto.txt").status_code == 404
    assert client.get("/api/runs/latest/tables/..%2F..%2Fconfig.yaml").status_code == 404
    assert client.get("/api/jobs/nada").status_code == 404


def test_config_endpoints(client: TestClient) -> None:
    default = client.get("/api/config/default").json()
    assert default["effects"]["covariance"] == "cluster"
    ok = client.post("/api/config/validate", json={"effects": {"alpha_remove": 0.1}})
    assert ok.status_code == 200
    assert ok.json()["config"]["effects"]["alpha_remove"] == 0.1
    leak = {"variables": {"candidates": ["avgDeathsPerYear"]}}
    assert client.post("/api/config/validate", json=leak).status_code == 422
    yaml_text = client.post("/api/config/yaml", json={"effects": {"alpha_remove": 0.1}}).text
    assert "alpha_remove: 0.1" in yaml_text


def test_job_lifecycle(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(config, registry, listener, persist=True):  # type: ignore[no-untyped-def]
        for k, (key, title) in enumerate(pipeline.STAGES):
            listener(pipeline.Event(key, "inicio", title, k / len(pipeline.STAGES)))
            time.sleep(0.01)
        listener(pipeline.Event("persistencia", "fin", "ok", 1.0))
        return "20260101-000000-aaaaaaaa", {}

    monkeypatch.setattr(pipeline, "run", fake_run)
    job = client.post("/api/jobs", json={"predictive": {"enabled": False}})
    assert job.status_code == 202
    jid = job.json()["id"]
    with client.stream("GET", f"/api/jobs/{jid}/events") as stream:
        lines = [ln for ln in stream.iter_lines() if ln.startswith("data:")]
    end = json.loads(lines[-1][5:])
    assert end["status"] == "terminada"
    assert end["run_id"] == "20260101-000000-aaaaaaaa"
    events = client.get(f"/api/jobs/{jid}").json()["events"]
    assert len(events) == len(pipeline.STAGES) + 1


def test_failed_job_reports_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(pipeline, "run", broken)
    jid = client.post("/api/jobs", json={}).json()["id"]
    for _ in range(100):
        state = client.get(f"/api/jobs/{jid}").json()
        if state["status"] != "en curso" and state["status"] != "en cola":
            break
        time.sleep(0.02)
    assert state["status"] == "error"
    assert "fallo simulado" in state["error"]
