"""GET /api/reload/busy (STUDIO_DIRECTOR_PLAN.md enhancement 4): the Electron
watchdog's own busy check, straight from `code_reload.busy_reason` -- the
thing it polls instead of quitting a working BuzzEdit the moment its Studio
parent disappears."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import health
from runtime import code_reload


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(health.router, prefix="/api")
    return TestClient(app)


def test_reload_busy_reports_free_when_nothing_is_busy(monkeypatch):
    monkeypatch.setattr(code_reload, "busy_reason", lambda route_tasks=0: None)
    resp = _client().get("/api/reload/busy")
    assert resp.status_code == 200
    assert resp.json() == {"busy": False, "reason": None}


def test_reload_busy_reports_the_reason_when_busy(monkeypatch):
    monkeypatch.setattr(code_reload, "busy_reason", lambda route_tasks=0: "a reel is rendering")
    resp = _client().get("/api/reload/busy")
    assert resp.json() == {"busy": True, "reason": "a reel is rendering"}
