"""routes/motion_gallery.py wiring (MOTION_GRAPHICS_PLAN.md M11): the gallery
refuses to start while a production render is running, and starts (with the
planned total up front) when nothing is. No real Node render is run."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from presentation import motion_gallery
from routes import motion_gallery as motion_gallery_routes
from routes import presentation as presentation_routes
from routes import rendering as rendering_routes


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(motion_gallery_routes.router, prefix="/api/motion")
    return TestClient(app)


def _reset():
    rendering_routes.jobs.clear()
    presentation_routes._jobs.clear()
    motion_gallery_routes._state.update(state="idle", folder=None, count=0, errors=[], total=0, message="")


def test_refuses_while_a_render_job_is_running():
    _reset()
    rendering_routes.jobs["render_p1"] = {"id": "render_p1", "status": "running"}
    try:
        resp = _client().post("/api/motion/gallery", json={})
        assert resp.status_code == 409
    finally:
        _reset()


def test_starts_with_the_planned_total_when_nothing_is_rendering(monkeypatch):
    _reset()
    monkeypatch.setattr(
        motion_gallery, "dump_compositions",
        lambda: [{"id": "KineticText", "defaultProps": {"theme": {}}, "fps": 30, "durationInFrames": 10}])

    async def fake_run(*args, **kwargs):
        return {"folder": "X", "count": 2, "errors": [], "rendered": 2}

    monkeypatch.setattr(motion_gallery_routes, "_run", fake_run)
    resp = _client().post("/api/motion/gallery", json={"themes": ["clean", "noir"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "running"
    assert body["total"] == 2
    _reset()
