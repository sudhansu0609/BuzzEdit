"""routes/supervisor.py (STUDIO_DIRECTOR_PLAN.md enhancement 4): the Studio
re-parents a headless BuzzEdit's Electron watchdog to its own pid instead of
spawning a second one; the pid is durable (survives a BuzzEdit restart)."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import supervisor


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(supervisor.router, prefix="/api/supervisor")
    return TestClient(app)


def test_parent_defaults_to_zero_when_never_set(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "STATE_PATH", tmp_path / "supervisor_parent.json")
    assert _client().get("/api/supervisor/parent").json() == {"pid": 0}


def test_setting_the_parent_persists_and_reads_back(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "STATE_PATH", tmp_path / "supervisor_parent.json")
    client = _client()
    posted = client.post("/api/supervisor/parent", json={"pid": 4242})
    assert posted.status_code == 200 and posted.json() == {"pid": 4242}
    assert client.get("/api/supervisor/parent").json() == {"pid": 4242}

    # Durable: a fresh read off disk (simulating a restart) sees it too.
    assert supervisor._load() == 4242
