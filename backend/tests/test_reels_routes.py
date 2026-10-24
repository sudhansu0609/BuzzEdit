"""routes/reels.py: the vibe reel job survives a restart (STUDIO_DIRECTOR_PLAN.md
enhancement 2, Fable review row 7) -- its own JSON on disk, not just the
in-memory `_jobs` dict, and a startup sweep that fails anything left running."""
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from presentation import vibe_reel
from routes import reels
from runtime import job_store


@pytest.fixture(autouse=True)
def _isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path / "jobs")
    reels._jobs.clear()
    yield
    reels._jobs.clear()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(reels.router, prefix="/api/reels")
    return TestClient(app)


def _request(tmp_path, **overrides) -> dict:
    body = dict(
        out_dir=str(tmp_path / "out"),
        name="reel",
        art_style="real",
        world="a quiet hill town at night",
        shots=[{"prompt": "a moonlit lake", "seconds": 6.0, "kind": "still"}],
        lines=[{"text": "line one", "start_s": 0.0, "end_s": 6.0}],
    )
    body.update(overrides)
    return body


def _wait_terminal(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = reels._jobs.get(job_id)
        if job and job["state"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_job_is_persisted_and_answers_from_disk_after_a_restart(tmp_path, monkeypatch):
    async def fake_render(req, progress_cb=None):
        if progress_cb:
            progress_cb(0.5, "halfway")
        return {"video": str(tmp_path / "reel.mp4"), "cover": None, "duration_s": 6.0}

    monkeypatch.setattr(vibe_reel, "render_vibe_reel", fake_render)
    client = _client()
    started = client.post("/api/reels/vibe", json=_request(tmp_path))
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    job = _wait_terminal(job_id)
    assert job["state"] == "done"

    on_disk = job_store.load(reels.JOB_KIND, job_id)
    assert on_disk is not None and on_disk["state"] == "done"
    assert on_disk["out_dir"] == str(tmp_path / "out")

    # Simulate a restart: the in-memory registry is gone, the route still answers.
    reels._jobs.clear()
    resp = client.get(f"/api/reels/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["state"] == "done"


def test_a_failed_render_is_persisted_too(tmp_path, monkeypatch):
    async def fake_render(req, progress_cb=None):
        raise vibe_reel.VibeReelError("1 shot(s) could not be made: shot0 (near-black frame)")

    monkeypatch.setattr(vibe_reel, "render_vibe_reel", fake_render)
    client = _client()
    started = client.post("/api/reels/vibe", json=_request(tmp_path))
    job_id = started.json()["job_id"]
    job = _wait_terminal(job_id)
    assert job["state"] == "failed" and "near-black" in job["error"]
    on_disk = job_store.load(reels.JOB_KIND, job_id)
    assert on_disk["state"] == "failed"


def test_startup_sweep_marks_a_stale_reel_job_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "JOBS_ROOT", tmp_path / "jobs")
    job_store.save(reels.JOB_KIND, "stale1", {"job_id": "stale1", "state": "running", "progress": 0.3})
    job_store.save(reels.JOB_KIND, "queued1", {"job_id": "queued1", "state": "queued"})
    marked = job_store.sweep_stale(reels.JOB_KIND)
    assert marked == 2
    for jid in ("stale1", "queued1"):
        on_disk = job_store.load(reels.JOB_KIND, jid)
        assert on_disk["state"] == "failed" and on_disk["error"] == job_store.RESTARTED_ERROR


def test_unknown_job_is_404():
    client = _client()
    assert client.get("/api/reels/jobs/does-not-exist").status_code == 404
