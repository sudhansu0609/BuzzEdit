"""Routes of the SFX library API (CLIPS_AND_SOUND_PLAN.md, brief S1): the
library folders, re-index, resolving an effect, and generating one that is
missing into the library.

Runs on a bare FastAPI app with only the sfx router. `sfx_library.LIBRARY_DIR`,
`CREATOR_LIBRARY` and `INDEX_PATH` are pointed into tmp_path (and its memo
cleared), and the `app_settings` singleton is pointed at a tmp JSON file, so
nothing here touches the real library or settings. ComfyUI, the audio
workflow and the GPU hand-over are monkeypatched; `_no_real_gpu_handover`
(tests/conftest.py) already keeps the real hand-over out of every test.
"""

import time
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from presentation import director, sfx_library, workflows
from store.app_settings import app_settings


def _wav(path: Path, seconds: float = 0.5, rate: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Jobs live in a module-level dict; never let one test's job leak into
    another's "is one already running?" check."""
    from routes import sfx as sfx_routes
    sfx_routes._jobs.clear()
    yield
    sfx_routes._jobs.clear()


@pytest.fixture
def env(tmp_path, monkeypatch):
    from routes import sfx as sfx_routes

    monkeypatch.setattr(sfx_library, "LIBRARY_DIR", tmp_path / "lib")
    monkeypatch.setattr(sfx_library, "CREATOR_LIBRARY", tmp_path / "creator_default")
    monkeypatch.setattr(sfx_library, "INDEX_PATH", tmp_path / "sfx_index.json")
    sfx_library._memo.clear()

    monkeypatch.setattr(app_settings, "settings_dir", tmp_path)
    monkeypatch.setattr(app_settings, "settings_file", tmp_path / "app_settings.json")
    monkeypatch.setattr(app_settings, "_cache", {})

    # The library's own seeded files: data/sfx (LIBRARY_DIR) is always a root.
    _wav(tmp_path / "lib" / "footSteps" / "walk1.wav")
    _wav(tmp_path / "lib" / "screams" / "scream_a.wav")

    app = FastAPI()
    app.include_router(sfx_routes.router, prefix="/api/sfx")
    with TestClient(app) as client:
        yield client, tmp_path, sfx_routes


@pytest.fixture
def ready_to_generate(monkeypatch, tmp_path):
    """ComfyUI online, an audio workflow configured, and a fake generation
    that writes a real 1 s WAV instead of calling Stable Audio."""
    monkeypatch.setattr(director, "_comfyui_online", lambda: True)
    monkeypatch.setattr(workflows, "resolve", lambda role: object())

    async def _fake_generated_effect(text, duration, seed=0):
        return _wav(tmp_path / "generated_takes" / f"take_{seed}.wav", seconds=1.0)

    monkeypatch.setattr(sfx_library, "generated_effect", _fake_generated_effect)


def _wait_for_job(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/sfx/jobs/{job_id}").json()
        if body["state"] in ("done", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


# --- library summary and configuration ------------------------------------------

def test_library_counts_the_seeded_files_and_lists_tags(env):
    client, _tmp, _routes = env
    body = client.get("/api/sfx/library").json()
    assert body["count"] == 2
    assert body["effects"] == 2
    assert body["beds"] == 0 and body["music"] == 0
    assert body["dirs"] == []
    assert body["default"] == [str(sfx_library.CREATOR_LIBRARY)]
    assert body["missing"] == []
    assert body["indexed_at"]
    tags = {t["tag"] for t in body["tags"]}
    # "footSteps" canonicalises to "footsteps"; "screams" to "shout" (CANON).
    assert "footsteps" in tags
    assert "shout" in tags


def test_put_library_refuses_a_missing_directory(env):
    client, tmp, _routes = env
    bad_dir = tmp / "nope"
    resp = client.put("/api/sfx/library", json={"dirs": [str(bad_dir)]})
    assert resp.status_code == 422
    assert "nope" in resp.json()["detail"]
    # Refused: the setting is untouched.
    assert client.get("/api/sfx/library").json()["dirs"] == []


def test_put_library_saves_a_real_directory_and_recounts(env):
    client, tmp, _routes = env
    extra = tmp / "extra_dir"
    _wav(extra / "doorKnock" / "knock1.wav")

    resp = client.put("/api/sfx/library", json={"dirs": [str(extra)]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dirs"] == [str(extra)]
    assert str(extra) in body["roots"]
    assert body["count"] == 3        # the 2 seeded files + the new one

    again = client.get("/api/sfx/library").json()
    assert again["dirs"] == [str(extra)] and again["count"] == 3

    # [] goes back to the default: the setting clears, roots fall back to it.
    back = client.put("/api/sfx/library", json={"dirs": []})
    assert back.status_code == 200
    assert back.json()["dirs"] == []
    assert str(sfx_library.CREATOR_LIBRARY) in back.json()["roots"]


def test_refresh_reindexes(env):
    client, _tmp, _routes = env
    resp = client.post("/api/sfx/library/refresh")
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


# --- resolving an effect -----------------------------------------------------------

def test_resolve_finds_a_library_match(env):
    client, _tmp, _routes = env
    resp = client.get("/api/sfx/resolve", params={"text": "footsteps on gravel"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "footsteps on gravel" and body["kind"] == "accent"
    assert body["source"] == "library"
    assert body["path"] is not None and "footSteps" in body["path"]
    assert body["duration_s"] is not None


# --- generating a missing effect ----------------------------------------------------

def test_generate_saves_into_the_library_and_is_then_findable(env, ready_to_generate):
    client, _tmp, _routes = env
    resp = client.post("/api/sfx/generate", json={"text": "a woman screams in terror"})
    assert resp.status_code == 200
    body = resp.json()
    job_id = body["job_id"]
    assert body["state"] in ("queued", "running", "done")

    job = _wait_for_job(client, job_id)
    assert job["state"] == "done", job.get("error")
    # Saved under LIBRARY_DIR/generated/<tag>/..., tag = canonical("screams") = "shout".
    assert "generated" in job["path"] and "shout" in job["path"]
    assert job["duration_s"] is not None

    # Now in the library's count...
    library = client.get("/api/sfx/library").json()
    assert library["count"] == 3

    # ...findable by the same text...
    resolved = client.get("/api/sfx/resolve", params={"text": "a woman screams in terror"}).json()
    assert resolved["source"] == "library"
    assert resolved["path"] == job["path"]

    # ...and servable.
    served = client.get("/api/sfx/file", params={"path": job["path"]})
    assert served.status_code == 200


def test_generate_refuses_empty_text(env, ready_to_generate):
    client, _tmp, _routes = env
    assert client.post("/api/sfx/generate", json={"text": "   "}).status_code == 400


def test_generate_refuses_when_comfyui_is_offline(env, monkeypatch):
    client, _tmp, _routes = env
    monkeypatch.setattr(director, "_comfyui_online", lambda: False)
    monkeypatch.setattr(workflows, "resolve", lambda role: object())
    resp = client.post("/api/sfx/generate", json={"text": "a scream"})
    assert resp.status_code == 503


def test_generate_refuses_without_an_audio_workflow(env, monkeypatch):
    client, _tmp, _routes = env
    monkeypatch.setattr(director, "_comfyui_online", lambda: True)
    monkeypatch.setattr(workflows, "resolve", lambda role: None)
    resp = client.post("/api/sfx/generate", json={"text": "a scream"})
    assert resp.status_code == 503


def test_generate_refuses_a_second_job_while_one_runs(env, ready_to_generate):
    client, _tmp, routes = env
    routes._jobs["already"] = {
        "job_id": "already", "state": "running", "stage": "generate", "text": "x",
        "kind": "accent", "seconds": 4.0, "path": None, "duration_s": None,
        "error": None, "started_at": time.time(), "ended_at": None,
    }
    resp = client.post("/api/sfx/generate", json={"text": "a scream"})
    assert resp.status_code == 409


# --- jobs and files -----------------------------------------------------------------

def test_job_404_for_an_unknown_id(env):
    client, _tmp, _routes = env
    assert client.get("/api/sfx/jobs/nope").status_code == 404


def test_get_file_refuses_a_path_outside_the_library(env, tmp_path_factory):
    client, _tmp, _routes = env
    outside_dir = tmp_path_factory.mktemp("outside")
    outside = _wav(outside_dir / "outside.wav")
    resp = client.get("/api/sfx/file", params={"path": str(outside)})
    assert resp.status_code == 403


def test_get_file_404_for_a_missing_file_inside_the_library(env):
    client, tmp, _routes = env
    missing = tmp / "lib" / "footSteps" / "nope.wav"
    resp = client.get("/api/sfx/file", params={"path": str(missing)})
    assert resp.status_code == 404
