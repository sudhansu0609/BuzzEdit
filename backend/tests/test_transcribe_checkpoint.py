"""A transcription interrupted part-way resumes from its last finished phase
(asr/transcribe_checkpoint.py), and a second POST for a project already
transcribing joins the first run instead of starting Whisper twice."""
import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from asr import transcribe_checkpoint as ckpt
from store.project_store import ProjectStore


# ───────────────────────────── the module ─────────────────────────────


def test_key_changes_with_source_and_settings(tmp_path):
    video = tmp_path / "take.mp4"
    video.write_bytes(b"x" * 10)
    a = ckpt.key_for(str(video), "hi", False, ["Sudhansu"])
    assert a == ckpt.key_for(str(video), "hi", False, ["Sudhansu"])
    assert a != ckpt.key_for(str(video), "en", False, ["Sudhansu"])
    assert a != ckpt.key_for(str(video), "hi", True, ["Sudhansu"])
    video.write_bytes(b"y" * 20)  # a new recording under the same name
    assert a != ckpt.key_for(str(video), "hi", False, ["Sudhansu"])


def test_phases_round_trip_and_a_different_key_is_ignored(tmp_path):
    key = {"source": "a.mp4", "size": 1, "mtime": 1, "language": "hi", "precut": False, "vocabulary": []}
    assert ckpt.load(tmp_path, "p1", key) == {}
    ckpt.save(tmp_path, "p1", key, ckpt.PHASE_WHISPER, {"words": [{"word": "hello"}], "language": "hi"})
    ckpt.save(tmp_path, "p1", key, ckpt.PHASE_PLAN, {"words": [{"word": "hello", "enabled": True}]})
    phases = ckpt.load(tmp_path, "p1", key)
    assert set(phases) == {ckpt.PHASE_WHISPER, ckpt.PHASE_PLAN}
    assert phases[ckpt.PHASE_WHISPER]["words"][0]["word"] == "hello"
    other = dict(key, language="en")
    assert ckpt.load(tmp_path, "p1", other) == {}
    # Saving under another key starts over: the old phases are not kept.
    ckpt.save(tmp_path, "p1", other, ckpt.PHASE_WHISPER, {"words": []})
    assert ckpt.load(tmp_path, "p1", key) == {}
    ckpt.clear(tmp_path, "p1")
    assert ckpt.load(tmp_path, "p1", other) == {}


def test_numpy_like_values_are_written(tmp_path):
    class Arr:
        def tolist(self):
            return [0.1, 0.2]

    key = {"source": "a.mp4", "size": 1, "mtime": 1, "language": "hi", "precut": False, "vocabulary": []}
    ckpt.save(tmp_path, "p1", key, ckpt.PHASE_PLAN, {"words": [], "report": {"energy": Arr()}})
    assert ckpt.load(tmp_path, "p1", key)[ckpt.PHASE_PLAN]["report"]["energy"] == [0.1, 0.2]


def test_a_damaged_checkpoint_is_no_checkpoint(tmp_path):
    key = {"source": "a.mp4", "size": 1, "mtime": 1, "language": "hi", "precut": False, "vocabulary": []}
    (tmp_path / "p1_transcribe_ckpt.json").write_text("{not json", encoding="utf-8")
    assert ckpt.load(tmp_path, "p1", key) == {}


# ───────────────────────────── the route ─────────────────────────────


@pytest.fixture
def route(tmp_path, monkeypatch):
    from main import app
    from routes import transcription as route_mod

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    store = ProjectStore(base_dir=str(projects_dir))
    monkeypatch.setattr(route_mod, "project_store", store)
    monkeypatch.setattr(route_mod, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(route_mod, "jobs", {})
    monkeypatch.setattr(route_mod, "_responses", {})

    video = tmp_path / "take.mp4"
    video.write_bytes(b"v" * 100)
    store.save_project("p1", {"id": "p1", "name": "T", "source_video": str(video),
                              "settings": {"auto_verify_cut": False}})

    monkeypatch.setattr(route_mod, "extract_audio", lambda src: str(tmp_path / "a.wav"))
    monkeypatch.setattr(route_mod, "get_video_info",
                        lambda src: {"duration": 3.0, "fps_num": 30, "fps_den": 1, "width": 1920,
                                     "height": 1080, "audio_codec": "aac"})

    words = [{"word": "Hello", "word_native": "Hello", "hinglish": "Hello", "start": 0.0, "end": 0.5},
             {"word": "world", "word_native": "world", "hinglish": "world", "start": 0.5, "end": 1.0}]
    calls = {"whisper": 0, "plan": 0}

    async def fake_whisper(audio_path, language=None, vocabulary=None, glossary=None):
        calls["whisper"] += 1
        return {"words": [dict(w) for w in words], "language": "hi", "native_text": "Hello world",
                "hinglish_text": "Hello world", "english_text": ""}

    class _Plan:
        def __init__(self, ws):
            self.words = ws
            self.report = {"planner": "classic"}

    async def fake_plan(ws, audio_path, aggressiveness=0.5, settings=None):
        calls["plan"] += 1
        return _Plan([dict(w, enabled=True) for w in ws])

    monkeypatch.setattr(route_mod.whisper_engine, "transcribe_full_async", fake_whisper)
    monkeypatch.setattr(route_mod, "plan_auto_edit", fake_plan)
    monkeypatch.setattr(route_mod, "record_cut_coverage", lambda tl, report: None)

    with TestClient(app) as test_client:
        yield test_client, store, projects_dir, calls


def test_a_run_that_dies_after_whisper_does_not_transcribe_again(route, monkeypatch):
    test_client, store, projects_dir, calls = route
    from routes import transcription as route_mod

    # First run: the planner "crashes" after Whisper has checkpointed.
    async def dying_plan(*args, **kwargs):
        calls["plan"] += 1
        raise RuntimeError("BuzzEdit died here")

    monkeypatch.setattr(route_mod, "plan_auto_edit", dying_plan)
    resp = test_client.post("/api/transcription/transcribe", json={"project_id": "p1", "language": "hi"})
    assert resp.status_code == 500
    assert calls == {"whisper": 1, "plan": 1}
    assert (projects_dir / "p1_transcribe_ckpt.json").exists()
    assert route_mod.jobs["trans_p1"]["status"] == "failed"

    # Second run (the Studio's retry): Whisper is not run again.
    async def good_plan(ws, audio_path, aggressiveness=0.5, settings=None):
        calls["plan"] += 1

        class _Plan:
            words = [dict(w, enabled=True) for w in ws]
            report = {"planner": "classic"}
        return _Plan()

    monkeypatch.setattr(route_mod, "plan_auto_edit", good_plan)
    resp = test_client.post("/api/transcription/transcribe", json={"project_id": "p1", "language": "hi"})
    assert resp.status_code == 200, resp.text
    assert calls == {"whisper": 1, "plan": 2}
    assert store.get_project("p1")["status"] == "transcribed"
    assert not (projects_dir / "p1_transcribe_ckpt.json").exists()  # cleared once saved
    assert route_mod.jobs["trans_p1"]["status"] == "completed"


def test_a_run_that_dies_after_planning_reuses_the_plan(route, monkeypatch):
    test_client, store, projects_dir, calls = route
    from routes import transcription as route_mod

    def dying_build(*args, **kwargs):
        raise RuntimeError("killed while building the timeline")

    real_build = route_mod.build_timeline_from_transcript
    monkeypatch.setattr(route_mod, "build_timeline_from_transcript", dying_build)
    resp = test_client.post("/api/transcription/transcribe", json={"project_id": "p1", "language": "hi"})
    assert resp.status_code == 500
    assert calls == {"whisper": 1, "plan": 1}

    monkeypatch.setattr(route_mod, "build_timeline_from_transcript", real_build)
    resp = test_client.post("/api/transcription/transcribe", json={"project_id": "p1", "language": "hi"})
    assert resp.status_code == 200, resp.text
    assert calls == {"whisper": 1, "plan": 1}  # neither phase ran again
    assert store.get_project("p1")["status"] == "transcribed"


def test_a_new_recording_starts_over(route):
    test_client, store, projects_dir, calls = route
    key = ckpt.key_for(str(Path(store.get_project("p1")["source_video"])), "hi", False, None)
    ckpt.save(projects_dir, "p1", dict(key, size=key["size"] + 1), ckpt.PHASE_WHISPER,
              {"words": [{"word": "stale", "word_native": "stale", "hinglish": "stale", "start": 0, "end": 1}],
               "language": "hi"})
    resp = test_client.post("/api/transcription/transcribe", json={"project_id": "p1", "language": "hi"})
    assert resp.status_code == 200, resp.text
    assert calls["whisper"] == 1


def test_a_second_post_joins_the_running_transcription(route, monkeypatch):
    test_client, store, projects_dir, calls = route
    from routes import transcription as route_mod

    monkeypatch.setattr(route_mod, "_JOIN_POLL_SECONDS", 0.01)
    route_mod.jobs["trans_p1"] = {"id": "trans_p1", "project_id": "p1", "status": "running", "stage": "planning"}

    async def finish_soon():
        await asyncio.sleep(0.05)
        route_mod._responses["trans_p1"] = {"status": "completed", "job_id": "trans_p1", "segments": 1}
        route_mod.jobs["trans_p1"]["status"] = "completed"

    async def joined():
        task = asyncio.create_task(finish_soon())
        try:
            return await route_mod.transcribe_video(route_mod.TranscribeJob(project_id="p1", language="hi"))
        finally:
            await task

    result = asyncio.run(joined())
    assert result["status"] == "completed" and result["segments"] == 1
    assert calls["whisper"] == 0  # Whisper never started a second time
