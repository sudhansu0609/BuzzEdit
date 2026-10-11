"""Routes of the audio suite (AUDIO_SUITE_PLAN.md §2.6, brief A1a): the
catalogue, user presets, a project's racks and the rendered rack preview.

The routes run on a bare FastAPI app with the project store, the projects
directory and the presets file pointed into tmp_path. Previews run a real
ffmpeg on synthetic tones; the measured gains force the compiler's
amplify-only fallback so they do not depend on `audio.compile`.
"""

import array
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import FFMPEG_BIN
from render.compiler import FilterGraphCompiler
from store.project_store import ProjectStore
from timeline.schema import AudioEffect, AudioMaster, SourceFile, Timeline, TimelineItem


@pytest.fixture
def env(tmp_path, monkeypatch):
    from routes import audio as audio_routes

    projects = tmp_path / "projects"
    projects.mkdir()
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(audio_routes, "project_store", store)
    monkeypatch.setattr(audio_routes, "PROJECTS_DIR", projects)
    monkeypatch.setattr(audio_routes, "PRESETS_PATH", tmp_path / "data" / "audio_presets.json")
    monkeypatch.setattr(audio_routes, "RACK_PRESETS_PATH",
                        tmp_path / "data" / "audio_rack_presets.json")
    app = FastAPI()
    app.include_router(audio_routes.router, prefix="/api/audio")
    return TestClient(app), store, tmp_path


@pytest.fixture
def no_audio_compile(monkeypatch):
    """Make `from audio.compile import compile_rack` fail, as before A1b."""
    monkeypatch.setitem(sys.modules, "audio.compile", None)


def _project(main_path: str = "C:/media/talk.mp4", bed_path: str = "C:/media/bed.mp3") -> Timeline:
    sources = {
        "main": SourceFile(id="main", path=main_path, duration_seconds=25.0),
        "bed": SourceFile(id="bed", path=bed_path, duration_seconds=25.0, kind="audio",
                          width=0, height=0),
    }
    span = dict(source_start_frame=0, source_end_frame=300, timeline_start_frame=0,
                timeline_end_frame=300)
    items = [
        TimelineItem(id="v1_0", track="V1", source_id="main", **span),
        TimelineItem(id="a1_0", track="A1", source_id="main", **span),
        TimelineItem(id="bed_0", track="A2", source_id="bed", **span),
    ]
    return Timeline(fps_num=30, fps_den=1, sources=sources, items=items,
                    duration_frames=300, revision=4)


def _save(store: ProjectStore, tl: Timeline, pid: str = "p1") -> None:
    store.save_project(pid, {"id": pid, "name": "T", "timeline": tl.model_dump()})


def _stored(store: ProjectStore, pid: str = "p1") -> Timeline:
    return Timeline.model_validate(store.get_project(pid)["timeline"])


def _amp(effect_id: str, gain_db: float = 3.0, **extra):
    return {"id": effect_id, "effect": "amplify", "params": {"gain_db": gain_db}, **extra}


# --- registration and catalogue --------------------------------------------------

def test_main_registers_the_audio_routes():
    from main import app

    paths = app.openapi()["paths"]
    expected = {
        "/api/audio/effects": {"get"},
        "/api/audio/presets": {"get", "post"},
        "/api/audio/presets/{effect}/{name}": {"delete"},
        "/api/audio/{project_id}/racks": {"get"},
        "/api/audio/{project_id}/rack": {"put"},
        "/api/audio/{project_id}/preview": {"post"},
    }
    for path, methods in expected.items():
        assert methods <= set(paths.get(path, {})), path


def test_effects_lists_the_catalogue(env):
    client, _store, _tmp = env
    resp = client.get("/api/audio/effects")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["effects"], list) and isinstance(body["categories"], list)
    assert "amplify" in {e["id"] for e in body["effects"]}


def test_effects_falls_back_to_amplify_without_the_catalogue(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "audio.catalogue", None)
    client, _store, _tmp = env
    body = client.get("/api/audio/effects").json()
    assert [e["id"] for e in body["effects"]] == ["amplify"]
    assert body["effects"][0]["params"][0]["key"] == "gain_db"
    assert body["categories"] == [body["effects"][0]["category"]]


# --- presets ------------------------------------------------------------------------

def test_presets_round_trip_through_the_presets_file(env):
    client, _store, tmp = env
    path = tmp / "data" / "audio_presets.json"
    assert client.get("/api/audio/presets").json() == {"presets": {}}

    resp = client.post("/api/audio/presets",
                       json={"effect": "amplify", "name": "Quiet", "params": {"gain_db": -6}})
    assert resp.status_code == 200
    client.post("/api/audio/presets",
                json={"effect": "amplify", "name": "Loud", "params": {"gain_db": 6}})
    expected = {"amplify": {"Quiet": {"gain_db": -6}, "Loud": {"gain_db": 6}}}
    assert client.get("/api/audio/presets").json() == {"presets": expected}
    assert json.loads(path.read_text(encoding="utf-8")) == expected

    assert client.delete("/api/audio/presets/amplify/Quiet").status_code == 200
    assert client.get("/api/audio/presets").json() == {
        "presets": {"amplify": {"Loud": {"gain_db": 6}}}}
    assert client.delete("/api/audio/presets/amplify/Quiet").status_code == 404
    assert client.delete("/api/audio/presets/amplify/Loud").status_code == 200
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_presets_refuse_names_the_delete_route_cannot_reach(env):
    client, _store, _tmp = env
    for body in ({"effect": "amplify", "name": "a/b"}, {"effect": "", "name": "x"},
                 {"effect": "amplify", "name": "  "}):
        assert client.post("/api/audio/presets", json=body).status_code == 422


# --- racks ----------------------------------------------------------------------------

def test_racks_of_a_project_without_racks(env):
    client, store, _tmp = env
    _save(store, _project())
    resp = client.get("/api/audio/p1/racks")
    assert resp.status_code == 200
    assert resp.json() == {
        "racks": {"A1": [], "MASTER": []},
        "items": {},
        "sources": {"main": {"audio_override": None, "history_len": 0},
                    "bed": {"audio_override": None, "history_len": 0}},
    }
    assert client.get("/api/audio/nope/racks").status_code == 404


def test_racks_report_the_processed_audio_of_each_source(env):
    client, store, _tmp = env
    tl = _project()
    tl.sources["main"].audio_override = "C:/proj/audio/versions/2_noise_reduction.wav"
    tl.sources["main"].audio_history = [
        {"path": "C:/proj/audio/versions/1_declip.wav", "effect": "declip", "params": {},
         "created": "2026-10-09T10:00:00"},
        {"path": "C:/proj/audio/versions/2_noise_reduction.wav", "effect": "noise_reduction",
         "params": {}, "created": "2026-10-09T10:05:00"}]
    _save(store, tl)
    sources = client.get("/api/audio/p1/racks").json()["sources"]
    assert sources["main"] == {"audio_override": "C:/proj/audio/versions/2_noise_reduction.wav",
                               "history_len": 2}


def test_put_rack_round_trips_every_scope(env):
    client, store, _tmp = env
    _save(store, _project())

    resp = client.put("/api/audio/p1/rack", json={"scope": "A1", "effects": [_amp("fx_a1")]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == "A1" and body["revision"] == 5
    assert body["effects"] == [{"id": "fx_a1", "effect": "amplify", "params": {"gain_db": 3.0},
                                "enabled": True, "start_s": None, "end_s": None, "preset": None}]

    window = _amp("fx_master", -1.5, start_s=2.0, end_s=4.0, preset="Quiet")
    assert client.put("/api/audio/p1/rack",
                      json={"scope": "MASTER", "effects": [window]}).json()["revision"] == 6
    assert client.put("/api/audio/p1/rack",
                      json={"scope": "item:bed_0", "effects": [_amp("fx_bed", -6)]}
                      ).json()["revision"] == 7
    assert client.put("/api/audio/p1/rack",
                      json={"scope": "A2", "effects": [_amp("fx_a2")]}).json()["revision"] == 8

    racks = client.get("/api/audio/p1/racks").json()
    assert [e["id"] for e in racks["racks"]["A1"]] == ["fx_a1"]
    assert racks["racks"]["MASTER"] == [{**window, "enabled": True}]
    assert [e["id"] for e in racks["racks"]["A2"]] == ["fx_a2"]
    assert {k: [e["id"] for e in v] for k, v in racks["items"].items()} == {"bed_0": ["fx_bed"]}

    stored = _stored(store)
    assert stored.revision == 8
    assert stored.audio_racks["A1"][0].params == {"gain_db": 3.0}
    assert next(i for i in stored.items if i.id == "bed_0").audio_effects[0].id == "fx_bed"

    # What was stored is what the render compiles.
    _inputs, graph, _v, _a = FilterGraphCompiler(stored).compile()
    for instance in ("@fx_a1_0", "@fx_master_0", "@fx_bed_0"):
        assert instance in graph

    # PUT replaces: an empty list clears the rack.
    assert client.put("/api/audio/p1/rack", json={"scope": "A1", "effects": []}).status_code == 200
    assert client.get("/api/audio/p1/racks").json()["racks"]["A1"] == []
    assert _stored(store).revision == 9


@pytest.mark.parametrize("body, status", [
    ({"scope": "V1", "effects": []}, 422),
    ({"scope": "master", "effects": []}, 422),
    ({"scope": "item:", "effects": []}, 422),
    ({"scope": "item:nope", "effects": []}, 404),
    ({"scope": "item:a1_0", "effects": [{"id": "fx_x", "effect": "amplify"}]}, 422),
    ({"scope": "item:v1_0", "effects": []}, 422),
    ({"scope": "A1", "effects": [{"id": "fx;x", "effect": "amplify"}]}, 422),
    ({"scope": "A1", "effects": [{"id": "fx x", "effect": "amplify"}]}, 422),
    ({"scope": "A1", "effects": [{"effect": "amplify"}]}, 422),
    ({"scope": "A1", "effects": [_amp("fx_1"), _amp("fx_1")]}, 422),
    ({"scope": "A1", "effects": [_amp("fx_taken")]}, 422),
])
def test_put_rack_refuses_bad_requests(env, body, status):
    client, store, _tmp = env
    tl = _project()
    tl.audio_racks = {"MASTER": []}
    next(i for i in tl.items if i.id == "bed_0").audio_effects = [
        AudioEffect(id="fx_taken", effect="amplify")]
    _save(store, tl)
    assert client.put("/api/audio/p1/rack", json=body).status_code == status
    assert _stored(store).revision == 4


def test_put_rack_on_a_missing_project(env):
    client, _store, _tmp = env
    assert client.put("/api/audio/nope/rack", json={"scope": "A1", "effects": []}).status_code == 404


def test_put_rack_may_keep_an_items_own_ids(env):
    """Re-saving a rack with the ids it already has is not a clash."""
    client, store, _tmp = env
    _save(store, _project())
    for _ in range(2):
        resp = client.put("/api/audio/p1/rack",
                          json={"scope": "item:bed_0", "effects": [_amp("fx_bed")]})
        assert resp.status_code == 200
        resp = client.put("/api/audio/p1/rack", json={"scope": "A1", "effects": [_amp("fx_a1")]})
        assert resp.status_code == 200


# --- rack presets (brief A5) ----------------------------------------------------------------

def test_main_registers_the_rack_preset_routes():
    from main import app

    paths = app.openapi()["paths"]
    expected = {
        "/api/audio/rack_presets": {"get", "post"},
        "/api/audio/rack_presets/{name}": {"delete"},
        "/api/audio/{project_id}/rack/apply_preset": {"post"},
    }
    for path, methods in expected.items():
        assert methods <= set(paths.get(path, {})), path


def test_rack_presets_round_trip_through_their_file(env):
    client, _store, tmp = env
    path = tmp / "data" / "audio_rack_presets.json"
    assert client.get("/api/audio/rack_presets").json() == {"presets": {}}

    voice = [_amp("fx_1", 2.0), {"id": "fx_2", "effect": "parametric_eq",
                                 "params": {"hp_on": True}, "enabled": False, "preset": "Podcast Voice"}]
    resp = client.post("/api/audio/rack_presets",
                       json={"name": "Voice", "effects": voice, "note": " for the talk "})
    assert resp.status_code == 200
    client.post("/api/audio/rack_presets", json={"name": "Quiet", "effects": [_amp("fx_q", -6.0)]})

    presets = client.get("/api/audio/rack_presets").json()["presets"]
    assert list(presets) == ["Voice", "Quiet"]
    assert presets["Voice"]["note"] == "for the talk" and presets["Quiet"]["note"] == ""
    assert presets["Voice"]["created"]
    assert [e["id"] for e in presets["Voice"]["effects"]] == ["fx_1", "fx_2"]
    assert presets["Voice"]["effects"][1] == {
        "id": "fx_2", "effect": "parametric_eq", "params": {"hp_on": True}, "enabled": False,
        "start_s": None, "end_s": None, "preset": "Podcast Voice"}
    assert json.loads(path.read_text(encoding="utf-8")) == presets

    # Saving under an existing name replaces that preset.
    client.post("/api/audio/rack_presets", json={"name": "Quiet", "effects": [_amp("fx_q", -12.0)]})
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["Quiet"]["effects"][0]["params"] == {"gain_db": -12.0}

    assert client.delete("/api/audio/rack_presets/Quiet").status_code == 200
    assert list(client.get("/api/audio/rack_presets").json()["presets"]) == ["Voice"]
    assert client.delete("/api/audio/rack_presets/Quiet").status_code == 404
    assert client.delete("/api/audio/rack_presets/Voice").json() == {"presets": {}}
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_rack_presets_refuse_names_the_delete_route_cannot_reach(env):
    client, _store, tmp = env
    for body in ({"name": "a/b", "effects": []}, {"name": "  ", "effects": []},
                 {"name": "x", "effects": [{"effect": "amplify"}]}):
        assert client.post("/api/audio/rack_presets", json=body).status_code == 422
    assert not (tmp / "data" / "audio_rack_presets.json").exists()


def test_apply_rack_preset_replaces_a_rack_with_fresh_ids(env):
    client, store, _tmp = env
    tl = _project()
    next(i for i in tl.items if i.id == "bed_0").audio_effects = [
        AudioEffect(id="fx_bed", effect="amplify")]
    _save(store, tl)
    client.put("/api/audio/p1/rack", json={"scope": "A1", "effects": [_amp("fx_old")]})
    window = _amp("fx_bed", -1.5, start_s=2.0, end_s=4.0, preset="Quiet")
    client.post("/api/audio/rack_presets",
                json={"name": "Voice", "effects": [_amp("fx_1", 3.0), window]})

    resp = client.post("/api/audio/p1/rack/apply_preset", json={"scope": "A1", "name": "Voice"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scope"] == "A1" and body["revision"] == 6
    ids = [e["id"] for e in body["effects"]]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert all(re.fullmatch(r"fx_[0-9a-f]{8}", i) for i in ids)
    assert not set(ids) & {"fx_1", "fx_bed", "fx_old"}
    # Everything but the id comes from the preset, windows and tags included.
    assert [{k: v for k, v in e.items() if k != "id"} for e in body["effects"]] == [
        {"effect": "amplify", "params": {"gain_db": 3.0}, "enabled": True,
         "start_s": None, "end_s": None, "preset": None},
        {"effect": "amplify", "params": {"gain_db": -1.5}, "enabled": True,
         "start_s": 2.0, "end_s": 4.0, "preset": "Quiet"}]

    stored = _stored(store)
    assert [e.id for e in stored.audio_racks["A1"]] == ids          # replaced, not appended
    assert next(i for i in stored.items if i.id == "bed_0").audio_effects[0].id == "fx_bed"
    _inputs, graph, _v, _a = FilterGraphCompiler(stored).compile()
    assert all(f"@{i}_0" in graph for i in ids)

    # Applied again, and to another scope: fresh ids every time, all unique in the project.
    again = client.post("/api/audio/p1/rack/apply_preset", json={"scope": "A1", "name": "Voice"}).json()
    master = client.post("/api/audio/p1/rack/apply_preset",
                         json={"scope": "MASTER", "name": "Voice"}).json()
    clip = client.post("/api/audio/p1/rack/apply_preset",
                       json={"scope": "item:bed_0", "name": "Voice"}).json()
    every = [e["id"] for r in (again, master, clip) for e in r["effects"]]
    assert len(set(every)) == 6 and not set(every) & set(ids)
    assert client.get("/api/audio/p1/racks").json()["items"]["bed_0"] == clip["effects"]
    assert _stored(store).revision == 9


@pytest.mark.parametrize("pid, body, status", [
    ("p1", {"scope": "A1", "name": "Nope"}, 404),
    ("nope", {"scope": "A1", "name": "Voice"}, 404),
    ("p1", {"scope": "V1", "name": "Voice"}, 422),
    ("p1", {"scope": "item:a1_0", "name": "Voice"}, 422),
    ("p1", {"scope": "item:nope", "name": "Voice"}, 404),
    ("p1", {"scope": "A1"}, 422),
])
def test_apply_rack_preset_refuses_bad_requests(env, pid, body, status):
    client, store, _tmp = env
    _save(store, _project())
    client.post("/api/audio/rack_presets", json={"name": "Voice", "effects": [_amp("fx_1")]})
    assert client.post(f"/api/audio/{pid}/rack/apply_preset", json=body).status_code == status
    assert _stored(store).revision == 4


# --- preview ------------------------------------------------------------------------------

def _ffmpeg(*args: str) -> None:
    result = subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", *args],
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-800:]


def _tone(path: Path, frequency: int, seconds: float, gain: float) -> Path:
    """A stereo tone: the preview writes stereo, and a mono source would come
    out 3 dB down from ffmpeg's centre-channel upmix."""
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={seconds}",
            "-af", f"volume={gain}", "-ac", "2", "-c:a", "pcm_s16le", str(path))
    return path


def _wav(path: Path):
    """(seconds, peak 0..1) of a 16-bit WAV."""
    with wave.open(str(path), "rb") as w:
        frames, rate = w.getnframes(), w.getframerate()
        samples = array.array("h", w.readframes(frames))
    return frames / rate, max(abs(s) for s in samples) / 32768.0


@pytest.fixture
def tones(tmp_path):
    """25 s of programme voice (peak ~0.25), its processed version (~0.0625)
    and a music bed (~0.125)."""
    media = tmp_path / "media"
    media.mkdir()
    paths = {"main": _tone(media / "voice.wav", 1000, 25, 2.0),
             "override": _tone(media / "1_processed.wav", 500, 25, 0.5),
             "bed": _tone(media / "bed.wav", 660, 25, 1.0)}
    return paths, {name: _wav(p)[1] for name, p in paths.items()}


def test_preview_of_20_seconds_renders_in_under_3_seconds(env, tones):
    client, store, tmp = env
    paths, peaks = tones
    _save(store, _project(str(paths["main"]), str(paths["bed"])))
    started = time.perf_counter()
    resp = client.post("/api/audio/p1/preview", json={"scope": "A1", "start_s": 2.0, "seconds": 20})
    elapsed = time.perf_counter() - started
    assert resp.status_code == 200, resp.text
    assert elapsed < 3.0
    body = resp.json()
    assert body == {"path": str(tmp / "projects" / "p1" / "audio" / "preview_A1.wav"),
                    "seconds": 20.0}
    seconds, peak = _wav(Path(body["path"]))
    assert seconds == pytest.approx(20.0, abs=0.05)
    assert peak == pytest.approx(peaks["main"], rel=0.03)      # empty rack, no voice master


def test_preview_reads_the_audio_override(env, tones):
    client, store, _tmp = env
    paths, peaks = tones
    tl = _project(str(paths["main"]), str(paths["bed"]))
    tl.sources["main"].audio_override = str(paths["override"])
    _save(store, tl)
    body = client.post("/api/audio/p1/preview",
                       json={"scope": "A1", "start_s": 0, "seconds": 5}).json()
    assert _wav(Path(body["path"]))[1] == pytest.approx(peaks["override"], rel=0.03)


def test_preview_runs_the_saved_or_the_unsaved_rack(env, tones, no_audio_compile):
    client, store, _tmp = env
    paths, peaks = tones
    tl = _project(str(paths["main"]), str(paths["bed"]))
    tl.audio_master = AudioMaster(voice_gain_db=6.0206)          # the voice chain: x2
    _save(store, tl)
    client.put("/api/audio/p1/rack", json={"scope": "A1", "effects": [_amp("fx_a1", -12.0412)]})

    saved = client.post("/api/audio/p1/preview", json={"scope": "A1", "start_s": 0, "seconds": 3})
    assert saved.status_code == 200, saved.text
    # Voice chain x2, then the saved A1 rack x0.25.
    assert _wav(Path(saved.json()["path"]))[1] == pytest.approx(peaks["main"] * 0.5, rel=0.03)

    unsaved = client.post("/api/audio/p1/preview", json={
        "scope": "A1", "start_s": 0, "seconds": 3, "effects": [_amp("fx_try", 0.0)]})
    assert _wav(Path(unsaved.json()["path"]))[1] == pytest.approx(peaks["main"] * 2.0, rel=0.03)
    # Previewing an unsaved rack leaves the saved one alone.
    assert [e["id"] for e in client.get("/api/audio/p1/racks").json()["racks"]["A1"]] == ["fx_a1"]

    master = client.post("/api/audio/p1/preview", json={
        "scope": "MASTER", "start_s": 0, "seconds": 3, "effects": [_amp("fx_m", -6.0206)]})
    assert master.json()["path"].endswith("preview_MASTER.wav")
    assert _wav(Path(master.json()["path"]))[1] == pytest.approx(peaks["main"], rel=0.03)


def test_preview_of_a_clip_runs_only_its_own_rack(env, tones, no_audio_compile):
    client, store, tmp = env
    paths, peaks = tones
    tl = _project(str(paths["main"]), str(paths["bed"]))
    tl.audio_master = AudioMaster(voice_gain_db=6.0206)          # not applied to a clip
    _save(store, tl)
    client.put("/api/audio/p1/rack", json={"scope": "item:bed_0", "effects": [_amp("fx_bed", -6.0206)]})
    resp = client.post("/api/audio/p1/preview",
                       json={"scope": "item:bed_0", "start_s": 1.0, "seconds": 2})
    assert resp.status_code == 200, resp.text
    path = Path(resp.json()["path"])
    assert path == tmp / "projects" / "p1" / "audio" / "preview_item_bed_0.wav"
    seconds, peak = _wav(path)
    assert seconds == pytest.approx(2.0, abs=0.05)
    assert peak == pytest.approx(peaks["bed"] * 0.5, rel=0.03)


def test_preview_refuses_what_it_cannot_play(env, tones):
    client, store, _tmp = env
    paths, _peaks = tones
    _save(store, _project(str(paths["main"]), str(paths["bed"])))
    post = lambda body: client.post("/api/audio/p1/preview", json=body).status_code
    assert post({"scope": "A2", "start_s": 0, "seconds": 2}) == 422
    assert post({"scope": "item:nope", "start_s": 0, "seconds": 2}) == 404
    assert post({"scope": "A1", "source_id": "nope", "start_s": 0, "seconds": 2}) == 404
    assert post({"scope": "A1", "start_s": 0, "seconds": 2,
                 "effects": [{"id": "fx,x", "effect": "amplify"}]}) == 422
    assert post({"scope": "A1"}) == 422                                    # start_s, seconds
    assert client.post("/api/audio/nope/preview",
                       json={"scope": "A1", "start_s": 0, "seconds": 2}).status_code == 404

    _save(store, _project("C:/media/missing.wav", str(paths["bed"])), pid="p2")
    assert client.post("/api/audio/p2/preview",
                       json={"scope": "A1", "start_s": 0, "seconds": 2}).status_code == 404
