"""The Voice panel's backend (VOICE_CHANGER_PLAN.md brief S2a, §2.7): voice regions (PUT checks
and trims them, apply converts them in ONE version and puts their FX on the A1 rack, undo takes it
all back), `program_range` on `/voice/convert`, whole-file and batch conversion into the media
pool.

The seed-vc worker is stubbed in process (`seedvc_env.WORKER`): every convert answers a sine at
22.05 kHz (seed-vc's rate) as long as its input -- 330 Hz in maya_voice, 550 Hz in ravi_voice --
so a converted stretch is easy to tell from the white-noise "speech" around it. The timeline cuts
the 12 s recording into two A1 clips (source 1-4 s and 6-10 s = programme 0-3 s and 3-7 s), so a
programme range across the cut maps to two source pieces. Projects, the voices library and the
job registry live in tmp_path. No GPU, no model.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio import noise, processes, seedvc_env, voice_conversion
from store.project_store import ProjectStore
from timeline.schema import AudioEffect, SourceFile, Timeline, TimelineItem

RATE = 48000
FPS = 30
DURATION_S = 12.0
KEPT = [(1.0, 4.0), (6.0, 10.0)]          # source seconds kept on A1: programme 0-3 s and 3-7 s
PROGRAMME_S = 7.0
TONES = {"maya_voice": 330.0, "ravi_voice": 550.0}
REGION_HALF = int(round(voice_conversion.REGION_FADE_S * RATE / 2.0))     # 15 ms seams
LANE_HALF = int(round(processes.FADE_S * RATE / 2.0))                     # the lane's 10 ms


def _noise(seconds: float, sr: int = RATE, seed: int = 0, level: float = 0.1) -> np.ndarray:
    return (np.random.default_rng(seed).uniform(-1.0, 1.0, int(round(seconds * sr))) * level).astype(np.float32)


def _peak_hz(samples: np.ndarray, sr: int = RATE) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / sr)[int(np.argmax(spectrum))])


def _samples(seconds: float) -> int:
    return int(round(seconds * RATE))


class StubWorker:
    """`seedvc_env.Worker.convert` in process: a sine at 22.05 kHz as long as the input, its
    frequency set by the reference voice. Jobs are recorded; refs in `fail` fail, and `busy`
    makes every job answer SeedVCBusy (a render started)."""

    def __init__(self):
        self.jobs = []
        self.fail = set()
        self.busy = False

    def convert(self, source, ref, out, params=None, progress=None, timeout_s=None):
        stem = Path(str(ref)).stem
        info = sf.info(str(source))
        self.jobs.append({"seconds": round(info.frames / info.samplerate, 3), "ref": stem,
                          "params": dict(params or {})})
        if self.busy:
            raise seedvc_env.SeedVCBusy("A render is running; voice conversion waits for it")
        if progress is not None:
            progress(0.5, "convert")
        if stem in self.fail:
            raise seedvc_env.SeedVCError(f"seed-vc convert failed: the stub refuses {stem}")
        n = int(round(info.frames * 22050 / info.samplerate))
        t = np.arange(n) / 22050.0
        sf.write(str(out), (0.3 * np.sin(2 * np.pi * TONES.get(stem, 330.0) * t)).astype(np.float32), 22050,
                 subtype="FLOAT")
        return {"ok": True, "op": "convert", "out": str(out), "seconds": 0.01, "device": "cuda", "sr": 22050}

    def status(self):
        return {"running": True, "device": "cuda", "idle_s": 0.0}

    def stop(self, wait_s=30.0, reason="stopped"):
        return {"running": False, "device": None, "idle_s": None}


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture
def library(tmp_path, monkeypatch):
    """data/ in tmp_path: the voices library starts empty."""
    monkeypatch.setattr(seedvc_env, "DATA_DIR", tmp_path / "data")
    return tmp_path / "data" / "voices"


@pytest.fixture
def worker(monkeypatch, library):
    stub = StubWorker()
    monkeypatch.setattr(seedvc_env, "WORKER", stub)
    monkeypatch.setattr(seedvc_env, "resolve", lambda: {
        "python": "python", "seedvc_dir": "stub", "cli": "cli.py", "source": "test", "reason": ""})
    return stub


def _voice_clip(path: Path, seconds: float, sr: int = 16000, seed: int = 1) -> Path:
    sf.write(str(path), _noise(seconds, sr, seed, 0.2), sr, subtype="PCM_16")
    return path


@pytest.fixture
def project(tmp_path, monkeypatch, library):
    """A 12 s white-noise recording cut into two A1 clips, a user effect on the A1 rack, a 3 s
    clip in the media pool, two voices in the library."""
    projects = tmp_path / "projects"
    projects.mkdir()
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(noise, "project_store", store)
    monkeypatch.setattr(noise, "PROJECTS_DIR", projects)
    recording = tmp_path / "recording.wav"
    sf.write(str(recording), _noise(DURATION_S, seed=7), RATE, subtype="FLOAT")
    clip = tmp_path / "clip.wav"
    sf.write(str(clip), _noise(3.0, seed=9), RATE, subtype="FLOAT")
    items, at = [], 0
    for k, (a, b) in enumerate(KEPT):
        length = int(round((b - a) * FPS))
        items.append(TimelineItem(id=f"a1_{k}", track="A1", source_id="main",
                                  source_start_frame=int(round(a * FPS)), source_end_frame=int(round(b * FPS)),
                                  timeline_start_frame=at, timeline_end_frame=at + length))
        at += length
    tl = Timeline(
        fps_num=FPS, fps_den=1,
        sources={"main": SourceFile(id="main", path=str(recording), duration_seconds=DURATION_S,
                                    kind="audio", width=0, height=0)},
        items=items, duration_frames=at, revision=3,
        audio_racks={"A1": [AudioEffect(id="fx_user0001", effect="parametric_eq", params={})]})
    pool = [{"id": "media_clip", "name": "clip.wav", "path": str(clip), "source_path": str(clip),
             "kind": "audio", "duration": 3.0, "width": 0, "height": 0, "fps": 0.0, "has_audio": True,
             "size_bytes": 0, "linked": True}]
    store.save_project("p1", {"id": "p1", "name": "Cut story", "timeline": tl.model_dump(), "settings": {},
                              "media_pool": pool})
    voice_conversion.add_voice("maya_voice", _voice_clip(tmp_path / "maya.wav", 4.0, seed=2), "woman", "test")
    voice_conversion.add_voice("ravi_voice", _voice_clip(tmp_path / "ravi.wav", 5.0, seed=3), "demon", "test")
    original, _ = sf.read(str(recording), dtype="float32")
    return SimpleNamespace(store=store, recording=recording, clip=clip, original=original, dir=projects / "p1")


def _timeline(store: ProjectStore) -> Timeline:
    return Timeline.model_validate(store.get_project("p1")["timeline"])


def _regions(store: ProjectStore):
    return store.get_project("p1")["settings"]["voice_regions"]


def _assert_only(version: np.ndarray, before: np.ndarray, pieces, half: int):
    """Outside every source piece (and its crossfade) sample for sample `before`; inside, the
    stub's tone of that piece's voice."""
    assert version.shape == before.shape
    untouched = np.ones(before.size, dtype=bool)
    for start_s, end_s, hz in pieces:
        a, b = _samples(start_s), _samples(end_s)
        untouched[a - half:b + half] = False
        inside = version[a + half:b - half]
        assert abs(_peak_hz(inside) - hz) < 3.0, (start_s, end_s, _peak_hz(inside))
        assert np.sqrt(np.mean(inside ** 2)) == pytest.approx(0.3 / np.sqrt(2.0), rel=0.05)
    assert np.array_equal(version[untouched], before[untouched])


REVERB = {"effect": "reverb", "params": {"room_size": 40.0, "wet": 20.0}}
DEMON_FX = [{"effect": "pitch_shifter", "params": {"semitones": -4.0}},
            {"id": "vc_demon_distortion", "effect": "distortion", "params": {"drive": 10.5}, "enabled": False}]


# --- regions: PUT ---------------------------------------------------------------------------

def test_put_regions_checks_clamps_and_trims_overlaps(project):
    before = _timeline(project.store).revision
    got = voice_conversion.set_regions("p1", [
        {"id": "late", "start_s": 5.0, "end_s": 6.5, "voice": "ravi_voice"},
        {"id": "r1", "start_s": 0.5, "end_s": 1.5, "voice": "maya_voice", "params": {"steps": 20, "extra": 1},
         "fx": [REVERB]},
        {"id": "r2", "start_s": 1.2, "end_s": 2.0, "voice": "", "preset": "Demon", "status": "failed",
         "error": "earlier"},
        {"id": "inside", "start_s": 1.6, "end_s": 1.9, "voice": "maya_voice"},      # covered by r1 + r2
        {"start_s": 6.8, "end_s": 9.0, "voice": "ravi_voice", "params": {"match_pitch": "yes"}},
        {"id": "gone", "start_s": 8.0, "end_s": 9.0, "voice": "ravi_voice"},       # after the programme
    ])["regions"]
    assert [(r["start_s"], r["end_s"]) for r in got] == [(0.5, 1.5), (1.5, 2.0), (5.0, 6.5), (6.8, PROGRAMME_S)]
    assert [r["id"] for r in got[:3]] == ["r1", "r2", "late"] and got[3]["id"].startswith("vr_")
    r1, r2, _late, last = got
    assert r1["params"] == {"steps": 20.0} and r1["status"] == "pending" and r1["preset"] is None
    assert r1["fx"][0]["effect"] == "reverb" and r1["fx"][0]["params"] == REVERB["params"]
    assert set(r1["fx"][0]) == {"id", "effect", "params", "enabled", "start_s", "end_s", "preset"}
    assert r2["voice"] == "" and r2["preset"] == "Demon" and r2["status"] == "failed" and r2["error"] == "earlier"
    assert last["params"] == {"match_pitch": True}
    assert _regions(project.store) == got
    assert voice_conversion.get_regions("p1") == {"regions": got}
    assert _timeline(project.store).revision == before + 1

    bad = [
        [{"start_s": "soon", "end_s": 2.0, "voice": "maya_voice"}],
        [{"start_s": 1.0, "end_s": float("nan"), "voice": "maya_voice"}],
        [{"start_s": 1.0, "end_s": 2.0}],                                          # no voice, no preset
        [{"start_s": 1.0, "end_s": 2.0, "voice": "maya_voice", "status": "done"}],
        [{"start_s": 1.0, "end_s": 2.0, "voice": "maya_voice", "fx": [{"params": {}}]}],
        [{"start_s": 1.0, "end_s": 2.0, "voice": "maya_voice", "params": {"steps": "many"}}],
        [{"id": "x", "start_s": 1.0, "end_s": 2.0, "voice": "maya_voice"},
         {"id": "x", "start_s": 3.0, "end_s": 4.0, "voice": "maya_voice"}],
        [{"id": "../x", "start_s": 1.0, "end_s": 2.0, "voice": "maya_voice"}],
        "not a list",
    ]
    for regions in bad:
        with pytest.raises(ValueError):
            voice_conversion.set_regions("p1", regions)
    assert _regions(project.store) == got
    assert voice_conversion.set_regions("p1", [])["regions"] == []


# --- regions: apply and undo ----------------------------------------------------------------

def test_apply_two_regions_one_across_a_cut_in_one_version_and_undo(project, worker):
    # an earlier version (the lane, 10 ms): the regions build on it and undo goes back to it
    earlier = voice_conversion.convert_range("p1", None, 0.2, 0.6, {"voice": "maya_voice"})
    v1 = earlier["audio_override"]
    before, _rate = sf.read(v1, dtype="float32")
    voice_conversion.set_regions("p1", [
        {"id": "r1", "start_s": 0.5, "end_s": 1.5, "voice": "maya_voice", "fx": [REVERB]},
        {"id": "r2", "start_s": 2.5, "end_s": 4.0, "voice": "ravi_voice", "preset": "Demon",
         "params": {"steps": 20}, "fx": DEMON_FX},
    ])
    worker.jobs.clear()
    versions_dir = project.dir / "audio" / "versions"
    files_before = set(versions_dir.glob("*.wav"))
    revision = _timeline(project.store).revision
    progress = []
    result = voice_conversion.apply_regions("p1", None, lambda f, m: progress.append(f))

    assert result["applied"] == ["r1", "r2"] and result["failed"] == []
    version = result["version"]
    assert set(versions_dir.glob("*.wav")) - files_before == {Path(version["path"])}     # exactly one
    assert version["voice_regions"] == ["r1", "r2"] and version["effect"] == "voice_conversion"
    tl = _timeline(project.store)
    source = tl.sources["main"]
    assert len(source.audio_history) == 2 and source.audio_override == version["path"]
    assert tl.revision == revision + 1
    converted, rate = sf.read(version["path"], dtype="float32")
    assert rate == RATE
    # programme 0.5-1.5 = source 1.5-2.5; programme 2.5-4.0 crosses the cut: source 3.5-4.0 + 6.0-7.0
    _assert_only(converted, before, [(1.5, 2.5, 330.0), (3.5, 4.0, 550.0), (6.0, 7.0, 550.0)], REGION_HALF)
    assert progress[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in progress)

    assert [(job["ref"], job["seconds"]) for job in worker.jobs] == [
        ("maya_voice", 2.0), ("ravi_voice", 1.5), ("ravi_voice", 2.0)]          # each piece + 0.5 s either side
    demon = worker.jobs[1]["params"]
    assert demon["pitch_shift"] == -5 and demon["f0_condition"] is True and demon["diffusion_steps"] == 20
    assert demon["length_adjust"] == 1.0

    rack = tl.audio_racks["A1"]
    assert rack[0].id == "fx_user0001" and rack[0].preset is None
    tagged = [(e.effect, e.preset, e.start_s, e.end_s, e.enabled) for e in rack[1:]]
    assert tagged == [("reverb", "voice_region:r1", 0.5, 1.5, True),
                      ("pitch_shifter", "voice_region:r2", 2.5, 4.0, True),
                      ("distortion", "voice_region:r2", 2.5, 4.0, False)]
    ids = [e.id for e in rack]
    assert len(set(ids)) == len(ids) and all(i.startswith("fx_") for i in ids)
    assert rack[3].params == {"drive": 10.5}
    regions = _regions(project.store)
    assert [(r["id"], r["status"], r.get("applied_version")) for r in regions] == [
        ("r1", "applied", version["path"]), ("r2", "applied", version["path"])]

    undone = voice_conversion.undo_regions("p1")
    assert [(r["id"], r["status"]) for r in undone["regions"]] == [("r1", "pending"), ("r2", "pending")]
    assert all("applied_version" not in r and "error" not in r for r in undone["regions"])
    tl = _timeline(project.store)
    assert tl.sources["main"].audio_override == v1 and len(tl.sources["main"].audio_history) == 1
    assert [(e.id, e.preset) for e in tl.audio_racks["A1"]] == [("fx_user0001", None)]
    assert _regions(project.store) == undone["regions"]
    assert Path(version["path"]).exists()                       # version files are kept
    with pytest.raises(noise.Conflict):                        # the newest version is not a regions apply
        voice_conversion.undo_regions("p1")
    processes.undo("p1")
    with pytest.raises(noise.Conflict):
        voice_conversion.undo_regions("p1")


def test_a_failed_region_keeps_the_original_and_is_flagged(project, worker):
    voice_conversion.set_regions("p1", [
        {"id": "ok", "start_s": 0.5, "end_s": 1.0, "voice": "ravi_voice", "fx": [REVERB]},
        {"id": "refused", "start_s": 2.0, "end_s": 2.5, "voice": "maya_voice", "fx": [REVERB]},
        {"id": "ghost", "start_s": 5.0, "end_s": 5.5, "voice": "nobody"},
    ])
    worker.fail = {"maya_voice"}
    result = voice_conversion.apply_regions("p1")
    assert result["applied"] == ["ok"]
    assert [f["id"] for f in result["failed"]] == ["refused", "ghost"]
    assert "refuses maya_voice" in result["failed"][0]["error"]
    assert "not in the voices library" in result["failed"][1]["error"]
    converted, _rate = sf.read(result["version"]["path"], dtype="float32")
    _assert_only(converted, project.original, [(1.5, 2.0, 550.0)], REGION_HALF)    # source 3.0-3.5 untouched
    regions = {r["id"]: r for r in _regions(project.store)}
    assert regions["ok"]["status"] == "applied"
    assert regions["refused"]["status"] == "failed" and "refuses" in regions["refused"]["error"]
    assert regions["ghost"]["status"] == "failed" and "applied_version" not in regions["ghost"]
    assert [e.preset for e in _timeline(project.store).audio_racks["A1"]] == [None, "voice_region:ok"]

    # nothing pending now; a listed (failed) region runs again once its voice converts
    with pytest.raises(noise.Conflict):
        voice_conversion.preflight_regions("p1")
    with pytest.raises(noise.NotFound):
        voice_conversion.preflight_regions("p1", ["missing"])
    assert voice_conversion.apply_regions("p1") == {"applied": [], "failed": [], "version": None}
    worker.fail = set()
    again = voice_conversion.apply_regions("p1", ["refused"])
    assert again["applied"] == ["refused"] and again["failed"] == []
    assert _timeline(project.store).sources["main"].audio_history[-1]["voice_regions"] == ["refused"]
    assert {r["id"]: r["status"] for r in _regions(project.store)} == {
        "ok": "applied", "refused": "applied", "ghost": "failed"}


def test_when_no_region_converts_nothing_is_written(project, worker):
    voice_conversion.set_regions("p1", [{"id": "a", "start_s": 0.5, "end_s": 1.0, "voice": "maya_voice"}])
    worker.fail = {"maya_voice"}
    result = voice_conversion.apply_regions("p1")
    assert result["applied"] == [] and result["version"] is None and result["failed"][0]["id"] == "a"
    assert not list((project.dir / "audio" / "versions").glob("*.wav"))
    assert _timeline(project.store).sources["main"].audio_override is None
    assert _regions(project.store)[0]["status"] == "failed"

    # an engine that cannot run (a render started) aborts: nothing written, statuses unchanged
    voice_conversion.set_regions("p1", [{"id": "b", "start_s": 0.5, "end_s": 1.0, "voice": "maya_voice"}])
    worker.fail, worker.busy = set(), True
    with pytest.raises(seedvc_env.SeedVCBusy):
        voice_conversion.apply_regions("p1")
    assert _regions(project.store)[0]["status"] == "pending"
    assert _timeline(project.store).sources["main"].audio_override is None


# --- whole files into the media pool --------------------------------------------------------

def _pool(store: ProjectStore):
    return {m["id"]: m for m in store.get_project("p1")["media_pool"]}


def test_convert_file_registers_a_media_item_of_the_same_length(project, worker):
    progress = []
    got = voice_conversion.convert_file("p1", media_id="media_clip", voice="maya_voice",
                                        progress=lambda f, m: progress.append(f))
    out = Path(got["path"])
    assert out.parent == project.dir / "audio" / "converted" and out.name == "clip_maya_voice.wav"
    assert got["seconds"] == 3.0
    info = sf.info(str(out))
    assert (info.samplerate, info.channels, info.frames) == (RATE, 1, 3 * RATE)
    samples, _rate = sf.read(str(out), dtype="float32")
    assert abs(_peak_hz(samples) - 330.0) < 3.0
    entry = _pool(project.store)[got["media_id"]]
    assert entry["path"] == str(out) and entry["kind"] == "audio" and entry["linked"] is False
    assert entry["duration"] == pytest.approx(3.0, abs=0.01) and entry["name"] == "clip_maya_voice.wav"
    assert "media_clip" in _pool(project.store)                     # the input stays
    assert progress[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in progress)

    # the project source (as it sounds now), named; the same name again gets a number
    whole = voice_conversion.convert_file("p1", voice="", preset="Demon", params={"voice": "ravi_voice"},
                                          name="Whole take")
    assert whole["seconds"] == DURATION_S and Path(whole["path"]).name == "Whole take.wav"
    assert _pool(project.store)[whole["media_id"]]["name"] == "Whole take"
    assert worker.jobs[-1]["ref"] == "ravi_voice" and worker.jobs[-1]["params"]["pitch_shift"] == -5
    again = voice_conversion.convert_file("p1", path=str(project.clip), voice="maya_voice", name="Whole take")
    assert Path(again["path"]).name == "Whole take_2.wav"
    assert len(_pool(project.store)) == 4

    with pytest.raises(noise.NotFound):
        voice_conversion.convert_file("p1", media_id="media_missing", voice="maya_voice")
    with pytest.raises(ValueError):
        voice_conversion.convert_file("p1", media_id="media_clip", path=str(project.clip), voice="maya_voice")
    with pytest.raises(ValueError, match="not in the voices library"):
        voice_conversion.convert_file("p1", media_id="media_clip", voice="nobody")
    assert len(_pool(project.store)) == 4


def test_batch_reports_per_item(project, worker):
    second = project.clip.with_name("second.wav")
    sf.write(str(second), _noise(2.0, seed=11), RATE, subtype="FLOAT")
    p_data = project.store.get_project("p1")
    p_data["media_pool"].append({**p_data["media_pool"][0], "id": "media_second", "name": "second.wav",
                                 "path": str(second), "source_path": str(second), "duration": 2.0})
    project.store.save_project("p1", p_data)
    progress = []
    got = voice_conversion.batch("p1", ["media_clip", "media_missing", "media_second"], voice="ravi_voice",
                                 progress=lambda f, m: progress.append((f, m)))
    first, missing, last = got["items"]
    assert first["media_id"] == "media_clip" and Path(first["path"]).name == "clip_ravi_voice.wav"
    assert missing["media_id"] == "media_missing" and "media_missing" in missing["error"]
    assert set(missing) == {"media_id", "error"}
    assert last["media_id"] == "media_second" and sf.info(last["path"]).frames == 2 * RATE
    pool = _pool(project.store)
    assert pool[first["out_media_id"]]["duration"] == pytest.approx(3.0, abs=0.01)
    assert pool[last["out_media_id"]]["duration"] == pytest.approx(2.0, abs=0.01)
    assert progress[-1][0] == 1.0 and [f for f, _m in progress] == sorted(f for f, _m in progress)
    assert any(m.startswith("3/3 media_second") for _f, m in progress)

    # a busy engine stops the batch: the items it did not reach carry its error, untried
    worker.jobs.clear()
    worker.busy = True
    stopped = voice_conversion.batch("p1", ["media_clip", "media_second"], voice="ravi_voice")["items"]
    assert [item["media_id"] for item in stopped] == ["media_clip", "media_second"]
    assert all("render is running" in item["error"] for item in stopped) and len(worker.jobs) == 1
    with pytest.raises(ValueError):
        voice_conversion.batch("p1", [], voice="ravi_voice")
    with pytest.raises(ValueError):
        voice_conversion.batch("p1", ["media_clip"], voice="nobody")


# --- routes -------------------------------------------------------------------------------------

@pytest.fixture
def client(project, worker):
    from routes import audio_process, audio_voice
    app = FastAPI()
    app.include_router(audio_process.router, prefix="/api/audio")
    app.include_router(audio_voice.router, prefix="/api/audio")
    return TestClient(app)


def _wait(client, job_id):
    for _ in range(600):
        job = client.get(f"/api/audio/p1/process/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_convert_with_program_range_maps_across_a_cut(client, project, worker):
    resp = client.post("/api/audio/p1/voice/convert", json={
        "start_s": 2.5, "end_s": 4.0, "program_range": True, "voice": "maya_voice", "params": {"steps": 12}})
    assert resp.status_code == 200, resp.text
    assert set(resp.json()) == {"job_id"}
    job = _wait(client, resp.json()["job_id"])
    assert job["status"] == "completed", job
    version = job["result"]["version"]
    assert version["ranges"] == [[3.5, 4.0], [6.0, 7.0]] and (version["start_s"], version["end_s"]) == (3.5, 7.0)
    assert job["result"]["history_len"] == 1
    converted, _rate = sf.read(job["result"]["audio_override"], dtype="float32")
    _assert_only(converted, project.original, [(3.5, 4.0, 330.0), (6.0, 7.0, 330.0)], LANE_HALF)
    assert [job["params"]["diffusion_steps"] for job in worker.jobs] == [12, 12]

    # one piece: that source range, as S1 ran it
    job = _wait(client, client.post("/api/audio/p1/voice/convert", json={
        "start_s": 0.5, "end_s": 1.5, "program_range": True, "params": {"voice": "ravi_voice"}}).json()["job_id"])
    assert job["status"] == "completed", job
    assert (job["result"]["version"]["start_s"], job["result"]["version"]["end_s"]) == (1.5, 2.5)
    assert "ranges" not in job["result"]["version"]
    converted, _rate = sf.read(job["result"]["audio_override"], dtype="float32")
    _assert_only(converted, project.original,
                 [(1.5, 2.5, 550.0), (3.5, 4.0, 330.0), (6.0, 7.0, 330.0)], LANE_HALF)

    assert client.post("/api/audio/p1/voice/convert", json={
        "start_s": 7.5, "end_s": 9.0, "program_range": True, "voice": "maya_voice"}).status_code == 422
    assert client.post("/api/audio/p1/voice/convert", json={
        "start_s": 2.0, "end_s": 1.0, "program_range": True, "voice": "maya_voice"}).status_code == 422


def test_region_routes_end_to_end(client, project, worker):
    assert client.get("/api/audio/p1/voice/regions").json() == {"regions": []}
    assert client.post("/api/audio/p1/voice/regions/apply").status_code == 409          # nothing pending
    resp = client.put("/api/audio/p1/voice/regions", json={"regions": [
        {"id": "r1", "start_s": 0.5, "end_s": 1.5, "voice": "maya_voice", "fx": [REVERB]},
        {"id": "r2", "start_s": 1.0, "end_s": 4.0, "voice": "ravi_voice"}]})
    assert resp.status_code == 200, resp.text
    regions = resp.json()["regions"]
    assert [(r["id"], r["start_s"], r["end_s"], r["status"]) for r in regions] == [
        ("r1", 0.5, 1.5, "pending"), ("r2", 1.5, 4.0, "pending")]
    assert client.get("/api/audio/p1/voice/regions").json() == {"regions": regions}
    assert client.put("/api/audio/p1/voice/regions",
                      json={"regions": [{"start_s": 1.0, "end_s": 2.0}]}).status_code == 422
    assert client.put("/api/audio/p1/voice/regions", json={}).status_code == 422
    assert client.post("/api/audio/p1/voice/regions/apply", json={"ids": ["nope"]}).status_code == 404
    assert client.post("/api/audio/p1/voice/regions/undo").status_code == 409            # nothing applied

    job = _wait(client, client.post("/api/audio/p1/voice/regions/apply").json()["job_id"])
    assert job["status"] == "completed", job
    assert set(job["result"]) == {"applied", "failed", "version"}
    assert job["result"]["applied"] == ["r1", "r2"] and job["result"]["failed"] == []
    converted, _rate = sf.read(job["result"]["version"]["path"], dtype="float32")
    # r2 (programme 1.5-4.0, trimmed by r1) = source 2.5-4.0 and, across the cut, 6.0-7.0
    _assert_only(converted, project.original, [(1.5, 2.5, 330.0), (2.5, 4.0, 550.0), (6.0, 7.0, 550.0)],
                 REGION_HALF)
    statuses = [r["status"] for r in client.get("/api/audio/p1/voice/regions").json()["regions"]]
    assert statuses == ["applied", "applied"]
    racks = _timeline(project.store).audio_racks["A1"]
    assert [(e.effect, e.preset, e.start_s, e.end_s) for e in racks[1:]] == [("reverb", "voice_region:r1", 0.5, 1.5)]

    undone = client.post("/api/audio/p1/voice/regions/undo")
    assert undone.status_code == 200, undone.text
    assert [r["status"] for r in undone.json()["regions"]] == ["pending", "pending"]
    assert _timeline(project.store).sources["main"].audio_override is None
    assert [e.id for e in _timeline(project.store).audio_racks["A1"]] == ["fx_user0001"]

    # listed ids only
    job = _wait(client, client.post("/api/audio/p1/voice/regions/apply", json={"ids": ["r2"]}).json()["job_id"])
    assert job["status"] == "completed" and job["result"]["applied"] == ["r2"]


def test_convert_file_and_batch_routes(client, project, worker):
    resp = client.post("/api/audio/p1/voice/convert_file",
                       json={"media_id": "media_clip", "voice": "maya_voice", "name": "Maya line"})
    assert resp.status_code == 200, resp.text
    job = _wait(client, resp.json()["job_id"])
    assert job["status"] == "completed", job
    assert set(job["result"]) == {"media_id", "path", "seconds"} and job["result"]["seconds"] == 3.0
    assert _pool(project.store)[job["result"]["media_id"]]["name"] == "Maya line"
    assert client.post("/api/audio/p1/voice/convert_file",
                       json={"media_id": "media_missing", "voice": "maya_voice"}).status_code == 404
    assert client.post("/api/audio/p1/voice/convert_file",
                       json={"media_id": "media_clip", "voice": "nobody"}).status_code == 422

    resp = client.post("/api/audio/p1/voice/batch",
                       json={"media_ids": ["media_clip", "media_missing"], "preset": "Demon",
                             "params": {"voice": "ravi_voice"}})
    assert resp.status_code == 200, resp.text
    job = _wait(client, resp.json()["job_id"])
    assert job["status"] == "completed", job
    good, bad = job["result"]["items"]
    assert good["media_id"] == "media_clip" and good["out_media_id"] in _pool(project.store)
    assert bad["media_id"] == "media_missing" and "error" in bad
    assert client.post("/api/audio/p1/voice/batch", json={"media_ids": [], "voice": "maya_voice"}).status_code == 422
    assert client.post("/api/audio/zz/voice/batch",
                       json={"media_ids": ["media_clip"], "voice": "maya_voice"}).status_code == 404


def test_new_routes_answer_503_without_seed_vc_and_409_during_a_render(client, project, monkeypatch):
    voice_conversion.set_regions("p1", [{"id": "r1", "start_s": 0.5, "end_s": 1.5, "voice": "maya_voice"}])
    monkeypatch.setattr(seedvc_env, "resolve", lambda: {
        "python": None, "seedvc_dir": None, "cli": "cli.py", "source": "unavailable", "reason": "no seed-vc found"})
    assert client.post("/api/audio/p1/voice/regions/apply").status_code == 503
    assert client.post("/api/audio/p1/voice/convert_file",
                       json={"media_id": "media_clip", "voice": "maya_voice"}).status_code == 503
    assert client.post("/api/audio/p1/voice/batch",
                       json={"media_ids": ["media_clip"], "voice": "maya_voice"}).status_code == 503
    assert client.post("/api/audio/p1/voice/convert", json={
        "start_s": 2.5, "end_s": 4.0, "program_range": True, "voice": "maya_voice"}).status_code == 503

    from routes import rendering
    monkeypatch.setitem(rendering.jobs, "render_r", {"id": "render_r", "job_type": "render", "status": "running"})
    for method, path in [("get", "/api/audio/p1/voice/regions"), ("post", "/api/audio/p1/voice/regions/undo"),
                         ("post", "/api/audio/p1/voice/regions/apply"), ("post", "/api/audio/p1/voice/batch")]:
        assert getattr(client, method)(path).status_code == 409, path
