"""The process lane and its routes (AUDIO_SUITE_PLAN.md §2.1, brief A2):
versions, `audio_override`, Undo, Normalize / Match Loudness, jobs and the
`/api/audio` analysis, noise and process routes.

The project store and projects directory point into tmp_path; the audio is a
synthetic 1 kHz `sine` gated off in three 2 s gaps over `anoisesrc` white
noise at -50 dBFS RMS.
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio import analysis, noise, processes
from config import FFMPEG_BIN
from store.project_store import ProjectStore
from timeline.schema import SourceFile, Timeline, TimelineItem

GAPS = [(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]
TONES = [(2.0, 4.0), (6.0, 8.0)]
NOISE_DBFS = -50.0
NOISE_AMPLITUDE = 10 ** (NOISE_DBFS / 20) * 3 ** 0.5


def _synth(path: Path) -> Path:
    gate = "+".join(f"between(t,{a:g},{b:g})" for a, b in GAPS)
    graph = (f"[0:a]volume=volume=0:enable='{gate}'[tone];"
             f"[tone][1:a]amix=inputs=2:normalize=0:duration=shortest")
    result = subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=10",
         "-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude={NOISE_AMPLITUDE:.6f}"
                             f":sample_rate=48000:duration=10:seed=11",
         "-filter_complex", graph, "-c:a", "pcm_f32le", str(path)],
        capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-600:]
    return path


def _level(path, start_s: float, end_s: float) -> float:
    samples, _rate = analysis.read_mono(str(path), start_s, end_s)
    return analysis.rms_dbfs(samples)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return _synth(tmp_path_factory.mktemp("synth") / "synth.wav")


@pytest.fixture
def project(tmp_path, monkeypatch, synth):
    projects = tmp_path / "projects"
    projects.mkdir()
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(noise, "project_store", store)
    monkeypatch.setattr(noise, "PROJECTS_DIR", projects)
    tl = Timeline(
        sources={"main": SourceFile(id="main", path=str(synth), duration_seconds=10.0, kind="audio",
                                    width=0, height=0)},
        items=[TimelineItem(id="a1_0", track="A1", source_id="main", source_start_frame=0,
                            source_end_frame=300, timeline_start_frame=0, timeline_end_frame=300)],
        duration_frames=300, revision=5)
    store.save_project("p1", {"id": "p1", "name": "Synthetic", "timeline": tl.model_dump()})
    return {"store": store, "dir": projects / "p1", "synth": synth}


def _source(store: ProjectStore) -> SourceFile:
    return Timeline.model_validate(store.get_project("p1")["timeline"]).sources["main"]


def _revision(store: ProjectStore) -> int:
    return Timeline.model_validate(store.get_project("p1")["timeline"]).revision


def _ebur128_i(path) -> float:
    return analysis.ebur128(str(path))["integrated_lufs"]


# --- the catalogue --------------------------------------------------------------

def test_processes_are_effect_defs_with_process_engine():
    # The five A2 ids, plus whatever registers beside them (voice_conversion, brief S1; the
    # processes_ext entries of brief A3): every entry is an EffectDef with engine "process".
    assert {"noise_reduction", "noise_reduction_ai", "noise_reduction_rnnoise_process", "normalize",
            "match_loudness"} <= set(processes.PROCESSES)
    for key, definition in processes.PROCESSES.items():
        assert isinstance(definition, dict)
        assert definition["id"] == key and definition["engine"] == "process"
        for field in ("name", "category", "description", "params", "presets", "windowable"):
            assert field in definition
        for param in definition["params"]:
            assert {"key", "label", "type", "min", "max", "step", "default", "unit", "live"} <= set(param)
            assert param["live"] is False
        for preset in definition["presets"].values():
            processes.resolve_params(key, preset)
    keys = [p["key"] for p in processes.PROCESSES["noise_reduction"]["params"]]
    assert keys == ["reduction_pct", "reduce_by_db", "spectral_decay_pct", "fft_size", "output_noise_only"]


def test_resolve_params_fills_defaults_clamps_and_refuses():
    values = processes.resolve_params("noise_reduction", {"reduction_pct": 250, "fft_size": "2048"})
    assert values["reduction_pct"] == 100.0 and values["fft_size"] == 2048
    assert values["reduce_by_db"] == 20.0 and values["output_noise_only"] is False
    with pytest.raises(ValueError):
        processes.resolve_params("noise_reduction", {"fft_size": 1000})
    with pytest.raises(ValueError):
        processes.resolve_params("no_such_effect", {})


# --- versions, override and undo ------------------------------------------------

def test_run_process_writes_a_version_sets_the_override_and_undo_restores(project):
    store, synth = project["store"], project["synth"]
    noise.capture_print("p1", None, 0.2, 1.8)
    before = _revision(store)
    progress = []
    result = processes.run_process("p1", None, "noise_reduction", {},
                                   progress=lambda f, m: progress.append(f))
    version_1 = project["dir"] / "audio" / "versions" / "1_noise_reduction.wav"
    assert version_1.exists()
    assert result["audio_override"] == str(version_1) and result["history_len"] == 1
    source = _source(store)
    assert source.audio_override == str(version_1)
    assert source.path == str(synth)                        # the original is untouched
    entry = source.audio_history[-1]
    assert set(entry) == {"path", "effect", "params", "start_s", "end_s", "created"}
    assert entry["effect"] == "noise_reduction" and entry["start_s"] is None
    assert entry["params"]["reduction_pct"] == 100.0
    assert _revision(store) == before + 1
    assert progress and progress[-1] == 1.0 and progress == sorted(progress)
    assert sf.info(str(version_1)).frames == sf.info(str(synth)).frames
    assert _level(version_1, 4.2, 5.8) < NOISE_DBFS - 20.0

    # A range process builds on version 1 and leaves everything outside it alone.
    result = processes.run_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -6.0},
                                   start_s=2.0, end_s=4.0)
    version_2 = project["dir"] / "audio" / "versions" / "2_normalize.wav"
    assert result["audio_override"] == str(version_2) and result["history_len"] == 2
    one, _ = sf.read(str(version_1), dtype="float32")
    two, _ = sf.read(str(version_2), dtype="float32")
    assert one.shape == two.shape
    edge = int(0.006 * 48000)
    assert np.array_equal(one[:96000 - edge], two[:96000 - edge])
    assert np.array_equal(one[192000 + edge:], two[192000 + edge:])
    peak = float(np.max(np.abs(two[96000 + edge:192000 - edge])))
    assert 20 * np.log10(peak) == pytest.approx(-6.0, abs=0.2)

    undone = processes.undo("p1")
    assert undone["undone"]["path"] == str(version_2)
    assert _source(store).audio_override == str(version_1)
    assert len(_source(store).audio_history) == 1
    processes.undo("p1")
    assert _source(store).audio_override is None and _source(store).audio_history == []
    assert processes.undo("p1")["undone"] is None              # nothing left: no change
    assert version_1.exists() and version_2.exists()           # files are kept

    # Numbers are never reused, so no version is overwritten.
    processes.run_process("p1", None, "normalize", {"mode": "rms", "rms_dbfs": -30.0})
    assert (project["dir"] / "audio" / "versions" / "3_normalize.wav").exists()


def test_noise_reduction_needs_a_print(project):
    with pytest.raises(noise.Conflict):
        processes.run_process("p1", None, "noise_reduction", {})
    assert _source(project["store"]).audio_override is None


def test_rnnoise_process_keeps_alignment_and_length(project, tmp_path):
    """arnndn lags by one 10 ms frame; the lane pads and cuts it off. Measured
    on broadband bursts, since a steady tone repeats every 48 samples."""
    from scipy.signal import correlate

    bursts = tmp_path / "bursts.wav"
    analysis.run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi", "-i",
                         "anoisesrc=color=pink:amplitude=0.3:sample_rate=48000:duration=4:seed=3",
                         "-af", "volume='if(between(t,1,1.5)+between(t,2.5,2.7),1,0.02)':eval=frame",
                         "-c:a", "pcm_f32le", str(bursts)])
    out = tmp_path / "rnnoise.wav"
    values = processes.resolve_params("noise_reduction_rnnoise_process", {"mix": 1.0})
    processes._apply("noise_reduction_rnnoise_process", values, bursts, out, "p1", None, None)
    original, _ = sf.read(str(bursts), dtype="float64")
    processed, _ = sf.read(str(out), dtype="float64")
    assert processed.shape == original.shape
    corr = correlate(processed, original, mode="full", method="fft")
    lags = np.arange(-len(original) + 1, len(original))
    near = np.abs(lags) < 4800
    assert int(lags[near][np.argmax(corr[near])]) == 0

    processes.run_process("p1", None, "noise_reduction_rnnoise_process", {"mix": 1.0})
    version = Path(_source(project["store"]).audio_override)
    assert sf.info(str(version)).frames == sf.info(str(project["synth"])).frames


# --- normalize and match loudness -------------------------------------------------

def test_normalize_to_minus_16_lufs_lands_within_half_a_lu(project):
    result = processes.run_process("p1", None, "normalize", {"mode": "lufs", "lufs": -16.0})
    assert _ebur128_i(result["audio_override"]) == pytest.approx(-16.0, abs=0.5)


def test_normalize_to_a_peak(project):
    result = processes.run_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -1.0,
                                                              "dc_remove": True})
    stats = analysis.astats(result["audio_override"])
    assert stats["Peak level dB"] == pytest.approx(-1.0, abs=0.1)


def test_match_loudness_hits_the_target_under_the_ceiling(project):
    result = processes.run_process("p1", None, "match_loudness",
                                   {"target_lufs": -14.0, "true_peak_db": -1.5})
    loudness = analysis.ebur128(result["audio_override"])
    assert loudness["integrated_lufs"] == pytest.approx(-14.0, abs=0.5)
    assert loudness["true_peak_dbtp"] <= -1.3


def test_match_loudness_limits_when_the_ceiling_is_in_the_way(project):
    # -10 LUFS would put the sine's peaks above -9 dBTP: loudnorm's limiter decides,
    # and the true peak back at 48 kHz stays under the ceiling.
    result = processes.run_process("p1", None, "match_loudness",
                                   {"target_lufs": -10.0, "true_peak_db": -9.0})
    assert result["info"]["mode"] == "dynamic"
    assert analysis.ebur128(result["audio_override"])["true_peak_dbtp"] <= -9.0 + 0.05
    assert sf.info(result["audio_override"]).frames == sf.info(str(project["synth"])).frames


# --- preview and jobs -------------------------------------------------------------

def test_preview_process_writes_only_the_preview(project):
    noise.capture_print("p1", None, 0.2, 1.8)
    result = processes.preview_process("p1", None, "noise_reduction", {}, 4.0, 2.0)
    path = project["dir"] / "audio" / "process_preview.wav"
    assert result["path"] == str(path) and path.exists()
    assert sf.info(str(path)).frames == 2 * 48000
    assert _level(path, 0.2, 1.8) < NOISE_DBFS - 20.0
    assert _source(project["store"]).audio_override is None
    assert not (project["dir"] / "audio" / "versions").exists()


def _wait(job_id: str, timeout_s: float = 60.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = processes.get_job(job_id)
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {processes.get_job(job_id)}")


def test_jobs_report_progress_and_failures(project):
    job = _wait(processes.start_job("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -3.0}))
    assert job["status"] == "completed" and job["progress"] == 1.0 and job["error"] is None
    assert job["result"]["history_len"] == 1
    failed = _wait(processes.start_job("p1", None, "noise_reduction", {}))
    assert failed["status"] == "failed" and "noise print" in failed["error"]


# --- routes -----------------------------------------------------------------------

@pytest.fixture
def client(project):
    from routes import audio_process

    app = FastAPI()
    app.include_router(audio_process.router, prefix="/api/audio")
    return TestClient(app)


def test_main_registers_the_process_routes():
    from main import app

    paths = app.openapi()["paths"]
    expected = {
        "/api/audio/processes": {"get"},
        "/api/audio/{project_id}/analyze": {"post"},
        "/api/audio/{project_id}/noise/detect": {"post"},
        "/api/audio/{project_id}/noise/print": {"get", "post"},
        "/api/audio/{project_id}/process": {"post"},
        "/api/audio/{project_id}/process/{job_id}": {"get"},
        "/api/audio/{project_id}/process/preview": {"post"},
        "/api/audio/{project_id}/undo": {"post"},
        "/api/audio/{project_id}/versions": {"get"},
    }
    for path, methods in expected.items():
        assert methods <= set(paths.get(path, {})), path


def test_routes_round_trip(client, project):
    resp = client.get("/api/audio/processes")
    assert resp.status_code == 200 and "noise_reduction" in resp.json()

    resp = client.post("/api/audio/p1/analyze", json={"start_s": 0.0, "end_s": 10.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source_id"] == "main" and len(body["spectrum"]["mean_db"]) == 1025
    png = Path(body["spectrum_png"])
    assert png.exists() and png.parent == project["dir"] / "audio"

    found = client.post("/api/audio/p1/noise/detect", json={}).json()
    assert any(a <= found["start_s"] and found["end_s"] <= b for a, b in GAPS), found

    assert client.get("/api/audio/p1/noise/print").status_code == 404
    assert client.post("/api/audio/p1/process", json={"effect": "noise_reduction"}).status_code == 409

    resp = client.post("/api/audio/p1/noise/print",
                       json={"start_s": found["start_s"], "end_s": found["end_s"]})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["bands_db"]) == 15
    assert client.get("/api/audio/p1/noise/print").json() == resp.json()

    resp = client.post("/api/audio/p1/process/preview",
                       json={"effect": "noise_reduction", "params": {}, "start_s": 4.0, "seconds": 2.0})
    assert resp.status_code == 200, resp.text
    assert Path(resp.json()["path"]).exists()

    resp = client.post("/api/audio/p1/process", json={"effect": "noise_reduction", "params": {}})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    job = None
    for _ in range(1200):
        job = client.get(f"/api/audio/p1/process/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    override = job["result"]["audio_override"]
    assert Path(override).exists()

    listed = client.get("/api/audio/p1/versions").json()
    assert listed["audio_override"] == override and len(listed["versions"]) == 1
    assert client.get("/api/audio/p1/versions", params={"source_id": "main"}).json() == listed

    undone = client.post("/api/audio/p1/undo", json={}).json()
    assert undone["audio_override"] is None and undone["history_len"] == 0


def test_routes_refuse_bad_requests(client):
    assert client.post("/api/audio/p1/process", json={"effect": "nope"}).status_code == 422
    assert client.post("/api/audio/p1/process",
                       json={"effect": "normalize", "start_s": 3.0, "end_s": 2.0}).status_code == 422
    assert client.post("/api/audio/p1/analyze", json={"start_s": 3.0, "end_s": 3.0}).status_code == 422
    assert client.post("/api/audio/p1/noise/detect", json={"source_id": "ghost"}).status_code == 404
    assert client.post("/api/audio/zz/noise/detect", json={}).status_code == 404
    assert client.get("/api/audio/p1/process/audio_process_missing").status_code == 404


def test_routes_refuse_while_a_render_is_running(client, monkeypatch):
    from routes import rendering

    monkeypatch.setitem(rendering.jobs, "render_other", {"id": "render_other", "job_type": "render",
                                                         "status": "running"})
    assert client.get("/api/audio/processes").status_code == 409
    assert client.post("/api/audio/p1/noise/detect", json={}).status_code == 409
    rendering.jobs["render_other"]["status"] = "completed"
    assert client.get("/api/audio/processes").status_code == 200


# --- several ranges in one version (VOICE_CHANGER_PLAN.md brief S2a) ---------------

from types import SimpleNamespace

HALF = int(round(processes.FADE_S * 48000 / 2.0))


def _only_inside(out: np.ndarray, before: np.ndarray, ranges, peak_dbfs: float) -> None:
    """Inside each range (past its crossfade) the peak is `peak_dbfs`; outside every range and
    its crossfade the samples are `before`'s, bit for bit."""
    assert out.shape == before.shape
    untouched = np.ones(before.size, dtype=bool)
    for start_s, end_s in ranges:
        a, b = int(round(start_s * 48000)), int(round(end_s * 48000))
        untouched[a - HALF:b + HALF] = False
        peak = float(np.max(np.abs(out[a + HALF:b - HALF])))
        assert 20 * np.log10(peak) == pytest.approx(peak_dbfs, abs=0.2)
    assert np.array_equal(out[untouched], before[untouched])


def test_run_process_ranges_splices_two_pieces_into_one_version(project):
    store, synth = project["store"], project["synth"]
    original, _ = sf.read(str(synth), dtype="float32")
    progress = []
    result = processes.run_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -6.0},
                                   progress=lambda f, m: progress.append(f),
                                   ranges=[(2.5, 3.5), (6.2, 7.2)])
    version = project["dir"] / "audio" / "versions" / "1_normalize.wav"
    assert result["audio_override"] == str(version) and result["history_len"] == 1
    assert list(version.parent.glob("*.wav")) == [version]                  # one version for both
    entry = _source(store).audio_history[-1]
    assert entry["ranges"] == [[2.5, 3.5], [6.2, 7.2]] and (entry["start_s"], entry["end_s"]) == (2.5, 7.2)
    assert len(result["info"]["ranges"]) == 2                              # each range measured on its own
    out, _ = sf.read(str(version), dtype="float32")
    _only_inside(out, original, [(2.5, 3.5), (6.2, 7.2)], -6.0)
    assert progress[-1] == 1.0 and progress == sorted(progress)

    # one range takes the one-range path: no `ranges` in the entry
    single = processes.run_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -3.0},
                                   ranges=[(2.5, 3.5)])
    assert "ranges" not in single["version"]
    assert (single["version"]["start_s"], single["version"]["end_s"]) == (2.5, 3.5)
    for bad in ([], [(3.0, 2.0)], [(1.0, float("inf"))], [(1.0,)], "2-3"):
        with pytest.raises(ValueError):
            processes.run_process("p1", None, "normalize", {}, ranges=bad)
    with pytest.raises(ValueError):
        processes.start_job("p1", None, "normalize", {}, ranges=[(3.0, 3.0)])
    assert len(_source(store).audio_history) == 2


def test_preview_process_ranges_puts_the_pieces_back_to_back(project):
    result = processes.preview_process("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -6.0},
                                       2.5, 2.0, ranges=[(2.5, 3.5), (6.2, 7.2)])
    path = project["dir"] / "audio" / "process_preview.wav"
    assert result["path"] == str(path) and result["seconds"] == 2.0 and result["start_s"] == 2.5
    assert result["ranges"] == [[2.5, 3.5], [6.2, 7.2]]
    clip, _ = sf.read(str(path), dtype="float32")
    assert clip.size == 2 * 48000
    for k in range(2):
        piece = clip[k * 48000 + HALF:(k + 1) * 48000 - HALF]
        assert 20 * np.log10(float(np.max(np.abs(piece)))) == pytest.approx(-6.0, abs=0.2)
    assert _source(project["store"]).audio_override is None
    assert not (project["dir"] / "audio" / "versions").exists()


def test_start_job_with_ranges_runs_one_version(project):
    job = _wait(processes.start_job("p1", None, "normalize", {"mode": "peak", "peak_dbfs": -6.0},
                                    ranges=[(2.5, 3.5), (6.2, 7.2)]))
    assert job["status"] == "completed", job
    assert job["result"]["history_len"] == 1 and job["result"]["version"]["ranges"] == [[2.5, 3.5], [6.2, 7.2]]


@pytest.fixture
def cut_project(tmp_path, monkeypatch, synth):
    """The synth cut into two A1 clips: source 2.0-3.0 s and 6.0-7.5 s = programme 0-1 s and 1-2.5 s."""
    from routes import audio_process

    projects = tmp_path / "projects"
    projects.mkdir()
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(noise, "project_store", store)
    monkeypatch.setattr(noise, "PROJECTS_DIR", projects)
    items = [TimelineItem(id="a1_0", track="A1", source_id="main", source_start_frame=60, source_end_frame=90,
                          timeline_start_frame=0, timeline_end_frame=30),
             TimelineItem(id="a1_1", track="A1", source_id="main", source_start_frame=180, source_end_frame=225,
                          timeline_start_frame=30, timeline_end_frame=75)]
    tl = Timeline(sources={"main": SourceFile(id="main", path=str(synth), duration_seconds=10.0, kind="audio",
                                              width=0, height=0)},
                  items=items, duration_frames=75, revision=5)
    store.save_project("p1", {"id": "p1", "name": "Cut", "timeline": tl.model_dump()})
    app = FastAPI()
    app.include_router(audio_process.router, prefix="/api/audio")
    return SimpleNamespace(client=TestClient(app), store=store, dir=projects / "p1", synth=synth)


def test_a_programme_range_across_a_cut_processes_both_pieces(cut_project):
    client = cut_project.client
    params = {"mode": "peak", "peak_dbfs": -6.0}
    resp = client.post("/api/audio/p1/process", json={"effect": "normalize", "params": params,
                                                     "program_range": True, "start_s": 0.5, "end_s": 1.5})
    assert resp.status_code == 200, resp.text                                # no 409: start_job takes ranges
    body = resp.json()
    assert body["source_id"] == "main"
    assert [(p["start_s"], p["end_s"]) for p in body["pieces"]] == [pytest.approx((2.5, 3.0)),
                                                                    pytest.approx((6.0, 6.5))]
    job = _wait(body["job_id"])
    assert job["status"] == "completed", job
    assert job["result"]["history_len"] == 1
    original, _ = sf.read(str(cut_project.synth), dtype="float32")
    out, _ = sf.read(job["result"]["audio_override"], dtype="float32")
    _only_inside(out, original, [(2.5, 3.0), (6.0, 6.5)], -6.0)

    resp = client.post("/api/audio/p1/process/preview", json={"effect": "normalize", "params": params,
                                                             "program_range": True, "start_s": 0.5,
                                                             "seconds": 1.0})
    assert resp.status_code == 200, resp.text
    assert resp.json()["seconds"] == 1.0 and len(resp.json()["ranges"]) == 2
