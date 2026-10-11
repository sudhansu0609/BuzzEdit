"""Programme time (AUDIO_SUITE_PLAN.md row L2): `a1_segments` is the render's A1 trim list,
and programme ranges map to source pieces across cuts and back."""

import re

import pytest

from backend.tests.test_audio_compiler import GOLDEN_BUILDERS
from render.compiler import FilterGraphCompiler
from timeline.program_time import a1_segments, program_to_source, source_to_program
from timeline.schema import SourceFile, Timeline, TimelineItem, TrackState

FPS = 30
_TRIM = re.compile(r"^\[(\d+):a\]atrim=(?:start=([\d.]+):end=([\d.]+)|duration=([\d.]+)),.*\[a1_(\d+)\]$")


def compiler_a1_trims(tl: Timeline):
    """(source start, source end) of every `[a1_k]` trim the compiler emits, in k order."""
    inputs, graph, _v, _a = FilterGraphCompiler(tl).compile()
    seek_of, index, pending = {}, 0, 0.0
    for k, arg in enumerate(inputs):
        if arg == "-ss":
            pending = float(inputs[k + 1])
        elif arg == "-i":
            seek_of[index] = pending
            index += 1
            pending = 0.0
    trims = {}
    for statement in graph.split(";"):
        m = _TRIM.match(statement)
        if m:
            src, start, end, duration, k = m.groups()
            if duration is not None:
                start = seek_of[int(src)]
                end = start + float(duration)
            trims[int(k)] = (round(float(start), 3), round(float(end), 3))
    return [trims[k] for k in sorted(trims)]


@pytest.mark.parametrize("name", ["mixed", "seeked", "plain"])
def test_a1_segments_are_the_compilers_a1_trims(name):
    tl = GOLDEN_BUILDERS[name]()
    segments = a1_segments(tl)
    assert [(round(s.src_start_s, 3), round(s.src_end_s, 3)) for s in segments] == compiler_a1_trims(tl)
    assert segments[0].prog_start_s == 0.0
    for before, after in zip(segments, segments[1:]):
        assert after.prog_start_s == pytest.approx(before.prog_end_s)


def cut_timeline(**extra) -> Timeline:
    """Two kept ranges of one 6 s source: 0.5-1.5 s and 4.5-5.5 s (programme 0-1 and 1-2)."""
    src = SourceFile(id="s", path="C:/media/two_tones.wav", duration_seconds=6.0, kind="audio")
    items = [TimelineItem(id=f"a1_{k}", track="A1", source_id="s",
                          source_start_frame=a, source_end_frame=b,
                          timeline_start_frame=k * 30, timeline_end_frame=k * 30 + 30)
             for k, (a, b) in enumerate([(15, 45), (135, 165)])]
    return Timeline(fps_num=FPS, fps_den=1, sources={"s": src}, items=items, duration_frames=60, **extra)


def test_program_to_source_across_a_cut_gives_two_pieces():
    pieces = program_to_source(cut_timeline(), 0.6, 1.4)
    assert [(sid, round(a, 6), round(b, 6)) for sid, _p, a, b in pieces] == [("s", 1.1, 1.5), ("s", 4.5, 4.9)]
    assert [round(b - a, 6) for _s, _p, a, b in pieces] == [0.4, 0.4]


def test_source_to_program_and_back():
    tl = cut_timeline()
    assert source_to_program(tl, "s", 1.2) == pytest.approx(0.7)
    assert source_to_program(tl, "s", 4.75) == pytest.approx(1.25)
    assert source_to_program(tl, "s", 3.0) is None          # cut away
    (_sid, _p, a, _b), = program_to_source(tl, 1.25, 1.3)
    assert a == pytest.approx(4.75)


def test_offset_override_mute_and_muted_track():
    tl = cut_timeline(program_offset_frames=15)
    tl.sources["s"].audio_override = "C:/media/processed.wav"
    segments = a1_segments(tl)
    assert [s.prog_start_s for s in segments] == pytest.approx([0.5, 1.5])
    assert {s.path for s in segments} == {"C:/media/processed.wav"}
    tl.items[0].mute = True                                  # the render leaves it out of the concat
    assert [(s.src_start_s, s.prog_start_s) for s in a1_segments(tl)] == pytest.approx([(4.5, 0.5)])
    tl.tracks["A1"] = TrackState(muted=True)
    assert a1_segments(tl) == []


# -- programme ranges through routes/audio_process.py ------------------------------------

import subprocess

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio import noise, processes
from config import FFMPEG_BIN
from routes import audio_process
from store.project_store import ProjectStore


@pytest.fixture(scope="module")
def two_tones(tmp_path_factory):
    """6 s stereo 48 kHz: 440 Hz for 0-2 s, silence 2-4 s, 880 Hz 4-6 s."""
    path = tmp_path_factory.mktemp("program_time") / "two_tones.wav"
    subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                    "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=2",
                    "-filter_complex", "[1:a]atrim=duration=2[gap];[0:a][gap][2:a]concat=n=3:v=0:a=1",
                    "-ar", "48000", "-ac", "2", str(path)], check=True)
    return path


@pytest.fixture
def api(tmp_path, two_tones, monkeypatch):
    projects = tmp_path / "projects"
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(noise, "project_store", store)
    monkeypatch.setattr(noise, "PROJECTS_DIR", projects)
    tl = cut_timeline()
    tl.sources["s"].path = str(two_tones)
    store.save_project("cut", {"id": "cut", "name": "cut", "source_video": str(two_tones),
                               "timeline": tl.model_dump()})
    app = FastAPI()
    app.include_router(audio_process.router, prefix="/api/audio")
    with TestClient(app) as c:
        yield c


def test_analyze_a_programme_range_hears_both_sides_of_the_cut(api):
    r = api.post("/api/audio/cut/analyze", json={"program_range": True, "start_s": 0.5, "end_s": 1.5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["program_range"] is True
    assert [(p["start_s"], p["end_s"]) for p in body["pieces"]] == [pytest.approx((1.0, 1.5)),
                                                                    pytest.approx((4.5, 5.0))]
    freqs = np.asarray(body["spectrum"]["freqs_hz"])
    db = np.asarray(body["spectrum"]["mean_db"])

    def energy(f):
        return float(db[np.abs(freqs - f) <= 15].max())

    floor = float(np.median(db))
    assert energy(440) > floor + 20 and energy(880) > floor + 20
    assert abs(energy(440) - energy(880)) < 3                  # half a second of each


@pytest.fixture
def fake_jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(processes, "resolve_params", lambda effect, params: dict(params or {}))
    monkeypatch.setattr(processes, "check_available", lambda effect: None)

    def start_job(project_id, source_id, effect, params, start_s=None, end_s=None):
        calls.append((source_id, start_s, end_s))
        return "job_1"

    monkeypatch.setattr(processes, "start_job", start_job)
    return calls


def test_process_on_one_programme_piece_runs_that_source_range(api, fake_jobs):
    r = api.post("/api/audio/cut/process",
                 json={"effect": "normalize", "program_range": True, "start_s": 0.2, "end_s": 0.8})
    assert r.status_code == 200, r.text
    assert fake_jobs == [("s", pytest.approx(0.7), pytest.approx(1.3))]


def test_process_across_a_cut_needs_ranges(api, fake_jobs, monkeypatch):
    body = {"effect": "normalize", "program_range": True, "start_s": 0.6, "end_s": 1.4}
    r = api.post("/api/audio/cut/process", json=body)
    assert r.status_code == 409 and r.json()["detail"] == audio_process.SPANS_A_CUT
    assert fake_jobs == []
    seen = []

    def start_job(project_id, source_id, effect, params, start_s=None, end_s=None, ranges=None):
        seen.append((source_id, ranges))
        return "job_2"

    monkeypatch.setattr(processes, "start_job", start_job)
    r = api.post("/api/audio/cut/process", json=body)
    assert r.status_code == 200, r.text
    (sid, ranges), = seen
    assert sid == "s" and ranges == [pytest.approx((1.1, 1.5)), pytest.approx((4.5, 4.9))]
