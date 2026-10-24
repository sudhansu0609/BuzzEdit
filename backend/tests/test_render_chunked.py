"""Chunk planning never cuts through anything genuinely time-dependent
(transitions, children, animated atmosphere), a chunk's timeline is the
programme's picture for that stretch starting at frame 0, and a boundary that
does cross an item's own animation or colour fades (Ken Burns, adjustment
layers) keeps that move's phase continuous across the cut."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from render import chunked as chunked_mod
from render.chunked import (
    CHUNK_TARGET_S,
    ChunkStalledError,
    _follow,
    plan_chunks,
    sub_timeline,
)
from render.runner import _render_canvas
from timeline.schema import (
    AtmosphereEffect,
    ColorGrade,
    SourceFile,
    Timeline,
    TimelineItem,
    Transform,
    Transition,
)

FPS = 24
PROJECT_25EFF765 = Path(__file__).resolve().parent.parent.parent / "data" / "projects" / "25eff765.json"


def _tl(extra=()):
    total = 300 * FPS
    sources = {
        "main": SourceFile(id="main", path="main.mp4", duration_seconds=300, width=3840,
                           height=2160, fps_num=24, fps_den=1, has_audio=True, kind="video"),
        "img": SourceFile(id="img", path="b.png", duration_seconds=5, width=1024, height=576,
                          fps_num=24, fps_den=1, has_audio=False, kind="image"),
    }
    items = [
        TimelineItem(id="v1", track="V1", source_id="main", source_start_frame=0,
                     source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total),
        TimelineItem(id="a1", track="A1", source_id="main", source_start_frame=0,
                     source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total),
        *extra,
    ]
    return Timeline(fps_num=24, fps_den=1, width=3840, height=2160, duration_frames=total,
                    sources=sources, items=items)


def test_chunks_cover_the_programme_without_gaps():
    tl = _tl()
    chunks = plan_chunks(tl)
    assert chunks[0][0] == 0 and chunks[-1][1] == tl.duration_frames
    assert all(a < b for a, b in chunks)
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
    assert len(chunks) >= 5


def test_a_boundary_may_now_cross_a_splittable_animated_overlay():
    """A plain animated overlay (no children, no transition, no atmosphere) is
    splittable since sub_timeline re-anchors its move across the cut, so a
    boundary landing inside it is no longer something plan_chunks avoids."""
    zoom = TimelineItem(id="kb", track="V3", source_id="img", source_start_frame=0,
                        source_end_frame=20 * FPS, timeline_start_frame=35 * FPS,
                        timeline_end_frame=55 * FPS,
                        transform=Transform(scale=1.0, scale_end=1.2))
    chunks = plan_chunks(_tl([zoom]))
    assert any(35 * FPS < a < 55 * FPS for a, _b in chunks[1:])
    # And the split itself is accepted rather than raising.
    a, _b = next((a, b) for a, b in chunks if 35 * FPS < a < 55 * FPS)
    sub_timeline(_tl([zoom]), 0, a)
    sub_timeline(_tl([zoom]), a, _tl([zoom]).duration_frames)


@pytest.mark.parametrize("make_unsplittable", [
    lambda item: item.model_copy(update={"transition": Transition()}),
    lambda item: item.model_copy(update={"children": [TimelineItem(
        id="kb_child", track="V3", timeline_start_frame=0, timeline_end_frame=1)]}),
    lambda item: item.model_copy(update={"atmosphere": [AtmosphereEffect(type="rain")]}),
])
def test_no_boundary_inside_a_genuinely_unsplittable_animated_overlay(make_unsplittable):
    zoom = TimelineItem(id="kb", track="V3", source_id="img", source_start_frame=0,
                        source_end_frame=20 * FPS, timeline_start_frame=35 * FPS,
                        timeline_end_frame=55 * FPS,
                        transform=Transform(scale=1.0, scale_end=1.2))
    zoom = make_unsplittable(zoom)
    chunks = plan_chunks(_tl([zoom]))
    for a, _b in chunks[1:]:
        assert not (35 * FPS < a < 55 * FPS)


def test_split_adjustment_layer_is_accepted_and_keeps_its_phase():
    adj = TimelineItem(id="mood", track="V8", kind="adjustment",
                       timeline_start_frame=35 * FPS, timeline_end_frame=95 * FPS,
                       transform=Transform(scale=1.0, scale_end=1.04),
                       color=ColorGrade(contrast=1.1))
    tl = _tl([adj])
    chunks = plan_chunks(tl)
    cuts_inside = [a for a, _b in chunks[1:] if 35 * FPS < a < 95 * FPS]
    assert cuts_inside  # an adjustment's interior is no longer fully blocked
    a = cuts_inside[0]
    left = sub_timeline(tl, 0, a)
    right = sub_timeline(tl, a, tl.duration_frames)
    left_item = next(i for i in left.items if i.id == "mood")
    right_item = next(i for i in right.items if i.id == "mood")
    assert left_item.transform.anim_total_frames == 60 * FPS
    assert right_item.transform.anim_total_frames == 60 * FPS
    assert left_item.transform.anim_offset_frames == 0
    assert right_item.transform.anim_offset_frames == a - 35 * FPS


def test_fade_split_keeps_fade_in_on_the_left_and_fade_out_on_the_right():
    still = TimelineItem(id="s", track="V3", source_id="img", source_start_frame=0,
                         source_end_frame=20 * FPS, timeline_start_frame=100 * FPS,
                         timeline_end_frame=120 * FPS,
                         color=ColorGrade(fade_in=1.0, fade_out=1.0))
    tl = _tl([still])
    cut = 110 * FPS  # well clear of both fade windows (1s each)
    left = sub_timeline(tl, 90 * FPS, cut)
    right = sub_timeline(tl, cut, 130 * FPS)
    left_item = next(i for i in left.items if i.id == "s")
    right_item = next(i for i in right.items if i.id == "s")
    assert left_item.color.fade_in == 1.0 and left_item.color.fade_out == 0.0
    assert right_item.color.fade_in == 0.0 and right_item.color.fade_out == 1.0


def test_plan_chunks_never_cuts_inside_a_fade_region():
    still = TimelineItem(id="s", track="V3", source_id="img", source_start_frame=0,
                         source_end_frame=20 * FPS, timeline_start_frame=100 * FPS,
                         timeline_end_frame=120 * FPS,
                         color=ColorGrade(fade_in=2.0, fade_out=2.0))
    chunks = plan_chunks(_tl([still]))
    fade_in_frames = range(100 * FPS, 100 * FPS + 2 * FPS)
    fade_out_frames = range(120 * FPS - 2 * FPS, 120 * FPS + 1)
    for a, _b in chunks[1:]:
        assert a not in fade_in_frames and a not in fade_out_frames


def test_a_chunk_timeline_starts_at_zero_and_keeps_only_its_items():
    still = TimelineItem(id="s", track="V3", source_id="img", source_start_frame=0,
                         source_end_frame=4 * FPS, timeline_start_frame=100 * FPS,
                         timeline_end_frame=104 * FPS)
    tl = _tl([still])
    part = sub_timeline(tl, 90 * FPS, 130 * FPS)
    ids = {i.id: i for i in part.items}
    assert set(ids) == {"v1", "s"}           # audio dropped, still kept
    assert ids["v1"].timeline_start_frame == 0
    assert ids["v1"].source_start_frame == 90 * FPS
    assert ids["v1"].duration_frames == 40 * FPS
    assert ids["s"].timeline_start_frame == 10 * FPS
    assert part.duration_frames == 40 * FPS and set(part.sources) == {"main", "img"}
    assert part.audio_master is None


def test_compose_at_the_delivery_size_only_when_smaller():
    assert _render_canvas((3840, 2160), (1920, 1080)) == (1920, 1080)
    assert _render_canvas((1920, 1080), (1920, 1080)) is None
    assert _render_canvas((1280, 720), (1920, 1080)) is None
    assert _render_canvas((2160, 3840), (1920, 1080)) == (606, 1080)
    assert _render_canvas((3840, 2160), None) is None


def test_parallel_chunks_follow_resolution_and_free_ram():
    from render.chunked import parallel_chunks
    assert parallel_chunks((1920, 1080), free_gb=24) == 4
    assert parallel_chunks((3840, 2160), free_gb=24) == 2
    assert parallel_chunks((3840, 2160), free_gb=12) == 1
    assert parallel_chunks((1920, 1080), free_gb=4) == 1


def test_text_is_scaled_to_a_4k_canvas_and_left_alone_at_1080p():
    from render.runner import _scale_text
    from timeline.schema import TextClip, TextStyle
    cap = TimelineItem(id="t", track="TC", kind="text", timeline_start_frame=0,
                       timeline_end_frame=24, text=TextClip(content="hi",
                       style=TextStyle(font_size=46, box_padding=16, stroke_width=0)))
    tl = _tl([cap])
    scaled = {i.id: i for i in _scale_text(tl, (3840, 2160)).items}["t"].text.style
    assert (scaled.font_size, scaled.box_padding, scaled.stroke_width) == (92, 32, 0)
    assert _scale_text(tl, (1920, 1080)) is tl


# --- the 25eff765 incident -------------------------------------------------

@pytest.mark.skipif(not PROJECT_25EFF765.exists(), reason="sample project data not present")
def test_25eff765_no_chunk_exceeds_3x_target_and_every_chunk_builds():
    """The incident: 32 V8 mood adjustments, 33 V2 autozoom adjustments and 234
    animated V1 face segments used to block almost the whole programme's
    interior, leaving chunks of 79/45/244/342/578/271/55/113s -- the 578s one
    alone ran ~40 minutes. None of those three categories is unsplittable any
    more, so planning here should land close to the 45s target throughout."""
    with open(PROJECT_25EFF765, "r", encoding="utf-8") as f:
        data = json.load(f)
    tl = Timeline.model_validate(data["timeline"])
    fps = tl.fps_num / max(1, tl.fps_den)
    chunks = plan_chunks(tl)
    assert len(chunks) > 8  # far more, far smaller chunks than the incident's 8
    lengths_s = [(b - a) / fps for a, b in chunks]
    assert max(lengths_s) <= CHUNK_TARGET_S * 3
    for a, b in chunks:
        part = sub_timeline(tl, a, b)
        assert part.duration_frames == b - a


# --- stall watchdog ---------------------------------------------------------

def test_stall_watchdog_kills_a_process_that_never_progresses(monkeypatch):
    monkeypatch.setattr(chunked_mod, "STALL_SECONDS", 2.0)
    monkeypatch.setattr(chunked_mod, "_STALL_POLL_S", 0.3)

    async def run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        with pytest.raises(ChunkStalledError):
            await _follow(proc, None, output_path=None)
        assert proc.returncode is not None  # the watchdog's kill was awaited

    asyncio.run(asyncio.wait_for(run(), timeout=15))


def test_stall_watchdog_leaves_real_progress_alone(monkeypatch):
    monkeypatch.setattr(chunked_mod, "STALL_SECONDS", 2.0)
    monkeypatch.setattr(chunked_mod, "_STALL_POLL_S", 0.3)

    async def run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import sys, time\n"
            "for i in range(6):\n"
            "    time.sleep(0.5)\n"
            "    print(f'out_time_ms={i * 500000}', flush=True)\n",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        seen = []
        await _follow(proc, seen.append, output_path=None)
        assert seen  # progress kept the watchdog from ever firing

    asyncio.run(asyncio.wait_for(run(), timeout=15))


# --- resume across a restart ------------------------------------------------

def test_resume_skips_chunks_whose_done_marker_and_graph_match(tmp_path, monkeypatch):
    monkeypatch.setattr(chunked_mod, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(chunked_mod, "MIN_PROGRAMME_S", 1.0)
    monkeypatch.setattr(chunked_mod, "_work_dir_hash", lambda *a, **k: "resumetest")
    monkeypatch.setattr(chunked_mod, "parallel_chunks", lambda *a, **k: 2)
    # Keep the work dir around after a "successful" run so its artefacts can
    # be inspected/edited to simulate a restart.
    monkeypatch.setattr(chunked_mod.shutil, "rmtree", lambda *a, **k: None)

    calls = []

    async def fake_run(cmd, on_progress=None, output_path=None):
        calls.append(str(output_path) if output_path else None)
        if output_path is not None:
            Path(output_path).write_bytes(b"0" * 100)
        if on_progress:
            on_progress(999_999.0)

    monkeypatch.setattr(chunked_mod, "_run", fake_run)

    tl = _tl()
    asyncio.run(chunked_mod.render_chunked(
        tl, str(tmp_path / "out1.mp4"), None, None, False, False))

    work = tmp_path / "chunks_resumetest"
    assert work.exists()  # rmtree mocked away, as if the job were interrupted

    # Simulate a restart that only chunk 0 (and not the audio mix, nor any
    # other chunk) had actually finished before: drop everyone else's marker.
    markers = sorted(work.glob("chunk_*.done"))
    assert len(markers) >= 2
    for marker in markers[1:]:
        marker.unlink()
        Path(str(marker)[:-len(".done")] + ".mp4").unlink(missing_ok=True)
    (work / "audio.done").unlink(missing_ok=True)
    (work / "audio.m4a").unlink(missing_ok=True)

    calls.clear()
    asyncio.run(chunked_mod.render_chunked(
        tl, str(tmp_path / "out2.mp4"), None, None, False, False))

    assert not any(c and "chunk_000.part.mp4" in c for c in calls)
    assert any(c and "chunk_001.part.mp4" in c for c in calls)
    assert any(c and "mix.part.wav" in c for c in calls)


# --- per-chunk retry ---------------------------------------------------------

def test_a_failed_chunk_is_retried_alone_and_can_still_succeed(tmp_path, monkeypatch):
    monkeypatch.setattr(chunked_mod, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(chunked_mod, "MIN_PROGRAMME_S", 1.0)
    monkeypatch.setattr(chunked_mod, "_work_dir_hash", lambda *a, **k: "retrytest")
    monkeypatch.setattr(chunked_mod, "parallel_chunks", lambda *a, **k: 2)

    attempts = {"n": 0}

    async def fake_run(cmd, on_progress=None, output_path=None):
        path = str(output_path) if output_path else ""
        if "chunk_000.part.mp4" in path:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("synthetic ffmpeg failure")
        if output_path is not None:
            Path(output_path).write_bytes(b"0" * 100)
        if on_progress:
            on_progress(999_999.0)

    monkeypatch.setattr(chunked_mod, "_run", fake_run)

    out = asyncio.run(chunked_mod.render_chunked(
        _tl(), str(tmp_path / "out.mp4"), None, None, False, False))
    assert out is not None
    assert attempts["n"] == 2  # failed once, then succeeded on the solo retry


def test_a_chunk_that_never_succeeds_fails_the_whole_render(tmp_path, monkeypatch):
    monkeypatch.setattr(chunked_mod, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(chunked_mod, "MIN_PROGRAMME_S", 1.0)
    monkeypatch.setattr(chunked_mod, "_work_dir_hash", lambda *a, **k: "retryfail")
    monkeypatch.setattr(chunked_mod, "parallel_chunks", lambda *a, **k: 2)

    attempts = {"n": 0}

    async def fake_run(cmd, on_progress=None, output_path=None):
        path = str(output_path) if output_path else ""
        if "chunk_000.part.mp4" in path:
            attempts["n"] += 1
            raise RuntimeError("synthetic ffmpeg failure")
        if output_path is not None:
            Path(output_path).write_bytes(b"0" * 100)
        if on_progress:
            on_progress(999_999.0)

    monkeypatch.setattr(chunked_mod, "_run", fake_run)

    with pytest.raises(RuntimeError):
        asyncio.run(chunked_mod.render_chunked(
            _tl(), str(tmp_path / "out.mp4"), None, None, False, False))
    # The first attempt plus every retry, no more -- render_chunked gave up.
    assert attempts["n"] == 1 + chunked_mod.MAX_CHUNK_RETRIES
    assert (tmp_path / "chunks_retryfail").exists()  # kept for a resume
