"""Camera angles on the transcript-built V1 (docs/MULTICAM_SYNC_PLAN.md §3-5).

Every source's frames are counted at the TIMELINE fps, so an angle piece is the
primary piece shifted by round(sync_offset_s * tl_fps) -- whatever rate the
angle was recorded at.
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import FFMPEG_BIN
from presentation.placement import BROLL_ORIGIN, jump_cut_coverage, jump_cut_times
from store.project_store import ProjectStore
from sync.audio_sync import SyncResult
from timeline import build_timeline_from_transcript
from timeline.authoring import _program_segments
from timeline.ops import (auto_angles, clear_angles, cut_program_range, rebuild_primary_tracks,
                          set_angle, toggle_word, v1_primary_span)
from timeline.schema import SourceFile, TimelineItem, Transform

FPS = 25


def _words(segment_lengths, word_s=0.48):
    """Runs of words separated by a disfluent "um" the edit cuts out."""
    words, t = [], 0.0
    for n, count in enumerate(segment_lengths):
        for k in range(count):
            text = f"w{n}_{k}" + ("." if k == count - 1 else "")
            words.append({"word": text, "start": round(t, 3), "end": round(t + word_s, 3)})
            t += word_s
        if n < len(segment_lengths) - 1:
            words.append({"word": "um", "start": round(t, 3), "end": round(t + word_s, 3),
                          "disfluency": True})
            t += word_s
    return words


def _timeline(segment_lengths=(3, 3, 3, 3, 3, 3), duration=60.0):
    tl = build_timeline_from_transcript(
        source_path="cam_a.mp4", duration_seconds=duration,
        transcript_words=_words(segment_lengths), fps_num=FPS, fps_den=1,
        pause_padding_seconds=0.0)
    return tl, list(tl.sources.keys())[0]


def _add_angle(tl, primary, offset_s=1.2, duration=60.0, fps=30, sid="src_camb"):
    tl.sources[sid] = SourceFile(id=sid, path="cam_b.mp4", duration_seconds=duration,
                                 width=1280, height=720, fps_num=fps, fps_den=1,
                                 synced_to=primary, sync_offset_s=offset_s, sync_confidence=6.0)
    return tl.sources[sid]


def _v1(tl):
    return sorted([i for i in tl.items if i.track == "V1"], key=lambda i: i.timeline_start_frame)


def _a1(tl):
    return sorted([i for i in tl.items if i.track == "A1"], key=lambda i: i.timeline_start_frame)


# ---------------------------------------------------------------- apply / rebuild

def test_spans_survive_a_word_toggle_rebuild():
    tl, primary = _timeline()
    angle = _add_angle(tl, primary)
    tl.angle_spans = [{"start": 0, "end": 60 * FPS, "source_id": angle.id}]
    rebuild_primary_tracks(tl, primary)
    assert all(i.source_id == angle.id for i in _v1(tl))

    word = next(w for w in tl.words if w.text == "w2_1")
    toggle_word(tl, word.id, False, primary)
    assert tl.angle_spans == [{"start": 0, "end": 60 * FPS, "source_id": angle.id}]
    v1 = _v1(tl)
    assert v1 and all(i.source_id == angle.id for i in v1)
    assert all(i.source_id == primary for i in _a1(tl))
    # The struck word is gone from the angle picture as well as the sound.
    assert not any(v1_primary_span(tl, i)[0] <= word.start_frame < v1_primary_span(tl, i)[1]
                   for i in v1)


def test_retranscribe_carries_angles_audio_and_manual_cuts():
    from routes.transcription import _carry_project_state
    old, old_primary = _timeline()
    angle = _add_angle(old, old_primary)
    old.sources[old_primary].audio_override = "ext.wav"
    old.sources[old_primary].audio_history = [{"path": "ext.wav", "effect": "sync"}]
    old.angle_spans = [{"start": 0, "end": 60 * FPS, "source_id": angle.id}]
    old.manual_cuts = [[0, 5]]
    rebuild_primary_tracks(old, old_primary)

    tl, primary = _timeline()                      # a fresh build names the primary anew
    _carry_project_state(old.model_dump(), tl, "cam_a.mp4")
    assert tl.sources[primary].audio_override == "ext.wav"
    assert tl.sources[primary].audio_history == [{"path": "ext.wav", "effect": "sync"}]
    assert tl.sources[angle.id].synced_to == primary
    assert tl.manual_cuts == [[0, 5]] and tl.angle_spans == old.angle_spans
    assert _v1(tl) and all(i.source_id == angle.id for i in _v1(tl))
    assert all(i.source_id == primary for i in _a1(tl))
    assert min(v1_primary_span(tl, i)[0] for i in _v1(tl)) >= 5

def test_frame_mapping_uses_the_timeline_fps_not_the_angle_recording_rate():
    tl, primary = _timeline()
    angle = _add_angle(tl, primary, offset_s=1.2, fps=30)        # 1.2 s = 30 frames at 25 fps
    first = _v1(tl)[0]
    tl.angle_spans = [{"start": 0, "end": 60 * FPS, "source_id": angle.id}]
    rebuild_primary_tracks(tl, primary)
    piece = _v1(tl)[0]
    assert piece.source_id == angle.id
    assert piece.source_start_frame == first.source_start_frame + 30
    assert piece.source_end_frame == first.source_end_frame + 30

    # The angle's own recorded rate plays no part: the per-clip fps= filter resamples.
    for recorded in (24, 30, 60):
        angle.fps_num = recorded
        rebuild_primary_tracks(tl, primary)
        assert _v1(tl)[0].source_start_frame == first.source_start_frame + 30


def test_a1_untouched_link_ids_and_carried_grade_survive():
    tl, primary = _timeline()
    angle = _add_angle(tl, primary)
    v1, a1 = _v1(tl), _a1(tl)
    for n, (v, a) in enumerate(zip(v1, a1)):
        v.link_id = a.link_id = f"lnk_{n}"
    v1[1].transform = Transform(scale=1.3)
    before_a1 = [(i.source_id, i.source_start_frame, i.source_end_frame,
                  i.timeline_start_frame, i.timeline_end_frame) for i in a1]

    seg = v1[1]
    mid = (seg.source_start_frame + seg.source_end_frame) // 2
    tl.angle_spans = [{"start": mid, "end": 60 * FPS, "source_id": angle.id}]
    from timeline.ops import apply_angle_spans
    apply_angle_spans(tl, primary)

    assert [(i.source_id, i.source_start_frame, i.source_end_frame, i.timeline_start_frame,
             i.timeline_end_frame) for i in _a1(tl)] == before_a1
    halves = [i for i in _v1(tl) if i.anchor_word_id == seg.anchor_word_id]
    assert [h.source_id for h in halves] == [primary, angle.id]
    assert {h.link_id for h in halves} == {"lnk_1"}
    assert all(h.transform and h.transform.scale == pytest.approx(1.3) for h in halves)
    assert halves[0].timeline_end_frame == halves[1].timeline_start_frame

    # A rebuild carries the grade by anchor word onto both pieces again.
    rebuild_primary_tracks(tl, primary)
    halves = [i for i in _v1(tl) if i.anchor_word_id == seg.anchor_word_id]
    assert [h.source_id for h in halves] == [primary, angle.id]
    assert all(h.transform and h.transform.scale == pytest.approx(1.3) for h in halves)


def test_a_piece_the_angle_did_not_record_stays_on_the_primary():
    tl, primary = _timeline()
    # Cam B started 2 s late (offset -2.0) and ran only 6 s.
    angle = _add_angle(tl, primary, offset_s=-2.0, duration=6.0)
    tl.angle_spans = [{"start": 0, "end": 60 * FPS, "source_id": angle.id}]
    rebuild_primary_tracks(tl, primary)
    for item in _v1(tl):
        p_start, p_end = v1_primary_span(tl, item)
        if item.source_id == angle.id:
            assert 2 * FPS <= p_start and p_end <= 8 * FPS
            assert 0 <= item.source_start_frame and item.source_end_frame <= 6 * FPS
        else:
            assert p_end <= 2 * FPS or p_start >= 8 * FPS
    assert any(i.source_id == angle.id for i in _v1(tl))


def test_captions_and_program_cuts_read_angle_pieces_on_the_primary_clock():
    tl, primary = _timeline()
    angle = _add_angle(tl, primary)
    plain = _program_segments(tl)
    set_angle(tl, 0, tl.duration_frames, angle.id, primary)
    assert all(i.source_id == angle.id for i in _v1(tl))
    assert _program_segments(tl) == plain
    assert jump_cut_times(tl) == [i.timeline_end_frame / FPS for i in _v1(tl)[:-1]]

    # Cutting a programme range lands on primary frames, and the angle stays.
    first = _v1(tl)[0]
    cut_program_range(tl, first.timeline_start_frame, first.timeline_start_frame + 5, primary)
    assert tl.manual_cuts[0] == [0, 5]
    assert all(i.source_id == angle.id for i in _v1(tl))


# ---------------------------------------------------------------- set_angle

def test_set_angle_takes_programme_frames_and_none_clears():
    tl, primary = _timeline()
    angle = _add_angle(tl, primary)
    v1 = _v1(tl)
    # Programme frames spanning the first jump cut: two primary spans, one per side.
    start, end = v1[0].timeline_end_frame - 10, v1[1].timeline_start_frame + 10
    set_angle(tl, start, end, angle.id, primary)
    # One span: the cut-out "um" between the two pieces is bridged.
    assert tl.angle_spans == [{"start": v1[0].source_end_frame - 10,
                               "end": v1[1].source_start_frame + 10, "source_id": angle.id}]
    on_angle = [i for i in _v1(tl) if i.source_id == angle.id]
    assert sum(i.duration_frames for i in on_angle) == 20
    assert all(i.source_id == primary for i in _a1(tl))

    # None over part of it carves the range back to the primary ...
    set_angle(tl, start, start + 5, None, primary)
    assert sum(i.duration_frames for i in _v1(tl) if i.source_id == angle.id) == 15
    # ... and over all of it leaves no span and no split behind.
    set_angle(tl, 0, tl.duration_frames, None, primary)
    assert tl.angle_spans == []
    assert [(i.source_start_frame, i.source_end_frame) for i in _v1(tl)] == \
        [(i.source_start_frame, i.source_end_frame) for i in v1]


def test_set_angle_overwrites_and_clear_angles():
    tl, primary = _timeline()
    b = _add_angle(tl, primary, sid="src_b")
    c = _add_angle(tl, primary, offset_s=-0.4, sid="src_c")
    total = tl.duration_frames
    set_angle(tl, 0, total, b.id, primary)
    set_angle(tl, 10, 30, c.id, primary)
    assert [(s["source_id"]) for s in tl.angle_spans] == ["src_b", "src_c", "src_b"]
    first = _v1(tl)[0]
    p_first = v1_primary_span(tl, first)[0]
    assert tl.angle_spans[1] == {"start": p_first + 10, "end": p_first + 30, "source_id": "src_c"}
    with pytest.raises(ValueError):
        set_angle(tl, 0, 10, "src_nope", primary)
    clear_angles(tl, primary)
    assert tl.angle_spans == [] and all(i.source_id == primary for i in _v1(tl))


# ---------------------------------------------------------------- auto

def test_hide_cuts_alternates_at_every_cut_with_long_shots():
    # Segments of 3 s: every cut is >= min_shot_s from the last switch.
    tl, primary = _timeline(segment_lengths=(6, 6, 6, 6, 6))
    angle = _add_angle(tl, primary)
    out = auto_angles(tl, primary, "hide_cuts", min_shot_s=2.5)
    v1 = _v1(tl)
    assert [i.source_id for i in v1] == [primary, angle.id, primary, angle.id, primary]
    assert out["summary"]["switches"] == 4
    assert out["summary"]["cuts_total"] == 4 and out["summary"]["cuts_hidden"] == 4
    assert jump_cut_times(tl) == [i.timeline_end_frame / FPS for i in v1[:-1]]
    assert jump_cut_coverage(tl)["angle_switches"] == 4


def test_hide_cuts_respects_min_shot_and_broll():
    # 1.5 s segments: only every second cut can switch with min_shot_s 2.5.
    tl, primary = _timeline(segment_lengths=(3, 3, 3, 3, 3, 3, 3))
    angle = _add_angle(tl, primary)
    cuts = jump_cut_times(tl)
    assert cuts == [1.44, 2.88, 4.32, 5.76, 7.2, 8.64]
    # A B-roll cutaway over the cut at 5.76 s.
    tl.items.append(TimelineItem(id="broll1", track="V2", source_id=angle.id, origin=BROLL_ORIGIN,
                                 timeline_start_frame=int(5.3 * FPS),
                                 timeline_end_frame=int(6.3 * FPS)))
    out = auto_angles(tl, primary, "hide_cuts", min_shot_s=2.5)
    v1 = _v1(tl)
    switch_times = [v1[k + 1].timeline_start_frame / FPS for k in range(len(v1) - 1)
                    if v1[k].source_id != v1[k + 1].source_id]
    assert switch_times == [2.88, 7.2]              # 5.76 is under B-roll, 4.32/8.64 too close
    assert all(b - a >= 2.5 for a, b in zip([0.0] + switch_times, switch_times))
    assert out["summary"] == {"mode": "hide_cuts", "switches": 2, "cuts_total": 6,
                              "cuts_by_angle": 2, "cuts_by_broll": 1, "cuts_hidden": 3}
    assert all(i.source_id == primary for i in _a1(tl))


def test_rhythm_switches_at_sentence_ends_and_off_clears():
    # Keep-full-source style: one long take, sentence every 2 s.
    tl, primary = _timeline(segment_lengths=(4,) * 10)
    tl.words = [w for w in tl.words if w.text != "um"]       # no cuts at all now
    for n, w in enumerate(tl.words):
        w.start_frame, w.end_frame = n * 12, n * 12 + 12
    rebuild_primary_tracks(tl, primary)
    assert len(_v1(tl)) == 1
    angle = _add_angle(tl, primary)
    out = auto_angles(tl, primary, "rhythm", cadence_s=5.0)
    v1 = _v1(tl)
    switches = [v1[k + 1].timeline_start_frame / FPS for k in range(len(v1) - 1)]
    assert out["summary"]["switches"] == len(switches) >= 2
    sentence_ends = {(n + 1) * 12 / FPS for n, w in enumerate(tl.words) if w.text.endswith(".")}
    assert all(any(abs(t - e) < 0.05 for e in sentence_ends) for t in switches)
    assert all(b - a >= 5.0 - 1e-6 for a, b in zip([0.0] + switches, switches))
    assert {i.source_id for i in v1} == {primary, angle.id}

    auto_angles(tl, primary, "off")
    assert tl.angle_spans == [] and len(_v1(tl)) == 1


# ---------------------------------------------------------------- routes

def _ffmpeg(*args):
    proc = subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-400:]


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("angles_media")
    cam_a, cam_b, wav = folder / "cam_a.mp4", folder / "cam_b.mp4", folder / "recorder.wav"
    for path, size in ((cam_a, "640x360"), (cam_b, "320x180")):
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
                "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=5",
            "-c:a", "pcm_s16le", str(wav))
    return {"cam_a": cam_a, "cam_b": cam_b, "wav": wav}


@pytest.fixture
def client(tmp_path, monkeypatch, media):
    from main import app
    from audio import noise
    from routes import timeline as timeline_route

    projects = tmp_path / "projects"
    projects.mkdir()
    store = ProjectStore(base_dir=str(projects))
    monkeypatch.setattr(timeline_route, "project_store", store)
    monkeypatch.setattr(noise, "project_store", store)
    monkeypatch.setattr(noise, "PROJECTS_DIR", projects)

    tl = build_timeline_from_transcript(
        source_path=str(media["cam_a"]), duration_seconds=4.0,
        transcript_words=_words((2, 2)), fps_num=FPS, fps_den=1, width=640, height=360,
        pause_padding_seconds=0.0)
    pool = [{"id": "media_b", "name": "cam_b.mp4", "path": str(media["cam_b"]), "kind": "video",
             "duration": 4.0, "width": 320, "height": 180, "fps": 25.0, "has_audio": True},
            {"id": "media_rec", "name": "recorder.wav", "path": str(media["wav"]), "kind": "audio",
             "duration": 5.0, "width": 0, "height": 0, "fps": 0.0, "has_audio": True}]
    store.save_project("p1", {"id": "p1", "name": "T", "timeline": tl.model_dump(),
                              "media_pool": pool})
    with TestClient(app) as test_client:
        yield test_client, store


def _fake_offset(monkeypatch, result):
    import sync.audio_sync as audio_sync
    monkeypatch.setattr(audio_sync, "find_offset", lambda ref, other, max_offset_s=900.0: result)


def test_routes_refuse_an_unreliable_match(client, monkeypatch):
    test_client, store = client
    _fake_offset(monkeypatch, SyncResult(offset_s=0.4, confidence=1.2, drift_ppm=0.0))
    before = store.get_project("p1")
    for path, media_id in (("angles/sync", "media_b"), ("sync/audio", "media_rec")):
        resp = test_client.post(f"/api/timeline/p1/{path}", json={"media_id": media_id})
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["confidence"] == pytest.approx(1.2)
    after = store.get_project("p1")
    assert after["timeline"] == before["timeline"]
    assert "angle" not in after["media_pool"][0]


def test_angle_routes_round_trip(client, monkeypatch):
    test_client, store = client
    _fake_offset(monkeypatch, SyncResult(offset_s=0.2, confidence=4.0, drift_ppm=0.0))
    resp = test_client.post("/api/timeline/p1/angles/sync", json={"media_id": "media_b"})
    assert resp.status_code == 200, resp.text
    sid = resp.json()["source_id"]
    pool = store.get_project("p1")["media_pool"]
    assert pool[0]["angle"]["synced_to"] and pool[0]["angle"]["offset_s"] == pytest.approx(0.2)

    listing = test_client.get("/api/timeline/p1/angles").json()
    assert listing["angles"][0]["source_id"] == sid
    assert listing["angles"][0]["media_id"] == "media_b"
    assert (listing["angles"][0]["width"], listing["angles"][0]["height"]) == (320, 180)
    assert listing["spans"] == []

    resp = test_client.post("/api/timeline/p1/angles/set",
                            json={"start_frame": 5, "end_frame": 20, "source_id": sid})
    assert resp.status_code == 200, resp.text
    spans = test_client.get("/api/timeline/p1/angles").json()["spans"]
    assert spans and spans[0]["source_id"] == sid and spans[0]["start"] == 5

    resp = test_client.post("/api/timeline/p1/angles/auto", json={"mode": "hide_cuts",
                                                                 "min_shot_s": 0.5})
    assert resp.status_code == 200, resp.text
    assert resp.json()["summary"]["cuts_total"] == 1

    resp = test_client.delete("/api/timeline/p1/angles")
    assert resp.status_code == 200 and resp.json()["angle_spans"] == []
    assert test_client.get("/api/timeline/p1/angles").json()["spans"] == []


def test_external_audio_becomes_the_programme_sound_and_undoes(client, monkeypatch):
    test_client, store = client
    _fake_offset(monkeypatch, SyncResult(offset_s=0.5, confidence=5.0, drift_ppm=0.0))
    resp = test_client.post("/api/timeline/p1/sync/audio", json={"media_id": "media_rec"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    version = Path(body["version_path"])
    assert version.exists() and version.parent.name == "versions"
    from sync.audio_sync import decode_mono
    assert abs(decode_mono(str(version), sr=48000).size - 4 * 48000) <= 2

    tl = store.get_project("p1")["timeline"]
    primary = tl["sources"][body["source_id"]]
    assert primary["audio_override"] == str(version)
    assert primary["audio_history"][-1]["effect"] == "external_audio_sync"
    assert primary["audio_history"][-1]["params"]["media_id"] == "media_rec"

    from routes.transcription import _transcription_audio
    p_data = store.get_project("p1")
    assert _transcription_audio(p_data, primary["path"]) == str(version)

    resp = test_client.post("/api/audio/p1/undo", json={"source_id": body["source_id"]})
    assert resp.status_code == 200, resp.text
    primary = store.get_project("p1")["timeline"]["sources"][body["source_id"]]
    assert primary["audio_override"] is None
