"""Tests for the automatic punch-ins and for when a job is allowed to start.

A zoom is the one thing in the presentation pass that touches the speaker's own
footage rather than layering over it, so nearly all of these check a refusal: a
punch that crosses a cut, or starts on one, or happens while a cutaway is on
screen, is worse than no punch at all.
"""

from datetime import datetime, timedelta

import pytest

from agents.scheduler import _is_due, resolve_start_at
from presentation.facezoom import (MIN_PUNCH_GAP_SECONDS, PUNCH_DEPTH_MAX,
                                   _to_canvas, apply_zooms, plan_zooms)
from presentation.models import (PresentationSettings, Program, ProgramSegment,
                                 ProgramWord, SegmentZoom, WindowZoom)
from timeline import build_timeline_from_transcript


def _program(segment_count: int = 6, segment_seconds: float = 10.0,
             loud_at: float = None) -> Program:
    """A programme of evenly spaced clips, optionally with one emphatic word."""
    words = []
    segments = []
    for index in range(segment_count):
        start = index * segment_seconds
        segments.append(ProgramSegment(
            item_id=f"v1_{index}",
            tl_start_s=start,
            tl_end_s=start + segment_seconds,
            source_start_frame=int(start * 30),
            source_end_frame=int((start + segment_seconds) * 30),
        ))
        step = 0.5
        clock = start
        while clock < start + segment_seconds - step:
            emphasis = 0.0
            if loud_at is not None and abs(clock - loud_at) < 0.4:
                emphasis = 2.5
            words.append(ProgramWord(
                text=f"w{len(words)}", tl_start_s=clock, tl_end_s=clock + 0.4,
                source_start_frame=int(clock * 30), emphasis_z=emphasis))
            clock += step

    return Program(duration_s=segment_count * segment_seconds, fps=30.0,
                   words=words, segments=segments, has_energy=True)


# --- aiming the zoom -------------------------------------------------------

def test_a_centred_face_gives_a_centred_anchor():
    assert _to_canvas(0.5, 0.5) == (0.0, 0.0)


def test_an_off_centre_face_is_pulled_back_toward_the_middle():
    """Zooming hard at a face near the edge of frame crops the composition
    badly, so the anchor is damped rather than used raw."""
    x, _ = _to_canvas(1.0, 0.5)
    assert 0.0 < x < 1.0


def test_the_anchor_can_never_run_off_the_canvas():
    for cx in (0.0, 1.0, -3.0, 5.0):
        x, y = _to_canvas(cx, cx)
        assert -0.35 <= x <= 0.35 and -0.35 <= y <= 0.35


# --- planning the moves ----------------------------------------------------

def test_every_segment_move_pushes_in_so_each_join_resets_wide():
    """Every clip starts wide and drifts tighter, so every V1 join steps back to
    wide by the whole depth of the move — a punch-out, the standard disguise for
    a talking-head jump cut. Alternating directions made the scale CONTINUOUS
    across every join (a push-in ends exactly where the next pull-back starts)
    and the head-position jump played bare. Variety comes from jittered depth."""
    segment_zooms, _ = plan_zooms(_program(), PresentationSettings(), [], seed=0)
    assert segment_zooms
    assert all(z.push_in for z in segment_zooms)
    if len(segment_zooms) > 2:
        assert len({z.depth for z in segment_zooms}) > 1, "depth should vary"
    assert all(0.06 <= z.depth <= 0.6 for z in segment_zooms)


def test_planning_is_deterministic():
    """A re-run must not reshuffle the edit under the user."""
    first, _ = plan_zooms(_program(), PresentationSettings(), [], seed=3)
    second, _ = plan_zooms(_program(), PresentationSettings(), [], seed=3)
    assert [z.item_id for z in first] == [z.item_id for z in second]


def test_a_punch_lands_on_the_emphatic_moment():
    _, punches = plan_zooms(_program(loud_at=25.0), PresentationSettings(), [], seed=0)
    assert punches
    assert any(p.start_s <= 25.0 <= p.end_s for p in punches)


def test_a_punch_never_crosses_a_cut():
    """The adjustment layer would carry the zoom over the join and the two shots
    would look like one moving frame."""
    program = _program(loud_at=19.8)     # right at the end of a segment
    _, punches = plan_zooms(program, PresentationSettings(), [], seed=0)
    for punch in punches:
        segment = program.segment_at(punch.start_s)
        assert segment is not None
        assert punch.end_s <= segment.tl_end_s
        assert punch.start_s >= segment.tl_start_s


def test_a_punch_keeps_clear_of_the_edges_of_its_shot():
    program = _program(loud_at=20.05)    # just after a cut
    _, punches = plan_zooms(program, PresentationSettings(), [], seed=0)
    for punch in punches:
        segment = program.segment_at(punch.start_s)
        assert punch.start_s >= segment.tl_start_s + 0.49


def test_no_punch_happens_while_a_cutaway_is_on_screen():
    """Nobody would see it, and it would be mid-move when the picture returns."""
    busy = [(20.0, 30.0)]
    _, punches = plan_zooms(_program(loud_at=25.0), PresentationSettings(), busy, seed=0)
    assert all(not (p.start_s < 30.0 and 20.0 < p.end_s) for p in punches)


def test_punches_are_spaced_out():
    program = _program(segment_count=2, segment_seconds=60.0)
    for word in program.words:
        word.emphasis_z = 3.0            # every word maximally emphatic
    _, punches = plan_zooms(program, PresentationSettings(), [], seed=0)
    ordered = sorted(punches, key=lambda p: p.start_s)
    for before, after in zip(ordered, ordered[1:]):
        assert after.start_s - before.end_s >= MIN_PUNCH_GAP_SECONDS - 0.01


def test_a_punch_never_zooms_further_than_the_footage_allows():
    program = _program(segment_count=2, segment_seconds=60.0)
    for word in program.words:
        word.emphasis_z = 9.0
    _, punches = plan_zooms(program, PresentationSettings(), [], seed=0)
    assert all(p.depth <= PUNCH_DEPTH_MAX for p in punches)


def test_no_energy_means_no_punches_but_still_a_rhythm():
    """An older project without a loudness envelope should still get baseline
    moves; there is simply nothing to time a punch to."""
    program = _program()
    program.has_energy = False
    segment_zooms, punches = plan_zooms(program, PresentationSettings(), [], seed=0)
    assert segment_zooms and punches == []


# --- applying them ---------------------------------------------------------

def _timeline():
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(80)]
    return build_timeline_from_transcript(
        source_path="C:/media/talk.mp4", duration_seconds=45.0,
        transcript_words=words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, 45.0)], pause_padding_seconds=0.0)


def test_a_punch_becomes_a_windowed_adjustment_layer():
    """It cannot be a clip transform: those animate across the whole clip, and
    V1 clips cannot be split because the word rebuild owns them."""
    timeline = _timeline()
    segments, punches = 0, [WindowZoom(start_s=5.0, end_s=6.5, depth=0.15)]
    applied_segments, applied_punches = apply_zooms(timeline, [], punches, {})
    assert applied_punches == 1
    adjustment = next(i for i in timeline.items if i.origin == "autozoom")
    assert adjustment.kind == "adjustment"
    assert adjustment.track == "V2"
    assert adjustment.transform.is_animated()


def test_a_segment_move_is_written_onto_the_clip_itself():
    timeline = _timeline()
    v1 = next(i for i in timeline.items if i.track == "V1")
    applied, _ = apply_zooms(timeline, [SegmentZoom(item_id=v1.id, push_in=True,
                                                    depth=0.1)], [], {})
    assert applied == 1
    assert v1.transform is not None and v1.transform.is_animated()


def test_re_running_replaces_the_previous_punches():
    timeline = _timeline()
    punches = [WindowZoom(start_s=5.0, end_s=6.5, depth=0.15)]
    for _ in range(3):
        apply_zooms(timeline, [], punches, {})
    # One punch is an ease-in, a hold and an ease-out layer.
    assert len([i for i in timeline.items if i.origin == "autozoom"]) == 3


def test_a_move_drifts_toward_the_face_rather_than_starting_on_it():
    """Beginning a shot already off-centre looks like a framing error."""
    timeline = _timeline()
    v1 = next(i for i in timeline.items if i.track == "V1")
    apply_zooms(timeline, [SegmentZoom(item_id=v1.id, push_in=True, depth=0.1)],
                [], {v1.id: (0.3, 0.2)})
    assert abs(v1.transform.pos_x) < abs(v1.transform.pos_x_end)


# --- when a job may start --------------------------------------------------

def test_a_bare_time_means_the_next_time_it_comes_round():
    resolved = resolve_start_at("23:00")
    assert resolved is not None
    when = datetime.fromisoformat(resolved)
    assert when > datetime.now()
    assert when.hour == 23 and when.minute == 0


def test_a_time_already_past_today_means_tomorrow():
    """A job queued at 23:30 asking for 23:00 must wait a day, not start now."""
    now = datetime.now()
    earlier = (now - timedelta(hours=1)).strftime("%H:%M")
    when = datetime.fromisoformat(resolve_start_at(earlier))
    assert when > now


def test_no_start_time_means_start_now():
    assert resolve_start_at(None) is None
    assert resolve_start_at("") is None
    assert _is_due({"start_at": None}) is True


def test_a_job_waits_until_its_time():
    future = (datetime.now() + timedelta(hours=2)).isoformat()
    assert _is_due({"start_at": future}) is False
    past = (datetime.now() - timedelta(minutes=1)).isoformat()
    assert _is_due({"start_at": past}) is True


def test_an_unreadable_start_time_runs_rather_than_waiting_forever():
    assert resolve_start_at("half past bananas") is None
    assert _is_due({"start_at": "not a date"}) is True


def test_a_punch_eases_in_holds_and_eases_back_out_without_a_jump():
    """The old punch ended at 1+depth and the next frame was back at 1.0x --
    the jolt that read as jitter. The pieces must join edge to edge at the
    same scale and end where the shot started."""
    timeline = _timeline()
    apply_zooms(timeline, [], [WindowZoom(start_s=5.0, end_s=6.6, depth=0.2)], {})
    layers = sorted((i for i in timeline.items if i.origin == "autozoom"),
                    key=lambda i: i.timeline_start_frame)
    assert len(layers) == 3
    for a, b in zip(layers, layers[1:]):
        assert a.timeline_end_frame == b.timeline_start_frame
        a_end = a.transform.scale_end if a.transform.scale_end is not None else a.transform.scale
        assert a_end == pytest.approx(b.transform.scale)
    assert layers[0].transform.scale == pytest.approx(1.0)
    assert layers[1].transform.scale == pytest.approx(1.2)
    assert layers[-1].transform.scale_end == pytest.approx(1.0)


def test_one_long_take_gets_no_whole_programme_drift():
    """A precut recording is one V1 segment; a segment move across all of it
    was invisible motion, a softer picture, and the costliest render chain."""
    from presentation.facezoom import MAX_ZOOM_SEGMENT_SECONDS, plan_zooms
    from presentation.models import PresentationSettings, Program, ProgramSegment
    program = Program(duration_s=473.0, fps=25.0, segments=[ProgramSegment(
        item_id="v1", tl_start_s=0.0, tl_end_s=473.0, source_start_frame=0, source_end_frame=11825)],
        words=[])
    segment_zooms, _ = plan_zooms(program, PresentationSettings(), [], 1, {})
    assert 473.0 > MAX_ZOOM_SEGMENT_SECONDS and segment_zooms == []
