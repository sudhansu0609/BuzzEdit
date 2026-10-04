"""Linked clips, ripple edits, the magnetic main track and paste.

These are the NLE gestures behind the timeline's context menu and shortcuts:
a video and its own sound behave as one clip, ripple delete closes the hole,
a hand-built V1 never shows a gap the render would not play, and pasted clips
come back with fresh identities on lanes where they fit.
"""

import pytest

from backend.timeline import clip_ops
from backend.timeline.schema import SourceFile, Timeline, TimelineItem


def _timeline(*items: TimelineItem) -> Timeline:
    src = SourceFile(id="src", path="C:/media/clip.mp4", duration_seconds=60.0,
                     width=1920, height=1080, fps_num=30, fps_den=1, kind="video",
                     has_audio=True)
    tl = Timeline(fps_num=30, fps_den=1, sources={"src": src}, items=list(items))
    tl.recalculate_duration()
    return tl


def _clip(cid, track, start, end, link=None, src_start=0, origin="manual"):
    return TimelineItem(id=cid, track=track, source_id="src",
                        source_start_frame=src_start, source_end_frame=src_start + end - start,
                        timeline_start_frame=start, timeline_end_frame=end,
                        origin=origin, link_id=link)


def _span(tl, cid):
    item = clip_ops.get_item(tl, cid)
    return item.timeline_start_frame, item.timeline_end_frame


# --- linked clips ------------------------------------------------------------

def test_moving_a_video_carries_its_linked_audio():
    tl = _timeline(_clip("v", "V2", 30, 90, "L"), _clip("a", "A2", 30, 90, "L"))
    clip_ops.move_item(tl, "v", 60)
    assert _span(tl, "v") == (60, 120)
    assert _span(tl, "a") == (60, 120)


def test_moving_to_another_lane_keeps_the_partner_on_its_own():
    tl = _timeline(_clip("v", "V2", 30, 90, "L"), _clip("a", "A2", 30, 90, "L"))
    clip_ops.move_item(tl, "v", 45, new_track="V3")
    assert clip_ops.get_item(tl, "v").track == "V3"
    assert clip_ops.get_item(tl, "a").track == "A2"
    assert _span(tl, "a") == (45, 105)


def test_a_group_move_stops_at_the_start_instead_of_squashing():
    tl = _timeline(_clip("x", "V2", 10, 40), _clip("y", "V3", 50, 80))
    clip_ops.move_items(tl, ["x", "y"], -30)
    assert _span(tl, "x") == (0, 30)
    assert _span(tl, "y") == (40, 70)


def test_unlinked_move_leaves_the_partner_alone():
    tl = _timeline(_clip("v", "V2", 30, 90, "L"), _clip("a", "A2", 30, 90, "L"))
    clip_ops.move_item(tl, "v", 60, linked=False)
    assert _span(tl, "a") == (30, 90)


def test_splitting_a_linked_pair_splits_both_and_keeps_halves_linked():
    tl = _timeline(_clip("v", "V2", 0, 90, "L"), _clip("a", "A2", 0, 90, "L"))
    clip_ops.split_item(tl, "v", 30)
    assert len(tl.items) == 4
    lefts = [i for i in tl.items if i.timeline_end_frame == 30]
    rights = [i for i in tl.items if i.timeline_start_frame == 30]
    assert {i.track for i in lefts} == {"V2", "A2"}
    assert {i.track for i in rights} == {"V2", "A2"}
    assert len({i.link_id for i in lefts}) == 1
    assert len({i.link_id for i in rights}) == 1
    assert lefts[0].link_id != rights[0].link_id


def test_deleting_a_video_deletes_its_linked_audio():
    tl = _timeline(_clip("v", "V2", 0, 90, "L"), _clip("a", "A2", 0, 90, "L"),
                   _clip("other", "A3", 0, 90))
    clip_ops.delete_item(tl, "v")
    assert [i.id for i in tl.items] == ["other"]


def test_trimming_a_video_edge_trims_the_audio_on_the_same_edge():
    tl = _timeline(_clip("v", "V2", 0, 90, "L"), _clip("a", "A2", 0, 90, "L"))
    clip_ops.trim_item(tl, "v", "end", 60)
    assert _span(tl, "a") == (0, 60)
    clip_ops.trim_item(tl, "a", "start", 15)
    assert _span(tl, "v") == (15, 60)
    assert clip_ops.get_item(tl, "v").source_start_frame == 15


def test_link_and_unlink():
    tl = _timeline(_clip("v", "V2", 0, 30), _clip("a", "A2", 0, 30))
    clip_ops.link_items(tl, ["v", "a"])
    assert clip_ops.get_item(tl, "v").link_id == clip_ops.get_item(tl, "a").link_id
    clip_ops.unlink_items(tl, ["v"])
    assert clip_ops.get_item(tl, "v").link_id is None
    assert clip_ops.get_item(tl, "a").link_id is None


def test_detach_is_refused_when_the_audio_is_already_its_own_clip():
    tl = _timeline(_clip("v", "V2", 0, 30, "L"), _clip("a", "A2", 0, 30, "L"))
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.detach_audio(tl, "v")


# --- ripple ------------------------------------------------------------------

def test_ripple_delete_closes_the_hole_on_both_linked_tracks():
    tl = _timeline(
        _clip("v1", "V2", 0, 30, "A"), _clip("a1", "A2", 0, 30, "A"),
        _clip("v2", "V2", 30, 60, "B"), _clip("a2", "A2", 30, 60, "B"),
    )
    clip_ops.delete_items(tl, ["v1"], ripple=True)
    assert _span(tl, "v2") == (0, 30)
    assert _span(tl, "a2") == (0, 30)


def test_plain_delete_leaves_the_hole():
    tl = _timeline(_clip("x", "V2", 0, 30), _clip("y", "V2", 30, 60))
    clip_ops.delete_items(tl, ["x"])
    assert _span(tl, "y") == (30, 60)


def test_ripple_never_slides_into_a_clip_that_stays_put():
    # y slides left with its partner b, but A2 has a locked clip right before b.
    tl = _timeline(_clip("x", "V2", 0, 30), _clip("y", "V2", 30, 60, "L"),
                   _clip("pin", "A2", 0, 20), _clip("b", "A2", 30, 60, "L"))
    clip_ops.get_item(tl, "pin").locked = True
    clip_ops.delete_items(tl, ["x"], ripple=True)
    assert _span(tl, "y") == (20, 50)       # only as far as the pinned clip allows
    assert _span(tl, "b") == (20, 50)       # still in sync


def test_ripple_trim_end_pulls_the_next_clip_in():
    tl = _timeline(_clip("x", "V2", 0, 60), _clip("y", "V2", 60, 90))
    clip_ops.trim_item(tl, "x", "end", 40, ripple=True)
    assert _span(tl, "x") == (0, 40)
    assert _span(tl, "y") == (40, 70)


def test_ripple_trim_start_keeps_the_left_edge_in_place():
    tl = _timeline(_clip("x", "V2", 30, 90), _clip("y", "V2", 90, 120))
    clip_ops.trim_item(tl, "x", "start", 50, ripple=True)
    x = clip_ops.get_item(tl, "x")
    assert (x.timeline_start_frame, x.timeline_end_frame) == (30, 70)
    assert x.source_start_frame == 20
    assert _span(tl, "y") == (70, 100)


def test_trim_to_playhead_only_touches_clips_under_it():
    tl = _timeline(_clip("x", "V2", 0, 60), _clip("y", "V3", 100, 160))
    clip_ops.trim_items(tl, ["x", "y"], "end", 30)
    assert _span(tl, "x") == (0, 30)
    assert _span(tl, "y") == (100, 160)


def test_close_gap():
    tl = _timeline(_clip("x", "V2", 0, 30), _clip("y", "V2", 75, 100))
    closed = clip_ops.close_gap(tl, "V2", 50)
    assert closed == 45
    assert _span(tl, "y") == (30, 55)


def test_close_gap_refuses_a_clip():
    tl = _timeline(_clip("x", "V2", 0, 30), _clip("y", "V2", 75, 100))
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.close_gap(tl, "V2", 10)


# --- magnetic main track -----------------------------------------------------

def test_a_hand_built_v1_has_no_leading_gap():
    tl = _timeline()
    clip_ops.add_media_item(tl, "src", "V1", 90, 0, 60)
    assert tl.items[0].timeline_start_frame == 0


def test_moving_a_v1_clip_past_its_neighbour_reorders_and_stays_packed():
    tl = _timeline(_clip("a", "V1", 0, 30, "LA"), _clip("aa", "A1", 0, 30, "LA"),
                   _clip("b", "V1", 30, 90, "LB"), _clip("bb", "A1", 30, 90, "LB"))
    clip_ops.move_item(tl, "a", 70)
    assert _span(tl, "b") == (0, 60)
    assert _span(tl, "a") == (60, 90)
    assert _span(tl, "bb") == (0, 60)       # the sound followed its picture
    assert _span(tl, "aa") == (60, 90)


def test_deleting_from_v1_closes_up_without_asking():
    tl = _timeline(_clip("a", "V1", 0, 30), _clip("b", "V1", 30, 90))
    clip_ops.delete_item(tl, "a")
    assert _span(tl, "b") == (0, 60)


def test_dropping_on_a_clips_leading_half_inserts_before_it():
    tl = _timeline(_clip("a", "V1", 0, 60))
    frame = clip_ops.main_track_insert_frame(tl, 10)
    new = clip_ops.add_media_item(tl, "src", "V1", frame, 0, 30)
    assert _span(tl, new.id) == (0, 30)
    assert _span(tl, "a") == (30, 90)


def test_an_auto_spine_is_never_packed():
    tl = _timeline(_clip("a", "V1", 0, 30, origin="auto"), _clip("b", "V1", 60, 90, origin="auto"))
    clip_ops.set_aspect_bars(tl, None)        # any op runs _finalize
    assert _span(tl, "b") == (60, 90)


# --- paste ---------------------------------------------------------------------

def _copied(tl, *ids):
    return [clip_ops.get_item(tl, i).model_dump() for i in ids]


def test_paste_gives_fresh_ids_and_a_fresh_link():
    tl = _timeline(_clip("v", "V2", 0, 30, "L"), _clip("a", "A2", 0, 30, "L"))
    pasted = clip_ops.paste_items(tl, _copied(tl, "v", "a"), at_frame=100)
    assert {p.id for p in pasted}.isdisjoint({"v", "a"})
    assert {(p.timeline_start_frame, p.timeline_end_frame) for p in pasted} == {(100, 130)}
    assert len({p.link_id for p in pasted}) == 1
    assert pasted[0].link_id != "L"
    assert {p.track for p in pasted} == {"V2", "A2"}   # the lanes were free there


def test_paste_over_an_occupied_lane_moves_up_a_lane():
    tl = _timeline(_clip("v", "V2", 0, 30))
    pasted = clip_ops.paste_items(tl, _copied(tl, "v"), at_frame=10)
    assert pasted[0].track == "V3"
    assert _span(tl, "v") == (0, 30)


def test_paste_of_a_transcript_clip_becomes_a_manual_overlay():
    tl = _timeline(_clip("v", "V1", 0, 30, origin="auto"))
    pasted = clip_ops.paste_items(tl, _copied(tl, "v"), at_frame=0)
    assert pasted[0].origin == "manual"
    assert pasted[0].track == "V2"


def test_paste_insert_splits_and_pushes_the_lane_right():
    tl = _timeline(_clip("x", "V2", 0, 60), _clip("y", "V2", 70, 90))
    pasted = clip_ops.paste_items(tl, _copied(tl, "y"), at_frame=30, insert=True)
    v2 = sorted((i for i in tl.items if i.track == "V2"), key=lambda i: i.timeline_start_frame)
    spans = [(i.timeline_start_frame, i.timeline_end_frame) for i in v2]
    assert spans == [(0, 30), (30, 50), (50, 80), (90, 110)]
    assert pasted[0].track == "V2"


def test_paste_rejects_clips_whose_media_is_gone():
    tl = _timeline(_clip("v", "V2", 0, 30))
    copied = _copied(tl, "v")
    copied[0]["source_id"] = "src_missing"
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.paste_items(tl, copied, at_frame=0)
