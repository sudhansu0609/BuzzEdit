import pytest
from backend.timeline.schema import Timeline, SourceFile, TimelineItem
from backend.timeline import clip_ops
from backend.render.compiler import FilterGraphCompiler


def _base_timeline():
    src = SourceFile(id="src_main", path="C:/media/main.mp4", duration_seconds=20.0,
                     width=1920, height=1080, fps_num=30, fps_den=1, kind="video")
    v1 = TimelineItem(id="v1_0", track="V1", source_id="src_main",
                      source_start_frame=0, source_end_frame=150,
                      timeline_start_frame=0, timeline_end_frame=150, origin="auto")
    a1 = TimelineItem(id="a1_0", track="A1", source_id="src_main",
                      source_start_frame=0, source_end_frame=150,
                      timeline_start_frame=0, timeline_end_frame=150, origin="auto")
    tl = Timeline(fps_num=30, fps_den=1, sources={"src_main": src}, items=[v1, a1])
    tl.recalculate_duration()
    return tl


def _add_manual_clip(tl, track="V2", start=30, src_start=0, src_end=60):
    return clip_ops.add_media_item(tl, "src_main", track, start, src_start, src_end)


def test_auto_items_are_protected():
    tl = _base_timeline()
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.delete_item(tl, "v1_0")
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.split_item(tl, "a1_0", 50)


def test_add_and_next_track():
    tl = _base_timeline()
    assert clip_ops.next_track(tl, "V") == "V2"
    assert clip_ops.next_track(tl, "A") == "A2"
    _add_manual_clip(tl, "V2")
    assert clip_ops.next_track(tl, "V") == "V3"


def test_split_manual_clip():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2", start=30, src_start=0, src_end=60)  # tl 30..90
    left, right = clip_ops.split_item(tl, clip.id, 60)
    assert left.timeline_start_frame == 30 and left.timeline_end_frame == 60
    assert right.timeline_start_frame == 60 and right.timeline_end_frame == 90
    # source split point continuity
    assert left.source_end_frame == right.source_start_frame == 30
    assert clip_ops.get_item(tl, clip.id) is None


def test_split_outside_bounds_rejected():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2", start=30, src_start=0, src_end=60)
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.split_item(tl, clip.id, 30)  # exactly on edge
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.split_item(tl, clip.id, 200)


def test_move_preserves_duration_and_blocks_cross_kind():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2", start=30, src_start=0, src_end=60)
    dur = clip.timeline_end_frame - clip.timeline_start_frame
    moved = clip_ops.move_item(tl, clip.id, 100, "V3")
    assert moved.track == "V3"
    assert moved.timeline_start_frame == 100
    assert moved.timeline_end_frame - moved.timeline_start_frame == dur
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.move_item(tl, clip.id, 0, "A2")  # video -> audio


def test_trim_edges():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2", start=30, src_start=10, src_end=70)  # tl 30..90
    clip_ops.trim_item(tl, clip.id, "start", 40)  # +10 frames
    assert clip.timeline_start_frame == 40 and clip.source_start_frame == 20
    clip_ops.trim_item(tl, clip.id, "end", 80)    # -10 frames
    assert clip.timeline_end_frame == 80 and clip.source_end_frame == 60


def test_trim_past_source_rejected():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2", start=0, src_start=0, src_end=60)
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.trim_item(tl, clip.id, "start", -10)  # negative timeline -> before source start


def test_delete_manual_clip():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2")
    clip_ops.delete_item(tl, clip.id)
    assert clip_ops.get_item(tl, clip.id) is None


def test_compiler_includes_overlay_and_audio_mix():
    tl = _base_timeline()
    # add a second video source (image) and audio source
    tl.sources["src_img"] = SourceFile(id="src_img", path="C:/media/broll.png",
                                       duration_seconds=5, width=1920, height=1080, kind="image")
    tl.sources["src_music"] = SourceFile(id="src_music", path="C:/media/music.mp3",
                                         duration_seconds=30, kind="audio")
    clip_ops.add_media_item(tl, "src_main", "V2", 30, 0, 60)     # video overlay
    clip_ops.add_media_item(tl, "src_img", "V3", 90, 0, 150)     # image overlay
    clip_ops.add_media_item(tl, "src_music", "A2", 0, 0, 900)    # audio mix

    inputs, fc, vlabel, alabel = FilterGraphCompiler(tl).compile()
    # video clip overlay uses PTS offset; image overlay uses raw stream
    assert "overlay=" in fc
    assert "setpts=PTS-STARTPTS+" in fc   # positioned video clip
    # audio mix present
    assert "amix=inputs=2:normalize=0" in fc
    assert "adelay=" in fc
    assert alabel == "[mix_a]"


# --- tracks (layers) ------------------------------------------------------

def test_track_flags_round_trip_and_reserve_empty_lane():
    tl = _base_timeline()
    state = clip_ops.set_track_flags(tl, "V1", hidden=True, muted=True)
    assert state.hidden and state.muted and not state.locked
    # a flagged lane with no clips is remembered so the UI keeps showing it
    clip_ops.set_track_flags(tl, "A3", muted=True)
    assert "A3" in tl.extra_tracks
    assert "V1" not in tl.extra_tracks     # it already holds clips


def test_locked_track_refuses_edits():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2")
    clip_ops.set_track_flags(tl, "V2", locked=True)
    assert clip_ops.is_track_locked(tl, "V2")
    for call in (
        lambda: clip_ops.delete_item(tl, clip.id),
        lambda: clip_ops.split_item(tl, clip.id, 45),
        lambda: clip_ops.move_item(tl, clip.id, 60),
        lambda: clip_ops.trim_item(tl, clip.id, "end", 70),
        lambda: clip_ops.add_media_item(tl, "src_main", "V2", 200, 0, 30),
    ):
        with pytest.raises(clip_ops.ClipOpError):
            call()
    clip_ops.set_track_flags(tl, "V2", locked=False)
    clip_ops.move_item(tl, clip.id, 60)     # unlocked again


def test_delete_track_removes_its_clips():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2")
    clip_ops.set_track_flags(tl, "V2", hidden=True)
    assert clip_ops.delete_track(tl, "V2") == 1
    assert clip_ops.get_item(tl, clip.id) is None
    assert "V2" not in tl.tracks and "V2" not in tl.extra_tracks


def test_delete_locked_track_is_refused():
    tl = _base_timeline()
    _add_manual_clip(tl, "V2")
    clip_ops.set_track_flags(tl, "V2", locked=True)
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.delete_track(tl, "V2")


def test_hidden_and_muted_tracks_drop_out_of_the_render():
    tl = _base_timeline()
    tl.sources["src_music"] = SourceFile(id="src_music", path="C:/media/music.mp3",
                                         duration_seconds=30, kind="audio")
    clip_ops.add_media_item(tl, "src_main", "V2", 30, 0, 60)
    clip_ops.add_media_item(tl, "src_music", "A2", 0, 0, 900)

    _, fc, _, alabel = FilterGraphCompiler(tl).compile()
    assert "overlay=" in fc and "amix=inputs=2" in fc

    clip_ops.set_track_flags(tl, "V2", hidden=True)
    clip_ops.set_track_flags(tl, "A2", muted=True)
    _, fc, _, alabel = FilterGraphCompiler(tl).compile()
    assert "overlay=" not in fc
    assert "amix=" not in fc     # only A1 is left, so no mixing is needed


# --- adjustment layers -----------------------------------------------------

def _adjustment(tl, start=60, frames=90, **kw):
    return clip_ops.add_adjustment_item(tl, start, frames, **kw)


def test_adjustment_takes_the_next_video_track_and_refuses_the_wrong_ones():
    tl = _base_timeline()
    assert _adjustment(tl).track == "V2"
    with pytest.raises(clip_ops.ClipOpError):
        _adjustment(tl, track="V1")     # nothing below it to adjust
    with pytest.raises(clip_ops.ClipOpError):
        _adjustment(tl, track="A2")
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.add_adjustment_item(tl, 0, 0)


def test_adjustment_can_be_moved_trimmed_and_split_like_any_clip():
    tl = _base_timeline()
    adj = _adjustment(tl)
    clip_ops.move_item(tl, adj.id, 120)
    clip_ops.trim_item(tl, adj.id, "end", 240)
    left, right = clip_ops.split_item(tl, adj.id, 180)
    assert left.kind == right.kind == "adjustment"
    # It carries no source, so trimming must not try to walk one.
    assert left.source_id is None


def test_a_neutral_adjustment_compiles_to_nothing():
    tl = _base_timeline()
    _adjustment(tl)
    _, fc, _, _ = FilterGraphCompiler(tl).compile()
    assert "split" not in fc and "adj" not in fc


def test_adjustment_treats_the_picture_only_inside_its_window():
    tl = _base_timeline()
    adj = _adjustment(tl, start=60, frames=90)      # 2s .. 5s at 30fps
    clip_ops.set_color(tl, adj.id, {"saturation": 0.0})

    _, fc, vlabel, _ = FilterGraphCompiler(tl).compile()
    assert "split[adjbase_0][adjsrc_0]" in fc
    assert "saturation=0" in fc
    assert "overlay=x=0:y=0:eof_action=pass:enable='between(t,2.000,5.000)'" in fc
    # Only the window is processed, not the whole programme.
    assert "trim=start=2.000:end=5.000" in fc
    assert vlabel == "[adj_0]"


def test_an_adjustment_reaches_the_layers_below_it_but_not_above():
    """Stacking order is the whole point: V2's adjustment must be applied before
    the V3 overlay is composited, and V4's after it."""
    tl = _base_timeline()
    tl.sources["src_img"] = SourceFile(id="src_img", path="C:/media/broll.png",
                                       duration_seconds=5, width=1920, height=1080, kind="image")
    clip_ops.add_media_item(tl, "src_img", "V3", 0, 0, 150)
    low = _adjustment(tl, track="V2")
    high = _adjustment(tl, track="V4")
    clip_ops.set_color(tl, low.id, {"saturation": 0.0})
    clip_ops.set_color(tl, high.id, {"contrast": 1.6})

    _, fc, _, _ = FilterGraphCompiler(tl).compile()
    below = fc.index("saturation=0")
    overlay = fc.index("overlay=x='")          # the V3 B-roll composite
    above = fc.index("contrast=1.6")
    assert below < overlay < above


def test_effects_attach_to_an_adjustment_layer_and_are_windowed():
    tl = _base_timeline()
    adj = _adjustment(tl)
    clip_ops.add_effect(tl, None, {"type": "rain"}, item_id=adj.id)
    assert len(clip_ops.get_item(tl, adj.id).atmosphere) == 1
    assert tl.effects == []                    # not the programme's

    _, fc, _, _ = FilterGraphCompiler(tl).compile()
    assert "blend=all_mode=screen" in fc
    assert "enable='between(t," in fc

    clip_ops.remove_effect(tl, 0, item_id=adj.id)
    assert clip_ops.get_item(tl, adj.id).atmosphere == []


def test_effects_cannot_be_hung_on_an_ordinary_clip():
    tl = _base_timeline()
    clip = _add_manual_clip(tl, "V2")
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.add_effect(tl, None, {"type": "rain"}, item_id=clip.id)


def test_adjustment_opacity_is_the_strength_of_the_treatment():
    tl = _base_timeline()
    adj = _adjustment(tl)
    clip_ops.set_color(tl, adj.id, {"saturation": 0.0})
    clip_ops.set_transform(tl, adj.id, {"opacity": 0.4})

    _, fc, _, _ = FilterGraphCompiler(tl).compile()
    assert "colorchannelmixer=aa=0.4" in fc
    # Opacity alone is not a geometry change, so no scale/pad/crop is emitted.
    # No geometry pad (a `pad=W:H` after scale) -- `tpad=`, the time pad on the
    # treated copy, is not one.
    assert ",pad=" not in fc and "]pad=" not in fc


def test_hiding_the_track_switches_the_adjustment_off():
    tl = _base_timeline()
    adj = _adjustment(tl, track="V2")
    clip_ops.set_color(tl, adj.id, {"saturation": 0.0})
    clip_ops.set_track_flags(tl, "V2", hidden=True)
    _, fc, _, _ = FilterGraphCompiler(tl).compile()
    assert "saturation=0" not in fc


def _cut_timeline(segments):
    src = SourceFile(id="src_main", path="C:/media/main.mp4", duration_seconds=600.0,
                     width=1920, height=1080, fps_num=30, fps_den=1, kind="video")
    items, t = [], 0
    for n in range(segments):
        s0 = n * 90                       # a 2s keep out of every 3s of source
        for kind in ("V1", "A1"):
            items.append(TimelineItem(id=f"{kind}_{n}", track=kind, source_id="src_main",
                                      source_start_frame=s0, source_end_frame=s0 + 60,
                                      timeline_start_frame=t, timeline_end_frame=t + 60,
                                      origin="auto"))
        t += 60
    tl = Timeline(fps_num=30, fps_den=1, sources={"src_main": src}, items=items)
    tl.recalculate_duration()
    return tl


def test_a_many_cut_edit_seeks_one_input_per_segment():
    """150 trims off one decode ran at 0.28x realtime; seeked inputs keep the
    cost proportional to the programme, not to the number of cuts."""
    inputs, fc, _, _ = FilterGraphCompiler(_cut_timeline(20)).compile()
    assert inputs.count("-ss") == 20                 # one per segment, shared by V1+A1
    assert inputs[inputs.index("-ss") + 1] == "0.000"
    assert "-threads" in inputs
    assert "trim=start=" not in fc.split("[v1_19]")[0]   # V1 no longer trims the shared stream
    assert "[1:v]trim=duration=2.000" in fc and "[1:a]atrim=duration=2.000" in fc


def test_a_short_edit_keeps_the_single_shared_input():
    inputs, fc, _, _ = FilterGraphCompiler(_cut_timeline(3)).compile()
    assert "-ss" not in inputs
    assert "[0:v]trim=start=0.000:end=2.000" in fc
