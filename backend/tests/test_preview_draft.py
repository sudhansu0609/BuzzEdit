"""The draft FX proxy: draft_timeline (canvas + text scaled, generated video
B-roll swapped to its still, noise atmosphere dropped), the still lookup,
the windowed timeline and proxy naming, and the draft encoder profile."""
import json

import pytest

from render import encoder as encoder_mod
from render import preview
from render.effects import ZOOM_SS_OVERRIDE, _zoom_ss
from timeline.schema import (
    AtmosphereEffect,
    SourceFile,
    TextClip,
    TextStyle,
    Timeline,
    TimelineItem,
    Transform,
)

FPS = 30


def _gen_folder(tmp_path, images_for_beat=1):
    folder = tmp_path / "assets" / "generated"
    folder.mkdir(parents=True)
    (folder / "clip1.mp4").write_bytes(b"\0")
    (folder / "clip1.json").write_text(json.dumps({"beat_id": "b1", "kind": "broll_video"}))
    for n in range(images_for_beat):
        (folder / f"img{n}.png").write_bytes(b"\0")
        (folder / f"img{n}.json").write_text(json.dumps({"beat_id": "b1", "kind": "broll_image"}))
    # Another beat's image, and an image with no sidecar match, are ignored.
    (folder / "other.png").write_bytes(b"\0")
    (folder / "other.json").write_text(json.dumps({"beat_id": "b2", "kind": "broll_image"}))
    return folder


def _tl(clip_path="clip.mp4", total_s=200):
    total = total_s * FPS
    sources = {
        "main": SourceFile(id="main", path="main.mp4", duration_seconds=total_s, width=1920,
                           height=1080, has_audio=True, kind="video"),
        "gen": SourceFile(id="gen", path=str(clip_path), duration_seconds=5, width=832,
                          height=464, has_audio=False, kind="video"),
        "music": SourceFile(id="music", path="music.mp3", duration_seconds=60, has_audio=True,
                            kind="audio"),
    }
    items = [
        TimelineItem(id="v1", track="V1", source_id="main", source_start_frame=0,
                     source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total),
        TimelineItem(id="a1", track="A1", source_id="main", source_start_frame=0,
                     source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total),
        TimelineItem(id="broll", track="V3", source_id="gen", source_start_frame=0,
                     source_end_frame=5 * FPS, timeline_start_frame=10 * FPS,
                     timeline_end_frame=15 * FPS, origin="broll",
                     transform=Transform(scale=0.8, pos_x=0.2)),
        TimelineItem(id="cap", track="CAP", kind="text", timeline_start_frame=20 * FPS,
                     timeline_end_frame=22 * FPS,
                     text=TextClip(content="hello", style=TextStyle(font_size=64,
                                                                    stroke_width=4))),
        TimelineItem(id="mood", track="V8", kind="adjustment", timeline_start_frame=0,
                     timeline_end_frame=30 * FPS,
                     atmosphere=[AtmosphereEffect(type="fog"), AtmosphereEffect(type="flicker")]),
        TimelineItem(id="mus", track="A2", source_id="music", source_start_frame=0,
                     source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total,
                     loop=True, audio_fade_in=1.0, audio_fade_out=2.0),
    ]
    return Timeline(fps_num=FPS, fps_den=1, width=1920, height=1080, duration_frames=total,
                    sources=sources, items=items,
                    effects=[AtmosphereEffect(type="grain"), AtmosphereEffect(type="VHS"),
                             AtmosphereEffect(type="flash"), AtmosphereEffect(type="shake")])


# --- still lookup ---------------------------------------------------------

def test_still_for_video_finds_the_one_matching_image(tmp_path):
    folder = _gen_folder(tmp_path)
    assert preview.still_for_video(str(folder / "clip1.mp4")) == str(folder / "img0.png")


def test_still_for_video_ambiguous_or_missing_keeps_clip(tmp_path):
    folder = _gen_folder(tmp_path, images_for_beat=2)
    assert preview.still_for_video(str(folder / "clip1.mp4")) is None
    assert preview.still_for_video(str(tmp_path / "nowhere" / "x.mp4")) is None
    # An image's own sidecar is not a video clip.
    assert preview.still_for_video(str(folder / "img0.png")) is None


# --- draft_timeline -------------------------------------------------------

def test_draft_drops_noise_and_keeps_cheap_effects(tmp_path):
    draft = preview.draft_timeline(_tl())
    assert [e.type for e in draft.effects] == ["flash", "shake"]
    mood = next(i for i in draft.items if i.id == "mood")
    assert [e.type for e in mood.atmosphere] == ["flicker"]


def test_draft_scales_canvas_and_text():
    original = _tl()
    draft = preview.draft_timeline(original, height=540)
    assert (draft.width, draft.height) == (960, 540)
    style = next(i for i in draft.items if i.id == "cap").text.style
    assert style.font_size == 32 and style.stroke_width == 2
    # The input is left alone.
    assert next(i for i in original.items if i.id == "cap").text.style.font_size == 64
    assert original.effects[0].type == "grain"


def test_draft_swaps_generated_video_to_its_still(tmp_path):
    folder = _gen_folder(tmp_path)
    draft = preview.draft_timeline(_tl(clip_path=folder / "clip1.mp4"))
    broll = next(i for i in draft.items if i.id == "broll")
    src = draft.sources[broll.source_id]
    assert src.kind == "image" and src.path == str(folder / "img0.png") and not src.has_audio
    assert (broll.timeline_start_frame, broll.timeline_end_frame) == (10 * FPS, 15 * FPS)
    assert broll.transform.scale == 0.8 and broll.transform.pos_x == 0.2
    assert "gen" in draft.sources            # the original source entry is untouched


def test_draft_keeps_clip_without_a_still(tmp_path):
    draft = preview.draft_timeline(_tl(clip_path=tmp_path / "x.mp4"))
    assert next(i for i in draft.items if i.id == "broll").source_id == "gen"
    off = preview.draft_timeline(_tl(clip_path=_gen_folder(tmp_path) / "clip1.mp4"),
                                 stills_for_video=False)
    assert next(i for i in off.items if i.id == "broll").source_id == "gen"


# --- windows --------------------------------------------------------------

def test_window_timeline_snaps_outward_and_keeps_audio():
    tl = _tl()
    # 21 s is inside the caption (20-22 s) and the mood layer (0-30 s): both
    # are unsplittable, so the window starts at 0 s, before both.
    sub, a, b = preview.window_timeline(tl, 21.0, 100.0)
    assert a == 0 and b == 100 * FPS
    sub, a, b = preview.window_timeline(tl, 40.0, 100.0)
    assert (a, b) == (40 * FPS, 100 * FPS)
    assert sub.duration_frames == 60 * FPS
    tracks = {i.track for i in sub.items}
    assert {"V1", "A1", "A2"} <= tracks
    mus = next(i for i in sub.items if i.id == "mus")
    assert mus.timeline_start_frame == 0 and mus.duration_frames == 60 * FPS
    assert mus.source_start_frame == (40 * FPS) % (60 * FPS)
    assert mus.audio_fade_in == 0.0 and mus.audio_fade_out == 0.0
    a1 = next(i for i in sub.items if i.id == "a1")
    assert a1.source_start_frame == 40 * FPS and a1.source_end_frame == 100 * FPS
    assert sub.audio_master == tl.audio_master and "music" in sub.sources


def test_window_proxy_paths_coexist():
    full = preview.proxy_path("p1")
    win = preview.proxy_path("p1", (300.0, 420.0))
    assert full.name == "p1_fx.mp4"
    assert win.name == "p1_fx_w300000-420000.mp4"
    assert win != full and win.parent == full.parent


# --- encoder / zoom --------------------------------------------------------

def test_draft_encoder_profile_flags(monkeypatch):
    monkeypatch.setattr(encoder_mod, "get_nvenc_available", lambda: True)
    flags = encoder_mod.get_encoder_flags(True, profile="draft")
    joined = " ".join(flags)
    for part in ("-c:v h264_nvenc", "-preset p1", "-rc vbr", "-cq:v 32", "-b:v 0", "-g 15",
                 "-b:a 96k"):
        assert part in joined
    monkeypatch.setattr(encoder_mod, "get_nvenc_available", lambda: False)
    joined = " ".join(encoder_mod.get_encoder_flags(True, profile="draft"))
    for part in ("-c:v libx264", "-preset ultrafast", "-crf 30", "-g 15", "-b:a 96k"):
        assert part in joined


def test_final_encoder_profile_unchanged(monkeypatch):
    monkeypatch.setattr(encoder_mod, "get_nvenc_available", lambda: True)
    assert encoder_mod.get_encoder_flags(True) == encoder_mod.get_encoder_flags(True, "final") == [
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq:v", "21", "-b:v", "0",
        "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000", "-movflags", "+faststart"]
    monkeypatch.setattr(encoder_mod, "get_nvenc_available", lambda: False)
    assert encoder_mod.get_encoder_flags(True) == [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart"]


def test_zoom_supersample_override_is_scoped():
    assert _zoom_ss(1920, 1080, 1.0) == 3
    token = ZOOM_SS_OVERRIDE.set(1)
    try:
        assert _zoom_ss(1920, 1080, 1.0) == 1
    finally:
        ZOOM_SS_OVERRIDE.reset(token)
    assert _zoom_ss(1920, 1080, 1.0) == 3
