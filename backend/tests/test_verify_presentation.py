"""Checks on the dressed programme: geometry, stacking, the mix, the render."""

import subprocess
from types import SimpleNamespace

import pytest

from backend.presentation import verify
from backend.presentation.models import PresentationSettings
from backend.presentation.program import build_program
from backend.timeline import build_timeline_from_transcript, clip_ops
from backend.timeline.schema import SourceFile


def _timeline(seconds: float = 30.0):
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i in range(int(seconds * 2) - 1)]
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", seconds, words, fps_num=30, fps_den=1, width=1920, height=1080,
        speech_regions=[(0.0, seconds)], pause_padding_seconds=0.0)


def test_text_boxes_follow_alignment_and_size():
    timeline = _timeline()
    item = clip_ops.add_text_item(timeline, "HELLO", 0, 30, track="T1",
                                  style={"font_size": 100, "pos_x": 0.0, "pos_y": 0.0})
    x0, y0, x1, y1 = verify.text_box(item, 1920, 1080)
    assert x1 - x0 == pytest.approx(5 * 100 * verify.CHAR_ADVANCE + 2 * item.text.style.box_padding)
    assert (x0 + x1) / 2 == pytest.approx(960) and (y0 + y1) / 2 == pytest.approx(540)
    left = clip_ops.add_text_item(timeline, "HELLO", 0, 30, track="T1",
                                  style={"font_size": 100, "pos_x": -0.9, "align": "left"})
    lx0, _, _, _ = verify.text_box(left, 1920, 1080)
    assert lx0 == pytest.approx(0.05 * 1920)


def test_text_over_the_face_is_caught_but_not_under_broll_or_in_the_caption_band():
    timeline = _timeline()
    program = build_program(timeline)
    face = {program.segments[0].item_id: (0.0, -0.2)}       # face upper-centre
    # A stat right on the face, a card in the corner, and a caption in the band.
    stat = clip_ops.add_text_item(timeline, "25%", 60, 90, track="TX",
                                  style={"font_size": 150, "pos_y": -0.2})
    stat.label = "stat: 25%"
    corner = clip_ops.add_text_item(timeline, "Cornell", 60, 90, track="TX",
                                    style={"font_size": 44, "pos_x": -0.92, "pos_y": -0.82,
                                           "align": "left"})
    corner.label = "location_card: Cornell"
    caption = clip_ops.add_text_item(timeline, "hello there", 60, 90, track="TC",
                                     style={"font_size": 84, "pos_y": 0.4})
    caption.origin = "caption"

    checks = {c.name: c for c in verify.verify_timeline(timeline, program, PresentationSettings(), face)}
    assert not checks["text_clear_of_face"].ok
    assert "stat: 25%" in checks["text_clear_of_face"].detail
    assert "Cornell" not in checks["text_clear_of_face"].detail
    assert checks["speaker_on_screen"].ok
    assert checks["assets_present"].ok

    # The same stat over a cutaway is fine: the face is not on screen.
    timeline.sources["b"] = SourceFile(id="b", path=__file__, duration_seconds=5, width=1920,
                                       height=1080, has_audio=False, kind="image")
    broll = clip_ops.add_media_item(timeline, "b", "V3", 45, 0, 120, origin="broll")
    checks = {c.name: c for c in verify.verify_timeline(timeline, program, PresentationSettings(), face)}
    assert checks["text_clear_of_face"].ok


def test_stacked_text_and_a_loud_bed_are_flagged():
    timeline = _timeline()
    program = build_program(timeline)
    a = clip_ops.add_text_item(timeline, "ONE", 60, 90, track="TX", style={"font_size": 120})
    a.label = "chapter_title: ONE"
    b = clip_ops.add_text_item(timeline, "TWO", 75, 90, track="TX", style={"font_size": 120})
    b.label = "quote_card: TWO"
    timeline.sources["m"] = SourceFile(id="m", path="C:/bed.wav", duration_seconds=5,
                                       has_audio=True, kind="audio", width=0, height=0)
    bed = clip_ops.add_media_item(timeline, "m", "A2", 0, 0, 300, origin="music")
    bed.volume = 1.0
    bed.duck = 0.0
    checks = {c.name: c for c in verify.verify_timeline(timeline, program, PresentationSettings())}
    assert not checks["text_not_stacked"].ok and "ONE" in checks["text_not_stacked"].detail
    assert not checks["music_under_voice"].ok
    assert "not ducked" in checks["music_under_voice"].detail
    bed.volume = 0.16
    bed.duck = 0.85
    checks = {c.name: c for c in verify.verify_timeline(timeline, program, PresentationSettings())}
    assert checks["music_under_voice"].ok


def test_loudness_is_parsed_from_ebur128_output(monkeypatch):
    summary = ("[Parsed_ebur128_0 @ x] Summary:\n\n  Integrated loudness:\n"
               "    I:         -13.4 LUFS\n    Threshold: -23.9 LUFS\n\n  Loudness range:\n"
               "    LRA:         6.2 LU\n    Threshold: -33.7 LUFS\n    LRA low:   -17.9 LUFS\n"
               "    LRA high:  -11.7 LUFS\n\n  True peak:\n    Peak:       -1.2 dBFS\n")
    monkeypatch.setattr(verify.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="", stderr=summary, returncode=0))
    measured = verify.measure_loudness("x.mp4")
    assert measured == {"integrated": -13.4, "range": 6.2, "peak": -1.2}


def test_the_render_checks_run_on_a_real_file(tmp_path):
    from backend.config import FFMPEG_BIN
    out = tmp_path / "x.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2,volume=0.3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0:
        pytest.skip("ffmpeg unavailable")
    checks = {c.name: c for c in verify.verify_render(str(out), PresentationSettings(loudness_lufs=-14))}
    assert checks["render_exists"].ok
    assert checks["loudness"].value is not None
    assert checks["av_length"].ok
    missing = verify.verify_render(str(tmp_path / "nope.mp4"), PresentationSettings())
    assert not missing[0].ok
