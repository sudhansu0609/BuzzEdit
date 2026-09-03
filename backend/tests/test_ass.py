"""The animated text engine: ASS script building, routing, and a real burn-in."""

import subprocess

import pytest

from backend.render import ass
from backend.render.compiler import FilterGraphCompiler
from backend.timeline import build_timeline_from_transcript, clip_ops, generate_captions
from backend.timeline.schema import TextClip, TextStyle


def _clip(animation: str, content: str = "Hello", **style) -> TextClip:
    return TextClip(content=content,
                    style=TextStyle(font_family="Arial", animation=animation, **style))


# --- primitives ------------------------------------------------------------------

def test_colours_and_times_take_the_ass_forms():
    assert ass.ass_colour("#FFE23A") == "&H003AE2FF"
    assert ass.ass_colour("white", 0.5) == "&H80FFFFFF"
    assert ass.ass_colour("black@0.6") == "&H66000000"
    assert ass.ass_colour("0x4FA8FFFF") == "&H00FFA84F"
    assert ass.ass_rgb("#FFE23A") == "&H3AE2FF&"
    assert ass.ass_time(0.0) == "0:00:00.00"
    assert ass.ass_time(83.456) == "0:01:23.46"
    assert ass.ass_time(3600.999) == "1:00:01.00"
    assert ass.escape_text("a {b}\nc") == "a (b)\\Nc"


def test_routing_sends_only_animated_clips_to_libass():
    assert ass.needs_ass(_clip("typewriter"))
    assert ass.needs_ass(_clip("glitch"))
    assert not ass.needs_ass(_clip("pop"))
    assert not ass.needs_ass(_clip("fade"))
    # Karaoke needs word timings; without them it is a plain caption.
    assert not ass.needs_ass(_clip("karaoke"))
    with_words = _clip("karaoke")
    with_words.words = [{"text": "Hello", "start_s": 0.0, "end_s": 0.4}]
    assert ass.needs_ass(with_words)


def test_style_line_carries_stroke_box_and_alignment():
    plain = TextStyle(font_family="Arial", font_size=64, color="#FF0000", stroke_width=3,
                      stroke_color="black", shadow_x=2, shadow_y=2, align="left", bold=True)
    line = ass.style_line("S1", plain)
    fields = line.split(",")
    assert fields[0].endswith("S1") and fields[2] == "64"
    assert fields[3] == "&H000000FF"            # red, BGR order
    assert fields[7] == "-1"                    # bold
    assert fields[15] == "1" and fields[16] == "3" and fields[17] == "2"
    assert fields[18] == "4"                    # left → middle-left anchor

    boxed = TextStyle(font_family="Arial", box=True, box_color="black@0.6", box_padding=14)
    fields = ass.style_line("S2", boxed).split(",")
    assert fields[15] == "3" and fields[16] == "14"
    assert fields[5] == "&H66000000" == fields[6]


# --- events ----------------------------------------------------------------------------

def test_typewriter_reveals_whole_devanagari_clusters_in_order():
    lines = ass.build_events(_clip("typewriter", "नमस्ते दो", animation_duration=0.9),
                             1.0, 5.0, "S", 1920, 1080)
    assert len(lines) == 1
    body = lines[0].split(",,")[-1]
    # न, म, स्ते, दो: the conjunct and the vowel signs stay with their base, so
    # four reveals, spaced across the 0.9s.
    reveals = body.count("\\alpha&HFF&\\t(")
    assert reveals == 4
    assert "स्ते" in body
    first = body.index("\\t(0,30")
    second = body.index("\\t(225,255")
    assert first < second


def test_karaoke_highlights_each_word_in_turn():
    clip = _clip("karaoke", "YEH KARAOKE HAI", highlight_color="#FFE23A", color="white")
    clip.words = [{"text": "YEH", "start_s": 0.0, "end_s": 0.4},
                  {"text": "KARAOKE", "start_s": 0.4, "end_s": 1.1},
                  {"text": "HAI", "start_s": 1.1, "end_s": 1.5}]
    lines = ass.build_events(clip, 10.0, 12.0, "S", 1920, 1080)
    body = lines[0].split(",,")[-1]
    assert body.count("\\c&H3AE2FF&") == 3           # each word lights up
    assert body.count("\\c&HFFFFFF&") == 6           # rests white, and goes back
    # Every span states its resting colour first, or the previous word's
    # animated yellow would carry into it.
    assert body.count("{\\c&HFFFFFF&\\fscx100\\fscy100\\t(") == 3
    assert "\\t(400,450,\\c&H3AE2FF&" in body
    assert "\\t(1100,1150,\\c&HFFFFFF&" in body
    assert lines[0].startswith("Dialogue: 0,0:00:10.00,0:00:12.00,S,")


def test_scale_blur_glitch_shake_and_flicker_build_their_tags():
    scale = ass.build_events(_clip("scale_in", animation_duration=0.35), 0, 3, "S", 1920, 1080)
    assert "\\fscx135\\fscy135\\t(0,350,\\fscx100\\fscy100)" in scale[0]
    assert "\\pos(960,540)" in scale[0]

    blur = ass.build_events(_clip("blur_in", animation_duration=0.8), 0, 3, "S", 1920, 1080)
    assert "\\blur14" in blur[0] and "\\blur0" in blur[0]

    glitch = ass.build_events(_clip("glitch"), 2.0, 5.0, "S", 1920, 1080)
    assert len(glitch) == 5                          # main + two copies × two bursts
    assert sum("\\c&H2020FF&" in line for line in glitch) == 2
    assert all(line.startswith("Dialogue: 1,") for line in glitch[1:])

    shake = ass.build_events(_clip("shake", animation_duration=0.4), 0, 3, "S", 1920, 1080)
    assert len(shake) >= 6
    positions = {line.split("\\pos(")[1].split(")")[0] for line in shake[:-1]}
    assert len(positions) > 1                        # it actually moves
    assert shake[-1].startswith("Dialogue: 0,0:00:00.40")

    flicker = ass.build_events(_clip("flicker"), 0, 2, "S", 1920, 1080)
    assert len(flicker) >= 10
    assert any("\\alpha&H00&" in line for line in flicker)
    assert any("\\alpha&H60&" in line or "\\alpha&H90&" in line for line in flicker)


def test_the_script_dedupes_styles_and_orders_events():
    a = _clip("scale_in", "A")
    b = _clip("scale_in", "B")
    c = _clip("blur_in", "C", color="#FF0000")
    script = ass.build_ass([(b, 5.0, 8.0), (a, 1.0, 3.0), (c, 2.0, 4.0)], 1280, 720)
    assert "PlayResX: 1280" in script and "PlayResY: 720" in script
    assert script.count("\nStyle: ") == 2
    events = [line for line in script.splitlines() if line.startswith("Dialogue:")]
    assert [e.split(",")[1] for e in events] == ["0:00:01.00", "0:00:02.00", "0:00:05.00"]


# --- compiler routing ----------------------------------------------------------------

def test_compiler_burns_animated_clips_with_one_ass_filter_and_the_rest_with_drawtext(tmp_path):
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(8)]
    timeline = build_timeline_from_transcript("C:/v.mp4", 6.0, words, pause_padding_seconds=0.0)
    clip_ops.add_text_item(timeline, "Typed", 0, 60, track="T1",
                           style={"font_family": "Arial", "animation": "typewriter"})
    clip_ops.add_text_item(timeline, "Faded", 30, 60, track="T1",
                           style={"font_family": "Arial", "animation": "fade"})
    _, graph, final_v, _ = FilterGraphCompiler(timeline, assets_dir=tmp_path).compile()
    assert graph.count("ass=filename=") == 1
    assert graph.count("drawtext=") == 1
    assert final_v == "[text_v]"
    assert list(tmp_path.glob("text_*.ass"))


def test_captions_carry_word_timings_and_the_karaoke_preset_is_routed():
    words = [{"word": "namaste", "start": 0.0, "end": 0.4},
             {"word": "dosto", "start": 0.5, "end": 0.9},
             {"word": "kaise", "start": 1.0, "end": 1.3},
             {"word": "ho", "start": 1.4, "end": 1.6}]
    timeline = build_timeline_from_transcript("C:/v.mp4", 3.0, words, pause_padding_seconds=0.0)
    items = generate_captions(timeline, preset="karaoke_pop")
    assert items
    first = items[0].text
    assert first.style.animation == "karaoke"
    assert [w["text"] for w in first.words][:2] == ["NAMASTE", "DOSTO"]
    assert first.words[0]["start_s"] == pytest.approx(0.0)
    assert first.words[1]["start_s"] == pytest.approx(0.5, abs=0.04)
    assert ass.needs_ass(first)


# --- end to end ------------------------------------------------------------------------------

def test_libass_burns_a_karaoke_caption_that_changes_over_time(tmp_path):
    """Real FFmpeg: the highlighted word moves, so two frames differ where the
    caption is drawn and nowhere else."""
    from backend.config import FFMPEG_BIN
    clip = TextClip(content="YEH KARAOKE HAI",
                    style=TextStyle(font_family="Arial", font_size=60, animation="karaoke",
                                    pos_y=0.0, stroke_width=3, color="white"))
    clip.words = [{"text": "YEH", "start_s": 0.0, "end_s": 0.8},
                  {"text": "KARAOKE", "start_s": 0.8, "end_s": 1.6},
                  {"text": "HAI", "start_s": 1.6, "end_s": 2.4}]
    script = ass.build_ass([(clip, 0.0, 3.0)], 640, 360)
    path = ass.write_ass_asset(script, tmp_path)
    out = tmp_path / "out.mp4"
    result = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i",
         "color=0x203040:s=640x360:d=3:r=30", "-vf", ass.build_ass_filter(path),
         "-frames:v", "75", "-pix_fmt", "yuv420p", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not out.exists():
        pytest.skip(f"ffmpeg/libass unavailable: {result.stderr[-200:]!r}")

    def frame_bytes(at: float) -> bytes:
        return subprocess.run(
            [FFMPEG_BIN, "-v", "error", "-ss", str(at), "-i", str(out), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    import numpy as np
    early = np.frombuffer(frame_bytes(0.4), dtype=np.uint8).reshape(360, 640, 3)
    late = np.frombuffer(frame_bytes(2.0), dtype=np.uint8).reshape(360, 640, 3)
    # Text is drawn: bright pixels where the background is a flat dark blue.
    assert int((early.min(axis=2) > 200).sum()) > 200

    # The yellow highlight has moved: the left third (first word) is yellow
    # early and white late; the right third the other way round.
    def yellow(frame, x0, x1):
        band = frame[150:210, x0:x1].astype(int)
        return int(((band[..., 0] > 180) & (band[..., 1] > 180) & (band[..., 2] < 140)).sum())

    assert yellow(early, 0, 260) > yellow(late, 0, 260)
    assert yellow(late, 380, 640) > yellow(early, 380, 640)
