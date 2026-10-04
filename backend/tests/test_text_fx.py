"""Text effects: the filtergraph fragments, the planner, the matte degrade
path, and face-avoidance."""

import random
import subprocess
from pathlib import Path

import pytest

from presentation import matte as matte_mod
from presentation import text_fx as tfx
from presentation import verify
from presentation.models import Beat, PresentationSettings
from presentation.program import build_program
from render import atmosphere
from timeline import build_timeline_from_transcript, clip_ops
from timeline.schema import AtmosphereEffect


# --- filtergraph fragments ---------------------------------------------------

def test_behind_head_is_text_between_two_video_layers_via_alphamerge():
    effect = AtmosphereEffect(type="behind_head", extra={
        "text": "TRUTH", "start_s": 1.0, "end_s": 3.0,
        "matte_path": "m.mp4", "matte_offset": 0.5})
    statements = atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 0,
                                         assets_dir=Path("."))
    joined = ";".join(statements)
    assert statements[0].startswith("[in]") and statements[-1].endswith("[out]")
    assert "drawtext=" in joined
    assert "movie='m.mp4'" in joined
    assert "alphamerge" in joined
    assert "split=2" in joined


def test_knockout_reveals_video_through_glyphs_on_black():
    effect = AtmosphereEffect(type="knockout", extra={"text": "BOOM", "start_s": 0.0, "end_s": 2.0})
    statements = atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 1,
                                         assets_dir=Path("."))
    joined = ";".join(statements)
    assert "color=black" in joined
    assert "drawtext=" in joined and "alpha='clip(" in joined     # the scale-in
    assert "alphamerge" in joined
    assert statements[-1].endswith("[out]")


def test_focus_with_matte_blurs_and_darkens_except_the_person():
    effect = AtmosphereEffect(type="focus", extra={
        "start_s": 0.0, "end_s": 2.0, "matte_path": "m.mp4", "matte_offset": 0.0})
    statements = atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 2)
    joined = ";".join(statements)
    assert "boxblur=" in joined
    assert "eq=brightness=" in joined
    assert "alphamerge" in joined


def test_focus_without_matte_falls_back_to_a_spotlight():
    effect = AtmosphereEffect(type="focus", extra={
        "start_s": 0.0, "end_s": 2.0, "face_x": 0.2, "face_y": -0.1})
    statements = atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 3)
    joined = ";".join(statements)
    assert "vignette=" in joined
    assert "matte" not in joined
    assert "boxblur" not in joined


@pytest.mark.parametrize("shape,needle", [
    # The underline swipes in on an overlay (x re-evaluated per frame); the
    # circle is a static drawbox outline -- drawbox geometry is fixed at
    # configure time, and `eval=frame` does not exist in this FFmpeg build.
    ("underline", "overlay=x='"), ("circle", "drawbox="),
])
def test_annotation_shapes_draw_on_near_the_word(shape, needle):
    effect = AtmosphereEffect(type="annotation", extra={
        "shape": shape, "start_s": 0.0, "end_s": 1.0, "cx": 0.5, "cy": 0.6})
    statements = atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 4)
    joined = ";".join(statements)
    assert needle in joined
    assert "eval=" not in joined


def test_annotation_arrow_draws_a_glyph_near_the_word():
    effect = AtmosphereEffect(type="annotation", extra={
        "shape": "arrow", "start_s": 0.0, "end_s": 1.0, "cx": 0.5, "cy": 0.6})
    statements = atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 5,
                                         assets_dir=Path("."))
    assert "drawtext=" in ";".join(statements)


def test_newspaper_sweep_is_a_bar_travelling_across_the_highlight_box():
    effect = AtmosphereEffect(type="newspaper_sweep", extra={
        "start_s": 1.0, "end_s": 1.6, "box": [0.1, 0.1, 0.6, 0.2]})
    statements = atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 6)
    joined = ";".join(statements)
    assert "overlay=x='" in joined and "clip((t-" in joined       # x animates per frame
    assert "eval=" not in joined
    assert any(st.startswith("[in]") for st in statements) and statements[-1].endswith("[out]")


def test_text_effects_degrade_to_a_pass_through_with_no_text_or_assets_dir():
    effect = AtmosphereEffect(type="behind_head", extra={"start_s": 0.0, "end_s": 1.0})
    assert atmosphere.build_effect(effect, "[in]", "[out]", 1920, 1080, 30.0, 10.0, 7) == ["[in]null[out]"]


def test_existing_atmosphere_effects_are_unaffected():
    """The pre-existing effects still take a plain positional call (no
    assets_dir) — behaviour for every caller before this pass is unchanged."""
    statements = atmosphere.build_effect(AtmosphereEffect(type="flash", intensity=0.6),
                                         "[in]", "[out]", 1920, 1080, 30.0, 10.0, 0)
    assert statements[0].startswith("[in]") and statements[0].endswith("[out]")


# --- the planner --------------------------------------------------------------

def _program(duration_s=180.0, gap=3.0):
    """A programme whose CUT duration (words only — the auto-edit trims the
    silence between them) is ~`duration_s`: each word is 0.5s of speech, so
    `duration_s / 0.5` of them, spaced `gap` seconds apart on the source before
    the cut removes the gaps."""
    words_list = ("this is an amazing incredible story about a huge massive discovery today "
                 "everyone absolutely should know because honestly nothing compares to this "
                 "unbelievable moment right now trust me it changes everything forever").split()
    n = int(duration_s / 0.5) + 4
    repeated = (words_list * (n // len(words_list) + 1))[:n]
    words = [{"word": w, "start": i * gap, "end": i * gap + 0.5} for i, w in enumerate(repeated)]
    timeline = build_timeline_from_transcript(
        "C:/media/talk.mp4", words[-1]["end"] + gap, words, fps_num=30, fps_den=1,
        width=1920, height=1080, speech_regions=[(0.0, words[-1]["end"] + gap)],
        pause_padding_seconds=0.0)
    program = build_program(timeline)
    rng = random.Random(0)
    for w in program.words:
        w.emphasis_z = rng.uniform(-1.0, 2.0)
    return timeline, program


def test_planner_count_follows_text_fx_per_minute():
    timeline, program = _program(duration_s=120.0)
    settings = PresentationSettings(text_fx=True, text_fx_per_minute=3.0)
    moments = tfx.plan_moments(program, [], timeline, [], settings, "vlog", {}, [], [], seed=1)
    assert len(moments) == round(3.0 * program.duration_s / 60.0)


def test_planner_off_switch_places_nothing():
    timeline, program = _program(duration_s=60.0)
    settings = PresentationSettings(text_fx=False, text_fx_per_minute=5.0)
    moments = tfx.plan_moments(program, [], timeline, [], settings, "vlog", {}, [], [], seed=1)
    assert moments == []


def test_planner_spaces_moments_at_least_six_seconds_apart():
    timeline, program = _program(duration_s=180.0)
    settings = PresentationSettings(text_fx=True, text_fx_per_minute=6.0)
    moments = tfx.plan_moments(program, [], timeline, [], settings, "general", {}, [], [], seed=2)
    assert len(moments) > 3
    ordered = sorted(moments, key=lambda m: m.start_s)
    for a, b in zip(ordered, ordered[1:]):
        assert b.start_s - a.end_s >= tfx.MIN_GAP_S - 1e-6


def test_planner_avoids_broll_and_card_windows():
    timeline, program = _program(duration_s=120.0)
    settings = PresentationSettings(text_fx=True, text_fx_per_minute=6.0)
    busy = [(0.0, program.duration_s)]           # the whole programme is "busy"
    moments = tfx.plan_moments(program, [], timeline, [], settings, "general", {}, busy, [], seed=3)
    assert moments == []


def test_planner_only_offers_styles_from_the_genre_palette():
    timeline, program = _program(duration_s=240.0)
    settings = PresentationSettings(text_fx=True, text_fx_per_minute=8.0)
    moments = tfx.plan_moments(program, [], timeline, [], settings, "documentary", {}, [], [], seed=4)
    styles = {m.style for m in moments}
    assert styles <= set(tfx.palette_for("documentary")[0]) or styles <= {
        s for s, _ in tfx.palette_for("documentary")}


def test_planner_restricts_to_explicit_text_fx_styles():
    timeline, program = _program(duration_s=120.0)
    settings = PresentationSettings(text_fx=True, text_fx_per_minute=6.0,
                                    text_fx_styles=["kinetic_words"])
    moments = tfx.plan_moments(program, [], timeline, [], settings, "horror", {}, [], [], seed=5)
    assert moments and {m.style for m in moments} == {"kinetic_words"}


def test_planner_favours_a_claim_beat_for_annotation():
    timeline, program = _program(duration_s=60.0)
    beat = Beat(kind="stat_callout", start_s=20.0, end_s=22.0, topic="a", text="25 percent",
               priority=0.9, origin="entity")
    settings = PresentationSettings(text_fx=True, text_fx_per_minute=1.0,
                                    text_fx_styles=["annotation"])
    moments = tfx.plan_moments(program, [beat], timeline, [], settings, "documentary", {}, [], [],
                               seed=6)
    assert len(moments) == 1 and moments[0].style == "annotation"
    assert moments[0].text == "25 percent"


# --- degrade without a matte --------------------------------------------------

def test_matte_module_degrades_to_off_with_no_backend(monkeypatch):
    monkeypatch.setattr(matte_mod, "detect_backend", lambda: "off")
    result = matte_mod.compute_person_matte("nope.mp4", [(0.0, 1.0)], Path("."), 100, 100)
    assert result is None


def test_apply_moments_skips_behind_head_and_softens_focus_without_a_matte(tmp_path):
    timeline, program = _program(duration_s=30.0)
    settings = PresentationSettings(text_fx=True, person_matte="off")
    moments = [
        tfx.Moment(1.0, 3.0, "behind_head", "loud"),
        tfx.Moment(10.0, 12.0, "focus", "word", face_x=0.1, face_y=-0.2),
        tfx.Moment(20.0, 22.0, "knockout", "big"),
    ]
    counts, backend, skipped = tfx.apply_moments(timeline, moments, settings, tmp_path,
                                                 None, 1920, 1080)
    assert backend == "off"
    assert counts == {"focus": 1, "knockout": 1}
    assert any(s["style"] == "behind_head" for s in skipped)
    focus_item = next(i for i in timeline.items if (i.label or "").startswith("text_fx focus"))
    assert focus_item.atmosphere[0].extra.get("matte_path") is None
    assert focus_item.atmosphere[0].extra.get("face_x") == pytest.approx(0.1)


def test_apply_moments_uses_a_real_matte_when_available(tmp_path, monkeypatch):
    from config import FFMPEG_BIN
    src = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=12:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not src.exists():
        pytest.skip("ffmpeg unavailable")

    # This exercises the CPU GrabCut path specifically; the torchvision path
    # (now real, and what a plain `detect_backend()` would pick when the venv
    # has it) has its own tests in test_matte_torchvision.py.
    monkeypatch.setattr(matte_mod, "detect_backend", lambda: "grabcut")

    timeline, program = _program(duration_s=6.0)
    settings = PresentationSettings(text_fx=True, person_matte="auto")
    moments = [tfx.Moment(1.0, 3.0, "behind_head", "loud")]
    counts, backend, skipped = tfx.apply_moments(timeline, moments, settings, tmp_path,
                                                 str(src), 320, 240)
    assert backend == "grabcut"
    assert counts == {"behind_head": 1}
    assert not skipped
    item = next(i for i in timeline.items if (i.label or "").startswith("text_fx behind_head"))
    assert Path(item.atmosphere[0].extra["matte_path"]).exists()


# --- face avoidance ------------------------------------------------------------

def test_dodge_face_clears_the_face_box_verify_checks():
    pos_x, pos_y = tfx._dodge_face(0.0, -0.2)          # face upper-centre
    cy = (pos_y + 1.0) / 2.0 * 1080
    face_cy = (-0.2 + 1.0) / 2.0 * 1080
    face_half_h = verify.FACE_HALF_H * 1080
    assert abs(cy - face_cy) > face_half_h + 60.0        # 60px: a caption's rough half-height

    pos_x2, pos_y2 = tfx._dodge_face(0.0, 0.4)           # face lower
    assert pos_y2 < 0.0 < pos_y                          # opposite bands


def test_kinetic_words_land_in_the_dodged_band(tmp_path):
    timeline, program = _program(duration_s=30.0)
    settings = PresentationSettings(text_fx=True)
    moment = tfx.Moment(5.0, 7.0, "kinetic_words", "big win", words=["big", "win"],
                        pos_x=0.0, pos_y=-0.68, face_x=0.0, face_y=0.2)
    tfx.apply_moments(timeline, [moment], settings, tmp_path, None, 1920, 1080)
    items = [i for i in timeline.items if i.origin == tfx.TEXT_FX_ORIGIN]
    assert items and all(i.text.style.pos_y == pytest.approx(-0.68) for i in items)


# --- a real render -------------------------------------------------------------

def test_a_knockout_effect_renders_for_real(tmp_path):
    """The whole point: the filtergraph is not just plausible-looking text —
    ffmpeg actually accepts and runs it."""
    from config import FFMPEG_BIN
    import asyncio
    from render.runner import render_timeline_async

    source = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x180:rate=25:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not source.exists():
        pytest.skip("ffmpeg unavailable")

    words = [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(8)]
    timeline = build_timeline_from_transcript(str(source), 4.0, words, fps_num=25, fps_den=1,
                                              width=320, height=180, pause_padding_seconds=0.0)
    adjustment = clip_ops.add_adjustment_item(timeline, 20, 50)     # 0.8s .. 2.8s @25fps
    adjustment.atmosphere = [AtmosphereEffect(type="knockout", enabled=True,
                                              extra={"text": "HI", "start_s": 0.8, "end_s": 2.8})]

    out = str(tmp_path / "out.mp4")
    asyncio.run(render_timeline_async(timeline, out, prefer_nvenc=False))
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_a_late_knockout_shows_the_word_not_a_black_frame(tmp_path):
    """Regression: the knockout's plate and mask were `color=` sources with
    their own clock from t=0, so a knockout later in the programme rendered
    solid black with no word. The glyphs must reveal the picture wherever the
    window sits."""
    from config import FFMPEG_BIN
    import asyncio
    from render.runner import render_timeline_async

    source = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=white:size=320x180:rate=25:duration=12",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not source.exists():
        pytest.skip("ffmpeg unavailable")

    words = [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(30)]
    timeline = build_timeline_from_transcript(str(source), 12.0, words, fps_num=25, fps_den=1,
                                              width=320, height=180, pause_padding_seconds=0.0)
    adjustment = clip_ops.add_adjustment_item(timeline, 225, 50)    # 9.0s .. 11.0s @25fps
    adjustment.atmosphere = [AtmosphereEffect(type="knockout", enabled=True, extra={
        "text": "WWW", "start_s": 9.0, "end_s": 11.0, "font_size": 110})]

    out = tmp_path / "out.mp4"
    asyncio.run(render_timeline_async(timeline, str(out), prefer_nvenc=False))
    frame = subprocess.run(
        [FFMPEG_BIN, "-v", "error", "-ss", "10.6", "-i", str(out), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE).stdout
    assert len(frame) == 320 * 180
    bright = sum(1 for b in frame if b > 128)
    # Black plate everywhere but the glyphs, which show the white source.
    assert 0.02 * len(frame) < bright < 0.8 * len(frame)


def test_filler_words_never_become_keyword_candidates():
    from presentation.text_fx import _is_content_word
    for filler in ("hain", "phir", "karna", "actually", "lekin", "kuch", "the"):
        assert not _is_content_word(filler)
    assert _is_content_word("perfection")


def test_claim_knockout_uses_the_marked_word_not_the_sentence():
    from presentation.text_fx import _hero_word
    assert _hero_word("Tumhara perfection ka *junoon* tumhe rok raha hai.") == "junoon"
    assert _hero_word("Perfection ek illusion hai.") == "Perfection"


def test_context_terms_come_from_the_plan():
    from presentation.models import Beat
    from presentation.text_fx import _context_terms
    beats = [Beat(start_s=0, end_s=2, kind="quote_card", text="Perfection ek illusion hai", topic="starting small")]
    terms = _context_terms(beats)
    assert {"perfection", "illusion", "starting", "small"} <= terms
    assert "hai" not in terms
