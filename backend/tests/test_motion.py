"""The Remotion motion-template engine's Python side (MOTION_GRAPHICS_PLAN.md
M1): theme building, the kinetic_text builder, the real render/fallback path,
and how placement treats a motion asset as a graphic rather than a clip.

A handful of these run the real Node engine (motion/render.mjs) -- skipped
automatically when it is not installed (`npm install` in motion/), never
mocked, because the one thing this module must prove is that a real render
produces a real file and a real failure falls back cleanly.
"""

import pytest

from presentation import motion
from presentation.models import Asset, Beat, PresentationSettings, Program
from presentation.placement import CLIP_NEEDS_5S_REASON, _held_window, _window_for

pytestmark = pytest.mark.skipif(not motion.available(),
                                reason="motion engine not installed (npm install in motion/)")


def _beat(**kwargs) -> Beat:
    base = dict(start_s=0.0, end_s=6.0, topic="a topic", kind="kinetic_text",
               priority=1.0, text="Yeh kahani hai sach *mein* darawani")
    base.update(kwargs)
    return Beat(**base)


def _ctx(**kwargs) -> motion.MotionContext:
    base = dict(width=320, height=180, fps=30.0, genre="horror",
               theme=motion.theme_for("horror", "real"), settings=PresentationSettings())
    base.update(kwargs)
    return motion.MotionContext(**base)


# --- theme ------------------------------------------------------------------

def test_theme_for_builds_a_complete_hex_theme():
    theme = motion.theme_for("horror", "real")
    for key in ("bg", "panel", "fg", "muted", "accent", "accent2", "water", "land", "border"):
        assert theme[key].startswith("#") and len(theme[key]) == 7, key
    assert theme["noir"] is True
    assert theme["fontHead"] and theme["fontBody"]

    clean = motion.theme_for("education", "real")
    assert clean["noir"] is False
    assert clean["bg"] != theme["bg"]


def test_theme_for_marks_hand_drawn_art_styles_as_paper():
    assert motion.theme_for("general", "stick_figure")["paper"] is True
    assert motion.theme_for("general", "real")["paper"] is False


# --- the kinetic_text builder -------------------------------------------------

def test_parse_marks_splits_marked_words():
    words = motion._parse_marks("This is *huge*")
    assert words == [{"text": "This", "mark": False}, {"text": "is", "mark": False},
                     {"text": "huge", "mark": True}]


def test_builders_registry_has_kinetic_text():
    assert motion.BUILDERS["kinetic_text"] is motion.build_kinetic_text


def test_build_kinetic_text_clamps_duration_and_carries_the_theme():
    ctx = _ctx()
    beat = _beat(planned_duration_s=30.0)  # over MOTION_BEAT_SECONDS_MAX
    job = motion.build_kinetic_text(beat, ctx)
    assert job["composition"] == "KineticText"
    assert job["props"]["theme"] == ctx.theme
    assert job["props"]["style"] == "slam"
    assert job["frames"] == round(motion.MOTION_BEAT_SECONDS_MAX * ctx.fps)
    assert job["transparent"] is False

    too_short = motion.build_kinetic_text(_beat(planned_duration_s=0.4), ctx)
    assert too_short["frames"] == round(motion.MOTION_BEAT_SECONDS_MIN * ctx.fps)


def test_build_kinetic_text_rejects_an_empty_beat():
    assert motion.build_kinetic_text(_beat(text="", topic=""), _ctx()) is None


def test_build_kinetic_text_falls_back_to_slam_for_an_unknown_style():
    job = motion.build_kinetic_text(_beat(data={"style": "explode"}), _ctx())
    assert job["props"]["style"] == "slam"


# --- the flow_diagram / icon_list builders (MOTION_GRAPHICS_PLAN.md M9) ----

def test_build_flow_diagram_and_icon_list_carry_their_props_and_reject_too_few():
    ctx = _ctx()
    flow = motion.BUILDERS["flow_diagram"](
        _beat(kind="flow_diagram", data={"steps": ["A", "B", "C"], "layout": "branch"}), ctx)
    assert flow["composition"] == "FlowDiagram"
    assert flow["props"]["steps"] == ["A", "B", "C"]
    assert flow["props"]["layout"] == "branch"
    assert flow["props"]["theme"] == ctx.theme

    too_few_steps = motion.BUILDERS["flow_diagram"](
        _beat(kind="flow_diagram", data={"steps": ["only one"]}), ctx)
    assert too_few_steps is None

    icons = motion.BUILDERS["icon_list"](
        _beat(kind="icon_list", data={"items": ["+Lower cost", "-Takes longer"], "mode": "pros_cons"}), ctx)
    assert icons["composition"] == "IconList"
    assert icons["props"]["mode"] == "pros_cons"
    assert icons["props"]["items"] == [
        {"label": "Lower cost", "icon": "wallet", "side": "pro"},
        {"label": "Takes longer", "icon": "hourglass", "side": "con"},
    ]

    too_few_items = motion.BUILDERS["icon_list"](
        _beat(kind="icon_list", data={"items": ["only one"]}), ctx)
    assert too_few_items is None


# --- the real render ----------------------------------------------------------

def test_render_motion_assets_produces_a_real_video_file(tmp_path):
    # Below MOTION_BEAT_SECONDS_MIN, so the asset is clamped to it.
    beat = _beat(id="b00", planned_duration_s=1.0)
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "horror", PresentationSettings())

    assert failures == []
    assert handled == {"b00"}
    assert len(assets) == 1
    asset = assets[0]
    assert asset.beat_id == "b00" and asset.kind == "video" and asset.source == "motion"
    assert asset.duration_s == pytest.approx(motion.MOTION_BEAT_SECONDS_MIN, abs=0.05)
    from pathlib import Path
    assert Path(asset.path).exists() and Path(asset.path).stat().st_size > 0


def test_render_motion_assets_caches_by_content(tmp_path):
    beat = _beat(id="b00", planned_duration_s=1.0)
    settings = PresentationSettings()
    first, _, _ = motion.render_motion_assets([beat], tmp_path, 320, 180, "horror", settings)
    second, _, _ = motion.render_motion_assets([beat], tmp_path, 320, 180, "horror", settings)
    assert first[0].path == second[0].path
    assert first[0].cache_hit is False and second[0].cache_hit is True


def test_cache_key_changes_when_the_engine_fingerprint_changes(monkeypatch):
    """FX1: a composition/dependency edit under motion/src must invalidate
    stale cached renders instead of silently reusing them."""
    spec = {"composition": "KineticText", "props": {}, "width": 320, "height": 180,
           "fps": 30.0, "frames": 10}
    monkeypatch.setattr(motion, "_engine_fingerprint_cache", "aaaaaaaaaaaa")
    monkeypatch.setattr(motion, "_engine_fingerprint", lambda: "aaaaaaaaaaaa")
    key1 = motion._cache_key(spec)
    monkeypatch.setattr(motion, "_engine_fingerprint", lambda: "bbbbbbbbbbbb")
    key2 = motion._cache_key(spec)
    assert key1 != key2


def test_render_motion_assets_reports_a_failure_per_beat_when_motion_graphics_is_off(tmp_path):
    # Still a failure (not a silent no-op): the beat came from a `[kinetic:]`
    # the user wrote, and director._draw_local_assets only tries a fallback
    # for a beat it finds in `failures`.
    beat = _beat(id="b00")
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "horror", PresentationSettings(motion_graphics=False))
    assert assets == [] and handled == set()
    assert failures == [{"beat_id": "b00", "reason": "motion graphics are off"}]


def test_render_motion_assets_skips_beats_no_builder_knows(tmp_path):
    beat = _beat(id="b00", kind="broll_image")
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "horror", PresentationSettings())
    assert (assets, failures, handled) == ([], [], set())


def test_render_motion_assets_leaves_the_beat_for_fallback_when_the_engine_is_unavailable(
        tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "available", lambda: False)
    beat = _beat(id="b00")
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "horror", PresentationSettings())
    assert assets == [] and handled == set()
    assert failures == [{"beat_id": "b00", "reason": "motion engine not installed"}]


# --- fallback ------------------------------------------------------------------

def test_render_fallback_draws_a_static_statement_style_card(tmp_path):
    beat = _beat(id="b00")
    asset = motion.render_fallback(beat, tmp_path, 320, 180)
    assert asset is not None
    assert asset.kind == "image" and asset.source == "motion_fallback" and asset.beat_id == "b00"
    from pathlib import Path
    assert Path(asset.path).exists()


def test_render_fallback_uses_the_parent_collages_own_sub_picture_or_drops_it(tmp_path):
    """paper_collage's fallback (M5) is one of its own already-generated
    sub-pictures -- stashed on `beat.data["fallback_image"]` by motion_
    collage.annotate_collage_assets right after that generation pass -- used
    as a plain still; a beat annotate_collage_assets never reached (no real
    sub-picture at all) has nothing to fall back to and is dropped."""
    from PIL import Image
    fallback_png = tmp_path / "sub0.png"
    Image.new("RGB", (40, 30), "red").save(fallback_png)
    beat = _beat(id="b00", kind="paper_collage", data={"fallback_image": str(fallback_png)})
    asset = motion.render_fallback(beat, tmp_path, 320, 180)
    assert asset is not None
    assert asset.kind == "image" and asset.source == "motion_fallback" and asset.beat_id == "b00"
    assert asset.path == str(fallback_png)

    empty = _beat(id="b01", kind="paper_collage", data={})
    assert motion.render_fallback(empty, tmp_path, 320, 180) is None


def test_render_fallback_has_nothing_for_a_kind_it_does_not_know():
    beat = _beat(id="b00", kind="broll_image")
    assert motion.render_fallback(beat, "/tmp", 320, 180) is None
    assert motion.render_fallback(None, "/tmp", 320, 180) is None


# --- placement: a motion asset is a graphic, not a clip ----------------------

def _motion_asset(beat_id: str, duration_s: float) -> Asset:
    return Asset(beat_id=beat_id, kind="video", path="motion.mp4", width=1920, height=1080,
                duration_s=duration_s, source="motion")


def test_a_motion_window_longer_than_the_render_holds_the_last_frame_instead_of_being_cut():
    """A plain (non-motion) video asset shorter than its window is cut to its
    own length; a motion one keeps the full window -- the overlay compositing
    then holds its last frame for the remainder (render/compiler.py). The
    asset is kept at or above the floor (broll_seconds_min) so the floor-raise
    a couple of lines below this cap in _window_for does not also widen it,
    muddying which behaviour is actually being tested."""
    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()
    beat = _beat(id="b00", start_s=0.0, end_s=20.0, planned_duration_s=6.0)

    plain = Asset(beat_id="b00", kind="video", path="clip.mp4", width=1920, height=1080,
                 duration_s=4.0, source="generated")
    _, plain_duration, _ = _window_for(beat, plain, program, [], settings)
    assert plain_duration == pytest.approx(4.0)

    moving = _motion_asset("b00", 4.0)
    _, motion_duration, _ = _window_for(beat, moving, program, [], settings)
    assert motion_duration == pytest.approx(6.0)


def test_a_motion_window_is_never_rejected_for_needing_five_seconds():
    """A 6s-or-longer video asset placed into a slot under 5s is normally
    rejected outright (CLIP_NEEDS_5S_REASON); a motion asset is a graphic, not
    a clip, and takes whatever room the beat has instead."""
    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()
    beat = _beat(id="b00", start_s=0.0, end_s=3.0)

    plain = Asset(beat_id="b00", kind="video", path="clip.mp4", width=1920, height=1080,
                 duration_s=6.0, source="generated")
    _, _, reason = _window_for(beat, plain, program, [], settings)
    assert reason == CLIP_NEEDS_5S_REASON

    moving = _motion_asset("b00", 6.0)
    _, duration, reason = _window_for(beat, moving, program, [], settings)
    assert reason == "" and duration == pytest.approx(3.0)


# --- director wiring: motion runs before, and instead of, the PIL cards ------

def test_draw_local_assets_renders_kinetic_text_before_the_pil_renderers(tmp_path):
    from presentation.director import _draw_local_assets

    beat = _beat(id="b00", planned_duration_s=1.0)
    assets, failures = _draw_local_assets([beat], tmp_path, 320, 180, "horror",
                                          PresentationSettings())
    assert failures == []
    assert len(assets) == 1
    assert assets[0].beat_id == "b00" and assets[0].source == "motion" and assets[0].kind == "video"


def test_draw_local_assets_falls_back_to_a_static_card_when_motion_is_unavailable(
        tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "available", lambda: False)
    beat = _beat(id="b00")
    from presentation.director import _draw_local_assets
    assets, failures = _draw_local_assets([beat], tmp_path, 320, 180, "horror",
                                          PresentationSettings())
    assert len(assets) == 1
    assert assets[0].source == "motion_fallback" and assets[0].kind == "image"
    assert failures and "drew a static card instead" in failures[0]["reason"]


def test_a_timed_motion_hold_also_keeps_the_full_window():
    """The `| seconds=N` path (_held_window) gets the same treatment."""
    program = Program(duration_s=100.0, fps=30.0)
    beat = _beat(id="b00", start_s=0.0, end_s=20.0, planned_duration_s=5.0,
                data={"hold": True, "style": "slam"})

    moving = _motion_asset("b00", 2.0)
    _, duration, reason = _held_window(beat, moving, program, [], start_s=0.0)
    assert reason == "" and duration == pytest.approx(5.0)
