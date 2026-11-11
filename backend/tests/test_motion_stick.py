"""StickScene: the Python side of the stick-figure motion template
(MOTION_GRAPHICS_PLAN.md M7) -- action/prop/ground normalisation, the
StickScene job builder, the real render, and the PIL fallback still.

The real-render cases are skipped automatically when the Node engine is not
installed (`npm install` in motion/), never mocked -- the point is to prove a
real render produces a real file and a real failure falls back cleanly."""

from pathlib import Path

import pytest

from presentation import motion, motion_stick
from presentation.models import Beat, PresentationSettings

pytestmark = pytest.mark.skipif(not motion.available(),
                                reason="motion engine not installed (npm install in motion/)")


def _beat(**kwargs) -> Beat:
    base = dict(start_s=0.0, end_s=6.0, topic="a scene", kind="stick_scene", priority=1.0,
               data={"action": "walk", "scene": "walking to the market", "prop": "house", "figures": 1})
    base.update(kwargs)
    return Beat(**base)


def _ctx(**kwargs) -> motion.MotionContext:
    base = dict(width=320, height=180, fps=30.0, genre="education",
               theme=motion.theme_for("education", "stick_figure"), settings=PresentationSettings())
    base.update(kwargs)
    return motion.MotionContext(**base)


# --- normalisation -----------------------------------------------------------

def test_normalize_action_falls_back_to_idle_for_unknown():
    assert motion_stick.normalize_action("swim") == "swim"
    assert motion_stick.normalize_action("SWIM") == "swim"
    assert motion_stick.normalize_action("flying") == "idle"
    assert motion_stick.normalize_action(None) == "idle"
    assert motion_stick.normalize_action("") == "idle"


def test_normalize_prop_falls_back_to_none_for_unknown():
    assert motion_stick.normalize_prop("Ship") == "ship"
    assert motion_stick.normalize_prop("spaceship") == "none"
    assert motion_stick.normalize_prop(None) == "none"


def test_ground_for_scene_picks_underwater_night_or_whiteboard():
    assert motion_stick.ground_for_scene("escaping a flooded house underwater") == "underwater"
    assert motion_stick.ground_for_scene("Lost at Sea") == "underwater"
    assert motion_stick.ground_for_scene("walking home at night") == "night"
    assert motion_stick.ground_for_scene("a sunny picnic") == "whiteboard"
    assert motion_stick.ground_for_scene(None) == "whiteboard"
    assert motion_stick.ground_for_scene("") == "whiteboard"


# --- the StickScene builder --------------------------------------------------

def test_builders_registry_has_stick_scene():
    assert motion.BUILDERS["stick_scene"] is motion_stick.build_stick_scene


def test_build_stick_scene_normalises_and_clamps_fields():
    ctx = _ctx()
    beat = _beat(data={"action": "SWIM", "scene": "escaping underwater", "prop": "boat", "figures": 7,
                       "caption": "x" * 100})
    job = motion_stick.build_stick_scene(beat, ctx)
    assert job["composition"] == "StickScene"
    assert job["props"]["action"] == "swim"
    assert job["props"]["ground"] == "underwater"
    assert job["props"]["prop"] == "none"  # "boat" is not in PROPS
    assert job["props"]["figures"] == 3  # clamped to 1-3
    assert len(job["props"]["caption"]) == 60
    assert job["props"]["theme"] == ctx.theme
    assert job["transparent"] is False


def test_build_stick_scene_clamps_duration():
    ctx = _ctx()
    job = motion_stick.build_stick_scene(_beat(planned_duration_s=30.0), ctx)  # over the max
    assert job["frames"] == round(motion.MOTION_BEAT_SECONDS_MAX * ctx.fps)

    too_short = motion_stick.build_stick_scene(_beat(planned_duration_s=0.4), ctx)
    assert too_short["frames"] == round(motion.MOTION_BEAT_SECONDS_MIN * ctx.fps)


def test_build_stick_scene_defaults_figures_to_one_and_caption_to_none():
    job = motion_stick.build_stick_scene(_beat(data={"action": "idle"}), _ctx())
    assert job["props"]["figures"] == 1
    assert job["props"]["caption"] is None


# --- the real render -----------------------------------------------------------

def test_render_motion_assets_produces_a_real_stick_scene_video(tmp_path):
    beat = _beat(id="b00", planned_duration_s=1.0)
    assets, failures, handled = motion.render_motion_assets(
        [beat], tmp_path, 320, 180, "education", PresentationSettings())
    assert failures == []
    assert handled == {"b00"}
    assert len(assets) == 1
    asset = assets[0]
    assert asset.beat_id == "b00" and asset.kind == "video" and asset.source == "motion"
    assert Path(asset.path).exists() and Path(asset.path).stat().st_size > 0


# --- the PIL fallback ---------------------------------------------------------

def test_render_stick_still_draws_a_connected_figure(tmp_path):
    beat = _beat(id="b00", data={"action": "idle", "scene": "a quiet night", "caption": "Help!"})
    image = motion_stick.render_stick_still(beat, 320, 180)
    assert image.size == (320, 180)


def test_render_fallback_draws_a_stick_still_for_stick_scene(tmp_path):
    beat = _beat(id="b00")
    asset = motion.render_fallback(beat, tmp_path, 320, 180)
    assert asset is not None
    assert asset.kind == "image" and asset.source == "motion_fallback" and asset.beat_id == "b00"
    assert Path(asset.path).exists()
