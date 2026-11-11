"""StickScene: era/outfit/bg additions (STICK_WORLD_PLAN.md SW2) -- outfit
normalisation against the shared vocabulary, the background prompt, and a
dry `build_stick_scene` job dict check. No Node/ComfyUI involved -- these
are the pure-Python pieces `presentation/director.py` wires together."""

from PIL import Image

from presentation import motion, motion_stick
from presentation.models import Beat, PresentationSettings


def _ctx(**kwargs) -> motion.MotionContext:
    base = dict(width=320, height=180, fps=30.0, genre="general",
               theme=motion.theme_for("general", "real"), settings=PresentationSettings())
    base.update(kwargs)
    return motion.MotionContext(**base)


def test_normalize_outfit_drops_unknown_tokens_and_a_second_piece_of_one_category():
    # "beard" and "moustache" are both "hair" -- the second is dropped.
    # "sandals" is not in the vocabulary at all, and is dropped too.
    assert motion_stick.normalize_outfit("chiton+laurel+beard+moustache+sandals") == \
        "chiton+laurel+beard"
    assert motion_stick.normalize_outfit("") == ""
    assert motion_stick.normalize_outfit(None) == ""


def test_compose_bg_prompt_adds_era_words_and_the_fixed_suffix_with_no_camera_words():
    prompt = motion_stick.compose_bg_prompt({"bg": "a library court", "era": "ancient_greece"})
    assert prompt.startswith("a library court, ancient Greece")
    assert "a wide view of the place with nobody in it" in prompt
    assert not any(word in prompt.lower() for word in ("camera", "lens", "shot on"))
    assert motion_stick.compose_bg_prompt({}) == ""


def test_build_stick_scene_dry_job_carries_era_people_background_and_luma(tmp_path):
    """A dry `build_stick_scene` job dict check (no Node render): `era` and
    `outfits` -> the `people` prop, each spec normalised; an already-
    generated `bg_image` -> the `background` data: URI and `bgLuma`."""
    bg_path = tmp_path / "bg.jpg"
    Image.new("RGB", (40, 40), (10, 20, 30)).save(bg_path, format="JPEG")

    beat = Beat(start_s=0.0, end_s=6.0, topic="a scene", kind="stick_scene", priority=1.0,
               data={"action": "point", "scene": "a Greek court", "figures": 2,
                     "era": "ancient_greece",
                     "outfits": ["chiton+laurel+scroll+beard", "toga+staff"],
                     "bg_image": str(bg_path), "bg_luma": 42.0})
    job = motion_stick.build_stick_scene(beat, _ctx())

    assert job["composition"] == "StickScene"
    assert job["props"]["era"] == "ancient_greece"
    assert job["props"]["people"] == [{"outfit": "chiton+laurel+scroll+beard"},
                                       {"outfit": "toga+staff"}]
    assert job["props"]["background"].startswith("data:image/jpeg;base64,")
    assert job["props"]["bgLuma"] == 42.0


def test_build_stick_scene_without_bg_image_falls_back_to_the_plain_ground():
    """No `bg_image` on the beat (STICK_WORLD_PLAN.md SW2's "no picture ->
    plain ground" fallback) -- `era` still fills `people` from the era's own
    defaults, but no `background`/`bgLuma` prop is sent at all."""
    beat = Beat(start_s=0.0, end_s=6.0, topic="a scene", kind="stick_scene", priority=1.0,
               data={"action": "idle", "scene": "a quiet office", "figures": 1, "era": "modern"})
    job = motion_stick.build_stick_scene(beat, _ctx())
    assert job["props"]["era"] == "modern"
    assert job["props"]["people"] == [{"outfit": "suit+phone"}]
    assert "background" not in job["props"]
    assert "bgLuma" not in job["props"]
