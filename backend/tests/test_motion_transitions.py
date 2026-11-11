"""M3 (MOTION_GRAPHICS_PLAN.md): the five Remotion transition stingers --
`[fx: shape_wipe|iris|ink|bars|paper_tear dur=0.8]` -> `build_transition_stinger`
-> `motion_transitions.place_stinger` -> a V10 overlay centred on the cut.

The one thing that actually matters is proven by a real render: the compiler
composites the ProRes 4444 overlay's alpha, so the footage underneath shows
through before and after the stinger's own midpoint, and the stinger's theme
colour fully covers the frame exactly at it. Skipped, never mocked, when the
Node engine is not installed (`npm install` in motion/), same as test_motion.py.
"""

import subprocess

import pytest

from presentation import motion, motion_transitions, script
from presentation.models import PresentationSettings, Program

pytestmark = pytest.mark.skipif(not motion.available(),
                                reason="motion engine not installed (npm install in motion/)")


def _ctx(**kwargs) -> motion.MotionContext:
    base = dict(width=640, height=360, fps=30.0, genre="general",
               theme=motion.theme_for("general", "real"), settings=PresentationSettings())
    base.update(kwargs)
    return motion.MotionContext(**base)


# --- directive grammar: `[fx: <name> dur=0.8]` -------------------------------

def test_the_five_stinger_names_default_to_point_eight_seconds():
    for name in motion_transitions.STINGER_NAMES:
        record = script._parse_fx("fx", name)
        assert record["kind"] == "fx"
        assert record["effect"] == name
        assert record["dur"] == pytest.approx(0.8)


def test_a_stinger_directive_can_still_override_its_duration():
    record = script._parse_fx("fx", "iris dur=1.5")
    assert record["effect"] == "iris" and record["dur"] == pytest.approx(1.5)


# --- the builder --------------------------------------------------------------

def test_builders_registry_has_transition_stinger():
    assert motion.BUILDERS["transition_stinger"] is motion_transitions.build_transition_stinger


def test_build_transition_stinger_rejects_an_unknown_name():
    beat = motion_transitions._StingerBeat(id="s1", data={"name": "not_a_stinger"})
    assert motion_transitions.build_transition_stinger(beat, _ctx()) is None


@pytest.mark.parametrize("name,composition", list(motion_transitions.STINGER_COMPOSITIONS.items()))
def test_build_transition_stinger_builds_a_transparent_job_at_its_own_duration(name, composition):
    beat = motion_transitions._StingerBeat(id="s1", data={"name": name}, planned_duration_s=0.8)
    job = motion_transitions.build_transition_stinger(beat, _ctx())
    assert job["composition"] == composition
    assert job["transparent"] is True
    assert job["props"]["theme"] == _ctx().theme
    # Never clamped into motion's 2.5-12s cutaway floor/ceiling -- a stinger
    # is a 0.8s overlay, not a cutaway.
    assert job["frames"] == round(0.8 * 30.0)


# --- the real render: placement -----------------------------------------------

def test_place_stinger_centres_the_overlay_on_the_cut(tmp_path):
    from timeline import clip_ops
    from timeline.schema import SourceFile, Timeline

    tl = Timeline(fps_num=30, fps_den=1, width=640, height=360)
    tl.sources["bg"] = SourceFile(id="bg", path="C:/bg.mp4", duration_seconds=4.0,
                                  width=640, height=360, kind="video", has_audio=False)
    clip_ops.add_media_item(tl, "bg", "V1", 0, 0, 120)
    program = Program(duration_s=4.0, fps=30.0)

    ok = motion_transitions.place_stinger(
        tl, "iris", at_s=2.0, dur_s=0.8, program=program, project_dir=tmp_path,
        width=640, height=360, genre="general", settings=PresentationSettings(),
        origin="test_stinger")
    assert ok is True

    placed = [i for i in tl.items if i.track == motion_transitions.STINGER_TRACK]
    assert len(placed) == 1
    item = placed[0]
    assert item.origin == "test_stinger"
    assert item.timeline_start_frame == round(1.6 * 30)
    assert item.timeline_end_frame == round(2.4 * 30)


def test_place_stinger_returns_false_for_an_unknown_name(tmp_path):
    from timeline.schema import Timeline
    tl = Timeline(fps_num=30, fps_den=1, width=640, height=360)
    program = Program(duration_s=4.0, fps=30.0)
    assert motion_transitions.place_stinger(
        tl, "not_a_stinger", 2.0, 0.8, program, tmp_path, 640, 360, "general",
        PresentationSettings(), origin="test_stinger") is False


# --- the real render: alpha compositing ---------------------------------------

def _frame_pixel(video_path: str, frame_index: int, x: int, y: int):
    """The exact RGB of one pixel of one exact output frame, selected by frame
    number (not a `-ss` time seek, which rounds) so the sample lands on the
    frame the test means."""
    from io import BytesIO

    from PIL import Image

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"select=eq(n\\,{frame_index})", "-vsync", "0", "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0 and proc.stdout, proc.stderr.decode(errors="replace")
    image = Image.open(BytesIO(proc.stdout)).convert("RGB")
    return image.getpixel((x, y))


def _close(color, target, tol=28):
    return all(abs(c - t) <= tol for c, t in zip(color, target))


def test_the_compiler_composites_the_stinger_alpha_over_the_footage(tmp_path):
    """Real render, real pixels: a solid-green V1 clip with an Iris stinger
    centred at 2.0s (0.8s window, 30fps -> frames 48..71, midpoint frame 60).
    A corner pixel -- never touched by the small early/late circle, only by
    the full-frame circle at coverage=1 -- is green well outside the stinger's
    own window, green again just inside each of its own edges, and the
    stinger's theme colour exactly at its own midpoint frame."""
    from timeline import clip_ops
    from timeline.schema import SourceFile, Timeline
    from render.runner import render_timeline_async
    import asyncio
    from config import FFMPEG_BIN

    width, height = 640, 360
    bg_path = tmp_path / "bg.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=0x00C000:s={width}x{height}:rate=30:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(bg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not bg_path.exists():
        pytest.skip("ffmpeg unavailable")

    tl = Timeline(fps_num=30, fps_den=1, width=width, height=height)
    tl.sources["bg"] = SourceFile(id="bg", path=str(bg_path), duration_seconds=4.0,
                                  width=width, height=height, kind="video", has_audio=False)
    clip_ops.add_media_item(tl, "bg", "V1", 0, 0, 120)
    program = Program(duration_s=4.0, fps=30.0)

    theme = motion.theme_for("general", "real")
    ok = motion_transitions.place_stinger(
        tl, "iris", at_s=2.0, dur_s=0.8, program=program, project_dir=tmp_path,
        width=width, height=height, genre="general", settings=PresentationSettings(),
        origin="test_stinger")
    assert ok is True

    out = str(tmp_path / "composited.mp4")
    asyncio.run(render_timeline_async(tl, out, prefer_nvenc=False))

    green = (0, 0xC0, 0)
    accent2 = tuple(int(theme["accent2"][i:i + 2], 16) for i in (1, 3, 5))
    corner = (round(width * 0.03), round(height * 0.03))

    # Well before/after the stinger's own 48..71 window: plain background.
    assert _close(_frame_pixel(out, 20, *corner), green), "no stinger yet: corner should be plain background"
    assert _close(_frame_pixel(out, 100, *corner), green), "stinger long gone: corner should be plain background"
    # Just inside its own start/end (its circle is still small): still shows
    # the footage through the overlay's alpha.
    assert _close(_frame_pixel(out, 49, *corner), green), "near the stinger's own start: corner still background"
    assert _close(_frame_pixel(out, 70, *corner), green), "near the stinger's own end: corner still background"
    # Exactly its own midpoint (frame 48+12=60): fully covered in its theme colour.
    assert _close(_frame_pixel(out, 60, *corner), accent2), "stinger's own midpoint: corner should be fully covered"
