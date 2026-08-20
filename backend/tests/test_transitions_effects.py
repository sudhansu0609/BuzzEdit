"""Transitions, atmosphere effects and cinematic letterboxing.

The two things most worth pinning here are timing and colour, because both failed
silently during development:

* transitions must not change the programme's length — a plain `xfade` consumes
  its overlap and shortens the picture while the audio concats at full length,
  drifting sound 0.12s per junction. The compiler pads each side with cloned
  frames so the xfade runs over the padding and both streams keep their length;
* `screen` is an RGB operation, and running it over YUV chroma planes turns the
  whole frame magenta.
"""

import subprocess

import pytest

from backend.render import transitions
from backend.render.atmosphere import build_aspect_bars, build_effect
from backend.render.compiler import FilterGraphCompiler
from backend.timeline import build_timeline_from_transcript, clip_ops
from backend.timeline.schema import AtmosphereEffect, Transition


def _program(segments: int = 4, clip: float = 0.9, gap: float = 0.6):
    words = [{"word": f"w{i}", "start": i * (clip + gap), "end": i * (clip + gap) + clip}
             for i in range(segments)]
    return build_timeline_from_transcript(
        "C:/video.mp4", segments * (clip + gap) + 1, words,
        fps_num=30, fps_den=1, pause_padding_seconds=0.0)


def _graph(timeline, tmp_path=None):
    return FilterGraphCompiler(timeline, assets_dir=tmp_path).compile()[1]


# --- transitions ----------------------------------------------------------

def test_no_transition_means_a_plain_concat():
    graph = _graph(_program())
    assert "xfade" not in graph
    assert "concat=n=4:v=1:a=0" in graph


def test_a_default_transition_applies_at_every_junction():
    timeline = _program()
    timeline.default_transition = Transition(type="fade", duration=0.4)
    graph = _graph(timeline)
    assert graph.count("xfade=") == 3          # 4 clips -> 3 junctions
    # Audio never crossfades: the picture keeps its full length (cloned-frame
    # padding under each xfade), so an acrossfade would shorten audio out of
    # step and smear speech over the join. Declick lives on the segments.
    assert "acrossfade" not in graph
    # Both sides of every junction are extended over cloned frames.
    assert graph.count("tpad=stop_mode=clone") == 3
    assert graph.count("tpad=start_mode=clone") == 3


def test_transition_offsets_keep_the_programme_full_length():
    """Transitions must not shorten the programme. Each junction's xfade runs
    over half-duration cloned extensions on both sides, so with 0.9s clips and
    0.4s fades the offsets sit at 0.9-0.2, 1.8-0.2, 2.7-0.2 — and the elapsed
    total stays 3.6s, exactly what the audio concat produces. The old
    shortening math drifted sound 0.4s per junction on this programme.
    """
    timeline = _program()
    timeline.default_transition = Transition(type="fade", duration=0.4)
    graph = _graph(timeline)
    assert "offset=0.700" in graph
    assert "offset=1.600" in graph
    assert "offset=2.500" in graph


def test_a_per_clip_transition_overrides_the_default():
    timeline = _program()
    timeline.default_transition = Transition(type="fade", duration=0.4)
    v1 = sorted([i for i in timeline.items if i.track == "V1"],
                key=lambda i: i.timeline_start_frame)
    clip_ops.set_transition(timeline, v1[2].id, updates={"type": "wipeleft", "duration": 0.3})
    graph = _graph(timeline)
    assert "transition=wipeleft" in graph
    assert graph.count("transition=fade") == 2


def test_a_transition_cannot_outlast_the_clips_it_joins():
    """xfade needs both sides to cover the overlap; asking for more breaks the graph."""
    timeline = _program(segments=2, clip=0.4, gap=0.6)
    timeline.default_transition = Transition(type="fade", duration=2.0)
    graph = _graph(timeline)
    assert "xfade" in graph
    duration = float(graph.split("xfade=transition=fade:duration=")[1].split(":")[0])
    assert duration <= 0.4


def test_a_zero_duration_transition_is_a_hard_cut():
    timeline = _program()
    clip_ops.set_transition(timeline, None, preset="cut")
    assert timeline.default_transition is None
    assert "xfade" not in _graph(timeline)


def test_only_real_xfade_names_are_accepted():
    assert transitions.is_valid("wipeleft")
    assert transitions.is_valid("circleopen")
    assert not transitions.is_valid("not_a_transition")


def test_an_unknown_transition_is_ignored_rather_than_breaking_the_render():
    timeline = _program()
    timeline.default_transition = Transition(type="teleport", duration=0.4)
    assert "xfade" not in _graph(timeline)


def test_clamp_duration_respects_both_clips():
    assert transitions.clamp_duration(1.0, 5.0, 0.5) < 0.5
    assert transitions.clamp_duration(0.4, 5.0, 5.0) == 0.4
    assert transitions.clamp_duration(99.0, 50.0, 50.0) == transitions.MAX_DURATION


# --- atmosphere -----------------------------------------------------------

def _effect_chain(kind: str, **kwargs) -> str:
    effect = AtmosphereEffect(type=kind, **kwargs)
    return ";".join(build_effect(effect, "[in]", "[out]", 1280, 720, 30.0, 5.0, 0))


def test_effects_composite_in_rgb_not_yuv():
    """screen over YUV chroma turns the picture magenta — every blend must be gbrp."""
    for kind in ("rain", "snow", "lightning", "fog", "sunlight", "light_leak", "wind"):
        chain = _effect_chain(kind)
        assert "format=gbrp" in chain, kind
        assert "blend=" in chain, kind


def test_blends_never_outlive_the_programme():
    """The generated layer is longer than the video; without shortest=1 blend pads
    the *programme* out to match and every transition's shortening is undone."""
    for kind in ("rain", "snow", "lightning", "fog", "sunlight", "light_leak", "wind"):
        assert "shortest=1" in _effect_chain(kind), kind


def test_coloured_effects_use_eight_digit_hex():
    """gradients reads 6-digit hex as alpha 0 and yields a transparent layer."""
    for kind in ("sunlight", "light_leak", "fog", "wind"):
        chain = _effect_chain(kind)
        for token in chain.split("c0=")[1:]:
            colour = token.split(":")[0].split(",")[0]
            assert len(colour) == 10 and colour.startswith("0x"), (kind, colour)


def test_rain_thresholds_above_a_grey_base():
    """noise on black tops out near 148, so a 230 threshold on black is empty."""
    chain = _effect_chain("rain")
    assert "color=gray" in chain
    threshold = int(chain.split("gt(val,")[1].split(")")[0])
    assert 200 < threshold < 250


def test_intensity_changes_the_result():
    light = _effect_chain("rain", intensity=0.1)
    heavy = _effect_chain("rain", intensity=1.0)
    assert light != heavy


def test_grain_needs_no_layer():
    chain = _effect_chain("grain")
    assert "noise=" in chain and "blend" not in chain


def test_an_unknown_effect_is_skipped():
    assert build_effect(AtmosphereEffect(type="unicorns"), "[in]", "[out]",
                        1280, 720, 30.0, 5.0, 0) == []


def test_effects_reach_the_compiled_graph():
    timeline = _program()
    clip_ops.add_effect(timeline, preset="heavy_rain")
    clip_ops.add_effect(timeline, preset="warm_leak")
    graph = _graph(timeline)
    assert graph.count("blend=all_mode=screen") >= 2


def test_a_disabled_effect_is_left_out():
    timeline = _program()
    clip_ops.add_effect(timeline, preset="heavy_rain")
    timeline.effects[0].enabled = False
    assert "blend=all_mode=screen" not in _graph(timeline)


# --- cinematic bars -------------------------------------------------------

def test_letterbox_keeps_the_output_resolution():
    bars = build_aspect_bars(2.39, 1920, 1080, "[in]", "[out]")
    assert "crop=1920:802" in bars          # 1920/2.39, rounded to even
    assert "pad=1920:1080" in bars


def test_a_taller_target_puts_bars_at_the_sides():
    bars = build_aspect_bars(1.0, 1920, 1080, "[in]", "[out]")
    assert "crop=1080:1080" in bars
    assert "pad=1920:1080" in bars


def test_matching_the_current_ratio_does_nothing():
    assert build_aspect_bars(16 / 9, 1920, 1080, "[in]", "[out]") is None
    assert build_aspect_bars(0, 1920, 1080, "[in]", "[out]") is None


def test_aspect_bars_reach_the_compiled_graph():
    timeline = _program()
    clip_ops.set_aspect_bars(timeline, 2.39)
    assert "pad=" in _graph(timeline)


def test_a_silly_ratio_is_refused():
    timeline = _program()
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.set_aspect_bars(timeline, 12.0)


# --- end to end -----------------------------------------------------------

def test_a_transitioned_effected_programme_renders_in_sync(tmp_path):
    """The whole point: picture and sound must come out the same length."""
    from backend.config import FFMPEG_BIN
    import asyncio
    from backend.render.runner import render_timeline_async

    source = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not source.exists():
        pytest.skip("ffmpeg unavailable")

    words = [{"word": f"w{i}", "start": i * 1.5, "end": i * 1.5 + 0.9} for i in range(4)]
    timeline = build_timeline_from_transcript(str(source), 8.0, words, fps_num=30, fps_den=1,
                                              width=320, height=180, pause_padding_seconds=0.0)
    timeline.default_transition = Transition(type="fade", duration=0.4)
    clip_ops.add_effect(timeline, preset="light_rain")
    clip_ops.set_aspect_bars(timeline, 2.39)

    out = str(tmp_path / "out.mp4")
    asyncio.run(render_timeline_async(timeline, out, prefer_nvenc=False))

    def duration(stream: str) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream,
             "-show_entries", "stream=duration", "-of", "csv=p=0", out],
            stdout=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    video, audio = duration("v:0"), duration("a:0")
    # 4 clips of 0.9s at full length — transitions ride on cloned padding and
    # must not shorten the programme.
    assert video == pytest.approx(3.6, abs=0.15)
    assert abs(video - audio) < 0.1, f"picture {video}s drifted from sound {audio}s"


def test_an_adjustment_layer_changes_only_its_own_stretch_of_the_programme(tmp_path):
    """Rendered for real and measured: the picture inside the layer's window is
    drained of colour, and the picture either side of it is untouched."""
    from backend.config import FFMPEG_BIN
    import asyncio
    from backend.render.runner import render_timeline_async
    from backend.timeline.schema import SourceFile, Timeline, TimelineItem

    source = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not source.exists():
        pytest.skip("ffmpeg unavailable")

    src = SourceFile(id="s", path=str(source), duration_seconds=8.0, width=320, height=180,
                     kind="video", has_audio=True)
    timeline = Timeline(fps_num=30, fps_den=1, sources={"s": src}, items=[
        TimelineItem(id="v1_0", track="V1", source_id="s", source_start_frame=0,
                     source_end_frame=180, timeline_start_frame=0, timeline_end_frame=180),
        TimelineItem(id="a1_0", track="A1", source_id="s", source_start_frame=0,
                     source_end_frame=180, timeline_start_frame=0, timeline_end_frame=180),
    ])
    timeline.recalculate_duration()
    adjustment = clip_ops.add_adjustment_item(timeline, 60, 60)      # 2s .. 4s
    clip_ops.set_color(timeline, adjustment.id, {"saturation": 0.0})

    out = str(tmp_path / "adjusted.mp4")
    asyncio.run(render_timeline_async(timeline, out, prefer_nvenc=False))

    def saturation_at(when: float) -> float:
        # -v info: the metadata filter prints at info level, and -v error hides it.
        result = subprocess.run(
            [FFMPEG_BIN, "-v", "info", "-ss", str(when), "-i", out, "-frames:v", "1",
             "-vf", "signalstats,metadata=print:key=lavfi.signalstats.SATAVG",
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in result.stdout.splitlines():
            if "SATAVG" in line:
                return float(line.rsplit("=", 1)[1])
        raise AssertionError("ffmpeg reported no saturation reading")

    assert saturation_at(3.0) < 5, "the adjustment layer did not reach the picture"
    assert saturation_at(1.0) > 40, "colour was drained before the layer starts"
    assert saturation_at(5.0) > 40, "colour was drained after the layer ends"
