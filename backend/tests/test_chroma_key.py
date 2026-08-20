"""Tests for the chroma keyer and the expanded colour grade.

A filter graph can compile to a plausible-looking string and still be wrong, so
these pin down the things that decide whether a key *looks* good: that the matte
operations are aimed at the alpha plane and nowhere else, that despill lands on
the keyed picture rather than before it, and that the clip leaves the chain in a
format that still has an alpha channel to composite with.
"""

import pytest

from render.compiler import FilterGraphCompiler
from render.effects import build_chroma_key_chain, build_color_chain
from timeline import clip_ops
from timeline.schema import (ChromaKey, ColorGrade, ColorWheel, SourceFile,
                             Timeline, TimelineItem)


def _chain(**kwargs) -> str:
    return ",".join(build_chroma_key_chain(ChromaKey(**kwargs)))


# --- the key itself --------------------------------------------------------

def test_a_disabled_key_compiles_to_nothing():
    assert build_chroma_key_chain(ChromaKey(enabled=False)) == []
    assert build_chroma_key_chain(None) == []


def test_a_green_screen_keys_in_yuv():
    """A lit green screen is never one RGB value — it falls off toward the edges
    of the frame — but its hue stays put, which is why the key is done on chroma."""
    chain = _chain(color="0x00FF00", similarity=0.2, blend=0.1)
    assert "format=yuva420p" in chain
    assert "chromakey=0x00FF00:0.2:0.1" in chain
    assert "colorkey" not in chain


def test_a_flat_graphic_colour_can_be_keyed_in_rgb_instead():
    chain = _chain(key_type="color", color="0xFF00FF")
    assert "colorkey=0xFF00FF:" in chain
    assert "chromakey" not in chain


def test_a_hash_colour_from_the_ui_is_normalised():
    """<input type="color"> hands back "#00ff00"; the key filters want 0x."""
    assert "chromakey=0x00ff00:" in _chain(color="#00ff00")


def test_a_bare_hex_colour_is_normalised_too():
    assert "chromakey=0x00FF00:" in _chain(color="00FF00")


# --- the edges, which are what decide whether a key looks real -------------

def test_the_choke_erodes_only_the_alpha_plane():
    """The whole chain has to stay linear — the compiler splices it into a
    comma-joined chain — which only works because erosion can be aimed at one
    plane. Eroding the colour planes as well would smear the picture."""
    chain = _chain(choke=0.5)
    assert "erosion=threshold0=0:threshold1=0:threshold2=0:threshold3=255" in chain


def test_a_stronger_choke_eats_more_pixels():
    """Erosion works a pixel at a time, so strength is a number of passes."""
    assert _chain(choke=0.34).count("erosion=") == 1
    assert _chain(choke=1.0).count("erosion=") == 3


def test_the_feather_blurs_only_the_matte():
    """planes=8 is the alpha plane of yuva420p. Without the restriction this
    would blur the picture itself, which is not what softening an edge means."""
    chain = _chain(feather=0.5)
    assert "gblur=sigma=" in chain and ":planes=8" in chain


def test_no_edge_work_is_done_when_the_sliders_are_at_zero():
    chain = _chain()
    assert "erosion" not in chain and "gblur" not in chain


# --- spill, which is not an edge problem -----------------------------------

def test_despill_runs_after_the_key_not_before_it():
    """Green reflected onto a shoulder is inside the subject. Removing it before
    the key would move the very colour the key is hunting for."""
    chain = _chain(spill=0.5)
    assert chain.index("chromakey") < chain.index("despill")


def test_a_green_screen_despills_green_and_a_blue_screen_despills_blue():
    assert "despill=type=green" in _chain(color="0x00FF00", spill=0.4)
    assert "despill=type=blue" in _chain(color="0x0000FF", spill=0.4)


def test_no_despill_when_the_slider_is_at_zero():
    assert "despill" not in _chain(spill=0.0)


# --- what the clip leaves the chain as -------------------------------------

def test_a_keyed_clip_ends_with_an_alpha_carrying_format():
    """The compositor overlays this clip. A format without alpha here silently
    throws away everything the key just did, and the green comes back."""
    assert _chain().endswith("format=rgba")
    assert _chain(spill=0.5, feather=0.3, choke=0.2).endswith("format=rgba")


def test_the_matte_preview_shows_the_matte_and_stops_there():
    """Judging a key against the composite hides every fault sitting on a dark
    area, so the matte is viewable on its own: white keeps, black drops."""
    chain = _chain(show_matte=True, spill=0.5)
    assert chain.endswith("alphaextract,format=yuv420p")
    assert "despill" not in chain, "there is no colour left to despill in a matte"


def test_the_matte_preview_still_shows_the_edge_work():
    """A matte that did not include the choke and feather would be a preview of
    a different key from the one that ships."""
    chain = _chain(show_matte=True, choke=0.4, feather=0.4)
    assert "erosion" in chain and "gblur" in chain


# --- the colour grade ------------------------------------------------------

def test_a_neutral_grade_compiles_to_nothing():
    assert build_color_chain(ColorGrade()) == []


def test_exposure_is_measured_in_stops():
    """The unit a camera works in, applied in linear light — so it behaves like
    the lens did rather than like a brightness slider bolted onto gamma."""
    assert "exposure=exposure=0.5" in ",".join(build_color_chain(ColorGrade(exposure=0.5)))


def test_tint_moves_green_against_magenta_where_temperature_cannot():
    chain = ",".join(build_color_chain(ColorGrade(tint=1.0)))
    assert "gamma_g=" in chain, "the green axis is the whole point of tint"


def test_temperature_still_works_on_its_own():
    chain = ",".join(build_color_chain(ColorGrade(temperature=1.0)))
    assert "gamma_r=1.3" in chain and "gamma_b=0.7" in chain


def test_the_colour_wheels_become_a_per_tonal_range_balance():
    """`eq` can only move the whole picture at once. Warming the highlights while
    cooling the shadows needs colorbalance, and that is what a grade is."""
    chain = ",".join(build_color_chain(ColorGrade(
        lift=ColorWheel(b=0.2), gain=ColorWheel(r=0.15))))
    assert "colorbalance=" in chain
    assert "bs=0.2" in chain and "rh=0.15" in chain


def test_an_untouched_wheel_contributes_nothing():
    chain = ",".join(build_color_chain(ColorGrade(saturation=1.2)))
    assert "colorbalance" not in chain


def test_shadow_and_highlight_recovery_use_a_curve():
    """Recovering a blown sky without flattening the face needs a curve that
    bends near one end, which brightness and gamma cannot do."""
    chain = ",".join(build_color_chain(ColorGrade(highlights=-0.5)))
    assert "curves=all='0/0 0.25/0.25 0.75/0.65 1/1'" in chain


def test_the_tone_curve_stays_monotonic_at_the_extremes():
    """A curve that doubles back posterises — a gradient turns into banded blocks."""
    chain = ",".join(build_color_chain(ColorGrade(shadows=1.0, highlights=-1.0)))
    points = chain.split("curves=all='")[1].split("'")[0].split()
    values = [float(p.split("/")[1]) for p in points]
    assert values == sorted(values)


def test_vibrance_is_separate_from_saturation():
    chain = ",".join(build_color_chain(ColorGrade(vibrance=0.6)))
    assert "vibrance=intensity=0.6" in chain
    assert "saturation" not in chain


def test_a_lut_is_applied_with_tetrahedral_interpolation():
    """The default trilinear interpolation visibly bands on smooth gradients."""
    chain = ",".join(build_color_chain(ColorGrade(lut_file="C:/luts/teal.cube")))
    assert "lut3d=file='C\\:/luts/teal.cube':interp=tetrahedral" in chain


def test_denoise_runs_before_anything_that_stretches_contrast():
    """Grading first amplifies the very grain the denoiser is there to remove."""
    chain = ",".join(build_color_chain(ColorGrade(denoise=0.5, contrast=1.6)))
    assert chain.index("hqdn3d") < chain.index("eq=")


def test_sharpening_runs_after_the_grade():
    chain = ",".join(build_color_chain(ColorGrade(sharpen=1.0, contrast=1.4)))
    assert chain.index("eq=") < chain.index("unsharp")


def test_the_old_grade_fields_still_compile_the_same_way():
    """Projects saved before the grade grew must keep rendering identically."""
    chain = build_color_chain(ColorGrade(brightness=0.1, contrast=1.2, saturation=1.3,
                                         gamma=1.1, hue=15, vignette=0.5))
    assert chain[0] == "eq=contrast=1.2:brightness=0.1:saturation=1.3:gamma=1.1"
    assert "hue=h=15" in chain
    assert any(f.startswith("vignette=angle=") for f in chain)


# --- where a key is allowed, and where it lands in the graph ---------------

def _timeline_with_overlay():
    src = SourceFile(id="src_main", path="C:/media/main.mp4", duration_seconds=20.0,
                     width=1920, height=1080, fps_num=30, fps_den=1, kind="video")
    v1 = TimelineItem(id="v1_0", track="V1", source_id="src_main",
                      source_start_frame=0, source_end_frame=150,
                      timeline_start_frame=0, timeline_end_frame=150, origin="auto")
    timeline = Timeline(fps_num=30, fps_den=1, sources={"src_main": src}, items=[v1])
    overlay = clip_ops.add_media_item(timeline, "src_main", "V2", 30, 0, 60)
    return timeline, overlay


def test_an_overlay_clip_can_be_keyed():
    timeline, overlay = _timeline_with_overlay()
    clip_ops.set_chroma(timeline, overlay.id, {"color": "0x00FF00", "similarity": 0.3})
    assert overlay.chroma is not None
    assert overlay.chroma.similarity == 0.3


def test_keying_the_programme_track_is_refused():
    """There is nothing behind V1 to show through the hole, and the transparent
    stream it produces cannot be concatenated with the opaque ones around it —
    the render would fail rather than merely look wrong."""
    timeline, _ = _timeline_with_overlay()
    with pytest.raises(clip_ops.ClipOpError) as err:
        clip_ops.set_chroma(timeline, "v1_0", {"color": "0x00FF00"})
    assert "V2" in str(err.value), "the error should say where to move the clip"


def test_disabling_the_key_clears_it_rather_than_leaving_a_dead_object():
    timeline, overlay = _timeline_with_overlay()
    clip_ops.set_chroma(timeline, overlay.id, {"color": "0x00FF00"})
    clip_ops.set_chroma(timeline, overlay.id, {"enabled": False})
    assert overlay.chroma is None


def test_the_key_runs_before_the_grade_in_the_compiled_graph():
    """Grading first moves the very colour the key is hunting for, so a warm
    grade over a green screen would leave green behind."""
    timeline, overlay = _timeline_with_overlay()
    clip_ops.set_chroma(timeline, overlay.id, {"color": "0x00FF00"})
    clip_ops.set_color(timeline, overlay.id, {"saturation": 1.4})

    _inputs, filter_complex, _v, _a = FilterGraphCompiler(timeline).compile()

    assert "chromakey=0x00FF00" in filter_complex
    assert filter_complex.index("chromakey") < filter_complex.index("saturation=1.4")
