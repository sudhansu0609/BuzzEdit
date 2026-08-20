import pytest

from backend.render.effects import (
    build_canvas_transform,
    build_color_chain,
    build_overlay_transform,
    escape_filter_path,
)
from backend.timeline.schema import ColorGrade, Transform


def _joined(filters):
    return ",".join(filters)


# --- colour ---------------------------------------------------------------

def test_neutral_grade_compiles_to_nothing():
    assert build_color_chain(None) == []
    assert build_color_chain(ColorGrade()) == []
    assert build_color_chain(ColorGrade(preset="cinematic")) == []


def test_grade_emits_eq_hue_sharpen_and_vignette():
    chain = _joined(build_color_chain(
        ColorGrade(contrast=1.2, saturation=1.4, brightness=0.1, gamma=0.9,
                   hue=15, sharpen=0.5, vignette=0.5)))
    assert "eq=" in chain and "contrast=1.2" in chain and "saturation=1.4" in chain
    assert "hue=h=15" in chain
    assert "unsharp=5:5:0.5:5:5:0" in chain
    assert "vignette=angle=" in chain


def test_temperature_splits_into_red_and_blue_gamma():
    warm = _joined(build_color_chain(ColorGrade(temperature=1.0)))
    cool = _joined(build_color_chain(ColorGrade(temperature=-1.0)))
    assert "gamma_r=1.3" in warm and "gamma_b=0.7" in warm
    assert "gamma_r=0.7" in cool and "gamma_b=1.3" in cool


def test_fade_out_is_anchored_to_the_clip_end():
    chain = _joined(build_color_chain(ColorGrade(fade_in=0.5, fade_out=1.0), clip_duration_sec=10.0))
    assert "fade=t=in:st=0:d=0.5" in chain
    assert "fade=t=out:st=9:d=1" in chain


def test_fade_out_needs_a_duration_to_place_itself():
    assert "fade=t=out" not in _joined(build_color_chain(ColorGrade(fade_out=1.0)))


# --- static geometry ------------------------------------------------------

def test_identity_transform_still_lands_on_the_canvas():
    chain = _joined(build_canvas_transform(Transform(), 1920, 1080, 100, 30))
    assert "scale=1920:1080" in chain
    assert "crop=1920:1080:0:0" in chain
    assert "zoompan" not in chain


def test_static_zoom_scales_then_crops_the_canvas_out():
    chain = _joined(build_canvas_transform(Transform(scale=2.0), 1920, 1080, 100, 30))
    assert "scale=3840:2160:force_original_aspect_ratio=decrease" in chain
    assert "crop=1920:1080:960:540" in chain          # centred in the slack


def test_pan_offsets_the_crop_within_the_slack():
    left = _joined(build_canvas_transform(Transform(scale=2.0, pos_x=-1.0), 1920, 1080, 100, 30))
    right = _joined(build_canvas_transform(Transform(scale=2.0, pos_x=1.0), 1920, 1080, 100, 30))
    assert "crop=1920:1080:0:540" in left             # flush left
    assert "crop=1920:1080:1920:540" in right         # flush right


def test_zoom_out_pads_rather_than_cropping_into_nothing():
    chain = _joined(build_canvas_transform(Transform(scale=0.5), 1920, 1080, 100, 30))
    assert "scale=960:540" in chain
    assert "pad=1920:1080" in chain
    assert "crop=1920:1080:0:0" in chain


def test_crop_fractions_eat_the_right_edges():
    chain = _joined(build_canvas_transform(
        Transform(crop_left=0.1, crop_right=0.1, crop_top=0.2), 1920, 1080, 100, 30))
    assert "crop=w=iw*0.8:h=ih*0.8:x=iw*0.1:y=ih*0.2" in chain


def test_rotation_grows_the_frame_to_fit():
    chain = _joined(build_canvas_transform(Transform(rotation=90), 1920, 1080, 100, 30))
    assert "rotate=" in chain and "ow=rotw(" in chain and "oh=roth(" in chain


# --- animated geometry ----------------------------------------------------

def test_animated_zoom_uses_zoompan_over_the_clip_length():
    chain = _joined(build_canvas_transform(
        Transform(scale=1.0, scale_end=1.5), 1920, 1080, 61, 30))
    assert "zoompan=" in chain
    assert "on/60" in chain                            # duration_frames - 1
    assert ":d=1:s=1920x1080:fps=30" in chain


def test_zoom_out_animation_prescales_so_zoompan_never_goes_below_one():
    # zoompan clamps z at 1.0, so a 0.5 -> 1.0 move has to pre-shrink to 0.5 and
    # zoom 1x -> 2x from there.
    chain = _joined(build_canvas_transform(
        Transform(scale=0.5, scale_end=1.0), 1920, 1080, 31, 30))
    # The animated path supersamples 3x on a 1080p canvas before zoompan
    # (pre-scale = canvas * base * 3, pad = canvas * 3) so the crop rounds at a
    # third of the error — base here is 0.5, so pre = 1920*0.5*3, 1080*0.5*3.
    assert "scale=2880:1620:force_original_aspect_ratio=decrease:flags=lanczos" in chain
    assert "pad=5760:3240" in chain
    # Progress is smoothstep-eased: p*p*(3-2p) with p = on/30.
    assert "z='1+(1)*(on/30)*(on/30)*(3-2*(on/30))'" in chain


def test_zoom_supersample_is_capped_so_4k_does_not_explode():
    # A 3x intermediate on a 4K canvas would be 11520 wide — too heavy — so the
    # supersample is held to 2x there (7680) while 1080p still gets 3x.
    hd = _joined(build_canvas_transform(Transform(scale=1.0, scale_end=1.5), 1920, 1080, 31, 30))
    assert "pad=5760:3240" in hd                      # 1920*3
    uhd = _joined(build_canvas_transform(Transform(scale=1.0, scale_end=1.5), 3840, 2160, 31, 30))
    assert "pad=7680:4320" in uhd                      # 3840*2, not *3


def test_animated_expressions_are_quoted():
    # zoompan x/y contain no commas here, but the option values must still be
    # quoted so future expression changes cannot break the filtergraph parse.
    chain = _joined(build_canvas_transform(
        Transform(scale=1.2, scale_end=1.0, pos_x=-0.5, pos_x_end=0.5), 1920, 1080, 31, 30))
    assert "x='(iw-iw/zoom)/2*(1+(-0.5+(1)*(on/30)*(on/30)*(3-2*(on/30))))'" in chain


# --- overlays -------------------------------------------------------------

def test_overlay_is_fitted_and_centred_by_default():
    filters, x_expr, y_expr = build_overlay_transform(None, 1920, 1080, 60, 30, 1.0, 3.0)
    assert "scale=1920:1080:force_original_aspect_ratio=decrease" in _joined(filters)
    assert x_expr == "(W-w)/2+(0)*W/2"
    assert y_expr == "(H-h)/2+(0)*H/2"


def test_overlay_scale_shrinks_the_box_for_picture_in_picture():
    filters, x_expr, _ = build_overlay_transform(
        Transform(scale=0.25, pos_x=0.5), 1920, 1080, 60, 30, 1.0, 3.0)
    assert "scale=480:270" in _joined(filters)
    assert x_expr == "(W-w)/2+(0.5)*W/2"


def test_overlay_pan_animates_on_main_stream_time():
    _filters, x_expr, _ = build_overlay_transform(
        Transform(pos_x=-0.5, pos_x_end=0.5), 1920, 1080, 60, 30, 2.0, 4.0)
    assert "clip((t-2)/2,0,1)" in x_expr


def test_overlay_with_animated_zoom_becomes_a_full_frame():
    filters, x_expr, y_expr = build_overlay_transform(
        Transform(scale=1.0, scale_end=1.4), 1920, 1080, 60, 30, 0.0, 2.0)
    assert "zoompan=" in _joined(filters)
    assert (x_expr, y_expr) == ("0", "0")


def test_overlay_opacity_adds_an_alpha_channel():
    filters, _x, _y = build_overlay_transform(
        Transform(opacity=0.5), 1920, 1080, 60, 30, 0.0, 2.0)
    chain = _joined(filters)
    assert "format=rgba" in chain and "colorchannelmixer=aa=0.5" in chain


# --- paths ----------------------------------------------------------------

def test_windows_paths_are_escaped_for_the_filtergraph():
    assert escape_filter_path(r"C:\Windows\Fonts\arial.ttf") == "C\\:/Windows/Fonts/arial.ttf"


# --- flip -----------------------------------------------------------------

def test_flip_is_identity_until_set():
    assert Transform().is_identity()
    assert not Transform(flip_h=True).is_identity()
    assert not Transform(flip_v=True).is_identity()


def test_flip_emits_hflip_vflip_on_canvas_and_overlay():
    chain = _joined(build_canvas_transform(Transform(flip_h=True, flip_v=True), 1920, 1080, 100, 30))
    assert "hflip" in chain and "vflip" in chain

    over = _joined(build_overlay_transform(Transform(flip_h=True), 1920, 1080, 100, 30)[0])
    assert "hflip" in over
    assert "vflip" not in over


def test_flip_runs_after_the_crop():
    """Crop coordinates must still refer to the original frame, so the mirror
    comes second: cropping the left then flipping keeps what was the left."""
    chain = _joined(build_canvas_transform(Transform(crop_left=0.2, flip_h=True), 1920, 1080, 100, 30))
    assert chain.index("crop=") < chain.index("hflip")


# --- animation offset (adjustment layers) ---------------------------------

def test_animation_counts_from_the_clips_own_first_frame_by_default():
    chain = _joined(build_canvas_transform(
        Transform(scale=1.0, scale_end=1.5), 1920, 1080, 100, 30))
    assert "on/99" in chain
    assert "clip(" not in chain


def test_an_offset_animation_is_counted_from_the_offset_and_held_either_side():
    """An adjustment layer's chain runs over the whole programme, so its move has
    to start at the frame the clip does — and hold still before and after it."""
    chain = _joined(build_canvas_transform(
        Transform(scale=1.0, scale_end=1.5), 1920, 1080, 100, 30, frame_offset=300))
    assert "clip((on-300)/99,0,1)" in chain
