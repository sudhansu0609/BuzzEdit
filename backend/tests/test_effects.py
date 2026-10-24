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
    assert "pad=1920:1080:480:270" in chain           # centred in the spare room
    assert "crop=1920:1080:0:0" in chain


def test_a_shrunk_clip_can_be_moved_off_centre():
    """Scaled under 1.0 the picture is smaller than the canvas, and every bit of
    room to move it is in the pad — the crop has no slack left to work with. Pan
    was ignored there, so two clips could be shrunk but never placed side by
    side."""
    left = _joined(build_canvas_transform(
        Transform(scale=0.5, pos_x=-1.0), 1920, 1080, 100, 30))
    right = _joined(build_canvas_transform(
        Transform(scale=0.5, pos_x=1.0), 1920, 1080, 100, 30))
    bottom = _joined(build_canvas_transform(
        Transform(scale=0.5, pos_y=1.0), 1920, 1080, 100, 30))
    assert "pad=1920:1080:0:270" in left              # flush left, still centred vertically
    assert "pad=1920:1080:960:270" in right           # flush right
    assert "pad=1920:1080:480:540" in bottom          # flush bottom


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
    # Even a fast move takes the 2x floor (at 1x the integer crop shimmers):
    # pre-scale = canvas * base (0.5) * 2, pad = canvas * 2.
    assert "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos" in chain
    assert "pad=3840:2160" in chain
    # Progress is smoothstep-eased: p*p*(3-2p) with p = on/30.
    assert "z='1+(1)*(on/30)*(on/30)*(3-2*(on/30))'" in chain


def test_zoom_supersample_is_capped_so_4k_does_not_explode():
    # A slow push (5% over 10s) judders at 1x, so it gets the full 3x on 1080p;
    # a 3x intermediate on a 4K canvas would be 11520 wide — too heavy — so the
    # supersample is held to 2x there (7680).
    slow = Transform(scale=1.0, scale_end=1.05)
    hd = _joined(build_canvas_transform(slow, 1920, 1080, 300, 30))
    assert "pad=5760:3240" in hd                      # 1920*3
    uhd = _joined(build_canvas_transform(slow, 3840, 2160, 300, 30))
    assert "pad=7680:4320" in uhd                      # 3840*2, not *3


def test_fast_zooms_take_the_2x_floor_and_slow_ones_the_full_3x():
    # Supersampling costs its square in pixels, so fast moves stop at the 2x
    # floor that hides the crop-size shimmer; slow moves get the full 3x.
    fast = _joined(build_canvas_transform(Transform(scale=1.0, scale_end=1.3), 1920, 1080, 25, 25))
    assert "pad=3840:2160" in fast
    slow = _joined(build_canvas_transform(Transform(scale=1.0, scale_end=1.05), 1920, 1080, 250, 25))
    assert "pad=5760:3240" in slow


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


# --- anim_total_frames / anim_offset_frames (chunked rendering) -----------
#
# These two Transform fields are only ever set by render/chunked.py's
# sub_timeline() when a chunk boundary cuts through this item's own
# animation. Left unset (every chain built before they existed, and every one
# built today for an item a chunk boundary never touches) they must have no
# effect at all -- the chains below are the exact strings build_canvas_transform
# and build_overlay_transform produced before anim_total_frames/anim_offset_frames
# were added, byte for byte.

def test_unset_anim_fields_reproduce_the_pre_chunking_canvas_filter():
    chain = _joined(build_canvas_transform(
        Transform(scale=1.0, scale_end=1.5), 1920, 1080, 100, 30))
    assert chain == (
        "scale=5760:3240:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=5760:3240:(ow-iw)/2:(oh-ih)/2:color=black,"
        "zoompan=z='1+(0.5)*(on/99)*(on/99)*(3-2*(on/99))':"
        "x='(iw-iw/zoom)/2*(1+(0+(0)*(on/99)*(on/99)*(3-2*(on/99))))':"
        "y='(ih-ih/zoom)/2*(1+(0+(0)*(on/99)*(on/99)*(3-2*(on/99))))':"
        "d=1:s=1920x1080:fps=30,setsar=1"
    )


def test_unset_anim_fields_reproduce_the_pre_chunking_offset_canvas_filter():
    chain = _joined(build_canvas_transform(
        Transform(scale=1.0, scale_end=1.5), 1920, 1080, 100, 30, frame_offset=300))
    assert chain == (
        "scale=5760:3240:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=5760:3240:(ow-iw)/2:(oh-ih)/2:color=black,"
        "zoompan=z='1+(0.5)*(clip((on-300)/99,0,1))*(clip((on-300)/99,0,1))*(3-2*(clip((on-300)/99,0,1)))':"
        "x='(iw-iw/zoom)/2*(1+(0+(0)*(clip((on-300)/99,0,1))*(clip((on-300)/99,0,1))*(3-2*(clip((on-300)/99,0,1)))))':"
        "y='(ih-ih/zoom)/2*(1+(0+(0)*(clip((on-300)/99,0,1))*(clip((on-300)/99,0,1))*(3-2*(clip((on-300)/99,0,1)))))':"
        "d=1:s=1920x1080:fps=30,setsar=1"
    )


def test_unset_anim_fields_reproduce_the_pre_chunking_overlay_pan():
    filters, x_expr, y_expr = build_overlay_transform(
        Transform(pos_x=-0.5, pos_x_end=0.5), 1920, 1080, 60, 30, 2.0, 4.0)
    assert filters == ["scale=1920:1080:force_original_aspect_ratio=decrease"]
    assert x_expr == "(W-w)/2+(-0.5+(1)*clip((t-2)/2,0,1))*W/2"
    assert y_expr == "(H-h)/2+(0+(0)*clip((t-2)/2,0,1))*H/2"


def test_split_canvas_zoom_keeps_the_original_moves_speed_and_phase():
    """A chunk boundary cuts this item's own stream in two: anim_total_frames
    is the move's original length and anim_offset_frames is how many of its
    frames already played, so the piece on either side of the cut picks up
    exactly where the other left off instead of restarting its own 0..1."""
    import re

    def _raw_progress_expr(z_expr: str) -> str:
        """The un-eased `on/span` or `clip(...)` progress term, found by
        matching parens rather than assuming no `clip(...)` call nests one --
        the smoothstep wrapper around it is not what needs to be linear."""
        idx = z_expr.find("clip(")
        if idx == -1:
            return re.search(r"on/\d+", z_expr).group(0)
        depth, start = 0, idx + len("clip")
        for j in range(start, len(z_expr)):
            if z_expr[j] == "(":
                depth += 1
            elif z_expr[j] == ")":
                depth -= 1
                if depth == 0:
                    return z_expr[idx:j + 1]
        raise ValueError(f"unbalanced clip() in {z_expr!r}")

    def _progress_at(chain, on_value):
        z_expr = re.search(r"z='([^']*)'", next(f for f in chain if f.startswith("zoompan="))).group(1)
        raw = _raw_progress_expr(z_expr)
        return eval(raw, {"clip": lambda a, b, c: max(b, min(c, a)), "on": on_value})

    whole = Transform(scale=1.0, scale_end=1.5)
    first_piece = Transform(scale=1.0, scale_end=1.5, anim_total_frames=100, anim_offset_frames=0)
    second_piece = Transform(scale=1.0, scale_end=1.5, anim_total_frames=100, anim_offset_frames=40)

    unsplit_chain = build_canvas_transform(whole, 1920, 1080, 100, 30)
    first_chain = build_canvas_transform(first_piece, 1920, 1080, 40, 30)
    second_chain = build_canvas_transform(second_piece, 1920, 1080, 60, 30)

    # The split piece that owns the move's first frames (anim_offset_frames=0)
    # tracks the unsplit chain exactly, frame for frame.
    for on in (0, 20, 39):
        assert _progress_at(first_chain, on) == pytest.approx(_progress_at(unsplit_chain, on))
    # The next piece's own frame 0 is the original move's frame 40 -- one
    # frame further than the first piece's last frame, not a restart at 0.
    last_of_first = _progress_at(first_chain, 39)
    first_of_second = _progress_at(second_chain, 0)
    assert first_of_second - last_of_first == pytest.approx(1 / 99)
    assert first_of_second == pytest.approx(_progress_at(unsplit_chain, 40))


def test_split_overlay_pan_keeps_the_original_moves_speed_and_phase():
    """The position-only overlay path works in seconds (`t`), not frames, but
    the same anim_total_frames/anim_offset_frames continuity has to hold once
    converted: one piece's last moment and the next piece's first are exactly
    one frame apart in the original move, never a restart."""
    fps = 30.0
    first_piece = Transform(pos_x=-0.5, pos_x_end=0.5,
                            anim_total_frames=100, anim_offset_frames=0)
    second_piece = Transform(pos_x=-0.5, pos_x_end=0.5,
                             anim_total_frames=100, anim_offset_frames=40)

    # Piece 1 is programme seconds [0, 40/30); piece 2 is [0, 60/30) on its own
    # clock, both passed their own clip_start/clip_end (as the compiler does).
    _f1, x1, _y1 = build_overlay_transform(first_piece, 1920, 1080, 40, fps, 0.0, 39 / fps)
    _f2, x2, _y2 = build_overlay_transform(second_piece, 1920, 1080, 60, fps, 0.0, 60 / fps)

    def _frac_at(x_expr: str, t_value: float) -> float:
        expr = x_expr.split("+", 1)[1].rsplit(")*W/2", 1)[0] + ")"
        return eval(expr, {"clip": lambda a, b, c: max(b, min(c, a)), "t": t_value})

    last_of_first = _frac_at(x1, 39 / fps)
    first_of_second = _frac_at(x2, 0.0)
    # One frame of progress at this move's own total span (99 frames). The
    # seconds-based path's expressions round to 4 decimal places (_f()), so
    # the match is close rather than exact.
    step = (0.5 - -0.5) / 99
    assert first_of_second - last_of_first == pytest.approx(step, abs=1e-3)
    assert first_of_second > last_of_first  # never a restart back toward 0
