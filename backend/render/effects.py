"""FFmpeg filter chains for clip geometry (crop / pan / zoom) and colour grading.

Everything here returns a list of filter strings meant to be joined with "," and
spliced into the single-pass filter_complex the compiler builds. Nothing shells
out, so the whole module is directly unit-testable against expected filter text.

Two geometry paths exist, and the split is deliberate:

*Static* transforms scale the original source once and crop the canvas out of it.
Scaling straight from the source keeps a 4K clip sharp when zoomed into a 1080p
canvas.

*Animated* transforms (Ken Burns) must go through `zoompan`, which only zooms
*in* — its `z` is clamped at 1.0. So the chain first pre-scales to the smaller of
the two scales, pads that to the canvas, and then zooms up from there. A
0.6 -> 0.9 move becomes "shrink to 0.6, then zoom 1.0x -> 1.5x", which lands on
the same picture while staying inside zoompan's legal range.
"""

import math
from typing import List, Optional, Tuple

from timeline.schema import ChromaKey, ColorGrade, Transform

# Supersampling factor for animated zooms. zoompan rounds its crop origin AND its
# crop *size* to whole input pixels every output frame, so a slow move steps the
# window down one pixel every few frames — a periodic ~25% motion spike that reads
# as judder (measured on lossless frames; it is not codec noise). Feeding zoompan
# a frame upscaled by this factor shrinks each step by the same factor, which is
# what turns the move into a glide.
#
# The lever has diminishing returns and a real cost — pixel throughput grows with
# its square — so it is chosen per canvas rather than fixed. 3x visibly steadies a
# 1080p/1440p zoom (the common case); 4K cannot afford 3x (an 11520-wide
# intermediate), so it is held to 2x. `_zoom_ss` caps the supersampled long edge
# to keep the intermediate sane on any canvas.
_ZOOM_SS_BASE = 3
_ZOOM_SS_MAX_EDGE = 7680


# The rounding step is one input pixel; the judder it causes is that step
# relative to how far the window moves per frame. A move of >= this many output
# pixels per frame at the chosen factor keeps a 1px step to <=20% speed ripple;
# the measured judder was a ~25% spike on SLOW moves. Fast moves -- a typical
# 10-20% punch over 1-2s travels 5-10px/frame -- need little or no
# supersampling, and 3x on every one was ~9x the pixel work for nothing.
_ZOOM_SS_MIN_TRAVEL_PX = 15.0
# Never below this for an animated zoom: at 1x the window's integer crop size
# changes aspect by a pixel every few frames, which on a face reads as a
# shimmer however fast the move. 2x keeps that under half a pixel.
_ZOOM_SS_FLOOR = 2


def _zoom_ss(canvas_w: int, canvas_h: int, base_scale: float,
             travel_px_per_frame: Optional[float] = None) -> int:
    """Supersampling factor for a zoom on this canvas, capped by intermediate size.

    `base_scale` is the pre-scale the move starts from (<=1 for a push-in), so the
    cap is measured against the frame zoompan actually receives, not the canvas.
    `travel_px_per_frame` (average window-edge travel in output pixels) picks
    the smallest factor that still hides the rounding; None keeps the base.
    """
    long_edge = max(canvas_w, canvas_h) * max(0.05, base_scale)
    ss = _ZOOM_SS_BASE
    if travel_px_per_frame is not None and travel_px_per_frame > 0:
        needed = int(math.ceil(_ZOOM_SS_MIN_TRAVEL_PX / travel_px_per_frame))
        ss = max(_ZOOM_SS_FLOOR, min(_ZOOM_SS_BASE, needed))
    while ss > 1 and long_edge * ss > _ZOOM_SS_MAX_EDGE:
        ss -= 1
    return ss


# Chroma-subsampled output needs even dimensions; odd intermediates make
# scale/overlay emit warnings or fail outright on some builds.
def _even(value: float, minimum: int = 2) -> int:
    return max(minimum, int(round(value / 2.0)) * 2)


def _f(value: float) -> str:
    """Compact fixed-point so filter strings stay readable and diffable."""
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def escape_filter_path(path: str) -> str:
    """Escape a filesystem path for use inside a quoted filter option.

    Windows paths carry both backslashes (filter escape characters) and a drive
    colon (the filter option separator), so both have to go.
    """
    return str(path).replace("\\", "/").replace(":", "\\:")


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def build_color_chain(color: Optional[ColorGrade], clip_duration_sec: float = 0.0) -> List[str]:
    """Filters implementing a ColorGrade. Empty list when the grade is neutral.

    `clip_duration_sec` is only needed to place a fade-out; pass 0 to skip it.

    The order below is the order a colourist works in, and it is not arbitrary:
    exposure and white balance first (fix the picture), then shadow/highlight
    recovery, then the colour wheels and saturation (grade it), then the look —
    LUT, sharpening, vignette, fades. Grading before correcting means every
    creative decision is made against a picture that is still wrong; sharpening
    before grading amplifies noise the grade would have buried.
    """
    if color is None or color.is_identity():
        return []

    filters: List[str] = []

    # --- correction ---------------------------------------------------------

    if color.denoise > 0.0:
        # Before anything that stretches contrast, or the grade amplifies the
        # very grain this is meant to remove.
        amount = _clamp(color.denoise, 0.0, 1.0)
        filters.append(f"hqdn3d={_f(4.0 * amount)}:{_f(3.0 * amount)}:"
                       f"{_f(6.0 * amount)}:{_f(4.5 * amount)}")

    if color.exposure != 0.0:
        # Stops, the unit a camera works in: +1 is twice the light. `exposure`
        # operates in linear light, so it behaves like the lens did rather than
        # like a brightness slider bolted onto gamma-encoded values.
        filters.append(f"exposure=exposure={_f(_clamp(color.exposure, -3.0, 3.0))}")

    eq_parts: List[str] = []
    if color.contrast != 1.0:
        eq_parts.append(f"contrast={_f(_clamp(color.contrast, 0.0, 3.0))}")
    if color.brightness != 0.0:
        eq_parts.append(f"brightness={_f(_clamp(color.brightness, -1.0, 1.0))}")
    if color.saturation != 1.0:
        eq_parts.append(f"saturation={_f(_clamp(color.saturation, 0.0, 3.0))}")
    if color.gamma != 1.0:
        eq_parts.append(f"gamma={_f(_clamp(color.gamma, 0.1, 10.0))}")
    if color.temperature != 0.0 or color.tint != 0.0:
        # Warm (+) lifts red and drops blue; cool (-) does the reverse. Tint is
        # the perpendicular axis — green against magenta — which is what actually
        # fixes fluorescent lighting, and no amount of temperature can do it.
        # Both work through eq's per-channel gamma rather than the
        # `colortemperature` filter, which is missing from older FFmpeg builds.
        temp = _clamp(color.temperature, -1.0, 1.0)
        tint = _clamp(color.tint, -1.0, 1.0)
        gamma_r = 1.0 + 0.30 * temp + 0.10 * tint
        gamma_g = 1.0 - 0.20 * tint
        gamma_b = 1.0 - 0.30 * temp + 0.10 * tint
        eq_parts.append(f"gamma_r={_f(_clamp(gamma_r, 0.1, 10.0))}")
        if gamma_g != 1.0:
            eq_parts.append(f"gamma_g={_f(_clamp(gamma_g, 0.1, 10.0))}")
        eq_parts.append(f"gamma_b={_f(_clamp(gamma_b, 0.1, 10.0))}")
    if eq_parts:
        filters.append("eq=" + ":".join(eq_parts))

    curve = _tone_curve(color.shadows, color.highlights)
    if curve:
        filters.append(curve)

    # --- grade --------------------------------------------------------------

    balance = _color_balance(color)
    if balance:
        filters.append(balance)

    if color.vibrance != 0.0:
        # Saturation pushed everything equally and blew out skin tones long
        # before the background got interesting. Vibrance leaves already-saturated
        # colours alone, which is why it is the one to reach for on faces.
        filters.append(f"vibrance=intensity={_f(_clamp(color.vibrance, -2.0, 2.0))}")

    if color.hue != 0.0:
        filters.append(f"hue=h={_f(color.hue)}")

    # --- look ---------------------------------------------------------------

    if color.lut_file:
        strength = _clamp(color.lut_strength, 0.0, 1.0)
        if strength > 0.0:
            # `interp=tetrahedral` is the accurate interpolation; the default
            # trilinear visibly banks on smooth gradients like a sky.
            filters.append(
                f"lut3d=file='{escape_filter_path(color.lut_file)}':interp=tetrahedral")
            if strength < 1.0:
                # A LUT has no strength control of its own. Anything short of a
                # split-and-blend graph is a lie, so a partial LUT is applied as a
                # saturation/contrast pull-back towards the original instead of
                # pretending to be a true mix. Full strength is the honest path.
                filters.append(f"eq=saturation={_f(0.6 + 0.4 * strength)}")

    if color.sharpen > 0.0:
        amount = _clamp(color.sharpen, 0.0, 2.0)
        filters.append(f"unsharp=5:5:{_f(amount)}:5:5:0")

    if color.vignette > 0.0:
        # vignette's angle runs 0..PI/2; anything past ~1.2rad is a black frame.
        angle = _clamp(color.vignette, 0.0, 1.0) * 1.15
        filters.append(f"vignette=angle={_f(angle)}")

    if color.fade_in > 0.0:
        filters.append(f"fade=t=in:st=0:d={_f(color.fade_in)}")
    if color.fade_out > 0.0 and clip_duration_sec > 0.0:
        start = max(0.0, clip_duration_sec - color.fade_out)
        filters.append(f"fade=t=out:st={_f(start)}:d={_f(color.fade_out)}")

    return filters


def _tone_curve(shadows: float, highlights: float) -> Optional[str]:
    """A `curves` filter lifting/crushing the ends of the tonal range.

    `eq`'s brightness and gamma move the whole picture; recovering a blown sky
    without flattening the face needs a curve that only bends near one end. The
    control points are the classic quarter/three-quarter tones, left at their
    neutral positions when the corresponding slider is at zero.
    """
    shadows = _clamp(shadows, -1.0, 1.0)
    highlights = _clamp(highlights, -1.0, 1.0)
    if shadows == 0.0 and highlights == 0.0:
        return None
    # 0.25 and 0.75 move by at most 0.2, which keeps the curve monotonic — a
    # non-monotonic curve posterises, turning a gradient into banded blocks.
    low = _clamp(0.25 + 0.20 * shadows, 0.0, 1.0)
    high = _clamp(0.75 + 0.20 * highlights, 0.0, 1.0)
    points = f"0/0 0.25/{_f(low)} 0.75/{_f(high)} 1/1"
    # Quoted: the point list contains no commas today, but `curves` takes them in
    # other forms and an unquoted value would end the filter at the first one.
    return f"curves=all='{points}'"


def _color_balance(color: ColorGrade) -> Optional[str]:
    """`colorbalance` from the three wheels — the only per-tonal-range colour tool.

    FFmpeg names the ranges shadows/midtones/highlights and the parameters
    `rs/gs/bs`, `rm/gm/bm`, `rh/gh/bh`. Each is -1..1 where positive pushes
    toward that primary and negative toward its complement.
    """
    lift = color.lift.clamped()
    mid = color.midtones.clamped()
    gain = color.gain.clamped()
    if lift.is_identity() and mid.is_identity() and gain.is_identity():
        return None
    parts = []
    for suffix, wheel in (("s", lift), ("m", mid), ("h", gain)):
        for channel, value in (("r", wheel.r), ("g", wheel.g), ("b", wheel.b)):
            if value != 0.0:
                parts.append(f"{channel}{suffix}={_f(value)}")
    return "colorbalance=" + ":".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Chroma key
# ---------------------------------------------------------------------------

def build_chroma_key_chain(key: Optional[ChromaKey]) -> List[str]:
    """Filters that key a green/blue screen out of a clip, edges included.

    The whole chain is linear — no split, no second branch — which matters
    because the compiler splices these into a comma-joined chain inside a much
    larger filter_complex. That is possible because `erosion` and `gblur` can be
    aimed at a single plane, so the matte can be choked and feathered *in place*
    on the alpha channel rather than being extracted and merged back.

    Order is the whole quality story:

      1. **Key** in YUV (`chromakey`). A lit green screen is never one RGB value —
         it falls off toward the edges of the frame — but its *hue* stays put, so
         keying on chroma alone tolerates the lighting a colour-distance key cannot.
      2. **Choke** (`erosion` on alpha). Removes the one-to-two pixel green fringe
         that no similarity setting reaches, because those pixels are genuinely
         half green and half subject.
      3. **Feather** (`gblur` on alpha). Without it the composite edge is a hard
         staircase that reads as a cut-out.
      4. **Despill** last, on the *keyed* picture. Green reflected onto a shoulder
         is not an edge problem — it is inside the subject, and any attempt to key
         it away cuts a hole in them.
    """
    if key is None or not key.enabled:
        return []

    color = _key_color(key.color)
    similarity = _clamp(key.similarity, 0.01, 1.0)
    blend = _clamp(key.blend, 0.0, 1.0)
    filters: List[str] = []

    if key.key_type == "color":
        # RGB distance. Right for a flat graphic colour, wrong for a lit screen.
        filters.append("format=rgba")
        filters.append(f"colorkey={color}:{_f(similarity)}:{_f(blend)}")
        filters.append("format=yuva420p")
    else:
        filters.append("format=yuva420p")
        filters.append(f"chromakey={color}:{_f(similarity)}:{_f(blend)}")

    # planes=8 is the alpha plane of yuva420p (1=Y, 2=U, 4=V, 8=A). Aiming these
    # at alpha alone is what keeps the chain linear.
    choke = _clamp(key.choke, 0.0, 1.0)
    if choke > 0.0:
        # Each pass eats one pixel, so this is quantised by nature; three passes
        # is already an aggressive choke on 1080p.
        for _ in range(max(1, int(round(choke * 3)))):
            filters.append("erosion=threshold0=0:threshold1=0:threshold2=0:threshold3=255")

    feather = _clamp(key.feather, 0.0, 1.0)
    if feather > 0.0:
        filters.append(f"gblur=sigma={_f(0.5 + 4.5 * feather)}:steps=2:planes=8")

    if key.show_matte:
        # The matte as a picture: white keeps, black drops. Judging a key against
        # the composite hides every fault that happens to sit on a dark area.
        filters.append("alphaextract")
        filters.append("format=yuv420p")
        return filters

    spill = _clamp(key.spill, 0.0, 1.0)
    if spill > 0.0:
        filters.append("format=rgba")
        filters.append(
            f"despill=type={'blue' if _is_blue(color) else 'green'}"
            f":mix={_f(spill)}:expand={_f(_clamp(key.spill_expand, 0.0, 1.0))}")

    # Leave the clip in rgba: the compositor overlays it, and an alpha-less
    # format here silently discards everything the key just did.
    filters.append("format=rgba")
    return filters


def _key_color(value: str) -> str:
    """Normalise a key colour to the 0xRRGGBB form the key filters expect."""
    text = str(value or "").strip()
    if text.startswith("#"):
        text = "0x" + text[1:]
    elif not text.lower().startswith("0x") and len(text) in (6, 8) and _is_hex(text):
        text = "0x" + text
    return text or "0x00FF00"


def _is_hex(text: str) -> bool:
    return all(c in "0123456789abcdefABCDEF" for c in text)


def _is_blue(color: str) -> bool:
    """Whether a key colour is a blue screen rather than a green one."""
    digits = color[2:] if color.lower().startswith("0x") else color
    if len(digits) < 6 or not _is_hex(digits[:6]):
        return False
    green = int(digits[2:4], 16)
    blue = int(digits[4:6], 16)
    return blue > green


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _crop_filter(t: Transform) -> Optional[str]:
    left = max(0.0, min(0.9, t.crop_left))
    top = max(0.0, min(0.9, t.crop_top))
    right = max(0.0, min(0.9, t.crop_right))
    bottom = max(0.0, min(0.9, t.crop_bottom))
    width_frac = 1.0 - left - right
    height_frac = 1.0 - top - bottom
    if width_frac >= 0.999 and height_frac >= 0.999:
        return None
    # Never let opposing crops collapse the frame to nothing.
    width_frac = max(0.05, width_frac)
    height_frac = max(0.05, height_frac)
    return (f"crop=w=iw*{_f(width_frac)}:h=ih*{_f(height_frac)}"
            f":x=iw*{_f(left)}:y=ih*{_f(top)}")


def _flip_filters(t: Transform) -> List[str]:
    """Mirror the picture. Applied straight after the crop so the crop edges still
    refer to the original frame — cropping the left then flipping keeps the crop on
    what was the left, which is what two independent controls should do."""
    filters: List[str] = []
    if t.flip_h:
        filters.append("hflip")
    if t.flip_v:
        filters.append("vflip")
    return filters


def _rotate_filter(t: Transform, transparent: bool) -> Optional[str]:
    if t.rotation % 360 == 0:
        return None
    rad = math.radians(t.rotation)
    fill = "none" if transparent else "black"
    return (f"rotate={_f(rad)}:ow=rotw({_f(rad)}):oh=roth({_f(rad)}):c={fill}")


def _opacity_filters(t: Transform) -> List[str]:
    if t.opacity >= 1.0:
        return []
    alpha = max(0.0, min(1.0, t.opacity))
    return ["format=rgba", f"colorchannelmixer=aa={_f(alpha)}"]


def build_opacity_filters(transform: Optional[Transform]) -> List[str]:
    """Alpha for compositing something back over the picture at partial strength.

    Used by adjustment layers, where opacity means "how much of the treatment to
    keep" rather than "how see-through this clip is".
    """
    return _opacity_filters(transform or Transform())


def _anim_pair(start: float, end: Optional[float]) -> Tuple[float, float]:
    return start, (start if end is None else end)


def build_canvas_transform(
    transform: Optional[Transform],
    canvas_w: int,
    canvas_h: int,
    duration_frames: int,
    fps: float,
    frame_offset: int = 0,
) -> List[str]:
    """Geometry chain whose output is exactly canvas_w x canvas_h.

    Used for the V1 program and for any overlay that animates its zoom (a
    full-frame Ken Burns move), where the result has to be a complete frame.

    `frame_offset` is for a chain that runs over a stream it does not start with —
    an adjustment layer treats the whole programme, so its animation has to be
    counted from the frame the clip begins at and held still either side of it.
    """
    t = transform or Transform()
    filters: List[str] = []

    crop = _crop_filter(t)
    if crop:
        filters.append(crop)
    filters.extend(_flip_filters(t))
    rotate = _rotate_filter(t, transparent=False)
    if rotate:
        filters.append(rotate)

    scale_start, scale_end = _anim_pair(t.scale, t.scale_end)
    pan_x_start, pan_x_end = _anim_pair(t.pos_x, t.pos_x_end)
    pan_y_start, pan_y_end = _anim_pair(t.pos_y, t.pos_y_end)

    if not t.is_animated():
        scale = max(0.05, scale_start)
        target_w, target_h = _even(canvas_w * scale), _even(canvas_h * scale)
        filters.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease")

        # Pad up to at least the canvas so the final crop always has pixels to
        # take, then cut the canvas out at the panned position.
        pad_w, pad_h = max(target_w, canvas_w), max(target_h, canvas_h)

        # Where the picture sits *inside* that pad. Zoomed in there is no room —
        # the pad is the picture — and the crop below does the panning. Zoomed
        # OUT the picture is smaller than the canvas and all the room is here, so
        # a centred pad was the reason pos_x/pos_y did nothing once scale went
        # under 1.0: you could shrink a clip but never move it off centre, which
        # makes laying two of them side by side impossible.
        room_x, room_y = pad_w - target_w, pad_h - target_h
        place_x = int(round(room_x / 2.0 * (1.0 + max(-1.0, min(1.0, pan_x_start)))))
        place_y = int(round(room_y / 2.0 * (1.0 + max(-1.0, min(1.0, pan_y_start)))))
        place_x = max(0, min(room_x, place_x))
        place_y = max(0, min(room_y, place_y))
        filters.append(f"pad={pad_w}:{pad_h}:{place_x}:{place_y}:color=black")

        slack_x, slack_y = pad_w - canvas_w, pad_h - canvas_h
        offset_x = int(round(slack_x / 2.0 * (1.0 + max(-1.0, min(1.0, pan_x_start)))))
        offset_y = int(round(slack_y / 2.0 * (1.0 + max(-1.0, min(1.0, pan_y_start)))))
        offset_x = max(0, min(slack_x, offset_x))
        offset_y = max(0, min(slack_y, offset_y))
        filters.append(f"crop={canvas_w}:{canvas_h}:{offset_x}:{offset_y}")
        filters.append("setsar=1")
        return filters

    # Animated: pre-scale down to the smaller end of the move so zoompan, which
    # cannot go below 1.0, has room to zoom up into both keyframes.
    #
    # Supersampled: zoompan's crop origin is rounded to whole pixels of its
    # INPUT every output frame, so at canvas resolution a slow zoom visibly
    # judders — the window snaps pixel by pixel. Feeding it a frame upscaled by
    # `_ZOOM_SS` and letting it output the canvas size shrinks that rounding
    # error by the same factor, which is what makes the move read as glide
    # instead of shake. Lanczos on the upscale keeps the picture sharp.
    base = min(scale_start, scale_end, 1.0)
    base = max(0.05, base)
    # Average travel of the crop window's edge per output frame: the zoom's
    # change of window width plus the pan's slide, both in output pixels.
    frames = max(1, duration_frames - 1)
    zoom_travel = canvas_w * abs(1.0 / max(1e-3, scale_start / base)
                                 - 1.0 / max(1e-3, scale_end / base)) / 2.0
    pan_travel = max(abs(pan_x_end - pan_x_start) * canvas_w,
                     abs(pan_y_end - pan_y_start) * canvas_h) / 2.0 * (1.0 - base / max(scale_start, scale_end, base))
    ss = _zoom_ss(canvas_w, canvas_h, base,
                  travel_px_per_frame=(zoom_travel + pan_travel) / frames)
    pre_w, pre_h = _even(canvas_w * base * ss), _even(canvas_h * base * ss)
    filters.append(f"scale={pre_w}:{pre_h}:force_original_aspect_ratio=decrease:flags=lanczos")
    filters.append(f"pad={canvas_w * ss}:{canvas_h * ss}:(ow-iw)/2:(oh-ih)/2:color=black")

    zoom_from, zoom_to = scale_start / base, scale_end / base
    span = max(1, duration_frames - 1)
    # Clamped only when offset: the plain form is what a clip that owns its whole
    # stream needs, and `on/span` reads better in the graph.
    raw = f"on/{span}" if frame_offset <= 0 else f"clip((on-{frame_offset})/{span},0,1)"
    # Smoothstep easing: a linear ramp starts and stops with a visible jolt,
    # which reads as part of the same shakiness. p*p*(3-2p) eases both ends.
    progress = f"({raw})*({raw})*(3-2*({raw}))"
    zoom_expr = f"{_f(zoom_from)}+({_f(zoom_to - zoom_from)})*{progress}"
    pan_x_expr = f"{_f(pan_x_start)}+({_f(pan_x_end - pan_x_start)})*{progress}"
    pan_y_expr = f"{_f(pan_y_start)}+({_f(pan_y_end - pan_y_start)})*{progress}"

    # zoompan crops an (iw/zoom x ih/zoom) window; centre it, then slide by the
    # pan fraction across whatever slack the current zoom leaves.
    x_expr = f"(iw-iw/zoom)/2*(1+({pan_x_expr}))"
    y_expr = f"(ih-ih/zoom)/2*(1+({pan_y_expr}))"

    filters.append(
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d=1:s={canvas_w}x{canvas_h}:fps={_f(fps)}"
    )
    filters.append("setsar=1")
    return filters


def build_overlay_transform(
    transform: Optional[Transform],
    canvas_w: int,
    canvas_h: int,
    duration_frames: int,
    fps: float,
    clip_start_sec: float = 0.0,
    clip_end_sec: float = 0.0,
) -> Tuple[List[str], str, str]:
    """Geometry chain plus the overlay x/y expressions for a composited clip.

    Returns (filters, x_expr, y_expr). A clip that animates its zoom is promoted
    to a full-canvas frame pinned at 0,0 — that is the B-roll Ken Burns case, and
    keeping zoompan on a canvas-sized frame avoids its poor alpha handling.
    Position-only animation stays cheap: `overlay` re-evaluates x/y every frame,
    so the pan rides on a time expression with no extra filter.
    """
    t = transform or Transform()

    if t.is_animated() and (t.scale_end is not None and t.scale_end != t.scale):
        filters = build_canvas_transform(t, canvas_w, canvas_h, duration_frames, fps)
        filters.extend(_opacity_filters(t))
        return filters, "0", "0"

    filters: List[str] = []
    crop = _crop_filter(t)
    if crop:
        filters.append(crop)
    filters.extend(_flip_filters(t))
    rotate = _rotate_filter(t, transparent=t.opacity < 1.0)
    if rotate:
        filters.append(rotate)

    scale = max(0.05, t.scale)
    box_w, box_h = _even(canvas_w * scale), _even(canvas_h * scale)
    filters.append(f"scale={box_w}:{box_h}:force_original_aspect_ratio=decrease")
    filters.extend(_opacity_filters(t))

    pan_x_start, pan_x_end = _anim_pair(t.pos_x, t.pos_x_end)
    pan_y_start, pan_y_end = _anim_pair(t.pos_y, t.pos_y_end)
    animated_pos = (pan_x_start != pan_x_end or pan_y_start != pan_y_end)

    if animated_pos and clip_end_sec > clip_start_sec:
        span = clip_end_sec - clip_start_sec
        progress = f"clip((t-{_f(clip_start_sec)})/{_f(span)},0,1)"
        x_frac = f"({_f(pan_x_start)}+({_f(pan_x_end - pan_x_start)})*{progress})"
        y_frac = f"({_f(pan_y_start)}+({_f(pan_y_end - pan_y_start)})*{progress})"
    else:
        x_frac = f"({_f(pan_x_start)})"
        y_frac = f"({_f(pan_y_start)})"

    # W/H are the main frame, w/h the overlay: centre it, then push by the
    # fraction across half the canvas so +1 parks it against the right edge.
    x_expr = f"(W-w)/2+{x_frac}*W/2"
    y_expr = f"(H-h)/2+{y_frac}*H/2"
    return filters, x_expr, y_expr


def build_fit_to_canvas(canvas_w: int, canvas_h: int) -> List[str]:
    """Letterbox an arbitrary source to the canvas without any user transform."""
    return [
        f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease",
        f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
    ]
