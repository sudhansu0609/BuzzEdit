"""Atmosphere effects — rain, snow, lightning, sunlight, light leaks, fog, grain.

Everything here is generated procedurally from lavfi sources; no stock footage or
overlay plates are needed, so an effect costs nothing but filter time and works at
any resolution.

Three things were learned the hard way while building these, and all three are
load-bearing:

1. **Blend in RGB, never in YUV.** `screen` is defined on light intensities. Applied
   to YUV chroma planes — which are signed offsets around 128 — screening two
   neutral greys gives 191, and the whole picture turns magenta. Every composite
   here converts to `gbrp` first.
2. **`gradients` needs 8-digit hex.** `c0=0xFF7A2A` is parsed with alpha 0 and
   silently yields a transparent (black) gradient; `0xFF7A2AFF` is what was meant.
   Named colours work too.
3. **Noise sits around its base value.** On a black source `noise=alls=60` never
   exceeds ~148, so a threshold above that produces an empty layer. Rain and snow
   start from grey and threshold near 230, which leaves the sparse bright specks
   that become drops.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from render.effects import escape_filter_path
from timeline.schema import AtmosphereEffect

logger = logging.getLogger("atmosphere")

# Effects that only add light. Their colour is irrelevant, so they are built as
# grey layers — cheaper, and it keeps them from tinting the picture.
_LUMA_EFFECTS = {"rain", "snow", "lightning", "fog", "grain"}


def _f(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _moving_box(input_label: str, output_label: str, tag: str, w: float, h: float,
                color: str, x_expr: str, y_expr: str, enable: str) -> List[str]:
    """A solid box whose position animates per frame.

    drawbox cannot do this: it evaluates x/y/w/h ONCE when the graph is
    configured (this FFmpeg 8.1 build has no `eval` option at all -- passing
    `eval=frame` aborted every render that contained one), so a box keyed to
    `t` just sat still. overlay re-evaluates x/y on every frame, so the box is
    a colour source slid over the picture instead. Size stays fixed; anything
    that should "grow" is expressed as a slide.
    """
    box = f"[mbox_{tag}]"
    return [
        f"color=c={color}:s={max(2, int(round(w)))}x{max(2, int(round(h)))},format=rgba{box}",
        f"{input_label}{box}overlay=x='{x_expr}':y='{y_expr}':shortest=1:"
        f"enable='{enable}'{output_label}",
    ]


def available() -> List[Dict[str, object]]:
    """Catalogue for the UI."""
    return [
        {"id": "rain", "label": "Rain", "description": "Falling streaks, screened over the picture"},
        {"id": "snow", "label": "Snow", "description": "Slow drifting flakes"},
        {"id": "lightning", "label": "Lightning", "description": "Periodic storm flashes"},
        {"id": "sunlight", "label": "Sunlight", "description": "Warm bloom from a corner"},
        {"id": "light_leak", "label": "Light Leak", "description": "Analogue colour wash across the frame"},
        {"id": "fog", "label": "Fog", "description": "Soft drifting haze"},
        {"id": "wind", "label": "Wind", "description": "Drifting haze with a slow sway"},
        {"id": "grain", "label": "Film Grain", "description": "Fine moving grain"},
        {"id": "flash", "label": "Flash Frame", "description": "A white (or coloured) hit; window it to a few frames"},
        {"id": "shutter", "label": "Camera Shutter", "description": "Black blades snap closed and reopen; window it to a hit"},
        {"id": "shake", "label": "Camera Shake", "description": "The frame jolts randomly; window it to a hit"},
        {"id": "glitch", "label": "Glitch", "description": "Colour planes tear apart on random frames"},
        {"id": "vhs", "label": "VHS / Found Footage", "description": "Scanlines, chroma smear and grain"},
        {"id": "flicker", "label": "Light Flicker", "description": "A failing light: brightness wobbles"},
        {"id": "redaction", "label": "Redaction Bar", "description": "A black censor bar wipes across the centre; window it to a beat"},
        {"id": "spotlight", "label": "Spotlight", "description": "Edges darken so the centre reads as lit; an emphasis beat"},
        {"id": "strobe", "label": "Strobe", "description": "Rapid white flashes; window it to a hit"},
    ]


def _rain_layer(width: int, height: int, fps: float, duration: float,
                intensity: float, speed: float) -> str:
    # A short noise field stretched tall turns each speck into a streak; scrolling
    # the tall field past the frame makes them fall.
    density = int(round(238 - 12 * intensity))          # lower threshold = more drops
    fall = int(round(500 * speed))
    tall = height * 4
    return (
        f"color=gray:s={width}x{max(60, height // 3)}:r={_f(fps)}:d={_f(duration)},"
        f"noise=alls=60,format=gray,"
        f"lutyuv=y='if(gt(val,{density}),255,0)',"
        f"scale={width}:{tall}:flags=bilinear,"
        f"lutyuv=y='min(255,val*2)',"
        f"crop={width}:{height}:0:'mod(t*{fall},{tall - height})'"
    )


def _snow_layer(width: int, height: int, fps: float, duration: float,
                intensity: float, speed: float) -> str:
    density = int(round(244 - 10 * intensity))
    fall = int(round(110 * speed))
    tall = height * 4
    return (
        f"color=gray:s={max(80, width // 4)}x{height}:r={_f(fps)}:d={_f(duration)},"
        f"noise=alls=60,format=gray,"
        f"lutyuv=y='if(gt(val,{density}),255,0)',"
        f"scale={width}:{tall}:flags=bicubic,gblur=sigma=1.6,"
        f"lutyuv=y='min(255,val*6)',"
        f"crop={width}:{height}:0:'mod(t*{fall},{tall - height})'"
    )


def _lightning_layer(width: int, height: int, fps: float, duration: float,
                     intensity: float, speed: float) -> str:
    # Built tiny and scaled up: the flash is uniform, so per-pixel geq work at full
    # resolution would be wasted. Two spikes per cycle reads as a real strike.
    period = _clamp(3.4 / max(0.15, speed), 0.8, 12.0)
    strength = _clamp(0.55 + 0.45 * intensity, 0.2, 1.0)
    flash = (f"max(0,1-14*abs(mod(T,{_f(period)})-{_f(period * 0.27)}))*{_f(strength)}"
             f"+max(0,1-18*abs(mod(T,{_f(period)})-{_f(period * 0.35)}))*{_f(strength * 0.7)}")
    return (
        f"color=gray:s=32x18:r={_f(fps)}:d={_f(duration)},format=gray,"
        f"geq=lum='255*({flash})',"
        f"scale={width}:{height}:flags=neighbor"
    )


def _sunlight_layer(width: int, height: int, fps: float, duration: float,
                    intensity: float, speed: float, color: str) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0={color}:c1=0x000000FF:type=radial:speed={_f(0.004 * speed)}:"
        f"x0={int(width * 0.22)}:y0={int(height * 0.2)}"
    )


def _light_leak_layer(width: int, height: int, fps: float, duration: float,
                      intensity: float, speed: float, color: str) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0={color}:c1=0x000000FF:type=linear:speed={_f(0.015 * speed)},"
        f"gblur=sigma=24"
    )


def _fog_layer(width: int, height: int, fps: float, duration: float,
               intensity: float, speed: float) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0=0xDFE8F0FF:c1=0x101418FF:type=linear:speed={_f(0.005 * speed)},"
        f"gblur=sigma=30"
    )


def _wind_layer(width: int, height: int, fps: float, duration: float,
                intensity: float, speed: float) -> str:
    return (
        f"gradients=s={width}x{height}:r={_f(fps)}:d={_f(duration)}:"
        f"c0=0xD8E2EAFF:c1=0x14181CFF:type=linear:speed={_f(0.06 * speed)},"
        f"gblur=sigma=26"
    )


# --- presentation text effects --------------------------------------------
#
# behind_head, knockout, focus, annotation and newspaper_sweep are driven by
# the presentation pass's text-fx planner (presentation/text_fx.py) through an
# adjustment clip's `atmosphere` list, exactly like the treatments above — the
# planner sets `effect.type` to one of these names and stashes everything else
# (the word, its own start/end seconds, a matte file, a face box, …) on
# `effect.extra`, since those parameters have nothing in common with
# intensity/speed/colour. They are the only effects that read `assets_dir`
# (they draw text, which needs somewhere to write the textfile FFmpeg reads
# from) or `effect.extra`.

def _text_filter(text: str, assets_dir: Path, font_size: int, color: str,
                 width: int, height: int, x_expr: str, y_expr: str,
                 enable: Optional[str] = None, bold: bool = True,
                 alpha_expr: Optional[str] = None,
                 extra_opts: Optional[List[str]] = None) -> Optional[str]:
    """A bare `drawtext=...` fragment (no `[in]`/`[out]`), or None with no font.

    Deliberately not `render.text.build_drawtext`: that one is shaped around a
    TextStyle and a -1..1 style anchor, where every caller here wants a fixed
    canvas position instead.

    `fontsize` is always a plain integer, never a `t`-driven expression: it
    crashes this build of ffmpeg (access violation) the instant the clip
    becomes visible — reproduced directly against ffmpeg, nothing to do with
    the rest of the graph. A "scale-in" reads instead as `alpha_expr` (and the
    caller's own `y_expr`, usually a small settle offset), the same alpha+
    offset trick `render/text.py`'s drawtext "pop" animation already uses.
    """
    from render.text import write_text_asset, escape_expansion
    from utils.fonts import resolve_font_file

    content = escape_expansion(text)
    if not content.strip():
        return None
    asset = write_text_asset(content, Path(assets_dir))
    font_file = (resolve_font_file("Montserrat", bold=bold)
                or resolve_font_file("Arial", bold=bold))
    if not font_file:
        logger.warning("No font file resolved for a text effect; skipping %r", text[:40])
        return None

    parts = [f"textfile='{escape_filter_path(asset)}'",
             f"fontfile='{escape_filter_path(font_file)}'",
             f"fontsize={max(4, int(font_size))}",
             f"fontcolor={color}", f"x='{x_expr}'", f"y='{y_expr}'",
             "borderw=3", "bordercolor=black@0.8"]
    if alpha_expr:
        parts.append(f"alpha='{alpha_expr}'")
    if extra_opts:
        parts.extend(extra_opts)
    if enable:
        parts.append(f"enable='{enable}'")
    return "drawtext=" + ":".join(parts)


def _behind_head(effect: AtmosphereEffect, input_label: str, output_label: str,
                 width: int, height: int, index) -> List[str]:
    """video, then a big word, then the same video masked to the person on top."""
    extra = effect.extra or {}
    assets_dir = extra.get("_assets_dir")
    matte_path = extra.get("matte_path")
    text = str(extra.get("text") or "").strip()
    if not matte_path or not text or not assets_dir:
        return [f"{input_label}null{output_label}"]
    start_s, end_s = float(extra.get("start_s", 0.0)), float(extra.get("end_s", 1.0))
    matte_offset = float(extra.get("matte_offset", 0.0))
    font_size = int(extra.get("font_size") or max(48, int(min(height * 0.26,
                                                               width * 0.85 / max(1, len(text)) / 0.72))))
    color = str(extra.get("color") or "white")
    window = f"between(t,{_f(start_s)},{_f(end_s)})"

    bg, fg_src = f"[bh_bg_{index}]", f"[bh_fg_{index}]"
    m_raw, m_scaled, m_ts = f"[bh_mraw_{index}]", f"[bh_mscale_{index}]", f"[bh_mts_{index}]"
    fg_rgba, bg_txt = f"[bh_fgr_{index}]", f"[bh_bgt_{index}]"

    txt = _text_filter(text, assets_dir, font_size, color, width, height,
                       x_expr="(w-text_w)/2", y_expr="(h-text_h)/2", enable=window, bold=True)
    if txt is None:
        return [f"{input_label}null{output_label}"]
    return [
        f"{input_label}split=2{bg}{fg_src}",
        f"{bg}{txt}{bg_txt}",
        f"movie='{escape_filter_path(matte_path)}':seek_point={_f(matte_offset)}{m_raw}",
        f"{m_raw}scale={width}:{height}{m_scaled}",
        f"{m_scaled}setpts=PTS-STARTPTS+{_f(start_s)}/TB,format=gray{m_ts}",
        f"{fg_src}{m_ts}alphamerge,format=yuva420p{fg_rgba}",
        f"{bg_txt}{fg_rgba}overlay=x=0:y=0:enable='{window}'{output_label}",
    ]


def _knockout(effect: AtmosphereEffect, input_label: str, output_label: str,
             width: int, height: int, fps: float, duration: float, index) -> List[str]:
    """Full-frame black; the video shows through the glyphs of a word scaling in."""
    extra = effect.extra or {}
    assets_dir = extra.get("_assets_dir")
    text = str(extra.get("text") or "").strip()
    if not text or not assets_dir:
        return [f"{input_label}null{output_label}"]
    start_s, end_s = float(extra.get("start_s", 0.0)), float(extra.get("end_s", 1.5))
    span = max(0.3, end_s - start_s)
    anim = min(0.5, span * 0.4)
    # The word should FILL the frame -- the picture is only seen through it --
    # so size it to the frame: up to ~55% of the height, narrowed for a long
    # word so it still fits the width (bold Montserrat caps run ~0.72 em/char).
    fill = min(height * 0.55, width * 0.9 / max(1, len(text)) / 0.72)
    base_size = int(extra.get("font_size") or max(56, int(fill)))
    window = f"between(t,{_f(start_s)},{_f(end_s)})"
    # A scale-in read through alpha + a settling y-offset rather than a real
    # per-frame fontsize change (see `_text_filter`'s docstring on why).
    progress = f"clip((t-{_f(start_s)})/{_f(anim)},0,1)"
    alpha_expr = progress
    y_settle = f"(h-text_h)/2+{_f(base_size * 0.35)}*(1-{progress})"

    txt = _text_filter(text, assets_dir, base_size, "white", width, height,
                       x_expr="(w-text_w)/2", y_expr=y_settle, enable=window,
                       bold=True, alpha_expr=alpha_expr)
    if txt is None:
        return [f"{input_label}null{output_label}"]

    # The black plate and the glyph mask are cut from the input itself (a
    # full-frame drawbox), not from `color=` sources: a generated source runs
    # its own clock from 0, so a knockout later than the first few seconds of
    # the programme never drew its word and the window rendered solid black.
    src, maskbg, plate = f"[ko_src_{index}]", f"[ko_mbg_{index}]", f"[ko_plate_{index}]"
    masktxt, maskgray, reveal = f"[ko_mtxt_{index}]", f"[ko_mgray_{index}]", f"[ko_rev_{index}]"
    fill_black = "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill"
    # The mask background must stay pure black — it is what the glyph cutout is
    # keyed against, not something shown. The visible plate is the one the user
    # can recolour (a knockout does not have to be black).
    plate_color = extra.get("plate_color")
    if isinstance(plate_color, str) and plate_color.startswith("#"):
        plate_color = "0x" + plate_color[1:]
    fill_plate = f"drawbox=x=0:y=0:w=iw:h=ih:color={plate_color or 'black'}:t=fill"
    return [
        f"{input_label}split=3{src}{maskbg}{plate}",
        f"{maskbg}{fill_black},{txt}{masktxt}",
        f"{masktxt}format=gray{maskgray}",
        f"{src}{maskgray}alphamerge,format=yuva420p{reveal}",
        f"{plate}{fill_plate}:enable='{window}'[ko_black_{index}]",
        f"[ko_black_{index}]{reveal}overlay=x=0:y=0:enable='{window}'{output_label}",
    ]


def _focus(effect: AtmosphereEffect, input_label: str, output_label: str,
          width: int, height: int, index) -> List[str]:
    """Blur+darken everything but the person (matte); a spotlight around the
    face when there is no matte to cut with."""
    extra = effect.extra or {}
    assets_dir = extra.get("_assets_dir")
    start_s, end_s = float(extra.get("start_s", 0.0)), float(extra.get("end_s", 1.0))
    window = f"between(t,{_f(start_s)},{_f(end_s)})"
    matte_path = extra.get("matte_path")
    caption = str(extra.get("caption") or "").strip()
    cap = None
    if caption and assets_dir:
        cap = _text_filter(
            caption, assets_dir, int(extra.get("caption_size") or max(28, int(height * 0.035))),
            "white", width, height, x_expr="(w-text_w)/2", y_expr="h*0.86", enable=window, bold=False)

    main_out = output_label if not cap else f"[fc_pre_{index}]"
    statements: List[str] = []

    if matte_path:
        blur_src, sharp_src = f"[fc_blur_{index}]", f"[fc_sharp_{index}]"
        blurred = f"[fc_blurred_{index}]"
        m_raw, m_scaled, m_ts = f"[fc_mraw_{index}]", f"[fc_mscale_{index}]", f"[fc_mts_{index}]"
        person = f"[fc_person_{index}]"
        matte_offset = float(extra.get("matte_offset", 0.0))
        radius = max(4, int(width * 0.012))
        dim = _f(0.18)
        statements.extend([
            f"{input_label}split=2{blur_src}{sharp_src}",
            f"{blur_src}boxblur=luma_radius={radius}:luma_power=2:enable='{window}',"
            f"eq=brightness=-{dim}:enable='{window}'{blurred}",
            f"movie='{escape_filter_path(matte_path)}':seek_point={_f(matte_offset)}{m_raw}",
            f"{m_raw}scale={width}:{height}{m_scaled}",
            f"{m_scaled}setpts=PTS-STARTPTS+{_f(start_s)}/TB,format=gray{m_ts}",
            f"{sharp_src}{m_ts}alphamerge,format=yuva420p{person}",
            f"{blurred}{person}overlay=x=0:y=0:enable='{window}'{main_out}",
        ])
    else:
        face_x, face_y = float(extra.get("face_x", 0.0)), float(extra.get("face_y", 0.0))
        cx = int(round((face_x + 1.0) / 2.0 * width))
        cy = int(round((face_y + 1.0) / 2.0 * height))
        angle, dim = _f(1.15), _f(0.16)
        statements.append(
            f"{input_label}vignette=angle={angle}:x0={cx}:y0={cy}:enable='{window}',"
            f"eq=brightness=-{dim}:enable='{window}'{main_out}"
        )

    if cap:
        statements.append(f"{main_out}{cap}{output_label}")
    return statements


def _annotation(effect: AtmosphereEffect, input_label: str, output_label: str,
                width: int, height: int, index) -> List[str]:
    """A hand-drawn-style circle / underline / arrow drawn on near a word."""
    extra = effect.extra or {}
    assets_dir = extra.get("_assets_dir")
    start_s, end_s = float(extra.get("start_s", 0.0)), float(extra.get("end_s", 1.0))
    span = max(0.2, end_s - start_s)
    anim = min(0.6, span * 0.5)
    shape = str(extra.get("shape") or "underline").lower()
    cx = _clamp(float(extra.get("cx", 0.5)), 0.0, 1.0) * width
    cy = _clamp(float(extra.get("cy", 0.5)), 0.0, 1.0) * height
    size = max(20, int(_clamp(float(extra.get("size", 0.16)), 0.03, 0.6) * width))
    color = str(extra.get("color") or "#FF3B30")
    window = f"between(t,{_f(start_s)},{_f(end_s)})"
    grow = f"clip((t-{_f(start_s)})/{_f(anim)},0,1)"

    if shape == "arrow":
        if not assets_dir:
            return [f"{input_label}null{output_label}"]
        glyph = "↙"  # points down-left at the word it sits above-right of
        # Pops in via alpha (a real per-frame fontsize crashes this build of
        # ffmpeg — see `_text_filter`'s docstring).
        txt = _text_filter(glyph, assets_dir, size, color, width, height,
                           x_expr=f"{_f(cx)}", y_expr=f"{_f(cy)}", enable=window, bold=True,
                           alpha_expr=grow)
        if txt is None:
            return [f"{input_label}null{output_label}"]
        return [f"{input_label}{txt}{output_label}"]

    thickness = max(3, int(size * 0.09))
    if shape == "circle":
        # No ellipse-outline filter ships with FFmpeg; an unfilled box around
        # the word reads as a rough hand-drawn circle at this thickness/size.
        # Static: drawbox evaluates its geometry once (see `_moving_box`), so
        # it appears for the window rather than scaling in.
        return [
            f"{input_label}drawbox=x={_f(cx - size / 2.0)}:y={_f(cy - size * 0.31)}:"
            f"w={_f(size)}:h={_f(size * 0.62)}:color={color}:t={thickness}:"
            f"enable='{window}'{output_label}"
        ]

    # underline (default): swipes in from the left over `anim` seconds, then
    # rests under the word.
    x_rest = cx - size / 2.0
    return _moving_box(input_label, output_label, f"ul_{index}", size, thickness, color,
                       x_expr=f"{_f(x_rest)}-{_f(size)}*(1-{grow})", y_expr=_f(cy),
                       enable=window)


def _newspaper_sweep(effect: AtmosphereEffect, input_label: str, output_label: str,
                     width: int, height: int, index) -> List[str]:
    """A highlighter bar sweeping across a newspaper clipping's highlighted phrase."""
    extra = effect.extra or {}
    start_s, end_s = float(extra.get("start_s", 0.0)), float(extra.get("end_s", 0.6))
    box = extra.get("box") or [0.1, 0.1, 0.9, 0.2]
    x0, y0, x1, y1 = (float(v) for v in box)
    bx0, by0 = x0 * width, y0 * height
    bw, bh = max(10.0, (x1 - x0) * width), max(6.0, (y1 - y0) * height)
    window = f"between(t,{_f(start_s)},{_f(end_s)})"
    span = max(0.15, end_s - start_s)
    progress = f"clip((t-{_f(start_s)})/{_f(span)},0,1)"
    sweep_w = bw * 0.22
    x_expr = f"{_f(bx0)}+({_f(bw - sweep_w)})*{progress}"
    color = str(extra.get("color") or "#FFE23A")
    return _moving_box(input_label, output_label, f"np_{index}", sweep_w, bh,
                       f"{color}@0.55", x_expr=x_expr, y_expr=_f(by0), enable=window)


def build_effect(
    effect: AtmosphereEffect,
    input_label: str,
    output_label: str,
    width: int,
    height: int,
    fps: float,
    duration: float,
    index: Union[int, str],
    assets_dir: Optional[Path] = None,
) -> List[str]:
    """Filter statements compositing one effect onto `input_label`.

    Returns complete `[in]…[out]` statements ready to join with ";".

    `index` only has to be unique within the graph — it names this effect's
    intermediate labels. Adjustment layers pass a compound id like "2_0" so their
    effects cannot collide with the programme-wide ones.

    `assets_dir` is only read by the presentation text-fx effects, which need
    somewhere to write the drawtext sidecar file for the word they draw.
    """
    kind = (effect.type or "").lower()
    intensity = _clamp(effect.intensity, 0.0, 1.0)
    speed = _clamp(effect.speed, 0.1, 4.0)
    duration = max(0.5, duration)

    if kind in ("behind_head", "knockout", "focus", "annotation", "newspaper_sweep"):
        # Smuggled through `extra` rather than a new build_effect parameter:
        # this dict is the effect's own private payload, and every other
        # caller of build_effect passes an effect with an empty one.
        if assets_dir is not None and "_assets_dir" not in effect.extra:
            effect.extra["_assets_dir"] = assets_dir
        if kind == "behind_head":
            return _behind_head(effect, input_label, output_label, width, height, index)
        if kind == "knockout":
            return _knockout(effect, input_label, output_label, width, height, fps, duration, index)
        if kind == "focus":
            return _focus(effect, input_label, output_label, width, height, index)
        if kind == "annotation":
            return _annotation(effect, input_label, output_label, width, height, index)
        return _newspaper_sweep(effect, input_label, output_label, width, height, index)

    # Grain needs no layer — it is applied straight to the picture.
    if kind == "grain":
        amount = int(round(4 + 26 * intensity))
        return [f"{input_label}noise=alls={amount}:allf=t+u{output_label}"]

    # --- the treatments: no layer, the picture itself is disturbed -----------
    # These are what a horror edit reaches for at a hit. They are meant to be
    # windowed by an adjustment clip (a flash is two frames, a shake half a
    # second); over a whole programme they would be unwatchable.
    if kind == "whiteflash":
        # A dip to white: brightness rises and falls over the window (a sine
        # hump), the documentary cut between two stills. `t` is local to the
        # window, as for the shutter below.
        peak = _f(_clamp(0.55 + 0.45 * intensity, 0.0, 1.0))
        return [f"{input_label}eq=brightness='{peak}*sin(PI*min(t/{_f(duration)},1))':eval=frame"
                f"{output_label}"]

    if kind == "flash":
        colour = (effect.color or "white").replace("0x", "#")[:7] if effect.color else "white"
        alpha = _f(_clamp(0.5 + 0.5 * intensity, 0.0, 1.0))
        return [f"{input_label}drawbox=color={colour}@{alpha}:t=fill{output_label}"]

    if kind == "shutter":
        # A fast camera-shutter snap: two black blades close from the top and
        # bottom edges to meet at the vertical centre by the window's
        # midpoint, then reopen by its end. `t` is local to this filter chain
        # (the window starts at zero), so one triangle expression over
        # t/duration — up 0→0.5, down 0.5→1 — drives both blades' height.
        # intensity scales how fully it closes. The blades are fixed-size
        # boxes slid in from above and below (`_moving_box`): drawbox cannot
        # animate its height.
        close = _f(_clamp(intensity, 0.05, 1.0))
        d = _f(duration)
        frac = f"if(lte(t/{d}\\,0.5)\\,(t/{d})*2\\,(1-t/{d})*2)"
        travel = f"({height}/2)*{close}*({frac})"
        half = height / 2.0
        mid = f"[shut_mid_{index}]"
        return (
            _moving_box(input_label, mid, f"shut_t_{index}", width, half, "black",
                        x_expr="0", y_expr=f"-{_f(half)}+{travel}", enable="1")
            + _moving_box(mid, output_label, f"shut_b_{index}", width, half, "black",
                          x_expr="0", y_expr=f"{height}-{travel}", enable="1")
        )

    if kind == "redaction":
        # A classified censor bar: a black band across the centre grows from
        # nothing to full width over the window. eval=frame is required or
        # drawbox only evaluates the width expression once, at t=0. The comma
        # inside min(...) has to be escaped, or ffmpeg reads it as ending this
        # drawbox and starting a new (invalid) filter — same reason the
        # shutter blades' frac escapes its commas.
        # The band slides in from the left edge (`_moving_box`): drawbox
        # cannot animate its width.
        d = _f(duration)
        band_h = _clamp(0.12 + 0.05 * intensity, 0.1, 0.22) * height
        alpha = _f(_clamp(0.85 + 0.15 * intensity, 0.0, 1.0))
        return _moving_box(input_label, output_label, f"redact_{index}", width, band_h,
                           f"black@{alpha}", x_expr=f"-{width}+{width}*min(1\\,t/{d})",
                           y_expr=_f(height * 0.44), enable="1")

    if kind == "spotlight":
        # Vignette pulls the edges down hard and a small brightness pull dims
        # the whole frame slightly, so the centre reads as the lit subject.
        angle = _f(_clamp(1.05 + 0.35 * intensity, 0.0, 1.4))
        dim = _f(0.04 + 0.06 * intensity)
        return [f"{input_label}vignette=angle='{angle}',eq=brightness=-{dim}{output_label}"]

    if kind == "strobe":
        # A gated white flash: enable() only passes the drawbox through for
        # the first 35% of each cycle, so the frame flicks between the source
        # and solid white at `rate` cycles per second. The commas inside
        # enable's expression are escaped the same way as the shutter
        # blades' frac.
        rate = _f(6.0 * speed)
        alpha = _f(_clamp(0.5 + 0.5 * intensity, 0.0, 1.0))
        return [
            f"{input_label}drawbox=color=white@{alpha}:t=fill:"
            f"enable='lt(mod(t*{rate}\\,1)\\,0.35)'{output_label}"
        ]

    if kind == "shake":
        # Crop a window that wanders randomly every frame, scale back up. The
        # amplitude scales with intensity; `random` is re-seeded per axis so the
        # two do not move together.
        amp = max(2, int(round(width * (0.006 + 0.03 * intensity))))
        return [
            f"{input_label}crop=w=iw-{2 * amp}:h=ih-{2 * amp}:"
            f"x='{amp}+{amp}*(random(1)-0.5)*2':y='{amp}+{amp}*(random(2)-0.5)*2',"
            f"scale={width}:{height}:flags=bilinear,setsar=1{output_label}"
        ]

    if kind == "glitch":
        # Red and blue planes torn apart on a random fifth of the frames, with
        # a burst of noise on the same frames so the tear reads as damage.
        shift = max(2, int(round(width * (0.003 + 0.012 * intensity))))
        chance = _f(0.08 + 0.3 * intensity)
        gate = f"lt(random(3),{chance})"
        return [
            f"{input_label}rgbashift=rh=-{shift}:bh={shift}:enable='{gate}',"
            f"noise=alls={int(round(30 + 40 * intensity))}:allf=t:enable='{gate}'{output_label}"
        ]

    if kind == "vhs":
        # Scanlines, softened chroma pushed sideways, grain and a slight
        # desaturation: found-footage in one chain. geq costs real time at
        # full resolution, so the scanline pass runs at 1/3 height first.
        dark = _f(1.0 - (0.18 + 0.2 * intensity))
        return [
            f"{input_label}chromashift=cbh=-{2 + int(3 * intensity)}:crh={2 + int(3 * intensity)},"
            f"eq=saturation={_f(0.85 - 0.2 * intensity)}:contrast={_f(1.0 + 0.08 * intensity)},"
            f"noise=alls={int(round(10 + 24 * intensity))}:allf=t+u,"
            f"geq=lum='if(mod(Y\\,3)\\,lum(X\\,Y)\\,lum(X\\,Y)*{dark})':cb='cb(X\\,Y)':cr='cr(X\\,Y)'"
            f"{output_label}"
        ]

    if kind == "flicker":
        # A failing light: brightness wobbles on two incommensurate sines.
        depth = _f(0.03 + 0.09 * intensity)
        rate = _f(23.0 * speed)
        return [
            f"{input_label}eq=brightness='-{depth}*0.5+{depth}*sin(t*{rate})*sin(t*{_f(7.3 * speed)})':"
            f"eval=frame{output_label}"
        ]

    # Wind sways the frame as well as hazing it, which is what sells the movement.
    if kind == "wind":
        drift_x = 0.012 + 0.03 * intensity
        drift_y = drift_x * 0.45
        over_w = int(round(width * (1 + drift_x * 2)))
        over_h = int(round(height * (1 + drift_y * 2)))
        over_w += over_w % 2
        over_h += over_h % 2
        sway = (
            f"{input_label}scale={over_w}:{over_h},"
            f"crop={width}:{height}:"
            f"'{(over_w - width) // 2}+{int((over_w - width) * 0.45)}*sin(t*1.7)':"
            f"'{(over_h - height) // 2}+{int((over_h - height) * 0.45)}*sin(t*1.1)'"
            f"[wind_sway{index}]"
        )
        layer = _wind_layer(width, height, fps, duration, intensity, speed)
        return [
            sway,
            f"{layer},format=gbrp[wind_haze{index}]",
            f"[wind_sway{index}]format=gbrp[wind_base{index}]",
            # shortest=1: the generated layer is built long enough to cover the
            # programme, and without this blend would stretch the *programme* out
            # to match the layer instead of the other way round.
            f"[wind_base{index}][wind_haze{index}]"
            f"blend=all_mode=screen:shortest=1:all_opacity={_f(0.10 + 0.25 * intensity)},"
            f"format=yuv420p{output_label}",
        ]

    builders = {
        "rain": lambda: _rain_layer(width, height, fps, duration, intensity, speed),
        "snow": lambda: _snow_layer(width, height, fps, duration, intensity, speed),
        "lightning": lambda: _lightning_layer(width, height, fps, duration, intensity, speed),
        "fog": lambda: _fog_layer(width, height, fps, duration, intensity, speed),
        "sunlight": lambda: _sunlight_layer(width, height, fps, duration, intensity, speed,
                                            effect.color or "0xFFE9B0FF"),
        "light_leak": lambda: _light_leak_layer(width, height, fps, duration, intensity, speed,
                                                effect.color or "0xFF7A2AFF"),
    }
    build = builders.get(kind)
    if build is None:
        logger.warning("Unknown atmosphere effect %r; skipping", effect.type)
        return []

    opacity = {
        "rain": 0.25 + 0.5 * intensity,
        "snow": 0.35 + 0.55 * intensity,
        "lightning": 0.45 + 0.5 * intensity,
        "fog": 0.12 + 0.35 * intensity,
        "sunlight": 0.15 + 0.45 * intensity,
        "light_leak": 0.12 + 0.4 * intensity,
    }[kind]

    layer = f"layer{index}"
    # gbrp on both sides: screen is an RGB operation, and running it over YUV
    # chroma is what turns the whole frame magenta.
    return [
        f"{builders[kind]()},format=gbrp[{layer}]",
        f"{input_label}format=gbrp[atmo_base{index}]",
        # shortest=1 is load-bearing: the layer is deliberately generated a little
        # longer than the programme, and blend otherwise pads the programme out to
        # the layer's length — which silently undid every transition's shortening
        # and left the video running 1.2s past its own audio.
        f"[atmo_base{index}][{layer}]"
        f"blend=all_mode=screen:shortest=1:all_opacity={_f(_clamp(opacity, 0.0, 1.0))},"
        f"format=yuv420p{output_label}",
    ]


def build_aspect_bars(ratio: float, width: int, height: int,
                      input_label: str, output_label: str) -> Optional[str]:
    """Letterbox to a cinematic aspect ratio, keeping the output resolution.

    Crops the picture to the target shape and pads the black bars back on, so the
    file stays 16:9 (or whatever it was) and simply *looks* like 2.39:1.
    """
    if ratio <= 0:
        return None
    current = width / max(1, height)
    if abs(current - ratio) < 0.01:
        return None

    if ratio > current:                      # wider target: bars top and bottom
        keep = int(round(width / ratio))
        keep -= keep % 2
        keep = max(2, min(height, keep))
        if keep >= height:
            return None
        return (f"{input_label}crop={width}:{keep}:0:{(height - keep) // 2},"
                f"pad={width}:{height}:0:{(height - keep) // 2}:color=black{output_label}")

    keep = int(round(height * ratio))         # taller target: bars left and right
    keep -= keep % 2
    keep = max(2, min(width, keep))
    if keep >= width:
        return None
    return (f"{input_label}crop={keep}:{height}:{(width - keep) // 2}:0,"
            f"pad={width}:{height}:{(width - keep) // 2}:0:color=black{output_label}")
