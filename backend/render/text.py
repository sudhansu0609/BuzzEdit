"""drawtext filter construction for text and caption clips.

Text content is written to a UTF-8 sidecar file and referenced with
`textfile=` rather than being inlined as `text=`. Inline text has to survive two
layers of filtergraph escaping, and this editor routinely carries Devanagari and
Hinglish transcript text with apostrophes and colons in it — the sidecar sidesteps
all of it. Files are named by content hash, so repeat renders reuse them and the
same project always produces the same filter string.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from timeline.schema import TextClip, TextStyle
from render.effects import escape_filter_path, _f
from utils.fonts import resolve_font_file

logger = logging.getLogger(__name__)


def resolve_style_font(style: TextStyle) -> Optional[str]:
    """The font file drawtext should load for this style."""
    if style.font_file and Path(style.font_file).exists():
        return style.font_file
    return resolve_font_file(style.font_family, bold=style.bold, italic=style.italic)


def write_text_asset(content: str, assets_dir: Path) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    path = assets_dir / f"text_{digest}.txt"
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


def _color_with_alpha(color: str, opacity: float) -> str:
    if opacity >= 1.0 or "@" in color:
        return color
    return f"{color}@{_f(max(0.0, min(1.0, opacity)))}"


def _position_expressions(style: TextStyle, animation_offset: str = "") -> tuple:
    """x/y expressions anchoring the text block by its alignment.

    pos_x/pos_y are -1..1 across the canvas. The anchor point is the left edge,
    centre or right edge of the block depending on `align`; vertically the block
    is always centred on its anchor, so pos_y=0 sits dead centre and pos_y=0.75
    lands in the usual subtitle band.
    """
    px = max(-1.0, min(1.0, style.pos_x))
    py = max(-1.0, min(1.0, style.pos_y))
    anchor_x = f"w*{_f((px + 1.0) / 2.0)}"
    anchor_y = f"h*{_f((py + 1.0) / 2.0)}"

    if style.align == "left":
        x_expr = anchor_x
    elif style.align == "right":
        x_expr = f"{anchor_x}-text_w"
    else:
        x_expr = f"{anchor_x}-text_w/2"

    y_expr = f"{anchor_y}-text_h/2"
    if animation_offset:
        y_expr = f"{y_expr}+({animation_offset})"
    return x_expr, y_expr


def _animation_parts(style: TextStyle, start: float, end: float) -> tuple:
    """(alpha_expression, y_offset_expression) for the configured animation.

    Movement is done with the y expression rather than an animated `fontsize`:
    drawtext re-rasterises the whole glyph set whenever fontsize changes per
    frame, which is ruinously slow once a video has a few hundred caption cards.
    """
    animation = (style.animation or "none").lower()
    duration = max(0.05, style.animation_duration)
    span = max(0.1, end - start)
    duration = min(duration, span / 2.0)

    if animation == "none":
        return None, ""

    fade = (f"min(clip((t-{_f(start)})/{_f(duration)},0,1),"
            f"clip(({_f(end)}-t)/{_f(duration)},0,1))")

    if animation == "fade":
        return fade, ""

    if animation == "pop":
        # Snap in faster than a fade and ride up the last few pixels, which reads
        # as a bounce without touching the glyph cache.
        quick = max(0.03, duration * 0.5)
        alpha = (f"min(clip((t-{_f(start)})/{_f(quick)},0,1),"
                 f"clip(({_f(end)}-t)/{_f(quick)},0,1))")
        offset = f"-{_f(style.font_size * 0.12)}*(1-clip((t-{_f(start)})/{_f(duration)},0,1))"
        return alpha, offset

    if animation == "slide-up":
        offset = f"{_f(style.font_size * 0.9)}*(1-clip((t-{_f(start)})/{_f(duration)},0,1))"
        return fade, offset

    return fade, ""


def build_drawtext(
    clip: TextClip,
    start_sec: float,
    end_sec: float,
    assets_dir: Path,
) -> Optional[str]:
    """One drawtext filter for a text clip, or None if there is nothing to draw."""
    content = (clip.content or "").strip("\n")
    if not content.strip():
        return None

    style = clip.style or TextStyle()
    asset = write_text_asset(content, assets_dir)

    alpha_expr, y_offset = _animation_parts(style, start_sec, end_sec)
    x_expr, y_expr = _position_expressions(style, y_offset)

    parts: List[str] = [f"textfile='{escape_filter_path(asset)}'"]

    font_file = resolve_style_font(style)
    if not font_file:
        # A drawtext with no fontfile falls back to fontconfig's default, which on
        # a machine with libass/fontconfig but no config (Windows ffmpeg builds)
        # aborts the whole render — "Cannot load default config file" → exit 139 →
        # a 500 with no captions at all. Drawing nothing is strictly better than
        # killing the render, so skip this clip when not one font could be found.
        logger.error("No font file resolved for family %r; skipping text %r to "
                     "avoid a fontconfig crash", style.font_family, content[:40])
        return None
    parts.append(f"fontfile='{escape_filter_path(font_file)}'")

    parts.append(f"fontsize={max(4, int(style.font_size))}")
    parts.append(f"fontcolor={_color_with_alpha(style.color, style.opacity)}")
    # Quoted because animated expressions contain commas, which the filtergraph
    # parser would otherwise read as the start of the next filter.
    parts.append(f"x='{x_expr}'")
    parts.append(f"y='{y_expr}'")

    if style.line_spacing:
        parts.append(f"line_spacing={int(style.line_spacing)}")
    if style.stroke_width > 0:
        parts.append(f"borderw={int(style.stroke_width)}")
        parts.append(f"bordercolor={style.stroke_color}")
    if style.shadow_x or style.shadow_y:
        parts.append(f"shadowx={int(style.shadow_x)}")
        parts.append(f"shadowy={int(style.shadow_y)}")
        parts.append(f"shadowcolor={style.shadow_color}")
    if style.box:
        parts.append("box=1")
        parts.append(f"boxcolor={style.box_color}")
        parts.append(f"boxborderw={int(style.box_padding)}")
    if alpha_expr:
        parts.append(f"alpha='{alpha_expr}'")

    parts.append(f"enable='between(t,{_f(start_sec)},{_f(end_sec)})'")
    return "drawtext=" + ":".join(parts)


def build_text_chain(entries: List[tuple], assets_dir: Path) -> List[str]:
    """drawtext filters for a list of (TextClip, start_sec, end_sec) entries."""
    filters = []
    for clip, start_sec, end_sec in entries:
        built = build_drawtext(clip, start_sec, end_sec, assets_dir)
        if built:
            filters.append(built)
    return filters
