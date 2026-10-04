"""Newspaper front-page stills drawn with PIL, for the clippings a speaker refers to.

A full-frame cutaway that reads as an aged newspaper clipping: masthead,
dateline, headline, a column or two of body copy and a halftone photo box.
Rendered once per beat and cached by its data, the same way charts are.

Two looks, matching the charts and maps: `clean` for explainers (cream aged
paper, black ink), `noir` for the dark genres (desaturated grey paper, near-
black ink).
"""

import hashlib
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .models import Asset, Beat

logger = logging.getLogger("presentation.newspaper")

STYLES: Dict[str, Dict[str, Any]] = {
    "clean": {"bg": (232, 222, 199), "ink": (30, 26, 20), "muted": (96, 88, 72),
              "rule": (30, 26, 20), "photo": (150, 142, 124), "highlight": (255, 232, 70)},
    "noir": {"bg": (150, 148, 142), "ink": (18, 17, 16), "muted": (78, 76, 72),
             "rule": (18, 17, 16), "photo": (90, 88, 84), "highlight": (196, 176, 60)},
}


def _font(size: int, bold: bool = True):
    from PIL import ImageFont
    try:
        from utils.fonts import resolve_font_file
        path = resolve_font_file("Montserrat", bold=bold) or resolve_font_file("Arial", bold=bold)
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _serif_font(size: int, bold: bool = True):
    from PIL import ImageFont
    try:
        from utils.fonts import resolve_font_file
        path = (resolve_font_file("Georgia", bold=bold) or resolve_font_file("Times New Roman", bold=bold)
                or resolve_font_file("Montserrat", bold=bold) or resolve_font_file("Arial", bold=bold))
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("times.ttf", size)
    except Exception:
        return ImageFont.load_default()


def style_for(genre: str) -> str:
    return "noir" if genre in ("horror", "true_crime") else "clean"


def _wrap_to_width(draw, text: str, font, max_width: float) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_newspaper(beat: Beat, width: int, height: int, style: str = "clean"):
    """One aged-newspaper still, drawn from a beat's data (or its text as a fallback)."""
    from PIL import Image, ImageDraw

    look = STYLES.get(style, STYLES["clean"])
    data = beat.data or {}
    ss = 2
    W, H = width * ss, height * ss
    image = Image.new("RGB", (W, H), look["bg"])
    draw = ImageDraw.Draw(image)
    scale = min(W, H) / (1080 * ss)

    margin = int(W * 0.08)
    y = int(H * 0.05)

    masthead = str(data.get("masthead") or "THE DAILY CHRONICLE")
    masthead_font = _serif_font(int(70 * ss * scale))
    tw = draw.textlength(masthead, font=masthead_font)
    draw.line([(margin, y), (W - margin, y)], fill=look["rule"], width=int(4 * ss * scale))
    y += int(14 * ss * scale)
    draw.text(((W - tw) / 2, y), masthead, font=masthead_font, fill=look["ink"])
    y += int(80 * ss * scale)
    draw.line([(margin, y), (W - margin, y)], fill=look["rule"], width=int(4 * ss * scale))
    y += int(20 * ss * scale)

    dateline = str(data.get("dateline") or "")
    dateline_font = _font(int(28 * ss * scale), bold=False)
    if dateline:
        dtw = draw.textlength(dateline, font=dateline_font)
        draw.text(((W - dtw) / 2, y), dateline, font=dateline_font, fill=look["muted"])
        y += int(44 * ss * scale)

    headline = str(data.get("headline") or beat.text or beat.topic or "UNTITLED")
    headline_font = _serif_font(int(66 * ss * scale))
    headline_lines = _wrap_to_width(draw, headline.upper(), headline_font, W - 2 * margin)

    highlight = data.get("highlight")
    highlight_target = str(highlight).strip() if isinstance(highlight, str) and highlight.strip() else None

    line_h = int(78 * ss * scale)
    headline_top = y
    highlight_box = None
    for i, line in enumerate(headline_lines):
        lw = draw.textlength(line, font=headline_font)
        lx = (W - lw) / 2
        ly = y + i * line_h
        if highlight_target and highlight_target.lower() in line.lower():
            pad = int(10 * ss * scale)
            box = (lx - pad, ly - pad * 0.3, lx + lw + pad, ly + line_h * 0.85)
            draw.rectangle(list(box), fill=look["highlight"])
            if highlight_box is None:
                # As fractions of the canvas: this still lines up once the
                # image is resized down to (width, height) below, and it is
                # what the text-fx pass's highlighter-sweep animation reads to
                # know where on screen to sweep.
                highlight_box = (box[0] / W, box[1] / H, box[2] / W, box[3] / H)
    if highlight_box is not None:
        beat.data["highlight_box"] = list(highlight_box)
    for i, line in enumerate(headline_lines):
        lw = draw.textlength(line, font=headline_font)
        lx = (W - lw) / 2
        ly = y + i * line_h
        draw.text((lx, ly), line, font=headline_font, fill=look["ink"])
    y = headline_top + max(1, len(headline_lines)) * line_h + int(8 * ss * scale)

    subhead = data.get("subhead")
    if subhead:
        subhead_font = _serif_font(int(36 * ss * scale), bold=False)
        subhead_lines = _wrap_to_width(draw, str(subhead), subhead_font, W - 2 * margin)
        for line in subhead_lines[:2]:
            lw = draw.textlength(line, font=subhead_font)
            draw.text(((W - lw) / 2, y), line, font=subhead_font, fill=look["muted"])
            y += int(46 * ss * scale)
    y += int(20 * ss * scale)
    draw.line([(margin, y), (W - margin, y)], fill=look["rule"], width=int(2 * ss * scale))
    y += int(24 * ss * scale)

    photo_h = int(H * 0.22)
    photo_w = int(W * 0.34)
    draw.rectangle([margin, y, margin + photo_w, y + photo_h], fill=look["photo"])
    for gx in range(margin, margin + photo_w, int(10 * ss * scale)):
        draw.line([(gx, y), (gx, y + photo_h)], fill=look["bg"], width=1)

    body_lines_data = data.get("body_lines")
    if isinstance(body_lines_data, list) and body_lines_data:
        body_text = " ".join(str(b) for b in body_lines_data)
    else:
        words = (headline.split() or ["The", "story", "continues"])
        filler_words = (words * 40)[:220]
        body_text = " ".join(filler_words)

    body_font = _font(int(24 * ss * scale), bold=False)
    col_gap = int(24 * ss * scale)
    text_top = y
    text_bottom = H - int(H * 0.05)
    col1_x0 = margin + photo_w + col_gap
    col_area_w = (W - margin) - col1_x0
    num_cols = 2 if col_area_w > int(360 * ss * scale) else 1
    col_w = (col_area_w - col_gap * (num_cols - 1)) / max(1, num_cols) if num_cols > 1 else col_area_w
    body_lines = _wrap_to_width(draw, body_text, body_font, col_w)
    body_line_h = int(34 * ss * scale)
    max_lines_per_col = max(1, int((text_bottom - text_top) / body_line_h))
    for ci in range(num_cols):
        cx = col1_x0 + ci * (col_w + col_gap)
        chunk = body_lines[ci * max_lines_per_col:(ci + 1) * max_lines_per_col]
        cy = text_top
        for line in chunk:
            draw.text((cx, cy), line, font=body_font, fill=look["ink"])
            cy += body_line_h

    below_photo_y = y + photo_h + int(16 * ss * scale)
    below_photo_h = max(0, int(text_bottom - below_photo_y))
    if below_photo_h > body_line_h:
        cy = below_photo_y
        extra_lines = _wrap_to_width(draw, body_text, body_font, photo_w)
        for line in extra_lines:
            if cy + body_line_h > text_bottom:
                break
            draw.text((margin, cy), line, font=body_font, fill=look["ink"])
            cy += body_line_h

    return image.resize((width, height), Image.LANCZOS)


def render_newspaper_assets(beats: Sequence[Beat], project_dir: Path, width: int, height: int,
                            genre: str = "general") -> Tuple[List[Asset], List[Dict[str, str]]]:
    assets: List[Asset] = []
    failures: List[Dict[str, str]] = []
    wanted = [b for b in beats if b.kind == "newspaper"]
    if not wanted:
        return assets, failures
    style = style_for(genre)
    directory = Path(project_dir) / "assets" / "newspaper"
    directory.mkdir(parents=True, exist_ok=True)
    for beat in wanted:
        data = beat.data or {}
        key = hashlib.sha1(
            f"{sorted(data.items())!r}:{beat.text}:{style}:{width}x{height}".encode()
        ).hexdigest()[:12]
        path = directory / f"newspaper_{key}.png"
        if not path.exists():
            try:
                image = render_newspaper(beat, width, height, style=style)
                image.save(path)
            except Exception as e:
                logger.warning("Newspaper for %r failed: %s", beat.topic, e)
                failures.append({"beat_id": beat.id, "reason": f"render failed: {e}"})
                continue
        assets.append(Asset(beat_id=beat.id, kind="image", path=str(path), width=width,
                            height=height, cache_hit=True))
    return assets, failures
