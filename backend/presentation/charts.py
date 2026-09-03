"""Chart and timeline cards drawn with PIL, for the numbers the speaker gives.

A statistic on its own is a stat call-out (a big number counting up). Two or
more comparable numbers are a chart: horizontal bars, the larger the longer,
labelled with the values as spoken. Three or more years are a timeline strip.
Both are full-frame cutaways with the same Ken Burns push the B-roll gets,
rendered once and cached by their data.

Two looks, matching the maps: `clean` for explainers, `noir` for the dark
genres.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Asset, Beat

logger = logging.getLogger("presentation.charts")

STYLES: Dict[str, Dict[str, Any]] = {
    "clean": {"bg": (246, 243, 236), "panel": (255, 255, 255), "bar": (224, 57, 62),
              "bar_alt": (46, 111, 171), "text": (28, 28, 30), "muted": (110, 106, 98),
              "grid": (222, 218, 208)},
    "noir": {"bg": (12, 13, 16), "panel": (22, 24, 29), "bar": (200, 16, 46),
             "bar_alt": (110, 116, 124), "text": (216, 208, 196), "muted": (140, 136, 128),
             "grid": (44, 48, 54)},
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


def _format(value: float, unit: str) -> str:
    text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"
    if unit in ("%",):
        return f"{text}%"
    return f"{text} {unit}".strip()


def render_bar_chart(labels: Sequence[str], values: Sequence[float], width: int, height: int,
                     unit: str = "", title: Optional[str] = None, style: str = "clean"):
    """Horizontal bars, largest value full width."""
    from PIL import Image, ImageDraw

    look = STYLES.get(style, STYLES["clean"])
    ss = 2
    W, H = width * ss, height * ss
    image = Image.new("RGB", (W, H), look["bg"])
    draw = ImageDraw.Draw(image)
    scale = min(W, H) / (1080 * ss)

    margin = int(W * 0.08)
    top = int(H * 0.14)
    title_font = _font(int(64 * ss * scale))
    label_font = _font(int(40 * ss * scale), bold=False)
    value_font = _font(int(44 * ss * scale))
    if title:
        draw.text((margin, int(H * 0.06)), title, font=title_font, fill=look["text"])
        top = int(H * 0.06) + int(64 * ss * scale) + int(40 * ss * scale)

    count = max(1, len(values))
    area_h = H - top - int(H * 0.08)
    row_h = area_h / count
    bar_h = min(row_h * 0.55, 120 * ss * scale)
    label_w = max(draw.textlength(str(l), font=label_font) for l in labels) if labels else 0
    bar_x0 = margin + label_w + int(30 * ss * scale)
    bar_max = W - margin - bar_x0 - int(220 * ss * scale)
    peak = max(abs(v) for v in values) or 1.0

    for index, (label, value) in enumerate(zip(labels, values)):
        cy = top + row_h * index + row_h / 2
        draw.text((margin, cy - int(22 * ss * scale)), str(label), font=label_font,
                  fill=look["text"])
        length = max(6 * ss, bar_max * abs(value) / peak)
        colour = look["bar"] if index == max(range(count), key=lambda i: abs(values[i])) else look["bar_alt"]
        draw.rounded_rectangle([bar_x0, cy - bar_h / 2, bar_x0 + length, cy + bar_h / 2],
                               radius=int(bar_h / 4), fill=colour)
        draw.text((bar_x0 + length + int(24 * ss * scale), cy - int(26 * ss * scale)),
                  _format(value, unit), font=value_font, fill=look["text"])

    return image.resize((width, height), Image.LANCZOS)


def render_timeline_strip(years: Sequence[str], width: int, height: int,
                          title: Optional[str] = None, style: str = "clean",
                          active: Optional[str] = None):
    """A horizontal line with the years on it; `active` is drawn large."""
    from PIL import Image, ImageDraw

    look = STYLES.get(style, STYLES["clean"])
    ss = 2
    W, H = width * ss, height * ss
    image = Image.new("RGB", (W, H), look["bg"])
    draw = ImageDraw.Draw(image)
    scale = min(W, H) / (1080 * ss)
    margin = int(W * 0.1)
    cy = int(H * 0.55)
    if title:
        draw.text((margin, int(H * 0.1)), title, font=_font(int(60 * ss * scale)), fill=look["text"])
    draw.line([(margin, cy), (W - margin, cy)], fill=look["grid"], width=int(8 * ss * scale))
    ordered = sorted(dict.fromkeys(years), key=lambda y: str(y))
    count = len(ordered)
    font = _font(int(40 * ss * scale))
    big = _font(int(64 * ss * scale))
    for index, year in enumerate(ordered):
        x = margin + (W - 2 * margin) * (index / max(1, count - 1) if count > 1 else 0.5)
        is_active = active is not None and str(year) == str(active)
        r = int((22 if is_active else 12) * ss * scale)
        draw.ellipse([x - r, cy - r, x + r, cy + r], fill=look["bar"] if is_active else look["bar_alt"])
        used = big if is_active else font
        tw = draw.textlength(str(year), font=used)
        draw.text((x - tw / 2, cy + r + int(18 * ss * scale)), str(year), font=used,
                  fill=look["text"] if is_active else look["muted"])
    return image.resize((width, height), Image.LANCZOS)


def style_for(genre: str) -> str:
    return "noir" if genre in ("horror", "true_crime") else "clean"


def render_chart_assets(beats: Sequence[Beat], project_dir: Path, width: int, height: int,
                        genre: str = "general") -> Tuple[List[Asset], List[Dict[str, str]]]:
    assets: List[Asset] = []
    failures: List[Dict[str, str]] = []
    wanted = [b for b in beats if b.kind == "chart" and b.data]
    if not wanted:
        return assets, failures
    style = style_for(genre)
    directory = Path(project_dir) / "assets" / "charts"
    directory.mkdir(parents=True, exist_ok=True)
    for beat in wanted:
        data = beat.data
        key = hashlib.sha1(f"{sorted(data.items())!r}:{style}:{width}x{height}".encode()).hexdigest()[:12]
        path = directory / f"chart_{key}.png"
        if not path.exists():
            try:
                if data.get("years"):
                    image = render_timeline_strip(data["years"], width, height,
                                                  title=data.get("title") or beat.text,
                                                  style=style, active=data.get("active"))
                else:
                    image = render_bar_chart(data["labels"], data["values"], width, height,
                                             unit=data.get("unit") or "",
                                             title=data.get("title") or None, style=style)
                image.save(path)
            except Exception as e:
                logger.warning("Chart for %r failed: %s", beat.topic, e)
                failures.append({"beat_id": beat.id, "reason": f"render failed: {e}"})
                continue
        assets.append(Asset(beat_id=beat.id, kind="image", path=str(path), width=width,
                            height=height, cache_hit=True))
    return assets, failures
