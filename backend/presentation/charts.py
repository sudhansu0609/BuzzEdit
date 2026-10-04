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


def _tint(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """Lighten (factor > 0, toward white) or darken (factor < 0, toward black) an RGB colour."""
    r, g, b = rgb
    if factor >= 0:
        r = r + (255 - r) * factor
        g = g + (255 - g) * factor
        b = b + (255 - b) * factor
    else:
        r = r * (1 + factor)
        g = g * (1 + factor)
        b = b * (1 + factor)
    return (int(max(0, min(255, r))), int(max(0, min(255, g))), int(max(0, min(255, b))))


def _pie_palette(look: Dict[str, Any], count: int, accent_index: int) -> List[Tuple[int, int, int]]:
    """Wedge colours: `bar` for the largest slice, `bar_alt` and derived tints for the rest."""
    factors = [0.0, 0.35, -0.25, 0.55, -0.45, 0.2, -0.15]
    colours: List[Tuple[int, int, int]] = []
    fi = 0
    for i in range(count):
        if i == accent_index:
            colours.append(look["bar"])
        else:
            colours.append(_tint(look["bar_alt"], factors[fi % len(factors)]))
            fi += 1
    return colours


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


def render_pie_chart(labels: Sequence[str], values: Sequence[float], width: int, height: int,
                     unit: str = "", title: Optional[str] = None, style: str = "clean"):
    """Filled wedges sized by share of total, with a legend and the largest wedge in accent."""
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
    label_font = _font(int(38 * ss * scale), bold=False)
    if title:
        draw.text((margin, int(H * 0.06)), title, font=title_font, fill=look["text"])
        top = int(H * 0.06) + int(64 * ss * scale) + int(40 * ss * scale)

    values = list(values) or [0.0]
    labels = list(labels)
    count = len(values)
    total = sum(abs(v) for v in values) or 1.0
    accent_index = max(range(count), key=lambda i: abs(values[i]))
    colours = _pie_palette(look, count, accent_index)

    bottom = H - int(H * 0.08)
    diameter = min(W * 0.46, bottom - top)
    diameter = max(diameter, 10 * ss)
    cx = margin + diameter / 2
    cy = top + (bottom - top) / 2
    bbox = [cx - diameter / 2, cy - diameter / 2, cx + diameter / 2, cy + diameter / 2]

    start_angle = -90.0
    for index, value in enumerate(values):
        share = abs(value) / total
        sweep = share * 360.0
        if sweep > 0:
            end_angle = start_angle + sweep
            draw.pieslice(bbox, start_angle, end_angle, fill=colours[index])
            start_angle = end_angle

    legend_x = cx + diameter / 2 + int(60 * ss * scale)
    legend_y = top + int(10 * ss * scale)
    row_h = max(int(56 * ss * scale), (bottom - top) / count)
    swatch = int(34 * ss * scale)
    for index, value in enumerate(values):
        label = labels[index] if index < len(labels) else ""
        ly = legend_y + row_h * index
        draw.rectangle([legend_x, ly, legend_x + swatch, ly + swatch], fill=colours[index])
        pct = abs(value) / total * 100.0
        text = f"{label} — {pct:.0f}%" if label else f"{pct:.0f}%"
        draw.text((legend_x + swatch + int(20 * ss * scale), ly - int(2 * ss * scale)), text,
                  font=label_font, fill=look["text"])

    return image.resize((width, height), Image.LANCZOS)


def render_line_chart(labels: Sequence[str], values: Sequence[float], width: int, height: int,
                      unit: str = "", title: Optional[str] = None, style: str = "clean"):
    """A single line with area fill, gridlines, point markers, and labels on the peak."""
    from PIL import Image, ImageDraw

    look = STYLES.get(style, STYLES["clean"])
    ss = 2
    W, H = width * ss, height * ss
    image = Image.new("RGB", (W, H), look["bg"])
    draw = ImageDraw.Draw(image)
    scale = min(W, H) / (1080 * ss)

    margin = int(W * 0.1)
    top = int(H * 0.14)
    title_font = _font(int(64 * ss * scale))
    label_font = _font(int(36 * ss * scale), bold=False)
    value_font = _font(int(40 * ss * scale))
    if title:
        draw.text((margin, int(H * 0.06)), title, font=title_font, fill=look["text"])
        top = int(H * 0.06) + int(64 * ss * scale) + int(40 * ss * scale)

    values = list(values) or [0.0]
    labels = list(labels)
    count = len(values)
    bottom = H - int(H * 0.14)
    left = margin
    right = W - margin

    vmax = max(values)
    vmin = min(values)
    if vmax == vmin:
        vmax += 1.0
        vmin -= 1.0
    span = vmax - vmin
    vmax += span * 0.15
    if vmin < 0:
        vmin -= span * 0.15

    def x_at(i: int) -> float:
        return left if count == 1 else left + (right - left) * (i / (count - 1))

    def y_at(v: float) -> float:
        return bottom - (bottom - top) * ((v - vmin) / (vmax - vmin))

    grid_lines = 4
    for g in range(grid_lines + 1):
        gy = top + (bottom - top) * g / grid_lines
        draw.line([(left, gy), (right, gy)], fill=look["grid"], width=max(1, int(2 * ss * scale)))

    points = [(x_at(i), y_at(v)) for i, v in enumerate(values)]

    if points:
        fill_colour = _tint(look["bar_alt"], 0.78)
        area = points + [(points[-1][0], bottom), (points[0][0], bottom)]
        draw.polygon(area, fill=fill_colour)

    if len(points) > 1:
        draw.line(points, fill=look["bar"], width=max(2, int(6 * ss * scale)), joint="curve")

    r = int(12 * ss * scale)
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=look["panel"], outline=look["bar"],
                     width=max(2, int(4 * ss * scale)))

    peak_val = max(values)
    for i, v in enumerate(values):
        if v == peak_val:
            x, y = points[i]
            text = _format(v, unit)
            tw = draw.textlength(text, font=value_font)
            draw.text((x - tw / 2, y - r - int(50 * ss * scale)), text, font=value_font, fill=look["text"])

    for i in range(count):
        label = labels[i] if i < len(labels) else ""
        if not label:
            continue
        x = x_at(i)
        tw = draw.textlength(str(label), font=label_font)
        draw.text((x - tw / 2, bottom + int(20 * ss * scale)), str(label), font=label_font,
                  fill=look["muted"])

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
        chart_type = str(data.get("chart_type", "")).lower()
        key = hashlib.sha1(
            f"{sorted(data.items())!r}:{style}:{width}x{height}:{chart_type}".encode()
        ).hexdigest()[:12]
        path = directory / f"chart_{key}.png"
        if not path.exists():
            try:
                if data.get("years"):
                    image = render_timeline_strip(data["years"], width, height,
                                                  title=data.get("title") or beat.text,
                                                  style=style, active=data.get("active"))
                elif chart_type == "pie":
                    image = render_pie_chart(data["labels"], data["values"], width, height,
                                             unit=data.get("unit") or "",
                                             title=data.get("title") or None, style=style)
                elif chart_type in ("line", "graph"):
                    image = render_line_chart(data["labels"], data["values"], width, height,
                                              unit=data.get("unit") or "",
                                              title=data.get("title") or None, style=style)
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
