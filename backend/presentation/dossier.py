"""Case-file / government-archive dossier stills drawn with PIL.

A full-frame cutaway that reads as a case file pulled from a drawer: a manila
or archive-paper background, a folder tab, a rotated rubber stamp, typed
monospace field rows and black redaction bars over the values the beat marks
as secret. Rendered once per beat and cached by its data, the same way charts
and maps are.

Two looks, matching the charts and maps: `clean` for explainers, `noir` for
the dark genres — here that means leaning the palette darker and colder for
horror and true crime rather than switching background tones outright, since
the dossier already has its own manila/archive choice.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .models import Asset, Beat

logger = logging.getLogger("presentation.dossier")

STYLES: Dict[str, Dict[str, Any]] = {
    "clean": {"manila": (196, 166, 108), "archive": (232, 226, 210), "ink": (35, 28, 18),
              "muted": (90, 80, 60), "tab": (216, 188, 130), "stamp": (168, 30, 30),
              "redaction": (18, 18, 18)},
    "noir": {"manila": (120, 104, 74), "archive": (176, 172, 158), "ink": (16, 14, 10),
             "muted": (70, 64, 52), "tab": (140, 122, 86), "stamp": (150, 20, 20),
             "redaction": (8, 8, 8)},
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


def _mono_font(size: int, bold: bool = False):
    from PIL import ImageFont
    try:
        from utils.fonts import resolve_font_file
        path = (resolve_font_file("Courier New", bold=bold) or resolve_font_file("Consolas", bold=bold)
                or resolve_font_file("Montserrat", bold=bold) or resolve_font_file("Arial", bold=bold))
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("cour.ttf", size)
    except Exception:
        return ImageFont.load_default()


def style_for(genre: str) -> str:
    return "noir" if genre in ("horror", "true_crime") else "clean"


def render_case_file(beat: Beat, width: int, height: int, style: str = "clean"):
    """One case-file dossier still, drawn from a beat's data (or its text as a fallback)."""
    from PIL import Image, ImageDraw

    look = STYLES.get(style, STYLES["clean"])
    data = beat.data or {}
    ss = 2
    W, H = width * ss, height * ss
    scale = min(W, H) / (1080 * ss)

    bg_kind = str(data.get("style") or "manila").lower()
    bg_colour = look.get(bg_kind, look["manila"])
    image = Image.new("RGB", (W, H), bg_colour)
    draw = ImageDraw.Draw(image)

    margin = int(W * 0.08)

    tab_w = int(W * 0.32)
    tab_h = int(H * 0.07)
    tab_x0 = margin
    tab_y0 = int(H * 0.03)
    draw.rectangle([tab_x0, tab_y0, tab_x0 + tab_w, tab_y0 + tab_h], fill=look["tab"])
    draw.rectangle([tab_x0, tab_y0, tab_x0 + tab_w, tab_y0 + tab_h], outline=look["ink"],
                   width=int(3 * ss * scale))
    title = str(data.get("title") or "CASE FILE")
    title_font = _font(int(34 * ss * scale))
    tw = draw.textlength(title, font=title_font)
    draw.text((tab_x0 + (tab_w - tw) / 2, tab_y0 + (tab_h - int(34 * ss * scale)) / 2),
              title, font=title_font, fill=look["ink"])

    body_top = tab_y0 + tab_h + int(30 * ss * scale)
    draw.rectangle([margin, body_top, W - margin, H - int(H * 0.05)],
                   outline=look["ink"], width=int(3 * ss * scale))

    fields = data.get("fields")
    redactions = data.get("redactions") or []
    redaction_strings = [str(r) for r in redactions] if isinstance(redactions, list) else []

    label_font = _mono_font(int(28 * ss * scale), bold=True)
    value_font = _mono_font(int(28 * ss * scale), bold=False)
    row_h = int(52 * ss * scale)
    y = body_top + int(30 * ss * scale)
    x0 = margin + int(30 * ss * scale)
    label_w = int(W * 0.20)

    if not isinstance(fields, list) or not fields:
        fields = [
            {"label": "SUBJECT", "value": beat.topic or beat.text or "UNKNOWN"},
            {"label": "SUMMARY", "value": beat.text or beat.summary or ""},
        ]

    max_rows = max(1, int((H - int(H * 0.08) - y) / row_h))
    for row in fields[:max_rows]:
        label = str(row.get("label", "")) if isinstance(row, dict) else ""
        value = str(row.get("value", "")) if isinstance(row, dict) else str(row)
        draw.text((x0, y), f"{label}:", font=label_font, fill=look["muted"])
        value_x = x0 + label_w
        is_redacted = any(r and r in value for r in redaction_strings)
        if is_redacted:
            vw = draw.textlength(value, font=value_font)
            pad = int(8 * ss * scale)
            bar_h = int(30 * ss * scale)
            draw.rectangle([value_x - pad, y - pad * 0.3, value_x + vw + pad, y + bar_h],
                           fill=look["redaction"])
        else:
            draw.text((value_x, y), value, font=value_font, fill=look["ink"])
        y += row_h

    stamp_text = str(data.get("stamp") or "CLASSIFIED").upper()
    stamp_font = _font(int(58 * ss * scale))
    sw = draw.textlength(stamp_text, font=stamp_font)
    sh = int(58 * ss * scale)
    pad = int(24 * ss * scale)
    stamp_w = int(sw + pad * 2)
    stamp_h = int(sh + pad * 2)
    stamp_img = Image.new("RGBA", (stamp_w, stamp_h), (0, 0, 0, 0))
    stamp_draw = ImageDraw.Draw(stamp_img)
    stamp_colour = look["stamp"]
    border_w = max(2, int(6 * ss * scale))
    stamp_draw.rectangle([border_w, border_w, stamp_w - border_w, stamp_h - border_w],
                         outline=(*stamp_colour, 230), width=border_w)
    stamp_draw.text((pad, pad * 0.6), stamp_text, font=stamp_font, fill=(*stamp_colour, 230))
    angle = -12
    rotated = stamp_img.rotate(angle, expand=True, resample=Image.BICUBIC)
    stamp_x = W - margin - rotated.width - int(20 * ss * scale)
    stamp_y = body_top + int(20 * ss * scale)
    image.paste(rotated, (max(margin, stamp_x), max(0, stamp_y)), rotated)

    return image.resize((width, height), Image.LANCZOS)


def render_case_file_assets(beats: Sequence[Beat], project_dir: Path, width: int, height: int,
                            genre: str = "general") -> Tuple[List[Asset], List[Dict[str, str]]]:
    assets: List[Asset] = []
    failures: List[Dict[str, str]] = []
    wanted = [b for b in beats if b.kind == "case_file"]
    if not wanted:
        return assets, failures
    style = style_for(genre)
    directory = Path(project_dir) / "assets" / "dossier"
    directory.mkdir(parents=True, exist_ok=True)
    for beat in wanted:
        data = beat.data or {}
        key = hashlib.sha1(
            f"{sorted(data.items())!r}:{beat.text}:{style}:{width}x{height}".encode()
        ).hexdigest()[:12]
        path = directory / f"dossier_{key}.png"
        if not path.exists():
            try:
                image = render_case_file(beat, width, height, style=style)
                image.save(path)
            except Exception as e:
                logger.warning("Case file for %r failed: %s", beat.topic, e)
                failures.append({"beat_id": beat.id, "reason": f"render failed: {e}"})
                continue
        assets.append(Asset(beat_id=beat.id, kind="image", path=str(path), width=width,
                            height=height, cache_hit=True))
    return assets, failures
