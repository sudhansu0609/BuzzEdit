"""Animated text through libass.

drawtext can fade, pop and slide a block of one colour. Everything a YouTube
edit actually does with text — the word being spoken lighting up, a location
card typing itself out, a title scaling in from large, a horror caption that
flickers or glitches — needs per-span colour, per-character timing and
animated transforms. ASS (Advanced SubStation Alpha) has all of that as
override tags, and FFmpeg's `ass` filter renders it with libass, which also
shapes Devanagari through HarfBuzz — so the native-script captions keep
working, and the font-file crash that haunts drawtext does not apply (libass
resolves families itself, with `fontsdir` as an extra search path).

One `.ass` file per render carries every animated clip; the compiler burns it
in with a single filter and leaves the plain clips to drawtext. Styles are
deduplicated by content, Dialogue lines are sorted by start, and every tag is
built from the same TextStyle the UI edits, so a card looks the same whichever
engine draws it.
"""

import hashlib
import logging
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from timeline.schema import TextClip, TextStyle

logger = logging.getLogger(__name__)

# Animations that only this engine can draw. A clip with any of these (or with
# word timings under a karaoke style) is routed here by the compiler.
ASS_ANIMATIONS = {"typewriter", "karaoke", "scale_in", "blur_in", "glitch", "shake", "flicker"}

_NAMED_COLOURS = {
    "white": "FFFFFF", "black": "000000", "red": "FF0000", "yellow": "FFFF00",
    "green": "00FF00", "blue": "0000FF", "orange": "FFA500", "gray": "808080",
    "grey": "808080", "cyan": "00FFFF", "magenta": "FF00FF", "gold": "FFD700",
}


def needs_ass(clip: Optional[TextClip]) -> bool:
    if clip is None or clip.style is None:
        return False
    if clip.second_line:
        return True
    animation = (clip.style.animation or "none").lower()
    if animation == "karaoke":
        return bool(clip.words)
    return animation in ASS_ANIMATIONS


# --- colours and times ---------------------------------------------------------------

def _parse_colour(value: Optional[str], opacity: float = 1.0) -> Tuple[str, float]:
    """(RRGGBB hex, alpha 0..1) from the colour forms the TextStyle carries:
    '#RRGGBB', '0xRRGGBB', a name, any of those with '@0.6' appended."""
    text = (value or "white").strip()
    alpha = max(0.0, min(1.0, opacity))
    if "@" in text:
        text, _, tail = text.partition("@")
        try:
            alpha *= max(0.0, min(1.0, float(tail)))
        except ValueError:
            pass
    text = text.strip().lower()
    if text.startswith("#"):
        text = text[1:]
    elif text.startswith("0x"):
        text = text[2:]
    if text in _NAMED_COLOURS:
        rgb = _NAMED_COLOURS[text]
    elif re.fullmatch(r"[0-9a-f]{6}", text):
        rgb = text.upper()
    elif re.fullmatch(r"[0-9a-f]{8}", text):
        rgb = text[:6].upper()
        alpha *= int(text[6:], 16) / 255.0
    else:
        rgb = "FFFFFF"
    return rgb.upper(), alpha


def ass_colour(value: Optional[str], opacity: float = 1.0) -> str:
    """'&HAABBGGRR' — ASS stores blue-green-red with an inverted alpha."""
    rgb, alpha = _parse_colour(value, opacity)
    aa = int(round((1.0 - alpha) * 255))
    return f"&H{aa:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def ass_rgb(value: Optional[str]) -> str:
    """'&HBBGGRR&' for inline \\c overrides."""
    rgb, _ = _parse_colour(value)
    return f"&H{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}&"


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rest = seconds % 60
    centis = int(round((rest - int(rest)) * 100))
    if centis == 100:
        centis = 0
        rest += 1
    return f"{hours}:{minutes:02d}:{int(rest):02d}.{centis:02d}"


def escape_text(text: str) -> str:
    """ASS text: braces would open an override block, newlines are \\N."""
    return (text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
            .replace("\r\n", "\n").replace("\n", "\\N"))


# --- styles -------------------------------------------------------------------------------

def _family_for(style: TextStyle) -> str:
    """The family libass should ask for: the one drawtext would have loaded, so
    a machine without the preset's face falls back the same way in both engines."""
    try:
        from utils.fonts import resolve_font_file, scan_fonts
        resolved = resolve_font_file(style.font_family, bold=style.bold, italic=style.italic)
        if resolved:
            for entry in scan_fonts():
                if resolved in (entry.get("regular"), entry.get("bold"),
                                entry.get("italic"), entry.get("bold_italic")):
                    return entry["family"]
    except Exception:
        pass
    return style.font_family or "Arial"


def style_line(name: str, style: TextStyle) -> str:
    """One `Style:` line. Positions come from \\pos on each event, so margins are 0."""
    primary = ass_colour(style.color, style.opacity)
    secondary = ass_colour(style.highlight_color, style.opacity)
    if style.box:
        outline = ass_colour(style.box_color)
        back = outline
        border_style = 3
        outline_px = max(0, int(style.box_padding))
        shadow_px = 0
    else:
        outline = ass_colour(style.stroke_color) if style.stroke_width > 0 else ass_colour("black", 0.0)
        back = ass_colour(style.shadow_color)
        border_style = 1
        outline_px = max(0, int(style.stroke_width))
        shadow_px = max(abs(int(style.shadow_x)), abs(int(style.shadow_y)))
    alignment = {"left": 4, "right": 6}.get(style.align, 5)
    return ("Style: {name},{font},{size},{primary},{secondary},{outline},{back},"
            "{bold},{italic},0,0,100,100,0,0,{border},{outline_px},{shadow_px},"
            "{align},0,0,0,1").format(
        name=name, font=_family_for(style), size=max(4, int(style.font_size)),
        primary=primary, secondary=secondary, outline=outline, back=back,
        bold=-1 if style.bold else 0, italic=-1 if style.italic else 0,
        border=border_style, outline_px=outline_px, shadow_px=shadow_px,
        align=alignment)


def _style_key(style: TextStyle) -> str:
    fields = style.model_dump()
    fields.pop("animation", None)
    fields.pop("animation_duration", None)
    fields.pop("pos_x", None)
    fields.pop("pos_y", None)
    fields.pop("line_spacing", None)
    digest = hashlib.sha1(repr(sorted(fields.items())).encode("utf-8")).hexdigest()[:8]
    return f"S{digest}"


# --- events ---------------------------------------------------------------------------------

def _position(style: TextStyle, width: int, height: int) -> Tuple[float, float]:
    px = max(-1.0, min(1.0, style.pos_x))
    py = max(-1.0, min(1.0, style.pos_y))
    return (px + 1.0) / 2.0 * width, (py + 1.0) / 2.0 * height


def _clusters(text: str) -> List[str]:
    """Grapheme-ish clusters: a base with its combining marks, and a Devanagari
    consonant joined to the one before it through a virama, so a typewriter
    never reveals half a conjunct."""
    out: List[str] = []
    for char in text:
        if out and (unicodedata.category(char).startswith("M") or out[-1].endswith("्")):
            out[-1] += char
        else:
            out.append(char)
    return out


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def _fade(duration_s: float, span_s: float) -> str:
    d = _ms(min(duration_s, span_s / 2.0))
    return f"\\fad({d},{d})"


def _typewriter(text: str, duration_s: float) -> str:
    """Each cluster appears in turn across `duration_s`: alpha-in per span, so
    outline and shadow arrive with the glyph rather than before it."""
    clusters = _clusters(text)
    visible = [c for c in clusters if c.strip()]
    if not visible:
        return escape_text(text)
    step = max(15.0, (duration_s * 1000.0) / max(1, len(visible)))
    parts: List[str] = []
    shown = 0
    for cluster in clusters:
        if cluster == "\n":
            parts.append("\\N")
            continue
        if not cluster.strip():
            parts.append(escape_text(cluster))
            continue
        at = int(shown * step)
        parts.append(f"{{\\alpha&HFF&\\t({at},{at + 30},\\alpha&H00&)}}{escape_text(cluster)}")
        shown += 1
    return "".join(parts)


def _karaoke(clip: TextClip, style: TextStyle) -> str:
    """The spoken word takes the highlight colour and swells a little; the
    words before it return to the base colour."""
    base = ass_rgb(style.color)
    hot = ass_rgb(style.highlight_color)
    parts: List[str] = []
    for word in clip.words:
        text = str(word.get("text", ""))
        if not text:
            continue
        start = _ms(float(word.get("start_s", 0.0)))
        end = _ms(float(word.get("end_s", start / 1000.0 + 0.2)))
        blend = 50
        # An emphasised word (a number, a shouted word) rests larger and in the
        # accent colour, and swells further when spoken.
        emphasis = bool(word.get("emphasis"))
        rest_scale = 118 if emphasis else 100
        hot_scale = 130 if emphasis else 108
        rest_colour = hot if emphasis else base
        # An animated value carries into the spans after it, so every word
        # states its own resting colour and size before its transforms.
        tags = (f"\\c{rest_colour}\\fscx{rest_scale}\\fscy{rest_scale}"
                f"\\t({start},{start + blend},\\c{hot}\\fscx{hot_scale}\\fscy{hot_scale})"
                f"\\t({end},{end + blend},\\c{rest_colour}\\fscx{rest_scale}\\fscy{rest_scale})")
        parts.append(f"{{{tags}}}{escape_text(text)}")
    body = " ".join(parts) if parts else escape_text(clip.content)
    return body + _second_line(clip, style)


def _second_line(clip: TextClip, style: TextStyle) -> str:
    """The translation line: smaller, plainer, under the caption."""
    text = (clip.second_line or "").strip()
    if not text:
        return ""
    size = max(12, int(style.font_size * 0.55))
    return (f"\\N{{\\r\\fs{size}\\b0\\c&HE6E6E6&\\bord{max(1, style.stroke_width // 2)}"
            f"\\fscx100\\fscy100}}{escape_text(text)}")


def _glitch_lines(base_tags: str, text: str, start: float, end: float, style_name: str,
                  x: float, y: float, rng: random.Random) -> List[str]:
    """Two coloured copies pushed a few pixels either way for a few frames,
    at the clip's start and once more a moment later."""
    lines: List[str] = []
    bursts = [(0.0, 0.12), (0.42, 0.5)]
    for offset, until in bursts:
        if start + offset >= end:
            continue
        a, b = start + offset, min(end, start + until)
        dx = rng.choice([5, 6, 7, 8])
        red = f"{{\\pos({x - dx:.0f},{y + 1:.0f})\\c&H2020FF&\\alpha&H70&\\blur1}}"
        cyan = f"{{\\pos({x + dx:.0f},{y - 1:.0f})\\c&HFFE020&\\alpha&H70&\\blur1}}"
        lines.append(f"Dialogue: 1,{ass_time(a)},{ass_time(b)},{style_name},,0,0,0,,{red}{text}")
        lines.append(f"Dialogue: 1,{ass_time(a)},{ass_time(b)},{style_name},,0,0,0,,{cyan}{text}")
    return lines


def _segmented(start: float, end: float, seconds: float, step_s: float,
               make_tags, style_name: str, text: str) -> List[str]:
    """Many short Dialogue lines whose tags change per segment (shake, flicker)."""
    lines: List[str] = []
    at = start
    limit = min(end, start + seconds)
    index = 0
    while at < limit - 1e-6:
        nxt = min(limit, at + step_s)
        lines.append(f"Dialogue: 0,{ass_time(at)},{ass_time(nxt)},{style_name},,0,0,0,,"
                     f"{{{make_tags(index, at - start)}}}{text}")
        at = nxt
        index += 1
    return lines


def build_events(clip: TextClip, start: float, end: float, style_name: str,
                 width: int, height: int, seed: int = 0) -> List[str]:
    """All Dialogue lines for one clip."""
    style = clip.style or TextStyle()
    animation = (style.animation or "none").lower()
    span = max(0.1, end - start)
    duration = max(0.05, min(style.animation_duration, span / 2.0))
    x, y = _position(style, width, height)
    pos = f"\\pos({x:.0f},{y:.0f})"
    rng = random.Random(f"{seed}:{clip.content}:{start:.3f}")
    text = escape_text(clip.content)
    lines: List[str] = []

    def dialogue(tags: str, body: str, layer: int = 0, a: float = start, b: float = end) -> str:
        return f"Dialogue: {layer},{ass_time(a)},{ass_time(b)},{style_name},,0,0,0,,{{{tags}}}{body}"

    if animation == "typewriter":
        body = _typewriter(clip.content, duration if style.animation_duration > 0.3 else 0.9)
        out_fade = _ms(min(0.25, span / 4.0))
        lines.append(dialogue(f"{pos}\\fad(0,{out_fade})", body))
    elif animation == "karaoke":
        lines.append(dialogue(f"{pos}\\fad(60,60)", _karaoke(clip, style)))
    elif animation == "scale_in":
        d = _ms(duration)
        lines.append(dialogue(
            f"{pos}\\fscx135\\fscy135\\t(0,{d},\\fscx100\\fscy100){_fade(min(0.15, duration), span)}",
            text))
    elif animation == "blur_in":
        d = _ms(duration)
        lines.append(dialogue(
            f"{pos}\\blur14\\alpha&H80&\\t(0,{d},\\blur0\\alpha&H00&)\\fad(0,{_ms(min(0.4, span / 3.0))})",
            text))
    elif animation == "glitch":
        lines.append(dialogue(f"{pos}\\fad(30,80)", text))
        lines.extend(_glitch_lines("", text, start, end, style_name, x, y, rng))
    elif animation == "shake":
        amplitude = max(3.0, style.font_size * 0.09)

        def shake_tags(index: int, elapsed: float) -> str:
            decay = max(0.15, 1.0 - elapsed / max(0.1, duration))
            dx = rng.uniform(-amplitude, amplitude) * decay
            dy = rng.uniform(-amplitude, amplitude) * decay
            return f"\\pos({x + dx:.0f},{y + dy:.0f})"

        lines.extend(_segmented(start, end, duration, 2.0 / 30.0, shake_tags, style_name, text))
        if start + duration < end:
            lines.append(dialogue(f"{pos}\\fad(0,{_ms(min(0.2, span / 4.0))})", text,
                                  a=start + duration))
    elif animation == "flicker":
        def flicker_tags(index: int, elapsed: float) -> str:
            level = rng.choice([0x00, 0x00, 0x10, 0x30, 0x60, 0x20, 0x90, 0x00])
            return f"{pos}\\alpha&H{level:02X}&"

        lines.extend(_segmented(start, end, span, rng.uniform(0.06, 0.16), flicker_tags,
                                style_name, text))
    else:
        lines.append(dialogue(f"{pos}{_fade(duration, span)}", text + _second_line(clip, style)))
    return lines


# --- the file ---------------------------------------------------------------------------------

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""

EVENTS_HEADER = """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(entries: Sequence[Tuple[TextClip, float, float]], width: int, height: int,
              seed: int = 0) -> str:
    """The whole script for a list of (clip, start_sec, end_sec)."""
    styles: Dict[str, str] = {}
    events: List[Tuple[float, str]] = []
    for clip, start, end in entries:
        if not clip or not (clip.content or "").strip() or end <= start:
            continue
        style = clip.style or TextStyle()
        name = _style_key(style)
        if name not in styles:
            styles[name] = style_line(name, style)
        for line in build_events(clip, start, end, name, width, height, seed):
            events.append((start, line))
    events.sort(key=lambda e: e[0])
    body = HEADER.format(width=width, height=height)
    body += "\n".join(styles.values()) + "\n"
    body += EVENTS_HEADER
    body += "\n".join(line for _, line in events) + "\n"
    return body


def write_ass_asset(script: str, assets_dir: Path) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(script.encode("utf-8")).hexdigest()[:16]
    path = assets_dir / f"text_{digest}.ass"
    if not path.exists():
        path.write_text(script, encoding="utf-8")
    return path


def fonts_dir() -> Optional[str]:
    """An extra font directory for libass: the app's bundled fonts, if any.

    Never the system folder. libass already finds system fonts through
    DirectWrite, and pointing `fontsdir` at C:\\Windows\\Fonts makes it open
    every file there — including the bitmap `.fon` files — and log two warning
    lines for each one. Thousands of lines of stderr filled the render pipe
    and FFmpeg sat blocked at frame 0 for half an hour.
    """
    try:
        bundled = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
        if bundled.exists() and any(bundled.iterdir()):
            return str(bundled)
    except Exception:
        pass
    return None


def build_ass_filter(path: Path) -> str:
    from render.effects import escape_filter_path
    parts = [f"filename='{escape_filter_path(path)}'"]
    extra = fonts_dir()
    if extra:
        parts.append(f"fontsdir='{escape_filter_path(extra)}'")
    return "ass=" + ":".join(parts)
