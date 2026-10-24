"""Emoji reactions: a colour emoji that pops in beside the speaker's head on
a key spoken word — money, growth, decline, a hard "no", a warning, surprise,
a percentage — the infotainment habit of reacting to your own line.

Three steps, like the other dressing passes:

* `plan_reactions` matches the programme's words (Devanagari, romanised Hindi
  or English) against `EMOJI_VOCAB` plus a percent rule, ranks the matches by
  how hard they were said (`ProgramWord.emphasis_z`) and by what they carry
  (a percentage or a big money figure outranks a plain word), then keeps the
  best ones that fit: `MIN_GAP_S` apart, at most `6 * emoji_density` in any
  minute, clear of cards/newspapers/text effects/overlays, never in the
  caption band, and only as a small corner badge while a B-roll cutaway hides
  the speaker.
* `render_emoji` draws each emoji once with Pillow from Segoe UI Emoji (in
  colour, `embedded_color=True`) into a transparent PNG with a soft shadow,
  cached under the project's assets like every other drawn graphic.
* `apply_reactions` puts each reaction on one fresh V track as a few short
  items sharing that PNG: the scale steps of the pop (with a small
  overshoot), a hold, and two opacity steps of the fade. The timeline's own
  scale animation (`Transform.scale_end`) cannot be used for this: it
  promotes an overlay to a full canvas padded with black (render/effects.py).

`settings.emoji_reactions` is "auto" (on for infotainment-type genres only,
`AUTO_GENRES`), "on" (any genre) or "off".
"""

import logging
import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from timeline import clip_ops
from timeline.authoring import clear_generated
from timeline.presets import CAPTION_POS_Y, CAPTION_PRESETS
from timeline.schema import SourceFile, Timeline, Transform, frame_to_time, time_to_frame

from . import genre as genre_mod
from .models import PresentationSettings, Program
from .verify import FACE_HALF_H, FACE_HALF_W

logger = logging.getLogger("presentation.emoji_fx")

EMOJI_ORIGIN = "emoji_fx"
EMOJI_FONT = r"C:\Windows\Fonts\seguiemj.ttf"

MIN_GAP_S = 3.0                 # from one reaction's end to the next one's start
PER_MINUTE_AT_DENSITY_1 = 6.0
BUSY_MARGIN_S = 0.3
SIZE = 0.18                     # emoji height as a fraction of the canvas height
CORNER_SIZE = 0.11              # smaller, in the corner, over a B-roll cutaway
WEAK_MIN_EMPHASIS = 0.5         # generic words react only when said hard
CAPTION_HALF_BAND = 0.2         # a two-line caption block, half its height in -1..1 units

# The pop, baked as steps (see the module docstring): (scale factor, seconds).
POP_STEPS: Tuple[Tuple[float, float], ...] = ((0.3, 0.06), (0.7, 0.06), (1.12, 0.07), (1.04, 0.06))
HOLD_S = 1.2
FADE_STEPS: Tuple[Tuple[float, float], ...] = ((0.66, 0.1), (0.33, 0.1))
REACTION_S = sum(d for _, d in POP_STEPS) + HOLD_S + sum(d for _, d in FADE_STEPS)

# The PNG: the glyph fills GLYPH_FRAC of a square canvas, the rest is room for
# the shadow; the canvas is drawn big enough for the overshoot to stay crisp.
GLYPH_FRAC = 0.8
MIN_RENDER_PX = 256

# Genres "auto" turns reactions on for: explainers, infotainment, comedy,
# motivation. Horror, true crime, mystery, devotional, documentary, news and
# geopolitics stay off unless the setting says "on".
AUTO_GENRES = frozenset({
    "general", "comedy", "gaming", "tech", "science_education", "finance",
    "motivational", "health_fitness", "cooking", "travel", "vlog",
})

# category -> emoji, its strong words (react whenever chosen) and weak words
# (everyday words that react only when stressed, WEAK_MIN_EMPHASIS). A term
# with a space is a phrase matched across consecutive words. Hindi in both
# Devanagari and romanised spellings, plus English. Grow it here.
EMOJI_VOCAB: Dict[str, Dict[str, Any]] = {
    "money": {"emoji": "\U0001F4B0", "words": (
        "paisa", "paise", "paisaa", "rupaye", "rupaiye", "rupay", "rupee", "rupees", "₹",
        "dollar", "dollars", "crore", "crores", "karod", "karor", "lakh", "lakhs",
        "million", "millions", "billion", "billions", "money", "kamai",
        "पैसा", "पैसे", "रुपये", "रुपए", "रुपया", "डॉलर", "करोड़", "लाख", "मिलियन", "बिलियन", "कमाई",
    ), "weak": ()},
    "growth": {"emoji": "\U0001F4C8", "words": (
        "badha", "badhi", "badhe", "badhta", "badhti", "badhega", "badhkar", "increase",
        "increased", "increases", "grow", "grows", "growth", "growing", "double", "doubled",
        "बढ़ा", "बढ़ी", "बढ़े", "बढ़ता", "बढ़ती", "बढ़ेगा", "डबल",
    ), "weak": ("zyada", "jyada", "zyaada", "ज़्यादा", "ज्यादा")},
    "decline": {"emoji": "\U0001F4C9", "words": (
        "gira", "giri", "gire", "girta", "ghata", "ghati", "decrease", "decreased", "drop",
        "dropped", "loss", "losses", "nuksaan", "nuksan", "crash", "crashed",
        "गिरा", "गिरी", "गिरे", "गिरता", "घटा", "घटी", "नुकसान",
    ), "weak": ("kam", "कम")},
    "negation": {"emoji": "\u274C", "words": (
        "never", "kabhi nahi", "kabhi nahin", "कभी नहीं", "कभी नही",
    ), "weak": ("nahi", "nahin", "no", "mat", "नहीं", "नही", "मत")},
    "warning": {"emoji": "\u26A0\uFE0F", "words": (
        "khatra", "khatre", "khatarnak", "khatarnaak", "danger", "dangerous", "warning",
        "savdhan", "saavdhan", "alert", "खतरा", "खतरे", "खतरनाक", "सावधान",
    ), "weak": ()},
    "surprise": {"emoji": "\U0001F92F", "words": (
        "shocking", "shocked", "hairaan", "hairan", "heraan", "unbelievable", "incredible",
        "insane", "हैरान",
    ), "weak": ("kya", "क्या")},
    "fire": {"emoji": "\U0001F525", "words": (
        "best", "zabardast", "jabardast", "amazing", "awesome", "kamaal", "kamal",
        "ज़बरदस्त", "जबरदस्त", "कमाल",
    ), "weak": ()},
    "brain": {"emoji": "\U0001F9E0", "words": (
        "dimaag", "dimag", "brain", "socho", "sochiye", "दिमाग", "सोचो", "सोचिए",
    ), "weak": ("think",)},
    "heart": {"emoji": "\u2764\uFE0F", "words": (
        "pyaar", "pyar", "love", "dil", "प्यार", "दिल",
    ), "weak": ()},
    "time": {"emoji": "\u23F0", "words": (), "weak": (
        "time", "samay", "ghanta", "ghante", "minute", "minutes", "second", "seconds",
        "समय", "घंटा", "घंटे", "मिनट", "सेकंड",
    )},
    "question": {"emoji": "\u2753", "words": (), "weak": (
        "kyun", "kyon", "why", "kaise", "how", "क्यों", "कैसे",
    )},
    "idea": {"emoji": "\U0001F4A1", "words": (
        "idea", "ideas", "trick", "tricks", "tip", "tips", "आइडिया", "ट्रिक", "टिप",
    ), "weak": ()},
}
PERCENT_EMOJI = "\U0001F4CA"
HUNDRED_EMOJI = "\U0001F4AF"
_PERCENT_WORDS = frozenset({"percent", "pratishat", "parsent", "prsent", "fisadi",
                            "प्रतिशत", "फीसदी", "परसेंट"})
_BIG_UNITS = frozenset({"crore", "crores", "karod", "karor", "lakh", "lakhs", "million",
                        "millions", "billion", "billions", "करोड़", "लाख", "मिलियन", "बिलियन"})
# Ranked like entities._rank_number: a percentage 3, a figure with a big unit 2.
_ENTITY_BONUS = {3: 0.6, 2: 0.4}
_STRONG_BONUS = 0.2


@dataclass
class Reaction:
    start_s: float
    end_s: float
    emoji: str
    word: str
    x: float                    # overlay centre, -1..1 across the canvas
    y: float
    size: float                 # emoji height as a fraction of the canvas height


# --- matching -----------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case, NFC, nukta folded (ज़ == ज), punctuation and symbols
    dropped — except ₹ and %, which carry meaning here. Devanagari vowel
    signs survive (str.isalnum would strip them)."""
    text = unicodedata.normalize("NFC", (text or "").lower()).replace("\u093C", "")
    return "".join(ch for ch in text
                   if ch in "₹%" or unicodedata.category(ch)[0] in "LMN")


def _lookup() -> Tuple[Dict[str, Tuple[str, bool]], Dict[Tuple[str, ...], Tuple[str, bool]]]:
    words: Dict[str, Tuple[str, bool]] = {}
    phrases: Dict[Tuple[str, ...], Tuple[str, bool]] = {}
    for category, entry in EMOJI_VOCAB.items():
        for strong, terms in ((True, entry.get("words", ())), (False, entry.get("weak", ()))):
            for term in terms:
                parts = tuple(normalise(p) for p in term.split())
                parts = tuple(p for p in parts if p)
                if not parts:
                    continue
                if len(parts) == 1:
                    words.setdefault(parts[0], (category, strong))
                else:
                    phrases.setdefault(parts, (category, strong))
    return words, phrases


def _number(token: str) -> Optional[float]:
    digits = token.replace("%", "").replace(",", "")
    try:
        return float(digits)        # float() reads Devanagari digits too
    except ValueError:
        return None


def match_words(words: Sequence[Any]) -> List[Dict[str, Any]]:
    """Every word of `words` (ProgramWord-like: .text, .tl_start_s, .emphasis_z)
    that earns an emoji: [{index, emoji, category, strong, rank, word}]."""
    vocab, phrases = _lookup()
    tokens = [normalise(getattr(w, "text", "")) for w in words]
    out: List[Dict[str, Any]] = []
    consumed = -1                   # last index a matched phrase covered
    for i, token in enumerate(tokens):
        if not token or i <= consumed:
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        prev = tokens[i - 1] if i > 0 else ""
        value = _number(token)
        if value is not None and (token.endswith("%") or nxt in _PERCENT_WORDS):
            out.append({"index": i, "emoji": HUNDRED_EMOJI if abs(value - 100.0) < 1e-9 else PERCENT_EMOJI,
                        "category": "percent", "strong": True, "rank": 3,
                        "word": getattr(words[i], "text", token)})
            continue
        hit = None
        for parts, info in phrases.items():
            if tuple(tokens[i:i + len(parts)]) == parts:
                hit = info
                consumed = i + len(parts) - 1
                break
        if hit is None:
            hit = vocab.get(token) or (vocab.get("₹") if token.startswith("₹") else None)
        if hit is None:
            continue
        category, strong = hit
        rank = 2 if token in _BIG_UNITS and _number(prev) is not None else 0
        out.append({"index": i, "emoji": EMOJI_VOCAB[category]["emoji"], "category": category,
                    "strong": strong, "rank": rank, "word": getattr(words[i], "text", token)})
    return out


# --- rules --------------------------------------------------------------------

def mode_for(settings: Any) -> str:
    raw = getattr(settings, "emoji_reactions", "auto")
    if raw is True:
        return "auto"
    if raw is False or raw is None:
        return "off"
    raw = str(raw).strip().lower()
    return raw if raw in ("auto", "on", "off") else "auto"


def enabled_for(settings: Any, genre: Optional[str]) -> bool:
    mode = mode_for(settings)
    if mode == "off":
        return False
    return mode == "on" or genre_mod.normalise(genre) in AUTO_GENRES


def caption_band(settings: Any) -> Optional[Tuple[float, float]]:
    """(top, bottom) of the caption block in -1..1 canvas units, or None with
    captions off. The preset's `pos_y` is the block's centre."""
    if not getattr(settings, "captions", True):
        return None
    preset = CAPTION_PRESETS.get(getattr(settings, "caption_preset", "") or "", {})
    pos_y = float((preset.get("style") or {}).get("pos_y", CAPTION_POS_Y))
    return pos_y - CAPTION_HALF_BAND, pos_y + CAPTION_HALF_BAND



def _overlaps(start: float, end: float, windows: Sequence[Tuple[float, float]],
              margin: float = BUSY_MARGIN_S) -> bool:
    return any(start < e + margin and s - margin < end for s, e in windows)


def _rate_ok(starts: Sequence[float], t: float, density: float) -> bool:
    """At most 6 * density reactions in any rolling minute (below one a
    minute: one per 60 / rate seconds)."""
    rate = PER_MINUTE_AT_DENSITY_1 * max(0.0, density)
    if rate <= 0:
        return False
    if rate < 1.0:
        window, limit = 60.0 / rate, 1
    else:
        window, limit = 60.0, int(math.floor(rate + 1e-9))
    all_starts = sorted(list(starts) + [t])
    for s in all_starts:
        if sum(1 for x in all_starts if s <= x < s + window) > limit:
            return False
    return True


def _place_beside(face: Tuple[float, float], size: float, aspect: float,
                  band: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Centre (x, y) beside the head, on the side with more room, out of the
    caption band; None when nothing fits."""
    fx, fy = face
    hh = size                       # half-height in -1..1 units (H/2 per unit)
    hw = size / aspect              # half-width (aspect = W / H)
    face_hw, face_hh = FACE_HALF_W * 2, FACE_HALF_H * 2
    margin = 0.04
    y = fy - 0.12
    y = max(-1.0 + hh + margin, y)
    if band is not None and y + hh > band[0] - 0.02:
        y = band[0] - 0.02 - hh
    if y - hh < -1.0 + 0.01:
        return None
    limit = 1.0 - hw - margin
    first = -1.0 if fx >= 0.0 else 1.0
    for side in (first, -first):
        x = fx + side * (face_hw + hw + margin)
        if abs(x) > limit:
            continue
        return round(x, 4), round(y, 4)
    # No room either side: above the head, if the frame has it.
    y_above = fy - face_hh - hh - margin
    if y_above - hh >= -1.0 + 0.01:
        return round(max(-limit, min(limit, fx)), 4), round(y_above, 4)
    return None


def _corner(size: float, aspect: float) -> Tuple[float, float]:
    """Top-right corner, inside a small margin."""
    return round(1.0 - size / aspect - 0.05, 4), round(-1.0 + size + 0.08, 4)


def plan_reactions(program: Program, settings: PresentationSettings,
                   busy_windows: Sequence[Tuple[float, float]],
                   face_anchors: Dict[str, Tuple[float, float]],
                   genre: Optional[str], canvas: Tuple[int, int],
                   broll_windows: Sequence[Tuple[float, float]] = (),
                   seed: int = 0) -> List[Reaction]:
    """The reactions to place, sorted by time. `busy_windows` are never
    touched; during `broll_windows` (the speaker hidden) a reaction may only
    sit in the corner, smaller."""
    if not enabled_for(settings, genre) or not program.words:
        return []
    if str(getattr(settings, "emoji_style", "pop") or "pop") != "pop":
        return []
    density = float(getattr(settings, "emoji_density", 1.0) or 0.0)
    if density <= 0:
        return []
    width, height = canvas
    aspect = (width or 1920) / float(height or 1080)
    band = caption_band(settings)
    words = sorted(program.words, key=lambda w: w.tl_start_s)

    rng = random.Random(f"emoji:{seed}")
    candidates = []
    for m in match_words(words):
        word = words[m["index"]]
        if not m["strong"] and word.emphasis_z < WEAK_MIN_EMPHASIS:
            continue
        priority = (word.emphasis_z + _ENTITY_BONUS.get(m["rank"], 0.0)
                    + (_STRONG_BONUS if m["strong"] else 0.0))
        candidates.append((priority, rng.random(), word.tl_start_s, m))
    candidates.sort(key=lambda c: (-c[0], c[1]))

    chosen: List[Reaction] = []
    for _priority, _tie, at_s, m in candidates:
        start = max(0.0, at_s - 0.05)
        end = min(program.duration_s, start + REACTION_S)
        if end - start < 0.8:
            continue
        if _overlaps(start, end, busy_windows):
            continue
        if any(start < r.end_s + MIN_GAP_S and r.start_s - MIN_GAP_S < end for r in chosen):
            continue
        if not _rate_ok([r.start_s for r in chosen], start, density):
            continue
        if _overlaps(start, end, broll_windows, margin=0.0):
            size = CORNER_SIZE
            pos = _corner(size, aspect)
        else:
            size = SIZE
            segment = program.segment_at(at_s)
            face = (face_anchors.get(segment.item_id) if segment else None) or (0.0, 0.0)
            pos = _place_beside(face, size, aspect, band)
            if pos is None:
                continue
        chosen.append(Reaction(start, end, m["emoji"], m["word"], pos[0], pos[1], size))
    chosen.sort(key=lambda r: r.start_s)
    return chosen


def busy_windows_of(timeline: Timeline) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """(hard, broll) windows from what is already on the timeline: cards,
    text effects, overlay graphics, pop-ups and newspaper/case-file cutaways
    are hard; every other B-roll cutaway is a corner-only window."""
    from .cards import CARD_ORIGIN
    from .graphics import OVERLAY_ORIGIN
    from .placement import BROLL_ORIGIN, POPUP_ORIGIN
    from .text_fx import TEXT_FX_ORIGIN
    hard_origins = {CARD_ORIGIN, OVERLAY_ORIGIN, POPUP_ORIGIN, TEXT_FX_ORIGIN}
    hard: List[Tuple[float, float]] = []
    broll: List[Tuple[float, float]] = []
    fn, fd = timeline.fps_num, timeline.fps_den
    for item in timeline.items:
        window = (frame_to_time(item.timeline_start_frame, fn, fd),
                  frame_to_time(item.timeline_end_frame, fn, fd))
        if item.origin in hard_origins:
            hard.append(window)
        elif item.origin == BROLL_ORIGIN:
            src = timeline.sources.get(item.source_id or "")
            parts = Path(src.path).parts if src else ()
            (hard if ("newspaper" in parts or "dossier" in parts) else broll).append(window)
    return hard, broll


# --- rendering ------------------------------------------------------------------

def render_px_for(size: float, canvas_h: int) -> int:
    """PNG side for an emoji shown at `size` of the canvas height: big enough
    for the overshoot step, in 64 px steps so sizes share a cache entry."""
    needed = size * canvas_h / GLYPH_FRAC * max(s for s, _ in POP_STEPS)
    return max(MIN_RENDER_PX, int(math.ceil(needed / 64.0)) * 64)


def render_emoji(char: str, size_px: int, cache_dir: Path) -> Path:
    """A transparent `size_px` square PNG of `char` in colour with a soft drop
    shadow, drawn once and cached by codepoints + size."""
    cache_dir = Path(cache_dir)
    codepoints = "-".join(f"{ord(c):x}" for c in char)
    path = cache_dir / f"emoji_{codepoints}_{int(size_px)}.png"
    if path.exists() and path.stat().st_size > 0:
        return path
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    glyph = char.replace("\uFE0F", "")   # no raqm: the selector would draw as a blank advance
    target = int(round(size_px * GLYPH_FRAC))
    font_px = target
    font = ImageFont.truetype(EMOJI_FONT, font_px)
    scratch = Image.new("RGBA", (font_px * 3, font_px * 3), (0, 0, 0, 0))
    ImageDraw.Draw(scratch).text((font_px * 1.5, font_px * 1.5), glyph, font=font,
                                 embedded_color=True, anchor="mm")
    box = scratch.getbbox()
    if box is None:
        raise RuntimeError(f"Emoji font drew nothing for {char!r}")
    art = scratch.crop(box)
    fit = target / float(max(art.width, art.height))
    art = art.resize((max(1, int(round(art.width * fit))), max(1, int(round(art.height * fit)))),
                     Image.LANCZOS)

    canvas = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    x = (size_px - art.width) // 2
    y = (size_px - art.height) // 2
    blur = max(2, size_px // 40)
    offset = max(2, size_px // 50)
    shadow = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    alpha = art.getchannel("A").point(lambda a: int(a * 0.55))
    shadow.paste((0, 0, 0, 255), (x + offset, y + offset), alpha)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(blur)))
    canvas.alpha_composite(art, (x, y))
    cache_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# --- placing ---------------------------------------------------------------------

def _register(timeline: Timeline, path: Path, size_px: int) -> str:
    source_id = f"src_emoji_{path.stem}"
    if source_id not in timeline.sources:
        timeline.sources[source_id] = SourceFile(
            id=source_id, path=str(path), duration_seconds=max(10.0, REACTION_S),
            width=size_px, height=size_px, fps_num=timeline.fps_num,
            fps_den=timeline.fps_den, has_audio=False, kind="image")
    return source_id


def _steps(r: Reaction) -> List[Tuple[float, float, float, float]]:
    """(start_s, end_s, scale_factor, opacity) for one reaction, cut short at
    its own end."""
    out = []
    t = r.start_s
    for factor, dur in POP_STEPS:
        out.append((t, t + dur, factor, 1.0))
        t += dur
    out.append((t, t + HOLD_S, 1.0, 1.0))
    t += HOLD_S
    for opacity, dur in FADE_STEPS:
        out.append((t, t + dur, 1.0, opacity))
        t += dur
    return [(s, min(e, r.end_s), f, o) for s, e, f, o in out if s < r.end_s]


def apply_reactions(timeline: Timeline, reactions: Sequence[Reaction], project_dir: Path) -> int:
    """Place every reaction (origin `emoji_fx`, cleared first); returns how
    many landed."""
    clear_generated(timeline, EMOJI_ORIGIN)
    if not reactions:
        return 0
    width, height = timeline.width or 1920, timeline.height or 1080
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    cache_dir = Path(project_dir) / "assets" / "graphics"
    track = clip_ops.next_track(timeline, "V")
    placed = 0
    for r in reactions:
        size_px = render_px_for(r.size, height)
        try:
            path = render_emoji(r.emoji, size_px, cache_dir)
        except Exception as e:
            logger.warning("Emoji %r could not be drawn: %s", r.emoji, e)
            continue
        source_id = _register(timeline, path, size_px)
        # A square PNG shows at scale * min(W, H) px; the glyph is GLYPH_FRAC of it.
        base_scale = r.size * height / GLYPH_FRAC / float(min(width, height))
        landed = False
        for start_s, end_s, factor, opacity in _steps(r):
            first = time_to_frame(start_s, fps_num, fps_den)
            frames = time_to_frame(end_s, fps_num, fps_den) - first
            if frames <= 0:
                continue
            item = clip_ops.add_media_item(timeline, source_id, track, first, 0, frames,
                                           origin=EMOJI_ORIGIN)
            item.label = f"emoji {r.emoji} {r.word[:24]}"
            item.transform = Transform(scale=round(base_scale * factor, 5), pos_x=r.x, pos_y=r.y,
                                       opacity=opacity)
            landed = True
        placed += int(landed)
    if placed:
        timeline.recalculate_duration()
        timeline.revision += 1
    logger.info("Emoji reactions: %d placed on %s", placed, track)
    return placed


def plan_and_apply(timeline: Timeline, program: Program, settings: PresentationSettings,
                   genre: Optional[str], face_anchors: Dict[str, Tuple[float, float]],
                   project_dir: Path, seed: int = 0) -> int:
    """The director's one call: clear the last run, read the busy windows off
    the timeline, plan, place."""
    clear_generated(timeline, EMOJI_ORIGIN)
    hard, broll = busy_windows_of(timeline)
    reactions = plan_reactions(program, settings, hard, face_anchors or {}, genre,
                               (timeline.width or 1920, timeline.height or 1080),
                               broll_windows=broll, seed=seed)
    return apply_reactions(timeline, reactions, project_dir)
