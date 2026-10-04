"""Text effects: text behind the speaker's head, knockout, focus, newspaper
highlighter sweeps, kinetic keyword pop-ins, hand-drawn annotations, and
typewriter/glitch reveals — chosen per genre and scaled by
`settings.text_fx_per_minute`.

Two passes. `plan_moments` picks WHERE and WHAT: it reads the programme's
words (`ProgramWord.emphasis_z`) for keyword-driven styles, the plan's own
stat/quote/definition beats for claim-driven styles, and whichever newspaper
cutaways already landed on the timeline for the highlighter sweep, then
spaces the result at least `MIN_GAP_S` apart and clear of B-roll and cards.
`apply_moments` turns that plan into timeline items: simple text (kinetic
words, typewriter, glitch) rides the existing drawtext/ASS machinery via
`clip_ops.add_text_item`; the effects that have to reach behind or around the
speaker (behind_head, knockout, focus, annotation, the newspaper sweep) go on
an adjustment clip whose `atmosphere` list is one `AtmosphereEffect` that
`render/atmosphere.py` knows how to turn into a filtergraph fragment — the
same mechanism the mood pass already uses for a windowed hit.

`behind_head` and the matte branch of `focus` need a person matte
(presentation/matte.py); when none is available they degrade — behind_head is
dropped (and reported in `text_fx_skipped`), focus falls back to a plain
spotlight around the detected face.
"""

import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from timeline import clip_ops
from timeline.presets import TEXT_ZONE_MAX_POS_Y
from timeline.authoring import clear_generated
from timeline.schema import AtmosphereEffect, Timeline, frame_to_time, time_to_frame

from . import genre as genre_mod
from . import matte as matte_mod
from .models import Beat, PresentationSettings, Program
from .placement import BROLL_ORIGIN
from .verify import FACE_HALF_H

logger = logging.getLogger("presentation.text_fx")

TEXT_FX_ORIGIN = "text_fx"
MIN_GAP_S = 6.0
BUSY_MARGIN_S = 0.3
DEFAULT_PER_MINUTE = 2.0

# Per-genre style palettes (see AGENTS.md-adjacent brief): a video's primary
# genre draws most of its effects from its own list; `settings.genre_secondary`
# (when the caller has set one) contributes its list too, at a lower weight.
#
# `focus` (blur + darken everything but the speaker) is left out of every
# automatic palette: on a real render it read as a smeared background rather
# than emphasis. It still runs when asked for by name in `text_fx_styles`.
_PALETTES: Dict[str, Tuple[str, ...]] = {
    "vlog": ("kinetic_words", "knockout", "behind_head", "annotation"),
    "motivational": ("kinetic_words", "knockout", "behind_head", "annotation"),
    "horror": ("glitch", "newspaper_highlight", "typewriter", "behind_head"),
    "mystery": ("glitch", "newspaper_highlight", "typewriter", "behind_head"),
    "true_crime": ("glitch", "newspaper_highlight", "typewriter", "behind_head"),
    "documentary": ("newspaper_highlight", "annotation", "typewriter"),
    "news": ("newspaper_highlight", "annotation", "typewriter"),
    "geopolitics": ("newspaper_highlight", "annotation", "typewriter"),
}
_GENERAL_PALETTE: Tuple[str, ...] = (
    "kinetic_words", "knockout", "behind_head", "annotation",
    "glitch", "typewriter", "newspaper_highlight",
)
# Styles drawn more often than their palette-mates when a genre uses them:
# the full-frame knockout word is the strongest beat in a talking-head edit.
_STYLE_WEIGHT_BOOST: Dict[str, int] = {"knockout": 2}
# Which candidate kind ("keyword" from emphasis / "claim" from a stat, quote or
# definition beat) each style is meant to draw its text from.
_STYLE_SOURCE: Dict[str, str] = {
    "kinetic_words": "keyword", "knockout": "keyword", "behind_head": "keyword",
    "typewriter": "keyword", "glitch": "keyword",
    "annotation": "claim", "newspaper_highlight": "newspaper",
}
ALL_STYLES: Tuple[str, ...] = tuple(_STYLE_SOURCE.keys())


def palette_for(genre: Optional[str], genre_secondary: Optional[str] = None) -> List[Tuple[str, int]]:
    """[(style, weight), ...] — the primary genre's styles at weight 3, a
    secondary genre's (when given) at weight 1, deduplicated."""
    primary = _PALETTES.get(genre_mod.normalise(genre), _GENERAL_PALETTE)
    weighted: Dict[str, int] = {s: 3 for s in primary}
    if genre_secondary:
        for s in _PALETTES.get(genre_mod.normalise(genre_secondary), ()):
            weighted[s] = max(weighted.get(s, 0), 1)
    if not weighted:
        weighted = {s: 1 for s in _GENERAL_PALETTE}
    for style, boost in _STYLE_WEIGHT_BOOST.items():
        if style in weighted:
            weighted[style] *= boost
    return sorted(weighted.items())


class Moment:
    __slots__ = ("start_s", "end_s", "style", "text", "words", "pos_x", "pos_y",
                "newspaper_box", "face_x", "face_y")

    def __init__(self, start_s: float, end_s: float, style: str, text: str,
                words: Optional[List[str]] = None, pos_x: float = 0.0, pos_y: float = 0.0,
                newspaper_box: Optional[List[float]] = None,
                face_x: float = 0.0, face_y: float = 0.0):
        self.start_s, self.end_s, self.style, self.text = start_s, end_s, style, text
        self.words = words or [text]
        self.pos_x, self.pos_y = pos_x, pos_y
        self.newspaper_box = newspaper_box
        self.face_x, self.face_y = face_x, face_y


# --- candidates -------------------------------------------------------------

def _duration_for(style: str) -> float:
    return {"knockout": 2.2, "behind_head": 2.6, "focus": 2.4, "annotation": 1.8,
           "newspaper_highlight": 0.7, "kinetic_words": 2.0, "typewriter": 2.2,
           "glitch": 1.6}.get(style, 2.0)


# Styles that blow ONE word up to fill the frame. A loud filler word ("hain",
# "phir", "actually") there reads as nonsense, so these only take a word the
# plan itself talks about (see `_context_terms`).
_BIG_WORD_STYLES = ("knockout", "behind_head")

# Function words and crutch words, English + romanized Hindi. Never a keyword.
_STOPWORDS = frozenset("""
the and but for with that this these those then than there their they them what when where which who whom
whose why how are was were been being have has had having does did doing will would shall should can could
may might must just very really actually basically literally honestly seriously okay yeah right like well
also into onto from about over under again more most some such only own same other each every because
while until your yours you our ours we its it's i'm i've don't didn't can't won't let's here thing things
something anything everything someone people kind sort lot lots much many even still though
hai hain hoon hun tha thi the thhe ho hota hoti hote hua hui huye hue raha rahi rahe rehta rehte rahta
kar karo karna karne karke karte karta karti kiya kiye kiyaa gaya gayi gaye jata jati jate jaata jaega
jayega jaenge sakta sakte sakti sakenge wala wale wali wahan yahan jahan kahan kaha kaise kaisa kaisi
aisa aise aisi waisa waise jaisa jaise jaisi kuch kuchh koi kisi kis kya kyun kyon kyunki kyuki jab tab
ab abhi phir fir aur ya yaa lekin magar par pe mein main mai mera meri mere mujhe mujhko hum ham humein
hamein hamara hamari humara humari apna apni apne aap aapka aapki aapke aapko tum tumhe tumhara tera
teri unka unki unke unko unhe unhein uska uski uske usko use isse iska iski iske isko ise isliye
woh wo yeh ye voh vo toh to bhi hi na nahi nahin mat bas matlab yaani yani haan acha achha arre yaar
wala ek do teen baat baar sab sabko saath liye liya lena dena diya dete deta hota ki ke ka ko se tak
chahiye chahta chahte chahti sochte socha sochta lagta lagti laga bahut zyada thoda thodi sirf
""".split())


def _clean_word(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _is_content_word(text: str) -> bool:
    clean = _clean_word(text)
    return len(clean) >= 4 and not clean.isdigit() and clean not in _STOPWORDS


def _context_terms(beats: Sequence[Beat]) -> set:
    """Content words the visual plan itself put on screen or described —
    card text, pop-ups, topics. A spoken word that also appears here is
    what the video is actually about at that point."""
    terms = set()
    for b in beats:
        for field in (b.text, b.subtext, b.popup_text, b.topic, b.summary):
            for tok in re.split(r"[^\w]+", field or ""):
                if _is_content_word(tok):
                    terms.add(_clean_word(tok))
    return terms


def _hero_word(text: str) -> str:
    """The one word of a claim to blow up: the `*emphasised*` one the writer
    marked, else the longest content word, else the text itself."""
    marked = re.findall(r"\*([^*]+)\*", text or "")
    if marked:
        return marked[0].strip()
    words = [w for w in re.split(r"\s+", re.sub(r"[*_]", "", text or "")) if _is_content_word(w)]
    return max(words, key=lambda w: len(_clean_word(w))) if words else (text or "")


def _keyword_candidates(program: Program, beats: Sequence[Beat] = ()) -> List[Tuple[float, float, str, List[str], bool]]:
    """(priority, time_s, text, [a few surrounding words], on_topic) sorted best
    first. Filler/function words are dropped; a word the plan also mentions
    gets a boost and is flagged `on_topic` (the only kind a full-frame
    style may use)."""
    terms = _context_terms(beats)
    words = sorted(program.words, key=lambda w: w.tl_start_s)
    out: List[Tuple[float, float, str, List[str], bool]] = []
    for i, w in enumerate(words):
        if not _is_content_word(w.text):
            continue
        on_topic = _clean_word(w.text) in terms
        group = [x.text for x in words[i:i + 4] if x.text.strip()][:4]
        out.append((w.emphasis_z + (1.0 if on_topic else 0.0), w.tl_start_s, w.text, group, on_topic))
    out.sort(key=lambda c: c[0], reverse=True)
    return out


def _claim_candidates(beats: Sequence[Beat]) -> List[Tuple[float, float, str, Beat]]:
    out = []
    for b in beats:
        if b.kind in ("stat_callout", "quote_card", "definition_card") and (b.text or "").strip():
            # +1.5: a curated claim/stat/quote beat should usually win a slot
            # over an ordinary loud word — `priority` is 0..1, the same range
            # ProgramWord.emphasis_z rarely clears on its own.
            out.append((b.priority + 1.5, b.start_s, (b.text or "").strip(), b))
    out.sort(key=lambda c: c[0], reverse=True)
    return out


def _newspaper_windows(timeline: Timeline, assets: Sequence[Any],
                       beats: Sequence[Beat]) -> List[Dict[str, Any]]:
    """Already-placed newspaper cutaways that carry a highlight box."""
    by_id = {b.id: b for b in beats}
    by_path = {getattr(a, "path", None): a for a in assets}
    out: List[Dict[str, Any]] = []
    for item in timeline.items:
        if item.kind != "media" or item.origin != BROLL_ORIGIN or not item.source_id:
            continue
        src = timeline.sources.get(item.source_id)
        if not src or "newspaper" not in Path(src.path).parts:
            continue
        asset = by_path.get(src.path)
        beat = by_id.get(getattr(asset, "beat_id", None)) if asset else None
        box = (beat.data or {}).get("highlight_box") if beat is not None else None
        if not box:
            continue
        out.append({
            "start_s": frame_to_time(item.timeline_start_frame, timeline.fps_num, timeline.fps_den),
            "end_s": frame_to_time(item.timeline_end_frame, timeline.fps_num, timeline.fps_den),
            "box": list(box),
            "text": (beat.data or {}).get("highlight") or beat.text or "",
        })
    return out


# --- face avoidance -----------------------------------------------------------

def _face_anchor(face_anchors: Dict[str, Tuple[float, float]], program: Program,
                 at_s: float) -> Tuple[float, float]:
    """The speaker's (x, y) at this moment, or frame centre with none detected."""
    segment = program.segment_at(at_s)
    anchor = face_anchors.get(segment.item_id) if segment else None
    return anchor if anchor else (0.0, 0.0)


def _dodge_face(face_x: float, face_y: float) -> Tuple[float, float]:
    """A (pos_x, pos_y) in the upper or lower band, whichever is farther from
    the speaker's face — clear of verify.py's own face box (FACE_HALF_H either
    side of the anchor) with room to spare."""
    del face_x
    band = min(0.9, 0.5 + FACE_HALF_H * 2)
    # Below the face never means into the caption band at the bottom.
    pos_y = -band if face_y >= 0.0 else min(band, TEXT_ZONE_MAX_POS_Y)
    return 0.0, pos_y


def _overlaps(a_start: float, a_end: float, windows: Sequence[Tuple[float, float]],
             margin: float = BUSY_MARGIN_S) -> bool:
    return any(a_start < e + margin and s - margin < a_end for s, e in windows)


# --- planning -----------------------------------------------------------------

def plan_moments(
    program: Program,
    beats: Sequence[Beat],
    timeline: Timeline,
    generated_assets: Sequence[Any],
    settings: PresentationSettings,
    genre: str,
    face_anchors: Dict[str, Tuple[float, float]],
    busy_windows: Sequence[Tuple[float, float]],
    card_windows: Sequence[Tuple[float, float]],
    seed: int = 0,
) -> List[Moment]:
    if not getattr(settings, "text_fx", True):
        return []
    minutes = max(0.0, program.duration_s) / 60.0
    # `effective_text_fx_per_minute` (density-aware) wins when this settings
    # object has it; otherwise fall back to the raw field, then the default —
    # the other agent's density work may or may not have landed yet.
    per_minute = getattr(settings, "effective_text_fx_per_minute", None)
    if per_minute is None:
        per_minute = getattr(settings, "text_fx_per_minute", None)
    per_minute = DEFAULT_PER_MINUTE if per_minute is None else max(0.0, float(per_minute))
    count = int(round(per_minute * minutes))
    if count <= 0:
        return []

    styles_setting = getattr(settings, "text_fx_styles", None)
    genre_secondary = getattr(settings, "genre_secondary", None)
    weighted = palette_for(genre, genre_secondary)
    if styles_setting:
        allowed = set(styles_setting)
        # Named styles run even when no automatic palette carries them (focus).
        kept = [(s, w) for s, w in weighted if s in allowed]
        weighted = kept + [(s, 1) for s in sorted(allowed - {s for s, _ in kept})]
    pool_by_source: Dict[str, List[str]] = {"keyword": [], "claim": [], "newspaper": []}
    for style, weight in weighted:
        source = _STYLE_SOURCE.get(style, "keyword")
        pool_by_source[source].extend([style] * weight)

    rng = random.Random(seed)
    busy = list(busy_windows) + list(card_windows)
    chosen: List[Moment] = []

    # Newspaper sweeps: one per already-placed newspaper cutaway with a
    # highlight box, independent of the keyword/claim budget below since they
    # ride an asset that already exists rather than competing for screen time.
    if pool_by_source["newspaper"]:
        for window in _newspaper_windows(timeline, generated_assets, beats):
            start_s = window["start_s"] + min(0.3, max(0.0, window["end_s"] - window["start_s"]) / 4)
            end_s = min(window["end_s"], start_s + _duration_for("newspaper_highlight"))
            if end_s <= start_s or len(chosen) >= count:
                continue
            if any(abs(start_s - m.start_s) < MIN_GAP_S for m in chosen if m.style == "newspaper_highlight"):
                continue
            chosen.append(Moment(start_s, end_s, "newspaper_highlight", window["text"],
                                 newspaper_box=window["box"]))

    def _spaced(start_s: float, end_s: float) -> bool:
        return not any(start_s < m.end_s + MIN_GAP_S and m.start_s - MIN_GAP_S < end_s for m in chosen)

    remaining = max(0, count - len(chosen))
    claim_candidates = _claim_candidates(beats) if pool_by_source["claim"] else []
    keyword_candidates = _keyword_candidates(program, beats) if pool_by_source["keyword"] else []
    candidates = (
        [(p, t, txt, ("claim", beat)) for p, t, txt, beat in claim_candidates]
        + [(p, t, txt, ("keyword", (words, on_topic))) for p, t, txt, words, on_topic in keyword_candidates]
    )
    candidates.sort(key=lambda c: c[0], reverse=True)

    for _priority, at_s, text, (source, payload) in candidates:
        if remaining <= 0:
            break
        pool = pool_by_source.get(source) or []
        if not pool:
            continue
        style = rng.choice(pool)
        if source == "keyword" and style in _BIG_WORD_STYLES and not payload[1]:
            # An off-topic word may still pop in small, never fill the frame.
            small = [st for st in pool if st not in _BIG_WORD_STYLES]
            if not small:
                continue
            style = rng.choice(small)
        duration = _duration_for(style)
        start_s = max(0.0, at_s - 0.15)
        end_s = min(program.duration_s, start_s + duration)
        if end_s <= start_s:
            continue
        if _overlaps(start_s, end_s, busy) or not _spaced(start_s, end_s):
            continue
        face_x, face_y = _face_anchor(face_anchors, program, at_s)
        pos_x, pos_y = _dodge_face(face_x, face_y)
        if source == "keyword":
            words = payload[0]
        elif style in _BIG_WORD_STYLES:
            words = [_hero_word(text)]
        else:
            words = [text]
        chosen.append(Moment(start_s, end_s, style, text, words=words, pos_x=pos_x, pos_y=pos_y,
                             face_x=face_x, face_y=face_y))
        remaining -= 1

    chosen.sort(key=lambda m: m.start_s)
    return chosen


# --- rendering ------------------------------------------------------------------

def _text_for(style: str, moment: "Moment") -> str:
    if style in ("knockout", "behind_head"):
        # One big word: its trailing full stop or comma would read as a glyph.
        return (moment.words[0] if moment.words else moment.text).strip(" .,!?;:\"'").upper()
    if style == "kinetic_words":
        return moment.text
    if style in ("typewriter", "glitch"):
        phrase = " ".join(moment.words[:4]) if moment.words else moment.text
        return phrase
    return moment.text[:60]


def _apply_kinetic_words(timeline: Timeline, moment: "Moment", track: str,
                         fps_num: int, fps_den: int) -> None:
    words = [w for w in (moment.words or [moment.text]) if w.strip()][:4] or [moment.text]
    span = max(0.3, moment.end_s - moment.start_s)
    step = span / max(1, len(words))
    for i, word in enumerate(words):
        start = moment.start_s + i * step
        end = min(moment.end_s, start + max(0.5, step))
        if end <= start:
            continue
        start_frame = time_to_frame(start, fps_num, fps_den)
        frames = max(1, time_to_frame(end, fps_num, fps_den) - start_frame)
        item = clip_ops.add_text_item(
            timeline, word.title(), start_frame, frames, track=track,
            style={"pos_x": moment.pos_x, "pos_y": moment.pos_y, "font_size": 90,
                  "bold": True, "animation": "pop", "animation_duration": 0.35,
                  "color": "#FFE23A", "stroke_width": 4})
        item.origin = TEXT_FX_ORIGIN
        item.label = f"text_fx kinetic_words: {word[:24]}"


def _apply_simple_text(timeline: Timeline, moment: "Moment", style: str, track: str,
                       fps_num: int, fps_den: int) -> None:
    animation = "typewriter" if style == "typewriter" else "glitch"
    start_frame = time_to_frame(moment.start_s, fps_num, fps_den)
    frames = max(1, time_to_frame(moment.end_s, fps_num, fps_den) - start_frame)
    item = clip_ops.add_text_item(
        timeline, _text_for(style, moment), start_frame, frames, track=track,
        style={"pos_x": moment.pos_x, "pos_y": moment.pos_y, "font_size": 76, "bold": True,
              "animation": animation, "animation_duration": 0.5})
    item.origin = TEXT_FX_ORIGIN
    item.label = f"text_fx {style}: {moment.text[:24]}"


def _apply_atmosphere_effect(timeline: Timeline, moment: "Moment", kind: str,
                             extra: Dict[str, Any], track: str,
                             fps_num: int, fps_den: int) -> None:
    start_frame = time_to_frame(moment.start_s, fps_num, fps_den)
    frames = max(1, time_to_frame(moment.end_s, fps_num, fps_den) - start_frame)
    item = clip_ops.add_adjustment_item(timeline, start_frame, frames, track=track)
    item.origin = TEXT_FX_ORIGIN
    item.label = f"text_fx {kind}: {moment.text[:24]}"
    extra = dict(extra)
    extra.setdefault("start_s", moment.start_s)
    extra.setdefault("end_s", moment.end_s)
    item.atmosphere = [AtmosphereEffect(type=kind, enabled=True, origin=TEXT_FX_ORIGIN, extra=extra)]


def apply_moments(
    timeline: Timeline,
    moments: Sequence[Moment],
    settings: PresentationSettings,
    project_dir: Path,
    source_video: Optional[str],
    canvas_w: int,
    canvas_h: int,
) -> Tuple[Dict[str, int], str, List[Dict[str, str]]]:
    """Places every moment; returns (counts_by_style, person_matte_used, skipped)."""
    clear_generated(timeline, TEXT_FX_ORIGIN)
    counts: Dict[str, int] = {}
    skipped: List[Dict[str, str]] = []
    if not moments:
        return counts, "off", skipped

    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    text_track = clip_ops.next_track(timeline, "T")
    fx_track = clip_ops.next_track(timeline, "V")

    matte_setting = str(getattr(settings, "person_matte", "auto") or "auto").lower()
    matte_needed = [m for m in moments if m.style in ("behind_head", "focus")]
    matte_result = None
    matte_backend = "off"
    if matte_setting != "off" and matte_needed and source_video:
        windows = [(m.start_s, m.end_s) for m in matte_needed]
        try:
            matte_result = matte_mod.compute_person_matte(
                source_video, windows, Path(project_dir), canvas_w, canvas_h)
        except Exception as e:
            logger.warning("Person matte computation failed: %s", e)
            matte_result = None
        matte_backend = matte_result.backend if matte_result else matte_mod.detect_backend()

    for moment in moments:
        style = moment.style
        if style == "kinetic_words":
            _apply_kinetic_words(timeline, moment, text_track, fps_num, fps_den)
        elif style in ("typewriter", "glitch"):
            _apply_simple_text(timeline, moment, style, text_track, fps_num, fps_den)
        elif style == "newspaper_highlight":
            _apply_atmosphere_effect(timeline, moment, "newspaper_sweep",
                                     {"box": moment.newspaper_box or [0.1, 0.1, 0.9, 0.2]},
                                     fx_track, fps_num, fps_den)
        elif style == "annotation":
            shape = random.Random(f"{moment.start_s}").choice(["underline", "circle", "arrow"])
            _apply_atmosphere_effect(
                timeline, moment, "annotation",
                {"shape": shape, "cx": (moment.pos_x + 1.0) / 2.0, "cy": (moment.pos_y + 1.0) / 2.0,
                "text": moment.text[:40]},
                fx_track, fps_num, fps_den)
        elif style == "behind_head":
            offset = matte_result.offset_for(moment.start_s, moment.end_s) if matte_result else None
            if offset is None:
                skipped.append({"style": "behind_head", "reason": "no person matte for this window"})
                continue
            _apply_atmosphere_effect(
                timeline, moment, "behind_head",
                {"text": _text_for("behind_head", moment), "matte_path": matte_result.path,
                "matte_offset": offset}, fx_track, fps_num, fps_den)
        elif style == "focus":
            offset = matte_result.offset_for(moment.start_s, moment.end_s) if matte_result else None
            if offset is not None:
                extra = {"matte_path": matte_result.path, "matte_offset": offset}
            else:
                extra = {"face_x": moment.face_x, "face_y": moment.face_y}
            _apply_atmosphere_effect(timeline, moment, "focus", extra, fx_track, fps_num, fps_den)
        elif style == "knockout":
            _apply_atmosphere_effect(
                timeline, moment, "knockout",
                {"text": _text_for("knockout", moment)}, fx_track, fps_num, fps_den)
        else:
            continue
        counts[style] = counts.get(style, 0) + 1

    return counts, matte_backend, skipped
