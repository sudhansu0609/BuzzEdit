"""Genre awareness — so a horror video gets horror pictures.

Every generated image is styled twice. First the language model is TOLD the
genre (a block appended to each prompt-writing system prompt), so it composes
scenes that belong in that kind of video. Then the genre's visual look is
APPENDED to the finished prompt anyway, because a local model told "this is a
horror video" will still happily write a sunny stock-photo prompt one time in
five — and one bright corporate hallway in the middle of a horror edit reads
as a mistake the whole video long.

The genre itself comes from the user when they set it, and from the transcript
when they do not: one cheap model call over the opening minutes, with a keyword
count as the fallback when the model is absent or answers nonsense. Detection
must never break a pass — anything unrecognised is simply "general", which
styles nothing and leaves the prompts exactly as they were.
"""

import logging
import math
import re
from typing import Dict, NamedTuple, Optional, Tuple

logger = logging.getLogger("presentation.genre")


class GenreStyle(NamedTuple):
    # Appended to every positive prompt so the pictures share the genre's look.
    look: str
    # Appended to every negative prompt: what would break the mood if it appeared.
    negative: str
    # Style phrase for the thumbnail prompt.
    thumbnail: str
    # Transcript words that vote for this genre when no model is available.
    # Includes romanised Hindi, because the transcripts are Hinglish.
    keywords: Tuple[str, ...]


# "general" is the identity genre: it styles nothing, so every prompt path can
# apply a genre unconditionally without special-casing the plain video.
GENRE_STYLES: Dict[str, GenreStyle] = {
    "general": GenreStyle("", "", "dramatic high contrast lighting", ()),
    "horror": GenreStyle(
        "dark ominous atmosphere, deep shadows, cold desaturated palette, fog, "
        "unsettling horror film mood, low-key lighting",
        "bright cheerful colors, sunny, cartoon",
        "dark horror atmosphere, eerie fog, glowing eyes in shadow, ominous "
        "red and black palette",
        ("horror", "bhoot", "bhootiya", "ghost", "haunted", "paranormal",
         "aatma", "atma", "chudail", "shaitan", "darawna", "darawni", "kabristan",
         "shamshan", "possessed", "exorcism", "demon", "pretatma", "tantrik",
         "darr", "khauf", "bhutiya", "spirit", "cursed", "creepy")),
    "true_crime": GenreStyle(
        "moody crime documentary tone, dim tungsten light, film noir shadows, "
        "muted desaturated palette, tense atmosphere",
        "bright cheerful colors, cartoon",
        "gritty crime documentary look, noir shadows, cold blue and amber tones",
        ("murder", "killer", "crime", "qatal", "khoon", "police", "victim",
         "investigation", "kidnap", "criminal", "jail", "chori", "mujrim",
         "gunah", "case", "forensic", "detective")),
    "comedy": GenreStyle(
        "bright vivid colors, playful energetic mood, warm saturated lighting, "
        "lighthearted comedic tone",
        "dark, gloomy, horror",
        "bright playful colors, exaggerated fun expression, pop-art energy",
        ("comedy", "funny", "joke", "jokes", "prank", "mazaak", "mazak",
         "hasna", "hansi", "roast", "meme", "hilarious", "chutkula")),
    "gaming": GenreStyle(
        "vibrant neon colors, dynamic video game concept art style, dramatic "
        "rim lighting, high energy",
        "dull, washed out",
        "vibrant neon gaming style, electric blues and purples, dynamic action",
        ("game", "gaming", "gameplay", "gamer", "minecraft", "gta", "pubg",
         "bgmi", "fortnite", "valorant", "esports", "noob", "level",
         "mission", "player", "stream")),
    "tech": GenreStyle(
        "clean modern tech aesthetic, sleek surfaces, cool blue accent "
        "lighting, minimalist composition",
        "cluttered, vintage, grainy",
        "sleek futuristic tech look, glowing blue circuit accents",
        ("technology", "tech", "smartphone", "iphone", "android", "laptop",
         "computer", "software", "app", "gadget", "processor", "chip",
         "artificial", "intelligence", "robot", "coding", "internet")),
    "science_education": GenreStyle(
        "clear educational illustration style, clean composition, bright even "
        "lighting, engaging documentary tone",
        "cluttered, chaotic",
        "bright curious science look, glowing diagram accents, deep blue depth",
        ("science", "experiment", "research", "study", "physics", "chemistry",
         "biology", "university", "professor", "theory", "brain", "space",
         "planet", "vigyan", "exam", "padhai", "concept", "history")),
    "finance": GenreStyle(
        "confident professional tone, city skyline and market imagery, warm "
        "golden accent lighting, crisp modern composition",
        "cartoon, messy",
        "bold wealth imagery, gold and deep green tones, rising energy",
        ("money", "paisa", "paise", "invest", "investment", "stock", "share",
         "market", "trading", "mutual", "fund", "profit", "business", "crore",
         "lakh", "salary", "savings", "wealth", "loan", "tax", "crypto")),
    "motivational": GenreStyle(
        "epic inspirational tone, golden hour light, heroic wide compositions, "
        "cinematic uplifting atmosphere",
        "gloomy, defeated",
        "epic sunrise energy, silhouetted figure on a summit, golden light",
        ("motivation", "success", "successful", "discipline", "mindset",
         "habits", "habit", "goal", "goals", "dream", "sapna", "mehnat",
         "struggle", "failure", "focus", "improve", "growth", "jeet")),
    "health_fitness": GenreStyle(
        "energetic fitness photography, crisp morning light, vibrant healthy "
        "tones, athletic dynamic composition",
        "junk food, sedentary",
        "high-energy fitness look, dynamic action, fresh vibrant tones",
        ("workout", "gym", "fitness", "exercise", "yoga", "diet", "protein",
         "muscle", "weight", "health", "sehat", "kasrat", "running", "calories",
         "body", "immunity")),
    "cooking": GenreStyle(
        "appetising food photography, warm inviting kitchen light, rich "
        "colours, shallow depth of field",
        "unappetising, messy",
        "mouth-watering food close-up, warm steam, rich golden light",
        ("recipe", "cooking", "cook", "khana", "masala", "tadka", "kitchen",
         "ingredients", "dish", "swad", "delicious", "restaurant", "chef",
         "biryani", "paneer", "roti", "sabzi")),
    "travel": GenreStyle(
        "sweeping travel photography, golden hour landscapes, vivid natural "
        "colours, sense of wonder",
        "indoor office, dull",
        "breathtaking destination vista, golden light, wanderlust energy",
        ("travel", "trip", "tour", "flight", "hotel", "beach", "mountain",
         "pahad", "safar", "yatra", "ghumne", "vacation", "visa", "passport",
         "destination", "explore")),
    "devotional": GenreStyle(
        "serene devotional atmosphere, warm diya candlelight, golden temple "
        "tones, reverent peaceful composition",
        "dark, frightening",
        "radiant divine glow, golden temple light, serene reverence",
        ("bhagwan", "mandir", "puja", "pooja", "aarti", "mantra", "shiv",
         "krishna", "ram", "hanuman", "durga", "bhakti", "prabhu", "ishwar",
         "dharma", "vrat", "prasad", "temple", "spiritual")),
    "news": GenreStyle(
        "serious journalistic tone, realistic documentary photography, neutral "
        "colour grade, clear composition",
        "cartoon, fantasy",
        "urgent newsroom look, bold red accents, documentary realism",
        ("news", "breaking", "government", "sarkar", "minister", "election",
         "parliament", "policy", "report", "andolan", "protest", "court",
         "supreme", "verdict", "economy")),
    "vlog": GenreStyle(
        "warm candid lifestyle photography, natural daylight, friendly "
        "personal tone, handheld documentary feel",
        "sterile, corporate",
        "warm personal lifestyle look, candid moment, natural light",
        ("vlog", "vlogging", "daily", "routine", "morning", "subscribe",
         "channel", "ghar", "family", "dost", "friends", "shopping")),
    "documentary": GenreStyle(
        "cinematic archival documentary tone, natural realistic light, "
        "restrained filmic colour, textured film grain, observational "
        "composition",
        "cartoon, fantasy",
        "cinematic archival documentary look, restrained filmic colour, "
        "textured film grain",
        ("documentary", "history", "itihaas", "archive", "true story", "real",
         "account", "incident", "kahani", "case")),
    "mystery": GenreStyle(
        "moody investigative mystery tone, low-key noir shadows, "
        "desaturated cold palette, pools of light, tense unresolved "
        "atmosphere",
        "bright cheerful colors, cartoon",
        "moody investigative mystery look, noir shadows, pools of light",
        ("mystery", "rahasya", "unsolved", "disappeared", "gayab", "clue",
         "suspense", "secret", "raaz", "conspiracy", "investigation",
         "evidence")),
    "geopolitics": GenreStyle(
        "serious geopolitical documentary tone, world maps and satellite "
        "imagery, muted institutional palette, cool clean realism, "
        "authoritative composition",
        "cartoon, fantasy",
        "serious geopolitical documentary look, world maps, muted "
        "institutional palette",
        ("geopolitics", "border", "seema", "war", "yudh", "nato", "china",
         "russia", "treaty", "sanction", "military", "strategy", "conflict",
         "alliance")),
}

GENRE_NAMES = tuple(GENRE_STYLES.keys())

# Fewer transcript votes than this and the keyword count is guessing, not
# detecting; the pass then styles nothing rather than styling wrongly.
MIN_KEYWORD_VOTES = 3

GENRE_SYSTEM = (
    "You classify a YouTube video by its transcript. The speaker may talk in "
    "Hindi written in Latin letters (Hinglish) with badly romanised spelling; "
    "read past the spelling.\n\n"
    "Choose the ONE genre that best fits what the video is about, from exactly "
    "this list: " + ", ".join(GENRE_NAMES) + ".\n"
    "Use 'general' when nothing fits clearly. Answer with JSON only:\n"
    '{"genre": ""}'
)

GENRE_SCHEMA = {
    "name": "genre",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"genre": {"type": "string", "enum": list(GENRE_NAMES)}},
        "required": ["genre"],
        "additionalProperties": False,
    },
}


# A subtle full-length atmosphere layer per genre — the automatic "look" of the
# video beyond colour. Deliberately sparse: most genres get gentle film grain
# (the universal documentary texture) or nothing, and every intensity is far
# below the preset defaults so the effect reads as texture, not weather. None
# means no layer at all.
_ATMOSPHERE: Dict[str, Optional[Tuple[str, float]]] = {
    "general": ("grain", 0.18),
    "horror": ("fog", 0.25),
    "true_crime": ("grain", 0.30),
    "comedy": None,
    "gaming": None,
    "tech": None,
    "science_education": ("grain", 0.15),
    "finance": None,
    "motivational": ("sunlight", 0.20),
    "health_fitness": None,
    "cooking": ("sunlight", 0.15),
    "travel": ("sunlight", 0.18),
    "devotional": ("light_leak", 0.18),
    "news": None,
    "vlog": ("grain", 0.15),
    "documentary": ("grain", 0.16),
    "mystery": ("grain", 0.24),
    "geopolitics": ("grain", 0.12),
}


def atmosphere_for(genre: Optional[str]) -> Optional[Tuple[str, float]]:
    """(effect type, intensity) for the genre's automatic atmosphere, or None."""
    return _ATMOSPHERE.get(normalise(genre))


# The programme-wide grade per genre: a named preset from
# timeline.presets.COLOR_PRESETS plus the few overrides that make it the
# genre's look rather than a generic one. Restraint on purpose — the grade
# sits under every frame, and a heavy one on talking-head footage reads as a
# camera fault, not a style.
_GRADE: Dict[str, Tuple[str, Dict[str, float]]] = {
    "general": ("punchy_talking_head", {}),
    "horror": ("moody", {"vignette": 0.5, "saturation": 0.78, "temperature": -0.32,
                         "contrast": 1.22}),
    "true_crime": ("cinematic", {"saturation": 0.86, "vignette": 0.38, "temperature": -0.18}),
    "comedy": ("vibrant", {}),
    "gaming": ("vibrant", {"contrast": 1.25}),
    "tech": ("clean", {"temperature": -0.12}),
    "science_education": ("punchy_talking_head", {}),
    "finance": ("cinematic", {"temperature": 0.1, "vignette": 0.18}),
    "motivational": ("teal_orange", {}),
    "health_fitness": ("vibrant", {"sharpen": 0.5}),
    "cooking": ("warm", {"saturation": 1.25}),
    "travel": ("teal_orange", {"saturation": 1.35}),
    "devotional": ("warm", {"vignette": 0.22}),
    "news": ("clean", {}),
    "vlog": ("warm", {"contrast": 1.05}),
    "documentary": ("cinematic", {"saturation": 0.92, "vignette": 0.14}),
    "mystery": ("moody", {"saturation": 0.8, "temperature": -0.2, "vignette": 0.42,
                          "contrast": 1.15}),
    "geopolitics": ("clean", {"temperature": -0.1, "vignette": 0.12}),
}


def grade_for(genre: Optional[str]) -> Tuple[str, Dict[str, float]]:
    """(colour preset name, overrides) for the genre's programme grade."""
    return _GRADE.get(normalise(genre), _GRADE["general"])


# The transition at a topic boundary, per genre: (xfade type, seconds). Inside
# a topic every join stays a hard cut (hidden under B-roll or disguised by the
# punch-out); these mark the chapter changes so the structure is felt. Horror
# breathes out into black; explainers whip forward; the rest dissolve.
_TOPIC_TRANSITION: Dict[str, Tuple[str, float]] = {
    "general": ("fade", 0.4),
    "horror": ("fadeblack", 0.7),
    "true_crime": ("fadeblack", 0.6),
    "comedy": ("slideleft", 0.35),
    "gaming": ("hlwind", 0.3),
    "tech": ("hlwind", 0.35),
    "science_education": ("hlwind", 0.35),
    "finance": ("zoomin", 0.4),
    "motivational": ("fade", 0.5),
    "health_fitness": ("slideleft", 0.35),
    "cooking": ("fade", 0.4),
    "travel": ("smoothleft", 0.5),
    "devotional": ("fadewhite", 0.7),
    "news": ("wipeleft", 0.3),
    "vlog": ("slideleft", 0.35),
    "documentary": ("fade", 0.5),
    "mystery": ("fadeblack", 0.6),
    "geopolitics": ("wipeleft", 0.35),
}


def topic_transition_for(genre: Optional[str]) -> Tuple[str, float]:
    return _TOPIC_TRANSITION.get(normalise(genre), _TOPIC_TRANSITION["general"])


# Auto-edit pacing per genre, (max_pause_seconds, pause_padding_seconds): a pause
# longer than the first is trimmed to the second either side of the speech. Only
# genres whose delivery lives in its silences are listed; the rest keep the
# Timeline defaults (0.40s / 0.12s), and a pacing a channel or the user chose
# always wins over these (asr.auto_edit.apply_genre_pacing).
#
# Horror is measured from the creator's own cut of Raat3Baje ep1 (2026-10-04):
# the pauses kept there have a median of 0.40s, p90 1.0s and p95 1.38s. Half of
# them ran past the default's 0.40s, and the default crushed every one of those
# to a ~0.24s beat — the dread before a reveal went with them. At 1.2s, 93% of
# the cut's pauses pass untouched; longer dead air still comes down, to a 0.9s beat.
_PACING: Dict[str, Tuple[float, float]] = {
    "horror": (1.2, 0.45),
}


def pacing_for(genre: Optional[str]) -> Optional[Tuple[float, float]]:
    """(max_pause_seconds, pause_padding_seconds) for the genre, or None when the
    Timeline defaults suit it."""
    return _PACING.get(normalise(genre))


# What the AI editor (asr/editor.py) should know about how a genre is delivered: guidance
# it weighs, not a rule that decides. Narrated stories repeat a phrase on purpose — on
# Raat3Baje ep1 most of what the editor wrongly cut was such repetition ("वहाँ नीचे... वहाँ
# नीचे") trimmed as if it were a stutter, after examples from a vlog taught it to.
_STORYTELLING = ("This is narrated storytelling. Repeating a phrase for effect (\"वहाँ नीचे... वहाँ नीचे\"), "
                 "a slow build and long dramatic pauses are part of the delivery: keep them. Trim a repeated "
                 "phrase only when its first copy is clearly broken off or stumbled.")
_EDITING_NOTES: Dict[str, str] = {
    "horror": _STORYTELLING,
    "mystery": _STORYTELLING,
    "true_crime": _STORYTELLING,
}


def editing_notes_for(genre: Optional[str]) -> str:
    """Delivery notes for the AI editor, or "" when the genre needs none."""
    return _EDITING_NOTES.get(normalise(genre), "")


# The recommended effect palette per genre — names an editor (human or model)
# can pick from when reaching for something beyond the automatic atmosphere
# and grade. Deliberately a suggestion list, not an enforced one.
_FX_PALETTE: Dict[str, Tuple[str, ...]] = {
    "general": ("grain",),
    "horror": ("flicker", "shutter", "glitch", "fog", "shake", "flash", "vhs", "strobe"),
    "true_crime": ("grain", "redaction", "spotlight", "flicker", "case_file"),
    "comedy": ("push", "zoom"),
    "gaming": ("glitch", "flash"),
    "tech": ("push", "clean"),
    "science_education": ("push", "spotlight"),
    "finance": ("push", "stat", "chart", "clean"),
    "motivational": ("push", "light_leak"),
    "health_fitness": ("push",),
    "cooking": ("push",),
    "travel": ("push", "light_leak"),
    "devotional": ("light_leak", "spotlight"),
    "news": ("grain", "push"),
    "vlog": ("push",),
    "documentary": ("grain", "light_leak", "spotlight", "push", "fade"),
    "mystery": ("grain", "flicker", "redaction", "spotlight", "shutter"),
    "geopolitics": ("grain", "spotlight", "push", "map", "newspaper"),
}


def fx_palette_for(genre: Optional[str]) -> Tuple[str, ...]:
    """The recommended effect names for the genre, or () when none apply."""
    return _FX_PALETTE.get(normalise(genre), ())


# --- secondary genre blend ---------------------------------------------------
#
# The primary genre alone drives the programme grade, atmosphere and topic
# transitions — one look has to hold the whole video together. The FX palette
# and B-roll style tags, in contrast, are lists a pass picks *from*, so a
# second genre can fold into them: the primary's own names stay in full, and
# roughly a 35/65 share of the secondary's own (non-overlapping) names are
# appended after them, so a scan of the list still reads primary-first.
_BLEND_PRIMARY_WEIGHT = 0.65
_BLEND_SECONDARY_WEIGHT = 0.35


def _blend(primary_names: Tuple[str, ...], secondary_names: Tuple[str, ...]) -> Tuple[str, ...]:
    extra = [name for name in secondary_names if name not in primary_names]
    if not extra:
        return primary_names
    ratio = _BLEND_SECONDARY_WEIGHT / _BLEND_PRIMARY_WEIGHT
    take = max(1, math.ceil(len(primary_names) * ratio)) if primary_names else len(extra)
    return tuple(list(primary_names) + extra[:take])


def blended_fx_palette(genre: Optional[str], secondary: Optional[str]) -> Tuple[str, ...]:
    """The FX palette for two genres, primary first and ~65/35 weighted.

    `secondary` is ignored when absent, "general", or the same as `genre` —
    a blend with nothing to add is just the primary's own palette.
    """
    primary = fx_palette_for(genre)
    if not secondary or normalise(secondary) in ("general", normalise(genre)):
        return primary
    return _blend(primary, fx_palette_for(secondary))


def _style_tags(genre: Optional[str]) -> Tuple[str, ...]:
    """A genre's `look` string, split into its comma-separated phrases."""
    return tuple(part.strip() for part in style_for(genre).look.split(",") if part.strip())


def blended_style_tags(genre: Optional[str], secondary: Optional[str]) -> Tuple[str, ...]:
    """B-roll style phrases for two genres, blended the same ~65/35 way as the
    FX palette — for `apply_look`, and for any other pass that wants the two
    genres' look words rather than a single genre's `look` string."""
    primary = _style_tags(genre)
    if not secondary or normalise(secondary) in ("general", normalise(genre)):
        return primary
    return _blend(primary, _style_tags(secondary))


def normalise(genre: Optional[str]) -> str:
    """A user- or model-supplied genre name, mapped onto one we style.

    Unknown names become 'general' rather than an error: a genre is a styling
    hint, and a hint we do not recognise must cost nothing.
    """
    if not genre:
        return "general"
    slug = re.sub(r"[^a-z]+", "_", str(genre).strip().lower()).strip("_")
    if slug in GENRE_STYLES:
        return slug
    aliases = {
        "scary": "horror", "horror_stories": "horror", "crime": "true_crime",
        "funny": "comedy", "education": "science_education",
        "science": "science_education", "educational": "science_education",
        "business": "finance", "money": "finance", "fitness": "health_fitness",
        "health": "health_fitness", "food": "cooking", "recipes": "cooking",
        "spiritual": "devotional", "religious": "devotional",
        "technology": "tech", "lifestyle": "vlog",
        "motivation": "motivational", "inspirational": "motivational",
        "geopolitical": "geopolitics", "geo": "geopolitics",
        "mystery_thriller": "mystery", "whodunit": "mystery",
        "financial_documentary": "finance", "financial": "finance",
    }
    return aliases.get(slug, "general")


def style_for(genre: Optional[str]) -> GenreStyle:
    return GENRE_STYLES[normalise(genre)]


def genre_block(genre: Optional[str]) -> str:
    """The paragraph appended to prompt-writing system prompts."""
    name = normalise(genre)
    if name == "general":
        return ""
    style = GENRE_STYLES[name]
    return (
        f"\n\nVIDEO GENRE: this is a {name.replace('_', ' ')} video. Every "
        f"scene you describe must belong in one — carry this mood in every "
        f"prompt: {style.look}."
    )


def apply_look(prompt: Optional[str], genre: Optional[str],
              secondary: Optional[str] = None) -> Optional[str]:
    """The genre's look stamped onto a positive prompt, exactly once.

    `secondary`, when given, blends its own style words in behind the
    primary's (see `blended_style_tags`) — the primary genre still leads.
    """
    if not prompt:
        return prompt
    look = (", ".join(blended_style_tags(genre, secondary)) if secondary
           else style_for(genre).look)
    if not look or look in prompt:
        return prompt
    return f"{prompt}, {look}"


def apply_negative(negative: str, genre: Optional[str],
                   secondary: Optional[str] = None) -> str:
    extra = style_for(genre).negative
    if secondary and normalise(secondary) not in ("general", normalise(genre)):
        sec_negative = style_for(secondary).negative
        if sec_negative and sec_negative not in extra:
            extra = f"{extra}, {sec_negative}" if extra else sec_negative
    if not extra or extra in (negative or ""):
        return negative
    return f"{negative}, {extra}" if negative else extra


def keyword_genre(text: str) -> str:
    """The genre the transcript's own words vote for, or 'general'."""
    words = re.split(r"[^\w]+", (text or "").lower())
    counts: Dict[str, int] = {}
    for word in words:
        for name, style in GENRE_STYLES.items():
            if word in style.keywords:
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "general"
    best = max(counts, key=lambda n: counts[n])
    if counts[best] < MIN_KEYWORD_VOTES:
        return "general"
    return best


async def detect_genre(program, ask) -> str:
    """The video's genre: the model's answer when there is one, else keywords.

    Never raises — a detection failure is a plain video, not a failed pass.
    """
    return await detect_genre_in_text(
        program.text_between(0.0, min(300.0, program.duration_s)),
        program.text_between(0.0, program.duration_s), ask)


async def detect_genre_in_text(excerpt: str, full_text: str, ask) -> str:
    """`detect_genre` over plain transcript text: `excerpt` (the opening minutes)
    for the model, `full_text` for the keyword fallback. For callers that have
    words but no Program yet — the auto-edit picks its pacing before any
    presentation exists."""
    if ask is not None and excerpt:
        try:
            answer = await ask(GENRE_SYSTEM, f"Transcript:\n{excerpt}\n\nGenre:",
                               GENRE_SCHEMA)
        except TypeError:
            try:
                answer = await ask(GENRE_SYSTEM, f"Transcript:\n{excerpt}\n\nGenre:")
            except Exception:
                answer = None
        except Exception:
            answer = None
        if answer:
            from .shotplan import first_json_object
            parsed = first_json_object(str(answer)) or {}
            name = normalise(parsed.get("genre"))
            if name != "general":
                logger.info("Genre (model): %s", name)
                return name
    name = keyword_genre(full_text)
    logger.info("Genre (keywords): %s", name)
    return name
