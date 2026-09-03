"""The emotional shape of the video, topic by topic, and what the edit does about it.

A story is not flat. A horror narration sets up, builds, turns, and lands;
an explainer opens a question, works through it, and pays it off. Nothing in
the pass knew that — every topic got the same treatment. This module tags
each topic with a **mood** and applies a **recipe** for it:

* a grade shift over the topic (desaturate and vignette as tension rises,
  wash out in the aftermath, warm up when things turn hopeful);
* a slow push-in or pull-back across the topic;
* an atmosphere window (fog for tension, lightning at the climax);
* at the topic's **key moment** — the reveal, the scare — a hit: a flash
  frame, a camera shake, a thunderclap or stinger, and a riser leading in;
* a transition choice for the topic boundary (white for a reveal, black for
  the aftermath).

Horror and true crime take the recipes at full strength. Everything else gets
the grade shifts at half strength and none of the hits: a flash frame in a
cooking video is a mistake, not a mood.

The tagging is one schema-constrained model call over the topic summaries
(the same form every other planner uses); without a model the story's arc is
assumed from position — calm, build, tense, climax, aftermath spread across
the topics for horror; calm throughout, ending hopeful, for the rest. A
`[mood: …]` stage direction in the script overrides either.
"""

import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from timeline import clip_ops
from timeline.authoring import clear_generated
from timeline.schema import AtmosphereEffect, Timeline, time_to_frame

from . import genre as genre_mod
from .models import PresentationSettings, Program, Topic

logger = logging.getLogger("presentation.mood")

AskJson = Callable[..., Awaitable[Optional[str]]]

MOODS = ("calm", "build", "tense", "reveal", "climax", "aftermath", "comedic", "hopeful")
MOOD_ORIGIN = "mood"
MOOD_TRACK = "V8"          # above B-roll (V3) and the punch-in layers, below cards' shade
HIT_TRACK = "V8"

MOOD_SYSTEM = (
    "You are the editor of a YouTube video. You are given its TOPICS in order, "
    "each with a one-sentence summary and the transcript of what the speaker says. "
    "The speaker may talk in Hindi written in Latin letters; read past the spelling.\n\n"
    "For each topic decide its MOOD — the emotional register the edit should carry:\n"
    "  calm      - ordinary, informative, relaxed\n"
    "  build     - something is starting; curiosity or unease rising\n"
    "  tense     - dread, suspense, danger approaching\n"
    "  reveal    - the twist, the answer, the thing the video was building to\n"
    "  climax    - the peak: the scare, the confrontation, the moment of impact\n"
    "  aftermath - after the peak: consequence, loss, reflection\n"
    "  comedic   - played for laughs\n"
    "  hopeful   - uplifting, encouraging, resolved\n\n"
    "Also give key_moment_s: the second (from the timestamps in the transcript) of "
    "the single strongest instant in the topic — where a scare or a reveal lands. "
    "Use the topic's start when nothing stands out.\n\n"
    "Answer with JSON only:\n"
    '{"topics": [{"index": 0, "mood": "calm", "key_moment_s": 0.0}]}'
)

MOOD_SCHEMA = {
    "name": "moods",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "mood": {"type": "string", "enum": list(MOODS)},
                        "key_moment_s": {"type": "number"},
                    },
                    "required": ["index", "mood", "key_moment_s"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["topics"],
        "additionalProperties": False,
    },
}

TOPIC_BATCH = 10
MAX_TOPIC_CHARS = 900

# Words that mark a turn, for the keyword fallback and for sharpening a key
# moment: the hit lands on the first of these inside the topic when the model
# gave none.
_TURN_WORDS = {"achanak", "suddenly", "phir", "tab", "lekin", "but", "aur", "then",
               "dekha", "saw", "awaaz", "sound", "darwaza", "door", "scream", "cheekh",
               "khoon", "blood", "twist", "reveal", "pata", "turned", "finally", "aakhir"}

# Genres whose recipes run at full strength, hits included.
_FULL_STRENGTH = {"horror", "true_crime"}


# --- recipes ---------------------------------------------------------------------------
#
# color: ColorGrade deltas over the topic; push: scale change across the topic
# (+ in, − out); atmosphere: windowed layers; hit: what fires at the key
# moment (visual kinds + sfx tag); riser: a riser ending on the hit;
# loop: a looped effect under the whole topic; transition: (type, seconds)
# INTO the topic.

RECIPES: Dict[str, Dict[str, Any]] = {
    "calm": {},
    "build": {
        "color": {"saturation": 0.92, "vignette": 0.15},
        "push": 0.04,
    },
    "tense": {
        "color": {"saturation": 0.84, "vignette": 0.3, "contrast": 1.08, "temperature": -0.12},
        "push": 0.07,
        "atmosphere": [("fog", 0.22)],
        "loop": ("heartbeat", 0.22),
        "transition": ("fadeblack", 0.5),
    },
    "reveal": {
        "color": {"contrast": 1.06, "vignette": 0.18},
        "push": 0.05,
        "hit": {"visual": ["flash"], "sfx": "stinger"},
        "riser": True,
        "transition": ("fadewhite", 0.35),
    },
    "climax": {
        "color": {"saturation": 0.78, "vignette": 0.42, "contrast": 1.16, "temperature": -0.2},
        "push": 0.09,
        "atmosphere": [("lightning", 0.5), ("flicker", 0.35)],
        "hit": {"visual": ["flash", "shake", "glitch"], "sfx": "thunder"},
        "riser": True,
        "loop": ("heartbeat", 0.35),
        "transition": ("fadeblack", 0.6),
    },
    "aftermath": {
        "color": {"saturation": 0.6, "brightness": -0.05, "vignette": 0.3},
        "push": -0.05,
        "atmosphere": [("grain", 0.25)],
        "transition": ("fadeblack", 0.9),
    },
    "comedic": {
        "color": {"saturation": 1.15, "brightness": 0.02},
        "transition": ("slideleft", 0.3),
    },
    "hopeful": {
        "color": {"temperature": 0.16, "saturation": 1.1, "brightness": 0.02},
        "push": -0.03,
        "atmosphere": [("sunlight", 0.2)],
        "transition": ("fadewhite", 0.5),
    },
}

HIT_FLASH_S = 0.08
HIT_SHAKE_S = 0.55
HIT_GLITCH_S = 0.35
RISER_S = 3.2


# --- tagging ---------------------------------------------------------------------------

async def tag_moods(topics: Sequence[Topic], program: Program, ask: Optional[AskJson],
                    genre: str = "general",
                    overrides: Optional[Dict[int, str]] = None) -> List[Topic]:
    """Fill each topic's `mood` and `key_moment_s`. Returns the same topics.

    `overrides` maps a topic index (in `topics` order) to a mood the user set
    in the script; those win over the model and the fallback.
    """
    ordered = sorted(topics, key=lambda t: t.start_s)
    for index, topic in enumerate(ordered):
        topic.mood = _fallback_mood(index, len(ordered), genre)
        topic.key_moment_s = None

    if ask is not None and ordered:
        try:
            await _ask_moods(ordered, program, ask, genre)
        except Exception as e:
            logger.warning("Mood tagging failed (%s); using the positional arc", e)

    for index, mood in (overrides or {}).items():
        if 0 <= index < len(ordered) and mood in MOODS:
            ordered[index].mood = mood

    for topic in ordered:
        if topic.key_moment_s is None or not (topic.start_s <= topic.key_moment_s < topic.end_s):
            topic.key_moment_s = _key_moment(topic, program)
    logger.info("Moods: %s", ", ".join(t.mood or "-" for t in ordered))
    return list(topics)


def _fallback_mood(index: int, count: int, genre: str) -> str:
    if count <= 0:
        return "calm"
    position = index / max(1, count - 1) if count > 1 else 0.0
    if genre in _FULL_STRENGTH:
        if count == 1:
            return "tense"
        arc = ["calm", "build", "tense", "climax", "aftermath"]
        return arc[min(len(arc) - 1, int(position * (len(arc) - 0.001)))]
    if genre == "comedy":
        return "comedic"
    if genre == "motivational":
        return "hopeful" if position >= 0.5 else "build"
    return "hopeful" if index == count - 1 and count > 1 else "calm"


async def _ask_moods(topics: List[Topic], program: Program, ask: AskJson, genre: str) -> None:
    from .shotplan import _ask, first_json_object
    system = MOOD_SYSTEM + genre_mod.genre_block(genre)
    for start in range(0, len(topics), TOPIC_BATCH):
        batch = topics[start:start + TOPIC_BATCH]
        listing = "\n\n".join(
            f"{start + i}. [{t.start_s:.1f}s–{t.end_s:.1f}s] {t.topic}: {t.summary}\n"
            f"{program.text_between(t.start_s, t.end_s)[:MAX_TOPIC_CHARS]}"
            for i, t in enumerate(batch))
        answer = await _ask(ask, system, f"Topics:\n{listing}\n\nMoods:", MOOD_SCHEMA)
        parsed = first_json_object(answer or "")
        if not parsed:
            continue
        for raw in parsed.get("topics", []) or []:
            try:
                index = int(raw.get("index"))
                mood = str(raw.get("mood", "")).strip().lower()
                key = float(raw.get("key_moment_s", -1))
            except (TypeError, ValueError):
                continue
            if not (0 <= index < len(topics)) or mood not in MOODS:
                continue
            topic = topics[index]
            topic.mood = mood
            if topic.start_s <= key < topic.end_s:
                topic.key_moment_s = key


def _key_moment(topic: Topic, program: Program) -> float:
    """The first turn word in the topic, else its loudest word, else its start."""
    words = program.words_between(topic.start_s, topic.end_s)
    if not words:
        return topic.start_s
    for word in words:
        if word.text.strip(".,!?").lower() in _TURN_WORDS and word.tl_start_s > topic.start_s + 1.5:
            return word.tl_start_s
    if program.has_energy:
        loudest = max(words, key=lambda w: w.emphasis_z)
        if loudest.emphasis_z > 0.5 and loudest.tl_start_s > topic.start_s + 1.0:
            return loudest.tl_start_s
    return topic.start_s + min(2.0, (topic.end_s - topic.start_s) / 2.0)


# --- applying ------------------------------------------------------------------------------

class MoodPlan:
    """What the visual pass produced, for the stages that follow it."""

    def __init__(self) -> None:
        self.sfx: List[Tuple[float, str]] = []
        self.loops: List[Tuple[float, float, str, float]] = []
        self.transitions: Dict[int, Tuple[str, float]] = {}
        self.layers = 0
        self.hits = 0
        self.moods: List[str] = []


def apply_moods(timeline: Timeline, topics: Sequence[Topic], program: Program,
                settings: PresentationSettings, genre: str, seed: int = 0) -> MoodPlan:
    """Put every topic's recipe on the timeline; return the sound and
    transition requests for the stages that own those."""
    clear_generated(timeline, MOOD_ORIGIN)
    plan = MoodPlan()
    strength = max(0.0, min(2.0, settings.mood_strength))
    if strength <= 0.0 or not topics:
        return plan
    full = genre in _FULL_STRENGTH
    scale = strength * (1.0 if full else 0.5)
    rng = random.Random(seed)
    fps_num, fps_den = timeline.fps_num, timeline.fps_den

    for index, topic in enumerate(sorted(topics, key=lambda t: t.start_s)):
        mood = topic.mood or "calm"
        plan.moods.append(mood)
        recipe = RECIPES.get(mood, {})
        if not recipe:
            continue
        start = max(0.0, topic.start_s)
        end = min(program.duration_s, topic.end_s)
        if end - start < 1.0:
            continue
        start_frame = time_to_frame(start, fps_num, fps_den)
        frames = max(1, time_to_frame(end - start, fps_num, fps_den))

        # The topic-long layer: grade, push, atmosphere.
        color = {k: _scaled(k, v, scale) for k, v in (recipe.get("color") or {}).items()}
        push = float(recipe.get("push", 0.0)) * scale
        atmosphere = [(kind, level * scale) for kind, level in (recipe.get("atmosphere") or [])
                      if full or kind in ("grain", "sunlight", "fog")]
        if color or abs(push) > 0.005 or atmosphere:
            layer = clip_ops.add_adjustment_item(timeline, start_frame, frames, track=MOOD_TRACK)
            layer.origin = MOOD_ORIGIN
            layer.label = f"mood: {mood}"
            if color:
                clip_ops.set_color(timeline, layer.id, color)
            if abs(push) > 0.005:
                if push > 0:
                    clip_ops.set_transform(timeline, layer.id,
                                           {"scale": 1.0, "scale_end": 1.0 + push})
                else:
                    clip_ops.set_transform(timeline, layer.id,
                                           {"scale": 1.0 - push, "scale_end": 1.0})
            for kind, level in atmosphere:
                layer.atmosphere.append(AtmosphereEffect(
                    type=kind, intensity=max(0.05, min(1.0, level)), origin=MOOD_ORIGIN))
            plan.layers += 1

        transition = recipe.get("transition")
        if transition and index > 0:
            plan.transitions[index] = (transition[0], transition[1])

        if not full:
            continue

        loop = recipe.get("loop")
        if loop:
            plan.loops.append((start, end, loop[0], loop[1] * scale))

        hit = recipe.get("hit")
        key = topic.key_moment_s if topic.key_moment_s is not None else start
        key = max(start, min(end - 0.5, key))
        if hit:
            for kind in hit.get("visual", []):
                seconds = {"flash": HIT_FLASH_S, "shake": HIT_SHAKE_S, "glitch": HIT_GLITCH_S}.get(kind, 0.2)
                level = {"flash": 0.9, "shake": 0.7, "glitch": 0.6}.get(kind, 0.5) * min(1.0, scale)
                item = clip_ops.add_adjustment_item(
                    timeline, time_to_frame(key, fps_num, fps_den),
                    max(1, time_to_frame(seconds, fps_num, fps_den)), track=HIT_TRACK)
                item.origin = MOOD_ORIGIN
                item.label = f"hit: {kind}"
                item.atmosphere.append(AtmosphereEffect(
                    type=kind, intensity=level, origin=MOOD_ORIGIN,
                    color="0xFFFFFFFF" if kind == "flash" else None))
            if hit.get("sfx"):
                plan.sfx.append((key, hit["sfx"]))
            plan.hits += 1
        if recipe.get("riser") and key - start >= 1.5:
            plan.sfx.append((max(start, key - RISER_S), "riser"))

    timeline.recalculate_duration()
    timeline.revision += 1
    logger.info("Moods applied: %d layers, %d hits (%s)", plan.layers, plan.hits,
                "full" if full else "subtle")
    return plan


def _scaled(key: str, value: float, scale: float) -> float:
    """A grade delta at `scale`: neutral is 1.0 for the multiplicative fields."""
    neutral = 1.0 if key in ("saturation", "contrast", "gamma") else 0.0
    return round(neutral + (float(value) - neutral) * scale, 4)
