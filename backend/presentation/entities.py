"""What the speaker names: people, places, dates, numbers, terms, quotes, sources.

Everything the text cards and the maps key off comes from this one pass. It
runs per topic with the model when there is one — a schema-constrained call,
the only form the local model answers reliably — and falls back to patterns
(years, percentages, large numbers) when there is not, so a night without the
model still gets its stat call-outs.

Each entity is **anchored** to the moment its word is spoken: the beat lands
when the speaker says "25 percent", not at the start of the topic. Anchoring
is a search over the programme words for the entity's key token; when nothing
matches (the model paraphrased), the topic's start is used.
"""

import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from . import genre as genre_mod
from .models import Beat, PresentationSettings, Program, Topic

logger = logging.getLogger("presentation.entities")

AskJson = Callable[..., Awaitable[Optional[str]]]

ENTITY_SYSTEM = (
    "You are the fact-checker for a YouTube video. You are given numbered TOPICS "
    "from the video, each with the transcript of what the speaker says in it. The "
    "speaker may talk in Hindi written in Latin letters (Hinglish) with badly "
    "romanised spelling; read past the spelling.\n\n"
    "For each topic list what the speaker actually NAMES — only things said in the "
    "transcript, never things you know from elsewhere:\n"
    "  people      - named persons: name, and their role in one to five words "
    "('Cornell psychologist', 'the victim's brother')\n"
    "  places      - named places: city, region, country, building\n"
    "  dates       - years or dates as the speaker says them ('1987', 'March 2020')\n"
    "  numbers     - statistics with their meaning: text as spoken ('25 percent', "
    "'3.5 crore'), meaning in three to six words in ENGLISH ('students who noticed')\n"
    "  terms       - a technical term the speaker DEFINES, with the definition in one "
    "short sentence in ENGLISH\n"
    "  quotes      - words the speaker attributes to somebody else, with who said them\n"
    "  sources     - a study, book, paper, institution or publication cited, with its "
    "year if given\n"
    "  comparisons - two things the speaker sets against each other ('iPhone vs Pixel', "
    "'before and after'): a, b, and for each a short scene a camera could photograph\n\n"
    "RULES:\n"
    "1. Only what is in the transcript. Empty lists are the normal answer.\n"
    "2. Names and places in their usual spelling (Latin letters).\n"
    "3. Answer with JSON only:\n"
    '{"topics": [{"index": 0, "people": [{"name": "", "role": ""}], "places": [""], '
    '"dates": [""], "numbers": [{"text": "", "meaning": ""}], '
    '"terms": [{"term": "", "definition": ""}], "quotes": [{"text": "", "who": ""}], '
    '"sources": [{"source": "", "year": ""}], '
    '"comparisons": [{"a": "", "b": "", "scene_a": "", "scene_b": ""}]}]}'
)

ENTITY_SCHEMA = {
    "name": "entities",
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
                        "people": {"type": "array", "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "role": {"type": "string"}},
                            "required": ["name", "role"], "additionalProperties": False}},
                        "places": {"type": "array", "items": {"type": "string"}},
                        "dates": {"type": "array", "items": {"type": "string"}},
                        "numbers": {"type": "array", "items": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}, "meaning": {"type": "string"}},
                            "required": ["text", "meaning"], "additionalProperties": False}},
                        "terms": {"type": "array", "items": {
                            "type": "object",
                            "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
                            "required": ["term", "definition"], "additionalProperties": False}},
                        "quotes": {"type": "array", "items": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}, "who": {"type": "string"}},
                            "required": ["text", "who"], "additionalProperties": False}},
                        "sources": {"type": "array", "items": {
                            "type": "object",
                            "properties": {"source": {"type": "string"}, "year": {"type": "string"}},
                            "required": ["source", "year"], "additionalProperties": False}},
                        "comparisons": {"type": "array", "items": {
                            "type": "object",
                            "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                                           "scene_a": {"type": "string"}, "scene_b": {"type": "string"}},
                            "required": ["a", "b", "scene_a", "scene_b"], "additionalProperties": False}},
                    },
                    "required": ["index", "people", "places", "dates", "numbers",
                                 "terms", "quotes", "sources", "comparisons"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["topics"],
        "additionalProperties": False,
    },
}

TOPIC_BATCH = 6
MAX_TOPIC_CHARS = 1400

# Card durations. A number is read in a glance; a definition has to be read.
STAT_SECONDS = 3.5
LOCATION_SECONDS = 4.0
CHARACTER_SECONDS = 4.0
SOURCE_SECONDS = 3.5
DEFINITION_SECONDS = 5.0
QUOTE_SECONDS = 5.0
CHAPTER_SECONDS = 2.6
END_SCREEN_SECONDS = 12.0

_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_PERCENT_RE = re.compile(
    r"\b(\d{1,3}(?:\.\d+)?)\s*(%|percent|pratishat|parsent|prsent)\b", re.IGNORECASE)
_BIG_NUMBER_RE = re.compile(
    r"\b(\d{1,4}(?:[.,]\d+)?)\s*(crore|lakh|lakhs|million|billion|thousand|hazaar|hazar|karod)\b",
    re.IGNORECASE)
_NUMBER_WORD = {"crore": "crore", "karod": "crore", "lakh": "lakh", "lakhs": "lakh",
                "million": "million", "billion": "billion", "thousand": "thousand",
                "hazaar": "thousand", "hazar": "thousand"}


class TopicEntities(dict):
    """A plain dict with the seven lists always present."""

    def __init__(self, index: int):
        super().__init__(index=index, people=[], places=[], dates=[], numbers=[],
                         terms=[], quotes=[], sources=[], comparisons=[])


# --- extraction ---------------------------------------------------------------------

async def extract_entities(topics: Sequence[Topic], program: Program,
                           ask: Optional[AskJson],
                           genre: str = "general") -> List[Dict[str, Any]]:
    """One entity record per topic, model-read when possible, patterns otherwise.

    The pattern pass always runs and is merged in: a model that reads
    "twenty-five percent" as a number is welcome, but the regex catches the
    "25%" it skipped.
    """
    records = [TopicEntities(i) for i in range(len(topics))]
    if ask is not None and topics:
        try:
            await _ask_entities(topics, program, ask, genre, records)
        except Exception as e:
            logger.warning("Entity extraction failed (%s); using patterns only", e)
    for index, topic in enumerate(topics):
        _pattern_entities(program.text_between(topic.start_s, topic.end_s), records[index])
    total = sum(len(r[k]) for r in records
                for k in ("people", "places", "dates", "numbers", "terms", "quotes", "sources",
                          "comparisons"))
    logger.info("Entities: %d across %d topics", total, len(topics))
    return records


async def _ask_entities(topics: Sequence[Topic], program: Program, ask: AskJson,
                        genre: str, records: List[TopicEntities]) -> None:
    from .shotplan import _ask, first_json_object
    system = ENTITY_SYSTEM + genre_mod.genre_block(genre)
    for start in range(0, len(topics), TOPIC_BATCH):
        batch = topics[start:start + TOPIC_BATCH]
        listing = "\n\n".join(
            f"{start + i}. {t.topic}\n{program.text_between(t.start_s, t.end_s)[:MAX_TOPIC_CHARS]}"
            for i, t in enumerate(batch))
        answer = await _ask(ask, system, f"Topics:\n{listing}\n\nEntities:", ENTITY_SCHEMA)
        parsed = first_json_object(answer or "")
        if not parsed:
            continue
        for raw in parsed.get("topics", []) or []:
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            if not (0 <= index < len(records)):
                continue
            record = records[index]
            for person in raw.get("people") or []:
                name = _clean(person.get("name") if isinstance(person, dict) else person)
                role = _clean(person.get("role") if isinstance(person, dict) else "")
                if name and len(name.split()) <= 5:
                    record["people"].append({"name": name, "role": role})
            for place in raw.get("places") or []:
                place = _clean(place)
                if place and len(place.split()) <= 4:
                    record["places"].append(place)
            for date in raw.get("dates") or []:
                date = _clean(date)
                if date:
                    record["dates"].append(date)
            for number in raw.get("numbers") or []:
                if not isinstance(number, dict):
                    continue
                text = _clean(number.get("text"))
                if text and re.search(r"\d", text):
                    record["numbers"].append({"text": text, "meaning": _clean(number.get("meaning"))})
            for term in raw.get("terms") or []:
                if isinstance(term, dict) and _clean(term.get("term")) and _clean(term.get("definition")):
                    record["terms"].append({"term": _clean(term["term"]),
                                            "definition": _clean(term["definition"])[:140]})
            for quote in raw.get("quotes") or []:
                if isinstance(quote, dict) and _clean(quote.get("text")):
                    record["quotes"].append({"text": _clean(quote["text"])[:160],
                                             "who": _clean(quote.get("who"))})
            for source in raw.get("sources") or []:
                if isinstance(source, dict) and _clean(source.get("source")):
                    record["sources"].append({"source": _clean(source["source"])[:80],
                                              "year": _clean(source.get("year"))})
            for comparison in raw.get("comparisons") or []:
                if not isinstance(comparison, dict):
                    continue
                a, b = _clean(comparison.get("a")), _clean(comparison.get("b"))
                if a and b and a.lower() != b.lower():
                    record["comparisons"].append({
                        "a": a[:40], "b": b[:40],
                        "scene_a": _clean(comparison.get("scene_a")) or f"a clear photograph of {a}",
                        "scene_b": _clean(comparison.get("scene_b")) or f"a clear photograph of {b}"})


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().strip('"\'')


def _digits(text: str) -> str:
    return re.sub(r"[^\d.]", "", text or "")


def _pattern_entities(text: str, record: TopicEntities) -> None:
    """Numbers and years the text carries in plain sight."""
    # Duplicates are judged on the digits: the model's "25 percent" and the
    # pattern's "25%" are one statistic.
    seen = {_digits(n["text"]) for n in record["numbers"]}
    for match in _PERCENT_RE.finditer(text):
        spoken = f"{match.group(1)}%"
        if _digits(spoken) not in seen:
            record["numbers"].append({"text": spoken, "meaning": "", "pattern": True})
            seen.add(_digits(spoken))
    for match in _BIG_NUMBER_RE.finditer(text):
        unit = _NUMBER_WORD.get(match.group(2).lower(), match.group(2).lower())
        spoken = f"{match.group(1)} {unit}"
        if _digits(spoken) not in seen:
            record["numbers"].append({"text": spoken, "meaning": "", "pattern": True})
            seen.add(_digits(spoken))
    years = {d for d in record["dates"]}
    for match in _YEAR_RE.finditer(text):
        if match.group(1) not in years:
            record["dates"].append(match.group(1))
            years.add(match.group(1))


# --- anchoring -------------------------------------------------------------------------

_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def _key(token: str) -> str:
    return _STRIP.sub("", token.lower())


def anchor_time(program: Program, topic: Topic, phrase: str,
                default: Optional[float] = None) -> float:
    """When the phrase's most distinctive token is spoken inside the topic."""
    tokens = [t for t in (_key(p) for p in phrase.split()) if len(t) >= 2]
    fallback = topic.start_s if default is None else default
    if not tokens:
        return fallback
    words = program.words_between(topic.start_s, topic.end_s)
    # Digits first (the surest anchor), then the phrase's own order — a name
    # anchors on its first word, which is when the speaker starts saying it.
    ordered = [t for t in tokens if t.isdigit()] + [t for t in tokens if not t.isdigit()]
    for token in ordered:
        for word in words:
            key = _key(word.text)
            if key == token or (len(token) >= 4 and (token in key or key in token) and len(key) >= 3):
                return word.tl_start_s
    return fallback


# --- beats ------------------------------------------------------------------------------

def _stat_data(text: str) -> Dict[str, Any]:
    from .script import _stat_data
    return _stat_data(text)


def _rank_number(number: Dict[str, Any]) -> int:
    text = number["text"].lower()
    if "%" in text or "percent" in text:
        return 3
    if any(u in text for u in _NUMBER_WORD.values()):
        return 2
    return 1


def entity_beats(records: Sequence[Dict[str, Any]], topics: Sequence[Topic],
                 program: Program, settings: PresentationSettings,
                 genre: str = "general") -> List[Beat]:
    """The text cards (and map requests) the entities justify, one per kind per
    topic at most, each anchored to its spoken moment."""
    beats: List[Beat] = []
    named_people: set = set()
    shown_sources: set = set()
    wants_map = settings.maps
    for topic, record in zip(topics, records):
        span_end = min(program.duration_s, topic.end_s)

        def make(kind: str, at: float, seconds: float, **fields) -> Beat:
            start = max(0.0, min(at, span_end - 1.0))
            base = dict(start_s=start, end_s=min(span_end, start + seconds),
                        topic=topic.topic, summary=topic.summary, priority=0.9,
                        origin="entity", negative_prompt="")
            base.update(fields)
            return Beat(kind=kind, **base)

        # Places: a map for the first, a location card for it (with the topic's
        # first date as the second line — "Rajasthan, 1987").
        if settings.location_cards and record["places"]:
            place = record["places"][0]
            date = record["dates"][0] if record["dates"] else None
            at = anchor_time(program, topic, place)
            beats.append(make("location_card", at, LOCATION_SECONDS, text=place,
                              subtext=date, place=place))
            if wants_map:
                beats.append(make("map", at, settings.broll_seconds_max + 1.0,
                                  place=place, text=place, priority=0.95))

        if settings.character_cards:
            for person in record["people"]:
                key = person["name"].lower()
                if key in named_people or not person.get("role"):
                    continue
                named_people.add(key)
                at = anchor_time(program, topic, person["name"])
                beats.append(make("character_card", at, CHARACTER_SECONDS,
                                  text=person["name"], subtext=person["role"]))
                break

        if settings.stat_callouts and record["numbers"]:
            best = max(record["numbers"], key=_rank_number)
            data = _stat_data(best["text"])
            if data:
                at = anchor_time(program, topic, best["text"])
                beats.append(make("stat_callout", at, STAT_SECONDS, text=best["text"],
                                  subtext=best.get("meaning") or None, data=data))

        if settings.source_cards:
            for source in record["sources"]:
                key = source["source"].lower()
                if key in shown_sources:
                    continue
                shown_sources.add(key)
                label = source["source"] + (f", {source['year']}" if source.get("year") else "")
                at = anchor_time(program, topic, source["source"])
                beats.append(make("source_card", at, SOURCE_SECONDS, text=f"Source: {label}"))
                break

        if settings.definition_cards and record["terms"]:
            term = record["terms"][0]
            at = anchor_time(program, topic, term["term"])
            beats.append(make("definition_card", at, DEFINITION_SECONDS,
                              text=term["term"], subtext=term["definition"]))

        if settings.split_screens and record.get("comparisons"):
            comparison = record["comparisons"][0]
            at = anchor_time(program, topic, comparison["a"])
            beats.append(make("split", at, settings.broll_seconds_max + 1.5,
                              image_prompt=comparison["scene_a"],
                              data={"left": comparison["scene_a"], "right": comparison["scene_b"],
                                    "labels": [comparison["a"], comparison["b"]]},
                              negative_prompt="text, watermark, logo, deformed hands, blurry",
                              priority=0.92))

        if settings.quote_cards and record["quotes"]:
            quote = record["quotes"][0]
            at = anchor_time(program, topic, quote["text"])
            beats.append(make("quote_card", at, QUOTE_SECONDS, text=f"“{quote['text']}”",
                              subtext=quote.get("who") or None))

    return beats


def structure_beats(topics: Sequence[Topic], program: Program,
                    settings: PresentationSettings) -> List[Beat]:
    """Chapter titles at every topic after the first, and the end screen."""
    beats: List[Beat] = []
    if settings.chapter_titles:
        for index, topic in enumerate(sorted(topics, key=lambda t: t.start_s)):
            if index == 0 or topic.start_s < 3.0:
                continue
            name = (topic.heading or topic.topic or "").strip()
            if not name:
                continue
            beats.append(Beat(
                kind="chapter_title", start_s=topic.start_s,
                end_s=min(program.duration_s, topic.start_s + CHAPTER_SECONDS),
                topic=topic.topic, text=_title_case(name), priority=0.95,
                origin="structure", negative_prompt=""))
    if settings.end_screen and program.duration_s > 30.0:
        start = max(0.0, program.duration_s - END_SCREEN_SECONDS)
        beats.append(Beat(
            kind="end_screen", start_s=start, end_s=program.duration_s,
            topic="end", text=settings.end_screen_text, priority=1.0,
            origin="structure", negative_prompt=""))
    return beats


def _title_case(name: str) -> str:
    words = name.split()
    if not words:
        return name
    if name.isupper() or len(words) > 7:
        return name
    small = {"a", "an", "the", "of", "in", "on", "and", "or", "to", "ka", "ki", "ke", "aur"}
    return " ".join(w if (i and w.lower() in small) else (w[:1].upper() + w[1:])
                    for i, w in enumerate(words))
