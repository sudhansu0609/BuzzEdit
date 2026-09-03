"""Entities → text cards: extraction, anchoring, placement, and the counter."""

import asyncio
import json
from pathlib import Path

import pytest

from backend.presentation import cards, entities
from backend.presentation.models import Beat, PresentationSettings, Topic
from backend.presentation.program import build_program
from backend.presentation.shotplan import sanitize_beats
from backend.render.text import build_drawtext, counter_text, escape_expansion
from backend.timeline import build_timeline_from_transcript
from backend.timeline.schema import TextClip, TextStyle

TEXT = ("Namaste dosto aaj hum Cornell University ki ek study ki baat karenge . "
        "Thomas Gilovich ne 2000 mein ek experiment kiya . "
        "Students ko lagta tha sab dekh rahe hain lekin sirf 25 percent logon ne notice kiya . "
        "Isko spotlight effect kehte hain matlab hum sochte hain sab humein dekh rahe hain . "
        "Aur ek study mein 3.5 crore log the .")


def _timeline():
    words = [{"word": t, "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i, t in enumerate(TEXT.split())]
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", 40.0, words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, 40.0)], pause_padding_seconds=0.0)


def _topics(program):
    return [Topic(start_s=0.0, end_s=19.0, topic="the Cornell study"),
            Topic(start_s=19.0, end_s=program.duration_s, topic="spotlight effect")]


# --- extraction ---------------------------------------------------------------

def test_patterns_find_percentages_big_numbers_and_years_without_a_model():
    timeline = _timeline()
    program = build_program(timeline)
    records = asyncio.run(entities.extract_entities(_topics(program), program, None))
    first, second = records
    assert "2000" in first["dates"]
    assert any(n["text"] == "25%" for n in first["numbers"])
    assert any(n["text"] == "3.5 crore" for n in second["numbers"])
    assert first["people"] == [] and first["places"] == []


def test_the_model_answer_is_merged_with_the_patterns():
    timeline = _timeline()
    program = build_program(timeline)

    async def ask(system, user, schema=None):
        assert schema["name"] == "entities"
        return json.dumps({"topics": [{
            "index": 0,
            "people": [{"name": "Thomas Gilovich", "role": "Cornell psychologist"}],
            "places": ["Cornell University"], "dates": [],
            "numbers": [{"text": "25 percent", "meaning": "students who noticed"}],
            "terms": [], "quotes": [],
            "sources": [{"source": "Journal of Personality", "year": "2000"}],
        }, {"index": 1, "people": [], "places": [], "dates": [], "numbers": [],
            "terms": [{"term": "spotlight effect",
                       "definition": "Believing everyone notices you more than they do"}],
            "quotes": [], "sources": []}]})

    records = asyncio.run(entities.extract_entities(_topics(program), program, ask))
    assert records[0]["people"][0]["name"] == "Thomas Gilovich"
    assert records[0]["places"] == ["Cornell University"]
    # The pattern pass still adds the year and does not duplicate the percent.
    assert "2000" in records[0]["dates"]
    assert sum(1 for n in records[0]["numbers"] if "25" in n["text"]) == 1
    assert records[1]["terms"][0]["term"] == "spotlight effect"


def test_beats_are_anchored_to_the_spoken_word():
    timeline = _timeline()
    program = build_program(timeline)
    topics = _topics(program)
    records = [entities.TopicEntities(0), entities.TopicEntities(1)]
    records[0]["people"].append({"name": "Thomas Gilovich", "role": "psychologist"})
    records[0]["places"].append("Cornell University")
    records[0]["dates"].append("2000")
    records[0]["numbers"].append({"text": "25%", "meaning": "noticed"})
    records[0]["sources"].append({"source": "Cornell study", "year": "2000"})
    records[1]["terms"].append({"term": "spotlight effect", "definition": "a bias"})
    records[1]["numbers"].append({"text": "3.5 crore", "meaning": ""})

    beats = entities.entity_beats(records, topics, program, PresentationSettings(maps=False))
    by_kind = {b.kind: b for b in beats}
    words = {w.text: w.tl_start_s for w in program.words}
    assert by_kind["character_card"].start_s == pytest.approx(words["Thomas"])
    assert by_kind["location_card"].start_s == pytest.approx(words["Cornell"])
    assert by_kind["location_card"].subtext == "2000"
    stats = [b for b in beats if b.kind == "stat_callout"]
    assert stats[0].start_s == pytest.approx(words["25"])
    assert stats[0].data["value"] == 25.0
    assert by_kind["definition_card"].start_s == pytest.approx(words["spotlight"])
    assert all(b.origin == "entity" for b in beats)
    # Only one stat per topic: the second topic's crore figure is its own.
    assert len(stats) == 2 and stats[1].data["decimals"] == 1

    with_maps = entities.entity_beats(records, topics, program, PresentationSettings())
    assert any(b.kind == "map" and b.place == "Cornell University" for b in with_maps)


def test_structure_beats_give_chapter_titles_and_an_end_screen():
    timeline = _timeline()
    program = build_program(timeline)
    topics = [Topic(start_s=0.0, end_s=10.0, topic="intro"),
              Topic(start_s=10.0, end_s=25.0, topic="the cornell study"),
              Topic(start_s=25.0, end_s=program.duration_s, topic="what it means",
                    heading="What It Means")]
    beats = entities.structure_beats(topics, program, PresentationSettings())
    chapters = [b for b in beats if b.kind == "chapter_title"]
    assert [b.text for b in chapters] == ["The Cornell Study", "What It Means"]
    assert chapters[0].start_s == 10.0
    end = next(b for b in beats if b.kind == "end_screen")
    assert end.end_s == program.duration_s and end.text == "SUBSCRIBE"
    # Short programmes get no end screen.
    short = program.model_copy(update={"duration_s": 20.0})
    assert not any(b.kind == "end_screen"
                   for b in entities.structure_beats(topics, short, PresentationSettings()))


# --- placement ------------------------------------------------------------------

def test_cards_collide_only_within_their_zone_and_reruns_replace():
    timeline = _timeline()
    program = build_program(timeline)
    beats = [
        Beat(kind="chapter_title", start_s=10.0, end_s=12.6, topic="a", text="Chapter", origin="structure"),
        Beat(kind="stat_callout", start_s=10.2, end_s=13.7, topic="a", text="25%",
             subtext="noticed", data={"value": 25, "suffix": "%"}, origin="entity"),
        Beat(kind="location_card", start_s=10.0, end_s=14.0, topic="a", text="Cornell",
             subtext="2000", origin="entity"),
        Beat(kind="source_card", start_s=10.0, end_s=13.5, topic="a", text="Source: x", origin="entity"),
        Beat(kind="end_screen", start_s=28.0, end_s=program.duration_s, topic="end",
             text="SUBSCRIBE", origin="structure"),
    ]
    counts = cards.place_cards(timeline, beats, program, PresentationSettings(), "general")
    assert counts["chapter_title"] == 1 and counts["location_card"] == 1
    assert counts["source_card"] == 1 and counts["end_screen"] == 1
    items = [i for i in timeline.items if i.origin == cards.CARD_ORIGIN and i.kind == "text"]
    # The stat slid forward past the chapter title (same zone) instead of overlapping.
    stat = next(i for i in items if i.label.startswith("stat:"))
    chapter = next(i for i in items if i.label.startswith("chapter_title"))
    assert stat.timeline_start_frame >= chapter.timeline_end_frame + 30
    assert stat.text.counter["to"] == 25.0 and stat.text.counter["suffix"] == "%"
    assert any(i.label.startswith("stat label") for i in items)
    # Corner cards share the moment with the chapter.
    location = next(i for i in items if i.label.startswith("location_card"))
    assert location.timeline_start_frame == chapter.timeline_start_frame
    assert location.text.content == "Cornell\n2000"
    assert {i.track for i in items} == {cards.CARD_TRACK}
    shade = [i for i in timeline.items if i.origin == cards.CARD_ORIGIN and i.kind == "adjustment"]
    assert len(shade) == 1 and shade[0].color.brightness < 0

    again = cards.place_cards(timeline, beats, program, PresentationSettings(), "horror")
    assert again == counts
    assert len([i for i in timeline.items if i.origin == cards.CARD_ORIGIN]) == len(items) + 1
    horror_chapter = next(i for i in timeline.items if (i.label or "").startswith("chapter_title"))
    assert horror_chapter.text.style.font_family == "Georgia"


def test_centre_cards_keep_clear_of_the_opening_title_but_not_of_popups():
    timeline = _timeline()
    program = build_program(timeline)
    beats = [Beat(kind="chapter_title", start_s=1.0, end_s=3.6, topic="a", text="Too early",
                  origin="structure"),
             Beat(kind="definition_card", start_s=20.0, end_s=25.0, topic="b", text="Term",
                  subtext="gloss", origin="entity")]
    counts = cards.place_cards(timeline, beats, program, PresentationSettings(), "general",
                               popup_windows=[(20.0, 23.5)])
    chapter = next(i for i in timeline.items if (i.label or "").startswith("chapter_title"))
    assert chapter.timeline_start_frame / 30 >= cards.TITLE_ZONE_S + cards.ZONE_GAP_S - 0.01
    definition = next(i for i in timeline.items if (i.label or "").startswith("definition"))
    # A pop-up in the upper third and a definition in the middle can share a moment.
    assert definition.timeline_start_frame / 30 == pytest.approx(20.0)
    assert counts == {"chapter_title": 1, "definition_card": 1}


def test_text_beats_survive_sanitising():
    timeline = _timeline()
    program = build_program(timeline)
    beats = [Beat(kind="stat_callout", start_s=5.0, end_s=8.5, topic="a", text="25%",
                  data={"value": 25}, origin="entity"),
             Beat(kind="stat_callout", start_s=5.0, end_s=8.5, topic="a", text="",
                  origin="entity"),
             Beat(kind="map", start_s=5.0, end_s=9.0, topic="a", origin="entity")]
    kept, dropped = sanitize_beats(beats, program, PresentationSettings())
    assert [b.kind for b in kept] == ["stat_callout"]
    assert {d["reason"] for d in dropped} == {"stat_callout with no text", "map with no place"}


# --- the counter ------------------------------------------------------------------

def test_counter_text_builds_a_drawtext_expansion():
    text = counter_text({"from": 0, "to": 25, "suffix": "%", "seconds": 1.2}, 10.0)
    assert text.startswith("%{expr_int_format:round(")
    assert text.endswith("\\%")
    assert "clip((t-10)/1.2\\,0\\,1)" in text
    decimal = counter_text({"to": 3.5, "suffix": "crore", "decimals": 1}, 0.0)
    assert decimal.count("%{expr_int_format") == 2 and decimal.endswith(" crore")
    assert escape_expansion("25% off") == "25\\% off"


def test_a_counter_clip_and_a_percent_caption_both_compile(tmp_path):
    clip = TextClip(content="25%", style=TextStyle(font_family="Arial"),
                    counter={"to": 25, "suffix": "%"})
    built = build_drawtext(clip, 1.0, 4.0, tmp_path)
    assert built and "textfile=" in built
    asset = next(tmp_path.glob("text_*.txt"))
    assert "expr_int_format" in asset.read_text(encoding="utf-8")

    plain = TextClip(content="sirf 25% log", style=TextStyle(font_family="Arial"))
    build_drawtext(plain, 1.0, 4.0, tmp_path)
    contents = [p.read_text(encoding="utf-8") for p in tmp_path.glob("text_*.txt")]
    assert any(c == "sirf 25\\% log" for c in contents)
