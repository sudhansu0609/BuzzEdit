"""Story structure: acts, the hook and cold open, chapters and the listing."""

import asyncio
import json

import pytest

from backend.presentation import structure
from backend.presentation.models import Beat, PresentationSettings, Topic
from backend.presentation.program import build_program
from backend.presentation.shotplan import plan_shots
from backend.timeline import apply_intro, build_timeline_from_transcript, clip_ops, remove_intro
from backend.timeline.ops import rebuild_primary_tracks

LINES = [
    "namaste dosto aaj ki kahani ek purane ghar ki hai .",
    "us ghar mein raat ko koi rehta tha jo din mein nahi dikhta tha .",
    "kya aap jaante hain ki 90 percent log is ghar se bhaag gaye ?",
    "sab kuch tab shuru hua jab ek parivaar wahan rehne aaya .",
    "achanak ek raat unhe darwaze par dastak sunai di lekin bahar koi nahi tha .",
    "phir jo hua usne unki zindagi badal di .",
    "aaj tak koi nahi jaanta us raat kya hua tha .",
    "video pasand aaye toh like aur subscribe zaroor karna .",
]


def _timeline(step: float = 0.45):
    words = []
    clock = 0.0
    for line in LINES:
        for token in line.split():
            words.append({"word": token, "start": round(clock, 3), "end": round(clock + 0.35, 3)})
            clock += step
        clock += 2.6                                  # a long pause ends the line
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", clock + 1.0, words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, clock + 1.0)], pause_padding_seconds=0.0,
        max_pause_seconds=5.0)


def _topics(program):
    total = program.duration_s
    edges = [0.0, total * 0.15, total * 0.35, total * 0.55, total * 0.75, total * 0.9, total]
    names = ["the house", "the family", "the knock", "the night", "the mystery", "outro"]
    return [Topic(start_s=a, end_s=b, topic=n) for a, b, n in zip(edges, edges[1:], names)]


# --- acts ---------------------------------------------------------------------

def test_positional_acts_follow_the_genre_and_spot_the_call_to_action():
    program = build_program(_timeline())
    topics = _topics(program)
    asyncio.run(structure.tag_structure(topics, program, None, "horror"))
    acts = [t.act for t in topics]
    assert acts[0] == "hook" and acts[-1] == "cta"
    assert "build" in acts and "climax" in acts
    plain = _topics(program)
    asyncio.run(structure.tag_structure(plain, program, None, "science_education"))
    assert [t.act for t in plain][0] == "hook" and plain[-1].act == "cta"
    assert "point" in [t.act for t in plain]
    assert structure.act_weight("build") == 1.0 and structure.act_weight("cta") < 0.5
    assert structure.act_weight(None) == 1.0


def test_the_models_acts_and_hook_are_taken_and_validated():
    program = build_program(_timeline())
    topics = _topics(program)
    line = next(l for l in structure.transcript_lines(program).splitlines() if "90 percent" in l)
    start = float(line.split("]")[0][1:])

    async def ask(system, user, schema=None):
        assert schema["name"] == "structure"
        return json.dumps({"acts": [{"index": 0, "act": "setup"}, {"index": 1, "act": "build"},
                                    {"index": 2, "act": "reveal"}, {"index": 3, "act": "climax"},
                                    {"index": 4, "act": "aftermath"}, {"index": 5, "act": "cta"},
                                    {"index": 9, "act": "climax"}],
                           "hook": {"start_s": start, "text": "kya aap jaante hain"}})

    hook = asyncio.run(structure.tag_structure(topics, program, ask, "horror"))
    assert [t.act for t in topics] == ["setup", "build", "reveal", "climax", "aftermath", "cta"]
    assert hook.source == "llm" and "90 percent" in hook.text
    assert hook.start_s == pytest.approx(start)
    assert structure.MIN_HOOK_S <= hook.end_s - hook.start_s <= structure.MAX_HOOK_S + 0.5


def test_a_hook_the_model_invented_is_refused_and_the_fallback_scores_the_question():
    program = build_program(_timeline())
    assert structure._validate_hook(program, 999.0, "not in the video") is None
    hook = structure.fallback_hook(program)
    assert hook is not None and hook.source == "fallback"
    assert "?" in hook.text                       # the question beats the rest


def test_acts_weigh_the_budget_toward_the_dense_parts():
    program = build_program(_timeline())
    topics = _topics(program)

    async def ask(system, user, schema=None):
        if schema and schema["name"] == "beats":
            names = [t.topic for t in topics]
            return json.dumps({"beats": [{"topic": n, "kind": "broll_image",
                                          "image_prompt": f"a picture of {n}", "video_prompt": None,
                                          "popup_text": None, "negative_prompt": "text",
                                          "style_hint": "photoreal"} for n in names]})
        return None

    async def enrich(found):
        for topic, act in zip(found, ["hook", "setup", "build", "climax", "aftermath", "cta"]):
            topic.act = act

    plan = asyncio.run(plan_shots(program, PresentationSettings(target_coverage=0.2, cards=False,
                                                                popups=False), ask, "horror",
                                  script_topics=[t.model_copy() for t in topics], enrich=enrich))
    by_topic = {b.topic: b for b in plan.beats if b.is_cutaway}
    kept = set(by_topic)
    assert "the knock" in kept                    # build: full weight
    assert "outro" not in kept                    # cta: weighed down and dropped first


# --- the cold open --------------------------------------------------------------------

def test_the_cold_open_copies_the_hook_to_the_front_and_shifts_everything_back():
    timeline = _timeline()
    program = build_program(timeline)
    hook = structure.fallback_hook(program)
    assert hook is not None
    clip_ops.add_text_item(timeline, "card", 300, 60, track="T1")
    music = clip_ops.add_media_item(timeline, next(iter(timeline.sources)), "A2", 0, 0,
                                    timeline.duration_frames)
    music.origin = "music"
    before_v1 = [(i.timeline_start_frame, i.source_start_frame)
                 for i in timeline.items if i.track == "V1"]

    done = structure.apply_cold_open(timeline, program, hook, PresentationSettings())
    assert done and done["text"] == hook.text
    offset = timeline.cold_open_frames
    assert offset == timeline.program_offset_frames > 0
    assert offset == int(round(done["seconds"] * 30)) + 18       # hook + 0.6 s of black
    # The programme moved back by the offset, keeping its source frames.
    after_v1 = sorted((i.timeline_start_frame, i.source_start_frame)
                      for i in timeline.items if i.track == "V1")
    assert after_v1 == sorted((s + offset, src) for s, src in before_v1)
    text = next(i for i in timeline.items if i.track == "T1")
    assert text.timeline_start_frame == 300 + offset
    # The hook itself sits at the head on its own tracks, video and audio.
    cold = [i for i in timeline.items if i.origin == structure.COLD_OPEN_ORIGIN]
    tracks = {i.track for i in cold}
    assert {"V2", "A5", "TC"} <= tracks
    video = [i for i in cold if i.track == "V2"]
    assert video[0].timeline_start_frame == 0
    assert sum(i.duration_frames for i in video) == offset - 18
    # The bed starts under the hook.
    assert music.timeline_start_frame == 0
    # Captions for the hook carry karaoke word timings.
    caption = next(i for i in cold if i.track == "TC")
    assert caption.text.words and caption.text.style.animation == "karaoke"

    # A transcript rebuild keeps the offset; removing the cold open restores.
    rebuild_primary_tracks(timeline, next(iter(timeline.sources)))
    assert min(i.timeline_start_frame for i in timeline.items if i.track == "V1") == offset
    structure.remove_cold_open(timeline)
    assert timeline.cold_open_frames == 0 and timeline.program_offset_frames == 0
    assert not [i for i in timeline.items if i.origin == structure.COLD_OPEN_ORIGIN]
    assert text.timeline_start_frame == 300


def test_an_intro_card_sits_after_the_cold_open_and_removes_cleanly():
    timeline = _timeline()
    program = build_program(timeline)
    hook = structure.fallback_hook(program)
    structure.apply_cold_open(timeline, program, hook, PresentationSettings())
    cold = timeline.cold_open_frames
    v1_start = min(i.timeline_start_frame for i in timeline.items if i.track == "V1")
    assert v1_start == cold

    apply_intro(timeline, "title_card", title="THE HOUSE")
    card = 90
    assert timeline.program_offset_frames == cold + card
    assert min(i.timeline_start_frame for i in timeline.items if i.track == "V1") == cold + card
    intro_text = next(i for i in timeline.items if i.origin == "intro")
    assert intro_text.timeline_start_frame >= cold
    hook_video = next(i for i in timeline.items if i.origin == structure.COLD_OPEN_ORIGIN
                      and i.track == "V2")
    assert hook_video.timeline_start_frame == 0

    remove_intro(timeline)
    assert timeline.program_offset_frames == cold == timeline.cold_open_frames
    assert min(i.timeline_start_frame for i in timeline.items if i.track == "V1") == cold


def test_a_hook_in_the_opening_seconds_is_not_cut_twice():
    timeline = _timeline()
    program = build_program(timeline)
    early = structure.Hook(start_s=3.0, end_s=6.0, text="x")
    assert structure.apply_cold_open(timeline, program, early, PresentationSettings()) is None
    assert timeline.cold_open_frames == 0
    off = structure.Hook(start_s=20.0, end_s=24.0, text="x")
    assert structure.apply_cold_open(timeline, program, off,
                                     PresentationSettings(cold_open=False)) is None


# --- chapters and the listing ------------------------------------------------------------

def test_chapters_start_at_zero_and_keep_ten_seconds_apart():
    topics = [Topic(start_s=0.0, end_s=8.0, topic="intro"),
              Topic(start_s=8.0, end_s=30.0, topic="the house", heading="The House"),
              Topic(start_s=30.0, end_s=95.0, topic="the night"),
              Topic(start_s=95.0, end_s=120.0, topic="outro")]
    chapters = structure.chapters_for(topics)
    assert chapters[0] == ("0:00", "intro")
    assert ("0:30", "the night") in chapters
    assert ("1:35", "outro") in chapters
    assert not any(name == "The House" for _, name in chapters)     # 8 s in: too close
    shifted = structure.chapters_for(topics, offset_s=5.0)
    assert ("0:35", "the night") in shifted
    assert structure._stamp(3725) == "1:02:05"


def test_the_listing_is_written_with_and_without_a_model(tmp_path):
    program = build_program(_timeline())
    topics = _topics(program)

    async def ask(system, user, schema=None):
        assert schema["name"] == "metadata"
        return json.dumps({"titles": ["Purane Ghar Ka Raaz", "The House That Knocks", ""],
                           "description": "Ek parivaar, ek ghar, ek raat.",
                           "tags": ["horror", "bhoot", "hindi horror story"]})

    payload = asyncio.run(structure.write_metadata(tmp_path, program, topics, ask, "horror",
                                                   "THE KNOCK"))
    assert payload["titles"][0] == "THE KNOCK" and "Purane Ghar Ka Raaz" in payload["titles"]
    assert payload["description"].startswith("Ek parivaar")
    assert "0:00" in payload["description"]
    assert (tmp_path / "metadata.json").exists() and (tmp_path / "chapters.txt").exists()
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["tags"][0] == "horror"

    plain = asyncio.run(structure.write_metadata(tmp_path, program, topics, None, "horror", None))
    assert plain["description"] and plain["chapters"][0]["at"] == "0:00"
    assert "bhoot" not in plain["tags"] or "horror" in plain["tags"]


def test_without_a_model_topics_still_come_from_the_pauses():
    from backend.presentation.shotplan import fallback_topics
    program = build_program(_timeline())
    topics = fallback_topics(program)
    assert 1 <= len(topics) <= 3
    assert topics[0].start_s == 0.0 and topics[-1].end_s == program.duration_s
    assert all(t.origin == "fallback" and t.topic for t in topics)
    for current, following in zip(topics, topics[1:]):
        assert current.end_s == following.start_s
        assert current.end_s - current.start_s >= 15.0
