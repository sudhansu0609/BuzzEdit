"""Moods per topic: tagging, the recipes, the hits, and the new treatments."""

import asyncio
import json

import pytest

from backend.presentation import mood, sound
from backend.presentation.models import PresentationSettings, Topic
from backend.presentation.program import build_program
from backend.render.atmosphere import build_effect
from backend.render.compiler import FilterGraphCompiler
from backend.timeline import build_timeline_from_transcript
from backend.timeline.schema import AtmosphereEffect


TEXT = ("raat ke do baje the aur ghar mein sab so rahe the . "
        "maine ek awaaz suni jaise koi darwaza khol raha ho . "
        "achanak darwaza khula aur andar koi khada tha . "
        "uske baad hum us ghar mein kabhi nahi gaye .")


def _timeline(loud_at=None):
    words = [{"word": t, "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i, t in enumerate(TEXT.split())]
    duration = len(words) * 0.5 + 1.0
    envelope = None
    if loud_at is not None:
        frames = int(duration * 50)
        db = [-30] * frames
        for i in range(int(loud_at * 50), int(loud_at * 50) + 25):
            db[i] = -6
        envelope = {"rate": 50.0, "db": db}
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", duration, words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, duration)], pause_padding_seconds=0.0,
        energy_envelope=envelope)


def _topics(program):
    return [Topic(start_s=0.0, end_s=5.5, topic="the night"),
            Topic(start_s=5.5, end_s=11.0, topic="the sound"),
            Topic(start_s=11.0, end_s=16.0, topic="the door"),
            Topic(start_s=16.0, end_s=program.duration_s, topic="after")]


# --- tagging ------------------------------------------------------------------

def test_without_a_model_horror_follows_the_arc_and_others_stay_calm():
    program = build_program(_timeline())
    topics = asyncio.run(mood.tag_moods(_topics(program), program, None, "horror"))
    assert [t.mood for t in topics] == ["calm", "build", "climax", "aftermath"]
    plain = asyncio.run(mood.tag_moods(_topics(program), program, None, "science_education"))
    assert [t.mood for t in plain] == ["calm", "calm", "calm", "hopeful"]
    assert all(t.key_moment_s is not None for t in topics)


def test_the_models_moods_and_key_moments_are_taken_when_valid():
    program = build_program(_timeline())

    async def ask(system, user, schema=None):
        assert schema["name"] == "moods"
        return json.dumps({"topics": [
            {"index": 0, "mood": "build", "key_moment_s": 2.0},
            {"index": 1, "mood": "tense", "key_moment_s": 8.0},
            {"index": 2, "mood": "climax", "key_moment_s": 99.0},     # outside → fallback
            {"index": 3, "mood": "nonsense", "key_moment_s": 17.0},   # ignored
        ]})

    topics = asyncio.run(mood.tag_moods(_topics(program), program, ask, "horror",
                                        overrides={3: "aftermath"}))
    assert [t.mood for t in topics] == ["build", "tense", "climax", "aftermath"]
    assert topics[1].key_moment_s == pytest.approx(8.0)
    # The third topic's key moment fell back to a turn word: "achanak" sits
    # right on the topic boundary (where the transition plays) and is passed
    # over; "aur" at 13.5s is the first one clear of it.
    aur = next(w.tl_start_s for w in program.words if w.text == "aur" and w.tl_start_s > 11.0)
    assert topics[2].key_moment_s == pytest.approx(aur)


def test_the_key_moment_falls_back_to_the_loudest_word():
    # The last topic has no turn words, so the loudest word carries the hit.
    program = build_program(_timeline(loud_at=18.0))
    topic = Topic(start_s=16.0, end_s=program.duration_s, topic="x")
    assert mood._key_moment(topic, program) == pytest.approx(18.0, abs=0.3)
    # With a flat envelope the hit lands a couple of seconds in.
    flat = build_program(_timeline())
    assert mood._key_moment(topic, flat) == pytest.approx(18.0, abs=0.01)


# --- recipes -------------------------------------------------------------------

def test_horror_gets_layers_hits_loops_and_transitions():
    timeline = _timeline()
    program = build_program(timeline)
    topics = _topics(program)
    for topic, name in zip(topics, ["build", "tense", "climax", "aftermath"]):
        topic.mood = name
        topic.key_moment_s = topic.start_s + 2.0
    plan = mood.apply_moods(timeline, topics, program, PresentationSettings(), "horror")

    layers = [i for i in timeline.items if i.origin == mood.MOOD_ORIGIN and i.label.startswith("mood:")]
    assert [i.label for i in layers] == ["mood: build", "mood: tense", "mood: climax", "mood: aftermath"]
    assert plan.layers == 4 and plan.hits == 1
    climax = layers[2]
    assert climax.color.saturation < 0.85 and climax.color.vignette > 0.3
    assert climax.transform.scale_end > climax.transform.scale
    assert {e.type for e in climax.atmosphere} == {"lightning", "flicker"}
    aftermath = layers[3]
    assert aftermath.transform.scale > aftermath.transform.scale_end     # a pull-back

    hits = [i for i in timeline.items if i.origin == mood.MOOD_ORIGIN and i.label.startswith("hit:")]
    assert {i.label for i in hits} == {"hit: flash", "hit: shake", "hit: glitch"}
    flash = next(i for i in hits if i.label == "hit: flash")
    assert flash.timeline_start_frame == 30 * 13 and flash.duration_frames <= 3
    assert dict(plan.sfx)[13.0] == "thunder"
    assert any(tag == "riser" for _, tag in plan.sfx)
    assert [l[2] for l in plan.loops] == ["heartbeat", "heartbeat"]
    assert plan.transitions == {1: ("fadeblack", 0.5), 2: ("fadeblack", 0.6),
                                3: ("fadeblack", 0.9)}

    # A re-run replaces the layers instead of stacking them.
    mood.apply_moods(timeline, topics, program, PresentationSettings(), "horror")
    assert len([i for i in timeline.items if i.origin == mood.MOOD_ORIGIN]) == len(layers) + len(hits)


def test_other_genres_get_half_strength_grades_and_no_hits():
    timeline = _timeline()
    program = build_program(timeline)
    topics = _topics(program)
    for topic, name in zip(topics, ["calm", "tense", "climax", "hopeful"]):
        topic.mood = name
        topic.key_moment_s = topic.start_s + 2.0
    plan = mood.apply_moods(timeline, topics, program, PresentationSettings(), "science_education")
    assert plan.hits == 0 and plan.loops == [] and plan.sfx == []
    layers = [i for i in timeline.items if i.origin == mood.MOOD_ORIGIN]
    assert all(i.label.startswith("mood:") for i in layers)
    tense = next(i for i in layers if i.label == "mood: tense")
    assert tense.color.saturation == pytest.approx(1.0 - (1.0 - 0.84) * 0.5)
    assert not any(e.type == "lightning" for i in layers for e in i.atmosphere)
    hopeful = next(i for i in layers if i.label == "mood: hopeful")
    assert any(e.type == "sunlight" for e in hopeful.atmosphere)

    off = mood.apply_moods(timeline, topics, program, PresentationSettings(mood_strength=0),
                           "horror")
    assert off.layers == 0 and not [i for i in timeline.items if i.origin == mood.MOOD_ORIGIN]


def test_loops_reach_the_sound_lanes(monkeypatch, tmp_path):
    monkeypatch.setattr(sound, "find_sfx", lambda tag, rng, library=None: None)
    monkeypatch.setattr(sound, "synth_path",
                        lambda kind: (tmp_path / f"{kind}_x.wav") if (tmp_path / f"{kind}_x.wav").write_bytes(b"x" * 2000) or True else None)
    monkeypatch.setattr(sound, "SYNTH_DIR", tmp_path)
    monkeypatch.setattr(sound, "find_music", lambda *a, **k: None)
    monkeypatch.setattr(sound, "find_ambience", lambda *a, **k: None)
    timeline = _timeline()
    program = build_program(timeline)
    # Placement mechanics, not sourcing policy: sfx_source="synth" keeps this
    # test on the old always-available FFmpeg fallback now that "auto" (the
    # default) no longer reaches for it — see test_sound.py's
    # test_sfx_source_auto_never_synthesises for that policy itself.
    settings = PresentationSettings(sfx_source="synth")
    plan = sound.plan_sound(timeline, program, settings, "horror", 1,
                            loops=[(11.0, 16.0, "heartbeat", 0.35), (1.0, 2.0, "heartbeat", 0.3)])
    assert len(plan.loops) == 1 and plan.loops[0].tag == "heartbeat"
    counts = sound.apply_sound(timeline, plan, settings)
    loop_items = [i for i in timeline.items if (i.label or "").startswith("SFX loop")]
    assert len(loop_items) == 1 and loop_items[0].loop and loop_items[0].track == "A3"
    assert loop_items[0].timeline_start_frame == 330


# --- render primitives -----------------------------------------------------------

@pytest.mark.parametrize("kind,needle", [
    ("flash", "drawbox=color=white@"),
    ("shake", "crop=w=iw-"),
    ("glitch", "rgbashift=rh=-"),
    ("vhs", "geq=lum="),
    ("flicker", "eq=brightness="),
])
def test_the_treatments_build_their_filters(kind, needle):
    statements = build_effect(AtmosphereEffect(type=kind, intensity=0.6), "[in]", "[out]",
                              1920, 1080, 30.0, 10.0, 0)
    assert len(statements) == 1
    assert statements[0].startswith("[in]") and statements[0].endswith("[out]")
    assert needle in statements[0]


def test_a_hit_compiles_as_a_windowed_adjustment():
    timeline = _timeline()
    program = build_program(timeline)
    topics = _topics(program)
    for topic, name in zip(topics, ["calm", "calm", "climax", "calm"]):
        topic.mood = name
        topic.key_moment_s = topic.start_s + 2.0
    mood.apply_moods(timeline, topics, program, PresentationSettings(), "horror")
    _, graph, _, _ = FilterGraphCompiler(timeline).compile()
    assert "drawbox=color=#FFFFFF@" in graph
    assert "enable='between(t,13.000,13.0" in graph
    assert "rgbashift=" in graph and "crop=w=iw-" in graph


def test_mood_hits_off_keeps_the_grade_and_push_but_no_hits():
    """Story channels: the calm mood layers stay; flash/shake hits, their
    stinger/thunder and the riser do not."""
    timeline = _timeline()
    program = build_program(timeline)
    topics = _topics(program)
    for topic, name in zip(topics, ["build", "tense", "climax", "aftermath"]):
        topic.mood = name
        topic.key_moment_s = topic.start_s + 2.0
    plan = mood.apply_moods(timeline, topics, program, PresentationSettings(mood_hits=False), "horror")

    assert plan.layers == 4 and plan.hits == 0
    assert not [i for i in timeline.items if i.origin == mood.MOOD_ORIGIN and i.label.startswith("hit:")]
    assert not [tag for _, tag in plan.sfx if tag in ("thunder", "stinger", "riser")]


def test_video_share_may_reach_every_cutaway():
    assert PresentationSettings(video_broll_share=1.0).video_broll_share == 1.0
    assert PresentationSettings(video_broll_share=3.0).video_broll_share == 1.0
