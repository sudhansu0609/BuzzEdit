"""Scene sounds (presentation/sound_design.py): the sound designer's pass, its
keyword fallback, and the script's [sfx: ...] cues that feed the same path."""

import asyncio
import json

from backend.presentation import script as script_stage
from backend.presentation import sound_design
from backend.presentation.models import Beat, PresentationSettings, Program, ProgramWord


def _program(words, duration):
    return Program(duration_s=duration, fps=30.0, words=[
        ProgramWord(text=text, tl_start_s=at, tl_end_s=at + 0.4, source_start_frame=0)
        for text, at in words])


def test_planner_anchors_cues_on_their_words_and_drops_unknown_anchors():
    said = "woh jungle ke raaste par chal raha tha tabhi baarish shuru ho gayi".split()
    program = _program([(w, i * 0.5) for i, w in enumerate(said)], 30.0)
    answer = json.dumps({"cues": [
        {"anchor": "chal raha tha", "effect": "footsteps on wet leaves", "kind": "accent",
         "level": "subtle", "seconds": 3},
        {"anchor": "Baarish shuru,", "effect": "rain", "kind": "bed", "level": "normal",
         "seconds": 90},
        {"anchor": "words that were never said", "effect": "door", "kind": "accent",
         "level": "normal", "seconds": 1},
        {"anchor": "ho gayi", "effect": "sad piano music", "kind": "bed", "level": "normal",
         "seconds": 10},
    ]})
    asked = []

    async def ask(system, user, schema=None):
        asked.append((system, user))
        return answer

    cues = asyncio.run(sound_design.plan_diegetic(program, [], ask, "horror"))
    assert [(c.effect, c.at_s, c.kind) for c in cues] == [
        ("footsteps on wet leaves", 2.5, "accent"), ("rain", 4.5, "bed")]
    assert cues[0].level == "subtle" and cues[0].seconds == 3.0
    assert cues[1].seconds == sound_design.BED_MAX_S
    assert all(c.source == "planner" for c in cues)
    system, user = asked[0]
    assert system == sound_design.SOUND_SYSTEM
    assert "[0:00]" in user and "[0:05]" in user and "On screen:" in user


def test_the_keyword_fallback_fires_without_a_model():
    beats = [Beat(start_s=3.0, end_s=8.0, image_prompt="A ranger walking through a misty trail"),
             Beat(start_s=20.0, end_s=26.0, image_prompt="An old forest jeep on a muddy road"),
             Beat(start_s=45.0, end_s=50.0, video_prompt="Heavy rain over the canopy")]
    cues = asyncio.run(sound_design.plan_diegetic(_program([("haan", 0.0)], 60.0), beats, None))
    got = [(c.effect, c.kind, c.at_s, c.seconds) for c in cues]
    assert got == [("footsteps", "accent", 3.0, None), ("jeep engine", "bed", 20.0, 8.0),
                   ("rain", "bed", 45.0, None)]
    assert all(c.level == "subtle" and c.source == "keyword" for c in cues)


def test_no_usable_answer_at_all_falls_back_to_keywords():
    beats = [Beat(start_s=3.0, end_s=8.0, image_prompt="Someone running in the dark")]

    async def ask(system, user, schema=None):
        return "sorry"

    cues = asyncio.run(sound_design.plan_diegetic(_program([("haan", 0.0)], 30.0), beats, ask))
    assert [(c.effect, c.source) for c in cues] == [("running footsteps", "keyword")]


def test_script_cues_win_over_the_planners_within_two_seconds():
    script = [sound_design.DiegeticCue(10.0, "door slam", source="script")]
    planned = [sound_design.DiegeticCue(11.5, "door"), sound_design.DiegeticCue(20.0, "dog barking")]
    merged = sound_design.merge_cues(script, planned)
    assert [(c.at_s, c.effect) for c in merged] == [(10.0, "door slam"), (20.0, "dog barking")]


def test_script_sfx_directives_split_into_tags_and_scene_sounds():
    ctx = script_stage.ScriptContext(
        paragraphs=[script_stage.ScriptParagraph(index=0, tl_start_s=0.0, tl_end_s=20.0, text="x")],
        directives=[
            script_stage.ScriptDirective(kind="sfx", arg="whoosh", tl_at_s=2.0),
            script_stage.ScriptDirective(
                kind="sfx", arg="footsteps on gravel | kind=accent | level=subtle | seconds=3",
                tl_at_s=5.0),
            script_stage.ScriptDirective(kind="sfx", arg="rain on the tin roof | kind=bed",
                                         tl_at_s=9.0),
        ])
    out = script_stage.directive_beats(ctx, _program([("haan", 0.0)], 20.0), PresentationSettings())
    assert out["sfx"] == [(2.0, "whoosh")]
    first, second = out["diegetic"]
    assert isinstance(first, sound_design.DiegeticCue)
    assert (first.at_s, first.effect, first.kind, first.level, first.seconds, first.source) == (
        5.0, "footsteps on gravel", "accent", "subtle", 3.0, "script")
    assert (second.effect, second.kind, second.level, second.seconds) == (
        "rain on the tin roof", "bed", "normal", None)
