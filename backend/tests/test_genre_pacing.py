"""Genre pacing: horror keeps its pauses, unless someone chose a pacing.

The default pacing trims any pause over 0.40s to a ~0.24s beat. Half the pauses
in the creator's own horror cut were longer than that, so a horror recording
gets its genre's pacing — but only when neither a channel nor the user set one.
"""

import asyncio

from asr import auto_edit
from asr.auto_edit import apply_genre_pacing, pacing_kwargs
from presentation import genre as g
from timeline import build_timeline_from_transcript

HORROR = (1.2, 0.45)


def _run(coro):
    return asyncio.run(coro)


def _words(text: str, seconds_each: float = 0.5):
    return [{"word": w, "start": i * seconds_each, "end": (i + 1) * seconds_each}
            for i, w in enumerate(text.split())]


def test_horror_has_its_own_pacing_and_other_genres_keep_the_defaults():
    assert g.pacing_for("horror") == HORROR
    assert g.pacing_for("scary") == HORROR          # an alias normalises to horror
    for other in ("vlog", "general", None, ""):
        assert g.pacing_for(other) is None


def test_a_named_genre_sets_the_pacing_and_says_where_it_came_from():
    settings = {"genre": "horror"}
    assert _run(apply_genre_pacing(settings, [])) == "horror"
    assert (settings["max_pause_seconds"], settings["pause_padding_seconds"]) == HORROR
    assert settings["pacing_source"] == "genre:horror"
    assert pacing_kwargs(settings) == {"max_pause_seconds": 1.2, "pause_padding_seconds": 0.45}


def test_a_channel_pacing_wins_over_the_genre():
    # Life3Baje's profile sends its own measured pacing; the genre must not override it.
    settings = {"genre": "horror", "max_pause_seconds": 1.0, "pause_padding_seconds": 0.35}
    assert _run(apply_genre_pacing(settings, [])) is None
    assert (settings["max_pause_seconds"], settings["pause_padding_seconds"]) == (1.0, 0.35)
    assert "pacing_source" not in settings


def test_a_user_who_moved_the_slider_keeps_their_choice():
    settings = {"genre": "horror", "max_pause_seconds": 0.8, "pause_padding_seconds": 0.45,
                "pacing_source": "genre:horror"}
    assert _run(apply_genre_pacing(settings, [])) is None
    assert settings["max_pause_seconds"] == 0.8


def test_a_genre_that_no_longer_applies_hands_back_the_defaults():
    settings = {"genre": "vlog", "max_pause_seconds": 1.2, "pause_padding_seconds": 0.45,
                "pacing_source": "genre:horror"}
    assert _run(apply_genre_pacing(settings, [])) == "vlog"
    assert "max_pause_seconds" not in settings
    assert "pause_padding_seconds" not in settings
    assert "pacing_source" not in settings


def test_without_a_genre_or_a_model_the_keywords_decide():
    settings = {}
    words = _words("ye bhoot ki kahani hai bhoot sach mein tha aur bhoot ne dekha")
    assert _run(apply_genre_pacing(settings, words)) == "horror"
    assert settings["pacing_source"] == "genre:horror"


def test_the_model_reads_a_horror_story_the_keywords_miss(monkeypatch):
    # The Raat3Baje story never says "bhoot" — keyword voting called it "cooking".
    story = _words("raat ke sawa do baje ek security guard akela corridor mein khada hai "
                   "uske peeche metal ka ek darwaza hai aur kadmon ki awaaz aa rahi hai")
    asked = []

    async def fake_ask(system_prompt, user_prompt, schema):
        asked.append(user_prompt)
        return '{"genre": "horror"}'

    from llm.client import lm_studio_client
    monkeypatch.setattr(lm_studio_client, "ask_with_schema", fake_ask)

    no_model = {}
    assert _run(apply_genre_pacing(no_model, story, model_ready=False)) != "horror"
    assert "max_pause_seconds" not in no_model
    assert not asked, "no model may be started when the planner's own was not up"

    settings = {}
    assert _run(apply_genre_pacing(settings, story, model_ready=True)) == "horror"
    assert (settings["max_pause_seconds"], settings["pause_padding_seconds"]) == HORROR
    assert asked and "security guard" in asked[0]


def test_plan_auto_edit_reports_the_genre_pacing(monkeypatch):
    async def fake_refine(words, **kwargs):
        refined = [dict(w) for w in words]
        refined[0]["_edit_report"] = {"used_llm": False}
        return refined

    monkeypatch.setattr(auto_edit, "refine_disfluencies", fake_refine)
    settings = {"genre": "horror"}
    plan = _run(auto_edit.plan_auto_edit(_words("ek do teen"), None, use_llm=False,
                                         settings=settings))
    assert plan.report["pacing"] == {"genre": "horror", "max_pause_seconds": 1.2,
                                     "pause_padding_seconds": 0.45}


def _v1(tl):
    return sorted((i for i in tl.items if i.track == "V1"), key=lambda i: i.timeline_start_frame)


def test_a_one_second_dramatic_pause_survives_horror_pacing_but_not_the_default():
    words = [{"word": "darwaza", "start": 0.0, "end": 0.5},
             {"word": "khula", "start": 1.5, "end": 2.0}]      # a 1.0s beat between them

    default = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1)
    items = _v1(default)
    assert len(items) == 2, "the default trims a 1.0s pause"
    trimmed = items[1].source_start_frame - items[0].source_end_frame
    assert trimmed > 0, "the default removed part of the beat between the two words"
    assert items[0].source_end_frame < 30 and items[1].source_start_frame > 30

    horror = build_timeline_from_transcript("v.mp4", 5.0, words, 30, 1,
                                            **pacing_kwargs({"genre": "horror",
                                                             "max_pause_seconds": 1.2,
                                                             "pause_padding_seconds": 0.45}))
    items = _v1(horror)
    assert len(items) == 1, "horror pacing keeps the 1.0s pause whole"
    assert items[0].source_start_frame == 0
    assert items[0].source_end_frame >= 60
