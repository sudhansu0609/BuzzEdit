"""Emoji reactions (presentation/emoji_fx.py): vocabulary, rules, rendering,
placement and the director wiring."""

import pytest

from presentation import emoji_fx
from presentation.emoji_fx import (EMOJI_ORIGIN, HUNDRED_EMOJI, MIN_GAP_S, PERCENT_EMOJI,
                                   match_words, plan_reactions)
from presentation.models import PresentationSettings, Program, ProgramSegment, ProgramWord

CANVAS = (1920, 1080)


def _w(text: str, t: float, z: float = 1.0) -> ProgramWord:
    return ProgramWord(text=text, tl_start_s=t, tl_end_s=t + 0.3, source_start_frame=int(t * 30),
                       emphasis_z=z)


def _program(words, duration: float = 120.0) -> Program:
    return Program(duration_s=duration, fps=30.0, words=list(words),
                   segments=[ProgramSegment(item_id="v1", tl_start_s=0.0, tl_end_s=duration,
                                            source_start_frame=0, source_end_frame=int(duration * 30))])


def _settings(**kw) -> PresentationSettings:
    kw.setdefault("emoji_reactions", "on")
    return PresentationSettings(**kw)


def _emojis(words):
    return [(m["word"], m["emoji"]) for m in match_words(words)]


# --- vocabulary ----------------------------------------------------------------

def test_vocabulary_matches_devanagari_romanised_english_and_percent():
    words = [_w(t, i) for i, t in enumerate(
        ["मेरे", "पैसे", "paisa,", "Money!", "ज़्यादा", "ज्यादा", "khatra", "Danger",
         "50%", "100", "percent", "१००", "प्रतिशत", "30", "pratishat", "kabhi", "nahi"])]
    got = [(m["word"], m["emoji"]) for m in match_words(words)]
    by_word = dict(got)
    assert by_word["पैसे"] == by_word["paisa,"] == by_word["Money!"] == "💰"
    assert by_word["ज़्यादा"] == by_word["ज्यादा"] == "📈"      # nukta folded
    assert by_word["khatra"] == by_word["Danger"] == "⚠️"
    assert ("50%", PERCENT_EMOJI) in got and ("30", PERCENT_EMOJI) in got
    assert ("100", HUNDRED_EMOJI) in got and ("१००", HUNDRED_EMOJI) in got
    assert "मेरे" not in by_word
    assert [w for w, _ in got].count("nahi") == 0, "a phrase (kabhi nahi) matches once"


# --- placement rules ---------------------------------------------------------------

def test_placement_rules():
    dense = _program([_w("zabardast", i * 2.0) for i in range(60)])
    reactions = plan_reactions(dense, _settings(), [], {}, "finance", CANVAS, seed=7)
    starts = [r.start_s for r in reactions]
    assert reactions and all(sum(1 for x in starts if s <= x < s + 60.0) <= 6 for s in starts)
    assert all(b.start_s - a.end_s >= MIN_GAP_S - 1e-6 for a, b in zip(reactions, reactions[1:]))
    again = plan_reactions(dense, _settings(), [], {}, "finance", CANVAS, seed=7)
    assert [(r.start_s, r.x, r.y) for r in again] == [(r.start_s, r.x, r.y) for r in reactions]

    # Busy window skipped, B-roll -> small corner badge, caption band and face side respected.
    program = _program([_w("paisa", 5.0), _w("shocking", 20.0), _w("danger", 40.0)])
    by_word = {r.word: r for r in plan_reactions(program, _settings(), [(4.0, 8.0)],
                                                 {"v1": (0.3, 0.6)}, "finance", CANVAS,
                                                 broll_windows=[(19.0, 24.0)])}
    assert "paisa" not in by_word
    assert by_word["shocking"].size < by_word["danger"].size and by_word["shocking"].x > 0.7
    top, _ = emoji_fx.caption_band(_settings())
    assert by_word["danger"].y + by_word["danger"].size <= top
    assert by_word["danger"].x < 0.3 - 0.22, "on the side away from the face"

    # Genre auto rule.
    auto = PresentationSettings()
    assert auto.emoji_reactions == "auto"
    assert plan_reactions(program, auto, [], {}, "comedy", CANVAS)
    assert plan_reactions(program, auto, [], {}, "horror", CANVAS) == []
    assert plan_reactions(program, _settings(emoji_reactions="on"), [], {}, "horror", CANVAS)


# --- the director ----------------------------------------------------------------------

async def _none():
    return None


async def _empty_assets(*args, **kwargs):
    return [], []


@pytest.mark.asyncio
async def test_director_adds_reactions_and_removes_them_on_rerun(tmp_path, monkeypatch):
    from presentation import director
    from store.project_store import ProjectStore
    from timeline import build_timeline_from_transcript
    from timeline.schema import Timeline as TL

    monkeypatch.setattr(director, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(director, "_llm_asker", lambda *a, **k: _none())
    monkeypatch.setattr(director, "_comfyui_online", lambda: False)
    monkeypatch.setattr(director.assets_stage, "generate_assets", _empty_assets)

    vocab = ["paisa", "shocking", "danger", "zabardast"]
    words, clock = [], 0.0
    for i in range(120):
        text = vocab[(i // 20) % 4] if i % 20 == 5 else f"shabd{i}"
        words.append({"word": text, "start": round(clock, 3), "end": round(clock + 0.32, 3),
                      "probability": 0.99})
        clock += 0.4
    timeline = build_timeline_from_transcript(
        source_path="C:/media/talk.mp4", duration_seconds=clock + 1.0, transcript_words=words,
        fps_num=30, fps_den=1, speech_regions=[(0.0, clock + 1.0)], pause_padding_seconds=0.0)
    store = ProjectStore(base_dir=str(tmp_path))
    store.save_project("p1", {"id": "p1", "name": "Test", "source_video": "C:/media/talk.mp4",
                              "timeline": timeline.model_dump(), "settings": {}})

    base = dict(render=False, thumbnail=False, face_zoom=False, genre="finance")
    report = await director.run_presentation_pass("p1", PresentationSettings(emoji_reactions="on", **base))
    assert report.emoji_reactions and report.emoji_reactions > 0
    saved = TL(**store.get_project("p1")["timeline"])
    first = [i for i in saved.items if i.origin == EMOJI_ORIGIN]
    assert first

    report = await director.run_presentation_pass("p1", PresentationSettings(emoji_reactions="on", **base))
    saved = TL(**store.get_project("p1")["timeline"])
    assert len([i for i in saved.items if i.origin == EMOJI_ORIGIN]) == len(first)

    report = await director.run_presentation_pass("p1", PresentationSettings(emoji_reactions="off", **base))
    assert report.emoji_reactions == 0
    saved = TL(**store.get_project("p1")["timeline"])
    assert not [i for i in saved.items if i.origin == EMOJI_ORIGIN]
