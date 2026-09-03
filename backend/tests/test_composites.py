"""Composites, caption extras, per-act music and the style profile hook."""

import asyncio
import json
from pathlib import Path

import pytest

from backend.presentation import captions, composite, sound
from backend.presentation.models import Asset, Beat, PresentationSettings, Topic
from backend.presentation.placement import BROLL_ORIGIN, place_broll
from backend.presentation.program import build_program
from backend.render import ass
from backend.timeline import build_timeline_from_transcript, clip_ops, generate_captions
from backend.timeline.schema import SourceFile


def _timeline(seconds: float = 40.0, loud_at=None):
    # Letters only: a digit anywhere in a word makes it a "number" for emphasis.
    words = [{"word": ("25" if i == 20 else f"w{chr(97 + i % 26)}{chr(97 + i // 26)}"),
              "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i in range(int(seconds * 2) - 1)]
    envelope = None
    if loud_at is not None:
        frames = int(seconds * 50)
        db = [-30] * frames
        for i in range(int(loud_at * 50), int(loud_at * 50) + 20):
            db[i] = -4
        envelope = {"rate": 50.0, "db": db}
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", seconds, words, fps_num=30, fps_den=1, width=1920, height=1080,
        speech_regions=[(0.0, seconds)], pause_padding_seconds=0.0, energy_envelope=envelope)


def _still(timeline, sid, path="C:/b.png"):
    timeline.sources[sid] = SourceFile(id=sid, path=path, duration_seconds=3, width=1920,
                                       height=1080, has_audio=False, kind="image")


# --- picture-in-picture --------------------------------------------------------------

def test_pip_puts_the_speaker_in_the_corner_over_long_cutaways_only():
    timeline = _timeline()
    program = build_program(timeline)
    _still(timeline, "b")
    clip_ops.add_media_item(timeline, "b", "V3", 60, 0, 150, origin="broll")     # 5 s
    clip_ops.add_media_item(timeline, "b", "V3", 600, 0, 60, origin="broll")     # 2 s
    count = composite.place_pip(timeline, program, PresentationSettings(), "science_education")
    assert count == 1
    pips = [i for i in timeline.items if i.origin == composite.COMPOSITE_ORIGIN]
    assert pips and all(i.track == composite.COMPOSITE_TRACK and i.mute for i in pips)
    assert pips[0].timeline_start_frame == 60
    assert sum(i.duration_frames for i in pips) == 150
    assert pips[0].transform.scale == pytest.approx(composite.PIP_SCALE)
    # The corner clip shows the same source frames as the programme beneath.
    v1 = next(i for i in timeline.items if i.track == "V1"
              and i.timeline_start_frame <= 60 < i.timeline_end_frame)
    assert pips[0].source_start_frame == v1.source_start_frame + (60 - v1.timeline_start_frame)
    # Horror keeps its pictures alone; "on" forces it anywhere.
    assert composite.place_pip(timeline, program, PresentationSettings(), "horror") == 0
    assert composite.place_pip(timeline, program, PresentationSettings(pip="on"), "horror") == 1


# --- split screen -----------------------------------------------------------------------

def test_split_beats_expand_into_a_pair_and_place_side_by_side(tmp_path):
    timeline = _timeline()
    program = build_program(timeline)
    split = Beat(id="s1", kind="split", start_s=5.0, end_s=11.0, topic="phones",
                 image_prompt="an iphone", planned_duration_s=3.0,
                 data={"left": "an iphone on a desk", "right": "a pixel phone on a desk",
                       "labels": ["iPhone", "Pixel"]})
    pair = composite.expand_splits([split, Beat(id="x", kind="popup", start_s=20, end_s=24,
                                                topic="t", popup_text="hi")])
    assert [b.kind for b in pair] == ["broll_image", "broll_image", "popup"]
    assert pair[0].id == "s1_l" and pair[1].id == "s1_r"
    assert pair[1].image_prompt == "a pixel phone on a desk" and pair[1].data["pair"] == "s1"

    left, right = tmp_path / "l.png", tmp_path / "r.png"
    left.write_bytes(b"x")
    right.write_bytes(b"x")
    assets = [Asset(beat_id="s1_l", kind="image", path=str(left), width=1920, height=1080),
              Asset(beat_id="s1_r", kind="image", path=str(right), width=1920, height=1080)]
    placed = place_broll(timeline, pair, assets, program, PresentationSettings())
    assert placed == 1
    halves = [i for i in timeline.items if i.origin == BROLL_ORIGIN]
    assert {i.track for i in halves} == {"V3", "V4"}
    assert {i.transform.pos_x for i in halves} == {-1.0, 1.0}
    assert all(i.transform.crop_left == 0.25 for i in halves)
    assert halves[0].timeline_start_frame == halves[1].timeline_start_frame
    labels = [i for i in timeline.items if i.origin == "split"]
    assert sorted(i.text.content for i in labels) == ["Pixel", "iPhone"]

    # A half whose partner never generated goes up on its own, full frame.
    placed = place_broll(timeline, pair, assets[:1], program, PresentationSettings())
    assert placed == 1
    assert not [i for i in timeline.items if i.origin == "split"]


def test_a_split_direction_in_the_script_becomes_a_split_beat():
    from backend.presentation import script as script_stage
    from backend.presentation.models import Program
    ctx = script_stage.ScriptContext(
        paragraphs=[script_stage.ScriptParagraph(index=0, tl_start_s=0, tl_end_s=30, text="x")],
        directives=[script_stage.ScriptDirective(kind="split", arg="an old haveli | a new flat",
                                                 tl_at_s=4.0)])
    program = build_program(_timeline())
    out = script_stage.directive_beats(ctx, program, PresentationSettings())
    beat = out["beats"][0]
    assert beat.kind == "split" and beat.data["right"] == "a new flat"
    assert beat.data["labels"][0] == "an old haveli"


# --- freeze frame -------------------------------------------------------------------------

def test_freeze_frames_land_under_stats_on_the_speaker(tmp_path):
    timeline = _timeline()
    program = build_program(timeline)
    _still(timeline, "b")
    clip_ops.add_media_item(timeline, "b", "V3", 600, 0, 90, origin="broll")
    beats = [Beat(kind="stat_callout", start_s=10.0, end_s=13.5, topic="a", text="25%",
                  data={"value": 25}, origin="entity"),
             Beat(kind="stat_callout", start_s=21.0, end_s=24.0, topic="b", text="40%",
                  data={"value": 40}, origin="entity")]          # under the cutaway
    grabbed = []

    def fake_grab(source, seconds, destination):
        grabbed.append(round(seconds, 2))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"png")
        return destination

    count = composite.place_freezes(timeline, program, beats, PresentationSettings(), tmp_path,
                                    grab=fake_grab)
    assert count == 1 and grabbed == [10.0]
    freeze = next(i for i in timeline.items if (i.label or "").startswith("freeze"))
    assert freeze.track == composite.COMPOSITE_TRACK and freeze.timeline_start_frame == 300
    assert freeze.duration_frames == int(composite.FREEZE_SECONDS * 30)
    assert timeline.sources[freeze.source_id].kind == "image"
    assert freeze.transform.scale_end > 1.0
    assert composite.place_freezes(timeline, program, beats, PresentationSettings(freeze_on_stats=False),
                                   tmp_path, grab=fake_grab) == 0


# --- caption emphasis and the bilingual line -----------------------------------------------

def test_numbers_and_loud_words_are_emphasised_at_most_two_per_card():
    timeline = _timeline(loud_at=3.0)          # the word at 3.0 s ("wga") is shouted
    program = build_program(timeline)
    generate_captions(timeline, preset="karaoke_pop")
    marked = captions.mark_emphasis(timeline, program)
    assert marked >= 2
    cards = [i for i in timeline.items if i.origin == "caption"]
    flagged = [w["text"] for c in cards for w in c.text.words if w.get("emphasis")]
    assert "25" in flagged and "WGA" in flagged
    assert all(sum(1 for w in c.text.words if w.get("emphasis")) <= 2 for c in cards)
    # The engine draws them larger and in the accent colour at rest.
    card = next(c for c in cards if any(w.get("emphasis") for w in c.text.words))
    body = ass.build_events(card.text, 0, 2, "S", 1920, 1080)[0]
    assert "\\fscx118\\fscy118" in body and "\\c&H3AE2FF&\\fscx118" in body


def test_translations_come_from_the_model_and_attach_by_time():
    timeline = _timeline()
    program = build_program(timeline)

    async def ask(system, user, schema=None):
        assert schema["name"] == "translations"
        count = user.count("\n") - 2
        return json.dumps({"lines": [{"index": i, "english": f"line {i} in english words here"}
                                     for i in range(count + 1)]})

    translations = asyncio.run(captions.translate_lines(program, ask))
    assert translations and translations[0][0] == 0.0
    generate_captions(timeline, preset="karaoke_pop")
    attached = captions.attach_second_lines(timeline, translations)
    assert attached > 0
    with_line = [i for i in timeline.items if i.origin == "caption" and i.text.second_line]
    assert with_line
    assert ass.needs_ass(with_line[0].text)
    body = ass.build_events(with_line[0].text, 0, 2, "S", 1920, 1080)[0]
    assert "\\N{\\r\\fs44" in body and "english" in body
    assert asyncio.run(captions.translate_lines(program, None)) == []


def test_transcript_english_is_used_without_a_model():
    timeline = _timeline()
    program = build_program(timeline)
    data = {"transcript": {"segments": [{"start": 0.0, "end": 5.0, "text": "x",
                                         "text_english": "Hello friends"},
                                        {"start": 5.0, "end": 10.0, "text": "y", "text_english": ""}]}}
    translations = captions.translations_from_transcript(data, program, timeline)
    assert translations == [(0.0, pytest.approx(4.97, abs=0.05), "Hello friends")]


# --- music per act ----------------------------------------------------------------------------

def test_the_bed_changes_cue_per_act_with_a_crossfade(monkeypatch, tmp_path):
    made = {}

    def fake_synth(kind):
        path = tmp_path / f"{kind}_x.wav"
        path.write_bytes(b"x" * 2000)
        made[kind] = path
        return path

    monkeypatch.setattr(sound, "synth_path", fake_synth)
    monkeypatch.setattr(sound, "find_music", lambda *a, **k: None)
    monkeypatch.setattr(sound, "find_ambience", lambda *a, **k: None)
    timeline = _timeline(seconds=200.0)
    program = build_program(timeline)
    sections = [(0, 12, "hook"), (12, 70, "setup"), (70, 140, "build"), (140, 175, "climax"),
                (175, 200, "cta")]
    plan = sound.plan_sound(timeline, program, PresentationSettings(sfx=False, ambience=False),
                            "horror", 1, sections=sections)
    acts = [c.act for c in plan.music]
    assert acts == ["setup", "build", "climax", "cta"]        # the short hook folded forward
    assert plan.music[0].start_s == 0.0 and plan.music[-1].end_s == program.duration_s
    assert plan.music[1].start_s == pytest.approx(70 - 1.0)   # overlap for the crossfade
    assert plan.music[0].end_s == pytest.approx(70 + 1.0)
    assert plan.music[1].fade_in_s == sound.MUSIC_CROSSFADE_S
    assert plan.music[2].gain > plan.music[0].gain             # the climax rises
    assert "drone_high" in made and "drone" in made
    # Off: one bed for the whole programme.
    single = sound.plan_sound(timeline, program, PresentationSettings(sfx=False, ambience=False,
                                                                       music_by_act=False),
                              "horror", 1, sections=sections)
    assert len(single.music) == 1 and single.music[0].act == ""


def test_music_sections_merge_same_acts_and_short_ones():
    assert sound.music_sections([(0, 30, "setup"), (30, 60, "setup"), (60, 70, "build"),
                                 (70, 130, "climax")], 130.0) == \
        [(0.0, 70.0, "setup"), (70.0, 130.0, "climax")]
    assert sound.music_sections(None, 50.0) == [(0.0, 50.0, "")]


# --- style profile hook ---------------------------------------------------------------------------

def test_the_style_profile_stage_applies_a_stored_profile(monkeypatch):
    # The director imports `style` by its bare name (backend/ is on sys.path),
    # so that is the module object to patch.
    import style as style_pkg
    from style import profile as profile_mod
    from style.profile import StyleProfile
    from presentation import director
    timeline = _timeline()
    fake = StyleProfile(id="ref", name="Reference Channel", source_path="C:/ref.mp4")
    monkeypatch.setattr(profile_mod, "load", lambda pid: fake if pid == "ref" else None)
    calls = {}

    def fake_apply(tl, prof, source, options):
        calls["motion"] = options.motion
        calls["captions"] = options.captions
        return {"profile": prof.name}

    monkeypatch.setattr(style_pkg, "apply_profile", fake_apply)
    name = director._apply_style_profile(timeline, PresentationSettings(style_profile="ref"), None)
    assert name == "Reference Channel"
    assert calls == {"motion": False, "captions": False}
    assert director._apply_style_profile(timeline, PresentationSettings(style_profile="nope"), None) == ""
