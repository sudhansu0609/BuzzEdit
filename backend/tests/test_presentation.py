"""Tests for the overnight presentation pass.

The danger in this pipeline is the opposite of the auto-edit's. The auto-edit can
only remove things, so its tests guard against cutting too much. This pass *adds*
things over the top of the video, and the failure it must never make is covering
the speaker with something the language model imagined. So most of what follows
checks that a bad plan produces nothing rather than something wrong.
"""

import json

import pytest

from presentation import workflows
from presentation.models import Asset, Beat, PresentationSettings, Program, ProgramWord
from presentation.placement import (BROLL_ORIGIN, POPUP_ORIGIN, broll_windows,
                                    place_broll, place_popups)
from presentation.program import build_program
from presentation.shotplan import (fallback_plan, first_json_object, plan_shots,
                                   validate_plan)
from timeline import build_timeline_from_transcript
from timeline.schema import Timeline


# --- fixtures --------------------------------------------------------------

def _words(count: int = 60, spacing: float = 0.4, loud_at=()):
    """A transcript with a controllable loudness bump."""
    words = []
    clock = 0.0
    for index in range(count):
        words.append({
            "word": f"shabd{index}",
            "start": round(clock, 3),
            "end": round(clock + spacing * 0.8, 3),
            "probability": 0.99,
        })
        clock += spacing
    return words


def _timeline(count: int = 60, with_energy: bool = True) -> Timeline:
    words = _words(count)
    duration = words[-1]["end"] + 1.0
    envelope = None
    if with_energy:
        # 50Hz of quiet speech with a loud stretch a third of the way in.
        frames = int(duration * 50)
        db = [-30] * frames
        for i in range(int(frames * 0.3), int(frames * 0.35)):
            db[i] = -8
        envelope = {"rate": 50.0, "db": db}
    return build_timeline_from_transcript(
        source_path="C:/media/talk.mp4",
        duration_seconds=duration,
        transcript_words=words,
        fps_num=30, fps_den=1,
        speech_regions=[(0.0, duration)],
        energy_envelope=envelope,
        pause_padding_seconds=0.0,
    )


def _program(**kwargs) -> Program:
    return build_program(_timeline(**kwargs))


def _beat(**kwargs) -> Beat:
    base = dict(start_s=5.0, end_s=11.0, topic="a topic", kind="broll_image",
                priority=0.8, image_prompt="a photograph of something")
    base.update(kwargs)
    return Beat(**base)


# --- Stage A: the programme ------------------------------------------------

def test_the_programme_places_words_in_the_cut_not_the_source():
    program = _program()
    assert program.words
    assert program.words[0].tl_start_s == pytest.approx(0.0, abs=0.1)
    assert program.duration_s > 0


def test_a_word_that_was_cut_never_appears_in_the_programme():
    """The programme is what the viewer hears. Planning a cutaway against a word
    that was edited out would place it at a time nobody says anything."""
    timeline = _timeline()
    timeline.words[10].enabled = False
    program = build_program(timeline)
    assert not any(w.text == "shabd10" for w in program.words)


def test_loudness_becomes_a_relative_emphasis_score():
    program = _program()
    assert program.has_energy
    scores = [w.emphasis_z for w in program.words]
    assert max(scores) > 0.5, "the loud stretch should stand out"
    assert min(scores) < 0.0


def test_a_project_with_no_energy_envelope_still_builds():
    """Older projects predate the envelope. They must plan, just without
    emphasis — not raise halfway through an overnight run."""
    program = _program(with_energy=False)
    assert program.words
    assert program.has_energy is False
    assert all(w.emphasis_z == 0.0 for w in program.words)


# --- Stage B: parsing and validation --------------------------------------

def test_json_is_found_inside_prose_and_code_fences():
    """Small models wrap answers in explanation however firmly they are told not
    to. Refusing those answers costs real content for no benefit."""
    assert first_json_object('Sure!\n```json\n{"beats": []}\n```') == {"beats": []}
    assert first_json_object('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert first_json_object("no json here at all") is None


def test_beats_are_clamped_to_the_programme():
    program = _program()
    kept, _ = validate_plan([_beat(start_s=-5.0, end_s=1e6)], program,
                            PresentationSettings())
    assert kept and kept[0].start_s == 0.0
    assert kept[0].end_s <= program.duration_s


def test_a_beat_over_silence_is_dropped():
    """A cutaway where nobody is speaking is a picture with no reason to exist.

    Built by hand rather than from a timeline: after the auto-edit ripples its
    cuts together the programme has no silent stretches to point at, and this
    rule exists for the model inventing a time rather than for a real gap.
    """
    program = Program(
        duration_s=60.0, fps=30.0,
        words=[ProgramWord(text="hello", tl_start_s=t, tl_end_s=t + 0.4,
                           source_start_frame=int(t * 30))
               for t in (0.0, 0.5, 1.0, 1.5, 2.0)],
        segments=[],
    )
    kept, dropped = validate_plan([_beat(start_s=30.0, end_s=36.0)], program,
                                  PresentationSettings())
    assert kept == []
    assert any("speech" in d["reason"] for d in dropped)


def test_a_beat_past_the_end_of_the_video_is_dropped():
    program = _program()
    beyond = program.duration_s + 30.0
    kept, dropped = validate_plan([_beat(start_s=beyond, end_s=beyond + 6)],
                                  program, PresentationSettings())
    assert kept == [] and dropped


def test_a_beat_missing_the_field_its_kind_needs_is_dropped():
    program = _program()
    kept, dropped = validate_plan(
        [_beat(kind="popup", popup_text=None), _beat(kind="broll_image", image_prompt=None)],
        program, PresentationSettings())
    assert kept == []
    assert len(dropped) == 2


def test_the_density_budget_caps_how_much_of_the_video_is_covered():
    """Left alone the model will illustrate everything, and the result stops
    being a talking head and becomes a slideshow."""
    program = _program(count=200)
    beats = [_beat(start_s=float(i * 2), end_s=float(i * 2 + 5), topic=f"topic {i}")
             for i in range(40)]
    kept, dropped = validate_plan(beats, program,
                                  PresentationSettings(max_broll_per_minute=2.0))
    minutes = program.duration_s / 60.0
    assert len(kept) <= max(1, round(minutes * 2.0)) + 1
    assert dropped


async def test_a_long_topic_becomes_many_short_shots_near_the_target():
    """A long story must fill its coverage budget with many distinct short images,
    not one long still. The slices tile into image+gap cells so placement keeps
    them all, landing near the coverage target instead of ~40%."""
    from presentation.shotplan import multiply_beats, budget_beats
    program = _program(count=200)                       # ~80s programme
    settings = PresentationSettings()
    beat = _beat(start_s=0.0, end_s=program.duration_s, topic="the whole story")
    multiplied = await multiply_beats([beat], settings, ask=None)   # no LLM: deterministic
    kept, _ = budget_beats(multiplied, program, settings)
    covered = sum(b.planned_duration_s or 0 for b in kept)
    assert len(kept) >= 8                               # many images, not one still
    assert covered / program.duration_s >= 0.6          # fills most of the budget


def test_cutaways_are_never_planned_back_to_back():
    # Spacing is judged on the on-screen cutaway, not the topic span (a shot is
    # planned shorter than its topic so the speaker keeps the gap). Two shots
    # that start within a cutaway-length of each other would bury the speaker
    # between them, and only the higher-priority one survives.
    program = _program(count=200)
    beats = [_beat(start_s=10.0, end_s=16.0, topic="first"),
             _beat(start_s=13.0, end_s=19.0, topic="second")]
    kept, _ = validate_plan(beats, program, PresentationSettings())
    assert len(kept) == 1


def test_two_beats_about_the_same_thing_collapse():
    program = _program(count=200)
    beats = [_beat(start_s=5.0, end_s=11.0, topic="cornwall university experiment"),
             _beat(start_s=40.0, end_s=46.0, topic="the cornwall university experiment")]
    kept, dropped = validate_plan(beats, program, PresentationSettings())
    assert len(kept) == 1
    assert any("duplicate" in d["reason"] for d in dropped)


def test_a_popup_longer_than_a_glance_is_trimmed():
    program = _program()
    kept, _ = validate_plan(
        [_beat(kind="popup", popup_text="one two three four five six seven eight",
               image_prompt=None)],
        program, PresentationSettings())
    assert kept and len(kept[0].popup_text.split()) <= 6


def test_graphics_stay_off_unless_asked_for():
    """An animated zoom on a transparent PNG composites a black box, so graphics
    are opt-in until that path is fixed."""
    program = _program()
    kept, dropped = validate_plan([_beat(kind="graphic")], program,
                                  PresentationSettings(graphics=False))
    assert kept == []
    assert any("switched off" in d["reason"] for d in dropped)


@pytest.mark.asyncio
async def test_a_garbage_answer_yields_no_beats_rather_than_bad_ones():
    program = _program()

    async def ask(_system, _user):
        return "I'm sorry, I can't help with that."

    plan = await plan_shots(program, PresentationSettings(), ask)
    assert plan.source == "fallback", "a useless model should fall back, not invent"


@pytest.mark.asyncio
async def test_no_model_at_all_still_produces_a_plan():
    """A night where LM Studio failed to load must not deliver a bare talking
    head — the keyword fallback is deliberately worse and deliberately present."""
    program = _program(count=200)
    plan = await plan_shots(program, PresentationSettings(), None)
    assert plan.source == "fallback"
    assert plan.beats
    assert all(b.image_prompt for b in plan.beats)


def test_the_fallback_needs_no_model_and_no_energy():
    program = _program(count=200, with_energy=False)
    beats = fallback_plan(program, PresentationSettings())
    assert beats


# --- Stage B: genre --------------------------------------------------------

def test_a_horror_transcript_is_detected_from_its_own_words():
    from presentation.genre import keyword_genre
    assert keyword_genre(
        "aaj hum ek haunted jagah gaye jahan bhoot dikha, "
        "paranormal cheezein hui aur sab darr gaye") == "horror"
    assert keyword_genre("aaj mausam accha tha aur hum ghar par rahe") == "general"


def test_unknown_genre_names_style_nothing_rather_than_erroring():
    from presentation.genre import apply_look, normalise
    assert normalise("scary") == "horror"
    assert normalise("Horror") == "horror"
    assert normalise("cats-knitting-asmr") == "general"
    assert normalise(None) == "general"
    prompt = "a quiet village street at dusk"
    assert apply_look(prompt, "unknown") == prompt, "an unknown genre must cost nothing"


def test_the_genre_look_is_stamped_once_not_twice():
    from presentation.genre import apply_look, style_for
    look = style_for("horror").look
    once = apply_look("an abandoned house", "horror")
    assert look in once
    assert apply_look(once, "horror") == once


@pytest.mark.asyncio
async def test_a_horror_video_gets_horror_prompts():
    """The whole point of genre awareness: whatever scene the model writes, the
    generated picture must carry the video's mood."""
    from presentation.genre import style_for
    program = _program()

    async def ask(system, _user, _schema=None):
        if "classify" in system.lower():
            return json.dumps({"genre": "horror"})
        if "researcher" in system:
            return json.dumps({"topics": [{
                "start_s": 0.0, "end_s": 20.0, "topic": "the haunted house",
                "summary": "A story about a haunted house.",
                "visual": "an abandoned house at night", "priority": 0.9}]})
        if "list of topics" in system:
            return json.dumps({"beats": [{
                "topic": "the haunted house", "kind": "broll_image",
                "image_prompt": "an abandoned house on a hill at night",
                "video_prompt": None, "popup_text": None,
                "negative_prompt": "text, watermark, logo",
                "style_hint": "photoreal"}]})
        return "no"

    plan = await plan_shots(program, PresentationSettings(), ask)
    assert plan.genre == "horror"
    cutaways = [b for b in plan.beats if b.is_cutaway]
    assert cutaways
    look = style_for("horror").look
    for beat in cutaways:
        assert look in beat.image_prompt
        assert "bright cheerful colors" in beat.negative_prompt


def test_the_fallback_prompts_carry_the_genre_too():
    from presentation.genre import style_for
    program = _program(count=200)
    beats = fallback_plan(program, PresentationSettings(), genre="horror")
    assert beats
    assert all(style_for("horror").look in b.image_prompt for b in beats)


@pytest.mark.asyncio
async def test_an_explicit_genre_setting_wins_without_asking_the_model():
    program = _program()
    plan = await plan_shots(program, PresentationSettings(), None, genre="comedy")
    assert plan.genre == "comedy"


# --- Stage B: the video share ----------------------------------------------

def _kept_beats(n=10, video=()):
    """n budgeted 2.5s cutaways with descending priority, some already video."""
    beats = []
    for i in range(n):
        beats.append(_beat(
            start_s=i * 4.0, end_s=i * 4.0 + 2.5,
            topic=f"topic {i}", priority=1.0 - i * 0.05,
            kind="broll_video" if i in video else "broll_image",
            planned_duration_s=2.5,
            image_prompt=f"a photograph of topic {i}"))
    return beats


def test_the_most_important_beats_get_video_at_the_asked_share():
    from presentation.shotplan import balance_video_share
    out = balance_video_share(_kept_beats(10), PresentationSettings())
    videos = [b for b in out if b.kind == "broll_video"]
    # 18% of 25s is 4.5s: exactly the two highest-priority beats at 2.5s each.
    assert [b.topic for b in videos] == ["topic 0", "topic 1"]
    assert all(b.video_prompt for b in videos), "a promoted beat needs a motion prompt"


def test_video_share_is_seconds_of_screen_time_within_one_beat():
    from presentation.shotplan import balance_video_share
    out = balance_video_share(_kept_beats(20), PresentationSettings())
    secs = sum(b.planned_duration_s for b in out if b.kind == "broll_video")
    total = sum(b.planned_duration_s for b in out if b.is_cutaway)
    assert 0.15 <= secs / total <= 0.25, "the target is 15-20%, one beat of slack"


def test_a_model_that_wants_everything_as_video_is_reined_in():
    from presentation.shotplan import balance_video_share
    out = balance_video_share(_kept_beats(10, video=range(10)), PresentationSettings())
    secs = sum(b.planned_duration_s for b in out if b.kind == "broll_video")
    total = sum(b.planned_duration_s for b in out if b.is_cutaway)
    assert secs / total <= 0.25, "excess video is demoted, lowest priority first"
    # What survives as video is the beats the planner cared most about.
    kept_videos = [b.topic for b in out if b.kind == "broll_video"]
    assert "topic 0" in kept_videos


def test_no_video_promotion_when_video_is_off():
    from presentation.shotplan import balance_video_share
    out = balance_video_share(
        _kept_beats(10), PresentationSettings(broll_video=False))
    assert all(b.kind == "broll_image" for b in out)
    out = balance_video_share(
        _kept_beats(10), PresentationSettings(video_broll_share=0.0))
    assert all(b.kind == "broll_image" for b in out)


def test_video_generation_size_is_capped_but_keeps_the_aspect():
    from presentation.assets import _video_canvas_default
    w, h = _video_canvas_default((1920, 1080))
    assert w * h <= 832 * 480 * 1.1, "a 1080p canvas must not generate at 1080p"
    assert abs(w / h - 16 / 9) < 0.1
    w, h = _video_canvas_default((1080, 1920))
    assert h > w, "a vertical short stays vertical"
    assert w % 16 == 0 and h % 16 == 0


# --- Stage C: the workflow registry ---------------------------------------

def test_bindings_are_detected_on_the_bundled_workflows():
    """The old loader hardcoded node 6 as the positive prompt and 7 as the
    negative. Detection has to agree with it, or every existing workflow breaks."""
    for name in ("broll_generation.json", "thumbnail.json"):
        detail = workflows.describe(name)
        assert detail["valid"], detail.get("error")
        assert detail["bindings"]["positive"] == ["6", "inputs", "text"]
        assert detail["bindings"]["negative"] == ["7", "inputs", "text"]
        assert detail["output_kind"] == "image"


def test_a_video_workflow_is_recognised_as_one():
    """A `length` input or a video save node is what separates a workflow that
    makes a clip from one that makes a still."""
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a long positive description of a scene", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": 848, "height": 480, "length": 73}},
        "5": {"class_type": "KSampler",
              "inputs": {"seed": 1, "steps": 20, "cfg": 6.0,
                         "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VHS_VideoCombine",
              "inputs": {"frame_rate": 24, "filename_prefix": "out", "images": ["5", 0]}},
    }
    bindings = workflows.guess_bindings(graph)
    assert bindings["output_kind"] == "video"
    assert bindings["positive"] == ["2", "inputs", "text"]
    assert bindings["negative"] == ["3", "inputs", "text"]
    assert bindings["length"] == ["4", "inputs", "length"]
    assert bindings["fps"] == ["6", "inputs", "frame_rate"]


def test_values_are_written_at_the_bound_paths():
    graph = workflows.load_graph("broll_generation.json")
    bindings = workflows.guess_bindings(graph)
    built = workflows.apply_bindings(graph, bindings, {
        "positive": "a cat", "negative": "no dogs", "width": 1920, "seed": 7})
    assert built["6"]["inputs"]["text"] == "a cat"
    assert built["7"]["inputs"]["text"] == "no dogs"
    assert built["5"]["inputs"]["width"] == 1920
    assert built["3"]["inputs"]["seed"] == 7
    # The original must be untouched, or the second generation inherits the first.
    assert graph["6"]["inputs"]["text"] != "a cat"


def test_a_binding_may_write_one_value_into_several_nodes():
    """A two-stage workflow (image model feeding a video model) needs the same
    prompt and size in both stages; a manifest binding that is a LIST of paths
    writes them all, and an empty list means 'never write this here'."""
    graph = {
        "a": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
        "b": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
        "s": {"class_type": "KSampler", "inputs": {"steps": 8}},
    }
    bindings = {
        "positive": [["a", "inputs", "text"], ["b", "inputs", "text"]],
        "steps": [],
    }
    built = workflows.apply_bindings(graph, bindings, {"positive": "new", "steps": 30})
    assert built["a"]["inputs"]["text"] == "new"
    assert built["b"]["inputs"]["text"] == "new"
    assert built["s"]["inputs"]["steps"] == 8, "an empty binding protects the value"


def test_the_bundled_video_workflow_is_recognised_and_configured():
    """The two-stage Z-Image -> Wan I2V workflow ships with the app and the
    manifest routes the broll_video role to it; a video beat must resolve to a
    workflow that really makes video."""
    detail = workflows.describe("zimage_wan22_i2v.json")
    assert detail["valid"], detail.get("error")
    assert detail["output_kind"] == "video"
    assert detail["roles_ok"]["broll_video"]
    resolved = workflows.resolve("broll_video")
    assert resolved is not None
    assert resolved.output_kind == "video"
    built = resolved.build(positive="a foggy street", negative="text",
                           width=832, height=480, seed=7, length=81, fps=16,
                           prefix="beat_b01")
    # The prompt must reach BOTH stages: the still generator and the animator.
    assert built["z_pos"]["inputs"]["text"] == "a foggy street"
    assert built["w_pos"]["inputs"]["text"] == "a foggy street"
    assert built["w_i2v"]["inputs"]["length"] == 81
    assert built["z_sample"]["inputs"]["seed"] == 7
    assert built["w_sample_high"]["inputs"]["noise_seed"] == 7
    assert built["v_save"]["inputs"]["filename_prefix"] == "beat_b01"
    # Global single-model overrides must not corrupt the tuned two-stage graph.
    built = resolved.build(positive="p", steps=30, cfg=7.5, model="foo.safetensors")
    assert built["z_sample"]["inputs"]["steps"] == 8
    assert built["z_sample"]["inputs"]["cfg"] == 1
    assert built["z_unet"]["inputs"]["unet_name"] == "z_image_turbo_bf16.safetensors"


def test_an_unresolvable_binding_is_skipped_rather_than_raising():
    """A workflow with no steps input keeps its own step count. That is a working
    generation; refusing to run would not be."""
    graph = workflows.load_graph("thumbnail.json")
    built = workflows.apply_bindings(graph, {"steps": ["999", "inputs", "steps"]},
                                     {"steps": 40})
    assert built is not None


def test_a_ui_export_is_rejected_with_an_explanation(tmp_path, monkeypatch):
    """The UI format has no node ids to bind to. Saying so beats a KeyError."""
    from config import WORKFLOWS_DIR
    bad = WORKFLOWS_DIR / "_test_ui_export.json"
    bad.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    try:
        detail = workflows.describe("_test_ui_export.json")
        assert detail["valid"] is False
        assert "Export (API)" in detail["error"]
    finally:
        bad.unlink(missing_ok=True)


# --- Stage D: placement ----------------------------------------------------

def _asset(beat_id: str, tmp_path, kind: str = "image") -> Asset:
    path = tmp_path / f"{beat_id}.png"
    path.write_bytes(b"not really a png")
    return Asset(beat_id=beat_id, kind=kind, path=str(path),
                 width=1920, height=1080, duration_s=0.0)


def test_a_cutaway_lands_on_v3_with_its_own_origin(tmp_path):
    timeline = _timeline(count=200)
    program = build_program(timeline)
    beats = [_beat(start_s=10.0, end_s=16.0)]
    beats[0].id = "b00"
    count = place_broll(timeline, beats, [_asset("b00", tmp_path)], program,
                        PresentationSettings())
    assert count == 1
    placed = [i for i in timeline.items if i.origin == BROLL_ORIGIN]
    assert len(placed) == 1
    assert placed[0].track == "V3"


def test_a_still_gets_a_ken_burns_move_and_a_clip_does_not(tmp_path):
    """A generated clip already moves; a zoom on top of its own motion reads as
    a mistake."""
    timeline = _timeline(count=200)
    program = build_program(timeline)
    still, clip = _beat(start_s=10.0, end_s=16.0), _beat(start_s=40.0, end_s=46.0)
    still.id, clip.id = "b00", "b01"
    video = _asset("b01", tmp_path, kind="video")
    video.duration_s = 5.0
    place_broll(timeline, [still, clip], [_asset("b00", tmp_path), video],
                program, PresentationSettings())
    placed = sorted([i for i in timeline.items if i.origin == BROLL_ORIGIN],
                    key=lambda i: i.timeline_start_frame)
    assert placed[0].transform is not None and placed[0].transform.is_animated()
    assert placed[1].transform is None


def test_a_second_run_replaces_the_first_rather_than_stacking(tmp_path):
    timeline = _timeline(count=200)
    program = build_program(timeline)
    beats = [_beat(start_s=10.0, end_s=16.0)]
    beats[0].id = "b00"
    for _ in range(3):
        place_broll(timeline, beats, [_asset("b00", tmp_path)], program,
                    PresentationSettings())
    assert len([i for i in timeline.items if i.origin == BROLL_ORIGIN]) == 1
    assert len([s for s in timeline.sources if s.startswith("src_broll_")]) == 1


def test_a_popup_with_no_clear_moment_draws_over_the_broll(tmp_path):
    """The pop-up prefers a moment with the speaker on screen, but when its
    whole span sits under cutaways it draws over them rather than vanishing —
    at 60-75% coverage the on-camera gaps are shorter than a pop-up, and
    skip-on-collision shipped videos with no pop-ups at all."""
    timeline = _timeline(count=200)
    program = build_program(timeline)
    cutaway = _beat(start_s=10.0, end_s=16.0)
    cutaway.id = "b00"
    popup = _beat(start_s=11.0, end_s=15.0, kind="popup", popup_text="An idea",
                  image_prompt=None)
    popup.id = "b01"

    place_broll(timeline, [cutaway], [_asset("b00", tmp_path)], program,
                PresentationSettings())
    placed = place_popups(timeline, [popup], program, PresentationSettings(),
                          broll_windows(timeline))
    assert placed == 1
    item = next(i for i in timeline.items if i.origin == POPUP_ORIGIN)
    assert item.timeline_start_frame / 30.0 == pytest.approx(11.0, abs=0.1)


def test_a_cutaway_is_snapped_to_a_word_boundary(tmp_path):
    """Cutting away mid-syllable draws attention to the cut itself."""
    timeline = _timeline(count=200)
    program = build_program(timeline)
    beat = _beat(start_s=10.13, end_s=16.0)
    beat.id = "b00"
    place_broll(timeline, [beat], [_asset("b00", tmp_path)], program,
                PresentationSettings())
    item = next(i for i in timeline.items if i.origin == BROLL_ORIGIN)
    start_s = item.timeline_start_frame / 30.0
    assert any(abs(w.tl_start_s - start_s) < 0.05 for w in program.words)


def test_a_popup_lands_on_its_own_text_track_above_the_captions():
    timeline = _timeline(count=200)
    program = build_program(timeline)
    popup = _beat(start_s=10.0, end_s=15.0, kind="popup", popup_text="An idea",
                  image_prompt=None)
    popup.id = "b00"
    assert place_popups(timeline, [popup], program, PresentationSettings()) == 1
    item = next(i for i in timeline.items if i.origin == POPUP_ORIGIN)
    assert item.track == "TP"
    # Captions own the lower third; a pop-up over them is unreadable.
    assert item.text.style.pos_y < 0


# --- Stage C: giving up on a broken night ----------------------------------
#
# One failed beat is one missing picture. But when nothing has generated for
# several beats in a row — ComfyUI died mid-run, the workflow is wrong — every
# further attempt burns a full timeout, and an overnight pass "completes" hours
# later with zero images. The stage stops instead, and says which beats it
# never tried.

class _FakeComfyQueue:
    """Stands in for the module-level queue_manager, scripting each submit."""

    def __init__(self, script):
        self.script = list(script)   # each entry: "ok" or "boom"
        self.submissions = 0
        self.client = self

    def is_connected(self, timeout: float = 10.0) -> bool:
        return True

    async def submit_and_wait(self, graph, timeout=0):
        step = self.script[self.submissions % len(self.script)]
        self.submissions += 1
        if step == "boom":
            raise ConnectionError("ComfyUI refused the connection")
        return [str(self._output)]


def _fake_image_workflow():
    return workflows.ResolvedWorkflow("broll_image", "fake.json", {}, {}, "image")


def _wire_assets(monkeypatch, tmp_path, queue):
    import comfyui_bridge
    from presentation import assets as assets_stage
    out = tmp_path / "out.png"
    out.write_bytes(b"not really a png")
    queue._output = out
    monkeypatch.setattr(comfyui_bridge, "queue_manager", queue)
    monkeypatch.setattr(assets_stage.workflows, "resolve",
                        lambda role: _fake_image_workflow() if role == "broll_image" else None)
    # Keep the app-settings store out of it.
    monkeypatch.setattr(assets_stage, "_setting", lambda key, default: default)
    return assets_stage


@pytest.mark.asyncio
async def test_repeated_failures_abort_the_night_instead_of_burning_it(tmp_path, monkeypatch):
    queue = _FakeComfyQueue(script=["boom"])
    assets_stage = _wire_assets(monkeypatch, tmp_path, queue)
    # Distinct ids (as validate_plan assigns) so each beat is its own generation
    # rather than a cache hit on the first.
    beats = [_beat(id=f"b{i:02d}", topic=f"topic {i}",
                   start_s=5.0 + 8 * i, end_s=11.0 + 8 * i) for i in range(6)]

    generated, failures = await assets_stage.generate_assets(
        beats, tmp_path, PresentationSettings())

    assert generated == []
    assert queue.submissions == assets_stage.MAX_CONSECUTIVE_FAILURES, \
        "after three straight failures no further beat may be attempted"
    assert len(failures) == 6, "every beat is accounted for, attempted or not"
    assert sum("not attempted" in f["reason"] for f in failures) == 3


@pytest.mark.asyncio
async def test_one_bad_beat_between_good_ones_costs_only_itself(tmp_path, monkeypatch):
    queue = _FakeComfyQueue(script=["boom", "ok"])
    assets_stage = _wire_assets(monkeypatch, tmp_path, queue)
    # Distinct ids (as validate_plan assigns) so each beat is its own generation
    # rather than a cache hit on the first.
    beats = [_beat(id=f"b{i:02d}", topic=f"topic {i}",
                   start_s=5.0 + 8 * i, end_s=11.0 + 8 * i) for i in range(6)]

    generated, failures = await assets_stage.generate_assets(
        beats, tmp_path, PresentationSettings())

    assert queue.submissions == 6, "a success resets the failure counter"
    assert len(generated) == 3 and len(failures) == 3
    assert not any("not attempted" in f["reason"] for f in failures)


# --- the start route refuses what cannot work --------------------------------

class _OfflineComfy:
    def __init__(self):
        self.client = self

    def is_connected(self, timeout: float = 10.0) -> bool:
        return False


@pytest.mark.asyncio
async def test_a_pass_wanting_pictures_is_refused_when_comfyui_is_off(monkeypatch):
    """A 503 now beats a degraded overnight run the user never asked for."""
    import comfyui_bridge
    from fastapi import HTTPException
    from routes import presentation as presentation_routes

    monkeypatch.setattr(comfyui_bridge, "queue_manager", _OfflineComfy())
    with pytest.raises(HTTPException) as err:
        await presentation_routes.start_pass(
            "p1", presentation_routes.RunRequest(settings={"broll": True}))
    assert err.value.status_code == 503
    assert "ComfyUI" in err.value.detail


@pytest.mark.asyncio
async def test_a_pass_without_pictures_starts_even_with_comfyui_off(monkeypatch):
    import comfyui_bridge
    from routes import presentation as presentation_routes

    monkeypatch.setattr(comfyui_bridge, "queue_manager", _OfflineComfy())
    started = []

    async def _no_run(project_id, job_id, settings):
        started.append(job_id)

    monkeypatch.setattr(presentation_routes, "_run_presentation", _no_run)
    answer = await presentation_routes.start_pass(
        "p1", presentation_routes.RunRequest(
            settings={"broll": False, "graphics": False, "thumbnail": False}))
    assert answer["status"] == "running"
    import asyncio
    await asyncio.sleep(0)          # let the stubbed job task run to completion
    assert started, "the job was accepted and started"


# --- the whole pass, degrading ---------------------------------------------

@pytest.mark.asyncio
async def test_the_pass_still_delivers_with_no_llm_and_no_comfyui(tmp_path, monkeypatch):
    """The failure mode that matters: the user wakes up to *something*.

    With the language model down and ComfyUI off there are no pictures and no
    topics, but the cut, the zooms and the captions are all still deliverable —
    and the report has to say plainly which parts degraded.
    """
    from presentation import director
    from store.project_store import ProjectStore

    monkeypatch.setattr(director, "PROJECTS_DIR", tmp_path)
    # No model, no picture generator, no render — the three external things.
    monkeypatch.setattr(director, "_llm_asker", lambda: _none())
    monkeypatch.setattr(director, "_comfyui_online", lambda: False)
    monkeypatch.setattr(director.assets_stage, "generate_assets", _empty_assets)

    timeline = _timeline(count=200)
    store = ProjectStore(base_dir=str(tmp_path))
    store.save_project("p1", {
        "id": "p1", "name": "Test", "source_video": "C:/media/talk.mp4",
        "timeline": timeline.model_dump(), "settings": {},
    })

    report = await director.run_presentation_pass(
        "p1", PresentationSettings(render=False, thumbnail=False, face_zoom=False))

    assert report.project_id == "p1"
    assert "llm_unavailable" in report.degraded
    assert report.comfyui_online is False
    assert "comfyui_offline" in report.degraded, \
        "a picture-less run must say WHY there are no pictures"
    assert report.captions > 0, "captions do not need the model or the GPU"
    saved = store.get_project("p1")
    assert saved["timeline"], "the timeline must be saved even when stages degrade"
    assert (tmp_path / "p1" / "presentation_report.json").exists()


async def _none():
    return None


async def _empty_assets(*args, **kwargs):
    return [], [{"beat_id": "b00", "reason": "ComfyUI offline"}]


# --- hiding the jump cuts ---------------------------------------------------
#
# A jump cut is invisible while a full-frame cutaway is over it. Placement
# therefore slides each window (within its beat) to straddle the nearest V1
# join; a shot that can reach no join runs exactly where the planner put it,
# which test_a_cutaway_is_snapped_to_a_word_boundary already locks in.


def _timeline_with_a_join():
    """A timeline whose auto-edit removed a stretch, leaving one V1 join."""
    from timeline.ops import rebuild_primary_tracks
    timeline = _timeline(count=200)
    source_id = next(iter(timeline.sources))
    for word in timeline.words[60:75]:
        word.enabled = False
    rebuild_primary_tracks(timeline, source_id)
    return timeline


def test_a_cutaway_slides_to_straddle_the_nearest_jump_cut(tmp_path):
    from presentation.placement import (CUT_COVER_MARGIN_S, jump_cut_coverage,
                                        jump_cut_times)
    timeline = _timeline_with_a_join()
    cuts = jump_cut_times(timeline)
    assert len(cuts) == 1
    cut = cuts[0]

    program = build_program(timeline)
    beat = _beat(start_s=cut - 4.0, end_s=cut + 4.0)
    beat.id = "b00"
    assert place_broll(timeline, [beat], [_asset("b00", tmp_path)], program,
                       PresentationSettings()) == 1
    item = next(i for i in timeline.items if i.origin == BROLL_ORIGIN)
    start_s = item.timeline_start_frame / 30.0
    end_s = item.timeline_end_frame / 30.0
    assert start_s + CUT_COVER_MARGIN_S - 0.05 <= cut <= end_s - CUT_COVER_MARGIN_S + 0.05
    assert jump_cut_coverage(timeline) == {"total": 1, "covered": 1}


def test_a_window_never_leaves_its_beat_to_chase_a_far_join(tmp_path):
    """The picture must stay on-topic: a join well outside the beat's span is
    another beat's business."""
    from presentation.placement import jump_cut_times
    timeline = _timeline_with_a_join()
    cut = jump_cut_times(timeline)[0]

    program = build_program(timeline)
    beat = _beat(start_s=cut + 8.0, end_s=cut + 14.0)
    beat.id = "b00"
    place_broll(timeline, [beat], [_asset("b00", tmp_path)], program,
                PresentationSettings())
    item = next(i for i in timeline.items if i.origin == BROLL_ORIGIN)
    start_s = item.timeline_start_frame / 30.0
    assert start_s >= cut + 8.0 - 0.6


# --- pop-in labels, the title, and the atmosphere ---------------------------


@pytest.mark.asyncio
async def test_topics_become_popups_when_the_model_plans_none():
    """The model routinely plans every beat as a cutaway — a real run planned
    18 beats and zero pop-ups — and the video then has no text accents at all.
    The topics themselves supply one label each."""
    from presentation.shotplan import plan_shots

    program = _program(count=200)

    async def ask(system, user, schema=None):
        if "researcher" in system:
            return json.dumps({"topics": [
                {"start_s": 1.0, "end_s": 30.0, "topic": "The Spotlight Effect",
                 "summary": "s", "visual": "a street", "priority": 0.9},
                {"start_s": 30.0, "end_s": 70.0, "topic": "Cornell Experiment",
                 "summary": "s", "visual": "a lab", "priority": 0.8},
            ]})
        if "art director" in system and "ONE scene" not in system:
            return json.dumps({"beats": [
                {"topic": "The Spotlight Effect", "kind": "broll_image",
                 "image_prompt": "a busy street, documentary photo",
                 "video_prompt": None, "popup_text": None,
                 "negative_prompt": "text", "style_hint": "photoreal"},
            ]})
        return None

    # Chapter titles carry the topic names when cards are on; this checks the
    # pop-up fallback itself, so cards are off here.
    plan = await plan_shots(program, PresentationSettings(cards=False), ask, genre="general")
    popups = [b for b in plan.beats if b.kind == "popup"]
    assert popups, "topic labels should fill in for the model's missing pop-ups"
    # The first topic starts inside the title zone and stays clear of it.
    assert all(b.start_s >= 3.0 for b in popups)
    assert any("Cornell" in (b.popup_text or "") for b in popups)


def test_a_popup_slides_off_a_cutaway_instead_of_dying(tmp_path):
    """At 60-75%% B-roll coverage a fixed start lands on a cutaway more often
    than not, and skip-on-collision shipped videos with no pop-ups at all."""
    timeline = _timeline(count=200)
    program = build_program(timeline)
    cutaway = _beat(start_s=10.0, end_s=16.0)
    cutaway.id = "b00"
    place_broll(timeline, [cutaway], [_asset("b00", tmp_path)], program,
                PresentationSettings())
    popup = _beat(start_s=10.0, end_s=25.0, kind="popup", popup_text="An idea",
                  image_prompt=None)
    popup.id = "p00"
    placed = place_popups(timeline, [popup], program, PresentationSettings(),
                          broll_windows(timeline))
    assert placed == 1
    from presentation.placement import POPUP_ORIGIN as _PO
    item = next(i for i in timeline.items if i.origin == _PO)
    start_s = item.timeline_start_frame / 30.0
    broll_end = max(end for _s, end in broll_windows(timeline))
    assert start_s >= broll_end - 0.01, "the pop-up must wait out the cutaway"


def test_the_atmosphere_layer_is_replaced_not_stacked():
    from presentation.director import _apply_atmosphere
    from timeline.schema import AtmosphereEffect
    timeline = _timeline(count=60)
    timeline.effects.append(AtmosphereEffect(type="rain", intensity=0.9))  # user's

    first = _apply_atmosphere(timeline, PresentationSettings(), "horror")
    second = _apply_atmosphere(timeline, PresentationSettings(), "science_education")

    assert first == "fog" and second == "grain"
    ours = [e for e in timeline.effects if e.origin == "presentation"]
    assert len(ours) == 1 and ours[0].type == "grain"
    assert any(e.type == "rain" for e in timeline.effects), "user effects stay"


def test_atmosphere_respects_genres_that_want_none():
    from presentation.director import _apply_atmosphere
    timeline = _timeline(count=60)
    assert _apply_atmosphere(timeline, PresentationSettings(), "finance") == ""
    assert not timeline.effects


def test_the_title_overlay_draws_the_thumbnail_title():
    from presentation.director import _apply_title
    timeline = _timeline(count=60)
    assert _apply_title(timeline, PresentationSettings(), "SPOTLIGHT EFFECT")
    intros = [i for i in timeline.items if i.origin == "intro"]
    assert intros and "SPOTLIGHT EFFECT" in intros[0].text.content
    # The default preset is an overlay: nothing shifted, no card in front.
    assert timeline.program_offset_frames == 0
