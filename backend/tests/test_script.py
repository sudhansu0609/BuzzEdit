"""The user's script: alignment, spelling repair, paragraphs and directions."""

import pytest

from backend.asr import script_align
from backend.presentation import script as script_stage
from backend.presentation.models import PresentationSettings
from backend.presentation.program import build_program
from backend.presentation.shotplan import plan_shots
from backend.timeline import build_timeline_from_transcript

SCRIPT = """# The T-shirt Study
Namaste dosto, aaj hum baat karenge spotlight effect ki. [map: Cornell University]
Thomas Gilovich ne ek experiment kiya tha. [broll: a student in a yellow t-shirt]

Students ko ek embarrassing t-shirt pehna kar bheja gaya. [sfx: whoosh]
Sirf 25 percent logon ne notice kiya. [stat: 25%]

Toh dosto, subscribe karna mat bhoolna. [title: SPOTLIGHT EFFECT]"""

HEARD = ("Namaste dosto aaj hum baat karenge spotlite effect ki Thommel Gilgovich ne ek "
         "experiment kiya tha students ko ek embarrasing teeshart pehna kar bheja gaya "
         "sirf 25 parsent logon ne notice kiya toh dosto sabsakraaib karna mat bhoolna")


def _heard_words(step: float = 0.5):
    words = []
    for index, text in enumerate(HEARD.split()):
        words.append({"text": text, "start": index * step, "end": index * step + 0.4})
    return words


# --- parsing ---------------------------------------------------------------

def test_parse_separates_spoken_tokens_from_directions_and_headings():
    parsed = script_align.parse_script(SCRIPT)
    assert [p["heading"] for p in parsed["paragraphs"]] == ["The T-shirt Study", None, None]
    tokens = script_align.flat_tokens(parsed)
    assert "[map:" not in " ".join(tokens)
    assert tokens[0] == "Namaste" and tokens[-1] == "bhoolna."
    kinds = [(d["kind"], d["arg"]) for d in parsed["directives"]]
    assert ("chapter", "The T-shirt Study") in kinds
    assert ("map", "Cornell University") in kinds
    assert ("sfx", "whoosh") in kinds
    # The map direction sits before the word "Thomas" (token 9).
    assert next(d for d in parsed["directives"] if d["kind"] == "map")["token"] == 9


def test_unknown_brackets_stay_as_spoken_text():
    parsed = script_align.parse_script("He said [laughs] fine [note: x]")
    assert script_align.flat_tokens(parsed) == ["He", "said", "[laughs]", "fine", "[note:", "x]"]
    assert parsed["directives"] == []


# --- alignment and spelling ------------------------------------------------

def test_the_script_fixes_the_transcripts_spelling():
    words = _heard_words()
    record = script_align.ingest(SCRIPT, words)
    assert record["aligned_words"] == record["token_count"]
    texts = [w["text"] for w in words]
    assert "subscribe" in texts and "sabsakraaib" not in texts
    assert "Thomas" in texts and "Gilovich" in texts
    assert "t-shirt" in texts
    assert record["spelling_fixed"] > 10
    # Timing is the audio's, never the script's.
    assert words[3]["start"] == pytest.approx(1.5)


def test_an_extra_word_on_either_side_does_not_derail_the_alignment():
    words = _heard_words()
    words.insert(5, {"text": "umm", "start": 2.45, "end": 2.5})          # transcript extra
    script = SCRIPT.replace("baat karenge", "baat zaroor karenge")           # script extra
    record = script_align.ingest(script, words)
    texts = [w["text"] for w in words]
    assert "subscribe" in texts
    assert "umm" in texts                                                    # untouched
    assert record["aligned_words"] >= record["token_count"] - 2


def test_a_devanagari_script_updates_the_native_spelling():
    words = [{"text": "namaste", "word_native": "नमस्ते", "start": 0, "end": 0.4},
             {"text": "dosto", "word_native": "दोस्तो", "start": 0.5, "end": 0.9},
             {"text": "aaj", "word_native": "आज", "start": 1.0, "end": 1.4}]
    record = script_align.ingest("नमस्ते दोस्तों, आज", words)
    assert words[1]["word_native"] == "दोस्तों,"
    assert words[1]["text"] == "dosto"
    assert record["aligned_words"] == 3


def test_paragraph_spans_and_directive_times_come_from_the_aligned_words():
    words = _heard_words()
    record = script_align.ingest(SCRIPT, words)
    spans = record["paragraphs"]
    assert len(spans) == 3
    assert spans[0]["start"] == pytest.approx(0.0)
    assert spans[1]["start"] == pytest.approx(8.0)
    assert spans[2]["start"] == pytest.approx(16.0)
    at = {d["kind"]: d["at"] for d in record["directives"]}
    assert at["map"] == pytest.approx(4.5)       # before "Thomas"
    assert at["broll"] == pytest.approx(8.0)     # end of the paragraph → next word
    assert at["stat"] == pytest.approx(16.0)


# --- the presentation side -------------------------------------------------

def _project_with_script():
    words = []
    for i, t in enumerate(HEARD.split()):
        shift = 0.5 if i >= 15 else 0.0            # room for the filler below
        words.append({"word": t, "start": i * 0.5 + shift, "end": i * 0.5 + 0.4 + shift})
    # A cut: a half-second filler between "kiya" and "tha" is removed.
    words.insert(15, {"word": "umm", "start": 7.45, "end": 7.95, "disfluency": True})
    timeline = build_timeline_from_transcript(
        "C:/media/talk.mp4", 20.0, words, fps_num=30, fps_den=1,
        speech_regions=[(0.0, 20.0)], pause_padding_seconds=0.0)
    data = {"script": {"text": SCRIPT}}
    record = script_stage.apply_project_script(data, timeline)
    return data, timeline, record


def test_applying_a_project_script_rewrites_the_timeline_words_and_stores_seconds():
    data, timeline, record = _project_with_script()
    assert record["aligned_words"] == record["token_count"]
    assert any(w.text == "subscribe" for w in timeline.words)
    stored = data["script"]
    assert stored["paragraphs"][1]["start"] == pytest.approx(8.5, abs=0.05)
    assert stored["directives"][1]["at"] == pytest.approx(4.5, abs=0.05)


def test_the_script_context_lives_on_the_cut_programmes_clock():
    data, timeline, _ = _project_with_script()
    ctx = script_stage.context_from(data, timeline)
    assert ctx.present
    program = build_program(timeline)
    # Everything after the cut filler moves half a second earlier: the stat
    # sits on a word spoken at 16.5s in the source, 16.0s in the cut.
    stat = next(d for d in ctx.directives if d.kind == "stat")
    stored = next(d for d in data["script"]["directives"] if d["kind"] == "stat")
    assert stored["at"] == pytest.approx(16.5, abs=0.05)
    assert stat.tl_at_s == pytest.approx(16.0, abs=0.1)
    # Paragraphs tile the programme.
    for current, following in zip(ctx.paragraphs, ctx.paragraphs[1:]):
        assert current.tl_end_s == pytest.approx(following.tl_start_s)
    topics = script_stage.paragraph_topics(ctx, program)
    assert topics[0].topic == "The T-shirt Study" and topics[0].origin == "script"


def test_stage_directions_become_beats_that_win_the_budget():
    data, timeline, _ = _project_with_script()
    ctx = script_stage.context_from(data, timeline)
    program = build_program(timeline)
    out = script_stage.directive_beats(ctx, program, PresentationSettings())
    kinds = {b.kind for b in out["beats"]}
    assert {"map", "broll_image", "stat_callout", "chapter_title"} <= kinds
    assert all(b.priority == 1.0 and b.origin == "script" for b in out["beats"])
    stat = next(b for b in out["beats"] if b.kind == "stat_callout")
    assert stat.data == {"value": 25.0, "prefix": "", "suffix": "%", "decimals": 0}
    assert out["title"] == "SPOTLIGHT EFFECT"
    assert out["sfx"] and out["sfx"][0][1] == "whoosh"

    import asyncio
    plan = asyncio.run(plan_shots(program, PresentationSettings(target_coverage=0.2, broll_seconds_min=2.0, broll_seconds_max=3.0),
                                  None, "general",
                                  script_topics=script_stage.paragraph_topics(ctx, program),
                                  extra_beats=out["beats"]))
    kept = {b.kind for b in plan.beats}
    assert "map" in kept and "stat_callout" in kept and "chapter_title" in kept
    assert any(b.origin == "script" and b.kind == "broll_image" for b in plan.beats)
    assert plan.topics and plan.topics[0].origin == "script"


def test_chart_and_stat_directions_parse_their_data():
    assert script_stage._chart_data("Notice rate: Students=25%, Guess=50%") == {
        "labels": ["Students", "Guess"], "values": [25.0, 50.0], "unit": "%",
        "title": "Notice rate"}
    assert script_stage._chart_data("only one=5") == {}
    assert script_stage._stat_data("3.5 crore")["value"] == 3.5
    assert script_stage._stat_data("$120")["prefix"] == "$"


def test_map_chart_and_timeline_accept_a_trailing_seconds_hold():
    """Item B2 (MIXED_ART_STYLE_PLAN): `| seconds=N` times these directions
    like a still (BuzzcafStudio uses it so a shot fills an exact share of the
    video); without it, behaviour is unchanged."""
    from backend.presentation.models import Program

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(kind: str, arg: str, at: float = 2.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind=kind, arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0]

    timed_map = _beat("map", "Nigeria | seconds=6")
    assert timed_map.place == "Nigeria" and timed_map.text == "Nigeria"
    assert timed_map.planned_duration_s == 6.0 and timed_map.data.get("hold") is True
    assert timed_map.end_s == pytest.approx(8.0)

    plain_map = _beat("map", "Cornell University")
    assert plain_map.place == "Cornell University" and plain_map.planned_duration_s is None
    assert plain_map.data == {}
    assert plain_map.end_s == pytest.approx(2.0 + script_stage.DIRECTIVE_CUTAWAY_SECONDS)

    timed_chart = _beat("chart", "bar: Days: Day1=1, Day2=3 | seconds=5")
    assert timed_chart.data["labels"] == ["Day1", "Day2"]
    assert timed_chart.data["values"] == [1.0, 3.0]
    assert timed_chart.data["chart_type"] == "bar"
    assert timed_chart.planned_duration_s == 5.0 and timed_chart.data["hold"] is True
    assert timed_chart.end_s == pytest.approx(7.0)

    plain_chart = _beat("chart", "Notice rate: Students=25%, Guess=50%")
    assert plain_chart.planned_duration_s is None and "hold" not in plain_chart.data

    timed_timeline = _beat("timeline", "1987, *1992*, 2001 | Title | seconds=4")
    assert timed_timeline.planned_duration_s == 4.0 and timed_timeline.data["hold"] is True
    assert timed_timeline.text == "Title" and timed_timeline.end_s == pytest.approx(6.0)


def test_kinetic_directive_parses_marked_words_style_and_seconds_hold():
    """`[kinetic: text with *marks* | style=... | seconds=N]`
    (MOTION_GRAPHICS_PLAN.md M1) -> a kinetic_text beat presentation/motion.py
    renders; unknown/absent style falls back to "slam"."""
    from backend.presentation.models import Program

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(arg: str, at: float = 3.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind="kinetic", arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0]

    plain = _beat("This is *huge*")
    assert plain.kind == "kinetic_text"
    assert plain.text == "This is *huge*"
    assert plain.data == {"style": "slam"}
    assert plain.planned_duration_s is None
    assert plain.priority == 1.0 and plain.origin == "script"
    assert plain.end_s == pytest.approx(3.0 + script_stage.DIRECTIVE_KINETIC_SECONDS)

    styled = _beat("Breaking *news* today | style=typewriter")
    assert styled.text == "Breaking *news* today" and styled.data["style"] == "typewriter"

    bad_style = _beat("no such style | style=explode")
    assert bad_style.data["style"] == "slam"

    timed = _beat("Yeh *sach* hai | style=reveal | seconds=5")
    assert timed.data == {"style": "reveal", "hold": True}
    assert timed.planned_duration_s == 5.0
    assert timed.end_s == pytest.approx(8.0)


def test_kinetic_is_in_the_bracket_vocabulary_and_full_frame_cutaway_kinds():
    from backend.asr.script_align import DIRECTIVES
    from backend.presentation.models import CUTAWAY_KINDS

    assert "kinetic" in DIRECTIVES
    assert "kinetic_text" in CUTAWAY_KINDS


def test_stick_directive_parses_action_scene_prop_figures_caption_and_seconds_hold():
    """`[stick: action=<pose> | scene=<short English scene> | prop=<prop> |
    figures=<1-3> | caption=<optional short text> | seconds=N]`
    (MOTION_GRAPHICS_PLAN.md M7) -> a stick_scene beat presentation/motion.py
    renders; the scene text is never shown -- only `ground_for_scene` reads
    it (presentation/motion_stick.py)."""
    from backend.presentation.models import Program

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(arg: str, at: float = 3.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind="stick", arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0]

    plain = _beat("action=swim | scene=escaping a flooded house | prop=ship | figures=2 | caption=Help!")
    assert plain.kind == "stick_scene"
    assert plain.data["action"] == "swim"
    assert plain.data["scene"] == "escaping a flooded house"
    assert plain.data["prop"] == "ship"
    assert plain.data["figures"] == 2
    assert plain.data["caption"] == "Help!"
    assert plain.text == "Help!"
    assert plain.planned_duration_s is None
    assert plain.priority == 1.0 and plain.origin == "script"
    assert plain.end_s == pytest.approx(3.0 + script_stage.DIRECTIVE_STICK_SECONDS)

    bare = _beat("action=shrug")
    assert bare.data == {"action": "shrug", "scene": "", "prop": "none", "figures": 1}
    assert bare.text == ""

    clamped = _beat("action=cheer | figures=9")
    assert clamped.data["figures"] == 3

    timed = _beat("action=point | scene=in the office | seconds=4")
    assert timed.data["hold"] is True
    assert timed.planned_duration_s == 4.0
    assert timed.end_s == pytest.approx(7.0)


def test_stick_directive_parses_era_outfit_and_bg():
    """STICK_WORLD_PLAN.md SW2: `era`/`outfit`/`bg` on `[stick: ...]` --
    `outfit` is one spec per figure (`;` between figures), normalised in
    presentation/motion_stick.py, not here; `figures` defaults to the
    number of outfits named, same as before when neither is given."""
    from backend.presentation.models import Program

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(arg: str, at: float = 3.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind="stick", arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0]

    dressed = _beat("action=point | scene=a Greek library court | era=ancient_greece | "
                    "outfit=chiton+laurel+scroll+beard ; toga+staff | "
                    "bg=ancient Greek library courtyard, marble columns | "
                    "caption=Athens, 399 BC")
    assert dressed.data["era"] == "ancient_greece"
    assert dressed.data["outfits"] == ["chiton+laurel+scroll+beard", "toga+staff"]
    assert dressed.data["figures"] == 2  # defaults to the number of outfits
    assert dressed.data["bg"] == "ancient Greek library courtyard, marble columns"

    era_only = _beat("action=idle | era=prehistoric")
    assert era_only.data["era"] == "prehistoric"
    assert "outfits" not in era_only.data and "bg" not in era_only.data
    assert era_only.data["figures"] == 1


def test_stick_is_in_the_bracket_vocabulary_and_full_frame_cutaway_kinds():
    from backend.asr.script_align import DIRECTIVES
    from backend.presentation.models import CUTAWAY_KINDS

    assert "stick" in DIRECTIVES
    assert "stick_scene" in CUTAWAY_KINDS


def test_collage_directive_parses_prompts_caption_and_seconds_hold():
    """`[collage: prompt A ; prompt B ; prompt C | caption=... | seconds=N]`
    (MOTION_GRAPHICS_PLAN.md M5) -> a paper_collage beat carrying 2-4 prompts
    presentation/motion_collage.py::expand_collages turns into generated-but-
    not-placed broll_image sub-beats."""
    from backend.asr.script_align import DIRECTIVES
    from backend.presentation.models import CUTAWAY_KINDS, Program

    assert "collage" in DIRECTIVES
    assert "paper_collage" in CUTAWAY_KINDS

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(arg: str, at: float = 3.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind="collage", arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0] if out["beats"] else None

    plain = _beat("an old haveli at night ; a torn letter on the desk ; a lantern "
                  "by the well | caption=Three clues he missed")
    assert plain.kind == "paper_collage"
    assert plain.data["prompts"] == ["an old haveli at night", "a torn letter on the desk",
                                     "a lantern by the well"]
    assert plain.data["caption"] == "Three clues he missed"
    assert plain.planned_duration_s is None
    assert plain.priority == 1.0 and plain.origin == "script"
    assert plain.end_s == pytest.approx(3.0 + script_stage.DIRECTIVE_COLLAGE_SECONDS)

    timed = _beat("a quiet street ; a locked door | seconds=6")
    assert timed.data["hold"] is True
    assert timed.planned_duration_s == 6.0
    assert "caption" not in timed.data

    too_few = _beat("only one prompt")
    assert too_few is None

    capped = _beat("a ; b ; c ; d ; e")
    assert capped.data["prompts"] == ["a", "b", "c", "d"]


def test_flow_and_icons_directives_parse_steps_layout_mode_and_seconds_hold():
    """`[flow: step A ; step B ; step C | layout=chain|cycle|branch |
    seconds=N]` -> a flow_diagram beat and `[icons: label ; label ; ... |
    mode=list|pros_cons | seconds=N]` -> an icon_list beat
    (MOTION_GRAPHICS_PLAN.md M9); both render through
    presentation/motion_diagrams.py."""
    from backend.asr.script_align import DIRECTIVES
    from backend.presentation.models import CUTAWAY_KINDS, Program

    assert "flow" in DIRECTIVES and "icons" in DIRECTIVES
    assert "flow_diagram" in CUTAWAY_KINDS and "icon_list" in CUTAWAY_KINDS

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(kind: str, arg: str, at: float = 3.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind=kind, arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0] if out["beats"] else None

    flow = _beat("flow", "Collect data ; Clean it up ; Train the model | layout=cycle")
    assert flow.kind == "flow_diagram"
    assert flow.data["steps"] == ["Collect data", "Clean it up", "Train the model"]
    assert flow.data["layout"] == "cycle"
    assert flow.text == "Collect data -> Clean it up -> Train the model"
    assert flow.planned_duration_s is None
    assert flow.priority == 1.0 and flow.origin == "script"
    assert flow.end_s == pytest.approx(3.0 + script_stage.DIRECTIVE_FLOW_SECONDS)

    bad_layout = _beat("flow", "Step A ; Step B | layout=sideways")
    assert bad_layout.data["layout"] == "chain"

    timed_flow = _beat("flow", "Step A ; Step B ; Step C | seconds=6")
    assert timed_flow.data["hold"] is True
    assert timed_flow.planned_duration_s == 6.0

    too_few_steps = _beat("flow", "only one step")
    assert too_few_steps is None

    capped_steps = _beat("flow", "a ; b ; c ; d ; e ; f ; g")
    assert capped_steps.data["steps"] == ["a", "b", "c", "d", "e", "f"]

    icons = _beat("icons", "+Lower cost ; -Takes longer | mode=pros_cons")
    assert icons.kind == "icon_list"
    assert icons.data["items"] == ["+Lower cost", "-Takes longer"]
    assert icons.data["mode"] == "pros_cons"
    assert icons.text == "+Lower cost, -Takes longer"
    assert icons.end_s == pytest.approx(3.0 + script_stage.DIRECTIVE_ICONS_SECONDS)

    bad_mode = _beat("icons", "Label A ; Label B | mode=sideways")
    assert bad_mode.data["mode"] == "list"

    too_few_items = _beat("icons", "only one label")
    assert too_few_items is None


def test_ui_directive_parses_variant_text_name_handle_meta_and_seconds_hold():
    """`[ui: <variant> | text=... | name=... | handle=... | meta=... |
    seconds=N]` -> a ui_mockup beat (MOTION_GRAPHICS_PLAN.md M10), rendered
    by presentation/motion_ui.py::build_ui_mockup."""
    from backend.asr.script_align import DIRECTIVES
    from backend.presentation.models import CUTAWAY_KINDS, Program

    assert "ui" in DIRECTIVES
    assert "ui_mockup" in CUTAWAY_KINDS

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(arg: str, at: float = 3.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind="ui", arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0]

    post = _beat("post | text=Finally finished the big project | name=Rhea Kapoor | "
                 "handle=@rhea | meta=2h . 14.2K likes")
    assert post.kind == "ui_mockup"
    assert post.data["variant"] == "post"
    assert post.data["text"] == "Finally finished the big project"
    assert post.data["name"] == "Rhea Kapoor"
    assert post.data["handle"] == "@rhea"
    assert post.data["meta"] == "2h . 14.2K likes"
    assert post.planned_duration_s is None
    assert post.priority == 1.0 and post.origin == "script"
    assert post.end_s == pytest.approx(3.0 + script_stage.DIRECTIVE_UI_SECONDS)

    bad_variant = _beat("tweet | text=Something")
    assert bad_variant.data["variant"] == "post"

    timed = _beat("headline | text=Big news today | seconds=5")
    assert timed.data["hold"] is True
    assert timed.planned_duration_s == 5.0


def test_callout_directive_parses_prompt_mark_region_label_and_seconds_hold():
    """`[callout: <image prompt> | mark=circle|arrow|magnify | at=<3x3
    region> | label=... | seconds=N]` -> a callout beat (MOTION_GRAPHICS_
    PLAN.md M10) presentation/motion_ui.py::expand_callouts turns into a
    generated-but-not-placed broll_image sub-beat."""
    from backend.asr.script_align import DIRECTIVES
    from backend.presentation.models import CUTAWAY_KINDS, Program

    assert "callout" in DIRECTIVES
    assert "callout" in CUTAWAY_KINDS

    program = Program(duration_s=100.0, fps=30.0)
    settings = PresentationSettings()

    def _beat(arg: str, at: float = 3.0):
        ctx = script_stage.ScriptContext(
            directives=[script_stage.ScriptDirective(kind="callout", arg=arg, tl_at_s=at)])
        out = script_stage.directive_beats(ctx, program, settings)
        return out["beats"][0] if out["beats"] else None

    plain = _beat("a cluttered office desk with a laptop | mark=circle | at=top_right | "
                  "label=The sticky note")
    assert plain.kind == "callout"
    assert plain.image_prompt == "a cluttered office desk with a laptop"
    assert plain.data["mark"] == "circle"
    assert plain.data["at"] == "top_right"
    assert plain.data["label"] == "The sticky note"
    assert plain.planned_duration_s is None
    assert plain.priority == 1.0 and plain.origin == "script"
    assert plain.end_s == pytest.approx(3.0 + script_stage.DIRECTIVE_CALLOUT_SECONDS)

    bad_options = _beat("a phone screen | mark=zoom | at=nowhere | seconds=5")
    assert bad_options.data["mark"] == "circle"
    assert bad_options.data["at"] == "center"
    assert bad_options.data["hold"] is True
    assert bad_options.planned_duration_s == 5.0

    no_prompt = _beat("| mark=arrow")
    assert no_prompt is None
