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
