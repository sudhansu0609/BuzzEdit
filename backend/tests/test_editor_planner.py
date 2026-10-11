"""The AI editor as the product's planner: agreement, the review list, fallback, answers, nudges."""

import asyncio
import itertools
import json
import threading

import numpy as np
import pytest

from asr import auto_edit, double_check, editor, editor_planner, retake_sweep, review, style
from asr.utterances import Utterance


def _words():
    words = [{"word": f"w{i}", "word_native": f"w{i}", "start": float(i), "end": i + 0.5} for i in range(10)]
    # a filler sound the audio pass found: stays cut, stays out of the editor's table
    words.insert(4, {"word": "[uh]", "word_native": "[uh]", "start": 2.6, "end": 2.9,
                     "disfluency": True, "reason": "filler_sound", "detected": True})
    return words


def _utts():
    # indices are into the editor's view (the words without the detected filler)
    return [Utterance(f"U{k + 1}", start=2.0 * k, end=2.0 * k + 1.5, word_indices=[2 * k, 2 * k + 1],
                      text=f"w{2 * k} w{2 * k + 1}", pause_before=0.5) for k in range(5)]


class _Reads:
    def __init__(self, answers, fail=0):
        self.answers = itertools.cycle(answers)
        self.fail = fail
        self.lock = threading.Lock()

    def ask(self, system, user, retries=1):
        with self.lock:
            if self.fail:
                self.fail -= 1
                raise RuntimeError("proxy served 'gpt-5' instead of 'claude-opus-5-5'")
            answer = next(self.answers)
        return json.dumps(answer), {"model": "claude-opus-5-5", "usage": {"prompt_tokens": 10,
                                                                          "completion_tokens": 5}}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """No audio analysis, no models: fixed speech map and utterances."""
    from asr import fumble_engine, gap_recovery, utterances
    import faster_whisper.audio

    def fake_audio_cuts(words, audio_path, report, detect_fillers=True):
        report.used_audio = True
        report.speech = [(0.0, 10.0)]
        report.energy = {"rate": 50, "db": [0] * 10}
        return words

    monkeypatch.setattr(fumble_engine, "plan_audio_cuts", fake_audio_cuts)
    monkeypatch.setattr(faster_whisper.audio, "decode_audio", lambda *a, **k: np.zeros(16000))
    monkeypatch.setattr(gap_recovery, "speech_regions", lambda audio: [(0.0, 10.0)])
    monkeypatch.setattr(utterances, "build", lambda words, regions, audio=None, script_text=None: _utts())
    monkeypatch.setattr(style, "store_path", lambda: tmp_path / "examples.jsonl")


def _plan(client, **kw):
    return editor_planner.plan_with_editor_sync(_words(), "x.wav", {"genre": "vlog"}, client=client, **kw)


def test_a_majority_of_reads_decides_and_nothing_is_left_in_doubt(offline):
    """2026-10-08: a line goes when more than half the reads removed it. Unanimity had kept
    an abandoned first attempt of a whole video (two reads cut it, one did not) and shipped
    the intro twice. A minority removal is kept with a note, and no review item is made."""
    a = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2", "why": "U1 stops mid-sentence"}],
         "drops": [{"utterances": ["U4"], "kind": "chatter", "why": "talking to the camera op"}],
         "trims": [{"utterance": "U5", "remove": "w8"}]}
    b = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}],
         "drops": [{"utterances": ["U3"], "kind": "false_start"}]}
    words = _plan(_Reads([a, b, a]))
    by = {w["word"]: w for w in words}
    assert not by["w0"]["enabled"] and by["w0"]["reason"] == "retake"
    assert "stops mid-sentence" in by["w0"]["note"] and by["w0"]["take"] == "U1"
    assert by["w2"]["enabled"]                                   # U2, the kept take
    assert not by["w6"]["enabled"] and by["w6"]["reason"] == "chatter"   # U4: 2 of 3 -> cut
    assert by["w4"]["enabled"] and not by["w4"].get("candidate")         # U3: 1 of 3 -> kept
    assert by["w4"]["note"].startswith("kept: 1 of 3 reads removed it")
    assert not by["w8"]["enabled"] and by["w9"]["enabled"]       # the trim 2 of 3 reads made
    assert not by["[uh]"]["enabled"]                             # the audio pass's filler stays cut
    report = words[0]["_edit_report"]
    assert report["planner"] == "editor" and report["editor"]["reads"] == 3
    assert report["editor"]["removed_takes"] == 2 and report["editor"]["kept_by_minority"] == 1
    assert report["review"] == []
    assert report["speech"] and report["energy"]                 # the timeline still gets them


def test_a_second_greeting_early_on_cuts_everything_before_it(offline, monkeypatch):
    """The creator restarts a video from its fixed greeting. On ep2 the editor's
    reads caught the duplicate intro in one run and missed it in the next, so
    the greeting decides: a second greeting in the first half followed by the
    opening said again, and all before the last one is an earlier attempt. The
    sign-off at the end also says the words, and is left alone."""
    from asr import utterances as utt_mod
    from asr.restart import is_greeting, restart_from_top

    greeting = "Hello doston mera naam hai Sudhansu aur aap dekh rahe hain Raat 3 Baje"
    lines = [greeting,                                                     # the first attempt's intro
             "aaj ki kahani ek purani haveli ki hai",
             greeting,                                                     # the restart, at 4 s of 10
             "aaj ki kahaani ek puraani haveli ki hai",                    # its opening, said again
             "mera naam hai Sudhansu aur aap dekh rahe the Raat 3 Baje"]   # the sign-off
    words, utts = [], []
    for k, line in enumerate(lines):
        first = len(words)
        words += [{"word": t, "word_native": t, "start": 2.0 * k + 0.1 * j, "end": 2.0 * k + 0.1 * j + 0.08}
                  for j, t in enumerate(line.split())]
        utts.append(Utterance(f"U{k + 1}", start=2.0 * k, end=2.0 * k + 1.5,
                              word_indices=list(range(first, len(words))), text=line, pause_before=0.5))
    assert is_greeting(greeting) and not is_greeting("mere saath Vikram naam ka ek purana ranger tha")
    assert set(restart_from_top(utts, 10.0)) == set(restart_from_top(utts, 10.0, words)) == {"U1", "U2"}
    monkeypatch.setattr(utt_mod, "build", lambda words, regions, audio=None, script_text=None: utts)

    words = editor_planner.plan_with_editor_sync(words, "x.wav", {"genre": "vlog"},
                                                 client=_Reads([{}, {}, {}]))
    at = {u.uid: [words[i] for i in u.word_indices] for u in utts}
    assert not any(w["enabled"] for w in at["U1"] + at["U2"])         # before the restart
    assert "restart from the top at U3" in at["U1"][0]["note"]
    assert at["U3"][0]["enabled"] and at["U5"][0]["enabled"]          # the restart, and the sign-off
    assert words[0]["_edit_report"]["editor"]["restart_cut"] == 2


def test_a_trim_takes_the_untranscribed_speech_between_its_words(offline, monkeypatch):
    """A stumble the transcript only half heard: the reads trim "w7 [speech, no transcript] w8",
    and the speech between the two words, which no word covers, is cut with them."""
    from asr import gap_recovery, utterances as utt_mod
    utts = _utts()[:3] + [Utterance("U4", start=6.0, end=9.5, word_indices=[6, 7, 8, 9],
                                    text="w6 w7 w8 w9", pause_before=0.5)]
    monkeypatch.setattr(utt_mod, "build", lambda words, regions, audio=None, script_text=None: utts)
    # speech under every word, and at 7.5-8.0 between w7 and w8 where no word is
    monkeypatch.setattr(gap_recovery, "speech_regions", lambda audio: [
        (0.0, 1.5), (2.0, 3.5), (4.0, 5.5), (6.0, 6.5), (7.0, 8.5), (9.0, 9.5)])
    trim = {"trims": [{"utterance": "U4", "remove": "w7 [speech, no transcript, 0.5 s] w8",
                       "kind": "false_start"}]}
    words = _plan(_Reads([trim, trim, {}]))
    holes = [(w["start"], w["end"], w["enabled"], w["reason"], w["take"]) for w in words if w["word"] == "[…]"]
    assert holes == [(7.5, 8.0, False, "false_start", "U4")]
    by = {w["word"]: w for w in words}
    assert not by["w7"]["enabled"] and not by["w8"]["enabled"]
    assert by["w6"]["enabled"] and by["w9"]["enabled"]
    assert words[0]["_edit_report"]["editor"]["untranscribed_cut_s"] == 0.5


def test_the_projects_script_reaches_the_utterance_table(offline, monkeypatch):
    from asr import utterances as utt_mod
    seen = []

    def build(words, regions, audio=None, script_text=None):
        seen.append(script_text)
        return _utts()

    monkeypatch.setattr(utt_mod, "build", build)
    script = "**VISUAL:** [Pitch black.]\n\n**V.O. (Narrator):** Raat ke sawa do baje (pause) ek guard."
    editor_planner.plan_with_editor_sync(_words(), "x.wav", {"genre": "vlog", "script_text": script},
                                         client=_Reads([{}, {}, {}]))
    _plan(_Reads([{}, {}, {}]))
    assert seen[0] == utt_mod.spoken_script(script) and "Pitch black" not in seen[0]
    assert seen[1] is None                                       # no script: no hints


def test_two_reads_must_both_agree_and_a_lone_read_cannot_cut(offline):
    a = {"drops": [{"utterances": ["U3"], "kind": "false_start"}]}
    words = _plan(_Reads([a, {}], fail=1))                       # one read failed: 2 answered
    assert next(w for w in words if w["word"] == "w4")["enabled"]      # 1 of 2 is not a majority
    words = _plan(_Reads([a, a], fail=1))
    assert not next(w for w in words if w["word"] == "w4")["enabled"]  # 2 of 2 is


def test_the_rule_based_stutter_pass_runs_over_what_the_editor_kept(offline):
    """The editor cuts a stutter only when enough reads quote the same words, which left
    "jo jo" / "thi thi" / "bahut" x10 in one cut. The deterministic rule runs after it."""
    from asr import editor_planner as ep

    def plan():
        words = _words()
        words[2]["word"] = words[2]["word_native"] = "w3"        # "w3 w3": the first copy goes
        return ep.plan_with_editor_sync(words, "x.wav", {"genre": "vlog"}, client=_Reads([{}, {}, {}]))

    words = plan()
    doubled = [w for w in words if w["word"] == "w3"]
    assert [w["enabled"] for w in doubled] == [False, True]
    assert doubled[0]["reason"] == "stutter"
    assert words[0]["_edit_report"]["editor"]["stutters"] == 1


def test_one_failed_read_still_leaves_an_agreement(offline):
    a = {"drops": [{"utterances": ["U3"], "kind": "false_start"}]}
    words = _plan(_Reads([a], fail=1))
    report = words[0]["_edit_report"]
    assert report["editor"]["reads"] == 2 and report["editor"]["reads_failed"] == 1
    assert not next(w for w in words if w["word"] == "w4")["enabled"]


def test_too_few_reads_or_no_key_means_the_editor_is_unavailable(offline, monkeypatch):
    with pytest.raises(editor_planner.EditorUnavailable, match="instead of"):
        _plan(_Reads([{}], fail=2))
    monkeypatch.setattr(editor, "proxy_config", lambda settings=None: {"base_url": "x", "api_key": ""})
    with pytest.raises(editor_planner.EditorUnavailable, match="no key"):
        editor_planner.plan_with_editor_sync(_words(), "x.wav", {})


def test_the_classic_planner_takes_over_and_the_report_says_so(monkeypatch):
    async def unavailable(words, audio_path, settings):
        raise editor_planner.EditorUnavailable("no key for the Claude proxy")

    async def classic(words, **kw):
        words = [dict(w, enabled=True) for w in words]
        words[0]["_edit_report"] = {"used_fluency": True}
        return words

    monkeypatch.setattr(editor_planner, "plan_with_editor", unavailable)
    monkeypatch.setattr(auto_edit, "refine_disfluencies", classic)
    monkeypatch.setattr(auto_edit, "release_asr_gpu", lambda: 0.0)
    plan = asyncio.run(auto_edit.plan_auto_edit(_words(), "x.wav", settings={"planner": "editor", "genre": "vlog"}))
    assert plan.report["planner"] == "classic" and "no key" in plan.report["planner_fallback"]
    assert any("AI editor could not run" in w for w in auto_edit.public_report(plan.report)["warnings"])


def test_planner_choice_project_then_app_then_editor(monkeypatch):
    from store.app_settings import AppSettings
    monkeypatch.setattr(AppSettings, "get", lambda self, key, default=None: "classic"
                        if key == "auto_cut_planner" else default)
    assert auto_edit.choose_planner({"planner": "editor"}) == "editor"
    assert auto_edit.choose_planner({}) == "classic"
    monkeypatch.setattr(AppSettings, "get", lambda self, key, default=None: default)
    assert auto_edit.choose_planner(None) == "editor"


def _timeline(words):
    from timeline import build_timeline_from_transcript
    return build_timeline_from_transcript("src.mp4", 10.0, words, speech_regions=[(0.0, 10.0)])


def test_answering_a_review_item_flips_the_whole_take_and_becomes_an_example(offline):
    # The planner no longer lists doubt items (the majority decides); the answer path stays
    # for items other checks add, so one is built by hand here.
    words = _plan(_Reads([{}, {}, {}]))
    tl = _timeline(words)
    tl.review = [{"take": "U4", "start": 6.0, "end": 7.5, "text": "w6 w7", "decision": "kept",
                  "votes": "1 of 3 reads removed it", "why": "talking to the camera op",
                  "twin": None, "twin_start": None, "context": "U4 w6 w7", "kind": "doubt",
                  "answer": None}]
    primary = next(iter(tl.sources))
    result = review.answer(tl, "U4", "cut", primary, channel="life3baje", source="proj1")
    assert result == {"changed": 2, "example": True}
    assert all(not w.enabled and w.reason == "review" for w in tl.words if w.take == "U4")
    assert tl.review[0]["answer"] == "cut" and review.open_items(tl) == []
    [example] = style.load_examples()
    assert example["kind"] == "correction" and example["channel"] == "life3baje"
    assert "What the creator did: U4: removed" in example["text"]
    with pytest.raises(KeyError):
        review.answer(tl, "U1", "keep", primary)
    with pytest.raises(ValueError):
        review.answer(tl, "U4", "maybe", primary)


def test_stored_style_puts_corrections_and_this_channel_first_and_leaves_one_out(offline):
    style.add_examples([
        {"kind": "hand_cut", "channel": "raat3baje", "source": "raat3baje_ep1", "text": "R-hand"},
        {"kind": "hand_cut", "channel": "life3baje", "source": "life3baje_ep1", "text": "L-hand"},
        {"kind": "correction", "channel": "life3baje", "source": "p9", "text": "L-fix"},
    ])
    text = style.stored_style("life3baje", notes="NOTES")
    assert text.split("\n\n") == ["NOTES", "L-fix", "L-hand", "R-hand"]
    assert "L-hand" not in style.stored_style("life3baje", exclude_source="life3baje_ep1")
    style.add_examples([{"kind": "hand_cut", "channel": "life3baje", "source": "life3baje_ep1",
                         "text": "L-hand-2"}], replace_source="life3baje_ep1")
    assert [e["text"] for e in style.load_examples()].count("L-hand") == 0


def test_a_small_leak_is_nudged_out_and_a_big_one_is_listed():
    from asr.cut_verify import nudge_cut_edges, struck_audio_leak

    words = [{"word": "a", "start": 0.0, "end": 1.0}, {"word": "b", "start": 1.0, "end": 2.0, "disfluency": True,
             "reason": "retake", "take": "U2"}, {"word": "c", "start": 2.0, "end": 3.0},
             {"word": "d", "start": 3.0, "end": 5.0, "disfluency": True, "reason": "retake", "take": "U4"},
             {"word": "e", "start": 5.0, "end": 6.0}]
    tl = _timeline(words)
    fps = tl.fps_num / tl.fps_den
    a1 = sorted([i for i in tl.items if i.track in ("A1", "V1")], key=lambda i: i.timeline_start_frame)
    # widen the kept span before "b" by 0.3 s into it (small) and the one before "d" by 1 s (big)
    for item in a1:
        if item.source_end_frame == int(round(1.0 * fps)):
            item.source_end_frame = int(round(1.3 * fps))
        elif item.source_end_frame == int(round(3.0 * fps)):
            item.source_end_frame = int(round(4.0 * fps))
    before = struck_audio_leak(tl)["leaked_struck_words"]
    assert before == ["b", "d"]
    out = nudge_cut_edges(tl, next(iter(tl.sources)))
    assert out["nudged"] >= 1 and out["leaks_listed"] == 1
    assert [r["take"] for r in out["review"] if r["kind"] == "leak"] == ["U4"]


class _Desk:
    """A fake EditorClient answering by the prompt it was given: the reads cycle through their
    answers; the sweep, the verifier and the final read each get their own. A prompt in `fail`
    raises, as a proxy that is down would."""

    model, effort = "claude-opus-5-5", "high"

    def __init__(self, reads, sweep=None, verify=None, final=None, fail=()):
        self.reads = itertools.cycle(reads)
        self.answers = {retake_sweep.SWEEP_SYSTEM: sweep or {}, double_check.VERIFY_SYSTEM: verify or {},
                        double_check.FINAL_READ_CHUNKED: final or {}}
        self.fail = set(fail)
        self.lock = threading.Lock()
        self.seen = []

    def ask(self, system, user, retries=1):
        with self.lock:
            self.seen.append((system, user))
            if system in self.fail:
                raise RuntimeError("editor unavailable: proxy down")
            answer = self.answers[system] if system in self.answers else next(self.reads)
        return json.dumps(answer), {"model": "claude-opus-5-5", "usage": {"prompt_tokens": 10,
                                                                          "completion_tokens": 5}}

    def users(self, system):
        return [user for s, user in self.seen if s == system]


def test_a_minority_removal_goes_only_when_the_verifier_finds_its_content_kept(offline):
    """V3 2.2: a line one read of three removed gets one focused question. It goes when a kept
    line says the same thing, and stays, with the reason the verifier gave, when it does not."""
    reads = [{"drops": [{"utterances": ["U2"], "kind": "false_start"}]},
             {"drops": [{"utterances": ["U4"], "kind": "chatter"}]}, {}]
    verify = {"U2": {"verdict": "remove", "covered_by": "U3", "why": "U3 says it again, complete"},
              "U4": {"verdict": "keep", "why": "the only place the name is said"}}
    desk = _Desk(reads, verify=verify)
    words = _plan(desk)
    by = {w["word"]: w for w in words}
    assert not by["w2"]["enabled"] and not by["w3"]["enabled"] and by["w2"]["reason"] == "false_start"
    assert by["w2"]["note"] == "U3 says it again, complete"
    assert by["w6"]["enabled"] and by["w6"]["note"] == "kept after verification -- the only place the name is said"
    [asked] = desk.users(double_check.VERIFY_SYSTEM)
    assert ">> U2" in asked and ">> U4" in asked and "(kept) U3" in asked
    ed = words[0]["_edit_report"]["editor"]
    assert (ed["candidates"], ed["verified_removed"], ed["verified_kept"]) == (2, 1, 1)
    assert ed["removed_takes"] == 1 and ed["kept_by_minority"] == 1 and ed["steps_failed"] == []
    # every call is counted: three reads, the sweep, the verifier, the final read
    assert (ed["tokens_in"], ed["tokens_out"]) == (60, 30)
    assert ed["model"] == "claude-opus-5-5" and ed["effort"] == "high"
    assert words[0]["_edit_report"]["review"] == []


def test_the_retake_sweep_puts_an_earlier_attempt_to_the_verifier(offline):
    """V3 2.3: a retake no read removed. The sweep groups U1 with U3; the earlier attempt is a
    candidate (the later one is the keep) and the verifier removes it."""
    sweep = {"groups": [{"utterances": ["U1", "U3"], "why": "the same line twice"}]}
    verify = {"U1": {"verdict": "remove", "covered_by": "U3", "why": "U3 is the complete attempt"}}
    desk = _Desk([{}, {}, {}], sweep=sweep, verify=verify)
    words = _plan(desk)
    by = {w["word"]: w for w in words}
    assert not by["w0"]["enabled"] and by["w0"]["reason"] == "retake"
    assert by["w0"]["note"] == "U3 is the complete attempt"
    assert by["w4"]["enabled"]                                   # U3, the later attempt
    [asked] = desk.users(double_check.VERIFY_SYSTEM)
    assert ">> U1" in asked and "(kept) U3" in asked
    ed = words[0]["_edit_report"]["editor"]
    assert (ed["sweep_groups"], ed["sweep_calls"], ed["candidates"], ed["verified_removed"]) == (1, 1, 1, 1)


def test_when_both_attempts_would_go_the_later_one_stays(offline):
    reads = [{"drops": [{"utterances": ["U2"]}]}, {"drops": [{"utterances": ["U4"]}]}, {}]
    verify = {"U2": {"verdict": "remove", "covered_by": "U4"}, "U4": {"verdict": "remove", "covered_by": "U2"}}
    words = _plan(_Desk(reads, verify=verify))
    by = {w["word"]: w for w in words}
    assert not by["w2"]["enabled"] and by["w6"]["enabled"]
    assert "the later attempt" in by["w6"]["note"]
    assert words[0]["_edit_report"]["editor"]["verified_removed"] == 1
    # the other way round: the line it leans on is an earlier majority removal
    reads = [{"drops": [{"utterances": ["U2", "U4"]}]}, {"drops": [{"utterances": ["U2"]}]}, {}]
    words = _plan(_Desk(reads, verify={"U4": {"verdict": "remove", "covered_by": "U2"}}))
    by = {w["word"]: w for w in words}
    assert not by["w2"]["enabled"] and by["w6"]["enabled"]


def test_the_final_read_restores_and_removes_only_against_a_kept_line(offline, monkeypatch):
    """V3 2.8: a removal needs a kept line that carries it; a restore brings a cut line back,
    except the lines the restart rule cut."""
    from asr import restart, utterances as utt_mod
    utts = _utts()
    # a restart at U3: U1 and U2 are the first attempt (the rule itself: test_restart.py)
    note = "restart from the top at U3 (4s): an earlier attempt"
    monkeypatch.setattr(restart, "restart_from_top", lambda utterances, total_seconds, words=None:
                        {"U1": note, "U2": note})
    monkeypatch.setattr(utt_mod, "build", lambda words, regions, audio=None, script_text=None: utts)
    chatter = {"drops": [{"utterances": ["U4"], "kind": "chatter"}]}
    final = {"remove": [{"utterance": "U3", "covered_by": "U4", "why": "U4 is cut: refused"},
                        {"utterance": "U5", "covered_by": "U3", "why": "U3 says it"}],
             "restore": ["U4", "U2"],
             "trims": [{"utterance": "U3", "remove": "w5", "why": "a doubled word"}]}
    desk = _Desk([chatter, chatter, {}], final=final)
    words = _plan(desk)
    by = {w["word"]: w for w in words}
    assert by["w4"]["enabled"] and not by["w5"]["enabled"] and by["w5"]["reason"] == "stutter"
    assert not by["w8"]["enabled"] and by["w8"]["reason"] == "final_read"      # U5, covered by U3
    assert by["w6"]["enabled"] and by["w6"]["note"] == "restored by the final read"   # U4 is back
    assert not by["w0"]["enabled"] and not by["w2"]["enabled"]      # U1, U2: the restore is refused
    [asked] = desk.users(double_check.FINAL_READ_CHUNKED)
    kept_part, removed_part = asked.split("Removed lines (for restoring only):")
    assert "U3 [" in kept_part and "U5 [" in kept_part and "U4 [" not in kept_part
    assert "U4 [" in removed_part and "U2 [" not in removed_part
    ed = words[0]["_edit_report"]["editor"]
    assert (ed["final_read_removed"], ed["final_read_restored"], ed["final_read_trims"]) == (1, 1, 1)
    assert ed["restart_cut"] == 2 and ed["removed_takes"] == 3


def test_the_report_counts_repeats_and_false_starts_that_still_play(offline, monkeypatch):
    """V3 2.10: what the creator measures a cut by. In a storytelling genre a clean, paused
    phrase repeat goes to the verifier; kept, it is counted, and so is a broken-off line whose
    restart also plays. public_report says both."""
    from asr import utterances as utt_mod
    utts = _utts()
    utts[0].notes = ["cut off mid-sound"]
    utts[1].same_opening = ["U1"]                    # U2 starts U1 over, and both play
    monkeypatch.setattr(utt_mod, "build", lambda words, regions, audio=None, script_text=None: utts)

    def plan(verify):
        words = _words()
        for k, text in ((2, "ab"), (3, "cd"), (5, "ab"), (6, "cd")):    # U2 then U3: "ab cd ab cd"
            words[k]["word"] = words[k]["word_native"] = text
        desk = _Desk([{}, {}, {}], verify=verify)
        return editor_planner.plan_with_editor_sync(words, "x.wav", {"genre": "horror"}, client=desk), desk

    words, desk = plan({})
    [asked] = desk.users(double_check.VERIFY_SYSTEM)
    assert "R1: \"ab cd\"" in asked and ">> U2" in asked
    ed = words[0]["_edit_report"]["editor"]
    assert ed["repeats_remaining"] == 1 and ed["repeats_examples"][0]["start"] == 2.0
    assert ed["false_starts_remaining"] == 1 and ed["false_starts_examples"][0]["take"] == "U1"
    warnings = auto_edit.public_report(words[0]["_edit_report"])["warnings"]
    assert "1 exact repeats remain in the cut (first at 2s)" in warnings
    assert "1 false starts remain in the cut" in warnings

    words, _ = plan({"R1": {"verdict": "remove", "why": "a stumble"}})
    first = next(w for w in words if w["start"] == 2.0)
    assert not first["enabled"] and first["reason"] == "repeat"
    ed = words[0]["_edit_report"]["editor"]
    assert ed["repeats_remaining"] == 0 and ed["phrase_repeats_cut"] == 1
    warnings = auto_edit.public_report(words[0]["_edit_report"]).get("warnings") or []
    assert not any("exact repeats" in w for w in warnings)


def test_a_failing_extra_call_leaves_the_edit_standing_and_is_reported(offline):
    """Every call after the reads is wrapped: the sweep, the verifier or the final read failing
    is listed in steps_failed, and the cut the reads agreed on ships."""
    chatter = {"drops": [{"utterances": ["U4"], "kind": "chatter"}]}
    words = _plan(_Desk([chatter, chatter, {}], fail={retake_sweep.SWEEP_SYSTEM}))
    by = {w["word"]: w for w in words}
    assert not by["w6"]["enabled"] and by["w0"]["enabled"]
    ed = words[0]["_edit_report"]["editor"]
    assert ed["steps_failed"] == ["sweep"] and ed["sweep_calls"] == 1 and ed["removed_takes"] == 1

    lone = {"drops": [{"utterances": ["U2"], "kind": "false_start"}]}
    down = {retake_sweep.SWEEP_SYSTEM, double_check.VERIFY_SYSTEM, double_check.FINAL_READ_CHUNKED}
    words = _plan(_Desk([chatter, lone, chatter], fail=down))
    by = {w["word"]: w for w in words}
    assert not by["w6"]["enabled"]                                   # U4: 2 of 3 reads
    assert by["w2"]["enabled"] and by["w2"]["note"].startswith("kept: 1 of 3 reads removed it")
    assert words[0]["_edit_report"]["editor"]["steps_failed"] == ["sweep", "verify", "final_read"]
