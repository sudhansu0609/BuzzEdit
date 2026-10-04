"""The AI editor as the product's planner: agreement, the review list, fallback, answers, nudges."""

import asyncio
import itertools
import json
import threading

import numpy as np
import pytest

from asr import auto_edit, editor, editor_planner, review, style
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


def test_only_what_every_read_removed_is_cut_and_the_rest_is_listed(offline):
    a = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2", "why": "U1 stops mid-sentence"}],
         "drops": [{"utterances": ["U4"], "kind": "chatter", "why": "talking to the camera op"}],
         "trims": [{"utterance": "U5", "remove": "w8"}]}
    b = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}],
         "trims": [{"utterance": "U5", "remove": "w8"}]}
    words = _plan(_Reads([a, b, a]))
    by = {w["word"]: w for w in words}
    assert not by["w0"]["enabled"] and by["w0"]["reason"] == "retake"
    assert "stops mid-sentence" in by["w0"]["note"] and by["w0"]["take"] == "U1"
    assert by["w2"]["enabled"]                                   # U2, the kept take
    assert by["w6"]["enabled"] and by["w6"]["candidate"]         # U4: 2 of 3 reads -> kept, flagged
    assert not by["w8"]["enabled"] and by["w9"]["enabled"]       # the trim every read made
    assert not by["[uh]"]["enabled"]                             # the audio pass's filler stays cut
    report = words[0]["_edit_report"]
    assert report["planner"] == "editor" and report["editor"]["reads"] == 3
    assert report["editor"]["removed_takes"] == 1 and report["editor"]["kept_under_doubt"] == 1
    [item] = report["review"]
    assert item["take"] == "U4" and item["decision"] == "kept" and item["votes"] == "2 of 3 reads removed it"
    assert "U4" in item["context"] and item["answer"] is None
    assert report["speech"] and report["energy"]                 # the timeline still gets them


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
    a = {"drops": [{"utterances": ["U4"], "kind": "chatter", "why": "talking to the camera op"}]}
    words = _plan(_Reads([a, {}, a]))
    tl = _timeline(words)
    tl.review = list(words[0]["_edit_report"]["review"])
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
