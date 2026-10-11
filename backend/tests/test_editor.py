"""The editor's answer contract: delete-only, validated, later take on ties, keeping wins."""

import json

import pytest

from asr import editor
from asr.utterances import Utterance


def _setup():
    words = [{"word": t, "word_native": t, "start": float(i), "end": i + 0.5}
             for i, t in enumerate("dosto kya ap dosto kya aapko pata hai main main bolunga".split())]
    utts = [
        Utterance("U1", 0.0, 2.5, [0, 1, 2], "dosto kya ap", 0.0),
        Utterance("U2", 3.0, 7.5, [3, 4, 5, 6, 7], "dosto kya aapko pata hai", 0.5),
        Utterance("U3", 8.0, 10.5, [8, 9, 10], "main main bolunga", 0.5),
        Utterance("U4", 11.0, 12.4, [], "[speech, no transcript, 1.4 s]", 0.5),
    ]
    return words, utts


def test_a_group_removes_every_attempt_but_the_kept_one():
    words, utts = _setup()
    d = editor.parse({"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}]}, utts, words)
    assert sorted(d.removed_words) == [0, 1, 2]
    assert set(d.removed_words.values()) == {"retake"}


def test_a_group_without_a_valid_keep_keeps_the_later_take():
    words, utts = _setup()
    d = editor.parse({"groups": [{"utterances": ["U1", "U2"], "keep": "U9"}]}, utts, words)
    assert sorted(d.removed_words) == [0, 1, 2]                   # U2, the later take, stays
    assert any("not a member" in r for r in d.refused)


def test_trims_must_quote_words_that_are_really_there():
    words, utts = _setup()
    ok = editor.parse({"trims": [{"utterance": "U3", "remove": "main"}]}, utts, words)
    assert sorted(ok.removed_words) == [8]
    invented = editor.parse({"trims": [{"utterance": "U3", "remove": "hum log"}]}, utts, words)
    assert invented.removed_words == {} and invented.refused
    whole = editor.parse({"trims": [{"utterance": "U3", "remove": "main main bolunga"}]}, utts, words)
    assert whole.removed_words == {} and "whole utterance" in whole.refused[0]


def test_a_trim_that_copies_the_tables_placeholder_matches_the_words_around_it():
    """Every sampled refusal on Raat3Baje ep2 was a trim whose quote copied the
    "[speech, no transcript, X s]" placeholder: the stumble was real, the quote is its words."""
    words, utts = _setup()
    middle = editor.parse({"trims": [{"utterance": "U2", "remove": "kya [speech, no transcript, 1.2 s] aapko"}]},
                          utts, words)
    assert sorted(middle.removed_words) == [4, 5] and not middle.refused
    end = editor.parse({"trims": [{"utterance": "U3", "remove": "main [speech, no transcript, 0.4 s]"}]},
                       utts, words)
    assert sorted(end.removed_words) == [8] and not end.refused


def test_a_trailing_comma_in_the_quote_still_matches():
    pairs = [("unhone", "उन्होंने"), ("kaha", "कहा"), ("ki", "कि"), ("vo", "वो"), ("aayenge", "आएंगे")]
    words = [{"word": r, "word_native": n, "start": float(i), "end": i + 0.5} for i, (r, n) in enumerate(pairs)]
    utts = [Utterance("U1", 0.0, 4.5, [0, 1, 2, 3, 4], "उन्होंने कहा कि वो आएंगे", 0.0)]
    d = editor.parse({"trims": [{"utterance": "U1", "remove": "उन्होंने, कहा"}]}, utts, words)
    assert sorted(d.removed_words) == [0, 1] and not d.refused


def test_a_one_letter_slip_matches_but_invented_words_and_whole_lines_do_not():
    words, utts = _setup()
    slip = editor.parse({"trims": [{"utterance": "U2", "remove": "dosti kya aapko"}]}, utts, words)
    assert sorted(slip.removed_words) == [3, 4, 5] and not slip.refused
    invented = editor.parse({"trims": [{"utterance": "U2", "remove": "hum log yahan aaye"}]}, utts, words)
    assert invented.removed_words == {} and invented.refused
    slips = editor.parse({"trims": [{"utterance": "U2", "remove": "dosti kya aapka pataa"}]}, utts, words)
    assert slips.removed_words == {} and slips.refused             # 3 slips in 4 tokens: not a quote
    whole = editor.parse({"trims": [{"utterance": "U2", "remove": "dosti kya aapko pata hai"}]}, utts, words)
    assert whole.removed_words == {} and "whole utterance" in whole.refused[0]


def test_applied_trims_record_their_word_range():
    words, utts = _setup()
    d = editor.parse({"trims": [{"utterance": "U3", "remove": "main"},
                                {"utterance": "U2", "remove": "kya [speech, no transcript, 1.2 s] aapko"},
                                {"utterance": "U2", "remove": "hum log"}]}, utts, words)
    assert d.trim_ranges == [("U3", 8, 8), ("U2", 4, 5)] and d.trims == 2


def _one_line(text):
    """U1 alone; a "[speech, no transcript, 0.8 s]" hole in its text is no word, so its word
    indices run straight across it."""
    tokens = text.replace("[speech, no transcript, 0.8 s]", " ").split()
    words = [{"word": t, "word_native": t, "start": float(i), "end": i + 0.5} for i, t in enumerate(tokens)]
    return words, [Utterance("U1", 0.0, len(tokens) - 0.5, list(range(len(tokens))), text, 0.0)]


def test_a_long_trim_removes_the_earlier_copy_of_a_doubled_phrase_even_when_the_later_is_quoted():
    words, utts = _one_line("toh main aapko ek kahani suna main aapko ek kahani sunata hoon")
    d = editor.parse({"trims": [{"utterance": "U1", "remove": "main aapko ek kahani sunata"}]}, utts, words)
    assert sorted(d.removed_words) == [1, 2, 3, 4, 5] and not d.refused        # the fuller copy stays
    assert d.trim_ranges == [("U1", 1, 5)] and set(d.removed_words.values()) == {"stutter"}


def test_a_long_trim_of_a_phrase_said_once_is_refused():
    """Measured 2026-10-09: long trims cut real content, e.g. the channel name."""
    words, utts = _one_line("हेलो दोस्तों मेरा चैनल है खयाल तीन बजे और आज मैं आपको एक कहानी सुनाता हूँ")
    d = editor.parse({"trims": [{"utterance": "U1", "remove": "मेरा चैनल है खयाल तीन"}]}, utts, words)
    assert d.removed_words == {} and d.trims == 0 and d.trim_ranges == []
    assert d.refused == ["trim on U1: 'मेरा चैनल है खयाल तीन' (5 words) is not a repeated phrase; "
                         "a trim removes only a stutter or a doubled phrase"]


def test_a_short_trim_still_removes_exactly_the_quoted_words():
    words, utts = _one_line("हेलो दोस्तों मेरा चैनल है खयाल तीन बजे और आज मैं आपको एक कहानी सुनाता हूँ")
    d = editor.parse({"trims": [{"utterance": "U1", "remove": "मेरा चैनल"}]}, utts, words)
    assert sorted(d.removed_words) == [2, 3] and d.trim_ranges == [("U1", 2, 3)] and not d.refused


def test_the_copies_of_a_doubled_phrase_may_sit_either_side_of_an_untranscribed_hole():
    words, utts = _one_line("toh main aapko ek kahani suna [speech, no transcript, 0.8 s] "
                            "main aapko ek kahani sunata hoon")
    d = editor.parse({"trims": [{"utterance": "U1", "remove": "main aapko ek kahani suna"}]}, utts, words)
    assert sorted(d.removed_words) == [1, 2, 3, 4, 5] and d.trim_ranges == [("U1", 1, 5)] and not d.refused


def test_keeping_wins_over_removing():
    words, utts = _setup()
    d = editor.parse({"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}],
                      "drops": [{"utterances": ["U2"], "kind": "repeat"}]}, utts, words)
    assert sorted(d.removed_words) == [0, 1, 2]                   # U2 survives the drop


def test_untranscribed_speech_can_be_dropped_as_a_span():
    words, utts = _setup()
    d = editor.parse({"drops": [{"utterances": ["U4"], "kind": "false_start"}]}, utts, words)
    assert d.removed_words == {} and d.removed_spans == [(11.0, 12.4, "false_start")]


def test_unknown_ids_and_prose_are_refused_not_guessed():
    words, utts = _setup()
    assert editor.parse("I think U1 should go.", utts, words).refused == ["no JSON object in the answer"]
    d = editor.parse({"drops": [{"utterances": ["U77"]}]}, utts, words)
    assert d.removed_words == {} and "unknown utterance" in d.refused[0]


def test_the_answer_is_found_inside_fences_and_prose():
    text = "Here you go:\n```json\n" + json.dumps({"drops": [{"utterances": ["U3"]}]}) + "\n```"
    assert editor.first_json(text) == {"drops": [{"utterances": ["U3"]}]}


def test_a_model_the_proxy_substituted_is_rejected(monkeypatch):
    class _Resp:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    payload = {"model": "gpt-6-astra", "choices": [{"message": {"content": "{}"}}], "usage": {}}
    monkeypatch.setattr(editor.urllib.request, "urlopen", lambda req, timeout=0: _Resp(payload))
    with pytest.raises(RuntimeError, match="instead of"):
        editor.EditorClient(api_key="k").ask("s", "u")


def test_the_speaker_section_comes_before_the_style_examples():
    text = editor.request("U1 [0.0-1.0 s] dosto", "EXAMPLE CUTS",
                          speaker_style="Fixed opening line (said once): Hello doston")
    assert (text.index("About this speaker (from their own channel notes):\nFixed opening line")
            < text.index("How this creator edits") < text.index("The recording:"))
    assert "About this speaker" not in editor.request("U1 [0.0-1.0 s] dosto", "EXAMPLE CUTS")


def test_the_editor_model_and_effort_come_from_the_app_setting_else_the_defaults(monkeypatch):
    from store.app_settings import AppSettings
    monkeypatch.delenv("CLAUDE_EDITOR_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_EDITOR_EFFORT", raising=False)
    monkeypatch.setattr(AppSettings, "get", lambda self, key, default=None: {
        "api_key": "k", "model": "claude-fable-5-1", "effort": "high"} if key == "auto_cut_editor" else default)
    proxy = editor.proxy_config()
    assert (proxy["model"], proxy["effort"], proxy["api_key"]) == ("claude-fable-5-1", "high", "k")
    monkeypatch.setattr(AppSettings, "get", lambda self, key, default=None: default)
    proxy = editor.proxy_config()
    assert (proxy["model"], proxy["effort"]) == (editor.DEFAULT_MODEL, editor.DEFAULT_EFFORT)
