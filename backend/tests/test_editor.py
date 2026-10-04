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
