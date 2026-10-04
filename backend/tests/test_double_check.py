"""The double-check around the editor: agreement, second opinion, content loss, final read."""

import itertools
import json
import threading

from asr import double_check as dc
from asr import editor
from asr.utterances import Utterance


def _setup():
    words = [{"word": f"w{i}", "word_native": f"w{i}", "start": float(i), "end": i + 0.5} for i in range(10)]
    utts = [Utterance(f"U{k + 1}", start=2.0 * k, end=2.0 * k + 1.5, word_indices=[2 * k, 2 * k + 1],
                      text=f"w{2 * k} w{2 * k + 1}", pause_before=0.5) for k in range(5)]
    return words, utts


class _Script:
    """A fake EditorClient answering by which prompt it was given."""

    def __init__(self, edits, second=None, content=None, final=None):
        self.edits = itertools.cycle(edits)
        self.answers = {dc.SECOND_OPINION: second or {}, dc.CONTENT_CHECK: content or {},
                        dc.FINAL_READ: final or {}}
        self.lock = threading.Lock()
        self.seen = []

    def ask(self, system, user, retries=1):
        with self.lock:
            self.seen.append(system)
            answer = next(self.edits) if system == editor.SYSTEM else self.answers[system]
        return json.dumps(answer), {"model": "claude-opus-5-5", "usage": {}, "seconds": 0.0}


def test_unanimous_cuts_stand_and_disputed_ones_get_a_second_opinion():
    words, utts = _setup()
    a = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}], "drops": [{"utterances": ["U4"]}]}
    b = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}]}
    client = _Script([a, b, a], second={"U4": "remove"})
    out, report = dc.check(utts, words, client, runs=3, final_read=False)
    assert report.unanimous == 1 and report.disputed == 1 and report.second_opinion_removed == 1
    assert sorted(out.removed_words) == [0, 1, 6, 7]              # U1 and U4


def test_a_disputed_line_the_second_opinion_keeps_stays():
    words, utts = _setup()
    a = {"drops": [{"utterances": ["U4"]}]}
    client = _Script([a, {}, {}], second={"U4": "keep"})
    out, _ = dc.check(utts, words, client, runs=3, final_read=False)
    assert out.removed_words == {}


def test_content_only_said_in_a_removed_line_comes_back():
    words, utts = _setup()
    a = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}], "drops": [{"utterances": ["U4"]}]}
    client = _Script([a], content={"U1": "covered", "U4": "lost: the guard's name"})
    out, report = dc.check(utts, words, client, runs=1, final_read=False)
    assert report.restored_by_content_check == 1
    assert sorted(out.removed_words) == [0, 1]                    # U4 restored, U1 still gone


def test_the_final_read_can_remove_a_survivor_and_restore_a_link():
    words, utts = _setup()
    a = {"groups": [{"utterances": ["U1", "U2"], "keep": "U2"}], "drops": [{"utterances": ["U5"]}]}
    client = _Script([a], content={"U1": "covered", "U5": "covered"},
                     final={"remove": ["U3"], "restore": ["U5"]})
    out, report = dc.check(utts, words, client, runs=1)
    assert report.final_read_removed == 1 and report.final_read_restored == 1
    assert sorted(out.removed_words) == [0, 1, 4, 5]              # U1 and U3; U5 back
    assert dc.FINAL_READ in client.seen
