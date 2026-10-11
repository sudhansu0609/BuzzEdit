"""The retake sweep: chunks, the lines they point at, merged groups (AUTO_CUT_V3_PLAN.md §2.3)."""

import json
import threading

from asr import retake_sweep
from asr.utterances import Utterance


def _utts(n):
    return [Utterance(f"U{k + 1}", start=2.0 * k, end=2.0 * k + 1.5, word_indices=[k], text=f"line {k + 1}",
                      pause_before=0.5) for k in range(n)]


class _Sweeper:
    """A fake EditorClient: the answer for a chunk is chosen by its first line; a chunk whose
    first line is in `fail` raises, as a proxy that is down would."""

    def __init__(self, by_first, fail=()):
        self.by_first, self.fail = by_first, set(fail)
        self.lock = threading.Lock()
        self.seen = []

    def ask(self, system, user, retries=1):
        assert system == retake_sweep.SWEEP_SYSTEM
        first = user.split("\n")[1].split(" ")[0]           # "The excerpt:\nU1 [0.0-1.5s, ..."
        with self.lock:
            self.seen.append(user)
        if first in self.fail:
            raise RuntimeError("editor unavailable: proxy down")
        return json.dumps(self.by_first.get(first, {})), {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def test_chunks_overlap_and_cover_every_utterance():
    utts = _utts(200)
    parts = retake_sweep.chunks(utts, size=80, overlap=25)
    assert [len(p) for p in parts] == [80, 80, 80, 35]
    for a, b in zip(parts, parts[1:]):
        assert [u.uid for u in a[-25:]] == [u.uid for u in b[:25]]
    assert {u.uid for p in parts for u in p} == {u.uid for u in utts}
    assert retake_sweep.chunks(_utts(5)) == [_utts(5)]
    assert retake_sweep.chunks([]) == []


def test_the_lines_a_chunk_points_at_elsewhere_are_shown_with_it():
    utts = _utts(200)
    utts[1].similar = [("U150", 0.8), ("U3", 0.6)]           # U3 is in the chunk already
    utts[2].same_opening = ["U160"]
    first = retake_sweep.chunks(utts)[0]
    assert [u.uid for u in retake_sweep.referenced(first, utts)] == ["U150", "U160"]
    client = _Sweeper({})
    retake_sweep.sweep(utts, client)
    asked = next(user for user in client.seen if user.startswith("The excerpt:\nU1 ["))
    excerpt, referenced = asked.split("\n\nReferenced lines from elsewhere in the recording:\n")
    assert "U150 [" in referenced and "U160 [" in referenced and "U150 [" not in excerpt
    others = [user for user in client.seen if not user.startswith("The excerpt:\nU1 [")]
    assert len(others) == 3 and not any("Referenced lines" in user for user in others)


def test_groups_that_share_a_line_are_merged_and_unknown_ids_are_dropped():
    utts = _utts(200)
    client = _Sweeper({
        "U1": {"groups": [{"utterances": ["U70", "U75"], "why": "said again"},
                          {"utterances": ["U3", "U999"]}]},              # one known line: no group
        "U56": {"groups": [{"utterances": ["U100", "U75", "U998"]}]},    # shares U75 with the first
        "U111": {"groups": [{"utterances": ["U121", "U120"]}]},
    })
    groups, meta = retake_sweep.sweep(utts, client)
    assert groups == [["U70", "U75", "U100"], ["U120", "U121"]]
    assert meta == {"calls": 4, "tokens_in": 28, "tokens_out": 12, "failed": 0}


def test_a_chunk_that_gets_no_answer_is_counted_and_skipped():
    utts = _utts(200)
    client = _Sweeper({"U1": {"groups": [{"utterances": ["U4", "U9"]}]},
                       "U56": {"groups": [{"utterances": ["U60", "U61"]}]}}, fail={"U56"})
    groups, meta = retake_sweep.sweep(utts, client)
    assert groups == [["U4", "U9"]]
    assert meta == {"calls": 4, "tokens_in": 21, "tokens_out": 9, "failed": 1}
