"""Worked examples from the creator's own cuts: labelling, picking, never the same recording."""

from asr import style
from asr.utterances import Utterance


def _utts(n=24):
    out = []
    for k in range(n):
        u = Utterance(f"U{k + 1}", start=k * 3.0, end=k * 3.0 + 2.5, word_indices=[k], text=f"line {k + 1}",
                      pause_before=0.5)
        out.append(u)
    out[4].similar = [("U6", 0.9)]          # U5 is an earlier attempt at U6
    return out


def test_labels_follow_the_cut_and_set_editorial_sections_apart():
    utts = _utts()
    removed = [(12.0, 14.5), (21.0, 21.8)]   # all of U5; a bit of U8
    decided = style.labels(utts, removed, editorial=[(60.0, 75.0)])
    assert decided["U5"] == "removed" and decided["U8"] == "trimmed" and decided["U1"] == "kept"
    assert decided["U21"] == "section" and decided["U25" if "U25" in decided else "U24"] in ("section", "kept")


def test_excerpts_prefer_retakes_and_say_what_the_creator_did():
    utts = _utts()
    decided = style.labels(utts, [(12.0, 14.5)])
    found = style.excerpts(utts, decided, "life3baje_ep1")
    assert found, "a window with a removed retake next to its kept twin is an example"
    assert "U5: removed (an earlier attempt; U6 kept)" in found[0]
    assert found[0].startswith("From life3baje_ep1:")


def test_a_recording_with_nothing_cut_gives_no_examples():
    utts = _utts()
    assert style.excerpts(utts, style.labels(utts, []), "x") == []
