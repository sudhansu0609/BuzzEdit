"""tools/autocut_dataset.py: the creator's removed spans become the editor's own
decisions (groups / drops / trims), cut into overlapping windows."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import autocut_dataset as D  # noqa: E402
from asr.utterances import Utterance  # noqa: E402


def _words(texts, start=0.0, step=0.5):
    out = []
    t = start
    for w in texts:
        out.append({"word": w, "start": round(t, 2), "end": round(t + step * 0.8, 2)})
        t += step
    return out


def _utt(uid, words, idxs, pause=0.3, notes=None, similar=None, same_opening=None):
    return Utterance(uid=uid, start=words[idxs[0]]["start"], end=words[idxs[-1]]["end"], word_indices=list(idxs),
                     text=" ".join(w["word"] for w in (words[i] for i in idxs)), pause_before=pause,
                     notes=notes or [], similar=similar or [], same_opening=same_opening or [])


def test_removed_spans_merge_and_skip_editorial():
    key = {"spans": [{"start": 10, "end": 12, "kind": "speech"}, {"start": 11.5, "end": 14, "kind": "silence"},
                     {"start": 20, "end": 30, "kind": "editorial"}, {"start": 40, "end": 41}]}
    assert D.removed_spans_of(key) == [(10.0, 14.0), (40.0, 41.0)]
    assert D.removed_share(10, 14, [(10.0, 14.0)]) == 1.0
    assert round(D.removed_share(12, 16, [(10.0, 14.0)]), 2) == 0.5


def test_a_cut_retake_joins_its_kept_line_and_a_broken_line_is_a_false_start():
    words = _words(["main", "wahan", "gaya", "tha",            # U1 0.0-1.9  (removed, retake of U3)
                    "toh", "phir",                               # U2 2.0-2.9  (removed, broke off)
                    "main", "wahan", "gaya", "tha", "raat", "ko",  # U3 3.0-5.9 kept
                    "aur", "um", "um", "darwaza", "khula"])      # U4 6.0-8.4 kept with a trim of "um um"
    utts = [
        _utt("U1", words, [0, 1, 2, 3], similar=[("U3", 0.9)]),
        _utt("U2", words, [4, 5], notes=["cut off mid-sound"]),
        _utt("U3", words, [6, 7, 8, 9, 10, 11], similar=[("U1", 0.9)]),
        _utt("U4", words, [12, 13, 14, 15, 16]),
    ]
    um = words[13]["start"], words[14]["end"]
    key = {"spans": [{"start": 0.0, "end": 1.95, "kind": "speech"}, {"start": 2.0, "end": 2.95, "kind": "speech"},
                     {"start": um[0], "end": um[1], "kind": "speech"}]}
    spans = D.removed_spans_of(key)
    states = D.classify(utts, words, spans)
    assert states["U1"]["state"] == "removed" and states["U2"]["state"] == "removed"
    assert states["U3"]["state"] == "kept" and states["U4"]["state"] == "trim"
    dec = D.decisions_for(utts, words, states)
    assert dec["groups"] == [{"utterances": ["U1", "U3"], "keep": "U3", "why": "earlier attempts at the kept line", "confidence": 0.9}]
    assert [d["utterances"] for d in dec["drops"]] == [["U2"]] and dec["drops"][0]["kind"] == "false_start"
    assert dec["trims"] == [{"utterance": "U4", "remove": "um um", "why": "the creator cut these words", "confidence": 0.85}]
    # The product's parser accepts exactly this answer.
    from asr import editor
    parsed = editor.parse(dec, utts, words)
    assert not parsed.refused and parsed.groups == 1 and parsed.drops == 1 and parsed.trims == 1
    assert set(parsed.removed_words) == {0, 1, 2, 3, 4, 5, 13, 14}


def test_windows_overlap_and_fold_a_tiny_tail():
    assert D.windows(0, 50, 12) == []
    assert D.windows(30, 50, 12) == [(0, 30)]
    assert D.windows(100, 50, 12) == [(0, 50), (38, 88), (76, 100)]
    assert D.windows(90, 50, 12) == [(0, 50), (38, 90)], "a 14-utterance tail (under a third of a window) folds into the one before"


def test_a_window_without_the_kept_line_turns_the_group_into_repeat_drops():
    full = {"groups": [{"utterances": ["U1", "U9"], "keep": "U9", "why": "x", "confidence": 0.9}],
            "drops": [{"utterances": ["U2"], "kind": "false_start", "why": "y", "confidence": 0.9},
                      {"utterances": ["U20"], "kind": "chatter", "why": "z", "confidence": 0.9}],
            "trims": [{"utterance": "U4", "remove": "um", "why": "w", "confidence": 0.8}]}
    w = D.window_decisions(full, ["U1", "U2", "U3", "U4"])
    assert w["groups"] == [] and [d["utterances"] for d in w["drops"]] == [["U1"], ["U2"]]
    assert w["drops"][0]["kind"] == "repeat" and w["trims"][0]["utterance"] == "U4"
    w2 = D.window_decisions(full, ["U1", "U4", "U9"])
    assert w2["groups"][0]["utterances"] == ["U1", "U9"] and w2["drops"] == []


def test_make_pairs_writes_the_editor_s_own_prompt_and_answer():
    words = _words(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    utts = [_utt(f"U{n + 1}", words, [n]) for n in range(10)]
    key = {"spans": [{"start": words[2]["start"], "end": words[2]["end"], "kind": "speech"}]}
    pairs, stats = D.make_pairs("t", utts, words, key, window=8, overlap=2)
    assert stats["windows"] == 2 and stats["removed"] == 1 and stats["drops"] == 1
    first = pairs[0]
    assert first["system"].startswith("You are the video editor") and "The recording:" in first["input"]
    assert "U1 [" in first["input"] and "Your decisions (JSON only):" in first["input"]
    out = json.loads(first["output"])
    assert out["drops"][0]["utterances"] == ["U3"] and out["groups"] == [] and out["trims"] == []
    assert json.loads(pairs[1]["output"]) == {"groups": [], "drops": [], "trims": []}
