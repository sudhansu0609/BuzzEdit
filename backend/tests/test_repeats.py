"""The deterministic phrase-repeat rule and the report's repeats_remaining measure."""

from asr.repeats import cut_phrase_repeats, exact_repeats, norm, repeats_remaining


def _line(text, start, dur=0.3):
    """Words said one after another from `start`, `dur` seconds each."""
    out, t = [], float(start)
    for tok in text.split():
        out.append({"start": round(t, 3), "end": round(t + dur, 3),
                    "word": tok, "word_native": tok})
        t += dur
    return out


def _hole(start, end):
    """A "[…]" placeholder: speech the recogniser heard but did not transcribe."""
    return {"start": start, "end": end, "word": "[…]", "word_native": "[…]",
            "untranscribed": True}


def _untouched(words):
    return all("enabled" not in w and "reason" not in w for w in words)


def _cut(words):
    return all(w["enabled"] is False and w["disfluency"] is True
               and w["candidate"] is False and w["reason"] == "repeat"
               for w in words)


def test_a_four_word_phrase_said_twice_is_found_and_its_first_copy_cut():
    words = _line("बारिश के बाद मिट्टी बारिश के बाद मिट्टी की खुशबू", 607.0)

    found = exact_repeats(words)
    assert len(found) == 1
    rep = found[0]
    assert rep["n"] == 4
    assert rep["first"] == [0, 1, 2, 3] and rep["second"] == [4, 5, 6, 7]
    assert rep["text"] == "बारिश के बाद मिट्टी"
    assert rep["start"] == 607.0
    assert rep["hole_between"] is False

    assert cut_phrase_repeats(words) == (1, [])
    assert _cut(words[:4])
    assert _untouched(words[4:])


def test_a_repeat_after_a_two_second_pause_is_not_one():
    words = _line("इस जोन में", 677.0) + _line("इस जोन में", 679.9)  # 2 s between

    assert exact_repeats(words) == []
    assert cut_phrase_repeats(words) == (0, [])
    assert _untouched(words)


def test_storytelling_keeps_a_clean_paused_repeat_but_cuts_one_across_a_hole():
    words = (_line("वहाँ नीचे", 100.0) + _line("वहाँ नीचे", 101.4)     # 0.8 s pause
             + _line("इस जोन में", 200.0) + [_hole(200.9, 201.5)]
             + _line("इस जोन में", 201.7))                            # 0.8 s, hole between

    cut, candidates = cut_phrase_repeats(words, storytelling=True)

    assert cut == 1
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["text"] == "वहाँ नीचे"
    assert cand["gap"] == 0.8 and cand["hole_between"] is False
    assert cand["first"] == [0, 1] and cand["second"] == [2, 3]
    assert _untouched(words[:4])
    assert _cut(words[4:7])
    assert _untouched(words[7:])


def test_words_already_disabled_are_invisible_to_the_scan():
    # The first copy is already removed: nothing plays twice.
    words = _line("मुझे एहसास हुआ कि अगर मुझे एहसास हुआ कि अगर", 2065.0)
    assert len(exact_repeats(words)) == 1
    for w in words[:5]:
        w["enabled"] = False
    assert exact_repeats(words) == []

    # A removed filler between the copies does not hide the repeat either.
    words = (_line("इस जोन में", 10.0)
             + [{"start": 10.9, "end": 11.3, "word": "um", "word_native": "उम्म",
                 "disfluency": True}]
             + _line("इस जोन में", 11.4))
    found = exact_repeats(words)
    assert [(r["first"], r["second"]) for r in found] == [([0, 1, 2], [4, 5, 6])]


def test_repeats_remaining_counts_what_survives_cut_phrase_repeats():
    words = (_line("वहाँ नीचे", 100.0) + _line("वहाँ नीचे", 101.4)            # 0.8 s pause
             + _line("बारिश के बाद मिट्टी", 200.0)
             + _line("बारिश के बाद मिट्टी", 201.3))                           # 0.1 s

    assert repeats_remaining(words)["count"] == 2
    assert len(repeats_remaining(words, max_examples=1)["examples"]) == 1

    cut_phrase_repeats(words, storytelling=True)
    assert repeats_remaining(words) == {
        "count": 1, "examples": [{"start": 100.0, "text": "वहाँ नीचे"}]}

    cut_phrase_repeats(words)
    assert repeats_remaining(words) == {"count": 0, "examples": []}


def test_ki_and_ka_are_different_tokens():
    assert norm("की") != norm("का")
    assert norm("की,") == "की"
    assert exact_repeats(_line("घर की घर का", 0.0)) == []
    assert len(exact_repeats(_line("घर की घर की", 0.0))) == 1


def test_a_repeat_made_only_of_reduplication_words_is_skipped():
    words = _line("धीरे धीरे धीरे धीरे", 0.0)

    assert exact_repeats(words) == []
    assert cut_phrase_repeats(words) == (0, [])
    assert _untouched(words)
