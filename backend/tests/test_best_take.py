"""Tests for choosing which attempt to keep, and for the cross-spelling fold.

"Keep the last take" is right almost always, and where it is wrong the cost of
fixing it is high: a bad swap deletes the good reading. So the mechanism only
engages on a like-for-like pair — a full-length alternative reading of the same
sentence — and most of these tests are about the cases it must not touch.
"""

import pytest

from asr.retakes import (
    MIN_TAKE_WORDS,
    choose_best_takes,
    parse_take_choices,
    phonetic,
    skeleton,
    token_similarity,
)


def _words(sentence: str, spacing: float = 0.35, cut: range = None):
    words = []
    clock = 0.0
    for index, token in enumerate(sentence.split()):
        word = {"word": token, "start": round(clock, 3),
                "end": round(clock + spacing * 0.85, 3), "probability": 0.99}
        if cut is not None and index in cut:
            word.update({"disfluency": True, "reason": "retake"})
        words.append(word)
        clock += spacing
    return words


def _kept(words):
    return [w["word"] for w in words if not w.get("disfluency")]


def _answering(reply):
    async def ask(_system, _user):
        return reply
    return ask


# --- parsing ---------------------------------------------------------------

def test_take_choices_are_parsed():
    assert parse_take_choices("1: 2\n2: take 1", 2) == {0: 1, 1: 0}


def test_an_unparseable_answer_chooses_nothing():
    assert parse_take_choices("I think the second one reads better", 1) == {}


# --- the fold across spellings ---------------------------------------------

def test_devanagari_left_in_a_romanised_token_is_dropped_before_comparing():
    """The ASR leaves stray native characters in romanised output. They compare
    against nothing, so the Latin skeleton of the same word is what is used."""
    assert phonetic("kaॉnvel").isascii()


def test_the_same_word_spelled_differently_between_takes_still_matches():
    """The known limit this closes, verbatim from the recording: one take says
    "cornwall ooniversity", the next says "kaॉnvel yoonivarsitee", and the repeat
    used to survive into the finished edit because the tokens shared no letters."""
    assert token_similarity("ooniversity", "yoonivarsitee") >= 0.8
    assert token_similarity("cornwall", "kaॉnvel") >= 0.8


def test_the_consonant_fold_does_not_invent_matches_between_short_words():
    """"kar", "kir" and "kur" all reduce to "kr", and Hindi is full of them. The
    keys really do collide; the token-length bar is what stops it mattering."""
    assert skeleton("kar") == skeleton("kur")
    assert token_similarity("kar", "kur") == 0.0


def test_two_unrelated_long_words_do_not_match():
    assert token_similarity("universitee", "responsibility") == 0.0
    assert token_similarity("kamare", "sunane") == 0.0
    assert token_similarity("dekhate", "samajhate") == 0.0


# --- choosing the take -----------------------------------------------------

CLEAN = "kabhee aisaa huaa hai ki aap kisee kamare men gae hoon"
GARBLED = "kabhee aisaa ha aap vah hai ki aap kisee kamare men gae"


@pytest.mark.asyncio
async def test_an_earlier_clean_reading_can_beat_a_garbled_final_take():
    """The case the reference recording ends on: the speaker's *last* attempt is
    itself fumbled and the only clean reading is an earlier one. Structure keeps
    the last take and ships the garbled version."""
    words = _words(f"{CLEAN} {GARBLED}", cut=range(len(CLEAN.split())))

    swaps = await choose_best_takes(words, _answering("1: 1"))

    assert swaps == 1
    assert _kept(words) == CLEAN.split()


@pytest.mark.asyncio
async def test_choosing_the_final_take_changes_nothing():
    words = _words(f"{CLEAN} {GARBLED}", cut=range(len(CLEAN.split())))
    before = _kept(words)

    assert await choose_best_takes(words, _answering("1: 2")) == 0
    assert _kept(words) == before


@pytest.mark.asyncio
async def test_a_silent_model_leaves_the_last_take_in_place():
    words = _words(f"{CLEAN} {GARBLED}", cut=range(len(CLEAN.split())))
    before = _kept(words)

    assert await choose_best_takes(words, _answering(None)) == 0
    assert _kept(words) == before


@pytest.mark.asyncio
async def test_a_run_up_is_never_offered_as_an_alternative_reading():
    """"dosto kya ap" is not a reading of the sentence, it is a run-up at it.
    The ordinary pile-up must never reach this mechanism at all."""
    pile_up = "dosto kya ap dosto kya dosto kya a dosto"
    final = "dosto kya apako pataa hai india men ek aisee jagah hai"
    words = _words(f"{pile_up} {final}", cut=range(len(pile_up.split())))

    asked = []

    async def ask(_system, user):
        asked.append(user)
        return "1: 1"

    assert await choose_best_takes(words, ask) == 0
    assert not asked, "a stub attempt must not even be put to the model"


@pytest.mark.asyncio
async def test_an_attempt_shorter_than_the_take_is_refused():
    """A truncated version of the take would swap away the sentence's ending."""
    truncated = " ".join(CLEAN.split()[:MIN_TAKE_WORDS])
    words = _words(f"{truncated} {CLEAN}", cut=range(MIN_TAKE_WORDS))

    assert await choose_best_takes(words, _answering("1: 1")) == 0


@pytest.mark.asyncio
async def test_a_different_sentence_is_never_swapped_in():
    """Two unrelated sentences are not two takes of one. Swapping them would put
    speech from somewhere else into the middle of the edit."""
    other = "lekin aaj hum kuchh aur baat karane vaale hain yahaan"
    words = _words(f"{other} {CLEAN}", cut=range(len(other.split())))

    assert await choose_best_takes(words, _answering("1: 1")) == 0


@pytest.mark.asyncio
async def test_nothing_is_asked_when_no_attempt_was_cut():
    words = _words(CLEAN)
    asked = []

    async def ask(_system, user):
        asked.append(user)
        return "1: 1"

    assert await choose_best_takes(words, ask) == 0
    assert not asked
