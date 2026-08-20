"""Tests for fluency-driven cut planning.

The model is asked what the edit should say; alignment turns its answer into
cuts. Most of these are about what happens when the model misbehaves, because
that is the whole risk: a model that paraphrases, translates or invents must not
be able to delete anything.
"""

import pytest

from backend.asr.fluency import (
    MIN_DELETED_RUN_ALLOWANCE,
    MIN_MATCH_RATIO,
    align_deletions,
    build_prompt_text,
    judge_window,
    plan_fluent_cuts,
    windows,
)


def _words(sentence: str, spacing: float = 0.35, pauses: dict = None):
    tokens = sentence.split()
    pauses = pauses or {}
    words = []
    clock = 0.0
    for index, token in enumerate(tokens):
        clock += pauses.get(index, 0.0)
        words.append({"word": token, "start": round(clock, 3),
                      "end": round(clock + spacing * 0.85, 3)})
        clock += spacing
    return words


# --- alignment -------------------------------------------------------------

def test_a_dropped_run_becomes_a_cut():
    original = "dosto kya ap dosto kya apko pata hai".split()
    cleaned = "dosto kya apko pata hai".split()
    deleted, ratio = align_deletions(original, cleaned)
    assert deleted == {0, 1, 2}
    assert ratio == pytest.approx(5 / 8)


def test_punctuation_and_case_do_not_block_a_match():
    deleted, ratio = align_deletions("Dosto, kya AP".split(), "dosto kya ap".split())
    assert deleted == set()
    assert ratio == 1.0


def test_a_reworded_span_is_kept_not_cut():
    """The model 'fixed' a word. That span is a guess, and cutting on a guess is
    how real content gets destroyed — the original stays."""
    original = "aap kisee kamare men gae ho".split()
    cleaned = "aap kisi kamre mein gaye ho".split()
    deleted, _ = align_deletions(original, cleaned)
    assert deleted == set()


def test_invented_words_cut_nothing():
    original = "dosto kya apko pata hai".split()
    cleaned = "friends do you know".split()          # translated, not transcribed
    deleted, ratio = align_deletions(original, cleaned)
    assert deleted == set()
    assert ratio == 0.0


def test_reordering_cannot_delete_the_moved_words():
    original = "one two three four".split()
    cleaned = "three four one two".split()
    deleted, _ = align_deletions(original, cleaned)
    # Whatever the aligner pairs up, it must never claim every word was dropped.
    assert len(deleted) < len(original)


# --- trust limits ----------------------------------------------------------

def test_a_translated_answer_is_discarded():
    plan = judge_window("dosto kya apko pata hai india me".split(),
                        "friends did you know that india has".split())
    assert not plan.trusted
    assert "verbatim" in plan.reason


def test_an_empty_answer_is_discarded():
    plan = judge_window("dosto kya apko".split(), [])
    assert not plan.trusted


def test_a_summarised_answer_is_discarded():
    """The model wrote a précis instead of an edit. Whichever limit catches it,
    nothing may be cut on that basis."""
    original = ("dosto aapake saath kabhee aisaa huaa hai ki aap kisee kamare men "
                "gae ho jahaan bahut saare log hain aur sab aapako dekh rahe hain").split()
    plan = judge_window(original, "dosto aapake saath".split())
    assert not plan.trusted
    assert plan.deleted == set()


def test_one_enormous_removal_is_discarded():
    """Most of the window echoed back verbatim — so the match ratio is healthy —
    but one removal runs far longer than any fumble. That is a model losing its
    place mid-answer, and it is not allowed to take a paragraph with it."""
    # Big enough that 40% of the window clears the absolute floor too.
    original = ["w%d" % i for i in range(400)]
    run = int(400 * 0.4) + 5
    assert run > MIN_DELETED_RUN_ALLOWANCE
    cleaned = original[:100] + original[100 + run:]
    plan = judge_window(original, cleaned)
    assert plan.match_ratio > MIN_MATCH_RATIO   # the ratio guard is happy...
    assert not plan.trusted                    # ...but the run-length guard is not
    assert "removal" in plan.reason


def test_a_real_pile_up_of_restarts_is_within_the_limit():
    """Five attempts at one sentence is 48 consecutive words on the recording
    this was measured from. That is a correct edit, not a runaway model, and an
    absolute 45-word ceiling threw the whole window away."""
    original = ["w%d" % i for i in range(240)]
    cleaned = original[48:]
    plan = judge_window(original, cleaned)
    assert plan.trusted
    assert len(plan.deleted) == 48


def test_a_sane_edit_is_trusted_and_offset_to_the_transcript():
    original = "dosto kya ap dosto kya apko pata hai india me".split()
    cleaned = "dosto kya apko pata hai india me".split()
    plan = judge_window(original, cleaned, start=100)
    assert plan.trusted
    assert plan.deleted == {100, 101, 102}


# --- windowing -------------------------------------------------------------

def test_a_short_transcript_is_one_window():
    assert windows(50) == [(0, 50, 0, 50)]


def test_windows_cover_every_word_exactly_once_in_their_cores():
    covered = []
    for _start, _end, core_start, core_end in windows(1000):
        covered.extend(range(core_start, core_end))
    assert covered == list(range(1000))


def test_windows_carry_context_around_their_core():
    spans = windows(1000)
    _start, _end, core_start, _core_end = spans[1]
    start, end = spans[1][0], spans[1][1]
    assert start < core_start           # context before
    assert end > spans[1][3]            # context after


# --- the prompt ------------------------------------------------------------

def test_pauses_are_shown_to_the_model():
    text = build_prompt_text(_words("dosto kya apko pata", pauses={2: 1.5}))
    assert text == "dosto kya | apko pata"


# --- end to end (with a stand-in model) ------------------------------------

@pytest.mark.asyncio
async def test_the_model_decides_the_cut():
    words = _words("dosto kya ap dosto kya apko pata hai india me")

    async def ask(_system, _user):
        return "dosto kya apko pata hai india me"

    decided = await plan_fluent_cuts(words, ask)
    assert {i for i, cut in decided.items() if cut} == {0, 1, 2}
    assert all(i in decided for i in range(len(words))), "every word was ruled on"


@pytest.mark.asyncio
async def test_a_model_that_will_not_answer_yields_none():
    """None means "structure decides", not "cut nothing" — the caller has to be
    able to tell those apart."""
    words = _words("dosto kya ap dosto kya apko pata hai")

    async def ask(_system, _user):
        return None

    assert await plan_fluent_cuts(words, ask) is None


@pytest.mark.asyncio
async def test_an_untrustworthy_answer_yields_none():
    words = _words("dosto kya ap dosto kya apko pata hai india me ek jagah")

    async def ask(_system, _user):
        return "friends do you know there is a place in india"

    assert await plan_fluent_cuts(words, ask) is None


@pytest.mark.asyncio
async def test_a_discarded_window_leaves_no_verdict_behind():
    """A window the model fumbled must be *absent* from the result, not reported
    as "keep everything" — otherwise it silently un-cuts the fumbles structure
    found there. Measured: 81 structural cuts were being resurrected this way."""
    words = _words(" ".join("w%d" % i for i in range(400)))

    async def ask(_system, user):
        # Answer the first window with a real edit (drop two words), gibberish
        # for the rest. A verbatim echo would count as no opinion, not a ruling.
        if "w0 " not in user:
            return "zzz yyy xxx"
        text = user.split("Transcript:")[1].split("Cleaned")[0]
        return text.replace("w1 ", "").replace("w2 ", "")

    decided = await plan_fluent_cuts(words, ask)
    assert decided is not None
    assert 0 in decided                    # first window ruled on
    assert 350 not in decided              # later window discarded, not "kept"


@pytest.mark.asyncio
async def test_an_echoing_model_has_no_opinion():
    """A model too small for the task echoes the transcript back verbatim. The
    window is technically perfect — 100% match, nothing deleted — but it proves
    nothing about any word, and reporting it as keeps handed such a model the
    power to un-cut every structural retake. An echo must decide nothing."""
    words = _words("dosto kya ap dosto kya apko pata hai india me")

    async def ask(_system, _user):
        return "dosto kya ap dosto kya apko pata hai india me"

    decided = await plan_fluent_cuts(words, ask)
    assert decided == {}      # ruled on nothing — structure's cuts all stand


# --- which take's audio survives -------------------------------------------

def test_the_last_take_is_the_one_kept_not_the_first():
    """Every attempt opens with the same words, so the model's answer matches all
    of them equally. A left-anchored diff keeps attempt one's opening and splices
    it onto the final attempt's continuation — the text reads right and the audio
    jumps between takes mid-sentence. Ties must fall to the LAST occurrence."""
    original = ("dosto aapake saath ki aap kisee "
                "dosto aapake saath ki aap kisee lut "
                "dosto aapake saath ki aap kamare men gae hoon").split()
    cleaned = "dosto aapake saath ki aap kamare men gae hoon".split()
    deleted, _ = align_deletions(original, cleaned)
    # The two abandoned run-ups (0-5 and 6-12, the second ending on "lut") go;
    # the final take survives whole and unbroken from index 13.
    assert deleted == set(range(0, 13))
    survivors = [i for i in range(len(original)) if i not in deleted]
    assert survivors == list(range(13, len(original)))


def test_a_kept_take_is_never_split_across_attempts():
    original = "ek experiment kiyaa ek experiment kiyaa gayaa thaa cornwell men".split()
    cleaned = "ek experiment kiyaa gayaa thaa cornwell men".split()
    deleted, _ = align_deletions(original, cleaned)
    survivors = [i for i in range(len(original)) if i not in deleted]
    # Contiguous: no stitching a word from the first attempt onto the second.
    assert survivors == list(range(survivors[0], survivors[-1] + 1))
    assert survivors[0] == 3
