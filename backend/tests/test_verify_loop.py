"""Tests for the repeated audit-repair loop and whole-fragment dropping.

One audit-repair pass was never enough for a mechanical reason: repair works
sentence by sentence, and removing debris from one sentence moves where the next
one begins. So the loop re-reads what actually survives each round.

Whole-fragment dropping is the most dangerous thing in the pipeline — it removes
a sentence the speaker said, rather than debris inside one — so most of these
tests are about the cases where it must refuse.
"""

import pytest

from backend.asr.verify import (
    MAX_FRAGMENT_WORDS,
    AuditResult,
    Issue,
    find_droppable_fragments,
    parse_drops,
    split_sentences,
    verify_until_clean,
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


def _live(words):
    return [w["word"] for w in words if not w.get("disfluency")]


def _scripted(answers):
    """An `ask` that replies with each answer in turn, recording the prompts."""
    calls = []

    async def ask(system, user):
        calls.append((system, user))
        return answers[len(calls) - 1] if len(calls) <= len(answers) else None

    ask.calls = calls
    return ask


# --- parsing ---------------------------------------------------------------

def test_drop_verdicts_are_parsed():
    assert parse_drops("1: DROP\n2: KEEP\n3: DROP", 3) == {0, 2}


def test_anything_that_is_not_drop_counts_as_keep():
    """The refusal has to be the default: a model that answers something the
    parser does not understand must not thereby delete a sentence."""
    assert parse_drops("1: maybe?\n2: I am not sure\n3: KEEP", 3) == set()


# --- the loop --------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_loop_stops_as_soon_as_nothing_is_broken():
    words = _words("dosto aapako pataa hai ki india men ek jagah hai.")
    ask = _scripted(["1: OK"])

    quality, cuts = await verify_until_clean(words, ask)

    assert cuts == []
    assert quality["verdict"] == "clean"
    assert len(ask.calls) == 1, "a clean edit must not cost a repair round"


@pytest.mark.asyncio
async def test_the_loop_repairs_then_re_reads_what_ships():
    """The score reported must be the score of the finished edit, never of the
    state it was in when the first complaint was made."""
    text = "dosto aapake saath kabhee aisaa ha aap vah hai ki aap kamare men gae."
    words = _words(text)
    ask = _scripted([
        "1: BROKEN - leftover stumble words",              # round 1 audit
        "1: " + text.replace(" ha aap vah", ""),           # round 1 repair
        "1: KEEP",                                         # fragment question
        "1: OK",                                           # round 2 audit
    ])

    quality, cuts = await verify_until_clean(words, ask)

    assert [words[c]["word"] for c in cuts] == ["ha", "aap", "vah"]
    assert quality["verdict"] == "clean"
    assert quality["rounds"][0]["broken"] == 1
    assert quality["rounds"][-1]["broken"] == 0
    assert quality["repairs"], "what was mended has to be reported, not only what is left"


@pytest.mark.asyncio
async def test_a_round_that_changes_nothing_ends_the_loop():
    """Asking the same question again gets the same answer. Without this the
    loop burns the model on a complaint it has already refused to act on."""
    words = _words("dosto aapako pataa hai ki india men ek jagah hai.")
    ask = _scripted([
        "1: BROKEN - something is wrong",
        "1: " + " ".join(_live(words)),      # repair changes nothing
        "1: KEEP",
    ])

    quality, cuts = await verify_until_clean(words, ask)

    assert cuts == []
    assert quality["verdict"].startswith("still_broken")
    # One audit and one repair. There is no second audit, and no fragment
    # question either — a single-sentence edit has nothing that could supersede
    # it, so there is nothing to ask about.
    assert len(ask.calls) == 2


@pytest.mark.asyncio
async def test_a_silent_model_verifies_nothing_and_cuts_nothing():
    words = _words("dosto aapako pataa hai ki india men ek jagah hai.")

    async def ask(_system, _user):
        return None

    quality, cuts = await verify_until_clean(words, ask)

    assert cuts == []
    assert quality["verdict"] == "not_verified"


@pytest.mark.asyncio
async def test_words_already_cut_are_invisible_to_the_loop():
    """The audit reads the *surviving* edit. A word the planner removed upstream
    must not be shown to the model, or it judges a sentence nobody will hear."""
    words = _words("dosto uh aapako pataa hai ki india men ek jagah hai.")
    words[1]["disfluency"] = True
    ask = _scripted(["1: OK"])

    await verify_until_clean(words, ask)

    assert "uh" not in ask.calls[0][1]


# --- dropping an abandoned sentence whole ----------------------------------

def _broken(words, sentence_index=0, problem="breaks off unfinished"):
    sentences = split_sentences(words)
    result = AuditResult(total=len(sentences), broken=1)
    result.issues.append(Issue(sentence_index, sentences[sentence_index].text, problem))
    return result


@pytest.mark.asyncio
async def test_an_abandoned_start_superseded_by_the_next_line_is_dropped():
    words = _words("aur vah kya hai ki. aur vah kya hai ki hum log ise dekh sakate hain.")
    ask = _scripted(["1: DROP"])

    positions = await find_droppable_fragments(words, _broken(words), ask)

    assert [words[p]["word"] for p in positions] == "aur vah kya hai ki.".split()


@pytest.mark.asyncio
async def test_a_fragment_is_kept_when_the_model_says_keep():
    words = _words("aur vah kya hai ki. hum log ise dekh sakate hain aaj.")
    ask = _scripted(["1: KEEP"])

    assert await find_droppable_fragments(words, _broken(words), ask) == []


@pytest.mark.asyncio
async def test_a_fragment_holding_a_negation_is_never_offered_for_dropping():
    """Same rule as repair: removing a negation reverses the speaker rather than
    tidying them. It must not even reach the model."""
    words = _words("hamane to notice naheen kiyaa. hum log ise dekh sakate hain aaj.")
    ask = _scripted(["1: DROP"])

    assert await find_droppable_fragments(words, _broken(words), ask) == []
    assert not ask.calls, "a protected fragment must not be asked about at all"


@pytest.mark.asyncio
async def test_a_long_sentence_is_never_dropped_whole():
    """Past a dozen words a 'fragment' is a thought with content in it, and
    dropping it takes something the speaker said."""
    long_text = " ".join(["shabd"] * (MAX_FRAGMENT_WORDS + 3))
    words = _words(f"{long_text}. hum log ise dekh sakate hain aaj.")
    ask = _scripted(["1: DROP"])

    assert await find_droppable_fragments(words, _broken(words), ask) == []


@pytest.mark.asyncio
async def test_the_last_sentence_is_never_dropped():
    """Nothing follows it, so nothing supersedes it — there is no evidence the
    speaker said it again properly."""
    words = _words("hum log ise dekh sakate hain aaj. aur vah kya hai ki.")
    ask = _scripted(["1: DROP"])

    assert await find_droppable_fragments(words, _broken(words, sentence_index=1), ask) == []


@pytest.mark.asyncio
async def test_a_sentence_repair_already_mended_is_not_dropped_as_well():
    """Mending beats deleting. A sentence the repair pass fixed is no longer a
    fragment, and offering it for dropping would remove the fixed version."""
    words = _words("aur vah kya hai ki. hum log ise dekh sakate hain aaj.")
    result = _broken(words)
    result.issues[0].repaired = True
    ask = _scripted(["1: DROP"])

    assert await find_droppable_fragments(words, result, ask) == []
