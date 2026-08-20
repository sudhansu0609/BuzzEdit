"""Tests for the grammar audit and repair of the finished edit.

The risk here is the opposite of the rest of the pipeline. Everything else can
only fail to cut; this pass acts on a judgement about *language*, and the ASR
romanises Hindi badly enough that half a real audit's complaints are spelling,
not speech. Cutting on those would delete perfectly good audio, so most of these
tests are about refusing to.
"""

import pytest

from backend.asr.verify import (
    MAX_REPAIR_FRACTION,
    Sentence,
    audit,
    parse_repairs,
    parse_verdicts,
    repair_deletions,
    split_sentences,
    verify_and_repair,
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


# --- splitting -------------------------------------------------------------

def test_sentences_split_on_terminal_punctuation():
    sentences = split_sentences(_words("dosto kya haal hai. aaj hum baat karenge."))
    assert [s.text for s in sentences] == ["dosto kya haal hai.", "aaj hum baat karenge."]


def test_sentences_split_on_a_long_pause_when_punctuation_is_missing():
    """The ASR punctuates unreliably, so a *long* silence is still a boundary —
    but it has to be long: see the mid-clause pause test below."""
    sentences = split_sentences(_words("dosto kya haal hai aaj hum baat karenge",
                                       pauses={4: 3.0}))
    assert len(sentences) == 2
    assert sentences[1].text == "aaj hum baat karenge"


# --- parsing the model's answers ------------------------------------------

def test_verdicts_are_parsed():
    verdicts = parse_verdicts("1: OK\n2: BROKEN - trails off\n3: OK", 3)
    assert verdicts[0] == (True, "")
    assert verdicts[1][0] is False and "trails off" in verdicts[1][1]


def test_an_unanswered_sentence_counts_as_fine():
    """The model failing to answer is not evidence against the speaker."""
    verdicts = parse_verdicts("1: OK", 3)
    assert 1 not in verdicts and 2 not in verdicts


def test_junk_around_the_answer_is_ignored():
    verdicts = parse_verdicts("Sure! Here you go:\n\n1: OK\n2: BROKEN - fragment\n", 2)
    assert len(verdicts) == 2 and verdicts[1][0] is False


def test_repairs_are_parsed():
    assert parse_repairs("1: dosto kya haal hai", 2) == {0: "dosto kya haal hai"}


# --- repair is delete-only ------------------------------------------------

def _sentence(text: str) -> Sentence:
    tokens = text.split()
    return Sentence(0, list(range(len(tokens))), text)


def test_a_repair_removes_the_debris_it_names():
    text = "dosto aapake saath kabhee aisaa ha aap vah hai ki aap kisee kamare men gae hoon"
    sentence = _sentence(text)
    cuts = repair_deletions(sentence, text.replace(" ha aap vah", ""), text.split())
    assert [text.split()[c] for c in cuts] == ["ha", "aap", "vah"]


def test_a_repair_that_respells_a_word_changes_nothing():
    """"jaz" is the ASR's spelling of *judge* — the speaker said it perfectly.
    A model that "corrects" it must not be able to cut it."""
    text = "sab aapako jaz kar rahe hain"
    cuts = repair_deletions(_sentence(text), "sab aapako judge kar rahe hain", text.split())
    assert cuts is None


def test_a_repair_that_translates_is_refused():
    text = "dosto aapake saath kabhee aisaa huaa hai ki aap kisee kamare men gae ho"
    cuts = repair_deletions(_sentence(text),
                            "friends has it ever happened that you went into a room",
                            text.split())
    assert cuts is None


def test_a_repair_that_guts_the_sentence_is_refused():
    text = "dosto aapake saath kabhee aisaa huaa hai ki aap kisee kamare men gae ho"
    tokens = text.split()
    keep = " ".join(tokens[:3])          # would delete far more than the limit
    cuts = repair_deletions(_sentence(text), keep, tokens)
    assert cuts is None
    assert MAX_REPAIR_FRACTION < 1.0


def test_a_repair_that_changes_nothing_is_not_a_cut():
    text = "dosto aapake saath kabhee aisaa huaa hai"
    assert repair_deletions(_sentence(text), text, text.split()) is None


# --- end to end with a stand-in model -------------------------------------

@pytest.mark.asyncio
async def test_a_broken_sentence_is_found_and_repaired():
    words = _words("dosto aapake saath kabhee aisaa ha aap vah hai ki aap kamare men gae hoon.")

    async def ask(system, _user):
        if "proof-reading" in system:
            return "1: BROKEN - leftover stumble words"
        return ("1: dosto aapake saath kabhee aisaa hai ki aap kamare men gae hoon.")

    result, cuts = await verify_and_repair(words, ask)
    assert result.broken == 1
    assert [words[c]["word"] for c in cuts] == ["ha", "aap", "vah"]
    assert result.issues[0].repaired is True


@pytest.mark.asyncio
async def test_a_clean_edit_is_left_alone():
    words = _words("dosto aapake saath kabhee aisaa huaa hai ki aap kamare men gae hoon.")

    async def ask(_system, _user):
        return "1: OK"

    result, cuts = await verify_and_repair(words, ask)
    assert result.broken == 0 and cuts == []


@pytest.mark.asyncio
async def test_a_silent_model_reports_nothing_verified_and_cuts_nothing():
    words = _words("dosto aapake saath kabhee aisaa huaa hai ki aap kamare men gae hoon.")

    async def ask(_system, _user):
        return None

    result, cuts = await verify_and_repair(words, ask)
    assert result.total == 0 and cuts == []


@pytest.mark.asyncio
async def test_an_interjection_is_not_reported_as_a_fragment():
    """"Bye." is complete speech; judging it as a sentence produces a false
    complaint on every video that ends politely."""
    words = _words("dosto aapake saath kabhee aisaa huaa hai ki aap kamare men gae ho. Bye.")

    asked = {}

    async def ask(system, user):
        asked["user"] = user
        return "1: OK"

    await audit(words, ask)
    assert "Bye" not in asked["user"]


@pytest.mark.asyncio
async def test_filtered_sentences_are_renumbered_contiguously():
    """Short sentences are filtered out before the audit, but the survivors used
    to keep their unfiltered numbers — the model saw "1, 3, 5…", answered about
    numbers never asked, and real verdicts were dropped. The prompt must always
    count 1..N, and an Nth-line verdict must land on the Nth surviving sentence."""
    words = _words(
        "Bye. dosto aapake saath kabhee aisaa huaa hai ki aap kamare men gae ho. "
        "Ok. sab log aapako hi dekh rahe hain aisaa lagataa hai.")

    asked = {}

    async def ask(_system, user):
        asked["user"] = user
        # Flag the SECOND surviving sentence by its prompt number.
        return "1: OK\n2: BROKEN - leftover stumble words"

    result = await audit(words, ask)

    numbers = [line.split(":")[0] for line in
               asked["user"].split("Sentences:\n")[1].splitlines()]
    assert numbers == [str(n + 1) for n in range(len(numbers))]
    assert result.broken == 1
    assert "dekh rahe hain" in result.issues[0].text


# --- the repair must not change what the speaker meant ---------------------

def test_a_repair_may_never_remove_a_negation():
    """Measured on the first live run: the model "repaired" "hamane to notice
    naheen kiyaa" (we did NOT notice) by deleting "naheen", which says the
    opposite. No grammatical judgement justifies that."""
    text = "unhonne kahaa ki hamane to notice naheen kiyaa thaa"
    tokens = text.split()
    cuts = repair_deletions(_sentence(text), text.replace(" naheen", ""), tokens)
    assert cuts is None


def test_english_negations_are_protected_too():
    text = "but they said that they did not notice it at all"
    cuts = repair_deletions(_sentence(text), text.replace(" not", ""), text.split())
    assert cuts is None


def test_ordinary_debris_is_still_removable():
    text = "dosto aapake saath kabhee aisaa ha aap vah hai ki aap kamare men gae hoon"
    tokens = text.split()
    cuts = repair_deletions(_sentence(text), text.replace(" ha aap vah", ""), tokens)
    assert [tokens[c] for c in cuts] == ["ha", "aap", "vah"]


def test_a_pause_mid_clause_does_not_split_the_sentence():
    """At 1.0s the splitter cut sentences in half and then reported both halves
    as fragments — 'ends with "ki"' was the split, not the edit."""
    sentences = split_sentences(_words(
        "aapako lagataa hai ki sab ke najare aapake hain", pauses={4: 1.2}))
    assert len(sentences) == 1


def test_the_tail_of_a_pause_split_sentence_is_protected():
    """A sentence the splitter ended at a pause looks unfinished whether or not
    it is — the thought continues in the next one. The live run deleted
    "ho sakataa hai ki vah sab" whose continuation was in the very next
    sentence."""
    text = "par aisaa hotaa hai ki sab hamen dekh rahe hain ho sakataa vah"
    tokens = text.split()
    keep = "par aisaa hotaa hai ki sab hamen dekh rahe hain"     # trims the tail
    soft = Sentence(0, list(range(len(tokens))), text, hard_boundary=False)
    assert repair_deletions(soft, keep, tokens) is None

    # The same trim is allowed where the speaker actually stopped.
    hard = Sentence(0, list(range(len(tokens))), text, hard_boundary=True)
    cuts = repair_deletions(hard, keep, tokens)
    assert cuts is not None and [tokens[c] for c in cuts] == ["ho", "sakataa", "vah"]


def test_debris_in_the_middle_is_removable_either_way():
    text = "dosto aapake saath ha aap vah hai ki aap kamare men gae hoon"
    tokens = text.split()
    soft = Sentence(0, list(range(len(tokens))), text, hard_boundary=False)
    cuts = repair_deletions(soft, text.replace(" ha aap vah", ""), tokens)
    assert [tokens[c] for c in cuts] == ["ha", "aap", "vah"]


def test_punctuation_marks_a_hard_boundary():
    sentences = split_sentences(_words("dosto kya haal hai. aaj hum baat karenge"))
    assert sentences[0].hard_boundary is True
    assert sentences[1].hard_boundary is False


@pytest.mark.asyncio
async def test_a_continuing_line_is_marked_for_the_auditor():
    """Otherwise every pause-split line comes back flagged "ends unfinished",
    which is a complaint about the splitter, not about the edit."""
    words = _words("aapako lagataa hai ki sab ke najare aapake hain aur sab aapako dekh rahe")
    asked = {}

    async def ask(_system, user):
        asked["user"] = user
        return "1: OK"

    await audit(words, ask)
    assert asked["user"].rstrip().endswith("...")
