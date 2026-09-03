"""Tests for the span contract — the model names the runs to delete.

The risk this contract trades for is different from the rewrite contract's. A
rewrite can only fail by being wrong *text*, which alignment catches. A span can
fail by pointing at the wrong *place*: a model that miscounts writes a
well-formed instruction to delete words it never looked at, and nothing about the
answer's shape betrays it.

The quoted ends are the whole defence, so most of these tests are about them.
"""

import pytest

from backend.asr.fluency import (
    MAX_DELETED_FRACTION,
    judge_spans,
    number_tokens,
    plan_fluent_cuts,
)
from backend.asr.spans import (
    Span,
    looks_like_spans,
    merge,
    parse_spans,
    verify_spans,
)
from backend.asr.fluency import normalise


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


# --- parsing ---------------------------------------------------------------


def test_a_range_a_single_word_and_a_reason_are_all_read():
    spans = parse_spans(
        "DELETE 0-4 | dosto ... kya | retake\n"
        "DELETE 7 | um | filler\n", total=20)
    assert [(s.start, s.end, s.reason) for s in spans] == [
        (0, 4, "retake"), (7, 7, "filler")]
    assert (spans[0].first, spans[0].last) == ("dosto", "kya")


def test_commentary_around_the_answer_is_ignored_not_fatal():
    """A model that explains itself has still answered."""
    spans = parse_spans(
        "Here are the runs I would remove:\n"
        "DELETE 2-3 | a ... b | retake\n"
        "Hope that helps!", total=10)
    assert [(s.start, s.end) for s in spans] == [(2, 3)]


def test_a_span_running_past_the_window_is_clamped_not_dropped():
    spans = parse_spans("DELETE 8-99 | h ... z | retake", total=10)
    assert [(s.start, s.end) for s in spans] == [(8, 9)]


def test_a_span_entirely_past_the_window_is_dropped():
    assert parse_spans("DELETE 40-50 | x ... y | retake", total=10) == []


def test_none_is_an_answer_but_a_rewrite_is_not():
    """The two must be told apart: NONE is a decision to respect, a rewrite is
    the signal to fall back to the older contract. Reading a rewrite as "no
    spans" would silently mean "cut nothing in this window"."""
    assert looks_like_spans("NONE")
    assert looks_like_spans("DELETE 1-2 | a ... b | retake")
    assert not looks_like_spans("dosto kya apko pata hai india me")
    assert not looks_like_spans("")


# --- verification: the guard that makes an index trustworthy ----------------


def test_a_span_whose_quotes_match_is_kept_where_it_is():
    tokens = "dosto kya ap dosto kya apko pata hai".split()
    spans = parse_spans("DELETE 0-2 | dosto ... ap | retake", len(tokens))
    kept, refused = verify_spans(spans, tokens, normalise)
    assert refused == []
    assert [(s.start, s.end, s.relocated) for s in kept] == [(0, 2, False)]


def test_a_miscounted_span_is_moved_to_where_its_words_actually_are():
    """The standard objection to index output — the model counts wrong — is
    detectable here, and usually repairable: the quoted words say what it meant."""
    tokens = "dosto kya ap dosto kya apko pata hai".split()
    spans = parse_spans("DELETE 2-4 | dosto ... ap | retake", len(tokens))
    kept, refused = verify_spans(spans, tokens, normalise)
    assert refused == []
    assert [(s.start, s.end, s.relocated) for s in kept] == [(0, 2, True)]


def test_a_span_quoting_words_that_are_not_in_the_window_is_refused():
    """A model describing a transcript it was not given must delete nothing."""
    tokens = "dosto kya ap dosto kya apko pata hai".split()
    spans = parse_spans("DELETE 1-3 | friends ... place | retake", len(tokens))
    kept, refused = verify_spans(spans, tokens, normalise)
    assert kept == []
    assert len(refused) == 1


def test_relocation_never_changes_the_length_of_a_span():
    """A miscounted offset is a slip; a floating length would turn it into an
    arbitrary cut of the model's choosing."""
    tokens = "a b c d e f a b c".split()
    spans = parse_spans("DELETE 4-6 | a ... c | retake", len(tokens))
    kept, _ = verify_spans(spans, tokens, normalise)
    assert [(s.start, s.end) for s in kept] == [(6, 8)]


def test_an_unquoted_span_is_taken_at_its_word():
    """Quoting is what makes an index checkable, so an unquoted span is
    unverified rather than wrong — the trust limits stand behind it instead."""
    tokens = "a b c d e".split()
    kept, refused = verify_spans(parse_spans("DELETE 1-2", len(tokens)),
                                 tokens, normalise)
    assert refused == []
    assert [(s.start, s.end) for s in kept] == [(1, 2)]


def test_touching_and_overlapping_spans_become_one():
    merged = merge([Span(0, 3), Span(4, 6), Span(2, 5), Span(9, 9)])
    assert [(s.start, s.end) for s in merged] == [(0, 6), (9, 9)]


# --- judging: the trust limits ---------------------------------------------


def test_a_window_the_model_would_gut_is_discarded():
    tokens = [f"w{i}" for i in range(20)]
    limit = int(len(tokens) * MAX_DELETED_FRACTION) + 2
    plan, reasons = judge_spans(tokens, f"DELETE 0-{limit} | w0 ... w{limit} | filler")
    assert not plan.trusted
    assert reasons == {}


def test_a_gut_labelled_retake_is_refused_as_an_orphan_instead():
    """A retake deleting most of the window has no surviving copy, so the
    orphan guard refuses it before the fraction limit is even reached — the
    window stays usable and structure decides in the refused region."""
    tokens = [f"w{i}" for i in range(20)]
    limit = int(len(tokens) * MAX_DELETED_FRACTION) + 2
    plan, reasons = judge_spans(tokens, f"DELETE 0-{limit} | w0 ... w{limit} | retake")
    assert plan.trusted
    assert plan.deleted == set()
    assert plan.orphan_runs == 1


def test_the_model_reason_is_carried_out_per_index():
    tokens = "dosto kya ap um dosto kya apko".split()
    plan, reasons = judge_spans(tokens,
                                "DELETE 0-2 | dosto ... ap | retake\n"
                                "DELETE 3 | um | filler")
    assert plan.trusted
    assert plan.deleted == {0, 1, 2, 3}
    assert reasons == {0: "retake", 1: "retake", 2: "retake", 3: "filler"}


def test_none_is_a_trusted_answer_that_still_does_not_overrule_structure():
    """NONE is a real statement — unlike an echo, which says nothing — so the
    window is trusted and cuts nothing. It is deliberately NOT allowed to restore
    the structural retakes in its window: one lazy NONE would put a whole pile-up
    of attempts back into the edit, which is the exact fault being fixed."""
    tokens = "dosto kya apko pata hai".split()
    plan, reasons = judge_spans(tokens, "NONE")
    assert plan.trusted
    assert plan.deleted == set()
    assert not plan.confirmed_keep
    assert reasons == {}


@pytest.mark.asyncio
async def test_a_none_window_leaves_the_structural_decision_alone():
    """The end-to-end consequence of the rule above: no verdict is written for a
    window the model declined, so `refine_disfluencies` keeps what structure
    decided there rather than un-cutting it."""
    words = _words(" ".join(f"w{i}" for i in range(30)))

    async def ask(_system, _user):
        return "NONE"

    stats = {}
    decided = await plan_fluent_cuts(words, ask, stats=stats)
    assert stats["none_answers"] == 1
    assert decided == {}


def test_indices_are_reported_against_the_whole_word_list():
    tokens = "a b c d".split()
    plan, reasons = judge_spans(tokens, "DELETE 1-2 | b ... c | retake", start=100)
    assert plan.deleted == {101, 102}
    assert set(reasons) == {101, 102}


# --- numbering --------------------------------------------------------------


def test_every_word_is_numbered_and_pauses_are_not():
    """A pause is not a word and cannot be deleted, so numbering it would open a
    gap in the sequence the model then has to reason about."""
    words = _words("dosto kya apko pata", pauses={2: 1.5})
    numbered = number_tokens(words)
    assert numbered == "[0]dosto [1]kya | [2]apko [3]pata"


# --- end to end through plan_fluent_cuts ------------------------------------


@pytest.mark.asyncio
async def test_the_planner_reads_a_span_answer():
    words = _words("dosto kya ap dosto kya apko pata hai india me ek jagah")

    async def ask(_system, _user):
        return "DELETE 0-2 | dosto ... ap | retake"

    reasons = {}
    decided = await plan_fluent_cuts(words, ask, reasons=reasons)
    assert decided is not None
    assert [i for i, cut in decided.items() if cut] == [0, 1, 2]
    assert reasons == {0: "retake", 1: "retake", 2: "retake"}


@pytest.mark.asyncio
async def test_a_model_that_rewrites_instead_is_still_understood():
    """Asking for spans does not guarantee getting them. A model that ignores the
    format has still done the job, in the older shape, and must not be read as
    having found nothing to cut."""
    words = _words("dosto kya ap dosto kya apko pata hai india me ek jagah")

    async def ask(_system, user):
        text = user.split("Transcript:")[1].split("Answer:")[0]
        # strip the numbering and the first attempt, as a rewrite would
        plain = " ".join(t.split("]")[-1] for t in text.split())
        return plain.replace("dosto kya ap dosto kya ", "dosto kya ", 1)

    stats = {}
    decided = await plan_fluent_cuts(words, ask, stats=stats)
    assert decided is not None
    assert stats["rewrites"] == 1 and stats["spans"] == 0
    assert any(cut for cut in decided.values())


@pytest.mark.asyncio
async def test_a_span_answer_cannot_invent_words():
    """Delete-only, like everywhere else: an answer naming words that are not in
    the window removes nothing at all."""
    words = _words("dosto kya ap dosto kya apko pata hai india me ek jagah")

    async def ask(_system, _user):
        return "DELETE 0-3 | friends ... place | retake"

    decided = await plan_fluent_cuts(words, ask)
    assert decided is not None
    assert not any(cut for cut in decided.values())


# --- the JSON contract -------------------------------------------------------


def test_json_deletions_are_parsed_and_clamped():
    from backend.asr.spans import parse_json_spans
    spans = parse_json_spans(
        '{"deletions": ['
        '{"first_index": 0, "last_index": 2, "first_word": "dosto", '
        '"last_word": "ap", "reason": "retake"}, '
        '{"first_index": 8, "last_index": 99, "first_word": "x", '
        '"last_word": "y", "reason": "filler"}]}', total=10)
    assert [(s.start, s.end, s.reason) for s in spans] == [
        (0, 2, "retake"), (8, 9, "filler")]


def test_a_non_json_answer_is_not_mistaken_for_no_deletions():
    """None means "not this shape, fall back"; [] means "nothing to delete".
    Confusing the two would read a failed call as a decision to cut nothing."""
    from backend.asr.spans import parse_json_spans
    assert parse_json_spans("dosto kya apko pata hai", total=10) is None
    assert parse_json_spans("", total=10) is None
    assert parse_json_spans('{"deletions": []}', total=10) == []


def test_a_span_off_by_a_little_at_one_end_is_stretched_to_its_quoted_word():
    """The dominant real miscount: one end verified in place, the other a token
    or two off. With the good end anchored, the bad end moves to the word the
    model actually quoted — both ends then match the transcript."""
    tokens = "dosto kya ap gae hain jismen aur".split()
    spans = parse_spans("DELETE 0-5 | dosto ... gae | retake", len(tokens))
    kept, refused = verify_spans(spans, tokens, normalise)
    assert refused == []
    assert [(s.start, s.end, s.relocated) for s in kept] == [(0, 3, True)]


def test_an_oversized_run_is_refused_alone_and_the_rest_stands():
    """The rewrite contract had to discard the whole window on one enormous
    removal; a span verified its own quoted ends, so only the run the model
    lost its place on is refused."""
    tokens = [f"w{i}" for i in range(300)]
    answer = ("DELETE 0-4 | w0 ... w4 | filler\n"
              "DELETE 10-250 | w10 ... w250 | filler")
    plan, reasons = judge_spans(tokens, answer)
    assert plan.trusted
    assert plan.deleted == {0, 1, 2, 3, 4}
    assert plan.oversized_runs == 1
    assert 10 in plan.no_opinion and 250 in plan.no_opinion
    assert set(reasons) == {0, 1, 2, 3, 4}


@pytest.mark.asyncio
async def test_a_refused_region_is_not_written_as_a_keep():
    """Structure's cuts must stand where the model's ruling was unusable — a
    refused span is "ruled on, unusable", which is no opinion, never a keep."""
    words = _words(" ".join(f"w{i}" for i in range(30)))

    async def ask(_system, _user):
        return ("DELETE 0-1 | w0 ... w1 | retake\n"
                "DELETE 5-8 | zebra ... yak | retake")

    decided = await plan_fluent_cuts(words, ask)
    assert decided is not None
    assert decided.get(0) is True and decided.get(1) is True
    for index in range(5, 9):
        assert index not in decided
    assert decided.get(3) is False


@pytest.mark.asyncio
async def test_the_constrained_json_path_is_preferred_and_verified():
    words = _words("dosto kya ap dosto kya apko pata hai india me ek jagah")
    calls = {}

    async def ask(_system, _user):
        calls["prose"] = True
        return "NONE"

    async def ask_json(_system, _user, schema):
        calls["json"] = schema["name"]
        return ('{"deletions": [{"first_index": 0, "last_index": 2, '
                '"first_word": "dosto", "last_word": "ap", "reason": "retake"}]}')

    stats = {}
    reasons = {}
    decided = await plan_fluent_cuts(words, ask, stats=stats, reasons=reasons,
                                     ask_json=ask_json)
    assert calls.get("json") == "deletions"
    assert "prose" not in calls
    assert [i for i, cut in decided.items() if cut] == [0, 1, 2]
    assert reasons[0] == "retake"
    assert stats["spans"] == 1 and stats["rewrites"] == 0


@pytest.mark.asyncio
async def test_a_failed_json_ask_falls_back_to_prose():
    words = _words("dosto kya ap dosto kya apko pata hai")

    async def ask_json(_system, _user, _schema):
        return None

    async def ask(_system, _user):
        return "DELETE 0-2 | dosto ... ap | retake"

    decided = await plan_fluent_cuts(words, ask, ask_json=ask_json)
    assert decided is not None
    assert [i for i, cut in decided.items() if cut] == [0, 1, 2]


def test_a_retake_span_with_no_surviving_copy_is_refused():
    """"Retake" claims a surviving copy exists — delete the earlier attempts,
    keep the last. A model once named BOTH attempts at a story as retakes and
    the story left the video entirely. With nothing surviving that resembles a
    span, the span is refused and structure (which keeps the last copy)
    decides there instead."""
    tokens = ("unhone bahut sare bachon ko ek tshirt pahnaya "
              "unhone bahut sare bachon ko ek tshirt pahnaya "
              "unko room me bheja gaya aur unse pucha gaya").split()
    answer = ("DELETE 0-7 | unhone ... pahnaya | retake\n"
              "DELETE 8-15 | unhone ... pahnaya | retake")
    plan, reasons = judge_spans(tokens, answer)
    assert plan.trusted
    assert plan.deleted == set()
    assert plan.orphan_runs == 2
    assert 0 in plan.no_opinion and 15 in plan.no_opinion


def test_a_retake_span_whose_copy_survives_is_kept():
    tokens = ("unhone bahut sare bachon ko ek tshirt pahnaya "
              "unhone bahut sare bachon ko ek tshirt pahnaya "
              "unko room me bheja gaya aur unse pucha gaya").split()
    plan, reasons = judge_spans(tokens, "DELETE 0-7 | unhone ... pahnaya | retake")
    assert plan.trusted
    assert plan.deleted == set(range(0, 8))
    assert plan.orphan_runs == 0
