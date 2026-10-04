"""Fluency-driven cut planning: the LLM decides, structure only guards.

The structural detector (retakes.py) asks "does this run of words reappear
shortly?". That question has no notion of a sentence, so it happily cuts *inside*
the surviving take and leaves a stitched-together fragment — measured on a real
recording as:

    aap kisee kamare [men gae hoon jahaan par bahut saare log hai phrikteev hol]
    men gae ho phir baahar gayaa hai

which is not something anybody said. The only component that can judge "is what
is left a complete, natural sentence?" is a language model, so it is asked
directly: here is what the speaker said, give back how it should read.

**The model may only delete.** It is asked for the cleaned transcript in the
speaker's own words, and the answer is aligned back onto the original token
sequence. Alignment turns that text into indices:

  - `equal`    -> those words are kept
  - `delete`   -> those words are cut          <- the only thing that cuts
  - `replace`  -> the model reworded; the ORIGINAL words are kept, because a
                  reworded span is a guess and cutting on a guess is how content
                  gets destroyed
  - `insert`   -> ignored outright; there is no video for words nobody said

So a model that paraphrases, translates or hallucinates cannot delete anything —
it can only fail to delete. That asymmetry is deliberate: a missed fumble is a
blemish, a deleted sentence is a defect.
"""

import difflib
import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("fluency")

# Fluency windows asked at once (LM Studio serves 4 parallel slots by default).
FLUENCY_CONCURRENCY = 4

_PUNCT_RE = re.compile(r"[^\w]", re.UNICODE)

# Windowing. Long enough that a pile-up of restarts is visible whole (the real
# case ran five attempts over ~50 words), short enough that a small local model
# does not lose its place.
WINDOW_WORDS = 200
OVERLAP_WORDS = 40

# A pause this long is shown to the model: it is the clearest signal of where one
# attempt stopped and the next began.
PAUSE_MARKER_SECONDS = 0.7
PAUSE_MARKER = "|"

# Trust limits. A window failing any of these is discarded and the structural
# decision stands for it.
MIN_MATCH_RATIO = 0.35      # how much of the window the model echoed verbatim
MAX_DELETED_FRACTION = 0.75  # beyond this it is summarising, not editing
# A single removal this large is the model losing its place rather than cutting a
# fumble. Measured as a *fraction* of the window, not an absolute: the real
# recording opens with five attempts at one sentence and the correct edit deletes
# 48 consecutive words, which a fixed 45-word ceiling threw away. What actually
# needs catching is a model that drops half the window in one swipe.
MAX_DELETED_RUN_FRACTION = 0.40
MIN_DELETED_RUN_ALLOWANCE = 60


def normalise(text: str) -> str:
    """Lower-cased letters/digits, keeping Indic combining vowel signs (matras).

    The old `[^\\w]` stripped Devanagari matras — नहीं became नह — because Python's
    `\\w` excludes combining marks. That was harmless while every pass judged the
    romanized "Hinglish" text, but the grammar passes now judge the native script,
    where dropping the matra changes the word (and would let a "repair" delete a
    negation it could no longer recognise). Keep Unicode letters (L), numbers (N)
    and marks (M) — the last is the vowel signs — plus the "_" placeholder. For
    Latin text this is identical to the old behaviour, so English is unaffected.
    """
    kept = [ch for ch in str(text or "")
            if ch == "_" or unicodedata.category(ch)[0] in ("L", "N", "M")]
    return "".join(kept).lower()


def _token(word: Dict[str, Any]) -> str:
    """The token to JUDGE — the native script when the word carries one.

    Whisper hears Hindi in Devanagari; the pipeline also keeps a romanized
    "Hinglish" copy for captions and the timeline. Judging fluency and grammar on
    that romanized copy is the bug: to the model "aapake saath kabhee" reads as
    broken English, so it scores good Hindi as wrong and tries to "fix" spellings
    that were only ever a romanization artefact. Judge the native script instead;
    fall back to the romanized / `text` form when there is no native (English
    speech, or a word added by hand).
    """
    return str(word.get("word_native") or word.get("word") or word.get("text", ""))


@dataclass
class WindowPlan:
    """What the model decided about one window, and whether to believe it."""
    start: int
    deleted: Set[int] = field(default_factory=set)
    trusted: bool = False
    reason: str = ""
    match_ratio: float = 0.0
    # True only when the answer is a real edit signal — the model deleted
    # something. A verbatim echo scores a perfect ratio and gets trusted, but it
    # proves nothing about any individual word: a small model that echoes must
    # not be allowed to overturn the structural cuts with "keep everything".
    confirmed_keep: bool = False
    # Indices (absolute) where the model's opinion was refused rather than
    # absent — a span whose quotes matched nothing, or a run so large the model
    # had lost its place. Never written as keeps: the structural decision stands
    # there, exactly as it does for a window that answered nothing at all.
    no_opinion: Set[int] = field(default_factory=set)
    # How many merged runs were refused for size alone. A window can be trusted
    # around them; the counter is surfaced so a model that routinely loses its
    # place is visible in the report.
    oversized_runs: int = 0
    # How many retake spans were refused because nothing surviving is a copy of
    # them — the model deleting the only telling of something (see judge_spans).
    orphan_runs: int = 0


def build_prompt_text(words: Sequence[Dict[str, Any]]) -> str:
    """The spoken text, with a marker where the speaker paused.

    Pause markers are stripped again before alignment, so the model dropping or
    keeping them changes nothing.
    """
    parts: List[str] = []
    previous_end: Optional[float] = None
    for word in words:
        start = float(word.get("start", 0.0) or 0.0)
        if previous_end is not None and start - previous_end >= PAUSE_MARKER_SECONDS:
            parts.append(PAUSE_MARKER)
        parts.append(_token(word).strip() or "_")
        previous_end = float(word.get("end", start) or start)
    return " ".join(parts)


def align_deletions(original: Sequence[str], cleaned: Sequence[str]) -> Tuple[Set[int], float]:
    """Which original indices the cleaned text dropped, and how much matched.

    Only `delete` opcodes cut. See the module docstring for why `replace` does
    not: the model reworded there, and the original is what is on tape.
    """
    left = [normalise(t) for t in original]
    right = [normalise(t) for t in cleaned if normalise(t)]
    if not left:
        return set(), 0.0

    # Aligned from the RIGHT, and that is load-bearing. Every attempt at a
    # sentence opens with the same words, so "keep dosto aapake saath kabhee
    # aisaa huaa hai ki aap" matches all five equally well and a left-anchored
    # diff picks the first. The text then reads perfectly while the *audio*
    # splices attempt one's opening at 1.5s onto attempt five's continuation at
    # 17.2s — two different takes joined mid-sentence, which is precisely the
    # seam a listener hears. Reversing makes ties fall to the last occurrence,
    # which is the take that actually ran to completion.
    count = len(left)
    matcher = difflib.SequenceMatcher(a=left[::-1], b=right[::-1], autojunk=False)
    deleted: Set[int] = set()
    equal = 0
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "delete":
            deleted.update(count - 1 - index for index in range(i1, i2))
        elif tag == "equal":
            equal += i2 - i1
    return deleted, equal / count


def _runs(indices: Sequence[int]) -> List[List[int]]:
    """Consecutive indices grouped into runs."""
    runs: List[List[int]] = []
    for index in sorted(indices):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _similar(left: Sequence[str], right: Sequence[str]) -> float:
    return difflib.SequenceMatcher(a=list(left), b=list(right), autojunk=False).ratio()


def snap_runs_to_the_final_take(tokens: Sequence[str], starts: Sequence[float],
                                deleted: Set[int],
                                min_jump_seconds: float = 2.0,
                                min_similarity: float = 0.6) -> Set[int]:
    """Pull each surviving run forward onto the take that follows it.

    The model writes the best possible *text*, which is not always speakable from
    a single take: it took "dosto aapake saath kabhee aisaa huaa hai ki aap aise"
    from the third attempt and "kamare men gae hoon" from the fifth, because
    neither attempt is clean on its own. The text reads perfectly and the audio
    jumps eight seconds backwards mid-sentence — a seam no listener misses.

    So wherever a surviving run is followed by a long jump, and the words
    immediately before the next run are a rough copy of it, that run is an
    earlier attempt at the same words: it moves to the later position. What
    survives then comes from one continuous stretch of speech, and the second
    fluency pass tidies up inside it.
    """
    survivors = [i for i in range(len(tokens)) if i not in deleted]
    if len(survivors) < 2:
        return deleted

    # Group survivors into blocks of continuous speech. Not raw runs: a fumble
    # cut *inside* a take splits it into several runs that are still a fraction
    # of a second apart, and each fragment alone is too short to recognise as an
    # earlier attempt. What matters is the audible jump between blocks.
    blocks: List[List[int]] = [[survivors[0]]]
    for previous, current in zip(survivors, survivors[1:]):
        if starts[current] - starts[previous] < min_jump_seconds:
            blocks[-1].append(current)
        else:
            blocks.append([current])

    for position, block in enumerate(blocks[:-1]):
        next_start = blocks[position + 1][0]
        # Where does the later attempt begin? At its own first word — the same
        # word this block opens with. Taking a fixed len(block) window instead
        # lands mid-phrase and beheads the sentence: the opening "dosto aapake
        # saath kabhee aisaa" became "aap vah hai ki aap", which reads as a cut
        # into the middle of a thought.
        opener = tokens[block[0]]
        candidate_start = next((p for p in range(next_start - 1, block[-1], -1)
                                if tokens[p] == opener), next_start - len(block))
        if candidate_start <= block[-1]:
            continue        # no room for a distinct earlier copy
        candidate = list(range(candidate_start, next_start))
        if _similar([tokens[i] for i in block], [tokens[i] for i in candidate]) < min_similarity:
            continue
        logger.info("Fluency: moving %d words from %.1fs to %.1fs so the take stays "
                    "continuous", len(block), starts[block[0]], starts[candidate_start])
        deleted = (deleted | set(block)) - set(candidate)

    return deleted


# How much of a deleted attempt's tail must reappear in the survivors for the
# deletion to count as a real retake. The tail, not the whole span: a span may
# merge five attempts, and only the last of them resembles the take that
# survives it.
_COPY_TAIL_TOKENS = 12
_COPY_MIN_RATIO = 0.55


def _copy_survives(span_tokens: List[str], survivors: Sequence[str]) -> bool:
    """Whether the survivors contain a rough copy of the span's final attempt."""
    tail = [t for t in span_tokens if t and t != "uh"][-_COPY_TAIL_TOKENS:]
    if not tail:
        return True                      # nothing substantive was deleted
    if len(survivors) < len(tail):
        return False
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(tail)
    for position in range(0, len(survivors) - len(tail) + 1):
        matcher.set_seq1(list(survivors[position:position + len(tail)]))
        if (matcher.real_quick_ratio() >= _COPY_MIN_RATIO
                and matcher.quick_ratio() >= _COPY_MIN_RATIO
                and matcher.ratio() >= _COPY_MIN_RATIO):
            return True
    return False


def _longest_run(indices: Set[int]) -> int:
    if not indices:
        return 0
    ordered = sorted(indices)
    longest = run = 1
    for previous, current in zip(ordered, ordered[1:]):
        run = run + 1 if current == previous + 1 else 1
        longest = max(longest, run)
    return longest


def judge_window(original: Sequence[str], cleaned: Sequence[str],
                 start: int = 0, starts: Optional[Sequence[float]] = None) -> WindowPlan:
    """Align one window's answer and decide whether it can be trusted.

    `starts` are the words' times; given them, surviving runs are snapped onto
    the take that follows so the edit plays from continuous speech.
    """
    plan = WindowPlan(start=start)
    deleted, ratio = align_deletions(original, cleaned)
    plan.match_ratio = round(ratio, 3)
    if starts is not None and len(starts) == len(original):
        deleted = snap_runs_to_the_final_take(original, starts, deleted)

    if not cleaned:
        plan.reason = "empty answer"
        return plan
    if ratio < MIN_MATCH_RATIO:
        plan.reason = f"only {ratio:.0%} of the window came back verbatim"
        return plan
    if original and len(deleted) / len(original) > MAX_DELETED_FRACTION:
        plan.reason = f"would delete {len(deleted)}/{len(original)} words"
        return plan
    run = _longest_run(deleted)
    allowed_run = max(MIN_DELETED_RUN_ALLOWANCE,
                      int(len(original) * MAX_DELETED_RUN_FRACTION))
    if run > allowed_run:
        plan.reason = f"single {run}-word removal (limit {allowed_run})"
        return plan

    plan.deleted = {start + index for index in deleted}
    plan.trusted = True
    plan.confirmed_keep = bool(deleted)
    return plan


def windows(total: int, size: int = WINDOW_WORDS,
            overlap: int = OVERLAP_WORDS) -> List[Tuple[int, int, int, int]]:
    """(start, end, core_start, core_end) spans covering `total` words.

    Each window carries context either side of the part it actually decides, so
    a restart straddling a boundary is still seen whole by the window that owns
    it.
    """
    if total <= 0:
        return []
    if total <= size:
        return [(0, total, 0, total)]
    spans: List[Tuple[int, int, int, int]] = []
    core_start = 0
    while core_start < total:
        core_end = min(total, core_start + size)
        start = max(0, core_start - overlap)
        end = min(total, core_end + overlap)
        spans.append((start, end, core_start, core_end))
        core_start = core_end
    return spans


SYSTEM_PROMPT = (
    "You are editing a transcript of unscripted speech so it can be cut in a video "
    "editor. The speaker fumbles: they begin a sentence, break off, and begin again, "
    "sometimes four or five times, before finally getting it out. They also make "
    "filler sounds.\n\n"
    "The speech is HINDI, written in its native Devanagari script, with English "
    "words mixed in as Hindi speakers naturally do. Judge it as Hindi: apply Hindi "
    "grammar and Hindi sentence sense, NOT English. (Occasionally a transcript is "
    "wholly English — judge that as English.) A '|' marks a pause in the recording.\n\n"
    "Give back the transcript as it should sound in the finished video.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Use ONLY the speaker's own words, spelled exactly as given, in the same "
    "order. Never translate, never romanize, never rephrase, never correct spelling "
    "or grammar. Never add a word that is not there.\n"
    "2. When a sentence is attempted more than once, keep ONLY the last attempt "
    "that runs to completion and delete every earlier attempt IN FULL. Never stitch "
    "parts of different attempts together — the result must be one clean take.\n"
    "3. Delete filler sounds and stutters.\n"
    "4. Keep anything the speaker meant: a list of examples, a phrase repeated for "
    "emphasis, a point restated. Repetition on purpose is not a fumble.\n"
    "5. What remains must read as a complete, fluent, grammatical HINDI sentence.\n"
    "6. Output the words only — no commentary, no quotes, no formatting.\n\n"
    "WORKED EXAMPLE — this is the whole job:\n"
    "Transcript:\n"
    "दोस्तों क्या आप | दोस्तों क्या | दोस्तों क्या अ | दोस्तों दोस्तों क्या आपको पता है इंडिया "
    "में एक ऐसी जगह है जो बहुत amazing है\n"
    "Cleaned transcript:\n"
    "दोस्तों क्या आपको पता है इंडिया में एक ऐसी जगह है जो बहुत amazing है\n\n"
    "Note what happened: the speaker tried the sentence four times and only the "
    "last one finished, so the first four attempts were deleted whole and nothing "
    "from them was reused. The English word 'amazing' was kept as spoken. Most "
    "transcripts you get will need cuts like this."
)


def user_prompt(text: str) -> str:
    return f"Transcript:\n{text}\n\nCleaned transcript:"


# --- the span contract ------------------------------------------------------
#
# The prompt above asks for the cleaned TEXT, and `align_deletions` then works
# out what was removed by diffing it back. The one below asks for the removals
# directly. See `asr/spans.py` for why: that diff is a reconstruction of a
# decision the model already made and was never asked to state, and the most
# fragile machinery in this module — right-anchored alignment, run snapping,
# shape-guessing about what counts as repair — exists only to survive it.

SPAN_SYSTEM_PROMPT = (
    "You are editing a transcript of unscripted speech so it can be cut in a "
    "video editor. The speaker fumbles: they begin a sentence, break off, and "
    "begin again, sometimes four or five times, before finally getting it out. "
    "They also make filler sounds.\n\n"
    "The speech is HINDI, written in its native Devanagari script, with English "
    "words mixed in as Hindi speakers naturally do. Judge it as Hindi, applying "
    "Hindi grammar and Hindi sentence sense, NOT English. (Occasionally a "
    "transcript is wholly English — judge that as English.) A '|' marks a pause "
    "in the recording.\n\n"
    "Every word is numbered: [0]word [1]word [2]word …\n\n"
    "Name the runs of words to DELETE, so that what remains plays as one "
    "continuous, fluent take. Answer with one line per run and nothing else:\n\n"
    "DELETE <first>-<last> | <first word> ... <last word> | <reason>\n\n"
    "The reason is one of: retake, filler, stutter, repeat.\n"
    "Quote the first and last word of the run exactly as they appear, so the "
    "numbers can be checked. For a single word write: DELETE <n> | <word> | "
    "<reason>\n"
    "If there is nothing to delete, answer with the single word NONE. That is a "
    "normal answer, not a failure.\n\n"
    "WHAT TO DELETE:\n"
    "1. When a sentence is attempted more than once, delete every earlier "
    "attempt IN FULL — the whole run, from where that attempt starts to where "
    "the next one begins — and keep only the last attempt that runs to "
    "completion. Never delete part of an attempt: half an attempt left behind "
    "is debris wedged into the middle of a good sentence.\n"
    "2. Filler sounds and stutters.\n"
    "3. A phrase the speaker ended up saying twice by accident — delete the "
    "first copy.\n\n"
    "WHAT TO KEEP:\n"
    "4. Anything the speaker meant: a list of examples, a phrase repeated for "
    "emphasis, a point restated. Repetition on purpose is not a fumble.\n"
    "5. Never delete a negation.\n"
    "6. Spelling is never a reason to delete anything.\n\n"
    "WORKED EXAMPLE — this is the whole job:\n"
    "[0]दोस्तों [1]क्या [2]आप [3]दोस्तों [4]क्या | [5]दोस्तों [6]क्या [7]अ | "
    "[8]दोस्तों [9]दोस्तों [10]क्या [11]आपको [12]पता [13]है [14]इंडिया [15]में "
    "[16]एक [17]ऐसी [18]जगह [19]है\n"
    "Answer:\n"
    "DELETE 0-4 | दोस्तों ... क्या | retake\n"
    "DELETE 5-7 | दोस्तों ... अ | retake\n"
    "DELETE 8 | दोस्तों | stutter\n\n"
    "Note what happened: the speaker tried the sentence four times and only the "
    "last one finished, so each earlier attempt was deleted as a whole run and "
    "nothing from any of them was reused."
)


# The same contract as SPAN_SYSTEM_PROMPT, answered as JSON under a server-side
# schema. This exists because the DELETE-line format is a request a small model
# is free to ignore — and the available 12B did, on every window of the real
# recording: it rewrote (paraphrased, 6-11% verbatim) instead, so the fluency
# layer shipped nothing. LM Studio's json_schema response format *constrains*
# generation, so the same model answers in span form every time. Verification is
# unchanged — quoted ends are still checked against the tokens — so the
# constrained answer is no more trusted, just reliably parseable.
JSON_SPAN_SYSTEM_PROMPT = (
    "You are editing a transcript of unscripted speech so it can be cut in a "
    "video editor. The speaker fumbles: they begin a sentence, break off, and "
    "begin again, sometimes four or five times, before finally getting it out. "
    "They also make filler sounds.\n\n"
    "The speech is HINDI, written in its native Devanagari script, with English "
    "words mixed in as Hindi speakers naturally do. Judge it as Hindi, applying "
    "Hindi grammar and Hindi sentence sense, NOT English. (Occasionally a "
    "transcript is wholly English — judge that as English.) A '|' marks a pause "
    "in the recording.\n\n"
    "Every word is numbered: [0]word [1]word [2]word …\n\n"
    "List the runs of words to DELETE, so that what remains plays as one "
    "continuous, fluent take. Each deletion gives the index AND the exact word "
    "at both ends (first_index/first_word, last_index/last_word) so the numbers "
    "can be checked, plus a reason: retake, filler, stutter or repeat.\n\n"
    "WHAT TO DELETE:\n"
    "1. When a sentence is attempted more than once, delete every earlier "
    "attempt IN FULL — the whole run, from where that attempt starts to where "
    "the next attempt begins — and keep only the last attempt that runs to "
    "completion. Each earlier attempt is its own deletion; never delete part "
    "of an attempt, and never fold the good take into a deletion.\n"
    "2. Filler sounds and stutters.\n"
    "3. A phrase the speaker ended up saying twice by accident — delete the "
    "first copy.\n\n"
    "WHAT TO KEEP:\n"
    "4. Anything the speaker meant: a list of examples, a phrase repeated for "
    "emphasis, a point restated. Repetition on purpose is not a fumble.\n"
    "5. Never delete a negation.\n"
    "6. Spelling is never a reason to delete anything — a machine wrote this "
    "transcript down and spells badly; the audio underneath is perfect.\n\n"
    "After your deletions, read what remains: it must be complete, fluent, "
    "grammatical HINDI. If deleting a run would leave a sentence without its "
    "ending, the run is wrong — make it smaller or larger until what remains "
    "is whole. If there is nothing to delete, return an empty list.\n\n"
    "WORKED EXAMPLE — this is the whole job:\n"
    "[0]दोस्तों [1]क्या [2]आप [3]दोस्तों [4]क्या | [5]दोस्तों [6]क्या [7]अ | "
    "[8]दोस्तों [9]दोस्तों [10]क्या [11]आपको [12]पता [13]है [14]इंडिया [15]में "
    "[16]एक [17]ऐसी [18]जगह [19]है\n"
    "Answer:\n"
    '{"deletions": ['
    '{"first_index": 0, "last_index": 4, "first_word": "दोस्तों", '
    '"last_word": "क्या", "reason": "retake"}, '
    '{"first_index": 5, "last_index": 7, "first_word": "दोस्तों", '
    '"last_word": "अ", "reason": "retake"}, '
    '{"first_index": 8, "last_index": 8, "first_word": "दोस्तों", '
    '"last_word": "दोस्तों", "reason": "stutter"}]}\n\n'
    "Note what happened: the speaker tried the sentence four times and only the "
    "last one finished, so each earlier attempt was deleted as its own whole "
    "run, and the surviving text — words 9 to 19 — reads as one complete "
    "sentence."
)

SPAN_SCHEMA = {
    "name": "deletions",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "deletions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "first_index": {"type": "integer"},
                        "last_index": {"type": "integer"},
                        "first_word": {"type": "string"},
                        "last_word": {"type": "string"},
                        "reason": {"type": "string",
                                   "enum": ["retake", "filler", "stutter",
                                            "repeat"]},
                    },
                    "required": ["first_index", "last_index", "first_word",
                                 "last_word", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["deletions"],
        "additionalProperties": False,
    },
}


def number_tokens(words: Sequence[Dict[str, Any]]) -> str:
    """The window as `[i]word` pairs, with a marker where the speaker paused.

    Pause markers carry no index — they are not words and cannot be deleted, so
    numbering them would open gaps in the sequence that the model then has to
    reason about. They sit between numbered tokens purely as the signal of where
    one attempt stopped and the next began.
    """
    parts: List[str] = []
    previous_end: Optional[float] = None
    for index, word in enumerate(words):
        start = float(word.get("start", 0.0) or 0.0)
        if previous_end is not None and start - previous_end >= PAUSE_MARKER_SECONDS:
            parts.append(PAUSE_MARKER)
        parts.append(f"[{index}]{_token(word).strip() or '_'}")
        previous_end = float(word.get("end", start) or start)
    return " ".join(parts)


def span_user_prompt(numbered: str) -> str:
    return f"Transcript:\n{numbered}\n\nAnswer:"


def judge_spans(tokens: Sequence[str], answer: str, start: int = 0,
                spans: Optional[List] = None) -> Tuple[WindowPlan, Dict[int, str]]:
    """Turn one window's span answer into a verdict, plus the reason per index.

    `spans` lets a caller hand in spans it already parsed (the JSON contract);
    otherwise the DELETE-line parser reads `answer`.

    The deleted-fraction limit from `judge_window` applies: a model asking to
    delete three-quarters of a window is summarising. `MIN_MATCH_RATIO` does NOT
    — there is no rewritten text here to have echoed, and every surviving word
    is an original token by construction. The single-enormous-run limit applies
    PER RUN rather than to the window: each span verified its own quoted ends
    independently, so one run the model lost its place on refuses that run and
    nothing else — where the rewrite contract had to throw the window away,
    because a diff gives its removals no independent evidence to stand on.
    Refused regions become `no_opinion`, never keeps: structure stands there.
    """
    from .spans import merge, parse_spans, verify_spans

    plan = WindowPlan(start=start, match_ratio=1.0)
    parsed = spans if spans is not None else parse_spans(answer, len(tokens))
    verified, refused = verify_spans(parsed, tokens, normalise)
    for span in refused:
        logger.warning("Fluency: refusing span %d-%d quoting %r...%r",
                       start + span.start, start + span.end, span.first, span.last)
        plan.no_opinion.update(start + index for index in span.indices())

    # A span that claims to remove a RETAKE is claiming a surviving copy exists
    # — "delete every earlier attempt, keep the last one" implies a last one is
    # kept. Verify the implication: the span's tail (its final attempt) must
    # have a rough copy in what the window's own plan lets survive. Measured
    # live: the model named BOTH attempts at the t-shirt story as retakes, the
    # spans verified (real quotes, legal sizes), and the story left the video
    # entirely. A retake with no surviving copy is the model deleting the only
    # telling of something; it is refused, and the structural decision — which
    # always keeps the last copy — stands there instead.
    surviving = set(range(len(tokens)))
    for span in verified:
        surviving -= span.indices()
    survivor_tokens = [normalise(tokens[index]) for index in sorted(surviving)]
    survivor_tokens = [t for t in survivor_tokens if t and t != "uh"]
    kept_spans: List = []
    for span in verified:
        if span.reason == "retake" and span.length >= 4 and not _copy_survives(
                [normalise(tokens[index]) for index in sorted(span.indices())],
                survivor_tokens):
            logger.warning("Fluency: refusing retake span %d-%d — nothing that "
                           "survives is a copy of it, so it would delete the "
                           "only telling", start + span.start, start + span.end)
            plan.orphan_runs += 1
            plan.no_opinion.update(start + index for index in span.indices())
            continue
        kept_spans.append(span)
    verified = kept_spans
    # Reasons come from the spans as the model wrote them, BEFORE merging. A
    # filler that happens to sit against the end of a retake is one touching run
    # after merging, and reading the reason off the merged span labels the "um"
    # a retake — losing the more specific answer the model actually gave.
    reasons: Dict[int, str] = {}
    for span in verified:
        for index in span.indices():
            key = start + index
            if reasons.get(key) in (None, "not_fluent"):
                reasons[key] = span.reason

    # Merged only for the extent and the run-length limit below, where touching
    # spans genuinely are one removal.
    deleted: Set[int] = set()
    allowed = max(MIN_DELETED_RUN_ALLOWANCE,
                  int(len(tokens) * MAX_DELETED_RUN_FRACTION))
    for span in merge(verified):
        if span.length > allowed:
            logger.warning("Fluency: refusing a single %d-word removal at %d-%d "
                           "(limit %d); the rest of the window stands",
                           span.length, start + span.start, start + span.end, allowed)
            plan.oversized_runs += 1
            plan.no_opinion.update(start + index for index in span.indices())
            continue
        deleted |= span.indices()

    if tokens and len(deleted) / len(tokens) > MAX_DELETED_FRACTION:
        plan.reason = f"would delete {len(deleted)}/{len(tokens)} words"
        return plan, {}

    plan.deleted = {start + index for index in deleted}
    plan.no_opinion -= plan.deleted
    reasons = {key: value for key, value in reasons.items() if key in plan.deleted}
    plan.trusted = True
    # An empty answer here is, unlike an echo, a real statement: a model that
    # writes NONE has read the window and said there is nothing to cut, where a
    # model echoing the transcript back has said nothing about any word in it.
    # The contract could therefore let a NONE restore the structural retakes in
    # its window — and it deliberately does not, yet. Restoring them is the safe
    # direction for the asymmetry rule but the wrong direction for the fault
    # actually being reported, which is repeated attempts surviving into the cut;
    # one lazy NONE would put a whole pile-up back. `none_answers` in the window
    # stats counts how often the model declines, so this can be decided on a
    # measurement rather than on the strength of the argument.
    plan.confirmed_keep = bool(deleted)
    return plan, reasons


async def plan_fluent_cuts(
    words: Sequence[Dict[str, Any]],
    ask,
    stats: Optional[Dict[str, int]] = None,
    reasons: Optional[Dict[int, str]] = None,
    use_spans: bool = True,
    ask_json=None,
) -> Optional[Dict[int, bool]]:
    """What the model decided, per word index, or None if it could not be used.

    The result maps only the indices the model actually ruled on — the cores of
    windows that passed the trust limits — to "cut" or "keep". Everything else is
    absent, and the caller must leave the structural decision alone there. A
    window the model fumbled must not silently un-cut the fumbles structure
    found in it.

    `ask` is an async callable taking (system_prompt, user_prompt) and returning
    the model's text, or None. Injected so the planning logic can be tested
    without a model.

    `stats`, when given, receives counters — answered / failed / untrusted
    windows, and how many windows each contract handled — so the caller can
    report how much of the edit actually got a fluency opinion rather than
    silently falling back to structure.

    `reasons`, when given, receives the model's own reason per cut index
    ("retake", "filler", "stutter"). Only the span contract can fill it: a
    rewritten transcript says what to remove but never why.

    `use_spans` asks the model to name the runs to delete rather than to rewrite
    the transcript (see `SPAN_SYSTEM_PROMPT`). A model that ignores the format
    and rewrites anyway is handled per window by falling back to the older
    contract, so this is safe to leave on with any model.

    `ask_json`, when given, is an async callable (system, user, schema) -> text
    that constrains the answer server-side (LM Studio's json_schema response
    format). This is how the span contract actually lands on a small local
    model: asked in prose, the available 12B ignored the DELETE-line format on
    every real window and rewrote instead. Constrained, it answers in span form
    every time — and the answer still goes through the same quoted-ends
    verification, so it is no more trusted, just reliably parseable. Falls back
    to `ask` per window when the constrained call fails.
    """
    if not words:
        return {}

    tokens = [_token(w) for w in words]
    decided: Dict[int, bool] = {}
    trusted_any = False
    counters = stats if stats is not None else {}
    counters.update(answered=0, failed=0, untrusted=0, spans=0, rewrites=0,
                    none_answers=0, oversized_runs=0, orphan_runs=0)
    found_reasons = reasons if reasons is not None else {}

    all_windows = list(windows(len(words)))
    # Every window's first JSON answer is asked for up front, a few at a time:
    # windows are independent, and asked one after another the pass took ~6
    # minutes of a real 13:50 take. Retries and the prose fallback below stay
    # sequential -- they are rare.
    first_answers: List[Optional[str]] = [None] * len(all_windows)
    if use_spans and ask_json is not None and all_windows:
        gate = asyncio.Semaphore(FLUENCY_CONCURRENCY)

        async def first(window_words) -> Optional[str]:
            async with gate:
                try:
                    return await ask_json(JSON_SPAN_SYSTEM_PROMPT,
                                          span_user_prompt(number_tokens(window_words)), SPAN_SCHEMA)
                except Exception as e:
                    logger.warning("Fluency: JSON request failed (%s)", e)
                    return None

        first_answers = list(await asyncio.gather(
            *(first(words[s:e]) for s, e, _cs, _ce in all_windows)))

    for window_number, (start, end, core_start, core_end) in enumerate(all_windows):
        window = words[start:end]
        plan = None
        window_reasons: Dict[int, str] = {}

        # The constrained JSON path first, when the caller can offer it.
        if use_spans and ask_json is not None:
            from .spans import parse_json_spans
            prompt = span_user_prompt(number_tokens(window))
            answer = first_answers[window_number]
            if not answer:
                logger.warning("Fluency: no JSON answer for words %d-%d; retrying once",
                               start, end)
                answer = await ask_json(JSON_SPAN_SYSTEM_PROMPT, prompt, SPAN_SCHEMA)
            parsed = parse_json_spans(answer or "", len(window))
            if parsed is not None:
                counters["answered"] += 1
                counters["spans"] += 1
                plan, window_reasons = judge_spans(tokens[start:end], "",
                                                   start=start, spans=parsed)
                if plan.trusted and not plan.deleted:
                    counters["none_answers"] += 1
            elif answer:
                logger.info("Fluency: JSON answer for words %d-%d was not the span "
                            "shape; asking in prose instead", start, end)

        if plan is None:
            if use_spans:
                system, prompt = SPAN_SYSTEM_PROMPT, span_user_prompt(number_tokens(window))
            else:
                system, prompt = SYSTEM_PROMPT, user_prompt(build_prompt_text(window))
            answer = await ask(system, prompt)
            if not answer:
                # One retry: a local server hiccup or timeout on one window should
                # not silently hand that window back to structure.
                logger.warning("Fluency: no answer for words %d-%d; retrying once", start, end)
                answer = await ask(system, prompt)
            if not answer:
                logger.warning("Fluency: no answer for words %d-%d", start, end)
                counters["failed"] += 1
                continue
            counters["answered"] += 1

            # Which contract did the answer actually arrive in? Asking for spans
            # does not guarantee getting them — a model that ignores the format
            # and rewrites the transcript has still done the job, just in the
            # older shape, and reading a rewrite as "no spans found" would
            # silently mean "cut nothing here". The shape of the answer decides
            # how it is read.
            from .spans import looks_like_spans
            if use_spans and looks_like_spans(answer):
                counters["spans"] += 1
                plan, window_reasons = judge_spans(tokens[start:end], answer, start=start)
                if plan.trusted and not plan.deleted:
                    counters["none_answers"] += 1
            else:
                if use_spans:
                    logger.info("Fluency: words %d-%d came back as a rewrite, not spans; "
                                "reading it as one", start, end)
                counters["rewrites"] += 1
                plan = judge_window(tokens[start:end], answer.split(), start=start,
                                    starts=[float(w.get("start", 0.0) or 0.0) for w in window])

        for index, reason in window_reasons.items():
            if core_start <= index < core_end:
                found_reasons[index] = reason
        counters["oversized_runs"] += plan.oversized_runs
        counters["orphan_runs"] += plan.orphan_runs
        if not plan.trusted:
            logger.warning("Fluency: discarding words %d-%d (%s)", start, end, plan.reason)
            counters["untrusted"] += 1
            continue
        trusted_any = True
        # Only the core of the window is decided here; the padding either side is
        # context, and belongs to its own window. A trusted window that deleted
        # nothing writes no keeps: an echo is "no opinion", never "keep
        # everything" — reporting it as keeps un-cut every structural retake the
        # window covered whenever a too-small model echoed the transcript back.
        # A refused region (quotes matching nothing, a run past the size limit)
        # is equally not a keep: the model ruled there and the ruling was
        # unusable, so the structural decision stands.
        for index in range(core_start, core_end):
            if index in plan.deleted:
                decided[index] = True
            elif index in plan.no_opinion:
                continue
            elif plan.confirmed_keep:
                decided[index] = False
        logger.info("Fluency: words %d-%d, %d cut (%.0f%% verbatim)",
                    core_start, core_end,
                    sum(1 for i in plan.deleted if core_start <= i < core_end),
                    100 * plan.match_ratio)

    if not trusted_any:
        return None
    return decided


# --- the last read ----------------------------------------------------------

# A stumble inside a take is short — "ha aap vah", "kisee", "ye vo". A longer run
# that is not a repetition is the model rewriting the sentence, and is refused.
MAX_STUMBLE_WORDS = 4
# How far either side to look for the other copy of a repeated word.
REPEAT_WINDOW = 15


def admissible_repair_cuts(tokens: Sequence[str], proposed: Set[int],
                           max_run: int = MAX_STUMBLE_WORDS,
                           nearby: int = REPEAT_WINDOW) -> Set[int]:
    """Filter proposed cuts down to the two shapes that are repair, not rewriting.

    A pass that re-reads an already-clean edit has almost nothing left to do, and
    a model with nothing to do starts improving the writing instead. Left
    unbounded, one such pass cut "dosto aapake saath kabhee aisaa" off the front
    of the video — the greeting. So only two shapes are accepted, both of which
    are unmistakably leftover debris:

      * a SHORT run — a stumble the speaker made inside an otherwise good take;
      * a word that RECURS nearby — the remains of a phrase said twice.

    Anything else is refused outright.
    """
    accepted: Set[int] = set()
    for run in _runs(sorted(proposed)):
        # A number is data and a bare symbol is a sound the page cannot show:
        # "%" reads as punctuation but the audio says "percent", and a live
        # repair pass cut two of them out of "more than 50% 60%". A run holding
        # one is not a stumble, so it cannot take the short-run shape; the
        # repeat shape below still catches a genuinely doubled number.
        carries_data = any(_data_token(tokens[index]) for index in run
                           if 0 <= index < len(tokens))
        if len(run) <= max_run and not carries_data:
            accepted.update(run)
            continue
        for index in run:
            if not (0 <= index < len(tokens)):
                continue
            token = tokens[index]
            if not token:
                continue
            window = range(max(0, index - nearby), min(len(tokens), index + nearby + 1))
            if any(other != index and tokens[other] == token for other in window):
                accepted.add(index)
    return accepted


def _data_token(token: str) -> bool:
    """A normalised token that is data rather than speech debris: empty (a bare
    symbol the normaliser erased) or carrying a digit."""
    return not token or any(ch.isdigit() for ch in token)


FINAL_READ_SYSTEM = (
    "You are doing the last read of a video edit before it ships. The cuts are "
    "already made. What you are given is what the viewer will hear, start to "
    "finish. The speech is HINDI in its native Devanagari script, with English "
    "words mixed in. Judge it as Hindi, applying Hindi grammar, not English. A '|' "
    "marks a pause.\n\n"
    "It must play as ONE continuous, fluent take. Give the text back with only "
    "the following removed:\n"
    "  - a phrase the speaker ended up saying twice;\n"
    "  - a word or two of stumble left wedged inside a sentence;\n"
    "  - the tail of an abandoned start that the next sentence says properly.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Use ONLY the words given, spelled exactly as given, in the same order. "
    "Never translate, romanize, rephrase, re-spell or add anything.\n"
    "2. This text has already been edited. Almost all of it is correct. If you "
    "find nothing of the kinds listed above, give it back unchanged — that is the "
    "expected answer, not a failure.\n"
    "3. Never remove a whole sentence that says something on its own.\n"
    "4. Never remove a negation ('नहीं', 'ना', 'मत', 'not', 'never').\n"
    "5. Output the words only — no commentary.\n"
)


async def final_read(words: Sequence[Dict[str, Any]], ask) -> Set[int]:
    """One narrow read of the finished edit; indices into `words` to remove.

    This is the gate the whole pipeline aims at: everything upstream proposes
    cuts against a transcript still full of debris, and none of it ever sees the
    result end to end. This does, and it is allowed to remove only what
    `admissible_repair_cuts` recognises as debris, so a model that decides to
    rewrite the video cannot.
    """
    if len(words) < 20:
        return set()

    tokens = [_token(w) for w in words]
    normalised = [normalise(t) for t in tokens]
    proposed: Set[int] = set()

    for start, end, core_start, core_end in windows(len(words)):
        window = words[start:end]
        answer = await ask(FINAL_READ_SYSTEM, user_prompt(build_prompt_text(window)))
        if not answer:
            continue
        plan = judge_window(tokens[start:end], answer.split(), start=start)
        if not plan.trusted:
            logger.warning("Final read: discarding words %d-%d (%s)",
                           start, end, plan.reason)
            continue
        proposed.update(index for index in plan.deleted if core_start <= index < core_end)

    accepted = admissible_repair_cuts(normalised, proposed)

    # The very end of the video is the sign-off, and nothing follows it to
    # supersede what is removed there — this pass once deleted the speaker's
    # "बाय बाय" as debris. A cut touching the last words stands only when the
    # word recurs just before it (the remains of a doubled phrase); anything
    # else at the tail is the speaker saying goodbye, and it stays.
    tail_start = len(words) - 2
    for index in [i for i in accepted if i >= tail_start]:
        token = normalised[index]
        nearby = [other for other in
                  range(max(0, index - REPEAT_WINDOW),
                        min(len(normalised), index + REPEAT_WINDOW + 1))
                  if other != index]
        if not token or not any(normalised[other] == token for other in nearby):
            accepted.discard(index)

    if proposed:
        logger.info("Final read: %d of %d proposed removals accepted as debris",
                    len(accepted), len(proposed))
    return accepted
