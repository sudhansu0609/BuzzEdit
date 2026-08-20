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
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("fluency")

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
    return _PUNCT_RE.sub("", str(text or "")).lower()


def _token(word: Dict[str, Any]) -> str:
    return str(word.get("word", word.get("text", "")))


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
    "filler sounds. The text may be English, romanized Hindi (Hinglish), or a mix. "
    "A '|' marks a pause in the recording.\n\n"
    "Give back the transcript as it should sound in the finished video.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Use ONLY the speaker's own words, spelled exactly as given, in the same "
    "order. Never translate. Never rephrase. Never correct spelling or grammar. "
    "Never add a word that is not there.\n"
    "2. When a sentence is attempted more than once, keep ONLY the last attempt "
    "that runs to completion and delete every earlier attempt IN FULL. Never stitch "
    "parts of different attempts together — the result must be one clean take.\n"
    "3. Delete filler sounds and stutters.\n"
    "4. Keep anything the speaker meant: a list of examples, a phrase repeated for "
    "emphasis, a point restated. Repetition on purpose is not a fumble.\n"
    "5. What remains must read as complete, fluent, grammatical speech.\n"
    "6. Output the words only — no commentary, no quotes, no formatting.\n\n"
    "WORKED EXAMPLE — this is the whole job:\n"
    "Transcript:\n"
    "dosto kya ap | dosto kya | dosto kya a | dosto dosto kya apko pata hai india "
    "me ek aisi jagah hai jo bahut amazing hai\n"
    "Cleaned transcript:\n"
    "dosto kya apko pata hai india me ek aisi jagah hai jo bahut amazing hai\n\n"
    "Note what happened: the speaker tried the sentence four times and only the "
    "last one finished, so the first four attempts were deleted whole and nothing "
    "from them was reused. Most transcripts you get will need cuts like this."
)


def user_prompt(text: str) -> str:
    return f"Transcript:\n{text}\n\nCleaned transcript:"


async def plan_fluent_cuts(
    words: Sequence[Dict[str, Any]],
    ask,
    stats: Optional[Dict[str, int]] = None,
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
    windows — so the caller can report how much of the edit actually got a
    fluency opinion rather than silently falling back to structure.
    """
    if not words:
        return {}

    tokens = [_token(w) for w in words]
    decided: Dict[int, bool] = {}
    trusted_any = False
    counters = stats if stats is not None else {}
    counters.update(answered=0, failed=0, untrusted=0)

    for start, end, core_start, core_end in windows(len(words)):
        window = words[start:end]
        answer = await ask(SYSTEM_PROMPT, user_prompt(build_prompt_text(window)))
        if not answer:
            # One retry: a local server hiccup or timeout on one window should
            # not silently hand that window back to structure.
            logger.warning("Fluency: no answer for words %d-%d; retrying once", start, end)
            answer = await ask(SYSTEM_PROMPT, user_prompt(build_prompt_text(window)))
        if not answer:
            logger.warning("Fluency: no answer for words %d-%d", start, end)
            counters["failed"] += 1
            continue
        counters["answered"] += 1
        plan = judge_window(tokens[start:end], answer.split(), start=start,
                            starts=[float(w.get("start", 0.0) or 0.0) for w in window])
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
        for index in range(core_start, core_end):
            if index in plan.deleted:
                decided[index] = True
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
        if len(run) <= max_run:
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


FINAL_READ_SYSTEM = (
    "You are doing the last read of a video edit before it ships. The cuts are "
    "already made. What you are given is what the viewer will hear, start to "
    "finish. The speech is Hindi in Latin letters (Hinglish), possibly mixed with "
    "English. A '|' marks a pause.\n\n"
    "It must play as ONE continuous, fluent take. Give the text back with only "
    "the following removed:\n"
    "  - a phrase the speaker ended up saying twice;\n"
    "  - a word or two of stumble left wedged inside a sentence;\n"
    "  - the tail of an abandoned start that the next sentence says properly.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Use ONLY the words given, spelled exactly as given, in the same order. "
    "Never translate, rephrase, re-spell or add anything.\n"
    "2. This text has already been edited. Almost all of it is correct. If you "
    "find nothing of the kinds listed above, give it back unchanged — that is the "
    "expected answer, not a failure.\n"
    "3. Never remove a whole sentence that says something on its own.\n"
    "4. Never remove a negation ('naheen', 'not', 'never').\n"
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
    if proposed:
        logger.info("Final read: %d of %d proposed removals accepted as debris",
                    len(accepted), len(proposed))
    return accepted
