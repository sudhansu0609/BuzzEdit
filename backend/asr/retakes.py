"""Retake detection — collapsing "I started that sentence four times" into one take.

The pattern this exists for, taken verbatim from a real recording:

    dosto kya ap · dosto kya · dosto kya a · dosto · dosto kya apko pata hai india…

Four aborted run-ups and then the sentence that actually lands. Everything before
the last attempt has to go, and what survives must be exactly:

    dosto kya apko pata hai india…

**Why not grammar checking.** The obvious idea is to parse the transcript and drop
the ungrammatical fragments, but that fails on three counts here: there is no
dependable parser for romanised Hinglish code-switching; the text is already noisy
from the ASR, so ungrammaticality is not evidence of a fumble; and grammar cannot
tell you *which* attempt to keep, which is the actual decision. Something
structural is needed instead.

**What works.** Speech research models a repair as
`reparandum → (interregnum) → repair`: an abandoned attempt, an optional "uh",
then a corrected version that *starts the same way as the thing it replaces*. That
leaves a detectable signature — a "rough copy", an approximate repetition where a
run of words reappears shortly afterwards. It needs no grammar and no vocabulary,
so it works the same in Hindi, English or a mix of both.

Matching is fuzzy on purpose. A speaker who cuts off mid-word leaves "a" where the
good take says "apko", and the ASR romanises the same sound differently between
attempts, so exact equality would miss most real restarts.
"""

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("retakes")

_PUNCT_RE = re.compile(r"[^\w]", re.UNICODE)

# How far ahead to look for a restart. Speakers routinely get most of a sentence
# out before abandoning it — one real example ran 16 words before starting over,
# and a 10-word take plus a 20-word flounder before the retry puts the repair 31
# words out (a real recording missed a doubled sentence by exactly one word at
# 30). The abandoned-tail allowance is what stops runaway matches, not this
# window, so it can afford to be generous.
MAX_LOOKAHEAD = 45
# A restart follows its abandoned attempt closely. A longer silence means a new
# sentence, and repetition across it is rhetorical rather than a fumble.
MAX_GAP_SECONDS = 2.5
# Unless the speaker spent that silence audibly floundering. A gap containing an
# editing term — "uh", "um", a stray noise, anything already marked for cutting —
# is the interregnum of a repair, not the end of a sentence, and those run long:
# the case this is measured from is "and you … [uh] … must feel that everyone is
# watching you", 5.3s from the abandoned attempt to the retry.
MAX_INTERREGNUM_GAP_SECONDS = 6.0
# The abandoned tail: the words the speaker got out before giving up, which the
# restart does not reproduce. The allowance scales with how much *was* reproduced,
# because a long verbatim match is proof on its own — nobody repeats seven words
# by accident, however far they had got before deciding to start again.
MAX_ABANDONED_TAIL = 3
TAIL_PER_MATCHED_WORD = 2.5


@dataclass
class Retake:
    start: int          # first word of the abandoned attempt
    end: int            # exclusive; where the surviving take begins
    match_length: int   # words the repair reproduced
    confidence: float
    quality: float = 1.0   # how cleanly the repair matched (skips discount it)

    @property
    def span(self) -> int:
        return self.end - self.start


def normalise(text: str) -> str:
    return _PUNCT_RE.sub("", str(text or "")).lower()


# Romanization is not stable between takes: the ASR writes the same spoken word
# as "woh" one time and "vo" the next, "kyaa"/"kya", "dostho"/"dosto". Folding
# the common Hindi-romanization equivalences before comparing lets the matcher
# see those as the same token. Applied as a secondary check only — the folds are
# deliberately coarse, and coarse equality alone must not outrank real equality.
_PHONETIC_FOLDS = (
    ("ph", "f"), ("th", "t"), ("sh", "s"), ("chh", "c"), ("ch", "c"),
    ("gh", "g"), ("kh", "k"), ("bh", "b"), ("dh", "d"), ("jh", "j"),
    ("w", "v"), ("z", "j"), ("q", "k"),
)
_REPEAT_RE = re.compile(r"(.)\1+")


def phonetic(text: str) -> str:
    folded = normalise(text)
    # Devanagari that survived romanisation ("kaॉnvel") compares against nothing.
    # Dropping it leaves the Latin skeleton of the same word, which is what the
    # other take is written in.
    folded = "".join(c for c in folded if c.isascii())
    for cluster, single in _PHONETIC_FOLDS:
        folded = folded.replace(cluster, single)
    folded = _REPEAT_RE.sub(r"\1", folded)     # long vowels: "kyaa" -> "kya"
    return folded.rstrip("h")                  # trailing aspiration: "voh" -> "vo"


# "y" counts as a vowel here: it is a semivowel the ASR sprinkles in and out of
# romanisations ("ooniversity" / "yoonivarsitee") and keeping it wrecks the frame.
_VOWELS = "aeiouy"


def skeleton(text: str) -> str:
    """The consonants of a token, folded — its shape with the vowels taken out.

    The ASR does not romanise the same sound the same way twice, and on borrowed
    words it barely tries: one take of the recording says "cornwall ooniversity"
    and the next says "kaॉnvel yoonivarsitee". Those share almost no letters, so
    the repeat survived into the finished edit and only the language model ever
    caught it.

    Vowels are where nearly all of that variation lives; the consonant frame is
    comparatively stable. `c` is folded onto `k` here and nowhere else — it is
    the difference between "cornwall" and "kaॉnvel", but merging /k/ with /tʃ/ is
    too coarse for the ordinary phonetic check, so it stays local to this key.

    A coarse key, used only as a last resort on long tokens; on short ones it
    collides with everything.
    """
    folded = phonetic(text).replace("c", "k")
    return "".join(c for c in folded if c not in _VOWELS)


# Below this a consonant skeleton is not evidence: "kar"/"kir"/"kur" all reduce
# to "kr", and Hindi is full of them.
_MIN_SKELETON_TOKEN = 5
_MIN_SKELETON_LENGTH = 3
# A frame this long can absorb one difference and still be the same word
# ("krnvl" / "knvl"). Shorter ones must match exactly.
_FUZZY_SKELETON_LENGTH = 4


def _edit_distance(a: str, b: str, limit: int = 2) -> int:
    """Levenshtein distance, giving up once it exceeds `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def token_similarity(a: str, b: str) -> float:
    """0..1 similarity between two normalised tokens.

    A cut-off word is a *prefix* of the word it was going to be ("a" for "apko"),
    which is the single most common shape in a restart, so prefixes score high
    without being treated as identical.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 2 and len(b) >= 2 and phonetic(a) == phonetic(b):
        return 0.85
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer.startswith(shorter):
        # A one-letter stub matches far too much to trust on its own.
        return 0.9 if len(shorter) >= 2 else 0.65
    if len(longer) >= 4:
        distance = _edit_distance(a, b, limit=2)
        if distance <= 1:
            return 0.85
        if distance == 2 and len(longer) >= 6:
            return 0.7
    # Last resort, for the words the ASR spells differently in each take. Scored
    # just above the matching threshold: enough to keep a run going through one
    # mangled word, not enough to open a match on its own (see `_match_run`,
    # where skips are rationed and never start a run).
    if len(shorter) >= _MIN_SKELETON_TOKEN:
        left, right = skeleton(a), skeleton(b)
        if len(left) >= _MIN_SKELETON_LENGTH and left == right:
            return 0.8
        if (min(len(left), len(right)) >= _FUZZY_SKELETON_LENGTH
                and _edit_distance(left, right, limit=1) <= 1):
            return 0.8
    return 0.0


def _match_run(norms: List[str], start: int, repair: int, limit: int,
               threshold: float = 0.8) -> Tuple[int, int, float, bool]:
    """How far the run at `repair` reproduces the run at `start`.

    Returns (matched_words, reparandum_words_consumed, quality, truncated).

    A retry is rarely word-for-word: the speaker reformulates as they restart —
    "ek experiment kiyaa thaa" comes back as "ek saaikalaॉjikal eksaperiment
    kiyaa gayaa thaa". Strict pairwise comparison dies at the first such
    difference and reports a one-word match, so the matcher tolerates single-word
    insertions, deletions and substitutions, each accepted only when the pair
    *after* the skip lines up again. Skips are rationed and never open a match.

    Also reports whether the match ended on a *truncated* word — "prob" where the
    good take says "probably". A clipped final word is direct evidence that the
    speaker broke off mid-word, which is worth more than the match length alone.
    """
    a_off = 0            # words consumed on the abandoned side
    b_off = 0            # words consumed on the retry side
    matched = 0
    skips = 0
    total = 0.0
    truncated = False

    def sim(a_index: int, b_index: int) -> float:
        if a_index >= repair or b_index >= limit:
            return 0.0
        return token_similarity(norms[a_index], norms[b_index])

    anchor = norms[start]

    def is_anchor(index: int) -> bool:
        return index < limit and token_similarity(norms[index], anchor) >= threshold

    while start + a_off < repair and repair + b_off < limit and matched < MAX_LOOKAHEAD:
        # A recurrence of the anchor word is the start of *another attempt*.
        # A run that continues past one — matching or skipping — straddles two
        # attempts, out-scores the true repair, and leaves a stray fragment of
        # the flounder in the final cut. Stop; the walk anchors there next.
        if (a_off > 0 and is_anchor(start + a_off)) or (b_off > 0 and is_anchor(repair + b_off)):
            break
        a, b = norms[start + a_off], norms[repair + b_off]
        score = token_similarity(a, b)
        if score >= threshold:
            truncated = (a != b and len(a) < len(b) and b.startswith(a))
            total += score
            matched += 1
            a_off += 1
            b_off += 1
            continue
        if matched >= 1 and skips <= matched // 3:
            # Insertion in the retry ("ek [saaikalaॉjikal] eksaperiment ...").
            if sim(start + a_off, repair + b_off + 1) >= threshold:
                b_off += 1
                skips += 1
                continue
            # Extra word in the abandoned attempt that the retry drops.
            if sim(start + a_off + 1, repair + b_off) >= threshold:
                a_off += 1
                skips += 1
                continue
            # One word replaced ("2000 year men" -> "2000 men"): both advance.
            if sim(start + a_off + 1, repair + b_off + 1) >= threshold:
                a_off += 1
                b_off += 1
                skips += 1
                continue
        # The word the speaker abandoned on: a strict prefix of the retry.
        if a and b and a != b and b.startswith(a):
            truncated = True
        break

    quality = (total / matched if matched else 0.0)
    if skips:
        # A patched-together match is worth less than a verbatim one.
        quality *= matched / (matched + skips)
    return matched, a_off, quality, truncated


def find_retakes(
    words: List[Dict[str, Any]],
    max_lookahead: int = MAX_LOOKAHEAD,
    max_gap_seconds: float = MAX_GAP_SECONDS,
) -> List[Retake]:
    """Locate abandoned attempts, keeping the last complete one.

    Walks left to right and, at each position, asks "does this phrase start again
    shortly?". The best restart wins and everything before it is dropped, then the
    scan resumes from the surviving take — which is what lets a pile-up of four
    successive false starts collapse in one pass.

    The walk is over the words that will *survive*: anything already marked for
    cutting is invisible here. A repair is reparandum → interregnum → repair, and
    the interregnum is made of exactly that material — "you must feel …" against
    "you [uh] must feel …" is one continuous match with the editing term taken
    out, and a two-word mismatch with it left in.
    """
    live = [i for i, w in enumerate(words)
            if normalise(w.get("word", w.get("text", ""))) and not w.get("disfluency")]
    if len(live) < 2:
        return []

    norms = [normalise(words[i].get("word", words[i].get("text", ""))) for i in live]
    starts = [float(words[i].get("start", 0.0) or 0.0) for i in live]
    ends = [float(words[i].get("end", 0.0) or 0.0) for i in live]
    # True where an editing term was lifted out of the gap before this word.
    bridged = [False] + [live[j] - live[j - 1] > 1 for j in range(1, len(live))]
    count = len(live)
    original = lambda position: live[position] if position < count else len(words)

    found: List[Retake] = []
    index = 0
    while index < count:
        if not norms[index]:
            index += 1
            continue

        best: Optional[Retake] = None
        worst_clean_gap = 0.0
        for repair in range(index + 1, min(count, index + max_lookahead + 1)):
            # A silence between the two attempts. Short is delivery; long used to
            # mean "moved on" and end the scan — but a speaker who flubs a take
            # on camera stops, composes themselves for several seconds, and says
            # it again. That silence is an edit point, not a sentence boundary,
            # so the scan continues through it and the *evidence bar* goes up
            # instead: a match found across a long clean silence has to be long
            # enough to be proof (see below). Gaps whose contents were already
            # marked for cutting were never boundaries at all.
            gap = starts[repair] - ends[repair - 1]
            if gap > MAX_INTERREGNUM_GAP_SECONDS:
                break
            if not bridged[repair]:
                worst_clean_gap = max(worst_clean_gap, gap)
            if not norms[repair]:
                continue
            if token_similarity(norms[index], norms[repair]) < 0.8:
                continue

            length, consumed, quality, truncated = _match_run(norms, index, repair, count)
            if length < 1:
                continue

            # Across a long clean silence, only a substantial reproduction is
            # evidence of a retake; a short echo after a real pause is far more
            # likely two genuine sentences.
            if worst_clean_gap > max_gap_seconds and length < 4:
                continue

            abandoned = (repair - index) - consumed
            allowed_tail = MAX_ABANDONED_TAIL + TAIL_PER_MATCHED_WORD * max(0, length - 1)
            if abandoned > allowed_tail:
                continue          # too much unrepeated material to be a restart

            # A one-word match only means anything back-to-back — that is a
            # stutter. Spaced apart it is just a common word recurring, and Hindi
            # sentences are full of "hai … hai": treating those as restarts would
            # eat the words between them.
            if length == 1 and repair != index + 1:
                continue

            # Seeing through an editing term is what lets a real restart be
            # recognised, but it also puts two ordinary sentences within reach of
            # each other. "they didn't notice it · uh · very few people noticed
            # it" is a two-word rough copy by every structural measure and both
            # halves are things the speaker meant. A match that had to bridge a
            # filler therefore has to be long enough to be proof on its own.
            if any(bridged[k] for k in range(index + 1, repair + 1)) and length < 3:
                continue

            # Confidence rises with how much of the attempt was repeated and how
            # cleanly it matched. Deliberately calibrated so a short match is *not*
            # enough on its own — two reasons, both from real recordings:
            # "it is what it is" is a rough copy by every structural measure, and a
            # three-word match on a phrase as ordinary as "unakaa naam thaa"
            # ("his name was") once cut the speaker's own subject out of the video.
            # Below the cut threshold these become LLM candidates rather than edits.
            confidence = (0.28 + 0.13 * length) * quality
            if length == 1:
                confidence *= 0.6

            # A word clipped mid-syllable is not something a fluent speaker does.
            if truncated:
                confidence += 0.15

            # Nothing left over: the speaker said the phrase, stopped dead, and
            # said exactly the same phrase again. That is a restart. It is the
            # leftover tail that makes a match ambiguous — "unakaa naam thaa
            # *thommel gilovich* | unakaa naam thaa …" might be two real clauses.
            if abandoned == 0 and length >= 2:
                confidence += 0.12

            # A multi-second silence before the retry is the on-camera
            # recompose. Combined with the length bar above, this is the
            # signature of a deliberate second take.
            if worst_clean_gap > max_gap_seconds and length >= 4:
                confidence += 0.1

            # A speaker who has to restart usually pauses first. Fluent delivery
            # through the "repeat" points at an idiom or a rhetorical echo.
            if starts[repair] - ends[repair - 1] >= 0.15:
                confidence += 0.15

            # The further away the restart, the likelier the "repeat" is just the
            # speaker coming back to a phrase. Long matches shrug this off; short
            # ones at a distance should not be cut on structure alone.
            if abandoned > MAX_ABANDONED_TAIL:
                confidence -= 0.05 * (abandoned - MAX_ABANDONED_TAIL) / max(1, length)

            candidate = Retake(index, repair, length,
                               round(min(1.0, confidence), 3), round(quality, 3))

            # Prefer the strongest *evidence*, not the longest run: a
            # skip-patched match can out-length a verbatim one by wandering
            # across attempt boundaries, and resuming from the wrong repair
            # leaves a stray word of the flounder in the final cut. Weighing
            # length by match quality keeps the clean repair ahead of the
            # tangled one (3×0.97 beats 4×0.67 on the recording this is from).
            def evidence(r: Retake) -> tuple:
                return (round(r.match_length * r.quality, 3), r.match_length, -r.start)
            if best is None or evidence(candidate) > evidence(best):
                best = candidate

        if best is not None:
            # Back to positions in the caller's list. The span swallows any
            # editing terms it bridged, which are being cut anyway.
            found.append(Retake(original(best.start), original(best.end),
                                best.match_length, best.confidence, best.quality))
            index = best.end        # resume from the surviving take
        else:
            index += 1

    _boost_chains(found)
    if found:
        logger.info("Retakes: %d abandoned attempts (%d words)",
                    len(found), sum(r.span for r in found))
    return found


def _boost_chains(found: List[Retake], bonus: float = 0.25) -> None:
    """Raise confidence for back-to-back restarts.

    This is the signal that separates floundering from ordinary repetition. A
    speaker fighting for a sentence tries it two, three, four times in a row —
    "dosto kya ap · dosto kya · dosto kya a · dosto · dosto kya apko…". A set
    phrase such as "it is what it is" happens once and moves on. Chained restarts
    are therefore trustworthy enough to cut without asking anyone.
    """
    run_start = 0
    for index in range(1, len(found) + 1):
        chained = (index < len(found) and found[index].start == found[index - 1].end)
        if chained:
            continue
        if index - run_start >= 2:
            for retake in found[run_start:index]:
                retake.confidence = round(min(1.0, retake.confidence + bonus), 3)
        run_start = index


def apply_retakes(
    words: List[Dict[str, Any]],
    cut_confidence: float = 0.72,
) -> List[Dict[str, Any]]:
    """Mark abandoned attempts on the word dicts.

    Confident retakes are cut outright; weaker ones become candidates for the LLM
    to judge in context, because the one thing this must not do is delete a phrase
    the speaker repeated deliberately.
    """
    for retake in find_retakes(words):
        for i in range(retake.start, retake.end):
            word = words[i]
            if word.get("disfluency"):
                continue
            if retake.confidence >= cut_confidence:
                word["disfluency"] = True
                word["reason"] = "retake"
            else:
                word["candidate"] = True
                if not word.get("reason"):
                    word["reason"] = "false_start"
            word["retake_confidence"] = retake.confidence
    return words


# --- which attempt to keep --------------------------------------------------
#
# Everything above assumes the last attempt is the good one, and almost always it
# is: a speaker restarts because the previous try went wrong. But not always. On
# the reference recording the *final* reading of the opening is itself garbled —
# "kabhee aisaa ha aap vah hai ki" — while an earlier attempt says the line
# cleanly. Structure cannot see that; only something that can read the sentence
# can, so where a genuine alternative reading exists the model is asked which one
# to keep.
#
# The danger is obvious: a wrong answer here deletes the good take. So this only
# ever runs on a *like-for-like* pair — an abandoned attempt long enough to be a
# whole reading, similar enough to be the same sentence, and no shorter than the
# take it would replace. A stub like "dosto kya ap" can never qualify, so the
# ordinary pile-up is untouched and the last take stands.

# An attempt shorter than this is a run-up, not a reading.
MIN_TAKE_WORDS = 4
# Below this the two are not the same sentence and swapping them changes content.
MIN_TAKE_SIMILARITY = 0.6
# The challenger must be at least this fraction of the surviving take's length,
# or it is a truncated version of it and swapping loses the ending.
MIN_TAKE_COVERAGE = 0.8


TAKE_SYSTEM = (
    "A speaker recording to camera said the same sentence more than once, "
    "restarting until they were happy. You are choosing which recording of it to "
    "put in the video. The text is Hindi written in Latin letters (Hinglish), "
    "possibly mixed with English.\n\n"
    "For each numbered group, pick the ONE take that is the most complete and "
    "fluent — the one a viewer should hear.\n\n"
    "How to judge:\n"
    "- Prefer the take that finishes its thought over one that breaks off.\n"
    "- Prefer the take without stumble words wedged into the middle of it.\n"
    "- **Spelling is never a reason to reject a take.** A machine wrote this text "
    "down and romanises Hindi badly ('jaz' = judge, 'sabsakraaib' = subscribe). "
    "Two takes spelled differently may be identical speech.\n"
    "- If the takes are equally good, choose the LAST one. It is the one the "
    "speaker settled on.\n\n"
    "Answer with one line per group and nothing else:\n"
    "<group number>: <take number>"
)

_TAKE_RE = re.compile(r"^\s*(\d+)\s*[:.\)-]\s*(?:take\s*)?(\d+)", re.IGNORECASE)


def parse_take_choices(answer: str, groups: int) -> Dict[int, int]:
    """Parse "<group>: <take>" lines into {group index: take index}, both 0-based."""
    choices: Dict[int, int] = {}
    for line in (answer or "").splitlines():
        match = _TAKE_RE.match(line)
        if not match:
            continue
        group = int(match.group(1)) - 1
        take = int(match.group(2)) - 1
        if 0 <= group < groups and take >= 0:
            choices[group] = take
    return choices


def _cut_runs(words: Sequence[Dict[str, Any]]) -> List[List[int]]:
    """Contiguous stretches of words cut as abandoned attempts."""
    runs: List[List[int]] = []
    for index, word in enumerate(words):
        if word.get("disfluency") and word.get("reason") == "retake":
            if runs and index == runs[-1][-1] + 1:
                runs[-1].append(index)
            else:
                runs.append([index])
    return runs


def _surviving_take(words: Sequence[Dict[str, Any]], after: int,
                    limit: int, max_gap: float = MAX_GAP_SECONDS) -> List[int]:
    """Indices of the take that survived, starting just after an abandoned run."""
    take: List[int] = []
    previous_end: Optional[float] = None
    for index in range(after, len(words)):
        word = words[index]
        if word.get("disfluency"):
            continue
        if not normalise(word.get("word", word.get("text", ""))):
            continue
        start = float(word.get("start", 0.0) or 0.0)
        if previous_end is not None and start - previous_end > max_gap:
            break               # a new thought, not more of this sentence
        take.append(index)
        previous_end = float(word.get("end", start) or start)
        if len(take) >= limit:
            break
    return take


def _text(words: Sequence[Dict[str, Any]], indices: Sequence[int]) -> str:
    return " ".join(str(words[i].get("word", words[i].get("text", ""))).strip()
                    for i in indices)


def _challengers(words: Sequence[Dict[str, Any]], run: List[int],
                 take: List[int]) -> List[List[int]]:
    """Earlier attempts in `run` that are genuine alternative readings of `take`.

    An attempt starts where the take's own opening word recurs, which is what the
    speaker did: they went back to the top of the sentence. Anything that is not
    a full-length, closely-matching reading is filtered out here, which is what
    keeps the ordinary run-up pile-up away from this whole mechanism.
    """
    if len(take) < MIN_TAKE_WORDS:
        return []
    tokens = {index: normalise(words[index].get("word", words[index].get("text", "")))
              for index in run + take}
    opener = tokens[take[0]]
    if not opener:
        return []

    found: List[List[int]] = []
    for position, index in enumerate(run):
        if token_similarity(tokens[index], opener) < 0.8:
            continue
        attempt = [i for i in run[position:] if tokens[i]]
        if len(attempt) < MIN_TAKE_WORDS:
            continue
        if len(attempt) < MIN_TAKE_COVERAGE * len(take):
            continue        # a truncated version of the take; swapping loses its end
        similarity = difflib.SequenceMatcher(
            a=[tokens[i] for i in attempt], b=[tokens[i] for i in take],
            autojunk=False).ratio()
        if similarity < MIN_TAKE_SIMILARITY:
            continue
        found.append(attempt)
    return found


async def choose_best_takes(words: List[Dict[str, Any]], ask) -> int:
    """Reconsider "keep the last take" where a real alternative reading exists.

    Runs after `apply_retakes` has cut. Returns how many swaps were made; the
    words are edited in place. Any failure — a silent model, an unparseable
    answer, a choice outside the offered takes — leaves the last take in place,
    which is the behaviour this refines rather than replaces.
    """
    groups: List[Tuple[List[List[int]], List[int]]] = []
    for run in _cut_runs(words):
        take = _surviving_take(words, run[-1] + 1, limit=len(run) + MIN_TAKE_WORDS)
        challengers = _challengers(words, run, take)
        if challengers:
            groups.append((challengers, take))
    if not groups:
        return 0

    prompt_parts: List[str] = []
    for number, (challengers, take) in enumerate(groups, 1):
        lines = [f"{number}:"]
        for take_number, attempt in enumerate(challengers, 1):
            lines.append(f"  take {take_number}: {_text(words, attempt)}")
        lines.append(f"  take {len(challengers) + 1}: {_text(words, take)}")
        prompt_parts.append("\n".join(lines))

    answer = await ask(TAKE_SYSTEM, "Groups:\n" + "\n\n".join(prompt_parts))
    if not answer:
        logger.info("Best take: no answer from the model; keeping the last take")
        return 0

    swaps = 0
    for group, choice in parse_take_choices(answer, len(groups)).items():
        challengers, take = groups[group]
        if choice >= len(challengers):
            continue            # chose the surviving take, which is already the plan
        winner = challengers[choice]
        for index in winner:
            words[index]["disfluency"] = False
            words[index]["candidate"] = False
            words[index]["reason"] = None
        for index in take:
            words[index]["disfluency"] = True
            words[index]["candidate"] = False
            words[index]["reason"] = "retake"
        swaps += 1
        logger.info("Best take: keeping the earlier reading %r over %r",
                    _text(words, winner), _text(words, take))
    return swaps
