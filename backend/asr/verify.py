"""Grammar verification and repair of the finished edit.

Everything before this proposes cuts. Nothing before this ever checked the
result. The gap was found by asking a plain question — "how do you know what is
left is a sentence?" — and the answer was that we did not: the pipeline measured
repeated n-grams, cut words still audible, segment lengths and A/V sync, none of
which can tell a sentence from a fragment. Audited for the first time, 8 of 15
sentences in a "clean" edit came back broken.

So the edit is now read back sentence by sentence:

  1. **Audit** — is each surviving sentence complete Hindi/Hinglish as spoken?
  2. **Repair** — for the broken ones, remove the leftover debris and no more.
  3. **Re-audit** — score what actually shipped, and report it.

**The one distinction that matters.** Two very different things make a sentence
look wrong in this transcript:

  * *debris* — a stumble or half-word left behind by the edit, "kabhee aisaa
    [ha aap vah] hai ki". The audio contains a fumble; cutting is correct.
  * *transliteration noise* — the ASR wrote "jaz" for *judge*, "sabsakraaib" for
    *subscribe*, "phinamaanaa" for *phenomenon*. The audio is perfect and only
    the spelling is wrong. Cutting here deletes good speech.

Roughly half of a real audit's complaints are the second kind, so the prompts say
so in as many words, and repair is delete-only and aligned — a model that ignores
the instruction and "corrects" spelling changes nothing, because a reworded span
keeps the original words.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fluency import align_deletions, normalise

logger = logging.getLogger("verify")

# A sentence ends on terminal punctuation, or on a long pause — the ASR punctuates
# unreliably, and a breath this long is a sentence boundary in speech whatever the
# text says.
_SENTENCE_END = re.compile(r"[.!?]$")
# Only a *long* silence is a sentence boundary on its own. Hindi speakers pause
# mid-clause constantly, and at 1.0s the splitter was cutting sentences in half
# and then reporting both halves as fragments — complaints like 'ends with "ki"'
# were the split, not the edit.
SENTENCE_PAUSE_SECONDS = 2.5
# Run-ons make the model judge several thoughts as one; split them for judging.
MAX_SENTENCE_WORDS = 40
# Below this a "sentence" is an interjection ("Bye.", "haan"), which is complete
# speech and must not be reported as a fragment.
MIN_JUDGED_WORDS = 4

# Repair limits. A repair removes debris; anything more is a rewrite.
MAX_REPAIR_FRACTION = 0.5
MIN_WORDS_AFTER_REPAIR = 3
MIN_REPAIR_MATCH_RATIO = 0.5
# The debris a repair exists for — a stumble, a half-word, the wedged remains of
# a false start — is short. A single deletion longer than this is the model
# re-cutting the sentence: a live repair removed eight consecutive words that
# were the researchers' names ("था थौमल गिलगोविच केनट सविट्सकी और विक्टोरिया
# मेडवेक"), which no grammatical judgement justifies. The percentage cap alone
# cannot catch this — a long sentence affords a long "repair".
MAX_REPAIR_RUN = 5

# Words a repair may NEVER remove. Deleting a negation does not tidy a sentence,
# it reverses it: the first live run of this pass "repaired" "hamane to notice
# naheen kiyaa" (we did NOT notice) into the opposite of what the speaker said.
# Nothing about a grammatical judgement justifies that, so the words are simply
# off limits — Hindi first, then the English a Hinglish speaker mixes in.
PROTECTED_WORDS = {
    # Hindi / Urdu negation and prohibition — native Devanagari, since the audit
    # now judges the native script (normalise keeps the matras, so these match).
    "नहीं", "नही", "नहि", "ना", "न", "मत", "बिना", "कभी", "कभ", "कोई", "कुछ",
    "केवल", "सिर्फ", "बिल्कुल",
    # The same words romanized, for English speech and any word without a native
    # form (normalise strips the matra off some, so keep the stripped shapes too).
    "naheen", "nahin", "nahi", "nah", "na", "mat", "bina", "binaa", "kabhee",
    "kabhi", "koee", "koi", "kuch", "kuchh",
    # English negation and quantity, which invert meaning just as hard
    "no", "not", "never", "none", "nothing", "dont", "doesnt", "didnt", "cant",
    "cannot", "wont", "isnt", "arent", "wasnt", "werent", "without", "neither",
    "nor", "only", "all", "every",
}


@dataclass
class Sentence:
    """One sentence of the edit, as word positions in the surviving list.

    `hard_boundary` means it ended on real terminal punctuation rather than on a
    pause or the length cap — i.e. the speaker actually finished there.
    """
    index: int
    positions: List[int]
    text: str
    hard_boundary: bool = False


@dataclass
class Issue:
    sentence: int
    text: str
    problem: str
    repaired: bool = False
    removed: List[str] = field(default_factory=list)


@dataclass
class AuditResult:
    total: int = 0
    broken: int = 0
    issues: List[Issue] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sentences": self.total,
            "broken": self.broken,
            "issues": [
                {"text": i.text, "problem": i.problem,
                 "repaired": i.repaired, "removed": i.removed}
                for i in self.issues
            ],
        }


def _token(word: Dict[str, Any]) -> str:
    """The token to JUDGE — the native Devanagari when the word carries one, so
    the audit reads real Hindi rather than its romanization. See fluency._token."""
    return str(word.get("word_native") or word.get("word") or word.get("text", ""))


def split_sentences(words: Sequence[Dict[str, Any]]) -> List[Sentence]:
    """Group surviving words into sentences by punctuation and by pause."""
    sentences: List[Sentence] = []
    current: List[int] = []
    previous_end: Optional[float] = None

    def flush(hard: bool = False) -> None:
        if current:
            sentences.append(Sentence(len(sentences), list(current),
                                      " ".join(_token(words[p]) for p in current),
                                      hard))
            current.clear()

    for position, word in enumerate(words):
        start = float(word.get("start", 0.0) or 0.0)
        if previous_end is not None and start - previous_end >= SENTENCE_PAUSE_SECONDS:
            flush()
        current.append(position)
        previous_end = float(word.get("end", start) or start)
        if _SENTENCE_END.search(_token(word).strip()):
            flush(hard=True)
        elif len(current) >= MAX_SENTENCE_WORDS:
            flush()
    flush()
    return sentences


AUDIT_SYSTEM = (
    "You are proof-reading the transcript of a finished video edit. The speaker "
    "talks in HINDI, written in its native Devanagari script, mixing in English "
    "words as Hindi speakers do.\n\n"
    "For each numbered sentence, decide ONE thing: read aloud, is it a complete, "
    "grammatical HINDI sentence? Apply HINDI grammar and Hindi sentence sense — "
    "not English. A sentence that is perfectly good spoken Hindi is OK even if a "
    "word-for-word English reading would sound odd.\n\n"
    "Judge it as SPEECH, not as writing:\n"
    "- Mixing English words into Hindi is normal and correct. Not a fault.\n"
    "- Missing punctuation is not a fault.\n"
    "- A line ending in '...' was cut off by a pause, not by the speaker: the "
    "thought carries on in the next line. Judge only whether the words present "
    "are good speech, and never call such a line unfinished.\n\n"
    "A sentence is BROKEN only when the words themselves do not form a complete "
    "Hindi thought — it breaks off unfinished, or it carries leftover stumble "
    "words from a false start, such as 'कभी ऐसा ह आप वह है कि' where 'ह आप वह' is "
    "debris left behind by the edit.\n\n"
    "Answer with one line per sentence and nothing else:\n"
    "<number>: OK\n"
    "<number>: BROKEN - <a few words on what is wrong>"
)


REPAIR_SYSTEM = (
    "You are repairing sentences from a video edit. The speech is HINDI in its "
    "native Devanagari script, with English words mixed in. Each sentence has "
    "leftover stumble words in it — debris from a false start the speaker made.\n\n"
    "For each numbered sentence, write it again with ONLY the debris words "
    "removed, so what is left is a complete, grammatical HINDI sentence.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Use only the words given, spelled exactly as given, in the same order. "
    "Delete words; never add, replace, reorder, re-spell, romanize or translate "
    "anything.\n"
    "2. Judge completeness by HINDI grammar, not English.\n"
    "3. Remove as little as possible — only what stops the sentence being a "
    "complete Hindi sentence.\n"
    "4. If nothing can be removed to fix it, write the sentence back unchanged.\n\n"
    "Answer with one line per sentence and nothing else:\n"
    "<number>: <the repaired sentence>"
)


def _numbered(sentences: Sequence[Sentence]) -> str:
    """Number the sentences 1..N, marking the ones that are still running.

    Numbered by position in the list given, NOT by `Sentence.index`: callers
    filter the split (short sentences out, broken ones only), and numbering the
    survivors by their unfiltered indices showed the model "1, 3, 7…" — it then
    answers about numbers that were never asked, and the answers for real ones
    get dropped. The caller maps ordinals back to `Sentence.index` itself.

    A line the splitter ended at a pause is not necessarily a finished thought —
    the speaker carries on in the next one. Without saying so, every one of them
    came back flagged "ends unfinished", which is a complaint about the splitter
    rather than about the edit.
    """
    return "\n".join(f"{position + 1}: {s.text}" + ("" if s.hard_boundary else " ...")
                     for position, s in enumerate(sentences))


_VERDICT_RE = re.compile(r"^\s*(\d+)\s*[:.\)-]\s*(.*)$")


def parse_verdicts(answer: str, count: int) -> Dict[int, Tuple[bool, str]]:
    """Parse "<n>: OK" / "<n>: BROKEN - reason" lines into {index: (ok, problem)}.

    Anything unparseable is simply absent, and an absent sentence counts as fine:
    the model failing to answer is not evidence against the speaker.
    """
    verdicts: Dict[int, Tuple[bool, str]] = {}
    for line in (answer or "").splitlines():
        match = _VERDICT_RE.match(line)
        if not match:
            continue
        number = int(match.group(1)) - 1
        if not 0 <= number < count:
            continue
        body = match.group(2).strip()
        ok = not re.match(r"(?i)^broken\b", body)
        problem = "" if ok else re.sub(r"(?i)^broken\s*[-–:]?\s*", "", body).strip()
        verdicts[number] = (ok, problem)
    return verdicts


def parse_repairs(answer: str, count: int) -> Dict[int, str]:
    repairs: Dict[int, str] = {}
    for line in (answer or "").splitlines():
        match = _VERDICT_RE.match(line)
        if not match:
            continue
        number = int(match.group(1)) - 1
        if 0 <= number < count and match.group(2).strip():
            repairs[number] = match.group(2).strip()
    return repairs


def repair_deletions(sentence: Sentence, repaired: str,
                     tokens: Sequence[str]) -> Optional[List[int]]:
    """Positions to cut so the sentence becomes the repaired one, or None.

    Delete-only and aligned, exactly as the fluency pass: the model can shorten
    the sentence and nothing else. A repair that rewrites, translates or guts the
    sentence fails the limits below and is discarded.
    """
    original = [tokens[p] for p in sentence.positions]
    cleaned = repaired.split()
    if not cleaned:
        return None

    deleted, ratio = align_deletions(original, cleaned)
    if not deleted:
        return None

    # Numbers and bare symbols are data, not debris, and the model cannot have
    # reasoned about the sound under them: "%" reads as punctuation on the page
    # but the audio says "percent" — a live repair deleted two of them and the
    # cut dropped the spoken word both times. They are lifted OUT of the
    # deletion set rather than refusing the whole repair, because unlike a
    # negation their removal cannot invert the sentence the model approved —
    # keeping them leaves that sentence exactly as it read.
    deleted = {offset for offset in deleted
               if normalise(original[offset])
               and not any(ch.isdigit() for ch in original[offset])}
    if not deleted:
        return None

    protected = sorted(normalise(original[offset]) for offset in deleted
                       if normalise(original[offset]) in PROTECTED_WORDS)
    if protected:
        # Refuse the whole repair rather than part of it: a repair is one
        # judgement about one sentence, and half of it is not a safer version.
        logger.info("Repair for sentence %d refused: it would remove %s",
                    sentence.index + 1, ", ".join(repr(w) for w in protected))
        return None
    if ratio < MIN_REPAIR_MATCH_RATIO:
        logger.info("Repair for sentence %d ignored: only %.0f%% verbatim",
                    sentence.index + 1, 100 * ratio)
        return None
    if len(deleted) / len(original) > MAX_REPAIR_FRACTION:
        logger.info("Repair for sentence %d ignored: would remove %d of %d words",
                    sentence.index + 1, len(deleted), len(original))
        return None
    longest = 0
    run = 0
    for offset in range(len(original)):
        run = run + 1 if offset in deleted else 0
        longest = max(longest, run)
    if longest > MAX_REPAIR_RUN:
        logger.info("Repair for sentence %d refused: a single %d-word removal "
                    "is a rewrite, not a repair", sentence.index + 1, longest)
        return None
    if len(original) - len(deleted) < MIN_WORDS_AFTER_REPAIR:
        logger.info("Repair for sentence %d ignored: too little left", sentence.index + 1)
        return None

    # A sentence that was cut off by a pause rather than by punctuation *looks*
    # unfinished at its tail whether or not it is: the thought carries on in the
    # next one. Trimming that tail deletes a real clause — the live run removed
    # "ho sakataa hai ki vah sab", whose continuation was sitting in the very
    # next sentence. Debris in the MIDDLE is still fair game; only the tail is
    # protected, and only when the speaker did not actually stop there.
    if not sentence.hard_boundary and (len(original) - 1) in deleted:
        logger.info("Repair for sentence %d refused: it trims the tail of a "
                    "sentence that a pause split, not the speaker", sentence.index + 1)
        return None

    return [sentence.positions[offset] for offset in sorted(deleted)]


async def audit(words: Sequence[Dict[str, Any]], ask) -> AuditResult:
    """Read the edit back and report which sentences are not sentences."""
    sentences = [s for s in split_sentences(words) if len(s.positions) >= MIN_JUDGED_WORDS]
    result = AuditResult(total=len(sentences))
    if not sentences:
        return result

    answer = await ask(AUDIT_SYSTEM, "Sentences:\n" + _numbered(sentences))
    if not answer:
        logger.warning("Audit: no answer from the model; edit not verified")
        result.total = 0
        return result

    verdicts = parse_verdicts(answer, len(sentences))
    for position, sentence in enumerate(sentences):
        ok, problem = verdicts.get(position, (True, ""))
        if not ok:
            result.broken += 1
            result.issues.append(Issue(sentence.index, sentence.text, problem))
    return result


async def repair_broken(words: List[Dict[str, Any]], result: AuditResult,
                        ask) -> List[int]:
    """Ask for repairs to the broken sentences and return the positions to cut.

    Mutates `result`'s issues to record what was removed, so the report can say
    which sentence was mended and with which words.
    """
    sentences = {s.index: s for s in split_sentences(words)}
    broken = [sentences[i.sentence] for i in result.issues if i.sentence in sentences]
    if not broken:
        return []

    answer = await ask(REPAIR_SYSTEM, "Sentences:\n" + _numbered(broken))
    # Ordinals in the prompt back to real sentence indices.
    repairs = {broken[position].index: text
               for position, text in parse_repairs(answer or "", len(broken)).items()}

    tokens = [_token(w) for w in words]
    cuts: List[int] = []
    for issue in result.issues:
        sentence = sentences.get(issue.sentence)
        repaired = repairs.get(issue.sentence)
        if sentence is None or not repaired:
            continue
        positions = repair_deletions(sentence, repaired, tokens)
        if not positions:
            continue
        issue.repaired = True
        issue.removed = [tokens[p] for p in positions]
        cuts.extend(positions)
        logger.info("Repaired sentence %d by removing: %s",
                    issue.sentence + 1, " ".join(issue.removed))

    return cuts


async def verify_and_repair(words: List[Dict[str, Any]], ask) -> Tuple[AuditResult, List[int]]:
    """Audit the surviving edit, repair what it can, and report the result.

    Returns (audit after repair, positions in `words` to cut). Positions are
    indices into the list handed in, which is the *surviving* words.
    """
    first = await audit(words, ask)
    if not first.total or not first.issues:
        return first, []

    logger.info("Audit: %d of %d sentences broken", first.broken, first.total)
    return first, await repair_broken(words, first, ask)


# --- incomplete sentences ---------------------------------------------------
#
# Repair removes debris from a sentence that is otherwise sound. It cannot help
# the other shape of damage: a sentence that is not damaged but *unfinished* —
# the speaker began a thought, abandoned it, and started over, and the abandoned
# start survived because it was too different from the retry for the retake
# matcher to pair them. There is nothing inside it to delete; the whole fragment
# has to go or nothing does.
#
# That is a far more dangerous cut than a repair, so it is fenced in hard: only
# short sentences, only ones the model says are superseded by what follows, never
# one holding a negation, and never more than a small share of the edit.

# Beyond this a "fragment" is a thought with content in it, and dropping it whole
# would take something the speaker said.
MAX_FRAGMENT_WORDS = 12
# A ceiling on the total share of the edit that whole-fragment drops may remove.
# If the model wants more than this, it is re-cutting the video, not tidying it.
MAX_FRAGMENT_FRACTION = 0.15


FRAGMENT_SYSTEM = (
    "You are checking a video edit for abandoned sentences. The speech is HINDI in "
    "its native Devanagari script, with English words mixed in; judge it as Hindi. "
    "The speaker sometimes starts a thought, gives up, and starts again in "
    "different words. The abandoned start is still in the edit and has to be "
    "removed whole.\n\n"
    "You are shown numbered pairs. For each pair:\n"
    "  A = a short line that may be an abandoned start.\n"
    "  B = the line that comes after it in the video.\n\n"
    "Answer DROP only when BOTH are true:\n"
    "  1. A is incomplete on its own — it breaks off and never finishes its thought.\n"
    "  2. B says the same thing properly, so nothing is lost by removing A.\n\n"
    "Answer KEEP for everything else. In particular answer KEEP when:\n"
    "  - A is a complete Hindi thought, even a very short one ('हाँ', 'तो दोस्तों').\n"
    "  - A says something B does not say.\n\n"
    "When you are unsure, answer KEEP.\n\n"
    "Answer with one line per pair and nothing else:\n"
    "<number>: DROP\n"
    "<number>: KEEP"
)


def parse_drops(answer: str, count: int) -> set:
    """Numbers the model answered DROP for. Anything else counts as KEEP."""
    drops = set()
    for line in (answer or "").splitlines():
        match = _VERDICT_RE.match(line)
        if not match:
            continue
        number = int(match.group(1)) - 1
        if 0 <= number < count and re.match(r"(?i)^drop\b", match.group(2).strip()):
            drops.add(number)
    return drops


def _fragment_candidates(words: Sequence[Dict[str, Any]], result: AuditResult,
                         sentences: Dict[int, Sentence]) -> List[Sentence]:
    """Broken sentences that repair could not save and that are safe to consider."""
    candidates: List[Sentence] = []
    for issue in result.issues:
        if issue.repaired:
            continue                     # mending it beats deleting it
        sentence = sentences.get(issue.sentence)
        if sentence is None or not sentence.positions:
            continue
        if len(sentence.positions) > MAX_FRAGMENT_WORDS:
            continue
        tokens = [normalise(_token(words[p])) for p in sentence.positions]
        if any(t in PROTECTED_WORDS for t in tokens):
            # Same rule as repair: removing a negation reverses the speaker
            # rather than tidying them, and no grammatical judgement is worth
            # that. See PROTECTED_WORDS.
            continue
        if any(any(ch.isdigit() for ch in _token(words[p]))
               for p in sentence.positions):
            # A number is data. A "fragment" carrying one is a thought with
            # content in it, whatever its grammar looks like.
            continue
        candidates.append(sentence)
    return candidates


async def find_droppable_fragments(words: Sequence[Dict[str, Any]], result: AuditResult,
                                   ask) -> List[int]:
    """Positions of whole abandoned sentences the model agrees are superseded."""
    sentences = {s.index: s for s in split_sentences(words)}
    candidates = _fragment_candidates(words, result, sentences)
    if not candidates or len(sentences) < 2:
        return []

    pairs: List[Tuple[Sentence, str]] = []
    for sentence in candidates:
        following = sentences.get(sentence.index + 1)
        if following is None:
            continue        # nothing supersedes the last line; keep it
        pairs.append((sentence, following.text))
    if not pairs:
        return []

    prompt = "\n\n".join(
        f"{number + 1}:\nA: {sentence.text}\nB: {after}"
        for number, (sentence, after) in enumerate(pairs)
    )
    answer = await ask(FRAGMENT_SYSTEM, "Pairs:\n" + prompt)
    if not answer:
        return []
    drops = parse_drops(answer, len(pairs))
    if not drops:
        return []

    # A share of the edit, but never less than one fragment's worth: as a pure
    # fraction the budget was under a single sentence on anything shorter than a
    # few hundred words, so short videos could never have a fragment dropped at
    # all while long ones could.
    budget = max(MAX_FRAGMENT_WORDS, int(len(words) * MAX_FRAGMENT_FRACTION))
    positions: List[int] = []
    for number in sorted(drops):
        sentence = pairs[number][0]
        if len(positions) + len(sentence.positions) > budget:
            logger.info("Fragment drop refused for sentence %d: over the %d-word "
                        "budget for whole-sentence removals",
                        sentence.index + 1, budget)
            continue
        positions.extend(sentence.positions)
        for issue in result.issues:
            if issue.sentence == sentence.index:
                issue.repaired = True
                issue.removed = [_token(words[p]) for p in sentence.positions]
        logger.info("Dropped abandoned sentence %d whole: %s",
                    sentence.index + 1, sentence.text)

    # Never leave the edit with nothing in it.
    if len(words) - len(positions) < MIN_WORDS_AFTER_REPAIR:
        logger.info("Fragment drops refused: too little of the edit would be left")
        return []
    return positions


# How many audit rounds to run. Each round re-reads what actually survives, so a
# sentence only exposed by the previous round's cut gets its turn. Three is the
# point of diminishing returns on real recordings — and the loop usually stops
# earlier, either because nothing is broken or because a round made no progress.
MAX_VERIFY_ROUNDS = 3


async def verify_until_clean(words: List[Dict[str, Any]], ask,
                             max_rounds: int = MAX_VERIFY_ROUNDS,
                             drop_fragments: bool = True) -> Tuple[Dict[str, Any], List[int]]:
    """Audit and repair the edit repeatedly until it stops improving.

    `words` is the FULL word list; words already marked `disfluency` are treated
    as gone. Returns (quality report, indices into `words` to cut).

    One audit-repair pass was never enough, for a mechanical reason: repair works
    sentence by sentence, and removing debris from one sentence changes where the
    *next* sentence begins. Sentences that were run together, or split by a pause
    that a cut has now closed, are only judged correctly on the following read.
    So it loops, and every round reads what actually survives.

    It stops on the first of: nothing broken, a round that changed nothing, the
    model going silent, or `max_rounds`. It always ends on a fresh audit, so the
    score reported is the score of what ships — never of an intermediate state.
    """
    cut: set = set()
    rounds: List[Dict[str, Any]] = []
    mended: List[Issue] = []
    latest = AuditResult()

    for round_number in range(1, max_rounds + 1):
        live = [i for i, w in enumerate(words)
                if not w.get("disfluency") and i not in cut]
        survivors = [words[i] for i in live]
        if len(survivors) < MIN_WORDS_AFTER_REPAIR:
            break

        latest = await audit(survivors, ask)
        rounds.append({"round": round_number,
                       "sentences": latest.total,
                       "broken": latest.broken})
        if not latest.total:
            logger.warning("Verification stopped at round %d: the model did not answer",
                           round_number)
            break
        logger.info("Verification round %d: %d of %d sentences broken",
                    round_number, latest.broken, latest.total)
        if not latest.issues:
            break
        if round_number == max_rounds:
            break               # this audit scored what ships; no more repairs

        positions = await repair_broken(survivors, latest, ask)
        if drop_fragments:
            positions = positions + await find_droppable_fragments(survivors, latest, ask)

        mended.extend(i for i in latest.issues if i.repaired)
        fresh = {live[p] for p in positions if 0 <= p < len(live)} - cut
        rounds[-1]["cut"] = len(fresh)
        if not fresh:
            # Nothing moved. Another identical round would ask the same question
            # and get the same answer, so stop rather than burn the model on it.
            logger.info("Verification stopped at round %d: no repair was accepted",
                        round_number)
            break
        cut |= fresh

    quality = latest.as_dict()
    quality["rounds"] = rounds
    # `issues` above is what is still wrong at the end. What got *fixed* along the
    # way is the more useful half of the story, and the last audit cannot show it.
    quality["repairs"] = [{"text": i.text, "problem": i.problem, "removed": i.removed}
                          for i in mended]
    quality["verdict"] = (
        "not_verified" if not latest.total
        else "clean" if latest.broken == 0
        else f"still_broken:{latest.broken}"
    )
    return quality, sorted(cut)
