"""A restart from the top, found deterministically.

The creator's channels open with a fixed greeting ("Hello doston, mera naam hai
Sudhansu aur aap dekh rahe hain Raat 3 Baje"). When a take goes wrong early he
starts the whole video again from that line. The editor's reads see the
duplicate intro, but not reliably: on Raat3Baje ep2 two of three reads removed
the abandoned first attempt in one run and none in the next (2026-10-08), and
either way the viewer would have heard the intro twice. So the greeting is the
rule, confirmed by what follows it: a later greeting inside the first half of
the recording is a restart when the lines after it repeat the lines after an
earlier greeting, and everything before the last such greeting is an earlier
attempt. A greeting-like line alone is not enough: on ep2 (2026-10-09) the rule
took one at 515.7 s and removed 141.7 s of speech the creator kept; his restart
was the greeting at 99.8 s.

The sign-off ("mera naam hai ... aur aap dekh rahe the Raat 3 Baje") also
matches the words, which is why only greetings in the first half count.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

from .utterances import _cosine, _key, _trigrams

# A name phrase ("mera naam", "naam hai") and a "you are watching" phrase
# ("dekh rahe"), in either script, at most GREETING_SPAN words apart. The ASR
# garbles the name itself ("sudhan so", "shudhan sho"), so it is not relied on;
# "naam" in a long story line with "dekh rahe" somewhere else is no greeting.
# "mera naam" / "naam hai" in either script; the recogniser also writes the
# creator's "mera naam" as "main naam" / "मैं नाम" (Raat3Baje ep2, 0.2 s), so
# a name word right after a first-person word counts too.
_NAME = re.compile(r"\b(?:mera|meri|main)\s+naa?m\b|\bnaam\s+hai\b|(?:मेरा|मेरी|मैं)\s+नाम|नाम\s+है")
_WATCHING = re.compile(r"\bdekha?\s+rahe\b|देख\s+रहे")
GREETING_SPAN = 12

#: A restart happens early; a greeting past this share of the recording is the
#: sign-off, or a reprise, not a restart.
RESTART_WINDOW = 0.5

#: The opening compared after each greeting: up to this many lines of at least
#: MIN_LINE_WORDS words, ending where the next greeting begins.
OPENING_LINES = 6
MIN_LINE_WORDS = 3
#: Character trigram cosine for a line said again (reworded, respelled).
LINE_MATCH = 0.5
#: Opening lines said again that confirm a restart.
PAIRS_NEEDED = 2


def _word_at(text: str, offset: int) -> int:
    return len(text[:offset].split())


def is_greeting(text: str) -> bool:
    t = (text or "").lower()
    names = [_word_at(t, m.start()) for m in _NAME.finditer(t)]
    watching = [_word_at(t, m.start()) for m in _WATCHING.finditer(t)]
    return any(abs(n - w) <= GREETING_SPAN for n in names for w in watching)


def _line(utterance: Any, words: Optional[Sequence[Dict[str, Any]]]) -> str:
    """The utterance as the spelling-folded keys of its words; its text without words."""
    if not words:
        return str(getattr(utterance, "text", "") or "")
    keys = (_key(words[i]) for i in getattr(utterance, "word_indices", ()) if 0 <= i < len(words))
    return " ".join(k for k in keys if k)


def confirms_restart(earlier_block: Sequence[str], later_block: Sequence[str]) -> bool:
    """Whether the later greeting starts the video over. A block is a greeting
    line followed by the lines of its opening. Two later opening lines that each
    match one of the earlier opening's lines confirm it."""
    if not earlier_block or not later_block:
        return False
    earlier = [_trigrams(line) for line in earlier_block[1:]]
    pairs = sum(1 for line in later_block[1:]
                if any(_cosine(_trigrams(line), e) >= LINE_MATCH for e in earlier))
    # The greeting line itself is a fixed line, so two greetings always match:
    # it is no evidence of a restart. Only the opening content decides (ep2:
    # a kept second greeting at 515 s opened a new part, not the video again).
    # An attempt abandoned after one line has only that line to repeat, so the
    # bar is the shorter opening's length, never below one.
    needed = max(1, min(PAIRS_NEEDED, len(earlier_block) - 1, len(later_block) - 1))
    return pairs >= needed


def restart_from_top(utterances: Sequence[Any], total_seconds: float,
                     words: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, str]:
    """uid -> note for every utterance before the last greeting in the first
    half of the recording whose opening repeats the opening after an earlier
    greeting (any earlier one, not only the previous). Empty when none does.
    `words` are what the utterances' word_indices point at (the editor's view);
    with them lines are compared as spelling-folded keys."""
    limit = max(0.0, float(total_seconds or 0.0)) * RESTART_WINDOW
    utterances = list(utterances)
    found = [n for n, u in enumerate(utterances)
             if float(u.start) <= limit and is_greeting(getattr(u, "text", ""))]
    if len(found) < 2:
        return {}
    blocks: List[List[str]] = []
    for k, n in enumerate(found):
        stop = found[k + 1] if k + 1 < len(found) else len(utterances)
        opening: List[str] = []
        for u in utterances[n + 1:stop]:
            line = _line(u, words)
            if len(line.split()) >= MIN_LINE_WORDS:
                opening.append(line)
                if len(opening) == OPENING_LINES:
                    break
        blocks.append([_line(utterances[n], words)] + opening)
    cut = None
    for k in range(1, len(found)):
        earlier = next((j for j in range(k) if confirms_restart(blocks[j], blocks[k])), None)
        if earlier is not None:
            cut = (utterances[found[k]], utterances[found[earlier]])
    if cut is None:
        return {}
    last, earlier = cut
    note = (f"restart from the top at {last.uid} ({float(last.start):.0f}s), repeating the opening "
            f"after {earlier.uid} ({float(earlier.start):.0f}s): an earlier attempt")
    return {u.uid: note for u in utterances if float(u.start) < float(last.start)}
