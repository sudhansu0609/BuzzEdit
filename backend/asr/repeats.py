"""Back-to-back phrase repeats: a deterministic cut and the report's measure.

The Raat3Baje ep2 cut of 2026-10-08 (project 656774a3) still played 14 exact
phrase repeats in what the viewer hears: "बारिश के बाद मिट्टी बारिश के बाद
मिट्टी" at 607 s (its trim was refused), "इस जोन में इस जोन में", "मुझे
एहसास हुआ कि अगर मुझे एहसास हुआ कि अगर". A phrase said twice back to back
needs no judgement to find, so it is found here by rule, after the editor,
like the stutter rule (AUTO_CUT_V3_PLAN.md §2, step 4):

- `exact_repeats()` lists the back-to-back n-gram repeats (n = 2..8) in what
  plays, read on the native script (`tokens.spoken`) and compared through
  `norm()`. One word said twice is the stutter rule's job, and a phrase made
  only of words Hindi doubles on purpose (`disfluency.REDUPLICATION_OK`,
  "धीरे धीरे") is grammar, not a repeat.
- `cut_phrase_repeats()` cuts the earlier copy. In a storytelling genre a
  clean, paused repeat may be said for effect ("वहाँ नीचे… वहाँ नीचे"), so
  there only a tight one (under 0.35 s between the copies) or one with an
  untranscribed hole between them is cut; the others are handed back as
  candidates for the verifier and left as they are.
- `repeats_remaining()` is the measure the report shows: the same scan over
  whatever is enabled now. It must read 0.
"""

import re
import unicodedata
from typing import Any, Dict, List, Tuple

from .disfluency import REDUPLICATION_OK
from .tokens import spoken

_NON_WORD_RE = re.compile(r"[^\w\u0900-\u097F]")

# Under this pause a repeat is a stumble even in a story: a phrase said again
# for effect is said after a breath, not on top of itself.
_TIGHT_GAP_S = 0.35


def norm(text: Any) -> str:
    """The token a repeat is compared on: NFC, lower case, punctuation removed.

    The Devanagari block is kept by name because the regex word class alone
    does not match the vowel signs: with it alone "की" and "का" both read "क",
    and two different phrases would count as one said twice.
    """
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    return _NON_WORD_RE.sub("", text)


# Compared in the same normal form as the tokens.
_REDUPLICATION = {norm(w) for w in REDUPLICATION_OK}


def _plays(w: Dict[str, Any]) -> bool:
    return bool(w.get("enabled", not w.get("disfluency", False)))


def _sec(w: Dict[str, Any], key: str) -> float:
    return float(w.get(key) or 0.0)


def exact_repeats(words: List[Dict[str, Any]], min_n: int = 2, max_n: int = 8,
                  max_gap: float = 1.5) -> List[Dict[str, Any]]:
    """Every back-to-back phrase repeat among the words that play, in order.

    Reads the words that are enabled and transcribed, in list order. A repeat
    at position i of length n is `norm` tokens [i:i+n] == [i+n:i+2n], with no
    empty token, at least one token longer than one character, and at most
    `max_gap` seconds from the end of the first copy to the start of the
    second. The longest n wins at each position and the scan resumes after the
    second copy, so results never overlap. n = 1 is never reported, and a
    phrase made only of reduplication words is skipped.

    Each result: `first` / `second` (indices into `words` of the two copies),
    `n`, `gap` (seconds), `text` (the first copy, spoken), `hole_between` (an
    untranscribed word sits between the copies in the full list) and `start`
    (the first copy's start, seconds).
    """
    keep = [i for i, w in enumerate(words)
            if _plays(w) and not w.get("untranscribed")]
    toks = [norm(spoken(words[i])) for i in keep]
    min_n = max(2, int(min_n))
    found: List[Dict[str, Any]] = []
    i = 0
    while i + 2 * min_n <= len(toks):
        hit = None
        for n in range(int(max_n), min_n - 1, -1):
            if i + 2 * n > len(toks):
                continue
            phrase = toks[i:i + n]
            if phrase != toks[i + n:i + 2 * n]:
                continue
            if not all(phrase) or not any(len(t) > 1 for t in phrase):
                continue
            if all(t in _REDUPLICATION for t in phrase):
                continue
            gap = round(_sec(words[keep[i + n]], "start")
                        - _sec(words[keep[i + n - 1]], "end"), 3)
            if gap > max_gap:
                continue
            hit = (n, gap)
            break
        if hit is None:
            i += 1
            continue
        n, gap = hit
        first, second = keep[i:i + n], keep[i + n:i + 2 * n]
        found.append({
            "first": first,
            "second": second,
            "n": n,
            "gap": gap,
            "text": " ".join(spoken(words[j]).strip() for j in first),
            "hole_between": any(words[k].get("untranscribed")
                                for k in range(first[-1] + 1, second[0])),
            "start": _sec(words[first[0]], "start"),
        })
        i += 2 * n
    return found


def cut_phrase_repeats(words: List[Dict[str, Any]], storytelling: bool = False
                       ) -> Tuple[int, List[Dict[str, Any]]]:
    """Cut the earlier copy of each back-to-back phrase repeat, in place.

    The earlier copy's words become `disfluency=True, candidate=False,
    enabled=False, reason="repeat"`. Outside storytelling every repeat is cut.
    In storytelling only a tight one (gap under 0.35 s) or one with an
    untranscribed hole between the copies is cut; any other -- clean and
    paused, possibly said on purpose ("वहाँ नीचे… वहाँ नीचे") -- is returned
    in `candidates` (the `exact_repeats` result dicts) and left untouched.

    One pass over `exact_repeats()`: of a phrase said three times only the
    first copy goes, and the pair left is what `repeats_remaining()` reports.
    Returns (repeats cut, candidates).
    """
    cut = 0
    candidates: List[Dict[str, Any]] = []
    for rep in exact_repeats(words):
        if storytelling and rep["gap"] >= _TIGHT_GAP_S and not rep["hole_between"]:
            candidates.append(rep)
            continue
        for j in rep["first"]:
            words[j]["disfluency"] = True
            words[j]["candidate"] = False
            words[j]["enabled"] = False
            words[j]["reason"] = "repeat"
        cut += 1
    return cut, candidates


def repeats_remaining(words: List[Dict[str, Any]], max_examples: int = 10
                      ) -> Dict[str, Any]:
    """How many exact phrase repeats still play: the report's measure, must be 0.

    `exact_repeats` over whatever is enabled now, with `max_gap` 1.5 s;
    `examples` carries the first `max_examples` as {"start": seconds, "text"}.
    """
    reps = exact_repeats(words, max_gap=1.5)
    return {
        "count": len(reps),
        "examples": [{"start": r["start"], "text": r["text"]}
                     for r in reps[:max(0, int(max_examples))]],
    }
