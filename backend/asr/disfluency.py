"""Deterministic disfluency detection (Layer 1 of the fumble pipeline).

Produces per-word annotations without doing anything risky on its own:

  - `disfluency`  : final auto-cut decision from rules ALONE. Conservative — only
                    non-lexical filler sounds ("um", "uh", "hmm", "aa") and clear
                    stutters/phrase-repeats. Safe to cut even with no LLM.
  - `candidate`   : an ambiguous cut candidate (soft filler like "like"/"matlab",
                    low-confidence micro-token, possible false start). Left for the
                    LLM adjudication layer to decide in context.
  - `reason`      : short human-readable tag for the UI / logs.

Works on romanized text, so Hinglish tokens ("matlab", "yaani", "haan") are
covered alongside English.
"""

import re
from typing import List, Dict, Any

from .retakes import apply_retakes

# Non-lexical vocalizations — these are (almost) never real words, safe to cut.
HARD_FILLERS = {
    "um", "umm", "ummm", "uh", "uhh", "uhm", "er", "err", "erm", "ah", "ahh",
    "hmm", "hmmm", "mm", "mmm", "huh", "eh", "ehh", "hm",
    # romanized Hindi hesitation sounds
    "aa", "aaa", "aaaa", "oo", "ooo",
}

# Ambiguous discourse markers / crutch words — only cut if the LLM (or an
# aggressive setting) confirms them in context. English + Hinglish.
SOFT_FILLERS = {
    # English
    "like", "basically", "actually", "literally", "seriously", "honestly",
    "so", "well", "right", "okay", "ok", "yeah", "yknow", "just", "really",
    # multi-word handled separately: "you know", "i mean", "sort of", "kind of"
    # Hinglish / Hindi crutch words
    "matlab", "yaani", "yani", "waise", "toh", "wo", "woh", "aisa", "aise",
    "haan", "bas", "arre", "arey", "acha", "achha", "yaar", "kya", "na",
}

# Multi-word soft fillers (checked on normalized bigrams).
SOFT_FILLER_PHRASES = {
    "you know", "i mean", "sort of", "kind of", "you see", "matlab ki",
    "i guess", "or something", "and stuff",
}

_PUNCT_RE = re.compile(r"[^\w]", re.UNICODE)


def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", str(text or "")).lower()


def _word_text(w: Dict[str, Any]) -> str:
    return str(w.get("word", w.get("text", "")))


def analyze_disfluencies(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotate words with disfluency / candidate / reason flags (deterministic)."""
    if not words:
        return words

    norms = [_norm(_word_text(w)) for w in words]
    out: List[Dict[str, Any]] = []

    for i, w in enumerate(words):
        w = dict(w)
        clean = norms[i]
        prob = float(w.get("probability", w.get("confidence", 1.0)) or 1.0)

        disfluency = False
        candidate = False
        reason = ""

        # 1. Hard non-lexical filler -> always cut.
        if clean and clean in HARD_FILLERS:
            disfluency, reason = True, "filler"

        # 2. Immediate stutter: same word repeated back-to-back ("I I", "the the").
        #    Back-to-back means back-to-back *in what survives* — "the uh the" is
        #    the same stutter with an editing term dropped into the middle of it,
        #    and comparing against the raw previous token misses every one of them.
        else:
            previous = next((p for p in reversed(out) if not p.get("disfluency")), None)
            if clean and previous is not None and _norm(_word_text(previous)) == clean:
                # cut the EARLIER duplicate(s); keep the last clean utterance.
                previous["disfluency"] = True
                previous["reason"] = "stutter"

        # 3. Soft filler -> candidate for LLM.
        if not disfluency and clean in SOFT_FILLERS:
            candidate, reason = True, "soft_filler"

        # 4. Multi-word soft filler (bigram with previous token).
        if not disfluency and i > 0:
            bigram = f"{norms[i - 1]} {clean}".strip()
            if bigram in SOFT_FILLER_PHRASES:
                candidate, reason = True, "soft_filler"
                if out:
                    out[-1]["candidate"] = True
                    out[-1].setdefault("reason", "soft_filler")

        # 5. Low-confidence micro-token (often a stray "uh"/noise mis-heard).
        if not disfluency and not candidate and clean and len(clean) <= 2 and prob < 0.35:
            candidate, reason = True, "low_confidence"

        w["disfluency"] = disfluency
        w["candidate"] = candidate
        w["reason"] = reason
        out.append(w)

    # 6. Abandoned attempts: the speaker starts a sentence, stops, and restarts —
    #    often several times before it lands. Handled by rough-copy detection in
    #    retakes.py rather than exact repeat matching, because a restart is a
    #    *prefix* of the good take with varying length and clipped words
    #    ("dosto kya ap · dosto kya · dosto kya apko pata hai…"), which exact
    #    n-gram comparison never matches.
    apply_retakes(out)

    return out
