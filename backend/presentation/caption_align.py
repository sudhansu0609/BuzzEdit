"""Phrase captions for a narrated vibe reel (REELS_DESK_PLAN.md VR1b).

The viewer must always read the GIVEN script's spelling -- never Whisper's --
so only the *timing* of each given word comes from what was heard: difflib
pairs the two normalised word lists (`align_given_words`), and a given word
with no heard match is spread evenly in time between its aligned neighbours
(`interpolate_times`). The timed words are then cut into short phrases a
caption can hold on screen (`break_phrases`), each trimmed so it never
overlaps the next (`pad_phrase_ends`).

Pure and GPU-free: the caller supplies the heard words (already transcribed
by `asr.faster_whisper_engine`) and the given text; nothing here touches a
model or the filesystem.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Devanagari combining marks this alignment treats as noise: the nukta (which
# turns ज into ज़, फ into फ़ ...) and the chandrabindu, folded onto the plain
# anusvara -- Whisper and a given script disagree on these far more often than
# on the word itself, and they carry no timing information.
NUKTA = "़"
CHANDRABINDU = "ँ"
ANUSVARA = "ं"

_STRIP_RE = re.compile(r"[^\wऀ-ॿ]+", re.UNICODE)
_WORD_SPLIT_RE = re.compile(r"\S+", re.UNICODE)

# A run of this many unmatched tokens on either side is not "the same word,
# misspelled" -- it is a different word, so a `replace` block is paired only
# inside the shorter span, in order, and only where the two keys still look
# alike.
_PAIR_MIN_RATIO = 0.5

PHRASE_MAX_WORDS = 4
PHRASE_MAX_S = 1.8
PAUSE_BREAK_S = 0.35
CAPTION_END_PAD_S = 0.15

# ।  Devanagari danda ("full stop"); the rest are the Latin punctuation a
# Hinglish line may end a clause on.
_BREAK_PUNCT_RE = re.compile(r"[।.?!,…]\s*$")


def normalize_word(word: str) -> str:
    """A matching key: lowercase, no nukta, chandrabindu folded to anusvara,
    punctuation stripped."""
    w = (word or "").strip().lower()
    w = w.replace(NUKTA, "").replace(CHANDRABINDU, ANUSVARA)
    return _STRIP_RE.sub("", w)


def split_words(text: str) -> List[str]:
    """Whitespace-separated tokens, punctuation kept (it decides phrase
    breaks) -- only `normalize_word` strips it, for matching."""
    return _WORD_SPLIT_RE.findall(text or "")


def _heard_key(word: Dict[str, Any]) -> str:
    native = str(word.get("word_native") or word.get("word") or "")
    return normalize_word(native)


def align_given_words(given_words: Sequence[str], heard_words: Sequence[Dict[str, Any]]
                      ) -> List[Tuple[int, Optional[int]]]:
    """[(given_index, heard_index)] in given order; `heard_index` is None
    where nothing in `heard_words` matched that given word."""
    given_keys = [normalize_word(w) for w in given_words]
    heard_keys = [_heard_key(w) for w in heard_words]
    matcher = difflib.SequenceMatcher(None, given_keys, heard_keys, autojunk=False)
    mapped: Dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                mapped[i1 + offset] = j1 + offset
        elif tag == "replace":
            span = min(i2 - i1, j2 - j1)
            for k in range(span):
                a_key, b_key = given_keys[i1 + k], heard_keys[j1 + k]
                if not a_key or not b_key:
                    continue
                ratio = difflib.SequenceMatcher(None, a_key, b_key).ratio()
                if ratio >= _PAIR_MIN_RATIO:
                    mapped[i1 + k] = j1 + k
    return [(i, mapped.get(i)) for i in range(len(given_words))]


def interpolate_times(given_words: Sequence[str], pairs: Sequence[Tuple[int, Optional[int]]],
                      heard_words: Sequence[Dict[str, Any]], total_duration: float
                      ) -> List[Dict[str, Any]]:
    """[{"text", "start", "end"}] for every given word, in order. An aligned
    word takes its heard word's own (start, end); a run of unaligned words is
    spread evenly across the gap between its matched neighbours (or the
    recording's own edges, for a run at the very start or end)."""
    n = len(given_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    for i, j in pairs:
        if j is not None:
            starts[i] = float(heard_words[j]["start"])
            ends[i] = float(heard_words[j]["end"])

    i = 0
    while i < n:
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < n and starts[j] is None:
            j += 1
        prev_end = ends[i - 1] if i > 0 else 0.0
        next_start = starts[j] if j < n else max(prev_end, total_duration)
        run_len = j - i
        span = max(0.0, next_start - prev_end)
        step = span / (run_len + 1)
        for k in range(run_len):
            starts[i + k] = prev_end + step * (k + 1)
            ends[i + k] = min(prev_end + step * (k + 2), next_start)
        i = j
    return [{"text": given_words[k], "start": starts[k] or 0.0, "end": ends[k] or 0.0}
            for k in range(n)]


def break_phrases(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[{"text", "start", "end"}] phrases of at most `PHRASE_MAX_WORDS` words
    and `PHRASE_MAX_S` seconds, breaking after sentence/clause punctuation and
    at a pause of more than `PAUSE_BREAK_S` between two words."""
    phrases: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    for idx, word in enumerate(words):
        current.append(word)
        is_last = idx == len(words) - 1
        span = current[-1]["end"] - current[0]["start"]
        ends_punct = bool(_BREAK_PUNCT_RE.search(word["text"]))
        too_many_words = len(current) >= PHRASE_MAX_WORDS
        too_long = span >= PHRASE_MAX_S
        pause = False
        if not is_last:
            gap = words[idx + 1]["start"] - word["end"]
            pause = gap > PAUSE_BREAK_S
        if is_last or ends_punct or too_many_words or too_long or pause:
            phrases.append({"text": " ".join(w["text"] for w in current),
                           "start": current[0]["start"], "end": current[-1]["end"]})
            current = []
    return phrases


def pad_phrase_ends(phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each phrase held `CAPTION_END_PAD_S` past its last word's end, but never
    into the next phrase's start."""
    out: List[Dict[str, Any]] = []
    for i, phrase in enumerate(phrases):
        end = phrase["end"] + CAPTION_END_PAD_S
        if i + 1 < len(phrases):
            end = min(end, phrases[i + 1]["start"])
        out.append({**phrase, "end": max(phrase["start"] + 0.05, end)})
    return out


def build_caption_phrases(given_text: str, heard_words: Sequence[Dict[str, Any]],
                          total_duration: float) -> List[Dict[str, Any]]:
    """The whole pipeline: split the given text into words, align them to what
    was heard, interpolate what did not match, and cut the result into timed
    on-screen phrases."""
    given_words = split_words(given_text)
    if not given_words:
        return []
    pairs = align_given_words(given_words, heard_words)
    timed = interpolate_times(given_words, pairs, heard_words, total_duration)
    phrases = break_phrases(timed)
    return pad_phrase_ends(phrases)
