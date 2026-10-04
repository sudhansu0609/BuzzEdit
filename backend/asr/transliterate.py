"""Romanize native-script transcripts into natural, readable Hinglish.

Whisper transcribes Hindi (and other Indic languages) in their native script
(Devanagari), including English words spoken mid-sentence ("लाइफ" for "life").
This module turns that into the spelling people actually type: English
loanwords spelled in English (not phonetic Hindi guesses), Hindi in standard
Hinglish spelling ("nahi", "hoga", "hoon" -- not "naheen", "hogaa", "hoo.n"),
and never a stray Devanagari character.

Four layers, each a fallback for the one before it:
1. A ~300-word table of the highest-frequency Hindi words, in the spelling
   people actually type (`hinglish_data.COMMON_WORDS`).
2. A curated table of common English loanwords, restored to English spelling
   (`hinglish_data.LOANWORDS`).
3. Deterministic ITRANS transliteration + colloquial cleanup rules (schwa
   deletion, long-vowel contraction, nukta handling) for everything else,
   with a consonant-skeleton match against the caller's own vocabulary and a
   bundled English wordlist to catch loanwords the curated table doesn't have.
4. A per-channel glossary override, applied last so it always wins.

It is intentionally dependency-light (no torch / neural models). Quality is
"clearly readable", not perfect.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .hinglish_data import COMMON_WORDS, LOANWORDS

logger = logging.getLogger("transliterate")

# Whisper language codes whose native script is Devanagari.
_DEVANAGARI_LANGS = {"hi", "mr", "ne", "sa"}

# Other Indic scripts we can transliterate if faster-whisper ever detects them.
_SCRIPT_BY_LANG = {
    "hi": "DEVANAGARI",
    "mr": "DEVANAGARI",
    "ne": "DEVANAGARI",
    "sa": "DEVANAGARI",
    "bn": "BENGALI",
    "ta": "TAMIL",
    "te": "TELUGU",
    "kn": "KANNADA",
    "ml": "MALAYALAM",
    "gu": "GUJARATI",
    "pa": "GURMUKHI",
    "or": "ORIYA",
}

_transliterate = None
_sanscript = None


def _load():
    """Lazily import indic-transliteration so a missing dependency degrades
    gracefully (returns native text) instead of breaking transcription."""
    global _transliterate, _sanscript
    if _transliterate is not None:
        return True
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        _sanscript = sanscript
        _transliterate = transliterate
        return True
    except Exception as e:  # pragma: no cover - only when dep missing
        logger.warning(f"indic-transliteration unavailable ({e}); Hinglish output falls back to native script.")
        return False


def is_transliterable(lang: Optional[str]) -> bool:
    return bool(lang) and lang in _SCRIPT_BY_LANG


# --- Devanagari-leftover safety net -----------------------------------------
#
# `indic_transliteration`'s ITRANS scheme has no mapping for a handful of
# marks, most commonly candra-O (ॉ), which shows up in almost every English
# loanword Hindi speech borrows ("कॉफी", "कॉल", "डॉक्टर") and leaks straight
# through untransliterated ("कॉफी" -> "kaॉphI"). A Latin caption with a stray
# Devanagari glyph in it is worse than an imperfect romanization, so this is a
# hard guarantee applied to every candidate spelling, not a best-effort clean.
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_KNOWN_LEFTOVERS = {
    "ॉ": "o",   # ॉ DEVANAGARI VOWEL SIGN CANDRA O
    "ऑ": "o",   # ऑ DEVANAGARI LETTER CANDRA A
    "ं": "n",   # ं anusvara, if it ever survives raw
    "ँ": "n",   # ँ chandrabindu
    "़": "",    # ़ nukta combining mark
}


def _strip_devanagari_leftovers(token: str) -> str:
    if not _DEVANAGARI_RE.search(token):
        return token
    return "".join(_KNOWN_LEFTOVERS.get(c, "") if "ऀ" <= c <= "ॿ" else c
                   for c in token)


def _clean_itrans(token: str) -> str:
    """Turn one ITRANS-transliterated token into readable, colloquial Hinglish.

    Beyond the mechanical ITRANS cleanup (nasal markers, final schwa
    deletion), this contracts the long-vowel spelling ITRANS produces into
    the shorter forms Hinglish typing actually uses: "hogaa" -> "hoga",
    "karoongaa" -> "karunga". `to_hinglish` checks the ~300-word common-word
    table and the loanword table *before* falling back to this, so this only
    ever runs on words neither table has an exact answer for.
    """
    # Nukta retroflex flap (ड़/ढ़): ITRANS spells it ".D"/".Dh". Hinglish typing
    # drops the flap distinction and just writes "d"/"dh" ("बड़े" -> "bade",
    # not "ba.de" -- the exact bug seen in real transcripts).
    token = token.replace(".Dh", "dh").replace(".D", "d")

    # Anusvara / nasal markers -> n; visarga -> h.
    token = token.replace("M", "n").replace(".n", "n").replace("~N", "n")
    token = token.replace("~n", "n").replace("N^", "n")
    token = token.replace("H", "h")

    # Nukta ख़ (aspirated velar fricative) -> "kh", the spelling people type
    # ("ख़ास" -> "khaas", not "kaas"). Plain ख already lowercases to "kh" on
    # its own; only the nukta form gets a capital K from the library.
    token = token.replace("K", "kh")

    # Final schwa deletion: Hindi drops the inherent 'a' at the end of a word
    # ("aaj" not "aaja"). Only drop it when it follows a consonant, so real
    # vowel endings ("namaste", "kya", "hua") are preserved. Done before long
    # vowels are expanded so 'A' (=aa) endings are never touched.
    if len(token) > 2 and token.endswith("a") and token[-2].lower() not in "aeiou":
        token = token[:-1]

    # Long vowels: ITRANS uses capitals. Expand, then lowercase everything.
    replacements = {
        "A": "aa", "I": "ee", "U": "oo",
        "RRi": "ri", "R^i": "ri", "LLi": "li",
        "ai": "ai", "au": "au",
    }
    for k, v in replacements.items():
        token = token.replace(k, v)

    token = token.lower()
    # Collapse runs like "aaa" -> "aa" that expansion can produce.
    token = re.sub(r"([a-z])\1{2,}", r"\1\1", token)

    # Colloquial contraction: "aa"->"a", "ee"/"ii"->"i", "oo"/"uu"->"u" -- the
    # gap between the textbook transliteration and how people actually type
    # Hinglish. Guarded to never shorten a token below 2 letters: a drawn-out
    # hesitation sound ("aa", "oo") must stay exactly that, or the filler
    # vocabularies in disfluency.HARD_FILLERS stop recognising it.
    contracted = token
    contracted = re.sub(r"aa+", "a", contracted)
    contracted = re.sub(r"(?:ee|ii)+", "i", contracted)
    contracted = re.sub(r"(?:oo|uu)+", "u", contracted)
    if len(contracted) >= 2:
        token = contracted

    return _strip_devanagari_leftovers(token)


def to_hinglish(
    text: str,
    lang: Optional[str],
    vocabulary: Optional[Iterable[str]] = None,
    glossary: Optional[Dict[str, str]] = None,
) -> str:
    """Romanize `text` from its native script into natural Hinglish-style Latin
    text: English loanwords spelled in English, Hindi in standard Hinglish
    spelling ("nahi", "hoga", "hoon"), never a stray Devanagari character.

    `vocabulary` is the Latin words from the user's own script (project
    settings) -- checked before the bundled English wordlist when restoring a
    loanword, since it is the exact spelling the speaker is going to use.
    `glossary` is a dict of overrides (keyed by either the native or the
    romanized spelling) applied last, so a channel-specific correction always
    wins over the tables and rules below it.

    Returns `text` unchanged for Latin-script languages (e.g. English) or when
    transliteration is unavailable, so callers can use it unconditionally.
    """
    if not text:
        return text
    if not is_transliterable(lang):
        return text
    if not _load():
        return text
    try:
        scheme = _SCRIPT_BY_LANG[lang]
        source = getattr(_sanscript, scheme)
        vocab_index = _skeleton_index(vocabulary)
        # Clean each whitespace-separated token independently so punctuation and
        # spacing survive.
        parts = re.split(r"(\s+)", text)
        cleaned = [p if p.isspace() or not p.strip()
                   else _apply_token(p, source, vocab_index, glossary)
                   for p in parts]
        return "".join(cleaned)
    except Exception as e:
        logger.warning(f"Transliteration failed for lang={lang}: {e}")
        return text


# Python's `\W` (even with re.UNICODE, the default) treats Devanagari combining
# marks -- matras, virama, anusvara, nukta -- as *not* word characters, so the
# old `\W*` boundary-strip used when this ran on already-Latin ITRANS output
# silently drops the last matra off a native word ("होगा" -> core "होग" + a
# stray trailing "ा"). `_split_word` strips only genuine boundary punctuation,
# by an explicit whitelist of the scripts this module transliterates, so a
# combining mark anywhere in the token stays part of the core.
_WORD_CHAR_RE = re.compile(
    r"[A-Za-z0-9"
    r"ऀ-ॿ"   # Devanagari
    r"ঀ-৿"   # Bengali
    r"਀-੿"   # Gurmukhi
    r"઀-૿"   # Gujarati
    r"଀-୿"   # Oriya
    r"஀-௿"   # Tamil
    r"ఀ-౿"   # Telugu
    r"ಀ-೿"   # Kannada
    r"ഀ-ൿ"   # Malayalam
    r"]"
)


def _split_word(token: str) -> tuple:
    """(leading punctuation, core, trailing punctuation)."""
    n = len(token)
    i = 0
    while i < n and not _WORD_CHAR_RE.match(token[i]):
        i += 1
    j = n
    while j > i and not _WORD_CHAR_RE.match(token[j - 1]):
        j -= 1
    return token[:i], token[i:j], token[j:]


def _apply_token(
    token: str,
    source,
    vocab_index: Dict[str, List[str]],
    glossary: Optional[Dict[str, str]],
) -> str:
    """Clean a token while keeping leading/trailing punctuation intact."""
    lead, core, trail = _split_word(token)
    if not core:
        return token
    return f"{lead}{_romanize_word(core, source, vocab_index, glossary)}{trail}"


def _romanize_word(
    core: str,
    source,
    vocab_index: Dict[str, List[str]],
    glossary: Optional[Dict[str, str]],
) -> str:
    """Romanize one word: common-word table, then loanword table, then the
    general rules (with loanword-skeleton restore), then the glossary."""
    from .tokens import is_latin

    if is_latin(core):
        # Whisper sometimes writes a code-switched English word straight in
        # Latin script even inside an "hi" pass, occasionally garbled
        # ("aactually" for "actually"). Repair, don't re-transliterate.
        candidate = _fix_latin_typo(core, vocab_index)
    elif core in COMMON_WORDS:
        candidate = COMMON_WORDS[core]
    elif core in LOANWORDS:
        candidate = LOANWORDS[core]
    else:
        try:
            raw = _transliterate(core, source, _sanscript.ITRANS)
        except Exception:
            raw = core
        cleaned = _clean_itrans(raw)
        candidate = _restore_loanword(cleaned, vocab_index) or cleaned

    candidate = _strip_devanagari_leftovers(candidate)

    if glossary:
        override = (glossary.get(core) or glossary.get(candidate)
                    or glossary.get(candidate.lower()))
        if override:
            candidate = override

    return candidate


# --- English-loanword restoration -------------------------------------------
#
# Whisper writes an English loanword out phonetically in Devanagari ("लाइफ"),
# and the rules above romanize that phonetically too ("laaiph"). The curated
# LOANWORDS table above catches the common ones exactly; this is the fallback
# for everything else -- comparing consonant *skeletons* (digraph folds, vowels
# stripped) against the project's own vocabulary and a bundled English
# wordlist. Reuses `retakes.skeleton`/`retakes.phonetic`, the same machinery
# that already repairs romanization drift between takes ("laaiph"/"life" and
# "cornwall"/"kaॉnvel yoonivarsitee" are the same kind of problem).
_ENGLISH_WORDLIST_PATH = Path(__file__).parent / "data" / "english_words.txt"
_MIN_RESTORE_SKELETON_LEN = 2

_english_wordlist_cache: Optional[List[str]] = None
_english_skeleton_index_cache: Optional[Dict[str, List[str]]] = None
_english_word_set_cache: Optional[set] = None


def _load_english_wordlist() -> List[str]:
    global _english_wordlist_cache
    if _english_wordlist_cache is None:
        try:
            with open(_ENGLISH_WORDLIST_PATH, "r", encoding="utf-8") as f:
                _english_wordlist_cache = [w.strip().lower() for w in f if w.strip()]
        except Exception as e:
            logger.warning(f"Could not load bundled English wordlist ({e}); "
                            "loanword restore falls back to curated tables only.")
            _english_wordlist_cache = []
    return _english_wordlist_cache


def _english_skeleton_index() -> Dict[str, List[str]]:
    global _english_skeleton_index_cache
    if _english_skeleton_index_cache is None:
        from .retakes import skeleton as _skeleton
        index: Dict[str, List[str]] = {}
        for w in _load_english_wordlist():
            index.setdefault(_skeleton(w), []).append(w)
        _english_skeleton_index_cache = index
    return _english_skeleton_index_cache


def _english_word_set() -> set:
    global _english_word_set_cache
    if _english_word_set_cache is None:
        _english_word_set_cache = set(_load_english_wordlist())
    return _english_word_set_cache


def _skeleton_index(vocabulary: Optional[Iterable[str]]) -> Dict[str, List[str]]:
    """Skeleton -> [original spellings] for the project's own vocabulary."""
    if not vocabulary:
        return {}
    from .retakes import skeleton as _skeleton
    index: Dict[str, List[str]] = {}
    for w in vocabulary:
        w = str(w or "").strip()
        if not w:
            continue
        index.setdefault(_skeleton(w), []).append(w)
    return index


_VOWEL_RUN_RE = re.compile(r"[aeiou]+")


def _vowel_pattern(word: str) -> str:
    return "".join(_VOWEL_RUN_RE.findall(word.lower()))


def _levenshtein(a: str, b: str, limit: int = 2) -> Optional[int]:
    """Edit distance, or None once it is certain to exceed `limit` (cheap
    early-out; these strings are always short words)."""
    if abs(len(a) - len(b)) > limit:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            best = min(best, cur[j])
        if best > limit:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= limit else None


def _closest_by_vowels(token: str, candidates: List[str]) -> Optional[str]:
    """Break a skeleton tie on which candidate's vowel pattern is closest to
    what was actually said. Requires a clear winner -- an ambiguous tie is
    not evidence, and leaves the rule-based spelling in place instead."""
    if len(candidates) == 1:
        return candidates[0]
    target = _vowel_pattern(token)
    scored = sorted(candidates, key=lambda c: _levenshtein(target, _vowel_pattern(c), 99) or 99)
    d0 = _levenshtein(target, _vowel_pattern(scored[0]), 99) or 99
    d1 = _levenshtein(target, _vowel_pattern(scored[1]), 99) or 99
    return scored[0] if d0 < d1 else None


# The bundled wordlist is generic (thousands of common English words with no
# idea what this recording is about), so a 2-letter skeleton match against it
# fires on real Hindi words too often -- "kyun" ("why") and "can", "rukh"
# ("stance") and "rich" collide at 2 letters once retakes.skeleton's w/v fold
# and y-as-vowel rule apply. The project's own vocabulary is small and
# deliberate (the speaker's own script), so it keeps the lower bar.
_MIN_VOCAB_SKELETON_LEN = _MIN_RESTORE_SKELETON_LEN
_MIN_WORDLIST_SKELETON_LEN = 3


# The generic wordlist has no idea what this recording is about, so a bare
# skeleton match against it turns real Hindi words into unrelated English
# ones just as often as it recovers a real loanword: "sakate" ("can") and
# "society", "hindi" (the language) and "hand", "taarget" and "throughout"
# were all real skeleton collisions measured on a sample transcript. Requiring
# the *vowel pattern* -- not just the skeleton -- to be a near-exact match
# keeps genuine phonetic drift ("dipharent" -> "different", one vowel off)
# while rejecting those. Vocabulary has no such gate: it is the small,
# deliberate list of words the speaker's own script uses, not a generic
# dictionary, so a skeleton hit against it is trusted directly.
_MAX_WORDLIST_VOWEL_DISTANCE = 1


def _restore_loanword(latin_token: str, vocab_index: Dict[str, List[str]]) -> Optional[str]:
    """Try to recover an English loanword from its romanized Hindi spelling."""
    from .retakes import skeleton as _skeleton
    key = _skeleton(latin_token)

    if len(key) >= _MIN_VOCAB_SKELETON_LEN:
        candidates = vocab_index.get(key)
        if candidates:
            best = _closest_by_vowels(latin_token, candidates)
            if best:
                return best

    if len(key) >= _MIN_WORDLIST_SKELETON_LEN:
        candidates = _english_skeleton_index().get(key)
        if candidates:
            best = _closest_by_vowels(latin_token, candidates)
            if best:
                target = _vowel_pattern(latin_token)
                dist = _levenshtein(target, _vowel_pattern(best), 99)
                if dist is not None and dist <= _MAX_WORDLIST_VOWEL_DISTANCE:
                    return best
    return None


def _fix_latin_typo(token: str, vocab_index: Dict[str, List[str]]) -> str:
    """Repair a Latin token Whisper mis-heard as a garbled English word
    ("aactually" for "actually"). Only fires within edit distance <=2 of a
    same-skeleton vocabulary/wordlist word -- conservative, so it never
    rewrites a real Hinglish word that is already in Latin script ("acha",
    "bas", "matlab")."""
    from .retakes import skeleton as _skeleton
    low = token.lower()
    if low in _english_word_set():
        return token
    key = _skeleton(low)
    for index, min_len in ((vocab_index, _MIN_VOCAB_SKELETON_LEN),
                            (_english_skeleton_index(), _MIN_WORDLIST_SKELETON_LEN)):
        if len(key) < min_len:
            continue
        candidates = index.get(key)
        if not candidates:
            continue
        best, best_dist = None, 3
        for c in candidates:
            d = _levenshtein(low, c.lower(), 2)
            if d is not None and d > 0 and d < best_dist:
                best, best_dist = c, d
        if best:
            return best
    return token
