"""Romanize native-script transcripts into readable "Hinglish"-style Latin text.

Whisper transcribes Hindi (and other Indic languages) in their native script
(Devanagari). Users often want the words spelled with Latin letters instead
(e.g. "नमस्ते दोस्तों" -> "namaste doston"). This module does a deterministic,
offline transliteration and then applies light phonetic cleanup — most
importantly *final schwa deletion* — so the output reads the way people
actually type Hinglish rather than a literal academic transliteration
("hama baata" -> "ham baat").

It is intentionally dependency-light (no torch / neural models). Quality is
"clearly readable", not perfect; medial schwa deletion and colloquial
spellings are out of scope.
"""

import re
import logging
from typing import Optional

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


def _clean_itrans(token: str) -> str:
    """Turn one ITRANS-transliterated token into readable Hinglish."""
    # Anusvara / nasal markers -> n; visarga -> h.
    token = token.replace("M", "n").replace(".n", "n").replace("~N", "n")
    token = token.replace("~n", "n").replace("N^", "n")
    token = token.replace("H", "h")

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
    return token


def to_hinglish(text: str, lang: Optional[str]) -> str:
    """Romanize `text` from its native script into Hinglish-style Latin text.

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
        raw = _transliterate(text, source, _sanscript.ITRANS)
        # Clean each whitespace-separated token independently so punctuation and
        # spacing survive.
        parts = re.split(r"(\s+)", raw)
        cleaned = [p if p.isspace() or not p.strip() else _apply_token(p) for p in parts]
        return "".join(cleaned)
    except Exception as e:
        logger.warning(f"Transliteration failed for lang={lang}: {e}")
        return text


def _apply_token(token: str) -> str:
    """Clean a token while keeping leading/trailing punctuation intact."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", token, re.UNICODE)
    if not m:
        return _clean_itrans(token)
    lead, core, trail = m.groups()
    if not core:
        return token
    return f"{lead}{_clean_itrans(core)}{trail}"
