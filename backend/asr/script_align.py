"""The user's script, aligned to what Whisper heard.

A script is the one source of truth the pipeline never had. Whisper writes
"sabsakraaib" for "subscribe" and "Thommel Gilgovich" for a name the speaker
knows perfectly well, and every stage downstream — captions, topic planning,
entity extraction — had to read past the misspellings. When the user has a
script, this module aligns it to the transcript words and:

* **fixes the caption spelling** — each aligned word takes the script's
  spelling, so captions read as what was said;
* **fixes the paragraph structure** — the script's paragraphs become the
  topic boundaries, which a model reading timestamps only ever estimated;
* **carries stage directions** — `[map: Jaipur]`, `[sfx: thunder]`,
  `[broll: an old haveli at night]`, `[text: 40% of students]` written into
  the script become beats at the moment the surrounding words are spoken.

Alignment is `difflib` over normalised tokens with a fuzzy pairing inside the
replaced runs. It does not need to be perfect: an unaligned word simply keeps
Whisper's spelling, and a directive anchors to the nearest aligned word.

Nothing here touches timing. The audio decides when a word was said; the
script only decides how it is spelled and what the speaker meant to show.
"""

import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("script_align")

# The bracket vocabulary. Anything else in brackets is left as spoken text
# (a script may legitimately contain "[laughs]"; that is not a directive).
DIRECTIVES = ("map", "sfx", "broll", "video", "text", "title", "card", "chapter",
              "music", "mood", "quote", "stat", "source", "character", "location",
              "chart", "definition", "split", "freeze", "newspaper", "case_file",
              "fx", "atmos", "grade",
              # designed graphics (presentation/graphics.py; grammar in
              # presentation/script.py::directive_beats)
              "statement", "canvas", "pills", "point", "name", "source_quote",
              "document", "timeline")
_DIRECTIVE_RE = re.compile(
    r"\[\s*(" + "|".join(DIRECTIVES) + r")\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_TOKEN_RE = re.compile(r"\S+")

# Below this similarity two tokens inside a replaced run are not the same word
# misspelled; they are different words, and the transcript keeps its own.
PAIR_MIN_RATIO = 0.5
# Inside an equal-length run, order is trusted down to this much similarity.
LOOSE_PAIR_RATIO = 0.34


# --- parsing ---------------------------------------------------------------------

def parse_script(text: str) -> Dict[str, Any]:
    """Paragraphs of spoken tokens plus the directives found among them.

    Returns {"paragraphs": [{"index", "text", "tokens", "heading"}],
             "directives": [{"kind", "arg", "paragraph", "token"}]}
    where `token` is the index (into the flat spoken-token list) of the word
    the directive sits before — the moment it should happen.
    """
    paragraphs: List[Dict[str, Any]] = []
    directives: List[Dict[str, Any]] = []
    flat_index = 0
    pending_heading: Optional[str] = None

    blocks = re.split(r"\n\s*\n", (text or "").replace("\r\n", "\n").strip())
    for block in blocks:
        lines = block.split("\n")
        spoken_lines: List[str] = []
        for line in lines:
            heading = _HEADING_RE.match(line)
            if heading:
                pending_heading = heading.group(1).strip()
                directives.append({"kind": "chapter", "arg": pending_heading,
                                   "paragraph": len(paragraphs), "token": flat_index})
                continue
            spoken_lines.append(line)
        body = "\n".join(spoken_lines).strip()
        if not body:
            continue

        tokens: List[str] = []
        cursor = 0
        for match in _DIRECTIVE_RE.finditer(body):
            before = body[cursor:match.start()]
            tokens.extend(_TOKEN_RE.findall(before))
            directives.append({
                "kind": match.group(1).lower(),
                "arg": match.group(2).strip(),
                "paragraph": len(paragraphs),
                "token": flat_index + len(tokens),
            })
            cursor = match.end()
        tokens.extend(_TOKEN_RE.findall(body[cursor:]))
        if not tokens:
            continue
        paragraphs.append({
            "index": len(paragraphs),
            "text": " ".join(tokens),
            "tokens": tokens,
            "first_token": flat_index,
            "heading": pending_heading,
        })
        pending_heading = None
        flat_index += len(tokens)

    return {"paragraphs": paragraphs, "directives": directives}


def flat_tokens(parsed: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for paragraph in parsed["paragraphs"]:
        out.extend(paragraph["tokens"])
    return out


# --- normalisation ----------------------------------------------------------------

def is_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(text or ""))


_STRIP_RE = re.compile(r"[^\wऀ-ॿ]+", re.UNICODE)


def _fold(token: str) -> str:
    """A spelling-tolerant key: lowercase, no punctuation, long vowels and
    doubled letters collapsed. 'sabsakraaib' and 'subscribe' still differ, but
    'jaz'/'jaz.' and 'Thommel'/'thommel' meet."""
    core = _STRIP_RE.sub("", (token or "").lower())
    core = re.sub(r"aa+", "a", core)
    core = re.sub(r"ee+", "i", core)
    core = re.sub(r"oo+", "u", core)
    core = re.sub(r"(.)\1+", r"\1", core)
    return core


def _romanise(token: str) -> str:
    if not is_devanagari(token):
        return token
    try:
        from .transliterate import to_hinglish
        return to_hinglish(token, "hi")
    except Exception:
        return token


def _word_keys(word: Dict[str, Any]) -> Tuple[str, str]:
    """(romanised key, native key) for a transcript word."""
    text = str(word.get("text") or word.get("word") or "")
    native = str(word.get("word_native") or "")
    roman = _fold(_romanise(text) if is_devanagari(text) else text)
    native_key = _fold(native) if native else ""
    return roman, native_key


# --- alignment ---------------------------------------------------------------------

def align(script_tokens: Sequence[str], words: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Pairs of (script token index, word index) that are the same spoken word.

    Exact matches on the folded key anchor the alignment; inside each replaced
    run, tokens are paired in order when they are similar enough, so a run of
    misspellings between two anchors still maps one to one.
    """
    if not script_tokens or not words:
        return []
    script_native = any(is_devanagari(t) for t in script_tokens)
    if script_native:
        script_keys = [_fold(t) for t in script_tokens]
        word_keys = [(_word_keys(w)[1] or _word_keys(w)[0]) for w in words]
    else:
        script_keys = [_fold(_romanise(t)) for t in script_tokens]
        word_keys = [_word_keys(w)[0] for w in words]

    matcher = difflib.SequenceMatcher(None, script_keys, word_keys, autojunk=False)
    pairs: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend((i, j1 + (i - i1)) for i in range(i1, i2))
        elif tag == "replace":
            pairs.extend(_pair_run(script_keys, word_keys, i1, i2, j1, j2))
    return pairs


def _pair_run(script_keys: List[str], word_keys: List[str],
              i1: int, i2: int, j1: int, j2: int) -> List[Tuple[int, int]]:
    """Pair two unequal runs in order, skipping whichever side is 'extra'."""
    pairs: List[Tuple[int, int]] = []
    i, j = i1, j1
    while i < i2 and j < j2:
        ratio = difflib.SequenceMatcher(None, script_keys[i], word_keys[j]).ratio()
        if ratio >= PAIR_MIN_RATIO:
            pairs.append((i, j))
            i += 1
            j += 1
            continue
        # Look one ahead on each side: is the next token a better partner?
        ahead_i = (difflib.SequenceMatcher(None, script_keys[i + 1], word_keys[j]).ratio()
                   if i + 1 < i2 else 0.0)
        ahead_j = (difflib.SequenceMatcher(None, script_keys[i], word_keys[j + 1]).ratio()
                   if j + 1 < j2 else 0.0)
        if ahead_i >= PAIR_MIN_RATIO and ahead_i >= ahead_j:
            i += 1              # the script has an extra word here
        elif ahead_j >= PAIR_MIN_RATIO:
            j += 1              # the transcript has an extra word here
        else:
            # Neither side pairs: when the runs are the same length, trust the
            # order (a badly romanised word can score under the ratio floor
            # against its own correct spelling) — unless the two share nothing
            # at all, which is an extra word on both sides, not a misspelling.
            if (i2 - i) == (j2 - j) and ratio >= LOOSE_PAIR_RATIO:
                pairs.append((i, j))
            i += 1
            j += 1
    return pairs


# --- applying the script -----------------------------------------------------------

def apply_spelling(words: List[Dict[str, Any]], script_tokens: Sequence[str],
                   pairs: Sequence[Tuple[int, int]]) -> int:
    """Give each aligned word the script's spelling. Mutates the dicts in place.

    A Devanagari script updates `word_native`; a romanised one updates `text`.
    Punctuation travels with the script token, so a sentence's full stop lands
    on the caption too. Returns how many words changed.
    """
    changed = 0
    for script_index, word_index in pairs:
        token = script_tokens[script_index]
        word = words[word_index]
        if is_devanagari(token):
            if (word.get("word_native") or "") != token:
                word["word_native"] = token
                changed += 1
        else:
            key = "text" if "text" in word else "word"
            if (word.get(key) or "") != token:
                word[key] = token
                changed += 1
        word["script_aligned"] = True
    return changed


def paragraph_spans(parsed: Dict[str, Any], words: Sequence[Dict[str, Any]],
                    pairs: Sequence[Tuple[int, int]],
                    start_key: str = "start", end_key: str = "end") -> List[Dict[str, Any]]:
    """Each paragraph's span in the WORDS' time base (source seconds/frames).

    A paragraph whose words were all unaligned inherits the gap between its
    neighbours rather than vanishing, so the topic boundaries stay complete.
    """
    by_script = dict(pairs)
    spans: List[Dict[str, Any]] = []
    for paragraph in parsed["paragraphs"]:
        first = paragraph["first_token"]
        last = first + len(paragraph["tokens"]) - 1
        aligned = [by_script[i] for i in range(first, last + 1) if i in by_script]
        entry: Dict[str, Any] = {
            "index": paragraph["index"],
            "heading": paragraph.get("heading"),
            "text": paragraph["text"],
            "aligned_words": len(aligned),
        }
        if aligned:
            entry["start"] = float(words[min(aligned)][start_key])
            entry["end"] = float(words[max(aligned)][end_key])
        spans.append(entry)

    # Fill the unaligned paragraphs from their neighbours.
    for index, entry in enumerate(spans):
        if "start" in entry:
            continue
        previous_end = next((spans[k]["end"] for k in range(index - 1, -1, -1)
                             if "end" in spans[k]), None)
        next_start = next((spans[k]["start"] for k in range(index + 1, len(spans))
                           if "start" in spans[k]), None)
        if previous_end is None and next_start is None:
            continue
        entry["start"] = previous_end if previous_end is not None else max(0.0, next_start - 1.0)
        entry["end"] = next_start if next_start is not None else entry["start"] + 1.0
        entry["estimated"] = True
    return [s for s in spans if "start" in s]


def directive_times(parsed: Dict[str, Any], words: Sequence[Dict[str, Any]],
                    pairs: Sequence[Tuple[int, int]],
                    start_key: str = "start") -> List[Dict[str, Any]]:
    """Each directive with the time of the word it sits before (words' base)."""
    by_script = dict(pairs)
    total = len(flat_tokens(parsed))
    out: List[Dict[str, Any]] = []
    for directive in parsed["directives"]:
        token = directive["token"]
        # The directive precedes `token`; the nearest aligned token at or after
        # it is its moment, falling back to the nearest one before.
        chosen = next((by_script[i] for i in range(token, total) if i in by_script), None)
        if chosen is None:
            chosen = next((by_script[i] for i in range(token - 1, -1, -1) if i in by_script), None)
        if chosen is None:
            continue
        entry = dict(directive)
        entry["at"] = float(words[chosen][start_key])
        out.append(entry)
    return out


def ingest(text: str, words: List[Dict[str, Any]],
           start_key: str = "start", end_key: str = "end") -> Dict[str, Any]:
    """Parse, align and apply a script to a word list in one go.

    Returns the record the project stores: the parsed structure with times in
    the words' time base, and the counts the UI reports.
    """
    parsed = parse_script(text)
    tokens = flat_tokens(parsed)
    pairs = align(tokens, words)
    fixed = apply_spelling(words, tokens, pairs)
    record = {
        "text": text,
        "token_count": len(tokens),
        "aligned_words": len(pairs),
        "spelling_fixed": fixed,
        "paragraphs": paragraph_spans(parsed, words, pairs, start_key, end_key),
        "directives": directive_times(parsed, words, pairs, start_key),
    }
    logger.info("Script: %d/%d tokens aligned, %d spellings fixed, %d paragraphs, "
                "%d directives", len(pairs), len(tokens), fixed,
                len(record["paragraphs"]), len(record["directives"]))
    return record
