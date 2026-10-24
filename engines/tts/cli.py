"""Local text-to-speech narration (AUTO_STORY_PLAN.md brief N1).

Runs under the TTS engine's own Python (`engines/tts/.venv`, torch + chatterbox-tts + kokoro),
never inside the backend: `backend/audio/narration.py` spawns it like the seed-vc CLI.

    python cli.py voices [--voices-dir DIR]
    python cli.py synthesize --text-file T --engine chatterbox|kokoro --voice <stem|preset|engine>
                             --language hi|en --exaggeration 0.5 --pace 1.0 --out OUT.wav

Protocol: JSON progress lines on stderr `{"progress": 0.4, "message": "..."}`, one JSON object on
stdout (`{"wav", "duration_s", "engine", "voice", "batches", "transliterated_text", ...}`), exit 0
ok / 1 error (stdout `{"error": ...}`) / 2 engine unavailable (its import failed). The model is
loaded once per process and the process exits after the job (nothing stays resident).

The text handling -- cues, sentence batches, Hinglish -> Devanagari, crossfaded joins -- is plain
Python (numpy at most) and imports nothing heavy at module level, so the backend's tests import it.

* Cues `(pause)` 0.6 s, `(pause 2s)` N s, `(beat)` 0.3 s become silences between batches;
  `(slow)` lowers the pace of what follows until `(normal)`; `(dramatic)`, `(intense)`,
  `(soft)`, `(calm)` switch a batch's exaggeration/cfg_weight/pace (layered on the base
  style) until `(normal)`, per `MOOD_CUES` (VOICE_DELIVERY_PLAN.md). None of them is spoken.
* `style` (mood preset from `MOODS` <- explicit keys: exaggeration, cfg_weight, temperature,
  pace, word_gap_ms, phrase_pause_s, sentence_pause_s) is resolved once by `resolve_style()`
  and carried through `--style '<json>'`; `--exaggeration`/`--pace` still work alone for
  backward compatibility. `space_out()` widens word/comma/sentence-length quiet runs already
  inside a generated batch; `insert_sentence_pauses()` adds the gap *between* batches.
* Batches are whole sentences (split at । . ? ! …) packed up to 220 characters, regardless of
  `sentence_pause_s` (never split one sentence per batch); a longer sentence is cut at commas,
  then spaces.
* Hinglish (`--language hi`): a word of plain ASCII letters that is not on the English list, not
  ALL CAPS, not a Capitalised name mid-sentence and without an English letter cluster is written
  in Devanagari (ITRANS after a few Hinglish spelling rules); everything else stays Latin.
* Every batch is synthesised (a failed or empty one is retried once), its own leading/trailing
  quiet trimmed to 60 ms (so it cannot stack with the gap inserted between batches), verified
  against what it actually said (chatterbox only -- Whisper transcribes it and scores the
  transcript against the script; dropped words regenerate it, babble past the real last word is
  cut there; a length-based guard regenerates, never cuts, if Whisper cannot load -- E4), spaced,
  joined with 60 ms crossfades and the cue silences, and written as 24 kHz mono 16-bit WAV.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOICES_DIR = ROOT / "data" / "voices"

SAMPLE_RATE = 24000
MAX_BATCH_CHARS = 220
XFADE_S = 0.06
PAUSE_S = 0.6
BEAT_S = 0.3
SLOW_FACTOR = 0.85            # (slow): pace x 0.85 until (normal)
SPACE_WIN_S = 0.01            # space_out()/trim_edges()/trim_babble(): 10 ms RMS envelope
SPACE_XFADE_S = 0.005         # space_out(): 5 ms crossfades, no clicks
SPACE_QUIET_REL_DB = 35.0     # quiet = below (batch peak dB - this) ...
SPACE_QUIET_ABS_DBFS = -40.0  # ... and below this
SPACE_WORD_GAP_MS = (60.0, 150.0)     # a quiet run in this range is a word gap
SPACE_PHRASE_RANGE_MS = (150.0, 350.0)  # a longer run, up to this, is a comma-length pause
SPACE_MAX_GAP_S = 1.6          # every resulting internal gap (run + inserted extra) is capped here
SPACE_EDGE_TARGET_S = 0.06     # trim_edges(): a batch's own leading/trailing quiet is cut to this
MIN_BATCH_S = 0.25            # a batch shorter than this came back empty: retried once
# Babble guard (VOICE_DELIVERY_PLAN.md V1), now the *fallback* path only (E4): when the Whisper
# verifier below cannot load, a batch is still checked by length -- expected seconds = spoken
# chars / (this * pace). Calibrated against one real Chatterbox batch: ~190 Devanagari chars at
# pace 1.0 took 11.9 s of audio (including ~4 s of its own pauses) -- 190 / 11.9 =~ 16 chars/s.
# The fallback never cuts (only `trim_babble`'s own tests call it now); it regenerates and keeps
# whichever attempt's length is closest to expected.
BABBLE_RATE_CHARS_PER_S = 16.0
BABBLE_OVER_FACTOR = 1.5       # a batch over this x expected ...
BABBLE_OVER_MARGIN_S = 1.0     # ... plus this many seconds is treated as hallucinated babble
BABBLE_REGEN_ATTEMPTS = 2      # extra synthesis attempts (new seed) in the fallback path
BABBLE_CUT_MIN_FRACTION = 0.85  # only cut at a quiet run past this fraction of expected_s
BABBLE_CUT_MIN_GAP_MS = 350.0   # ... and only a quiet run at least this long (sentence-like)
BABBLE_UNDER_FACTOR = 0.6       # a batch under this x expected stopped early (sentences lost) ...
BABBLE_UNDER_MIN_EXPECTED_S = 2.0  # ... checked only when at least this much speech is expected
# Whisper verification (VOICE_DELIVERY_PLAN.md E4): every chatterbox batch is transcribed and
# scored against its own script; see `score_transcript`/`verify_decision`.
WHISPER_SR = 16000
WHISPER_BEAM_SIZE = 5
# (model, device, compute_type) attempts, in order, for the verifier: GPU fp16 first, then a
# lighter GPU quant, then CPU on a smaller model.
WHISPER_ATTEMPTS: List["tuple[str, str, str]"] = [
    ("large-v3-turbo", "cuda", "float16"),
    ("large-v3-turbo", "cuda", "int8_float16"),
    ("small", "cpu", "int8"),
]
VERIFY_MIN_COVERAGE = 0.85     # fraction of script words Whisper must match, in order, to accept
VERIFY_MAX_EXTRA_WORDS = 1     # words heard past the last matched one before it is "babble"
VERIFY_FUZZY_RATIO = 0.6       # two normalised words count as the same word at this char-ratio+
VERIFY_CUT_PAD_S = 0.12        # cut this long after the last matched word's end timestamp
# Clean endings (2026-10-23: a breathy "whizz" after the last word, or a hard stop mid-vowel):
# whatever runs past the last matched word + TAIL_KEEP_S is cut even when Whisper heard no extra
# word in it, every spoken batch fades in/out, and the narration ends on real silence.
TAIL_KEEP_S = 0.25
TAIL_FADE_IN_S = 0.01
TAIL_FADE_OUT_S = 0.12
END_SILENCE_S = 0.5
VERIFY_REGEN_ATTEMPTS = 3      # extra synthesis attempts (new seed) while coverage is short
_VERIFY_NUKTA = "़"       # ़  combining nukta: dropped (ख़ और ख compare equal)
_VERIFY_CHANDRABINDU = "ँ"  # ँ  folded to anusvara below (Whisper is inconsistent about it)
_VERIFY_ANUSVARA = "ं"    # ं
CFG_WEIGHT = 0.5
EXIT_ERROR = 1
EXIT_UNAVAILABLE = 2
ENGINES = ("chatterbox", "kokoro")
LANGUAGES = ("hi", "en")
ENGINE_VOICE = "engine"       # the engine's own default voice (no reference clip)

# Inline (dramatic)/(intense)/(soft)/(calm) cues: exaggeration/cfg_weight/pace layered on the
# base style for the batches they cover, plus a silence said to come "before it" (0 = none).
MOOD_CUES: Dict[str, Dict[str, float]] = {
    "dramatic": {"exaggeration": 0.85, "cfg_weight": 0.4, "pace_factor": 0.95, "before_s": 0.3},
    "intense": {"exaggeration": 1.0, "cfg_weight": 0.5, "pace_factor": 1.05, "before_s": 0.0},
    "soft": {"exaggeration": 0.3, "cfg_weight": 0.35, "pace_factor": 0.9, "before_s": 0.2},
    "calm": {"exaggeration": 0.35, "cfg_weight": 0.35, "pace_factor": 0.92, "before_s": 0.0},
}

# Kokoro's presets; the first letter is the pipeline's lang_code (h = Hindi, a/b = English).
KOKORO_VOICES: Dict[str, List[str]] = {
    "hi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "en": ["af_heart", "af_bella", "am_michael", "am_fenrir", "bf_emma", "bm_george"],
}
KOKORO_DEFAULT = {"hi": "hm_omega", "en": "af_heart"}

# English words that also turn up in Hinglish scripts; kept Latin. Words that are Hindi too
# ("do", "par", "mat", "the", "me", "so", "to") are deliberately not here.
ENGLISH_WORDS = frozenset("""
a an and are at be been but by call camera car channel college comment doctor door family flat floor
for friend from game ghost good hello horror hospital hotel house how i if in is it just
laptop lift light like lock message minute minutes mobile night of office ok okay on one or
phone please police real really road room rule rules school share sir sorry station story
subscribe thank thanks that this time train unlock video was what when where which who why
will window with yes you your
""".split())
# Letter groups that do not occur in romanised Hindi.
ENGLISH_CLUSTERS = re.compile(r"ck|tion|sion|ght|wh|ph|x|ous|ble|nce|ow|ew|c(?!h)")
# Common words whose Hinglish spelling does not map onto ITRANS: word -> ITRANS.
HINDI_OVERRIDES: Dict[str, str] = {
    "hai": "hai", "hain": "haiM", "main": "maiM", "mein": "meM", "me": "meM", "nahi": "nahIM",
    "nahin": "nahIM", "hoon": "hU.N", "hun": "hU.N", "kahan": "kahA.N", "yahan": "yahA.N",
    "wahan": "vahA.N", "haan": "hA.N", "kyun": "kyoM", "kyon": "kyoM", "kya": "kyA",
    "bhi": "bhI", "hi": "hI", "ki": "kI", "ji": "jI", "bhai": "bhAI", "theek": "ThIk",
    "thik": "ThIk", "tum": "tuma", "hum": "hama", "aur": "aura", "par": "para",
}

_CUE_RE = re.compile(
    r"\(\s*(?:(?P<pause>pause)(?:\s+(?P<secs>\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?)?"
    r"|(?P<beat>beat)|(?P<slow>slow)|(?P<normal>normal)"
    r"|(?P<mood>dramatic|intense|soft|calm))\s*\)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[।.?!…])\s+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_PUNCT = "।.?!…"


# --------------------------------------------------------------------------- protocol

def emit_progress(fraction: float, message: str) -> None:
    print(json.dumps({"progress": round(float(fraction), 3), "message": message}, ensure_ascii=False),
          file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- text handling

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_cues(text: str) -> List[Dict[str, Any]]:
    """The script as `{"text", "slow", "mood"}` and `{"silence": seconds}` items, in order.
    Cues are removed from the spoken text; `(slow)` / `(normal)` switch the pace of what
    follows, `(dramatic)`/`(intense)`/`(soft)`/`(calm)` switch its `mood` (also `MOOD_CUES`)
    until `(normal)`, which ends both."""
    items: List[Dict[str, Any]] = []
    slow = False
    mood: Optional[str] = None
    pos = 0
    for match in _CUE_RE.finditer(text):
        chunk = _clean(text[pos:match.start()])
        if chunk:
            items.append({"text": chunk, "slow": slow, "mood": mood})
        pos = match.end()
        if match.group("pause"):
            secs = match.group("secs")
            items.append({"silence": float(secs) if secs else PAUSE_S})
        elif match.group("beat"):
            items.append({"silence": BEAT_S})
        elif match.group("slow"):
            slow = True
        elif match.group("normal"):
            slow = False
            mood = None
        elif match.group("mood"):
            mood = match.group("mood").lower()
            before_s = MOOD_CUES[mood]["before_s"]
            if before_s:
                items.append({"silence": before_s})
    chunk = _clean(text[pos:])
    if chunk:
        items.append({"text": chunk, "slow": slow, "mood": mood})
    return items


def split_sentences(text: str) -> List[str]:
    return [s for s in (_clean(p) for p in _SENTENCE_END.split(text)) if s]


def _split_long(sentence: str, limit: int) -> List[str]:
    """A sentence over `limit` characters: cut at commas, then at spaces, then hard."""
    if len(sentence) <= limit:
        return [sentence]
    pieces: List[str] = []
    for part in re.split(r"(?<=[,;:])\s+", sentence):
        if len(part) <= limit:
            pieces.append(part)
            continue
        line = ""
        for word in part.split(" "):
            while len(word) > limit:
                if line:
                    pieces.append(line)
                    line = ""
                pieces.append(word[:limit])
                word = word[limit:]
            candidate = f"{line} {word}".strip()
            if len(candidate) > limit:
                pieces.append(line)
                line = word
            else:
                line = candidate
        if line:
            pieces.append(line)
    return _pack(pieces, limit)


def _pack(pieces: Sequence[str], limit: int) -> List[str]:
    batches: List[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > limit:
            batches.append(current)
            current = piece
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def plan_batches(text: str, limit: int = MAX_BATCH_CHARS) -> List[Dict[str, Any]]:
    """The synthesis plan: `{"text", "slow", "mood"}` batches of whole sentences (<= `limit`
    chars; never spanning a mood change or a cue silence), packed as many to a batch as fit, and
    `{"silence": s}` items from the cues, in order. Adjacent silences merge. A batch is never
    split per sentence -- sentence and comma spacing, and the pause between sentences, are
    applied afterwards on the generated audio (`space_out`, `insert_sentence_pauses`)."""
    plan: List[Dict[str, Any]] = []
    for item in parse_cues(text):
        if "silence" in item:
            if plan and "silence" in plan[-1]:
                plan[-1]["silence"] += item["silence"]
            else:
                plan.append(dict(item))
            continue
        sentences: List[str] = []
        for sentence in split_sentences(item["text"]):
            sentences.extend(_split_long(sentence, limit))
        for batch in _pack(sentences, limit):
            plan.append({"text": batch, "slow": item["slow"], "mood": item["mood"]})
    return plan


def insert_sentence_pauses(plan: Sequence[Dict[str, Any]],
                           sentence_pause_s: float) -> List[Dict[str, Any]]:
    """Add the pause between batches (a packed batch boundary is a sentence end): between two
    consecutive spoken batches with no cue silence between them, insert a `{"silence":
    sentence_pause_s}` item; where a cue silence already sits there, raise it to
    `max(cue_silence, sentence_pause_s)` instead of adding to it -- never both. Never before the
    first batch (a leading cue silence there is a mood's own lead-in, not a sentence gap).
    `sentence_pause_s <= 0` returns the plan unchanged."""
    if sentence_pause_s <= 0:
        return list(plan)
    out: List[Dict[str, Any]] = []
    prev_was_text = False
    for item in plan:
        if "silence" in item:
            if prev_was_text:
                item = dict(item)
                item["silence"] = max(item["silence"], sentence_pause_s)
            out.append(item)
            prev_was_text = False
            continue
        if prev_was_text:
            out.append({"silence": sentence_pause_s})
        out.append(dict(item))
        prev_was_text = True
    return out


def is_hindi_word(word: str, sentence_start: bool = False) -> bool:
    """Should this Roman-script word be written in Devanagari? Deterministic: plain ASCII letters,
    not ALL CAPS, not on the English list, not a Capitalised name mid-sentence, no English letter
    cluster."""
    if not word or not word.isascii() or not word.isalpha():
        return False
    if len(word) > 1 and word.isupper():
        return False
    low = word.lower()
    if low in HINDI_OVERRIDES:
        return True
    if low in ENGLISH_WORDS:
        return False
    if word[0].isupper() and not sentence_start:
        return False
    if ENGLISH_CLUSTERS.search(low):
        return False
    return True


def to_itrans(word: str) -> str:
    """Hinglish spelling -> ITRANS: w is v, a final a / i is long (darwaza, apni)."""
    low = word.lower()
    if low in HINDI_OVERRIDES:
        return HINDI_OVERRIDES[low]
    s = low.replace("w", "v")
    if len(s) > 1 and s.endswith("a") and not s.endswith("aa"):
        s = s[:-1] + "A"
    elif len(s) > 1 and s.endswith("i") and not s.endswith(("ii", "ai", "ei")):
        s = s[:-1] + "I"
    return s


def _devanagari(itrans: str) -> Optional[str]:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        return None
    out = transliterate(itrans, sanscript.ITRANS, sanscript.DEVANAGARI)
    return _schwa_spelling(out)


_HALANT = "्"
_KEEP_BEFORE = "यर"          # य र: real conjuncts (kya, satrah)
_NASALS = "नम"              # न म before a consonant: andar, ambar


def _schwa_spelling(word: str) -> str:
    """ITRANS writes every vowel-less consonant with a halant; Hindi spells most of them as full
    letters (bajkar -> बजकर, not बज्कर). Keep the halant only before य / र, after न / म and
    in a doubled consonant (pakka); never on the last letter."""
    chars = list(word)
    out = []
    for i, ch in enumerate(chars):
        if ch != _HALANT:
            out.append(ch)
            continue
        if i == len(chars) - 1:
            continue
        prev = chars[i - 2] if i >= 2 and chars[i - 1] == "़" else chars[i - 1]
        nxt = chars[i + 1]
        if nxt in _KEEP_BEFORE or prev in _NASALS or nxt == prev:
            out.append(ch)
    return "".join(out)


def transliterate_text(text: str) -> str:
    """Hindi words of Roman-script Hinglish in Devanagari; English words and names stay Latin.
    Without indic_transliteration the text comes back unchanged."""
    def replace(match: "re.Match[str]") -> str:
        word = match.group(0)
        before = text[:match.start()].rstrip()
        start = not before or before[-1] in _SENTENCE_PUNCT or before[-1] in "\"'“‘:("
        if not is_hindi_word(word, sentence_start=start):
            return word
        converted = _devanagari(to_itrans(word))
        return converted if converted else word
    return _WORD_RE.sub(replace, text)


def plan_text(plan: Sequence[Dict[str, Any]]) -> str:
    """The plan as one reviewable line: spoken text with `(pause Ns)` for silences."""
    parts = []
    for item in plan:
        if "silence" in item:
            parts.append(f"(pause {item['silence']:g}s)")
        else:
            parts.append(("(slow) " if item.get("slow") else "") + item.get("spoken", item["text"]))
    return " ".join(parts)


# --------------------------------------------------------------------------- style and moods

DEFAULT_MOOD = "steady"
# Engine defaults, before any mood or explicit key is layered on (VOICE_DELIVERY_PLAN.md's
# "Contract: style" table).
STYLE_DEFAULTS: Dict[str, float] = {
    "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "pace": 1.0,
    "word_gap_ms": 0.0, "phrase_pause_s": 0.0, "sentence_pause_s": 0.0,
}
# Mood presets for the request-level `style.mood` (distinct from the inline `MOOD_CUES`); the
# single source for both the engine and `GET /api/narration/moods`. Tune by ear.
# `phrase_pause_s`/`sentence_pause_s` are *additions* to the model's own pauses (`space_out`'s
# comma-/sentence-length quiet runs, and the gap `insert_sentence_pauses` puts between batches),
# not replacements -- halved from the first cut of these numbers for that reason (V1).
MOODS: Dict[str, Dict[str, Any]] = {
    "steady": {
        "id": "steady", "label": "Steady storyteller (default)",
        "description": "Even pacing and moderate expressiveness; the default delivery.",
        "style": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "pace": 0.95,
                  "word_gap_ms": 40, "phrase_pause_s": 0.125, "sentence_pause_s": 0.225},
    },
    "calm": {
        "id": "calm", "label": "Calm, relaxed",
        "description": "Slower and softer, with more room between phrases.",
        "style": {"exaggeration": 0.35, "cfg_weight": 0.35, "temperature": 0.6, "pace": 0.9,
                  "word_gap_ms": 60, "phrase_pause_s": 0.175, "sentence_pause_s": 0.3},
    },
    "warm": {
        "id": "warm", "label": "Warm, conversational",
        "description": "Natural pace, close and friendly, little extra space.",
        "style": {"exaggeration": 0.55, "cfg_weight": 0.5, "temperature": 0.8, "pace": 1.0,
                  "word_gap_ms": 20, "phrase_pause_s": 0.075, "sentence_pause_s": 0.15},
    },
    "eerie": {
        "id": "eerie", "label": "Slow and eerie",
        "description": "Deliberate and unsettled, long pauses between phrases and sentences.",
        "style": {"exaggeration": 0.4, "cfg_weight": 0.3, "temperature": 0.6, "pace": 0.85,
                  "word_gap_ms": 80, "phrase_pause_s": 0.25, "sentence_pause_s": 0.4},
    },
    "dramatic": {
        "id": "dramatic", "label": "Dramatic",
        "description": "Raised stakes, weighted pauses ahead of the turn.",
        "style": {"exaggeration": 0.8, "cfg_weight": 0.4, "temperature": 0.8, "pace": 0.95,
                  "word_gap_ms": 40, "phrase_pause_s": 0.15, "sentence_pause_s": 0.3},
    },
    "intense": {
        "id": "intense", "label": "Intense, urgent",
        "description": "Fast and charged, almost no breathing room.",
        "style": {"exaggeration": 1.0, "cfg_weight": 0.5, "temperature": 0.8, "pace": 1.05,
                  "word_gap_ms": 0, "phrase_pause_s": 0.05, "sentence_pause_s": 0.125},
    },
    "horror_narrator": {
        "id": "horror_narrator", "label": "Horror / detective narrator",
        "description": "Slow, grave, low-key: the voice of a case file read back in the dark.",
        # Tempo picked by the creator 2026-10-24 ("faster_b"): 0.85 with 0.6 s sentence pauses
        # plus the director's pause cues read "very slow" in a 90 s reel; 0.93 then needed to be
        # "slightly slower" (same day), so 0.90 with a little more air between words and lines.
        "style": {"exaggeration": 0.3, "cfg_weight": 0.25, "temperature": 0.6, "pace": 0.90,
                  "word_gap_ms": 50, "phrase_pause_s": 0.25, "sentence_pause_s": 0.5},
    },
}


def list_moods() -> List[Dict[str, Any]]:
    """`GET /api/narration/moods` payload: the presets in table order."""
    return [dict(mood) for mood in MOODS.values()]


def resolve_style(style: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolution: engine defaults <- mood preset (default `DEFAULT_MOOD`) <- explicit keys.
    Unknown style keys are ignored (the caller validates ranges and the mood id); an unknown
    mood raises ValueError. The result always carries `mood` plus all of `STYLE_DEFAULTS`."""
    style = dict(style or {})
    mood = str(style.get("mood") or DEFAULT_MOOD).strip().lower()
    if mood not in MOODS:
        raise ValueError(f"Unknown mood {mood!r}: one of {', '.join(MOODS)}")
    resolved = dict(STYLE_DEFAULTS)
    resolved.update(MOODS[mood]["style"])
    for key in STYLE_DEFAULTS:
        if key in style:
            resolved[key] = float(style[key])
    resolved["mood"] = mood
    return resolved


# --------------------------------------------------------------------------- audio joining

def silence(seconds: float, sr: int = SAMPLE_RATE, xfade_s: float = XFADE_S) -> Any:
    """A cue's silence, padded by both crossfades so the audible gap is `seconds`."""
    import numpy as np
    return np.zeros(int(round((seconds + 2 * xfade_s) * sr)), dtype=np.float32)


def join_audio(chunks: Sequence[Any], sr: int = SAMPLE_RATE, xfade_s: float = XFADE_S) -> Any:
    """Concatenate mono float chunks with linear crossfades of `xfade_s` (shorter when a chunk is)."""
    import numpy as np
    out = np.zeros(0, dtype=np.float32)
    n = int(round(xfade_s * sr))
    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            continue
        k = min(n, out.size, chunk.size)
        if k <= 0:
            out = np.concatenate([out, chunk])
            continue
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        mixed = out[-k:] * (1.0 - ramp) + chunk[:k] * ramp
        out = np.concatenate([out[:-k], mixed, chunk[k:]])
    return out


def _quiet_frames(audio: Any, sr: int) -> "tuple[int, Any]":
    """10 ms RMS envelope -> `(win, quiet)`: `quiet[i]` is True when frame `i` is below (this
    audio's peak dB - `SPACE_QUIET_REL_DB`) and below `SPACE_QUIET_ABS_DBFS`. Shared by
    `space_out`/`trim_edges`/`trim_babble`. `quiet` is empty when there are fewer than 3 frames."""
    import numpy as np
    win = max(1, int(round(SPACE_WIN_S * sr)))
    n_frames = audio.size // win
    if n_frames < 3:
        return win, np.zeros(0, dtype=bool)
    frames = audio[:n_frames * win].reshape(n_frames, win).astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak_db = 20.0 * np.log10(max(float(np.max(np.abs(audio))), 1e-9))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return win, (rms_db < peak_db - SPACE_QUIET_REL_DB) & (rms_db < SPACE_QUIET_ABS_DBFS)


def _quiet_runs(quiet: Any) -> List["tuple[int, int]"]:
    """`quiet` (a bool array of frames) -> `[(start, end), ...]` of its contiguous True runs."""
    runs: List["tuple[int, int]"] = []
    i = 0
    n = len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def space_out(audio: Any, sr: int = SAMPLE_RATE, word_gap_ms: float = 0.0,
              phrase_pause_s: float = 0.0, sentence_pause_s: float = 0.0) -> Any:
    """Widen the quiet gaps already inside one spoken batch (sentences packed together are not
    split into their own batches any more), before batches are joined: `_quiet_frames` finds the
    runs; a run touching either edge of the batch is a breath or the engine's own lead/tail
    silence, not a gap between words, and is left alone (`trim_edges` handles those). A run of
    `SPACE_WORD_GAP_MS` (60-150 ms) is a word gap: `word_gap_ms` is inserted at its middle. A run
    in `SPACE_PHRASE_RANGE_MS` (150-350 ms) is a natural (comma-length) pause: `phrase_pause_s`
    instead. A longer run is a sentence-length pause: `sentence_pause_s` instead. Every resulting
    gap (the run plus whatever is inserted) is capped at `SPACE_MAX_GAP_S`. Pure numpy; all three
    `<= 0` returns `audio` unchanged. Returns `(audio, word_gaps, phrase_gaps, sentence_gaps)`."""
    if word_gap_ms <= 0 and phrase_pause_s <= 0 and sentence_pause_s <= 0:
        return audio, 0, 0, 0
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio, 0, 0, 0

    win, quiet = _quiet_frames(audio, sr)
    n_frames = len(quiet)
    if n_frames == 0:
        return audio, 0, 0, 0
    runs = _quiet_runs(quiet)

    pieces: List[Any] = []
    cursor = 0
    word_gaps = 0
    phrase_gaps = 0
    sentence_gaps = 0
    word_lo, word_hi = SPACE_WORD_GAP_MS
    phrase_lo, phrase_hi = SPACE_PHRASE_RANGE_MS
    for start, end in runs:
        if start == 0 or end == n_frames:                      # touches an edge: leave alone
            continue
        duration_ms = (end - start) * win / sr * 1000.0
        if word_lo <= duration_ms <= word_hi and word_gap_ms > 0:
            extra_s, kind = word_gap_ms / 1000.0, "word"
        elif phrase_lo < duration_ms <= phrase_hi and phrase_pause_s > 0:
            extra_s, kind = phrase_pause_s, "phrase"
        elif duration_ms > phrase_hi and sentence_pause_s > 0:
            extra_s, kind = sentence_pause_s, "sentence"
        else:
            continue
        extra_s = min(extra_s, SPACE_MAX_GAP_S - duration_ms / 1000.0)   # cap the resulting gap
        if extra_s <= 0:
            continue
        if kind == "word":
            word_gaps += 1
        elif kind == "phrase":
            phrase_gaps += 1
        else:
            sentence_gaps += 1
        mid = ((start + end) * win) // 2
        pieces.append(audio[cursor:mid])
        pieces.append(silence(extra_s, sr=sr, xfade_s=SPACE_XFADE_S))
        cursor = mid
    if word_gaps == 0 and phrase_gaps == 0 and sentence_gaps == 0:
        return audio, 0, 0, 0
    pieces.append(audio[cursor:])
    return join_audio(pieces, sr=sr, xfade_s=SPACE_XFADE_S), word_gaps, phrase_gaps, sentence_gaps


def trim_edges(audio: Any, sr: int = SAMPLE_RATE, target_s: float = SPACE_EDGE_TARGET_S) -> Any:
    """Trim a generated batch's own leading and trailing quiet (the engine's natural lead-in and
    tail-off) down to `target_s`, so it cannot stack with the silence inserted between batches
    (`insert_sentence_pauses`) -- the defect was each batch keeping its full lead/trail silence
    with a sentence/phrase pause then added on top. Same detector as `space_out`
    (`_quiet_frames`); a batch too short to measure, or with no edge quiet, comes back unchanged."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    win, quiet = _quiet_frames(audio, sr)
    n_frames = len(quiet)
    if n_frames == 0:
        return audio

    lead = 0
    while lead < n_frames and quiet[lead]:
        lead += 1
    trail = 0
    while trail < n_frames - lead and quiet[n_frames - 1 - trail]:
        trail += 1

    target_frames = max(0, int(round(target_s * sr / win)))
    cut_lead = max(0, lead - target_frames) * win
    cut_trail = max(0, trail - target_frames) * win
    end = audio.size - cut_trail
    if cut_lead >= end:
        return audio
    return audio[cut_lead:end]


def spoken_char_count(text: str) -> int:
    """How many characters of `text` are actually said: everything except whitespace and
    punctuation (Unicode category Z*/P*). Devanagari matras and the nukta count -- they are part
    of a syllable's sound, unlike `str.isalpha()` which excludes them."""
    return sum(1 for ch in text if unicodedata.category(ch)[0] not in ("Z", "P"))


def trim_babble(audio: Any, sr: int = SAMPLE_RATE, expected_s: float = 0.0) -> "tuple[Any, bool]":
    """The babble guard: Chatterbox keeps talking past the end of a short batch. If `audio` runs
    longer than `BABBLE_OVER_FACTOR * expected_s + BABBLE_OVER_MARGIN_S`, cut after the *last*
    quiet run of at least `BABBLE_CUT_MIN_GAP_MS` whose start falls past
    `BABBLE_CUT_MIN_FRACTION * expected_s` (keeping the part before it, including that pause) --
    real speech is expected to be mostly done by then, so a late, sentence-length silence is
    where the babble starts. Returns `(audio, cut)`; `cut` is False when `expected_s <= 0`, the
    audio was not over length, or no such quiet run exists -- the caller then decides whether to
    regenerate instead. Pure numpy; unchanged (`cut=False`) returns the same object, not a copy."""
    if expected_s <= 0 or audio.size == 0:
        return audio, False
    limit_s = BABBLE_OVER_FACTOR * expected_s + BABBLE_OVER_MARGIN_S
    if audio.size / sr <= limit_s:
        return audio, False

    import numpy as np
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    win, quiet = _quiet_frames(flat, sr)
    n_frames = len(quiet)
    if n_frames == 0:
        return audio, False
    min_gap_frames = max(1, int(round(BABBLE_CUT_MIN_GAP_MS / 1000.0 * sr / win)))
    min_start_frame = (BABBLE_CUT_MIN_FRACTION * expected_s * sr) / win

    best_end = None
    for start, end in _quiet_runs(quiet):
        if (end - start) >= min_gap_frames and start >= min_start_frame:
            best_end = end                                      # keep the last qualifying run
    if best_end is None:
        return audio, False
    return flat[:best_end * win], True


def batch_ok(audio: Any, sr: int = SAMPLE_RATE) -> bool:
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return audio.size >= int(MIN_BATCH_S * sr) and bool(np.isfinite(audio).all())


# --------------------------------------------------------------------------- narration verification (Whisper, E4)
# Chatterbox is checked against what it actually said, not its length: every batch is
# transcribed (word timestamps) and scored against its own script (`score_transcript`); a batch
# that drops words is regenerated, one that babbles past the end is cut at the real last word,
# never inside matched speech (`synthesize`). Pure scoring/decision functions take plain dicts
# (`{"word", "start", "end"}`), so they are unit-tested without loading any model.

_verifier_state: Dict[str, Any] = {}


def load_whisper_verifier() -> "tuple[Optional[Any], str]":
    """The Whisper model that verifies a chatterbox batch, loaded once per process (module-level
    cache, mirrors `cli_module()`'s own cache on the backend side). Tries `WHISPER_ATTEMPTS` in
    order -- GPU fp16, a lighter GPU quant, then CPU on a smaller model -- and keeps the first
    that loads. Returns `(model, descriptor)`; `model` is `None` when none of them did, and
    `descriptor` then starts with `"unavailable: "` (the caller falls back to the length guard)."""
    if "model" in _verifier_state:
        return _verifier_state["model"], _verifier_state["descriptor"]
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        _verifier_state["model"] = None
        _verifier_state["descriptor"] = f"unavailable: faster-whisper not installed ({exc})"
        return None, _verifier_state["descriptor"]
    errors: List[str] = []
    for model_name, device, compute_type in WHISPER_ATTEMPTS:
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:  # noqa: BLE001 -- tried in order, reported if every one fails
            errors.append(f"{model_name}/{device}/{compute_type}: {type(exc).__name__}: {exc}")
            continue
        _verifier_state["model"] = model
        _verifier_state["descriptor"] = f"{model_name}/{device}/{compute_type}"
        return model, _verifier_state["descriptor"]
    _verifier_state["model"] = None
    _verifier_state["descriptor"] = "unavailable: " + "; ".join(errors)
    return None, _verifier_state["descriptor"]


def unload_whisper_verifier() -> None:
    """Free the verifier's VRAM at the end of a synthesis run; a no-op when it was never loaded."""
    model = _verifier_state.pop("model", None)
    _verifier_state.pop("descriptor", None)
    if model is None:
        return
    del model
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def transcribe_words(model: Any, audio: Any, sr: int, language: str) -> List[Dict[str, Any]]:
    """`audio` (mono float32 at `sr`) -> Whisper's own words in order, each `{"word", "start",
    "end"}` (seconds, same timeline regardless of sample rate). faster-whisper never resamples a
    raw array itself (only a file path), so this does: it expects `WHISPER_SR`."""
    import numpy as np
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr != WHISPER_SR:
        import librosa
        flat = librosa.resample(flat, orig_sr=sr, target_sr=WHISPER_SR).astype(np.float32)
    segments, _ = model.transcribe(flat, language=language, word_timestamps=True,
                                   vad_filter=False, beam_size=WHISPER_BEAM_SIZE)
    words: List[Dict[str, Any]] = []
    for segment in segments:
        for w in (segment.words or []):
            words.append({"word": str(w.word), "start": float(w.start), "end": float(w.end)})
    return words


def _verify_normalize(text: str) -> str:
    """One word, comparable: Unicode whitespace/punctuation dropped (`spoken_char_count`'s own
    Z*/P* rule -- cue marks never reach here anyway, they are stripped by `parse_cues`), the
    nukta combining mark dropped and chandrabindu folded to anusvara (Whisper is inconsistent
    about marking a Devanagari word's nasal/retroflex the creator always says the same way),
    Latin lower-cased."""
    out = []
    for ch in unicodedata.normalize("NFC", text):
        if unicodedata.category(ch)[0] in ("Z", "P"):
            continue
        if ch == _VERIFY_NUKTA:
            continue
        if ch == _VERIFY_CHANDRABINDU:
            ch = _VERIFY_ANUSVARA
        out.append(ch.lower())
    return "".join(out)


def verify_words(text: str) -> List[str]:
    """`text` (already cue-free) as the word list `score_transcript` compares against Whisper's
    own: `_verify_normalize` per whitespace-split word, empties (pure punctuation) dropped."""
    return [w for w in (_verify_normalize(word) for word in text.split()) if w]


def _words_fuzzy_equal(a: str, b: str) -> bool:
    """Same word at `VERIFY_FUZZY_RATIO`+ char-ratio (Whisper misspells Hindi: खामुशी/ख़ामोशी,
    तीम/तीन) -- exact match is the common case and short-circuits the ratio call."""
    if a == b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= VERIFY_FUZZY_RATIO


def _align_words(script_words: Sequence[str], heard_words: Sequence[str]) -> List[Optional[int]]:
    """For each `script_words[i]`, the index into `heard_words` it matches, or `None`: a
    longest-common-subsequence alignment under `_words_fuzzy_equal` (classic LCS DP -- batches
    are at most a couple of hundred words, so O(n*m) is fine), so a match always respects order
    and never reuses a word on either side."""
    n, m = len(script_words), len(heard_words)
    if n == 0 or m == 0:
        return [None] * n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, next_row = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            if _words_fuzzy_equal(script_words[i], heard_words[j]):
                row[j] = next_row[j + 1] + 1
            else:
                row[j] = max(next_row[j], row[j + 1])
    alignment: List[Optional[int]] = [None] * n
    i = j = 0
    while i < n and j < m:
        if _words_fuzzy_equal(script_words[i], heard_words[j]):
            alignment[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return alignment


def score_transcript(script_words: Sequence[str], heard_words: Sequence[Dict[str, Any]]
                     ) -> Dict[str, Any]:
    """Score one Whisper transcript (`heard_words`: `{"word", "start", "end"}` in order, raw --
    not yet normalised) against a batch's own `script_words` (already `verify_words()`-normalised):
    `coverage` = the fraction of script words matched in order (`_align_words`, fuzzy); `extra` =
    how many heard words come after the last matched one; `cut_at_s` = that word's own end
    timestamp, or `None` when nothing matched at all (nothing safe to cut to)."""
    heard_pairs = [(_verify_normalize(str(w.get("word", ""))), w) for w in heard_words]
    heard_pairs = [pair for pair in heard_pairs if pair[0]]
    heard_norm = [norm for norm, _ in heard_pairs]
    alignment = _align_words(script_words, heard_norm)
    matched = [i for i in alignment if i is not None]
    coverage = (len(matched) / len(script_words)) if script_words else 1.0
    if matched:
        last = max(matched)
        extra = len(heard_norm) - last - 1
        cut_at_s = float(heard_pairs[last][1].get("end", 0.0) or 0.0)
    else:
        extra = len(heard_norm)
        cut_at_s = None
    return {"coverage": coverage, "extra": extra, "cut_at_s": cut_at_s}


def verify_decision(score: Dict[str, Any]) -> str:
    """`"accept"` / `"cut"` / `"regenerate"` from one `score_transcript()` result. Coverage below
    `VERIFY_MIN_COVERAGE` always means `"regenerate"` -- a cut cannot put missing words back.
    Coverage fine but more than `VERIFY_MAX_EXTRA_WORDS` heard past the last matched word means
    `"cut"` (or `"regenerate"` on the rare case nothing matched to cut to, so `cut_at_s` is None
    even though `coverage` alone would read as fine -- an empty `script_words` making it 1.0)."""
    if score["coverage"] < VERIFY_MIN_COVERAGE:
        return "regenerate"
    if score["extra"] > VERIFY_MAX_EXTRA_WORDS:
        return "cut" if score["cut_at_s"] is not None else "regenerate"
    return "accept"


def cut_at_time(audio: Any, sr: int, cut_s: float) -> Any:
    """`audio` truncated at `cut_s` seconds (clamped to `[0, audio length]`); pure numpy."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    idx = max(0, min(audio.size, int(round(cut_s * sr))))
    return audio[:idx]


def fade_edges(audio: Any, sr: int = SAMPLE_RATE, fade_in_s: float = TAIL_FADE_IN_S,
               fade_out_s: float = TAIL_FADE_OUT_S) -> Any:
    """`audio` with a linear fade-in and a squared (fast-dying) fade-out, so a batch never starts
    or stops mid-waveform; each fade is capped at a quarter of the audio. Pure numpy."""
    import numpy as np
    audio = np.array(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    k_in = min(int(fade_in_s * sr), audio.size // 4)
    k_out = min(int(fade_out_s * sr), audio.size // 4)
    if k_in > 0:
        audio[:k_in] *= np.linspace(0.0, 1.0, k_in, dtype=np.float32)
    if k_out > 0:
        audio[-k_out:] *= np.linspace(1.0, 0.0, k_out, dtype=np.float32) ** 2
    return audio


# --------------------------------------------------------------------------- voices

def list_voices(voices_dir: Path = DEFAULT_VOICES_DIR) -> List[Dict[str, Any]]:
    voices: List[Dict[str, Any]] = []
    folder = Path(voices_dir)
    if folder.is_dir():
        for path in sorted(folder.glob("*.wav"), key=lambda p: p.stem.casefold()):
            voices.append({"id": path.stem, "kind": "reference", "path": str(path)})
    for language, presets in KOKORO_VOICES.items():
        for preset in presets:
            voices.append({"id": preset, "kind": "kokoro", "language": language})
    return voices


def kokoro_preset(voice: str) -> bool:
    return any(voice in presets for presets in KOKORO_VOICES.values())


def reference_path(voice: str, voices_dir: Path) -> Optional[Path]:
    """A reference clip by stem (case-insensitive) or by path; None when there is none."""
    direct = Path(voice)
    if direct.suffix.lower() == ".wav" and direct.is_file():
        return direct
    folder = Path(voices_dir)
    if folder.is_dir():
        for path in folder.glob("*.wav"):
            if path.stem.casefold() == voice.casefold():
                return path
    return None


# --------------------------------------------------------------------------- engines

class EngineUnavailable(RuntimeError):
    pass


def load_engine(engine: str, language: str, voice: str,
                voices_dir: Path) -> Callable[..., Any]:
    """Load the model once; return say(text, pace, exaggeration, cfg_weight, temperature=None)
    -> mono float32 at 24 kHz. exaggeration/cfg_weight/temperature are per call (a mood cue
    layers over the base style, batch by batch); Kokoro ignores them but honours pace."""
    import numpy as np
    if engine == "chatterbox":
        try:
            import torch
            import torchaudio
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as exc:
            raise EngineUnavailable(f"chatterbox unavailable: {exc}") from exc
        ref: Optional[Path] = None
        if voice != ENGINE_VOICE:
            ref = reference_path(voice, voices_dir)
            if ref is None:
                raise ValueError(f"No reference voice {voice!r} in {voices_dir}")
        model = ChatterboxMultilingualTTS.from_pretrained(
            device="cuda" if torch.cuda.is_available() else "cpu")

        def say(text: str, pace: float, exaggeration: float, cfg_weight: float,
                temperature: Optional[float] = None) -> Any:
            kwargs: Dict[str, Any] = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if ref is not None:
                kwargs["audio_prompt_path"] = str(ref)
            wav = model.generate(text, language_id=language, **kwargs)
            if int(model.sr) != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, int(model.sr), SAMPLE_RATE)
            audio = wav.detach().reshape(-1).cpu().numpy().astype(np.float32)
            if abs(pace - 1.0) > 0.01:
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=float(pace)).astype(np.float32)
            return audio
        return say

    if engine == "kokoro":
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise EngineUnavailable(f"kokoro unavailable: {exc}") from exc
        preset = KOKORO_DEFAULT[language] if voice == ENGINE_VOICE else voice
        if not kokoro_preset(preset):
            raise ValueError(f"{preset!r} is not a Kokoro preset")
        pipe = KPipeline(lang_code=preset[0])

        def say(text: str, pace: float, exaggeration: float, cfg_weight: float,
                temperature: Optional[float] = None) -> Any:
            # Kokoro has no exaggeration/cfg_weight/temperature controls; pace still applies.
            parts = []
            for _, _, audio in pipe(text, voice=preset, speed=float(pace)):
                if audio is None:
                    continue
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return say

    raise ValueError(f"Unknown engine {engine!r}")


def synthesize(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import soundfile as sf

    t0 = time.time()
    text = Path(args.text_file).read_text(encoding="utf-8")
    if args.style:
        style = resolve_style(json.loads(args.style))
    else:
        # Backward compatible: no --style at all behaves exactly as before it existed.
        style = {"mood": None, "exaggeration": float(args.exaggeration), "cfg_weight": CFG_WEIGHT,
                 "temperature": None, "pace": float(args.pace), "word_gap_ms": 0.0,
                 "phrase_pause_s": 0.0, "sentence_pause_s": 0.0}
    plan = plan_batches(text)
    plan = insert_sentence_pauses(plan, style["sentence_pause_s"])
    batches = [item for item in plan if "text" in item]
    if not batches:
        raise ValueError("Nothing to say: the text is empty once the cues are removed")
    for item in batches:
        item["spoken"] = transliterate_text(item["text"]) if args.language == "hi" else item["text"]

    emit_progress(0.02, f"loading {args.engine}")
    say = load_engine(args.engine, args.language, args.voice, Path(args.voices_dir))
    emit_progress(0.1, f"{args.engine} loaded in {time.time() - t0:.0f} s")

    # Whisper verification (VOICE_DELIVERY_PLAN.md E4): chatterbox hallucinates, Kokoro does not,
    # so only chatterbox is checked, lazily, once for the whole run.
    verifier_model: Optional[Any] = None
    verify_descriptor = "skipped: kokoro does not hallucinate"
    if args.engine == "chatterbox":
        verifier_model, verify_descriptor = load_whisper_verifier()
        emit_progress(0.1, f"verify: {verify_descriptor}")

    chunks: List[Any] = []
    retries = 0
    done = 0
    word_gaps = 0
    phrase_gaps = 0
    sentence_gaps = 0
    verify_batches = 0
    verify_regenerated = 0
    verify_cut = 0
    verify_min_coverage: Optional[float] = None
    verify_flagged: List[int] = []
    try:
        for item in plan:
            if "silence" in item:
                chunks.append(silence(float(item["silence"])))
                continue
            pace = style["pace"] * (SLOW_FACTOR if item.get("slow") else 1.0)
            exaggeration = style["exaggeration"]
            cfg_weight = style["cfg_weight"]
            cue = MOOD_CUES.get(item.get("mood") or "")
            if cue:
                exaggeration = cue["exaggeration"]
                cfg_weight = cue["cfg_weight"]
                pace *= cue["pace_factor"]

            def _generate(seed: Optional[int] = None) -> Any:
                if seed is not None and args.engine == "chatterbox":
                    try:
                        import torch
                        torch.manual_seed(seed)
                    except Exception:
                        pass
                return say(item["spoken"], pace, exaggeration, cfg_weight, style["temperature"])

            audio = None
            for attempt in (1, 2):
                try:
                    audio = _generate()
                    if batch_ok(audio):
                        break
                    error = "empty audio"
                except Exception as exc:  # noqa: BLE001 -- retried once, then reported
                    error = f"{type(exc).__name__}: {exc}"
                audio = None
                if attempt == 1:
                    retries += 1
                    emit_progress(0.1 + 0.85 * done / len(batches), f"batch {done + 1} failed ({error}); retrying")
                else:
                    raise RuntimeError(f"batch {done + 1} failed twice: {error}")

            audio = trim_edges(audio, SAMPLE_RATE)

            if verifier_model is not None:
                # Verified against what Whisper actually heard, not length: a batch that drops
                # words is regenerated (up to VERIFY_REGEN_ATTEMPTS extra tries, keeping the best
                # by coverage then fewest extra words); one that babbles past the real last word
                # is cut there, never inside matched speech.
                script_words = verify_words(item["spoken"])
                score = score_transcript(
                    script_words, transcribe_words(verifier_model, audio, SAMPLE_RATE, args.language))
                attempts = [(audio, score)]
                if score["coverage"] < VERIFY_MIN_COVERAGE:
                    for seed in range(1, VERIFY_REGEN_ATTEMPTS + 1):
                        emit_progress(
                            0.1 + 0.85 * done / len(batches),
                            f"batch {done + 1}: regenerating (coverage {attempts[-1][1]['coverage']:.2f})")
                        try:
                            regen = _generate(seed=2000 + seed)
                        except Exception:
                            continue
                        if not batch_ok(regen):
                            continue
                        regen = trim_edges(regen, SAMPLE_RATE)
                        regen_score = score_transcript(
                            script_words,
                            transcribe_words(verifier_model, regen, SAMPLE_RATE, args.language))
                        attempts.append((regen, regen_score))
                        if regen_score["coverage"] >= VERIFY_MIN_COVERAGE:
                            break
                    verify_regenerated += 1
                audio, score = max(attempts, key=lambda pair: (pair[1]["coverage"], -pair[1]["extra"]))
                if score["extra"] > VERIFY_MAX_EXTRA_WORDS and score["cut_at_s"] is not None:
                    audio = trim_edges(
                        cut_at_time(audio, SAMPLE_RATE, score["cut_at_s"] + VERIFY_CUT_PAD_S), SAMPLE_RATE)
                    verify_cut += 1
                    emit_progress(0.1 + 0.85 * done / len(batches),
                                 f"batch {done + 1}: trimmed {score['extra']} unmatched word(s)")
                elif (score["cut_at_s"] is not None
                      and audio.size / SAMPLE_RATE > score["cut_at_s"] + TAIL_KEEP_S):
                    # No extra word, but a breath / hiss tail past the last word: cut it too.
                    audio = cut_at_time(audio, SAMPLE_RATE, score["cut_at_s"] + TAIL_KEEP_S)
                verify_batches += 1
                verify_min_coverage = (score["coverage"] if verify_min_coverage is None
                                       else min(verify_min_coverage, score["coverage"]))
                if score["coverage"] < VERIFY_MIN_COVERAGE:
                    verify_flagged.append(done + 1)
            elif args.engine == "chatterbox":
                # The verifier could not load: length-based fallback (VOICE_DELIVERY_PLAN.md E4,
                # V1's original guard) -- regenerate only, never cut (`trim_babble` unused here).
                expected_s = (spoken_char_count(item["spoken"]) / (BABBLE_RATE_CHARS_PER_S * pace)
                             if pace > 0 else 0.0)
                over = (expected_s > 0 and audio.size / SAMPLE_RATE
                       > BABBLE_OVER_FACTOR * expected_s + BABBLE_OVER_MARGIN_S)
                under = (expected_s >= BABBLE_UNDER_MIN_EXPECTED_S
                        and audio.size / SAMPLE_RATE < BABBLE_UNDER_FACTOR * expected_s)
                verify_batches += 1
                if over or under:
                    best, best_gap = audio, abs(audio.size / SAMPLE_RATE - expected_s)
                    for seed in range(1, BABBLE_REGEN_ATTEMPTS + 1):
                        try:
                            regen = _generate(seed=1000 + seed)
                        except Exception:
                            continue
                        if not batch_ok(regen):
                            continue
                        regen = trim_edges(regen, SAMPLE_RATE)
                        gap = abs(regen.size / SAMPLE_RATE - expected_s)
                        if gap < best_gap:
                            best, best_gap = regen, gap
                    audio = best
                    verify_regenerated += 1

            audio = fade_edges(audio, SAMPLE_RATE)
            audio, w, p, sg = space_out(audio, SAMPLE_RATE, style["word_gap_ms"],
                                        style["phrase_pause_s"], style["sentence_pause_s"])
            word_gaps += w
            phrase_gaps += p
            sentence_gaps += sg
            chunks.append(audio)
            done += 1
            emit_progress(0.1 + 0.85 * done / len(batches), f"batch {done}/{len(batches)}")
    finally:
        unload_whisper_verifier()

    audio = np.clip(join_audio(chunks), -1.0, 1.0)
    audio = np.concatenate([audio, np.zeros(int(END_SILENCE_S * SAMPLE_RATE), dtype=np.float32)])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
    emit_progress(1.0, "written")
    return {"wav": str(out), "duration_s": round(audio.size / SAMPLE_RATE, 3), "engine": args.engine,
            "voice": args.voice, "language": args.language, "batches": len(batches),
            "retries": retries, "transliterated_text": plan_text(plan),
            "style": style, "word_gaps": word_gaps, "phrase_gaps": phrase_gaps,
            "sentence_gaps": sentence_gaps,
            "verify": {"engine": verify_descriptor, "batches": verify_batches,
                      "regenerated": verify_regenerated, "cut": verify_cut,
                      "min_coverage": (round(verify_min_coverage, 3)
                                       if verify_min_coverage is not None else None),
                      "flagged": verify_flagged},
            "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- entry point

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local TTS narration (AUTO_STORY_PLAN.md N1)")
    commands = parser.add_subparsers(dest="command", required=True)
    voices = commands.add_parser("voices", help="reference clips and engine presets")
    voices.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR))
    synth = commands.add_parser("synthesize", help="text file -> 24 kHz mono WAV")
    synth.add_argument("--text-file", required=True)
    synth.add_argument("--engine", choices=ENGINES, default="chatterbox")
    synth.add_argument("--voice", default="sudhansu")
    synth.add_argument("--language", choices=LANGUAGES, default="hi")
    synth.add_argument("--exaggeration", type=float, default=0.5)
    synth.add_argument("--pace", type=float, default=1.0)
    synth.add_argument("--style", default=None,
                       help="JSON: mood, exaggeration, cfg_weight, temperature, pace, "
                            "word_gap_ms, phrase_pause_s, sentence_pause_s "
                            "(VOICE_DELIVERY_PLAN.md); overrides --exaggeration/--pace")
    synth.add_argument("--out", required=True)
    synth.add_argument("--voices-dir", default=str(DEFAULT_VOICES_DIR))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    try:
        if args.command == "voices":
            result: Dict[str, Any] = {"voices": list_voices(Path(args.voices_dir))}
        else:
            result = synthesize(args)
    except EngineUnavailable as exc:
        print(json.dumps({"error": str(exc), "unavailable": True}, ensure_ascii=False), flush=True)
        return EXIT_UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 -- the protocol reports it
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
        return EXIT_ERROR
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
