"""The editor's view of a recording: numbered utterances, with what the audio says.

The editor (AUTO_CUT_EDITOR_PLAN.md §4.4) reads the whole recording at once and decides by
take, not by word. The bare transcript can't give it three things, and this module adds them:

- **Every stretch of speech, including the ones with no words.** A region the VAD calls
  speech but no word covers becomes `[speech, no transcript, 2.4 s]` inside its utterance.
  The restarts Whisper skips are usually exactly what an edit removes, so they must be
  visible to be judged.
- **What the audio shows, in words:** how an utterance ends (cut off while still loud, fading,
  pitch falling or rising), whether it restarts with a pitch reset after the previous one,
  fillers, a drawn-out word, and speaking rate and loudness against the speaker's own norm.
  These are measurements written for a reader. Nothing here decides anything.
- **Which other utterances say the same thing, anywhere in the recording.** A retake can be a
  minute away and reworded; in a 300-line table it gets missed unless pointed at. Similarity
  is character-trigram cosine over the spelling-folded romanisation, so the same words spelled
  differently between takes, or in a different script, still meet.

Plus the script paragraph each utterance follows, when there is a script. That is a hint,
never a filter: in Raat3Baje ep1's final cut, 37% of what was said was ad-lib.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16000
PAUSE_SPLIT = 0.30             # seconds of silence that end an utterance
FRAME = 0.010                  # 10 ms analysis hop
SIMILAR_MIN = 0.50             # trigram cosine worth pointing the editor at (0.4 pointed at
                               # unrelated lines of the same story on Raat3Baje ep1)
SIMILAR_TOP = 3
SEMITONE_SHIFT = 2.0           # pitch movement worth a note
RESET_SEMITONES = 3.0          # a restart sounds like a pitch reset of at least this
LOUD_DB = 10.0                 # within this of the utterance's level counts as still sounding
SILENT_DB = 25.0               # this far below the level counts as gone
CUT_OFF_FRAMES = 4             # sound gone within 40 ms of its last loud frame: cut off
FADE_DB = 10.0                 # dropping at least this over the last 400 ms
DRAWN_OUT_S = 1.2              # a single word held this long
FILLERS = {"[uh]", "uh", "um", "umm", "hmm", "aa", "aaa", "उम", "अ", "आ", "हम्म"}


@dataclass
class Utterance:
    uid: str
    start: float
    end: float
    word_indices: List[int]
    text: str
    pause_before: float
    notes: List[str] = field(default_factory=list)
    similar: List[Tuple[str, float]] = field(default_factory=list)
    same_opening: List[str] = field(default_factory=list)
    script: Optional[str] = None
    untranscribed_s: float = 0.0


# --- grouping ----------------------------------------------------------------------

def build(words: Sequence[Dict[str, Any]], speech: Sequence[Tuple[float, float]] = (),
          audio: Optional[np.ndarray] = None, script_text: Optional[str] = None) -> List[Utterance]:
    """Utterances over `words` (dicts with start/end/word/word_native) and the speech
    regions no word covers; acoustic notes when `audio` (16 kHz mono) is given; script
    tags when `script_text` is."""
    from .gap_recovery import uncovered

    items: List[Tuple[float, float, Optional[int], str]] = []
    for index, w in enumerate(words):
        text = str(w.get("word_native") or w.get("word") or "").strip()
        if text:
            items.append((float(w["start"]), float(w["end"]), index, text))
    for start, end in uncovered(speech, words) if speech else []:
        items.append((start, end, None, f"[speech, no transcript, {end - start:.1f} s]"))
    items.sort(key=lambda it: (it[0], it[1]))
    if not items:
        return []

    groups: List[List[Tuple[float, float, Optional[int], str]]] = [[items[0]]]
    for item in items[1:]:
        if item[0] - max(i[1] for i in groups[-1]) >= PAUSE_SPLIT:
            groups.append([item])
        else:
            groups[-1].append(item)

    utterances: List[Utterance] = []
    previous_end = 0.0
    for n, group in enumerate(groups, 1):
        start = group[0][0]
        end = max(i[1] for i in group)
        utterances.append(Utterance(
            uid=f"U{n}", start=start, end=end,
            word_indices=[i[2] for i in group if i[2] is not None],
            text=" ".join(i[3] for i in group),
            pause_before=max(0.0, start - previous_end),
            untranscribed_s=sum(i[1] - i[0] for i in group if i[2] is None)))
        previous_end = end

    _similarity(utterances, words)
    if audio is not None:
        _acoustics(utterances, words, audio)
    else:
        _text_notes(utterances, words)
    if script_text:
        _script_tags(utterances, words, script_text)
    return utterances


# --- similarity --------------------------------------------------------------------

def _key(word: Dict[str, Any]) -> str:
    from .script_align import _word_keys
    roman, native = _word_keys(word)
    return roman or native


def _trigrams(text: str) -> Counter:
    padded = f"  {text}  "
    return Counter(padded[i:i + 3] for i in range(len(padded) - 2))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def _similarity(utterances: List[Utterance], words: Sequence[Dict[str, Any]]) -> None:
    keys = [[_key(words[i]) for i in u.word_indices if _key(words[i]) not in FILLERS] for u in utterances]
    grams = [_trigrams(" ".join(k)) if len(k) >= 2 else Counter() for k in keys]
    openings = [_trigrams(" ".join(k[:3])) if len(k) >= 3 else Counter() for k in keys]
    for i, u in enumerate(utterances):
        if not grams[i]:
            continue
        scored = []
        for j, v in enumerate(utterances):
            if i == j or not grams[j]:
                continue
            score = _cosine(grams[i], grams[j])
            if score >= SIMILAR_MIN:
                scored.append((v.uid, round(score, 2)))
            if openings[i] and openings[j] and _cosine(openings[i], openings[j]) >= 0.8:
                u.same_opening.append(v.uid)
        scored.sort(key=lambda s: -s[1])
        u.similar = scored[:SIMILAR_TOP]
        u.same_opening = u.same_opening[:SIMILAR_TOP]


# --- acoustics ---------------------------------------------------------------------

def _frame_db(audio: np.ndarray) -> np.ndarray:
    hop = int(FRAME * SAMPLE_RATE)
    n = len(audio) // hop
    frames = audio[:n * hop].reshape(n, hop).astype(np.float64)
    return 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-10)


def _pitch(segment: np.ndarray, max_frames: int = 60) -> float:
    """Median f0 in Hz over voiced 40 ms frames of `segment` (at most `max_frames` of them,
    spread evenly), or NaN when too little of it is voiced. Autocorrelation by FFT."""
    win, hop = int(0.04 * SAMPLE_RATE), int(0.01 * SAMPLE_RATE)
    lo, hi = SAMPLE_RATE // 400, SAMPLE_RATE // 75
    starts = np.arange(0, max(0, len(segment) - win), hop)
    if starts.size < 3:
        return float("nan")
    if starts.size > max_frames:
        starts = starts[np.linspace(0, starts.size - 1, max_frames).astype(int)]
    frames = np.stack([segment[s:s + win] for s in starts]).astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    energy = np.sum(frames * frames, axis=1)
    spectrum = np.fft.rfft(frames, n=2 * win, axis=1)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :win]
    lags = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(lags)), lags] / np.maximum(energy, 1e-12)
    voiced = (energy > 1e-6) & (strength > 0.45)
    if voiced.sum() < 3:
        return float("nan")
    return float(np.median(SAMPLE_RATE / lags[voiced]))


def _semitones(a: float, b: float) -> float:
    return 12.0 * math.log2(a / b) if a > 0 and b > 0 and not (math.isnan(a) or math.isnan(b)) else 0.0


def _word_notes(u: Utterance, words: Sequence[Dict[str, Any]]) -> List[str]:
    notes = []
    fillers = sum(1 for i in u.word_indices if _key(words[i]) in FILLERS
                  or str(words[i].get("word") or "").strip() in FILLERS)
    if fillers:
        notes.append(f"fillers ×{fillers}")
    longest = max((float(words[i]["end"]) - float(words[i]["start"]) for i in u.word_indices), default=0.0)
    if longest >= DRAWN_OUT_S:
        notes.append(f"a word held {longest:.1f} s")
    if u.untranscribed_s:
        notes.append(f"{u.untranscribed_s:.1f} s of speech the transcript missed")
    return notes


def _text_notes(utterances: List[Utterance], words: Sequence[Dict[str, Any]]) -> None:
    for u in utterances:
        u.notes = _word_notes(u, words)


def _acoustics(utterances: List[Utterance], words: Sequence[Dict[str, Any]], audio: np.ndarray) -> None:
    db = _frame_db(audio)

    def frames(a: float, b: float) -> np.ndarray:
        return db[max(0, int(a / FRAME)):max(0, int(b / FRAME))]

    levels = [float(np.median(frames(u.start, u.end))) for u in utterances if u.end - u.start > 0.2]
    median_level = float(np.median(levels)) if levels else 0.0
    rates = [len(u.word_indices) / (u.end - u.start) for u in utterances
             if u.word_indices and u.end - u.start > 1.0]
    median_rate = float(np.median(rates)) if rates else 0.0

    def cut_off(level: float, end: float) -> bool:
        """The sound stops dead: from its last loud frame to silence in under 40 ms.
        A finished sentence decays over 80-200 ms; a line abandoned mid-word does not."""
        window = frames(end - 0.25, end + 0.30)
        loud = np.nonzero(window >= level - LOUD_DB)[0]
        if not loud.size:
            return False
        after = window[loud[-1] + 1:]
        gone = np.nonzero(after <= level - SILENT_DB)[0]
        return bool(gone.size) and int(gone[0]) < CUT_OFF_FRAMES

    previous_tail_f0 = float("nan")
    previous_unfinished = False
    for u in utterances:
        notes = []
        unfinished = False
        body = frames(u.start, u.end)
        if body.size >= 20:
            level = float(np.median(body))
            last = frames(u.end - 0.40, u.end)
            if cut_off(level, u.end):
                notes.append("cut off mid-sound")
                unfinished = True
            elif last.size >= 20 and float(np.mean(last[:10])) - float(np.mean(last[-10:])) >= FADE_DB:
                notes.append("fades out")
            if level < median_level - 6.0:
                notes.append(f"quieter ({level - median_level:.0f} dB)")
        segment = lambda a, b: audio[int(max(0.0, a) * SAMPLE_RATE):int(b * SAMPLE_RATE)]
        f0_all = _pitch(segment(u.start, u.end)) if u.end - u.start >= 0.5 else float("nan")
        f0_head = _pitch(segment(u.start, min(u.end, u.start + 0.35)))
        f0_tail = _pitch(segment(max(u.start, u.end - 0.35), u.end))
        # A falling end is how a statement normally finishes, so only the unusual endings
        # are worth a note: level or rising, the sound of a line left unfinished.
        shift = _semitones(f0_tail, f0_all)
        if not math.isnan(f0_tail) and shift > -SEMITONE_SHIFT / 2:
            notes.append("ends on a rising pitch" if shift >= SEMITONE_SHIFT else "ends on a level pitch")
            unfinished = True
        # A pitch jump at the start is the normal reset of a new sentence; it marks a
        # restart only after a line that was left unfinished.
        reset = _semitones(f0_head, previous_tail_f0)
        if previous_unfinished and reset >= RESET_SEMITONES and u.pause_before < 3.0:
            notes.append(f"restarts {reset:.0f} semitones higher after an unfinished line")
        if median_rate and u.word_indices and u.end - u.start > 1.0:
            rate = len(u.word_indices) / (u.end - u.start)
            if rate > 1.3 * median_rate:
                notes.append("rushed")
            elif rate < 0.7 * median_rate:
                notes.append("slow")
        u.notes = notes + _word_notes(u, words)
        if not math.isnan(f0_tail):
            previous_tail_f0 = f0_tail
        previous_unfinished = unfinished


# --- script ------------------------------------------------------------------------

def _script_tags(utterances: List[Utterance], words: Sequence[Dict[str, Any]], script_text: str) -> None:
    from .script_align import align, flat_tokens, parse_script

    parsed = parse_script(script_text)
    tokens = flat_tokens(parsed)
    if not tokens:
        return
    pairs = dict((w, s) for s, w in align(tokens, list(words)))
    starts = [p["first_token"] for p in parsed["paragraphs"]]

    def paragraph_of(token: int) -> int:
        k = 0
        while k + 1 < len(starts) and starts[k + 1] <= token:
            k += 1
        return k

    for u in utterances:
        hits = [paragraph_of(pairs[i]) for i in u.word_indices if i in pairs]
        if not u.word_indices:
            continue
        if len(hits) < max(1, len(u.word_indices) // 4):
            u.script = "ad-lib"
            continue
        para, count = Counter(hits).most_common(1)[0]
        u.script = f"P{para + 1} ({100 * len(hits) // len(u.word_indices)}% matched)"


# --- the table the editor reads -----------------------------------------------------

def table(utterances: Sequence[Utterance]) -> str:
    lines = []
    for u in utterances:
        line = f"{u.uid} [{u.start:.1f}-{u.end:.1f}s, pause before {u.pause_before:.1f}s] {u.text}"
        extra = []
        if u.notes:
            extra.append("; ".join(u.notes))
        if u.similar:
            extra.append("similar to " + ", ".join(f"{uid} ({score:.2f})" for uid, score in u.similar))
        if u.same_opening:
            extra.append("same opening as " + ", ".join(u.same_opening))
        if u.script:
            extra.append(f"script {u.script}")
        if extra:
            line += "  || " + " | ".join(extra)
        lines.append(line)
    return "\n".join(lines)


_STUDIO_VO = re.compile(r"\*\*V\.O\.[^*]*\*\*\s*(.+)")


def spoken_script(text: str) -> str:
    """The lines a Studio narration script (`script/narration.md`) means to be *said*:
    the V.O. lines, minus stage notes like "(pause)". Any other text is returned as is."""
    spoken = [m.group(1) for m in (_STUDIO_VO.match(line.strip()) for line in (text or "").splitlines()) if m]
    if not spoken:
        return text or ""
    return "\n\n".join(re.sub(r"\([^)]*\)", " ", line).strip() for line in spoken)
