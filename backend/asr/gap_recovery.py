"""Gap recovery: speech the transcript never wrote down.

Whisper decodes a long recording window by window and quietly drops what it takes for
repetition or noise. A passage said twice keeps one telling, a restart vanishes, and the
forced aligner — which fits the whole transcript to the whole recording in one pass —
then stretches the neighbouring words across the hole. Measured on 2026-10-04 with
Silero VAD: 13% of the speech in a 28.5-minute Raat3Baje recording had no word on it.
An editor cannot cut, or keep, what it cannot see; and the speech Whisper skips is
disproportionately the material an edit removes, because repeated takes are exactly
what its repetition guards suppress.

So after the main pass and the alignment: find the speech no aligned word covers, decode
only those stretches again (faster-whisper's `clip_timestamps` — same model, same
settings, a little context either side), keep the words that land inside the holes, and
let the caller align everything again together. What still has no words is reported, and
the editor sees it as a speech region without text rather than as nothing.

Speech presence comes from Silero VAD (bundled with faster-whisper), a learned detector:
the energy threshold in `vad.py` reads room noise as speech on some recordings.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("gap_recovery")

SAMPLE_RATE = 16000
GAP_MIN_SECONDS = 0.4          # shorter uncovered speech is a word edge, not a skipped phrase
CLIP_CONTEXT_SECONDS = 0.25    # decoded either side of a gap so its first word is whole
STRETCH_CAP_SECONDS = 1.5      # a word claiming longer than this covers only this much
INSIDE_TOLERANCE = 0.1         # a recovered word's midpoint may sit this far outside its gap
MIN_PROBABILITY = 0.3          # below this a recovered word is more likely noise than speech
DUPLICATE_OVERLAP = 0.5        # a recovered word this much inside an existing one is that word

Interval = Tuple[float, float]


def speech_regions(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> List[Interval]:
    """Where the recording has speech, by Silero VAD, in seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(threshold=0.5, min_speech_duration_ms=200,
                         min_silence_duration_ms=250, speech_pad_ms=30)
    stamps = get_speech_timestamps(audio, options, sampling_rate=sample_rate)
    return [(s["start"] / sample_rate, s["end"] / sample_rate) for s in stamps]


def _covered(words: Sequence[Dict[str, Any]]) -> List[Interval]:
    spans = []
    for w in words:
        start = float(w.get("start") or 0.0)
        end = float(w.get("end") or start)
        if end > start:
            spans.append((start, min(end, start + STRETCH_CAP_SECONDS)))
    spans.sort()
    merged: List[Interval] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def uncovered(regions: Sequence[Interval], words: Sequence[Dict[str, Any]],
              min_gap: float = GAP_MIN_SECONDS) -> List[Interval]:
    """Stretches of speech that no word covers, at least `min_gap` long."""
    covered = _covered(words)
    gaps: List[Interval] = []
    k = 0
    for start, end in regions:
        cursor = start
        while k < len(covered) and covered[k][1] <= start:
            k += 1
        j = k
        while j < len(covered) and covered[j][0] < end:
            c0, c1 = covered[j]
            if c0 > cursor and c0 - cursor >= min_gap:
                gaps.append((cursor, c0))
            cursor = max(cursor, c1)
            j += 1
        if end - cursor >= min_gap:
            gaps.append((cursor, end))
    return gaps


def _clips(gaps: Sequence[Interval], duration: float) -> List[Interval]:
    """The gaps widened by a little context, overlapping ones merged."""
    clips: List[Interval] = []
    for start, end in gaps:
        a = max(0.0, start - CLIP_CONTEXT_SECONDS)
        b = min(duration, end + CLIP_CONTEXT_SECONDS)
        if clips and a <= clips[-1][1]:
            clips[-1] = (clips[-1][0], max(clips[-1][1], b))
        else:
            clips.append((a, b))
    return clips


def _inside(mid: float, gaps: Sequence[Interval]) -> bool:
    return any(a - INSIDE_TOLERANCE <= mid <= b + INSIDE_TOLERANCE for a, b in gaps)


def _duplicate(word: Dict[str, Any], existing: Sequence[Interval]) -> bool:
    start, end = float(word["start"]), float(word["end"])
    length = max(1e-3, end - start)
    return any(max(0.0, min(end, b) - max(start, a)) / length >= DUPLICATE_OVERLAP for a, b in existing)


def recover_gaps(
    model,
    audio_path: str,
    words: List[Dict[str, Any]],
    language: Optional[str],
    make_word: Callable[[Any], Dict[str, Any]],
    audio: Optional[np.ndarray] = None,
    initial_prompt: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (`words` plus the words recovered from the holes, a report).

    `make_word` turns a faster-whisper word into this pipeline's word dict (romanised the
    same way as the main pass). Recovered words carry `recovered: True`. Never raises.
    """
    report: Dict[str, Any] = {"gaps": 0, "gap_seconds": 0.0, "recovered_words": 0,
                              "still_uncovered_seconds": 0.0}
    try:
        if audio is None:
            from faster_whisper.audio import decode_audio
            audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        regions = speech_regions(audio)
        report["speech_seconds"] = round(sum(b - a for a, b in regions), 1)
        gaps = uncovered(regions, words)
        report["gaps"] = len(gaps)
        report["gap_seconds"] = round(sum(b - a for a, b in gaps), 1)
        report["gap_list"] = [(round(a, 3), round(b, 3)) for a, b in gaps]
        if not gaps:
            return words, report

        clips = _clips(gaps, len(audio) / SAMPLE_RATE)
        flat = [round(t, 3) for clip in clips for t in clip]
        segments, _info = model.transcribe(
            audio_path, language=language, word_timestamps=True, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0, initial_prompt=initial_prompt,
            clip_timestamps=flat)
        existing = _covered(words)
        found: List[Dict[str, Any]] = []
        for seg in segments:
            for w in (seg.words or []):
                if float(w.probability or 0.0) < MIN_PROBABILITY:
                    continue
                if not _inside((float(w.start) + float(w.end)) / 2, gaps):
                    continue
                word = make_word(w)
                if not str(word.get("word") or "").strip():
                    continue
                word["recovered"] = True
                if not _duplicate(word, existing):
                    found.append(word)
        report["recovered_words"] = len(found)
        merged = sorted(words + found, key=lambda w: float(w.get("start") or 0.0))
        report["still_uncovered_seconds"] = round(sum(b - a for a, b in uncovered(regions, merged)), 1)
        logger.info("Gap recovery: %d gaps (%.1fs of speech with no words) -> %d words recovered, "
                    "%.1fs still without words", report["gaps"], report["gap_seconds"],
                    len(found), report["still_uncovered_seconds"])
        return merged, report
    except Exception as e:  # noqa: BLE001 - the main transcript stands on its own
        logger.warning("Gap recovery skipped (%s); keeping the main pass.", e)
        report["error"] = str(e)
        return words, report
