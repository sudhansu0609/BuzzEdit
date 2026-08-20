"""Reconcile ASR word timings with what the audio actually contains.

Whisper's word timestamps are approximate, and when they go wrong they go wrong
in one particular way: a word absorbs the pause or the un-transcribed speech
around it. On a real 195-second recording from this project, 23 words claimed
over a second each and one claimed 10.13 seconds — 76 seconds of runtime, 39% of
the video, sealed inside "words".

That matters more than it sounds. The editor removes dead air by looking at the
gaps *between* words, so any silence a word has swallowed is invisible and can
never be cut. Clamping each word back to the speech it really covers is what
exposes that time — no new cutting logic required, the existing ripple does the
rest once the timings are honest.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .vad import SpeechMap

logger = logging.getLogger("align")

# A word covering less speech than this is mostly air; its timing is not to be
# trusted and it gets clamped hard.
_SUSPECT_SPEECH_FRACTION = 0.85
# Words shorter than this are left alone — clamping them risks clipping onsets.
_MIN_WORD_SECONDS = 0.08


@dataclass
class RepairReport:
    repaired: int = 0
    seconds_recovered: float = 0.0
    dropped: int = 0
    total: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "repaired": self.repaired,
            "seconds_recovered": round(self.seconds_recovered, 2),
            "dropped": self.dropped,
            "total": self.total,
        }


def _bounds(word: Dict[str, Any]) -> Tuple[float, float]:
    return float(word.get("start", 0.0) or 0.0), float(word.get("end", 0.0) or 0.0)


def repair_word_timings(
    words: List[Dict[str, Any]],
    speech_map: Optional[SpeechMap],
) -> Tuple[List[Dict[str, Any]], RepairReport]:
    """Clamp each word to the speech inside its span.

    A word is only touched when it is demonstrably padded — when most of what it
    covers is not speech. Words that already sit on speech are left exactly as
    they are, because Whisper's alignment is good most of the time and nudging
    every word would trade a real problem for a subtle one.
    """
    report = RepairReport(total=len(words))
    if not words or speech_map is None or not speech_map.speech:
        return words, report

    repaired: List[Dict[str, Any]] = []
    for word in words:
        start, end = _bounds(word)
        span = end - start
        if span <= _MIN_WORD_SECONDS:
            repaired.append(word)
            continue

        fraction = speech_map.speech_fraction(start, end)
        if fraction >= _SUSPECT_SPEECH_FRACTION:
            repaired.append(word)
            continue

        clamped = speech_map.clamp_to_speech(start, end)
        updated = dict(word)
        if clamped is None:
            # No speech at all under this word. Keep it — dropping a word would
            # silently delete transcript text — but shrink it to a point so it
            # stops holding a pause open.
            updated["end"] = start + _MIN_WORD_SECONDS
            updated["timing_suspect"] = True
            report.dropped += 1
            report.seconds_recovered += max(0.0, span - _MIN_WORD_SECONDS)
        else:
            new_start, new_end = clamped
            if new_end - new_start < _MIN_WORD_SECONDS:
                new_end = new_start + _MIN_WORD_SECONDS
            updated["start"], updated["end"] = round(new_start, 3), round(new_end, 3)
            updated["timing_repaired"] = True
            report.repaired += 1
            report.seconds_recovered += max(0.0, span - (new_end - new_start))
        repaired.append(updated)

    # Clamping can leave a word starting before the previous one ended; keep the
    # sequence monotonic so downstream frame maths never sees a negative gap.
    for previous, current in zip(repaired, repaired[1:]):
        if current.get("start", 0.0) < previous.get("end", 0.0):
            current["start"] = previous["end"]
            if current["end"] < current["start"] + _MIN_WORD_SECONDS:
                current["end"] = current["start"] + _MIN_WORD_SECONDS

    if report.repaired or report.dropped:
        logger.info("Word timing repair: %d clamped, %d emptied, %.1fs of hidden air recovered",
                    report.repaired, report.dropped, report.seconds_recovered)
    return repaired, report


def find_unvoiced_speech(
    words: List[Dict[str, Any]],
    speech_map: Optional[SpeechMap],
    min_duration: float = 0.18,
    max_duration: float = 2.5,
) -> List[Tuple[float, float]]:
    """Speech the transcript has no words for — the fillers Whisper threw away.

    Whisper is trained to produce clean, readable text, so it silently drops
    "um", "uh", stutters and false starts. That is exactly why matching filler
    words against the transcript finds almost nothing: on the real project it
    flagged 4 words out of 351. The sounds are still in the audio though, and
    they show up as speech regions that no word claims.

    Bounded at both ends on purpose: shorter than `min_duration` is a breath or a
    consonant tail, longer than `max_duration` is real speech the ASR missed
    rather than a filler, and deleting that would cut the user's content.
    """
    if speech_map is None or not speech_map.speech:
        return []

    spans = sorted((_bounds(w) for w in words), key=lambda b: b[0])
    unvoiced: List[Tuple[float, float]] = []

    for region_start, region_end in speech_map.speech:
        cursor = region_start
        for word_start, word_end in spans:
            if word_end <= cursor:
                continue
            if word_start >= region_end:
                break
            if word_start > cursor:
                unvoiced.append((cursor, min(word_start, region_end)))
            cursor = max(cursor, word_end)
            if cursor >= region_end:
                break
        if cursor < region_end:
            unvoiced.append((cursor, region_end))

    return [(round(s, 3), round(e, 3)) for s, e in unvoiced
            if min_duration <= (e - s) <= max_duration]
