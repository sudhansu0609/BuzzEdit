import logging
import re
from typing import List, Dict, Any
from models import TranscriptSegment

logger = logging.getLogger(__name__)

FUMBLE_PATTERNS = [
    r'\b(um|uh|uhuh|umm|uhh|hmm|like|so|you know|I mean|basically|actually|literally)\b',
    r'\b(\w+)\s+\1\b',
    r'(?:\w+\s+){0,3}\.\.\.',
    r'\b(well|let me see|let me think|how do I say this|what\'s the word)\b',
]

FUMBLE_KEYWORDS = [
    "um", "uh", "uhh", "umm", "hmm", "like", "you know",
    "I mean", "basically", "actually", "literally", "so",
    "well", "right", "okay", "alright",
]


def detect_fumbles(
    transcript_segments: List[TranscriptSegment],
    min_pause: float = 0.8,
    max_fumble_duration: float = 3.0,
) -> List[Dict[str, Any]]:
    fumble_segments = []

    for seg in transcript_segments:
        if not seg.text:
            continue

        detected = _analyze_segment(seg)
        if detected:
            for abs_start, abs_end, reason, confidence in detected:
                duration = abs_end - abs_start

                if 0.1 <= duration <= max_fumble_duration:
                    fumble_segments.append({
                        "start": round(abs_start, 2),
                        "end": round(abs_end, 2),
                        "duration": round(duration, 2),
                        "confidence": round(confidence, 2),
                        "reason": reason,
                    })

    if transcript_segments:
        for i in range(len(transcript_segments) - 1):
            gap = transcript_segments[i + 1].start - transcript_segments[i].end
            if min_pause <= gap <= 2.5:
                text = transcript_segments[i].text.lower()
                if any(kw in text for kw in ["um", "uh", "like", "so", "I mean"]):
                    fumble_segments.append({
                        "start": round(transcript_segments[i].end, 2),
                        "end": round(transcript_segments[i + 1].start, 2),
                        "duration": round(gap, 2),
                        "confidence": 0.6,
                        "reason": "mid-sentence pause with filler",
                    })

    fumble_segments.sort(key=lambda s: s["start"])
    fumble_segments = _merge_overlapping(fumble_segments)

    logger.info(f"Detected {len(fumble_segments)} fumble segments")
    return fumble_segments


def _analyze_segment(seg: TranscriptSegment) -> List[tuple]:
    results = []
    text = seg.text
    duration = seg.end - seg.start

    if seg.words and len(seg.words) > 0:
        for i in range(len(seg.words)):
            word = seg.words[i]
            w_text = word.get("word", "").lower().strip(".,!?;:")

            if w_text in FUMBLE_KEYWORDS:
                w_start = word.get("start", seg.start)
                w_end = word.get("end", seg.end)
                results.append((
                    round(w_start, 2),
                    round(w_end, 2),
                    f"filler word: {w_text}",
                    0.7,
                ))

            if i > 0:
                prev_word = seg.words[i - 1]
                prev_text = prev_word.get("word", "").lower().strip(".,!?;:")
                if prev_text and w_text and w_text == prev_text and len(w_text) > 2:
                    w_start = prev_word.get("start", seg.start)
                    w_end = word.get("end", seg.end)
                    results.append((
                        round(w_start, 2),
                        round(w_end, 2),
                        f"repeated word: {w_text}",
                        0.8,
                    ))

            if w_text and len(w_text) > 1:
                conf = word.get("confidence", 1.0)
                if conf < 0.3:
                    w_start = word.get("start", seg.start)
                    w_end = word.get("end", seg.end)
                    results.append((
                        round(w_start, 2),
                        round(w_end, 2),
                        "low confidence recognition",
                        0.5,
                    ))

        for i in range(len(seg.words) - 1):
            curr_end = seg.words[i].get("end", 0)
            next_start = seg.words[i + 1].get("start", 0)
            pause = next_start - curr_end

            if 0.5 < pause < 2.0:
                curr_text = seg.words[i].get("word", "").lower().strip(".,!?;:")
                if any(kw in curr_text for kw in ["um", "uh", "like", "so"]):
                    results.append((
                        round(curr_end, 2),
                        round(next_start, 2),
                        "filler pause",
                        0.6,
                    ))

    text_lower = text.lower()
    for pattern in FUMBLE_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            char_start = match.start()
            char_end = match.end()
            total_chars = len(text)
            if total_chars > 0 and duration > 0:
                time_start = seg.start + (char_start / total_chars) * duration
                time_end = seg.start + (char_end / total_chars) * duration
                results.append((
                    round(time_start, 2),
                    round(time_end, 2),
                    f"pattern: {match.group()}",
                    0.5,
                ))

    return results


def _merge_overlapping(segments: List[Dict[str, Any]], overlap_threshold: float = 0.3) -> List[Dict[str, Any]]:
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"] + overlap_threshold:
            last["end"] = max(last["end"], seg["end"])
            last["duration"] = round(last["end"] - last["start"], 2)
            last["confidence"] = max(last["confidence"], seg["confidence"])
        else:
            merged.append(seg.copy())

    return merged
