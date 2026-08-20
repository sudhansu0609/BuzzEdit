"""Stage A — the finished cut, as something the director can reason about.

Everything downstream needs two things the timeline does not offer directly:

**Time in the cut video, not in the source.** Words carry source timestamps, and
every removed fumble ripples the programme earlier — so a B-roll clip placed at a
word's raw timestamp would drift further out of sync with every cut before it.
The projection through the surviving V1 cuts already exists for captions
(`timeline.authoring.source_to_timeline_frame`) and is reused here.

**How emphatic each word is.** The auto-edit already measures loudness at 50Hz
and stores it on the timeline as `energy_envelope`, but nothing has ever read it
except the cut-placement nudge. Joining it to the word list gives a per-word
emphasis score for free, which is what decides where a punch-in lands.
"""

import logging
import math
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from timeline.authoring import _program_segments, source_to_timeline_frame
from timeline.schema import Timeline, frame_to_time

from .models import Program, ProgramSegment, ProgramWord

logger = logging.getLogger("presentation.program")

# Window used for the local speaking-rate curve. Long enough to be a rate rather
# than a single gap, short enough to track a change of pace.
RATE_WINDOW_SECONDS = 3.0
# Smoothing applied to the emphasis curve before peaks are picked. One word is
# noise; a run of loud words is emphasis.
SMOOTH_WORDS = 5


def build_program(timeline: Timeline) -> Program:
    """Project the surviving words onto the cut, and score how they were said."""
    fps = timeline.fps_num / max(1, timeline.fps_den)
    segments_raw = _program_segments(timeline)

    segments: List[ProgramSegment] = []
    for item in sorted(
        [i for i in timeline.items
         if i.track == "V1" and i.enabled and i.kind == "media"],
        key=lambda i: i.timeline_start_frame,
    ):
        segments.append(ProgramSegment(
            item_id=item.id,
            tl_start_s=frame_to_time(item.timeline_start_frame, timeline.fps_num, timeline.fps_den),
            tl_end_s=frame_to_time(item.timeline_end_frame, timeline.fps_num, timeline.fps_den),
            source_start_frame=item.source_start_frame,
            source_end_frame=item.source_end_frame,
            anchor_word_id=item.anchor_word_id,
        ))

    envelope = timeline.energy_envelope or {}
    db: List[Any] = envelope.get("db") or []
    rate = float(envelope.get("rate") or 0.0)
    has_energy = bool(db and rate > 0)

    words: List[ProgramWord] = []
    for word in timeline.words:
        if not word.enabled:
            continue
        tl_start_frame = source_to_timeline_frame(segments_raw, word.start_frame)
        if tl_start_frame is None:
            continue        # this word was cut out of the programme
        tl_end_frame = source_to_timeline_frame(segments_raw, max(word.start_frame,
                                                                  word.end_frame - 1))
        if tl_end_frame is None or tl_end_frame < tl_start_frame:
            tl_end_frame = tl_start_frame + max(1, word.end_frame - word.start_frame)

        words.append(ProgramWord(
            text=word.text,
            tl_start_s=frame_to_time(tl_start_frame, timeline.fps_num, timeline.fps_den),
            tl_end_s=frame_to_time(tl_end_frame + 1, timeline.fps_num, timeline.fps_den),
            source_start_frame=word.start_frame,
            db_mean=_word_loudness(word.start_frame, word.end_frame, fps, rate, db),
        ))

    words.sort(key=lambda w: w.tl_start_s)
    _score_emphasis(words)
    _score_rate(words)

    duration_s = frame_to_time(timeline.duration_frames, timeline.fps_num, timeline.fps_den)
    if not duration_s and segments:
        duration_s = segments[-1].tl_end_s

    program = Program(
        duration_s=duration_s,
        fps=fps,
        words=words,
        segments=segments,
        has_energy=has_energy,
    )
    logger.info("Programme: %.1fs, %d words, %d segments, energy=%s",
                duration_s, len(words), len(segments), has_energy)
    return program


def _word_loudness(start_frame: int, end_frame: int, fps: float,
                   rate: float, db: List[Any]) -> float:
    """Mean dB under a word.

    The envelope is indexed in *source* time at its own frame rate, and the word
    carries *source* frames — so both convert through seconds. Getting this wrong
    silently shifts every emphasis score by the length of the removed material.
    """
    if not db or rate <= 0 or fps <= 0:
        return 0.0
    first = int(start_frame / fps * rate)
    last = int(end_frame / fps * rate)
    window = [float(v) for v in db[first:max(first + 1, last)]]
    return mean(window) if window else 0.0


def _score_emphasis(words: List[ProgramWord]) -> None:
    """Loudness as a z-score, smoothed. Mutates the words in place."""
    if not words:
        return
    values = [w.db_mean for w in words]
    centre = mean(values)
    spread = pstdev(values) if len(values) > 1 else 0.0
    if spread <= 0.0:
        # A flat or missing envelope. Everything is equally emphatic, which the
        # zoom planner reads as "space the moves evenly" rather than failing.
        for word in words:
            word.emphasis_z = 0.0
        return

    raw = [(value - centre) / spread for value in values]
    half = SMOOTH_WORDS // 2
    for index, word in enumerate(words):
        lo = max(0, index - half)
        hi = min(len(raw), index + half + 1)
        # Hann weights: the word itself counts most, its neighbours support it.
        weights = [0.5 - 0.5 * math.cos(2 * math.pi * (i + 1) / (hi - lo + 1))
                   for i in range(hi - lo)]
        total = sum(weights) or 1.0
        word.emphasis_z = sum(r * w for r, w in zip(raw[lo:hi], weights)) / total


def _score_rate(words: List[ProgramWord]) -> None:
    """Local speaking rate in words per second. Mutates the words in place."""
    if not words:
        return
    starts = [w.tl_start_s for w in words]
    half = RATE_WINDOW_SECONDS / 2.0
    for index, word in enumerate(words):
        lo = word.tl_start_s - half
        hi = word.tl_start_s + half
        count = 0
        # Walk outwards from the word rather than scanning the whole list.
        for other in range(index, -1, -1):
            if starts[other] < lo:
                break
            count += 1
        for other in range(index + 1, len(starts)):
            if starts[other] > hi:
                break
            count += 1
        word.rate_wps = count / RATE_WINDOW_SECONDS


def intensity_at(program: Program, time_s: float) -> float:
    """Smoothed emphasis at a moment; 0.0 where nobody is speaking."""
    best: Optional[ProgramWord] = None
    for word in program.words:
        if word.tl_start_s <= time_s < word.tl_end_s:
            return word.emphasis_z
        if word.tl_start_s > time_s:
            break
        best = word
    return best.emphasis_z if best is not None else 0.0


def transcript_lines(program: Program, pause_seconds: float = 2.5) -> str:
    """The programme as timestamped lines, for the language model to read.

    Grouped on the same long-pause rule the grammar audit uses, so a "line" is
    roughly a sentence and the timestamps the model quotes back land on sentence
    boundaries rather than mid-clause.
    """
    lines: List[str] = []
    current: List[str] = []
    started = 0.0
    previous_end: Optional[float] = None

    for word in program.words:
        if previous_end is not None and word.tl_start_s - previous_end >= pause_seconds:
            if current:
                lines.append(f"[{started:.1f}] {' '.join(current)}")
            current = []
        if not current:
            started = word.tl_start_s
        current.append(word.text)
        previous_end = word.tl_end_s
        if word.text.strip().endswith((".", "?", "!")) and len(current) >= 4:
            lines.append(f"[{started:.1f}] {' '.join(current)}")
            current = []

    if current:
        lines.append(f"[{started:.1f}] {' '.join(current)}")
    return "\n".join(lines)
