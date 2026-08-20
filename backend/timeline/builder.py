import uuid
from typing import Any, Dict, List, Optional, Tuple
from .schema import Timeline, SourceFile, WordItem, Transition, time_to_frame
from .ops import rebuild_primary_tracks

def build_timeline_from_transcript(
    source_path: str,
    duration_seconds: float,
    transcript_words: List[Dict[str, Any]],
    fps_num: int = 30,
    fps_den: int = 1,
    width: int = 1920,
    height: int = 1080,
    has_audio: bool = True,
    speech_regions: Optional[List[Tuple[float, float]]] = None,
    max_pause_seconds: Optional[float] = None,
    pause_padding_seconds: Optional[float] = None,
    energy_envelope: Optional[Dict[str, Any]] = None,
    default_transition: Optional[Transition] = None,
) -> Timeline:
    """
    Build an initial Timeline object from source video metadata and timestamped words.
    """
    source_id = f"src_main_{uuid.uuid4().hex[:6]}"
    source_file = SourceFile(
        id=source_id,
        path=source_path,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        fps_num=fps_num,
        fps_den=fps_den,
        has_audio=has_audio,
    )

    words: List[WordItem] = []
    for idx, w in enumerate(transcript_words):
        start_sec = float(w.get("start", 0.0))
        end_sec = float(w.get("end", 0.0))
        text = str(w.get("word", w.get("text", ""))).strip()
        if not text:
            continue
        
        start_f = time_to_frame(start_sec, fps_num, fps_den)
        end_f = time_to_frame(end_sec, fps_num, fps_den)
        if end_f <= start_f:
            end_f = start_f + 1

        disfluency = bool(w.get("disfluency", False))
        enabled = not disfluency  # Auto-disable initial disfluency words if flagged

        words.append(WordItem(
            id=f"w_{idx}_{uuid.uuid4().hex[:4]}",
            text=text,
            start_frame=start_f,
            end_frame=end_f,
            enabled=enabled,
            disfluency=disfluency,
            reason=w.get("reason") or None,
            candidate=bool(w.get("candidate", False)),
        ))

    timeline = Timeline(
        fps_num=fps_num,
        fps_den=fps_den,
        width=width,
        height=height,
        sources={source_id: source_file},
        words=words,
        default_transition=default_transition,
    )

    # Speech regions let the rebuild cut silence out of the middle of a word,
    # not merely between words — see Timeline.speech_regions.
    if speech_regions:
        timeline.speech_regions = [
            [time_to_frame(start, fps_num, fps_den), time_to_frame(end, fps_num, fps_den)]
            for start, end in speech_regions
        ]
    if energy_envelope:
        timeline.energy_envelope = energy_envelope
    if max_pause_seconds is not None:
        timeline.max_pause_seconds = float(max_pause_seconds)
    if pause_padding_seconds is not None:
        timeline.pause_padding_seconds = float(pause_padding_seconds)

    rebuild_primary_tracks(timeline, source_id)
    return timeline
