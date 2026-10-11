"""Programme time of the A1 voice (AUDIO_SUITE_PLAN.md §4, row L2).

The render builds the A1 audio as a plain concat of the A1 trims (render/compiler.py,
"3. Build A1 trims" / "4. Join the A1 clips"): the enabled media items on track A1 (compound
children flattened by the compiler's own `flatten_items`), whose source has audio and is not
an image, sorted by timeline start, with muted items left out; each contributes its
source_start_frame..source_end_frame of the source (of the source's `audio_override` WAV when
it has one), and a positive `program_offset_frames` puts that much silence first. A muted or
hidden A1 track contributes nothing.

Programme time is the time on that concat: what the timeline shows, and what a rack window
`enable='between(t,a,b)'` after the concat means in the render. The live engine plays the
same pieces, so a window is heard live on the same words it affects in the render.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from timeline.schema import Timeline, TimelineItem, frame_to_time


@dataclass(frozen=True)
class Segment:
    source_id: str
    path: str                 # the source's audio_override when set, else its file
    src_start_s: float
    src_end_s: float
    prog_start_s: float
    prog_end_s: float

    @property
    def duration_s(self) -> float:
        return self.src_end_s - self.src_start_s


Piece = Tuple[str, str, float, float]   # (source_id, path, source start s, source end s)


def program_offset_s(timeline: Timeline) -> float:
    """The silence the render puts before the first A1 trim."""
    return frame_to_time(max(0, timeline.program_offset_frames), timeline.fps_num, timeline.fps_den)


def _a1_items(timeline: Timeline) -> List[TimelineItem]:
    from render.compiler import flatten_items   # the render's own flattening rules

    state = timeline.tracks.get("A1")
    if state is not None and (state.muted or state.hidden):
        return []

    def has_audio(item: TimelineItem) -> bool:
        src = timeline.sources.get(item.source_id or "")
        return bool(src and src.has_audio and src.kind != "image")

    return sorted((i for i in flatten_items(timeline.items)
                   if i.kind == "media" and i.track == "A1" and has_audio(i)),
                  key=lambda i: i.timeline_start_frame)


def a1_segments(timeline: Timeline) -> List[Segment]:
    """The A1 trims the render concatenates, in order, with their programme times."""
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    segments: List[Segment] = []
    prog = program_offset_s(timeline)
    for item in _a1_items(timeline):
        if item.mute:
            continue
        src = timeline.sources[item.source_id]
        start = frame_to_time(item.source_start_frame, fps_num, fps_den)
        end = frame_to_time(item.source_end_frame, fps_num, fps_den)
        if end <= start:
            continue                                   # an empty trim adds nothing
        path = getattr(src, "audio_override", None) or src.path
        segments.append(Segment(item.source_id, path, start, end, prog, prog + (end - start)))
        prog += end - start
    return segments


def pieces_of(segments: List[Segment], a: float, b: float) -> List[Piece]:
    """The source pieces programme [a, b) plays, in order (the leading silence has none)."""
    pieces: List[Piece] = []
    for seg in segments:
        lo, hi = max(a, seg.prog_start_s), min(b, seg.prog_end_s)
        if hi - lo > 1e-6:
            pieces.append((seg.source_id, seg.path,
                           seg.src_start_s + (lo - seg.prog_start_s),
                           seg.src_start_s + (hi - seg.prog_start_s)))
    return pieces


def program_to_source(timeline: Timeline, a: float, b: float) -> List[Piece]:
    """Programme range [a, b) as source pieces, split at every cut."""
    return pieces_of(a1_segments(timeline), a, b)


def source_to_program(timeline: Timeline, source_id: str, t: float) -> Optional[float]:
    """Where source time t of `source_id` plays in the programme; None when it was cut away."""
    for seg in a1_segments(timeline):
        if seg.source_id == source_id and seg.src_start_s <= t < seg.src_end_s:
            return seg.prog_start_s + (t - seg.src_start_s)
    return None
