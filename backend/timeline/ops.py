import uuid
from typing import List, Optional
from .schema import Timeline, TimelineItem, WordItem, SourceFile

def rebuild_primary_tracks(timeline: Timeline, primary_source_id: str) -> None:
    """
    Rebuild V1 (video) and A1 (audio) tracks from enabled words.
    Contiguous enabled words are merged into continuous video/audio segments.
    Timeline start frames ripple so there are no empty gaps between enabled dialogue cuts.
    """
    # Remove existing V1 and A1 items
    timeline.items = [item for item in timeline.items if item.track not in ("V1", "A1")]
    
    enabled_words = [w for w in timeline.words if w.enabled]
    if not enabled_words:
        timeline.recalculate_duration()
        return

    # Group contiguous words into segments
    segments: List[List[WordItem]] = []
    current_segment: List[WordItem] = [enabled_words[0]]

    for i in range(1, len(enabled_words)):
        prev_word = enabled_words[i - 1]
        curr_word = enabled_words[i]
        # If words are contiguous (within 5 frames), group them
        if curr_word.start_frame - prev_word.end_frame <= 5:
            current_segment.append(curr_word)
        else:
            segments.append(current_segment)
            current_segment = [curr_word]
    segments.append(current_segment)

    # Build V1 and A1 items for each segment
    current_timeline_frame = 0
    for idx, seg in enumerate(segments):
        src_start = seg[0].start_frame
        src_end = seg[-1].end_frame
        seg_duration = src_end - src_start
        if seg_duration <= 0:
            continue

        tl_start = current_timeline_frame
        tl_end = tl_start + seg_duration

        # Video Item (V1)
        v_item = TimelineItem(
            id=f"v1_{idx}_{uuid.uuid4().hex[:6]}",
            track="V1",
            source_id=primary_source_id,
            source_start_frame=src_start,
            source_end_frame=src_end,
            timeline_start_frame=tl_start,
            timeline_end_frame=tl_end,
            enabled=True,
            anchor_word_id=seg[0].id
        )

        # Audio Item (A1)
        a_item = TimelineItem(
            id=f"a1_{idx}_{uuid.uuid4().hex[:6]}",
            track="A1",
            source_id=primary_source_id,
            source_start_frame=src_start,
            source_end_frame=src_end,
            timeline_start_frame=tl_start,
            timeline_end_frame=tl_end,
            enabled=True,
            anchor_word_id=seg[0].id
        )

        timeline.items.extend([v_item, a_item])
        current_timeline_frame = tl_end

    timeline.recalculate_duration()
    timeline.revision += 1

def toggle_word(timeline: Timeline, word_id: str, enabled: bool, primary_source_id: str) -> bool:
    """Toggle a word's enabled status and rebuild primary tracks."""
    for word in timeline.words:
        if word.id == word_id:
            word.enabled = enabled
            rebuild_primary_tracks(timeline, primary_source_id)
            return True
    return False

def toggle_word_range(timeline: Timeline, word_ids: List[str], enabled: bool, primary_source_id: str) -> int:
    """Toggle multiple words by ID and rebuild primary tracks."""
    count = 0
    id_set = set(word_ids)
    for word in timeline.words:
        if word.id in id_set:
            word.enabled = enabled
            count += 1
    if count > 0:
        rebuild_primary_tracks(timeline, primary_source_id)
    return count

def add_broll_item(
    timeline: Timeline,
    broll_source_id: str,
    timeline_start_frame: int,
    duration_frames: int,
    anchor_word_id: Optional[str] = None
) -> TimelineItem:
    """Add a V2 B-roll overlay item to the timeline."""
    broll_item = TimelineItem(
        id=f"v2_broll_{uuid.uuid4().hex[:6]}",
        track="V2",
        source_id=broll_source_id,
        source_start_frame=0,
        source_end_frame=duration_frames,
        timeline_start_frame=timeline_start_frame,
        timeline_end_frame=timeline_start_frame + duration_frames,
        enabled=True,
        anchor_word_id=anchor_word_id
    )
    timeline.items.append(broll_item)
    timeline.recalculate_duration()
    timeline.revision += 1
    return broll_item
