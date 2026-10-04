"""
GET /api/projects/{project_id}/kept_words

Words that survived the cut (BuzzcafAI's produce_video pipeline needs these,
not the full pre-cut transcript, to re-plan visuals against what the speaker
actually said in the take). Derived from the project's timeline: the `enabled`
words, in programme order, each with source time (`start`/`end`, i.e. where the
word sits in the original recording) and, where it falls inside a surviving V1
segment, programme time (`tl_start`/`tl_end`, i.e. where it lands in the edited
video) too.
"""

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from config import PROJECTS_DIR
from store.project_store import ProjectStore
from timeline.schema import Timeline, TimelineItem, WordItem, frame_to_time

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))


def _v1_items(timeline: Timeline) -> List[TimelineItem]:
    """Enabled V1 media segments, in programme order -- these are what a kept
    word's source frames get mapped through to find where it landed."""
    return sorted(
        (i for i in timeline.items if i.track == "V1" and i.kind == "media" and i.enabled),
        key=lambda i: i.timeline_start_frame,
    )


def _programme_span(word: WordItem, items: List[TimelineItem]) -> Optional[Tuple[int, int]]:
    """Where `word`'s source frames land on the programme timeline, via the V1
    item whose source range contains its start -- there is no speed change
    within a segment, so the mapping is a linear offset. None if the word does
    not fall inside any surviving segment (e.g. a dropped sliver)."""
    for item in items:
        if item.source_start_frame <= word.start_frame < item.source_end_frame:
            offset_start = max(0, word.start_frame - item.source_start_frame)
            offset_end = max(offset_start, min(item.source_end_frame, word.end_frame) - item.source_start_frame)
            tl_start = item.timeline_start_frame + offset_start
            tl_end = item.timeline_start_frame + offset_end
            return tl_start, tl_end
    return None


@router.get("/{project_id}/kept_words")
async def get_kept_words(project_id: str) -> Dict[str, Any]:
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    tl_data = p_data.get("timeline")
    if not tl_data:
        return {"words": []}

    timeline = Timeline.model_validate(tl_data)
    items = _v1_items(timeline)

    kept = sorted((w for w in timeline.words if w.enabled), key=lambda w: w.start_frame)

    words: List[Dict[str, Any]] = []
    for word in kept:
        entry: Dict[str, Any] = {
            "word": word.text,
            "word_native": word.word_native,
            "start": frame_to_time(word.start_frame, timeline.fps_num, timeline.fps_den),
            "end": frame_to_time(word.end_frame, timeline.fps_num, timeline.fps_den),
            "tl_start": None,
            "tl_end": None,
        }
        span = _programme_span(word, items)
        if span is not None:
            entry["tl_start"] = frame_to_time(span[0], timeline.fps_num, timeline.fps_den)
            entry["tl_end"] = frame_to_time(span[1], timeline.fps_num, timeline.fps_den)
        words.append(entry)

    return {"words": words}
