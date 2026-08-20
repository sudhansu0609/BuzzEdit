"""Clip-level timeline operations for manual (multi-track) editing.

These operate directly on `Timeline.items` and are the backend for the NLE-style
gestures in the UI: add media to a track, split, delete, move, and trim clips.

Design note (see the "Separate tracks" decision): V1/A1 are AI/word-managed and
rebuilt from the transcript (origin="auto"), so manual edits live on overlay/mix
tracks (V2+/A2+, origin="manual"). These ops refuse to mutate auto/locked items
to prevent the next word-rebuild from silently clobbering hand edits.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from .presets import (color_preset_values, effect_preset, text_preset_style,
                      transition_preset)
from .schema import (AtmosphereEffect, ChromaKey, ColorGrade, TextClip, TextStyle,
                     Timeline, TimelineItem, TrackState, Transform, Transition)


class ClipOpError(ValueError):
    """Raised when an operation is invalid (bad id, locked item, bad bounds)."""


def get_item(timeline: Timeline, item_id: str) -> Optional[TimelineItem]:
    for item in timeline.items:
        if item.id == item_id:
            return item
    return None


def _require_editable(item: Optional[TimelineItem], item_id: str,
                      timeline: Optional[Timeline] = None) -> TimelineItem:
    if item is None:
        raise ClipOpError(f"Item {item_id} not found")
    if item.origin == "auto" or item.locked:
        raise ClipOpError(
            f"Item {item_id} is on an AI-managed/locked track ({item.track}); "
            f"copy it to a manual track (V2+/A2+) to edit."
        )
    # A locked *track* protects everything on it, including clips added later.
    if timeline is not None and is_track_locked(timeline, item.track):
        raise ClipOpError(f"{item.track} is locked — unlock the track to edit its clips")
    return item


_KIND_ORDER = {"T": 0, "V": 1, "A": 2}


def track_kind(track: str) -> str:
    """'V' for video/caption tracks, 'A' for audio, 'T' for text overlays."""
    upper = track.upper()
    if upper.startswith("A"):
        return "A"
    if upper.startswith("T"):
        return "T"
    return "V"


def list_tracks(timeline: Timeline) -> List[str]:
    """All track names present, ordered text, then video, then audio, ascending."""
    names = {item.track for item in timeline.items}
    def sort_key(name: str):
        num = int("".join(ch for ch in name if ch.isdigit()) or 0)
        return (_KIND_ORDER[track_kind(name)], num, name)
    return sorted(names, key=sort_key)


def next_track(timeline: Timeline, kind: str) -> str:
    """Next free track name for a kind, e.g. 'V3', 'A2' or 'T1'."""
    kind = kind.upper()
    prefix = kind if kind in ("A", "T") else "V"
    max_num = 0
    for item in timeline.items:
        if track_kind(item.track) == prefix:
            digits = "".join(ch for ch in item.track if ch.isdigit())
            if digits:
                max_num = max(max_num, int(digits))
    return f"{prefix}{max_num + 1}"


def _finalize(timeline: Timeline) -> None:
    timeline.recalculate_duration()
    timeline.revision += 1


# ---------------------------------------------------------------------------
# Tracks (layers)
# ---------------------------------------------------------------------------

def track_state(timeline: Timeline, track: str) -> TrackState:
    """The switches for a track, defaulting to all-off for one never touched."""
    return timeline.tracks.get(track) or TrackState()


def is_track_locked(timeline: Timeline, track: str) -> bool:
    return track_state(timeline, track).locked


def set_track_flags(timeline: Timeline, track: str, **flags: Any) -> TrackState:
    """Hide, lock or mute a whole track.

    Stored per track rather than per clip: a lane can be empty and still need to
    remember it is locked, and flagging every clip would be undone by the next
    clip dropped onto it.
    """
    state = timeline.tracks.get(track) or TrackState()
    for key in ("hidden", "locked", "muted"):
        if flags.get(key) is not None:
            setattr(state, key, bool(flags[key]))
    timeline.tracks[track] = state
    # Keep an empty-but-configured lane alive so it survives a reload.
    if track not in {i.track for i in timeline.items} and track not in timeline.extra_tracks:
        timeline.extra_tracks.append(track)
    _finalize(timeline)
    return state


def add_track(timeline: Timeline, kind: str) -> str:
    """Reserve the next empty lane of a kind so it persists before anything lands on it."""
    name = next_track(timeline, kind)
    if name not in timeline.extra_tracks:
        timeline.extra_tracks.append(name)
    _finalize(timeline)
    return name


def delete_track(timeline: Timeline, track: str) -> int:
    """Remove a track and every clip on it. Returns how many clips went.

    V1/A1 are allowed to go too — they are the programme, and a user who asks to
    delete the layer means it. Transcribing again rebuilds them from the words,
    which is the documented behaviour of the primary tracks.
    """
    if is_track_locked(timeline, track):
        raise ClipOpError(f"{track} is locked — unlock it before deleting")

    removed = [i for i in timeline.items if i.track == track]
    if not removed and track not in timeline.extra_tracks and track not in timeline.tracks:
        raise ClipOpError(f"No track named {track}")

    timeline.items = [i for i in timeline.items if i.track != track]
    timeline.tracks.pop(track, None)
    timeline.extra_tracks = [t for t in timeline.extra_tracks if t != track]
    _finalize(timeline)
    return len(removed)


def add_media_item(
    timeline: Timeline,
    source_id: str,
    track: str,
    timeline_start_frame: int,
    source_start_frame: int,
    source_end_frame: int,
    origin: str = "manual",
) -> TimelineItem:
    """Place a clip from an already-registered source onto a track."""
    if source_id not in timeline.sources:
        raise ClipOpError(f"Source {source_id} not registered in timeline")
    if is_track_locked(timeline, track):
        raise ClipOpError(f"{track} is locked — unlock it to add clips")
    duration = source_end_frame - source_start_frame
    if duration <= 0:
        raise ClipOpError("Clip duration must be positive")
    timeline_start_frame = max(0, timeline_start_frame)
    item = TimelineItem(
        id=f"clip_{uuid.uuid4().hex[:8]}",
        track=track,
        source_id=source_id,
        source_start_frame=source_start_frame,
        source_end_frame=source_end_frame,
        timeline_start_frame=timeline_start_frame,
        timeline_end_frame=timeline_start_frame + duration,
        enabled=True,
        origin=origin,
    )
    timeline.items.append(item)
    _finalize(timeline)
    return item


def split_item(timeline: Timeline, item_id: str, at_timeline_frame: int) -> Tuple[TimelineItem, TimelineItem]:
    """Split a clip into two at an absolute timeline frame (the playhead)."""
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    if not (item.timeline_start_frame < at_timeline_frame < item.timeline_end_frame):
        raise ClipOpError("Split point must be strictly inside the clip")

    offset = at_timeline_frame - item.timeline_start_frame  # frames into the clip
    src_split = item.source_start_frame + offset

    left = item.model_copy(deep=True, update={
        "id": f"clip_{uuid.uuid4().hex[:8]}",
        "source_end_frame": src_split,
        "timeline_end_frame": at_timeline_frame,
    })
    right = item.model_copy(deep=True, update={
        "id": f"clip_{uuid.uuid4().hex[:8]}",
        "source_start_frame": src_split,
        "timeline_start_frame": at_timeline_frame,
    })
    if item.kind == "compound":
        # Children are positioned relative to their parent's start, so the right
        # half has to be rebased or every child would jump forward by the offset.
        _rebase_children(right, -offset)
    timeline.items = [i for i in timeline.items if i.id != item_id]
    timeline.items.extend([left, right])
    _finalize(timeline)
    return left, right


def delete_item(timeline: Timeline, item_id: str) -> None:
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    timeline.items = [i for i in timeline.items if i.id != item.id]
    _finalize(timeline)


def move_item(
    timeline: Timeline,
    item_id: str,
    new_timeline_start_frame: int,
    new_track: Optional[str] = None,
) -> TimelineItem:
    """Move a clip in time and optionally to another track. Duration is preserved."""
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    duration = item.timeline_end_frame - item.timeline_start_frame
    new_start = max(0, new_timeline_start_frame)
    item.timeline_start_frame = new_start
    item.timeline_end_frame = new_start + duration
    if new_track:
        if track_kind(new_track) != track_kind(item.track):
            raise ClipOpError("Cannot move a video clip to an audio track (or vice-versa)")
        item.track = new_track
    _finalize(timeline)
    return item


def _rebase_children(item: TimelineItem, shift: int) -> None:
    """Slide a compound's children within the parent, keeping them where they play."""
    for child in item.children:
        child.timeline_start_frame += shift
        child.timeline_end_frame += shift


def trim_item(timeline: Timeline, item_id: str, edge: str, new_timeline_frame: int) -> TimelineItem:
    """Trim one edge of a clip to a new timeline frame, adjusting the source in/out
    so the visible content stays anchored (like dragging a clip handle)."""
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)

    # Text has no source to run out of, and a compound's material lives in its
    # children — only real media clips are bounded by a source file.
    trims_source = item.kind == "media"
    max_src = None
    if trims_source:
        source = timeline.sources.get(item.source_id or "")
        if source is not None:
            fps = timeline.fps_num / max(1, timeline.fps_den)
            max_src = int(round(source.duration_seconds * fps))

    if edge == "start":
        if new_timeline_frame >= item.timeline_end_frame:
            raise ClipOpError("Start edge must stay before the end")
        if new_timeline_frame < 0:
            raise ClipOpError("A clip cannot start before the timeline does")
        delta = new_timeline_frame - item.timeline_start_frame
        if trims_source:
            new_src_start = item.source_start_frame + delta
            if new_src_start < 0:
                raise ClipOpError("Trim would run past the start of the source media")
            item.source_start_frame = new_src_start
        elif item.kind == "compound":
            # The parent's start moved, so rebase children to hold them in place.
            _rebase_children(item, -delta)
        item.timeline_start_frame = new_timeline_frame
    elif edge == "end":
        if new_timeline_frame <= item.timeline_start_frame:
            raise ClipOpError("End edge must stay after the start")
        delta = new_timeline_frame - item.timeline_end_frame
        if trims_source:
            new_src_end = item.source_end_frame + delta
            if new_src_end <= item.source_start_frame:
                raise ClipOpError("Trim would collapse the clip")
            if max_src is not None and new_src_end > max_src:
                raise ClipOpError("Trim would run past the end of the source media")
            item.source_end_frame = new_src_end
        item.timeline_end_frame = new_timeline_frame
    else:
        raise ClipOpError("edge must be 'start' or 'end'")

    _finalize(timeline)
    return item


# ---------------------------------------------------------------------------
# Effects: transform, colour, text
# ---------------------------------------------------------------------------

def _merged(model_cls, current: Optional[BaseModel], *updates: Dict[str, Any]):
    """Layer partial dicts onto a model, dropping keys the model does not define.

    Presets and UI panels both send sparse patches, and the UI happily rounds
    trips extra keys (`preset`, ids) that the model has no field for — silently
    ignoring them beats a 500 on every save.
    """
    data = (current.model_dump() if current is not None else model_cls().model_dump())
    allowed = set(model_cls.model_fields)
    for update in updates:
        if not update:
            continue
        for key, value in update.items():
            if key not in allowed:
                continue
            # Nested models (the colour wheels) get merged, not replaced. The UI
            # sends the one component the user dragged, and replacing the whole
            # wheel with `{"r": 0.2}` would silently zero its green and blue.
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
    return model_cls(**data)


def _require_effectable(timeline: Timeline, item_id: str) -> TimelineItem:
    """Any item can take effects — even AI-managed ones, unless explicitly locked.

    Transform and colour are non-destructive and do not fight the word rebuild
    (ops.rebuild_primary_tracks carries them across by anchor word), so there is
    no reason to block them on V1 the way structural edits are blocked.
    """
    item = get_item(timeline, item_id)
    if item is None:
        raise ClipOpError(f"Item {item_id} not found")
    if item.locked:
        raise ClipOpError(f"Item {item_id} is locked")
    if is_track_locked(timeline, item.track):
        raise ClipOpError(f"{item.track} is locked — unlock the track to change its clips")
    return item


def set_transform(timeline: Timeline, item_id: str, updates: Dict[str, Any],
                  reset: bool = False) -> TimelineItem:
    """Patch a clip's geometry (crop / scale / pan / rotation / opacity)."""
    item = _require_effectable(timeline, item_id)
    if reset:
        item.transform = None
    else:
        item.transform = _merged(Transform, item.transform, updates)
        if item.transform.is_identity():
            item.transform = None
    _finalize(timeline)
    return item


def set_color(timeline: Timeline, item_id: str, updates: Dict[str, Any],
              preset: Optional[str] = None, reset: bool = False) -> TimelineItem:
    """Patch a clip's grade. A preset seeds the values; explicit updates win."""
    item = _require_effectable(timeline, item_id)
    if reset:
        item.color = None
    else:
        previous_preset = item.color.preset if item.color else None
        base = ColorGrade() if preset else item.color
        item.color = _merged(ColorGrade, base, color_preset_values(preset), updates)
        item.color.preset = preset or previous_preset
        if item.color.is_identity():
            item.color = None
    _finalize(timeline)
    return item


def set_chroma(timeline: Timeline, item_id: str, updates: Dict[str, Any],
               reset: bool = False) -> TimelineItem:
    """Patch a clip's chroma key.

    Keying is refused on V1. There is nothing behind the programme track to show
    through the hole, and the transparent stream it produces cannot be concatenated
    with the opaque ones around it — the render would fail rather than look wrong,
    which is a confusing way to learn that green screen belongs on an overlay.
    """
    item = _require_effectable(timeline, item_id)
    if track_kind(item.track) != "V" or item.track.upper() == "V1":
        raise ClipOpError(
            f"Chroma key needs a video overlay track; {item.track} is the "
            f"programme track. Move the clip to V2 or above to key it."
        )
    if reset:
        item.chroma = None
    else:
        item.chroma = _merged(ChromaKey, item.chroma, updates)
        if not item.chroma.enabled:
            item.chroma = None
    _finalize(timeline)
    return item


def set_master_color(timeline: Timeline, updates: Dict[str, Any],
                     preset: Optional[str] = None, reset: bool = False) -> Optional[ColorGrade]:
    """Grade the whole program. Survives transcript rebuilds, unlike per-clip grades."""
    if reset:
        timeline.master_color = None
    else:
        base = ColorGrade() if preset else timeline.master_color
        timeline.master_color = _merged(ColorGrade, base, color_preset_values(preset), updates)
        timeline.master_color.preset = preset or timeline.master_color.preset
    _finalize(timeline)
    return timeline.master_color


def set_master_transform(timeline: Timeline, updates: Dict[str, Any],
                         reset: bool = False) -> Optional[Transform]:
    """Reframe the whole program (crop to 9:16, push in, slow drift)."""
    if reset:
        timeline.master_transform = None
    else:
        timeline.master_transform = _merged(Transform, timeline.master_transform, updates)
        if timeline.master_transform.is_identity():
            timeline.master_transform = None
    _finalize(timeline)
    return timeline.master_transform


def build_text_clip(content: str, preset: Optional[str] = None,
                    style: Optional[Dict[str, Any]] = None,
                    current: Optional[TextClip] = None) -> TextClip:
    # A named preset resets the styling before the caller's overrides land, so
    # picking a preset is a clean swap rather than a merge with the old look.
    base = None if preset else (current.style if current else None)
    merged = _merged(TextStyle, base, text_preset_style(preset), style or {})
    return TextClip(content=content, style=merged,
                    preset=preset or (current.preset if current else None))


def add_text_item(
    timeline: Timeline,
    content: str,
    timeline_start_frame: int,
    duration_frames: int,
    track: Optional[str] = None,
    preset: Optional[str] = None,
    style: Optional[Dict[str, Any]] = None,
) -> TimelineItem:
    """Place a text element on a text track (T1, T2, ...)."""
    if duration_frames <= 0:
        raise ClipOpError("Text duration must be positive")
    track = track or next_track(timeline, "T")
    if track_kind(track) != "T":
        raise ClipOpError(f"{track} is not a text track — text lives on T1, T2, ...")
    start = max(0, timeline_start_frame)
    item = TimelineItem(
        id=f"txt_{uuid.uuid4().hex[:8]}",
        track=track,
        source_id=None,
        kind="text",
        timeline_start_frame=start,
        timeline_end_frame=start + duration_frames,
        origin="manual",
        text=build_text_clip(content, preset, style),
        label=content[:40],
    )
    timeline.items.append(item)
    _finalize(timeline)
    return item


def add_adjustment_item(
    timeline: Timeline,
    timeline_start_frame: int,
    duration_frames: int,
    track: Optional[str] = None,
    preset: Optional[str] = None,
) -> TimelineItem:
    """Place an adjustment layer: a clip with no picture that treats what is below it.

    It goes on a video track like any other clip, but the compiler applies its
    transform, grade and atmosphere to everything already composited beneath it
    rather than drawing anything of its own. Which track it sits on therefore
    decides its reach — V3 treats V1 and V2, and leaves V4 alone.
    """
    if duration_frames <= 0:
        raise ClipOpError("An adjustment layer needs a positive duration")
    track = track or next_track(timeline, "V")
    if track_kind(track) != "V":
        raise ClipOpError(f"{track} is not a video track — adjustment layers live on V2, V3, ...")
    if track.upper() == "V1":
        raise ClipOpError("An adjustment layer has to sit above the picture it adjusts, not on V1")
    if is_track_locked(timeline, track):
        raise ClipOpError(f"{track} is locked — unlock it to add an adjustment layer")

    start = max(0, timeline_start_frame)
    item = TimelineItem(
        id=f"adj_{uuid.uuid4().hex[:8]}",
        track=track,
        source_id=None,
        kind="adjustment",
        timeline_start_frame=start,
        timeline_end_frame=start + duration_frames,
        origin="manual",
        label="Adjustment",
    )
    if preset:
        item.color = _merged(ColorGrade, None, color_preset_values(preset))
        item.color.preset = preset
    timeline.items.append(item)
    _finalize(timeline)
    return item


def set_text(timeline: Timeline, item_id: str, content: Optional[str] = None,
             preset: Optional[str] = None, style: Optional[Dict[str, Any]] = None) -> TimelineItem:
    """Edit a text clip's copy and/or styling."""
    item = _require_effectable(timeline, item_id)
    if item.kind != "text" or item.text is None:
        raise ClipOpError(f"Item {item_id} is not a text clip")
    new_content = item.text.content if content is None else content
    item.text = build_text_clip(new_content, preset, style, current=item.text)
    item.label = new_content[:40]
    _finalize(timeline)
    return item


def set_transition(timeline: Timeline, item_id: Optional[str],
                   preset: Optional[str] = None,
                   updates: Optional[Dict[str, Any]] = None,
                   reset: bool = False) -> Optional[Transition]:
    """Set the transition into a clip, or the programme default when item_id is None.

    A duration of zero means a hard cut, which is stored as "no transition" so the
    compiler can concatenate that junction instead of overlapping it.
    """
    data: Dict[str, Any] = {}
    data.update(transition_preset(preset))
    data.update(updates or {})

    if item_id:
        item = _require_effectable(timeline, item_id)
        if reset or (data.get("duration") == 0):
            item.transition = None
        else:
            item.transition = _merged(Transition, item.transition, data)
        _finalize(timeline)
        return item.transition

    if reset or (data.get("duration") == 0):
        timeline.default_transition = None
    else:
        timeline.default_transition = _merged(Transition, timeline.default_transition, data)
    _finalize(timeline)
    return timeline.default_transition


def _effect_list(timeline: Timeline, item_id: Optional[str]) -> List[AtmosphereEffect]:
    """Where an effect lives: the programme, or one adjustment layer's own list."""
    if not item_id:
        return timeline.effects
    item = _require_effectable(timeline, item_id)
    if item.kind != "adjustment":
        raise ClipOpError(
            f"Item {item_id} is a {item.kind} clip — effects attach to the programme "
            f"or to an adjustment layer"
        )
    return item.atmosphere


def add_effect(timeline: Timeline, preset: Optional[str] = None,
               updates: Optional[Dict[str, Any]] = None,
               item_id: Optional[str] = None) -> AtmosphereEffect:
    """Add a generated atmosphere effect to the programme, or to an adjustment layer."""
    data: Dict[str, Any] = {"type": "rain"}
    data.update(effect_preset(preset))
    data.update(updates or {})
    if not data.get("type"):
        raise ClipOpError("An effect needs a type")
    # Built directly rather than through _merged: AtmosphereEffect has a required
    # field, so there is no valid empty instance to layer updates onto.
    allowed = set(AtmosphereEffect.model_fields)
    effect = AtmosphereEffect(**{k: v for k, v in data.items() if k in allowed})
    _effect_list(timeline, item_id).append(effect)
    _finalize(timeline)
    return effect


def update_effect(timeline: Timeline, index: int, updates: Dict[str, Any],
                  item_id: Optional[str] = None) -> AtmosphereEffect:
    effects = _effect_list(timeline, item_id)
    if not 0 <= index < len(effects):
        raise ClipOpError(f"No effect at position {index}")
    effects[index] = _merged(AtmosphereEffect, effects[index], updates)
    _finalize(timeline)
    return effects[index]


def remove_effect(timeline: Timeline, index: int, item_id: Optional[str] = None) -> None:
    effects = _effect_list(timeline, item_id)
    if not 0 <= index < len(effects):
        raise ClipOpError(f"No effect at position {index}")
    effects.pop(index)
    _finalize(timeline)


def set_aspect_bars(timeline: Timeline, ratio: Optional[float]) -> Optional[float]:
    """Letterbox the programme to a cinematic shape, or clear it with None."""
    if ratio is not None and not (0.3 <= float(ratio) <= 4.0):
        raise ClipOpError("Aspect ratio must be between 0.3 and 4.0")
    timeline.aspect_bars = float(ratio) if ratio else None
    _finalize(timeline)
    return timeline.aspect_bars


def set_item_flags(timeline: Timeline, item_id: str, **flags: Any) -> TimelineItem:
    """Toggle enabled / locked / mute, set volume or rename a clip."""
    item = get_item(timeline, item_id)
    if item is None:
        raise ClipOpError(f"Item {item_id} not found")
    for key in ("enabled", "locked", "mute", "volume", "label"):
        if key in flags and flags[key] is not None:
            setattr(item, key, flags[key])
    _finalize(timeline)
    return item


# ---------------------------------------------------------------------------
# Detach audio / compound clips
# ---------------------------------------------------------------------------

def detach_audio(timeline: Timeline, item_id: str, track: Optional[str] = None) -> TimelineItem:
    """Split a video clip's audio onto its own track so it can be edited alone.

    Video tracks in this compiler are picture-only — a clip dropped on V2 renders
    silent by design, which keeps the filter graph predictable. Detaching is how
    you get at that clip's sound: it creates the matching A-track item and marks
    the video half as detached so a second detach is refused.
    """
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    if track_kind(item.track) != "V" or item.kind != "media":
        raise ClipOpError("Only video clips can have their audio detached")
    if item.mute:
        raise ClipOpError("This clip's audio is already detached")

    source = timeline.sources.get(item.source_id or "")
    if source is None:
        raise ClipOpError("Clip has no registered source")
    if not source.has_audio or source.kind == "image":
        raise ClipOpError(f"{source.kind} source has no audio to detach")

    audio_track = track or next_track(timeline, "A")
    if track_kind(audio_track) != "A":
        raise ClipOpError("Detached audio must land on an audio track")

    audio_item = item.model_copy(deep=True, update={
        "id": f"clip_{uuid.uuid4().hex[:8]}",
        "track": audio_track,
        "transform": None,
        "color": None,
        "text": None,
        "mute": False,
        "label": item.label or f"audio of {item.id[-4:]}",
    })
    item.mute = True
    timeline.items.append(audio_item)
    _finalize(timeline)
    return audio_item


def make_compound(timeline: Timeline, item_ids: List[str],
                  label: Optional[str] = None) -> TimelineItem:
    """Group clips into one block that moves, trims and deletes as a unit.

    Children are stored with timeline frames *relative* to the group's start, so
    sliding the parent slides everything without touching the children. The
    compiler flattens the group back out at render time (render.compiler.flatten_items).
    """
    if len(item_ids) < 2:
        raise ClipOpError("Select at least two clips to compound")

    members: List[TimelineItem] = []
    for item_id in item_ids:
        item = _require_editable(get_item(timeline, item_id), item_id, timeline)
        if item.kind == "compound":
            raise ClipOpError("Uncompound the nested group before compounding again")
        members.append(item)

    start = min(i.timeline_start_frame for i in members)
    end = max(i.timeline_end_frame for i in members)

    # The group needs a home lane; use the topmost track any member occupies so
    # the block appears where the user was already looking.
    host_track = sorted(
        {i.track for i in members},
        key=lambda t: (_KIND_ORDER[track_kind(t)], int("".join(c for c in t if c.isdigit()) or 0)),
    )[0]

    children = []
    for item in members:
        child = item.model_copy(deep=True)
        child.timeline_start_frame = item.timeline_start_frame - start
        child.timeline_end_frame = item.timeline_end_frame - start
        children.append(child)

    member_ids = {i.id for i in members}
    timeline.items = [i for i in timeline.items if i.id not in member_ids]

    parent = TimelineItem(
        id=f"cmp_{uuid.uuid4().hex[:8]}",
        track=host_track,
        source_id=None,
        kind="compound",
        timeline_start_frame=start,
        timeline_end_frame=end,
        origin="manual",
        children=children,
        label=label or f"Compound ({len(children)} clips)",
    )
    timeline.items.append(parent)
    _finalize(timeline)
    return parent


def break_compound(timeline: Timeline, item_id: str) -> List[TimelineItem]:
    """Dissolve a compound back into independent clips at their current positions."""
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    if item.kind != "compound":
        raise ClipOpError(f"Item {item_id} is not a compound clip")

    restored: List[TimelineItem] = []
    for child in item.children:
        abs_start = item.timeline_start_frame + child.timeline_start_frame
        abs_end = item.timeline_start_frame + child.timeline_end_frame
        # Honour a trim applied to the group: children outside it stay gone.
        if abs_end <= item.timeline_start_frame or abs_start >= item.timeline_end_frame:
            continue
        head = max(0, item.timeline_start_frame - abs_start)
        tail = max(0, abs_end - item.timeline_end_frame)

        clone = child.model_copy(deep=True)
        clone.id = f"clip_{uuid.uuid4().hex[:8]}"
        clone.timeline_start_frame = abs_start + head
        clone.timeline_end_frame = abs_end - tail
        clone.source_start_frame = child.source_start_frame + head
        clone.source_end_frame = child.source_end_frame - tail
        if clone.transform is None:
            clone.transform = item.transform
        if clone.color is None:
            clone.color = item.color
        restored.append(clone)

    timeline.items = [i for i in timeline.items if i.id != item.id]
    timeline.items.extend(restored)
    _finalize(timeline)
    return restored
