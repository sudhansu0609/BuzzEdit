"""Clip-level timeline operations for manual (multi-track) editing.

These operate directly on `Timeline.items` and are the backend for the NLE-style
gestures in the UI: add media to a track, split, delete, move, and trim clips.

Design note (see the "Separate tracks" decision): V1/A1 are AI/word-managed and
rebuilt from the transcript (origin="auto"), so manual edits live on overlay/mix
tracks (V2+/A2+, origin="manual"). These ops refuse to mutate auto/locked items
to prevent the next word-rebuild from silently clobbering hand edits.
"""

import uuid
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
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


def _finalize(timeline: Timeline, first: Iterable[str] = ()) -> None:
    pack_main_track(timeline, first)
    timeline.recalculate_duration()
    timeline.revision += 1


def _new_id(prefix: str = "clip") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def new_link_id() -> str:
    return f"lnk_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Linked clips (a video and its own sound, moved as one)
# ---------------------------------------------------------------------------

def linked_partners(timeline: Timeline, item: TimelineItem) -> List[TimelineItem]:
    if not item.link_id:
        return []
    return [i for i in timeline.items if i.link_id == item.link_id and i.id != item.id]


def with_partners(timeline: Timeline, item_ids: Iterable[str]) -> List[TimelineItem]:
    """The named items plus everything linked to them, each once."""
    seen: Set[str] = set()
    out: List[TimelineItem] = []
    for item_id in item_ids:
        item = get_item(timeline, item_id)
        if item is None:
            raise ClipOpError(f"Item {item_id} not found")
        for member in [item, *linked_partners(timeline, item)]:
            if member.id not in seen:
                seen.add(member.id)
                out.append(member)
    return out


def link_items(timeline: Timeline, item_ids: List[str]) -> str:
    """Link clips so they move, trim, split and delete together. Anything already
    linked to one of them joins the new set rather than being orphaned."""
    members = with_partners(timeline, item_ids)
    if len(members) < 2:
        raise ClipOpError("Select at least two clips to link")
    for member in members:
        _require_editable(member, member.id, timeline)
    link_id = new_link_id()
    for member in members:
        member.link_id = link_id
    _finalize(timeline)
    return link_id


def unlink_items(timeline: Timeline, item_ids: List[str]) -> int:
    """Break the link on these clips (and their partners) so each edits alone."""
    members = with_partners(timeline, item_ids)
    for member in members:
        member.link_id = None
    _finalize(timeline)
    return len(members)


# ---------------------------------------------------------------------------
# The magnetic main track and ripple edits
# ---------------------------------------------------------------------------

MAIN_TRACK = "V1"


def is_magnetic_main(timeline: Timeline) -> bool:
    """True when V1 holds hand-placed clips only.

    The compiler renders V1 as a straight concatenation of its clips — a gap on
    the lane is simply not there in the export — so a hand-built V1 is kept
    packed: first clip at the start, no holes, which makes the lane show what
    the render will actually play. A transcript-built spine (origin "auto") has
    its own ripple rules (ops.cut_program_range) and is left alone.
    """
    v1 = [i for i in timeline.items if i.track == MAIN_TRACK]
    return bool(v1) and all(i.origin != "auto" for i in v1)


def pack_main_track(timeline: Timeline, first: Iterable[str] = ()) -> None:
    """Close every gap on a hand-built V1, carrying linked partners along.

    Order is by start frame; on a tie the clips in `first` (the ones just
    dropped, pasted or moved) go ahead of the clip already there, which is what
    makes dropping onto a clip's leading edge insert before it.
    """
    if not is_magnetic_main(timeline):
        return
    first = set(first)
    v1 = sorted(
        (i for i in timeline.items if i.track == MAIN_TRACK),
        key=lambda i: (i.timeline_start_frame, 0 if i.id in first else 1, i.timeline_end_frame),
    )
    cursor = max(0, timeline.program_offset_frames)
    shifted: Set[str] = set()
    for item in v1:
        shift = cursor - item.timeline_start_frame
        cursor = item.timeline_end_frame + shift
        if not shift:
            continue
        item.timeline_start_frame += shift
        item.timeline_end_frame += shift
        for partner in linked_partners(timeline, item):
            # A partner on V1 gets its own turn in this loop; one on another
            # track follows the first V1 clip it is linked to, exactly once.
            if partner.track == MAIN_TRACK or partner.id in shifted:
                continue
            partner.timeline_start_frame += shift
            partner.timeline_end_frame += shift
            shifted.add(partner.id)


def main_track_insert_frame(timeline: Timeline, frame: int) -> int:
    """Where a clip dropped at `frame` lands on a magnetic V1: the nearest edit
    point — before a clip if dropped on its first half, after it otherwise, and
    on the end of the track if dropped past everything."""
    v1 = sorted((i for i in timeline.items if i.track == MAIN_TRACK),
                key=lambda i: i.timeline_start_frame)
    if not v1:
        return max(0, timeline.program_offset_frames)
    for item in v1:
        if item.timeline_start_frame <= frame < item.timeline_end_frame:
            mid = (item.timeline_start_frame + item.timeline_end_frame) / 2
            return item.timeline_start_frame if frame < mid else item.timeline_end_frame
    return v1[-1].timeline_end_frame if frame >= v1[-1].timeline_end_frame else v1[0].timeline_start_frame


def _movable(timeline: Timeline, item: TimelineItem) -> bool:
    return item.origin != "auto" and not item.locked and not is_track_locked(timeline, item.track)


def _ripple(timeline: Timeline, tracks: Iterable[str], from_frame: int, delta: int,
            include: Iterable[TimelineItem] = (), exclude: Iterable[str] = ()) -> int:
    """Slide every clip on `tracks` that starts at or after `from_frame` by
    `delta` frames (negative = left), together with linked partners — and with
    the rest of each partner's track from that point, so a ripple never shears a
    linked pair or slides one clip into the next.

    A leftward slide is shortened rather than let a moving clip land on one that
    is staying put or run past frame 0. Every moving clip moves by the same
    amount, so sync between tracks always survives. Returns the delta applied.
    """
    if delta == 0:
        return 0
    exclude = set(exclude)
    tracks = set(tracks)
    moving: Dict[str, TimelineItem] = {i.id: i for i in include if i.id not in exclude}
    changed = True
    while changed:
        changed = False
        for item in timeline.items:
            if item.id in moving or item.id in exclude or not _movable(timeline, item):
                continue
            if item.track in tracks and item.timeline_start_frame >= from_frame:
                moving[item.id] = item
                changed = True
        for item in list(moving.values()):
            tracks.add(item.track)
            for partner in linked_partners(timeline, item):
                if partner.id not in moving and partner.id not in exclude and _movable(timeline, partner):
                    moving[partner.id] = partner
                    changed = True
    if not moving:
        return 0

    if delta < 0:
        room = min(m.timeline_start_frame for m in moving.values())
        for m in moving.values():
            for other in timeline.items:
                if other.id in moving or other.track != m.track:
                    continue
                if other.timeline_end_frame <= m.timeline_start_frame:
                    room = min(room, m.timeline_start_frame - other.timeline_end_frame)
        delta = -min(-delta, max(0, room))
    else:
        room = None
        for m in moving.values():
            for other in timeline.items:
                if other.id in moving or other.track != m.track:
                    continue
                if other.timeline_start_frame >= m.timeline_end_frame:
                    gap = other.timeline_start_frame - m.timeline_end_frame
                    room = gap if room is None else min(room, gap)
        if room is not None:
            delta = min(delta, room)

    for m in moving.values():
        m.timeline_start_frame += delta
        m.timeline_end_frame += delta
    return delta


def close_gap(timeline: Timeline, track: str, at_frame: int) -> int:
    """Delete the empty stretch of `track` around `at_frame`, sliding everything
    after it left (a ripple). Returns how many frames were closed."""
    on_track = [i for i in timeline.items if i.track == track]
    if any(i.timeline_start_frame <= at_frame < i.timeline_end_frame for i in on_track):
        raise ClipOpError("That is a clip, not a gap")
    later = [i.timeline_start_frame for i in on_track if i.timeline_start_frame > at_frame]
    if not later:
        raise ClipOpError("There is nothing after this gap to close it up")
    gap_end = min(later)
    gap_start = max([i.timeline_end_frame for i in on_track if i.timeline_end_frame <= at_frame],
                    default=0)
    applied = _ripple(timeline, {track}, gap_end, -(gap_end - gap_start))
    if applied == 0:
        raise ClipOpError("Can't close this gap — a locked clip or a linked clip on "
                          "another track is in the way")
    _finalize(timeline)
    return -applied


def track_fits(timeline: Timeline, track: str, start: int, end: int, ignore: Set[str] = frozenset()) -> bool:
    return not any(
        i.track == track and i.id not in ignore
        and i.timeline_start_frame < end and start < i.timeline_end_frame
        for i in timeline.items
    )


def track_has_auto(timeline: Timeline, track: str) -> bool:
    return any(i.track == track and i.origin == "auto" for i in timeline.items)


def _tracks_of_kind(timeline: Timeline, kind: str) -> List[str]:
    names = {i.track for i in timeline.items if track_kind(i.track) == kind and i.track.upper() != "CAP"}
    names.update(t for t in timeline.extra_tracks if track_kind(t) == kind)
    return sorted(names, key=lambda t: int("".join(c for c in t if c.isdigit()) or 0))


def free_track(timeline: Timeline, kind: str, start: int, end: int,
               prefer: Iterable[str] = (), skip: Iterable[str] = ()) -> str:
    """The first track of `kind` with nothing in [start, end): the preferred
    ones first, then existing overlay lanes bottom-up, then a brand new lane.
    Never V1/A1 unless asked for (they are the programme), never a locked lane."""
    skip = set(skip)
    candidates = [*prefer, *(t for t in _tracks_of_kind(timeline, kind) if t.upper() not in ("V1", "A1"))]
    for track in candidates:
        if track in skip or track_kind(track) != kind or is_track_locked(timeline, track):
            continue
        if track_has_auto(timeline, track):
            continue
        if track_fits(timeline, track, start, end):
            return track
    taken = set(_tracks_of_kind(timeline, kind)) | skip
    num = max([int("".join(c for c in t if c.isdigit()) or 0) for t in taken] + [1]) + 1
    return f"{kind}{num}"


def a1_follows_v1(timeline: Timeline) -> bool:
    """A1 is the main track's own sound: empty, or only clips linked to V1 clips.
    Only then can a V1 clip's audio go on A1 without landing on music."""
    if track_has_auto(timeline, "A1"):
        return False
    v1_links = {i.link_id for i in timeline.items if i.track == MAIN_TRACK and i.link_id}
    return all(i.link_id in v1_links for i in timeline.items if i.track == "A1")


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
    # Dropped on a clip's leading edge of a magnetic V1, it goes in front of it.
    _finalize(timeline, first={item.id})
    return item


def _split_one(timeline: Timeline, item: TimelineItem,
               at_timeline_frame: int) -> Tuple[TimelineItem, TimelineItem]:
    offset = at_timeline_frame - item.timeline_start_frame  # frames into the clip
    src_split = item.source_start_frame + offset

    left = item.model_copy(deep=True, update={
        "id": _new_id(item.id.split("_")[0] or "clip"),
        "source_end_frame": src_split,
        "timeline_end_frame": at_timeline_frame,
    })
    right = item.model_copy(deep=True, update={
        "id": _new_id(item.id.split("_")[0] or "clip"),
        "source_start_frame": src_split,
        "timeline_start_frame": at_timeline_frame,
        # The transition belongs to the clip's leading edge, which only the
        # left half still has; a split must not invent a dissolve mid-shot.
        "transition": None,
    })
    if item.kind == "compound":
        # Children are positioned relative to their parent's start, so the right
        # half has to be rebased or every child would jump forward by the offset.
        _rebase_children(right, -offset)
    # Keep the halves where the original sat in the list: the compiler's V1
    # concat must never see the right half ahead of the left.
    idx = next(n for n, i in enumerate(timeline.items) if i.id == item.id)
    timeline.items[idx:idx + 1] = [left, right]
    return left, right


def split_items(timeline: Timeline, item_ids: List[str], at_timeline_frame: int,
                linked: bool = True) -> List[TimelineItem]:
    """Split every named clip (and, with `linked`, its partners) that the frame
    falls strictly inside. Halves of a linked pair stay linked to each other:
    the left halves keep the original link, the right halves share a new one."""
    targets = with_partners(timeline, item_ids) if linked else [
        get_item(timeline, i) for i in item_ids]
    targets = [t for t in targets
               if t is not None and t.timeline_start_frame < at_timeline_frame < t.timeline_end_frame]
    if not targets:
        raise ClipOpError("Split point must be strictly inside the clip")
    for target in targets:
        _require_editable(target, target.id, timeline)

    rights: Dict[str, List[TimelineItem]] = {}
    out: List[TimelineItem] = []
    for target in targets:
        left, right = _split_one(timeline, target, at_timeline_frame)
        out.extend([left, right])
        if target.link_id:
            rights.setdefault(target.link_id, []).append(right)
    for halves in rights.values():
        new_link = new_link_id() if len(halves) > 1 else None
        for half in halves:
            half.link_id = new_link
    _finalize(timeline)
    return out


def split_item(timeline: Timeline, item_id: str, at_timeline_frame: int,
               linked: bool = True) -> Tuple[TimelineItem, TimelineItem]:
    """Split a clip into two at an absolute timeline frame (the playhead)."""
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    if not (item.timeline_start_frame < at_timeline_frame < item.timeline_end_frame):
        raise ClipOpError("Split point must be strictly inside the clip")
    halves = split_items(timeline, [item_id], at_timeline_frame, linked=linked)
    # The named clip's own halves come first: it was the first target.
    return halves[0], halves[1]


def delete_items(timeline: Timeline, item_ids: List[str], ripple: bool = False,
                 linked: bool = True) -> int:
    """Delete clips (with their linked partners). With `ripple`, each hole is
    closed by sliding the rest of its track left — Filmora's Ripple Delete."""
    members = with_partners(timeline, item_ids) if linked else [
        get_item(timeline, i) for i in item_ids]
    if any(m is None for m in members):
        raise ClipOpError("Item not found")
    for member in members:
        _require_editable(member, member.id, timeline)
    gone = {m.id for m in members}
    timeline.items = [i for i in timeline.items if i.id not in gone]

    if ripple:
        # One ripple per distinct hole, latest first, so closing a later hole
        # never moves the edges of an earlier one. A video and its linked audio
        # share a hole and therefore a single ripple.
        holes: Dict[Tuple[int, int], Set[str]] = {}
        for m in members:
            holes.setdefault((m.timeline_start_frame, m.timeline_end_frame), set()).add(m.track)
        for (start, end), tracks in sorted(holes.items(), key=lambda kv: -kv[0][0]):
            _ripple(timeline, tracks, end, -(end - start))
    _finalize(timeline)
    return len(members)


def delete_item(timeline: Timeline, item_id: str, ripple: bool = False,
                linked: bool = True) -> None:
    delete_items(timeline, [item_id], ripple=ripple, linked=linked)


def move_items(timeline: Timeline, item_ids: List[str], delta_frames: int,
               track_map: Optional[Dict[str, str]] = None,
               linked: bool = True) -> List[TimelineItem]:
    """Slide clips (with their linked partners) by the same number of frames,
    optionally re-homing some onto other lanes of the same kind."""
    members = with_partners(timeline, item_ids) if linked else [
        get_item(timeline, i) for i in item_ids]
    if any(m is None for m in members):
        raise ClipOpError("Item not found")
    for member in members:
        _require_editable(member, member.id, timeline)
    track_map = track_map or {}
    for item_id, track in track_map.items():
        item = get_item(timeline, item_id)
        if item is None or not track:
            continue
        if track_kind(track) != track_kind(item.track):
            raise ClipOpError("Cannot move a video clip to an audio track (or vice-versa)")
        if is_track_locked(timeline, track):
            raise ClipOpError(f"{track} is locked — unlock it to move clips onto it")

    # The group moves as a block: it stops at frame 0 rather than squashing.
    delta = max(delta_frames, -min(m.timeline_start_frame for m in members))
    for member in members:
        member.timeline_start_frame += delta
        member.timeline_end_frame += delta
        if track_map.get(member.id):
            member.track = track_map[member.id]
    _finalize(timeline, first={m.id for m in members})
    return members


def move_item(
    timeline: Timeline,
    item_id: str,
    new_timeline_start_frame: int,
    new_track: Optional[str] = None,
    linked: bool = True,
) -> TimelineItem:
    """Move a clip in time and optionally to another track. Duration is preserved,
    and linked partners slide by the same amount (on their own tracks)."""
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    delta = max(0, new_timeline_start_frame) - item.timeline_start_frame
    move_items(timeline, [item_id], delta,
               track_map={item_id: new_track} if new_track else None, linked=linked)
    return item


def _rebase_children(item: TimelineItem, shift: int) -> None:
    """Slide a compound's children within the parent, keeping them where they play."""
    for child in item.children:
        child.timeline_start_frame += shift
        child.timeline_end_frame += shift


def trim_item(timeline: Timeline, item_id: str, edge: str, new_timeline_frame: int,
              linked: bool = True, ripple: bool = False) -> TimelineItem:
    """Trim one edge of a clip to a new timeline frame, adjusting the source in/out
    so the visible content stays anchored (like dragging a clip handle).

    Linked partners whose same edge sits on the same frame are trimmed with it,
    so the picture and its sound stay cut together. With `ripple` the rest of
    the track follows the edge: shortening pulls later clips in, lengthening
    pushes them out, and a trimmed head keeps the clip's left edge in place.
    """
    item = _require_editable(get_item(timeline, item_id), item_id, timeline)
    if edge not in ("start", "end"):
        raise ClipOpError("edge must be 'start' or 'end'")
    edge_of = (lambda i: i.timeline_start_frame) if edge == "start" else (lambda i: i.timeline_end_frame)
    old = edge_of(item)
    group = [item]
    if linked:
        group += [p for p in linked_partners(timeline, item) if edge_of(p) == old]
    for member in group:
        _require_editable(member, member.id, timeline)
    for member in group:
        _trim_one(timeline, member, edge, new_timeline_frame)

    if ripple and new_timeline_frame != old:
        tracks = {m.track for m in group}
        if edge == "end":
            _ripple(timeline, tracks, old, new_timeline_frame - old, exclude={m.id for m in group})
        else:
            # The clip itself slides back to where its head used to be.
            _ripple(timeline, tracks, new_timeline_frame, old - new_timeline_frame,
                    include=group)
    _finalize(timeline)
    return item


def trim_items(timeline: Timeline, item_ids: List[str], edge: str, at_frame: int,
               ripple: bool = False) -> List[TimelineItem]:
    """Trim the start or end of every named clip the frame falls inside to that
    frame — "trim start/end to playhead"."""
    done: Set[str] = set()
    trimmed: List[TimelineItem] = []
    for item_id in item_ids:
        if item_id in done:
            continue
        item = get_item(timeline, item_id)
        if item is None or not (item.timeline_start_frame < at_frame < item.timeline_end_frame):
            continue
        group = [item, *linked_partners(timeline, item)]
        done.update(m.id for m in group)
        trim_item(timeline, item_id, edge, at_frame, linked=True, ripple=ripple)
        trimmed.append(item)
    if not trimmed:
        raise ClipOpError("Put the playhead inside the clip to trim it")
    return trimmed


def _trim_one(timeline: Timeline, item: TimelineItem, edge: str, new_timeline_frame: int) -> None:
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
    # `extra` is a free-form payload (the knockout/behind-head/etc. text-fx
    # params): a caller editing one key (say `plate_color`) must not blow away
    # every other key already sitting there, which a plain `_merged` replace
    # would do. Merged by hand instead, with a None value deleting the key —
    # the one way a caller can ever remove something from `extra`.
    extra_update = updates.get("extra")
    rest = {k: v for k, v in updates.items() if k != "extra"}
    effects[index] = _merged(AtmosphereEffect, effects[index], rest)
    if isinstance(extra_update, dict):
        merged_extra = dict(effects[index].extra or {})
        for key, value in extra_update.items():
            if value is None:
                merged_extra.pop(key, None)
            else:
                merged_extra[key] = value
        effects[index].extra = merged_extra
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
    """Toggle enabled / locked / mute / loop, set volume, fades, ducking or rename a clip."""
    item = get_item(timeline, item_id)
    if item is None:
        raise ClipOpError(f"Item {item_id} not found")
    for key in ("enabled", "locked", "mute", "volume", "label",
                "loop", "audio_fade_in", "audio_fade_out", "duck"):
        if key not in flags or flags[key] is None:
            continue
        value = flags[key]
        if key == "duck":
            value = max(0.0, min(1.0, float(value)))
        elif key in ("audio_fade_in", "audio_fade_out"):
            value = max(0.0, float(value))
        elif key == "volume":
            value = max(0.0, min(4.0, float(value)))
        setattr(item, key, value)
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
    if any(track_kind(p.track) == "A" for p in linked_partners(timeline, item)):
        raise ClipOpError("This clip's audio is already on its own track — unlink "
                          "the two to edit them separately")

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


# ---------------------------------------------------------------------------
# Clipboard: paste / paste-insert
# ---------------------------------------------------------------------------

def _fresh_identity(item: TimelineItem, link_map: Dict[str, str]) -> None:
    """A pasted clip is a new, hand-placed clip — never a transcript-managed one."""
    item.id = _new_id(item.id.split("_")[0] or "clip")
    item.origin = "manual"
    item.locked = False
    item.anchor_word_id = None
    if item.link_id:
        item.link_id = link_map.setdefault(item.link_id, new_link_id())
    for child in item.children:
        child.id = _new_id(child.id.split("_")[0] or "clip")
        child.anchor_word_id = None
        if child.link_id:
            child.link_id = link_map.setdefault(child.link_id, new_link_id())


def paste_items(timeline: Timeline, clips: List[Dict[str, Any]], at_frame: int,
                insert: bool = False) -> List[TimelineItem]:
    """Paste copied clips so the earliest lands on `at_frame`, keeping their
    spacing, lanes and links.

    Each clip goes back on the lane it was copied from when that lane is free
    for the pasted span; otherwise the whole lane's worth moves up to the first
    free lane of its kind. A hand-built V1 is magnetic, so pasting onto it
    inserts at the next edit point. With `insert`, the target lanes are split at
    the playhead and pushed right to make room (Paste Insert).
    """
    if not clips:
        raise ClipOpError("Nothing to paste")
    try:
        pasted = [TimelineItem.model_validate(c) for c in clips]
    except Exception as e:  # pydantic ValidationError, bad shapes from the client
        raise ClipOpError(f"Clipboard contents are not clips: {e}")
    for clip in pasted:
        if clip.kind == "media" and clip.source_id not in timeline.sources:
            raise ClipOpError("A copied clip's media is no longer on this timeline")
        if clip.timeline_end_frame <= clip.timeline_start_frame:
            raise ClipOpError("A copied clip has no length")

    shift = max(0, at_frame) - min(c.timeline_start_frame for c in pasted)
    link_map: Dict[str, str] = {}
    for clip in pasted:
        _fresh_identity(clip, link_map)
        clip.timeline_start_frame += shift
        clip.timeline_end_frame += shift
        # Text belongs on a text lane even when it was copied off the captions.
        if clip.kind == "text" and track_kind(clip.track) != "T":
            clip.track = "T1"
    start = min(c.timeline_start_frame for c in pasted)
    end = max(c.timeline_end_frame for c in pasted)

    by_track: Dict[str, List[TimelineItem]] = {}
    for clip in pasted:
        by_track.setdefault(clip.track, []).append(clip)

    main_magnetic = is_magnetic_main(timeline) or not any(i.track == MAIN_TRACK for i in timeline.items)
    a1_free_to_follow = a1_follows_v1(timeline)
    used: Set[str] = set()
    a1_following = False
    for track in sorted(by_track, key=lambda t: (_KIND_ORDER[track_kind(t)],
                                                 int("".join(c for c in t if c.isdigit()) or 0))):
        group = by_track[track]
        kind = track_kind(track)
        g_start = min(c.timeline_start_frame for c in group)
        g_end = max(c.timeline_end_frame for c in group)
        target = None
        if track == MAIN_TRACK and main_magnetic and not track_has_auto(timeline, track):
            target = track
        elif track == "A1" and MAIN_TRACK in used and a1_free_to_follow:
            target = track
            a1_following = True
        elif (track not in used and not is_track_locked(timeline, track)
              and not track_has_auto(timeline, track)
              and (insert or all(track_fits(timeline, track, c.timeline_start_frame, c.timeline_end_frame)
                                 for c in group))):
            target = track
        if target is None:
            target = free_track(timeline, kind, g_start, g_end, skip=used)
        used.add(target)
        for clip in group:
            clip.track = target

    if insert:
        # The magnetic V1 (and the A1 that follows it) make room by packing.
        targets = {c.track for c in pasted} - {MAIN_TRACK} - ({"A1"} if a1_following else set())
        straddling = [i.id for i in timeline.items
                      if i.track in targets and _movable(timeline, i)
                      and i.timeline_start_frame < start < i.timeline_end_frame]
        if straddling:
            split_items(timeline, straddling, start)
        _ripple(timeline, targets, start, end - start)

    timeline.items.extend(pasted)
    _finalize(timeline, first={c.id for c in pasted})
    return pasted
