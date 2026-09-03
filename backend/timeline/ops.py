import logging
import uuid
from typing import List, Optional
from .schema import Timeline, TimelineItem, WordItem, SourceFile, Transform

logger = logging.getLogger("timeline.ops")


def apply_auto_zoom(
    timeline: Timeline,
    depth: float = 0.10,
    min_segment_seconds: float = 0.5,
    every: int = 1,
) -> int:
    """Give the auto-edit its motion: a gentle Ken Burns push-in / pull-back on
    each cut segment.

    A single-camera talking head has no angles to cut between, so an auto-edit is
    one long series of jump cuts that sit dead still. A slow zoom across each
    segment is what editors add to give that footage life. The move is animated
    with `scale`/`scale_end` on the V1 clip's Transform, which the compiler already
    renders via `zoompan`.

    Every move pushes IN, deliberately: each clip starts wide and drifts
    tighter, so at every join the frame steps back to wide by the whole depth
    of the move — a punch-out, the standard disguise for a talking-head jump
    cut. Alternating directions (the old behaviour) made the scale CONTINUOUS
    across every join — a push-in ends at 1+depth exactly where the following
    pull-back starts — so the head-position jump played completely bare. Depth
    varies a little per segment so the rhythm does not feel mechanical, and
    since each move resets to wide there is no ratchet. Segments too short to
    read as a move (a fragment between two cuts) are left flat. Only V1 media
    segments with no transform of their own are touched, so a move the user
    placed — or one carried across a rebuild on an anchor word — is never
    overwritten.

    Returns the number of segments moved.
    """
    depth = max(0.0, min(0.6, float(depth)))
    if depth <= 0.0:
        return 0

    fps = timeline.fps_num / max(1, timeline.fps_den)
    min_frames = int(round(max(0.0, min_segment_seconds) * fps))
    every = max(1, int(every))

    segments = sorted(
        [i for i in timeline.items
         if i.track == "V1" and i.enabled and i.kind == "media"],
        key=lambda i: i.timeline_start_frame,
    )

    applied = 0
    for index, segment in enumerate(segments):
        if segment.transform is not None and not segment.transform.is_identity():
            continue                       # keep a move the user (or a carry) set
        if segment.duration_frames < min_frames:
            continue
        if index % every != 0:
            continue
        jitter = 0.75 + 0.5 * ((index * 7) % 5) / 4.0   # 0.75..1.25, deterministic
        segment.transform = Transform(
            scale=1.0,
            scale_end=round(min(1.6, 1.0 + max(0.05, depth * jitter)), 3),
        )
        applied += 1

    if applied:
        timeline.revision += 1
    return applied


# Named reasons for non-lexical auto-detections — an inaudible pop the merge
# floor may bridge back rather than pay a jump-cut for.
_BRIDGEABLE_REASONS = {"filler", "filler_sound", "stutter", "sliver"}


def _is_bridgeable_removal(word) -> bool:
    """Whether a disabled word is a cosmetic auto-filler the merge floor may keep.

    True only for non-lexical auto-detections: a named filler/stutter/sliver, or a
    bare `disfluency` flag with no specific reason (an "uh" the detector marked).
    Everything else disabled is an *intentional* cut — a manual strike (which
    `toggle_word` leaves with no reason and no disfluency flag) or a lexical
    planner cut ("retake", "not_fluent", "not_grammatical", even when it also
    carries the disfluency flag) — and must always be honoured, however short, or
    the render plays words the transcript shows struck out.
    """
    return word.reason in _BRIDGEABLE_REASONS or (word.disfluency and not word.reason)


def _removed_speech_intervals(timeline: Timeline, hard_only: bool = False) -> List[List[int]]:
    """Source-frame spans of *deliberately removed* speech.

    These are the disabled words — clamped to actual speech when the map is
    known, so a disabled word floating in silence does not count. The rebuild
    needs them to tell two kinds of gap apart: a gap that exists because the
    speaker paused (bridge it, keep the breath) and a gap that exists because a
    fumble was cut (never bridge it — bridging plays the fumble).

    `hard_only` keeps just the intentional cuts (manual strikes + lexical planner
    cuts), dropping the short auto-fillers the merge floor may bridge. The merge
    uses it to guarantee an intentional strike is cut no matter how short.
    """
    disabled = [(w.start_frame, w.end_frame) for w in timeline.words
                if not w.enabled and w.end_frame > w.start_frame
                and (not hard_only or not _is_bridgeable_removal(w))]
    if not disabled:
        return []
    spans: List[List[int]] = []
    if timeline.speech_regions:
        regions = sorted((int(a), int(b)) for a, b in timeline.speech_regions if b > a)
        for word_start, word_end in disabled:
            for region_start, region_end in regions:
                start = max(word_start, region_start)
                end = min(word_end, region_end)
                if end > start:
                    spans.append([start, end])
    else:
        spans = [[s, e] for s, e in disabled]
    spans.sort()
    merged: List[List[int]] = []
    for span in spans:
        if merged and span[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span[1])
        else:
            merged.append(span)
    return merged


def _removed_inside(removed: List[List[int]], lo: int, hi: int) -> int:
    """Frames of removed speech falling inside (lo, hi)."""
    total = 0
    for start, end in removed:
        overlap = min(end, hi) - max(start, lo)
        if overlap > 0:
            total += overlap
    return total


def _snap_cut(frame: int, lo: int, hi: int, timeline: Timeline) -> int:
    """Move a cut point to the quietest moment within ±4 frames.

    ASR word boundaries are ±50ms guesses; the real articulation boundary is the
    energy dip between the two sounds. Cutting there is what makes an edit land
    between phonemes instead of through one. No stored envelope, no movement.
    """
    envelope = timeline.energy_envelope or {}
    db = envelope.get("db") or []
    rate = float(envelope.get("rate") or 0.0)
    if not db or rate <= 0:
        return frame
    fps = timeline.fps_num / max(1, timeline.fps_den)
    lo = max(lo, frame - 4)
    hi = min(hi, frame + 4)
    if hi < lo:
        return frame

    def loudness(candidate: int) -> float:
        # A timeline frame spans more than one envelope frame; take the quietest
        # instant under it, or sampling only the frame's start misses the dip.
        first = int(candidate / fps * rate)
        last = max(first, int((candidate + 1) / fps * rate))
        window = [db[i] for i in range(first, last + 1) if 0 <= i < len(db)]
        return min(window) if window else float("inf")

    baseline = loudness(frame)
    best_frame, best_db = frame, baseline
    for candidate in range(lo, hi + 1):
        value = loudness(candidate)
        if value < best_db:
            best_db, best_frame = value, candidate
    # Only move for a real dip. On flat energy every candidate ties within
    # noise, and drifting the cut early would shave the kept word. 1dB is
    # deliberate: 2dB refused genuine articulation dips on breathy recordings,
    # and anything looser starts eating consonants.
    if baseline - best_db < 1.0:
        return frame
    return best_frame

def audit_cut_coverage(timeline: Timeline, limit: int = 5) -> dict:
    """Check the render against the plan: are the cut words really gone?

    The word list says what should be removed; the V1 items say what will
    actually be played. Those two disagreed for a long time and nothing noticed,
    because the transcript panel reads the *plan*: the rebuild bridged any gap
    under `max_pause_seconds`, and a removed filler is such a gap, so every short
    cut was quietly glued back into the render while the UI showed it struck out.

    A cut word is only counted when it has speech under it — a disabled word
    floating in silence carries no audio to leak — and it counts as still audible
    when a V1 segment covers the middle of that speech. The reverse direction is
    audited too: a word still marked kept whose speech no V1 segment plays
    (a dropped sliver the plan forgot about) is counted as `kept_but_dropped`.
    Returns `{checked, still_audible, examples, kept_but_dropped, dropped_examples}`.
    """
    regions = sorted((int(a), int(b)) for a, b in (timeline.speech_regions or []) if b > a)
    covered = [(item.source_start_frame, item.source_end_frame)
               for item in timeline.items if item.track == "V1"]

    dropped: List[str] = []
    if covered:
        for word in timeline.words:
            if not word.enabled or word.end_frame <= word.start_frame:
                continue
            midpoint = (word.start_frame + word.end_frame) // 2
            if not any(a <= midpoint < b for a, b in covered):
                dropped.append(word.text)

    disabled = [w for w in timeline.words if not w.enabled and w.end_frame > w.start_frame]
    if not disabled:
        return {"checked": 0, "still_audible": 0, "examples": [],
                "kept_but_dropped": len(dropped), "dropped_examples": dropped[:limit]}

    checked = 0
    audible: List[str] = []
    for word in disabled:
        # The speech actually sitting under this word. With no map, the word's own
        # span is the best available answer.
        spans = []
        if regions:
            for region_start, region_end in regions:
                start = max(word.start_frame, region_start)
                end = min(word.end_frame, region_end)
                if end > start:
                    spans.append((start, end))
        else:
            spans = [(word.start_frame, word.end_frame)]
        if not spans:
            continue
        checked += 1
        start, end = max(spans, key=lambda s: s[1] - s[0])
        midpoint = (start + end) // 2
        if any(a <= midpoint < b for a, b in covered):
            audible.append(word.text)

    return {"checked": checked, "still_audible": len(audible), "examples": audible[:limit],
            "kept_but_dropped": len(dropped), "dropped_examples": dropped[:limit]}


def rebuild_primary_tracks(timeline: Timeline, primary_source_id: str) -> None:
    """
    Rebuild V1 (video) and A1 (audio) tracks from enabled words.
    Contiguous enabled words are merged into continuous video/audio segments.
    Timeline start frames ripple so there are no empty gaps between enabled dialogue cuts.
    """
    # A rebuild throws the primary items away, which would also throw away any
    # zoom or grade the user put on a segment. Anchor words identify segments
    # across rebuilds, so stash the effects against them and hand them back to
    # whichever new item lands on the same word.
    carried: dict = {}
    for item in timeline.items:
        if item.track in ("V1", "A1") and item.anchor_word_id:
            if item.transform is not None or item.color is not None:
                carried[(item.track, item.anchor_word_id)] = (item.transform, item.color)

    # Remove existing V1 and A1 items
    timeline.items = [item for item in timeline.items if item.track not in ("V1", "A1")]

    enabled_words = [w for w in timeline.words if w.enabled]
    if not enabled_words:
        timeline.recalculate_duration()
        return

    fps = timeline.fps_num / max(1, timeline.fps_den)
    # Pauses shorter than this are part of natural delivery and stay untouched.
    # Longer ones are collapsed to a padding beat either side, which is what
    # separates a tightened edit from a breathless one. The old rule broke a
    # segment at any gap over 5 frames (0.17s) and then deleted the gap whole,
    # so every natural beat in the delivery was stripped out.
    max_pause_frames = max(1, int(round(timeline.max_pause_seconds * fps)))
    padding_frames = max(0, int(round(timeline.pause_padding_seconds * fps)))

    # Keep only the parts of each enabled word that are actually speech. Without
    # this, silence a word has swallowed rides along untouched — on a real
    # recording that left 29.6s of dead air inside the kept segments.
    pieces: List[tuple] = []
    if timeline.speech_regions:
        regions = sorted((int(a), int(b)) for a, b in timeline.speech_regions if b > a)
        for word in enabled_words:
            overlapped = False
            for region_start, region_end in regions:
                if region_end <= word.start_frame:
                    continue
                if region_start >= word.end_frame:
                    break
                start = max(word.start_frame, region_start)
                end = min(word.end_frame, region_end)
                if end > start:
                    pieces.append((start, end, word.id))
                    overlapped = True
            if not overlapped:
                # No speech under this word at all. Keep a sliver so its text is
                # still represented rather than silently vanishing from the edit.
                pieces.append((word.start_frame, word.start_frame + 1, word.id))
    else:
        pieces = [(w.start_frame, w.end_frame, w.id) for w in enabled_words]

    pieces = [p for p in sorted(pieces) if p[1] > p[0]]
    if not pieces:
        timeline.recalculate_duration()
        return

    # A gap between kept pieces can exist for two reasons, and they must be
    # treated oppositely. A natural pause is bridged so the delivery keeps its
    # breath. A gap left by a *removed word* must never be bridged: bridging it
    # puts the fumble straight back in the render — which is exactly what
    # happened to every filler and stutter shorter than the pause threshold, and
    # why "cut" words could still be heard in the finished video. Removals of a
    # frame or two are the exception: a 2-frame cut is an inaudible pop and a
    # visible video jump, so those are deliberately left in place.
    removed = _removed_speech_intervals(timeline)
    # Intentional cuts only (manual strikes + lexical planner cuts). These are
    # never bridged, whatever their length — the floor below is for cosmetic
    # auto-fillers, not for a word the user or planner deliberately struck.
    hard_removed = _removed_speech_intervals(timeline, hard_only=True)
    # A cut costs a visible jump in the picture. Removing a couple of frames of
    # speech is not worth one: the viewer never hears the difference and always
    # sees the join. So a removal shorter than this is simply not made — the
    # material stays and the two segments become one. Measured effect on the real
    # recording: cuts fall from 35 to a calmer edit with no orphan flashes.
    blocking_frames = max(2, int(round(timeline.min_removal_seconds * fps)))

    segments: List[List[int]] = []
    for start, end, word_id in pieces:
        # Bridge a gap only when it holds nothing but a short auto-filler: under
        # the floor AND with no intentional cut inside it. A struck or lexically
        # cut word in the gap always forces a real cut, so the render matches the
        # struck-out transcript instead of quietly gluing the word back in.
        if (segments and start - segments[-1][1] <= max_pause_frames
                and _removed_inside(removed, segments[-1][1], start) < blocking_frames
                and _removed_inside(hard_removed, segments[-1][1], start) == 0):
            segments[-1][1] = max(segments[-1][1], end)
        else:
            segments.append([start, end, word_id])

    # Drop slivers. A kept run this short is a fragment of a word between two
    # removals; on screen it is a flash of a different head position, and losing
    # it merges two cuts into one. Only ever dropped when it is *surrounded* by
    # removals — a short segment at the edge of a real pause is fine.
    min_segment_frames = max(1, int(round(timeline.min_segment_seconds * fps)))
    if len(segments) > 2:
        kept_segments = [segments[0]]
        dropped_segments: List[List[int]] = []
        for index in range(1, len(segments) - 1):
            segment = segments[index]
            if segment[1] - segment[0] < min_segment_frames:
                dropped_segments.append(segment)
                continue
            kept_segments.append(segment)
        kept_segments.append(segments[-1])
        if dropped_segments:
            logger.debug("Dropped %d sliver segments", len(dropped_segments))
            # The audio for these words is gone, so the plan must say so too —
            # a word left `enabled` here shows as kept in the transcript panel
            # while the render plays silence. Only words with no footprint in
            # any surviving segment flip; a word straddling a kept segment
            # still plays and stays enabled.
            for word in timeline.words:
                if not word.enabled:
                    continue
                in_dropped = any(word.start_frame < seg[1] and word.end_frame > seg[0]
                                 for seg in dropped_segments)
                if not in_dropped:
                    continue
                in_kept = any(word.start_frame < seg[1] and word.end_frame > seg[0]
                              for seg in kept_segments)
                if not in_kept:
                    word.enabled = False
                    word.disfluency = True
                    word.reason = "sliver"
        segments = kept_segments

    # Build V1 and A1 items for each segment, starting after any intro card.
    current_timeline_frame = max(0, timeline.program_offset_frames)
    source_limit = None
    source = timeline.sources.get(primary_source_id)
    if source is not None:
        source_limit = int(round(source.duration_seconds * fps))

    previous_src_end = 0
    for idx, seg in enumerate(segments):
        # Keep a breath either side of the cut. Slicing exactly on the speech
        # clips consonant onsets and makes every join audible.
        src_start = seg[0] - padding_frames
        src_end = seg[1] + padding_frames
        anchor_id = seg[2]

        # Never pad into a neighbouring segment, or past the media.
        if idx > 0:
            src_start = max(src_start, segments[idx - 1][1])
        if idx + 1 < len(segments):
            src_end = min(src_end, segments[idx + 1][0])
        src_start = max(0, src_start)
        if source_limit is not None:
            src_end = min(src_end, source_limit)

        # Padding is a breath, not a licence to replay removed material. Without
        # this, the 0.12s pad either side re-covered most of a just-cut filler.
        for removed_start, removed_end in removed:
            if removed_start < seg[0] and removed_end > src_start:
                src_start = min(seg[0], max(src_start, removed_end))
            if removed_end > seg[1] and removed_start < src_end:
                src_end = max(seg[1], min(src_end, removed_start))

        # Land each cut on the quietest nearby moment (needs a stored envelope).
        # The hunt is bounded by removed speech exactly as the padding is: a dip
        # search that wanders into the removal gap from both sides re-covers a
        # short fumble entirely — measured as 0.13s "cut" fragments still
        # playing. A better cut point only legitimately lives in silence or in
        # the kept material's own edge.
        lo_bound = previous_src_end
        next_limit = segments[idx + 1][0] if idx + 1 < len(segments) else (
            source_limit if source_limit is not None else src_end + 2)
        hi_bound = next_limit
        # One frame of grace into the removal: the dip at a fumble's onset is
        # often the true articulation boundary, and 33ms of pre-fumble breath is
        # a smoother join. More than that starts replaying the fumble.
        for removed_start, removed_end in removed:
            if removed_end <= seg[0] and removed_end - 1 > lo_bound:
                lo_bound = removed_end - 1
            if removed_start >= seg[1] and removed_start + 1 < hi_bound:
                hi_bound = removed_start + 1
        src_start = _snap_cut(src_start, lo_bound, seg[0], timeline)
        # When the padding after this segment was clamped away entirely (a
        # neighbour or the media edge sits right there), src_end IS the word's
        # own end — letting the snap pull it 2 frames earlier shaved 66ms off a
        # final consonant. Only allow the snap to reach inside the word when
        # there is padding to trade.
        end_floor = seg[1] if src_end <= seg[1] else max(src_start + 1, seg[1] - 2)
        src_end = _snap_cut(src_end, end_floor, hi_bound, timeline)
        src_start = max(0, src_start)
        if source_limit is not None:
            src_end = min(src_end, source_limit)

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
            anchor_word_id=anchor_id
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
            anchor_word_id=anchor_id
        )

        for item in (v_item, a_item):
            restored = carried.get((item.track, item.anchor_word_id))
            if restored:
                item.transform, item.color = restored

        timeline.items.extend([v_item, a_item])
        current_timeline_frame = tl_end
        previous_src_end = src_end

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

def _subtract_regions(span: List[int], regions: List[List[int]]) -> List[List[int]]:
    """[start,end] with every overlapping [a,b] in `regions` cut out of it.

    Returns the surviving sub-spans, in order. A region that splits the span in
    two yields two pieces; one that swallows it yields none.
    """
    pieces = [list(span)]
    for a, b in regions:
        out: List[List[int]] = []
        for s, e in pieces:
            if b <= s or a >= e:
                out.append([s, e])                    # no overlap
                continue
            if a > s:
                out.append([s, min(a, e)])            # keep the head
            if b < e:
                out.append([max(b, s), e])            # keep the tail
        pieces = out
    return [p for p in pieces if p[1] > p[0]]


def trim_source_regions(timeline: Timeline, regions: List[List[int]],
                        primary_source_id: str) -> int:
    """Cut leaked source-frame regions out of the V1/A1 program and re-ripple.

    The auto-edit's cut-verify finds moments the video plays that the transcript
    removed — a fumble a wrong timestamp let through. This surgically removes
    those source spans from the primary tracks without touching the word plan or
    re-running the edit, then closes the timeline gaps they leave. Returns the
    number of leaked regions applied.
    """
    merged = sorted([[int(a), int(b)] for a, b in regions if b > a])
    if not merged:
        return 0
    collapsed: List[List[int]] = []
    for region in merged:
        if collapsed and region[0] <= collapsed[-1][1]:
            collapsed[-1][1] = max(collapsed[-1][1], region[1])
        else:
            collapsed.append(region)

    carried: dict = {}
    for item in timeline.items:
        if item.track in ("V1", "A1") and item.anchor_word_id and (
                item.transform is not None or item.color is not None):
            carried[(item.track, item.anchor_word_id)] = (item.transform, item.color)

    v1 = sorted([i for i in timeline.items if i.track == "V1"],
                key=lambda i: i.timeline_start_frame)
    segments: List[tuple] = []
    for item in v1:
        for start, end in _subtract_regions(
                [item.source_start_frame, item.source_end_frame], collapsed):
            segments.append((start, end, item.anchor_word_id))
    if len(segments) == len(v1):
        return 0                                       # nothing overlapped

    timeline.items = [i for i in timeline.items if i.track not in ("V1", "A1")]
    current = max(0, timeline.program_offset_frames)
    for idx, (src_start, src_end, anchor) in enumerate(segments):
        duration = src_end - src_start
        if duration <= 0:
            continue
        tl_start, tl_end = current, current + duration
        for track in ("V1", "A1"):
            item = TimelineItem(
                id=f"{track.lower()}_{idx}_{uuid.uuid4().hex[:6]}",
                track=track, source_id=primary_source_id,
                source_start_frame=src_start, source_end_frame=src_end,
                timeline_start_frame=tl_start, timeline_end_frame=tl_end,
                enabled=True, anchor_word_id=anchor)
            restored = carried.get((track, anchor))
            if restored:
                item.transform, item.color = restored
            timeline.items.append(item)
        current = tl_end

    timeline.recalculate_duration()
    timeline.revision += 1
    return len(collapsed)


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
        anchor_word_id=anchor_word_id,
        origin="manual"
    )
    timeline.items.append(broll_item)
    timeline.recalculate_duration()
    timeline.revision += 1
    return broll_item
