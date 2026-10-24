import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
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


def _lay_full_source(timeline: Timeline, primary_source_id: str, carried: dict) -> None:
    """One V1 and one A1 clip over the whole source (Timeline.keep_full_source).

    The user cut this recording themselves, so nothing is trimmed: not the
    pauses, not the fumbles, not the silence before the first word. Every word
    is marked enabled because every word plays -- a struck word here would show
    as cut in the transcript panel while the render still played it.
    """
    fps = timeline.fps_num / max(1, timeline.fps_den)
    source = timeline.sources.get(primary_source_id)
    if source is not None and source.duration_seconds > 0:
        src_end = int(round(source.duration_seconds * fps))
    else:
        src_end = max((w.end_frame for w in timeline.words), default=0)
    for word in timeline.words:
        word.enabled = True
    if src_end <= 0:
        timeline.recalculate_duration()
        return

    anchor_id = timeline.words[0].id if timeline.words else None
    tl_start = max(0, timeline.program_offset_frames)
    for track, prefix in (("V1", "v1"), ("A1", "a1")):
        item = TimelineItem(
            id=f"{prefix}_0_{uuid.uuid4().hex[:6]}",
            track=track,
            source_id=primary_source_id,
            source_start_frame=0,
            source_end_frame=src_end,
            timeline_start_frame=tl_start,
            timeline_end_frame=tl_start + src_end,
            enabled=True,
            anchor_word_id=anchor_id,
        )
        restored = carried.get((track, anchor_id))
        if restored:
            item.transform, item.color = restored
        timeline.items.append(item)

    timeline.recalculate_duration()
    timeline.revision += 1


def rebuild_primary_tracks(timeline: Timeline, primary_source_id: str) -> None:
    """Rebuild V1/A1 from the word list, then reapply any manual cuts on top.

    `manual_cuts` are hand-made removals on the AI-managed tracks (via
    `cut_program_range`) that the word-driven rebuild knows nothing about; every
    word toggle throws V1/A1 away and rebuilds them from scratch, which would
    silently bring a manually cut span back. Reapplying `trim_source_regions`
    here is what makes a manual cut survive that rebuild.
    """
    _rebuild_primary_tracks_inner(timeline, primary_source_id)
    if timeline.manual_cuts:
        trim_source_regions(timeline, timeline.manual_cuts, primary_source_id)
    # Camera angles last: they repoint pieces of the finished cut, never change it.
    apply_angle_spans(timeline, primary_source_id)


def _rebuild_primary_tracks_inner(timeline: Timeline, primary_source_id: str) -> None:
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

    if timeline.keep_full_source:
        _lay_full_source(timeline, primary_source_id, carried)
        return

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

def _uncut_word_span(timeline: Timeline, word) -> None:
    """Re-enabling a word must also lift any manual cut sitting over it.

    Otherwise `rebuild_primary_tracks` would rebuild the word back in and then
    immediately trim it straight back out again via `manual_cuts`.
    """
    if word.end_frame <= word.start_frame or not timeline.manual_cuts:
        return
    span = [word.start_frame, word.end_frame]
    survivors: List[List[int]] = []
    for region in timeline.manual_cuts:
        survivors.extend(_subtract_regions(region, [span]))
    timeline.manual_cuts = survivors


def toggle_word(timeline: Timeline, word_id: str, enabled: bool, primary_source_id: str) -> bool:
    """Toggle a word's enabled status and rebuild primary tracks."""
    for word in timeline.words:
        if word.id == word_id:
            word.enabled = enabled
            if enabled:
                _uncut_word_span(timeline, word)
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
            if enabled:
                _uncut_word_span(timeline, word)
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
    # Camera-angle pieces are read back in primary frames, and the pieces one
    # segment was split into for them are joined again; the angles are laid
    # back over the result at the end.
    spans_now = _merge_angle_pieces([(*v1_primary_span(timeline, i), i.anchor_word_id)
                                     for i in v1])
    segments: List[tuple] = []
    for p_start, p_end, anchor in spans_now:
        for start, end in _subtract_regions([p_start, p_end], collapsed):
            segments.append((start, end, anchor))
    if [(a, b) for a, b, _ in segments] == [(i.source_start_frame, i.source_end_frame) for i in v1]:
        return 0                                       # nothing overlapped (an edge trim keeps the
                                                       # count, so compare the spans themselves)

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

    apply_angle_spans(timeline, primary_source_id)
    timeline.recalculate_duration()
    timeline.revision += 1
    return len(collapsed)


def _merge_regions(regions: List[List[int]]) -> List[List[int]]:
    """Sort and coalesce overlapping/adjacent [start,end) spans."""
    merged: List[List[int]] = []
    for start, end in sorted([r for r in regions if r[1] > r[0]]):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def cut_program_range(timeline: Timeline, start_frame: int, end_frame: int,
                      primary_source_id: str) -> int:
    """Delete a span of the finished programme — V1/A1 *and* every other track.

    `start_frame`/`end_frame` are TIMELINE (programme) frames: a "delete this
    span of the finished video" edit, not a transcript one. The span is mapped
    across the current V1 items into source-frame spans, folded into
    `manual_cuts` (so the cut survives every future word-toggle rebuild) and
    applied to V1/A1 via `trim_source_regions`. Every other item then ripples:
    an item entirely inside the cut is dropped, an item entirely after it
    shifts left to close the gap, and an item straddling an edge keeps only its
    surviving part (its source region trimmed to match, for a media item).
    Words wholly inside the newly cut source span read as a manual strike.

    Returns the number of programme frames removed.
    """
    start_frame, end_frame = int(start_frame), int(end_frame)
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("cut_program_range: invalid or empty range")
    cut_length = end_frame - start_frame

    v1 = sorted([i for i in timeline.items if i.track == "V1"],
                key=lambda i: i.timeline_start_frame)
    new_spans: List[List[int]] = []
    for item in v1:
        overlap_start = max(item.timeline_start_frame, start_frame)
        overlap_end = min(item.timeline_end_frame, end_frame)
        if overlap_end <= overlap_start:
            continue
        primary_start, _primary_end = v1_primary_span(timeline, item)
        src_start = primary_start + (overlap_start - item.timeline_start_frame)
        src_end = primary_start + (overlap_end - item.timeline_start_frame)
        if src_end > src_start:
            new_spans.append([src_start, src_end])

    timeline.manual_cuts = _merge_regions(timeline.manual_cuts + new_spans)
    if new_spans:
        trim_source_regions(timeline, timeline.manual_cuts, primary_source_id)

    fps = timeline.fps_num / max(1, timeline.fps_den)

    def map_frame(frame: int) -> int:
        # Collapses anything inside the cut to its start, and shifts anything
        # after it left by the cut's length — the same ripple in one formula.
        if frame <= start_frame:
            return frame
        if frame >= end_frame:
            return frame - cut_length
        return start_frame

    kept_items: List[TimelineItem] = []
    for item in timeline.items:
        if item.track in ("V1", "A1"):
            kept_items.append(item)          # already rippled by trim_source_regions
            continue
        old_start, old_end = item.timeline_start_frame, item.timeline_end_frame
        if old_end <= start_frame:
            kept_items.append(item)          # entirely before the cut
            continue

        new_start, new_end = map_frame(old_start), map_frame(old_end)
        if new_end <= new_start:
            continue                          # entirely inside the cut -> dropped

        overlap_start = max(old_start, start_frame)
        overlap_end = min(old_end, end_frame)
        overlap_len = max(0, overlap_end - overlap_start)

        if item.kind == "media" and item.source_id and overlap_len > 0:
            if old_start < start_frame:
                item.source_end_frame -= overlap_len     # tail of the item cut away
            else:
                item.source_start_frame += overlap_len   # head of the item cut away

        shift_frames = old_start - new_start
        if shift_frames and item.atmosphere:
            shift_s = shift_frames / fps
            for fx in item.atmosphere:
                extra = fx.extra or {}
                if "start_s" in extra:
                    extra["start_s"] = float(extra["start_s"]) - shift_s
                if "end_s" in extra:
                    extra["end_s"] = float(extra["end_s"]) - shift_s

        item.timeline_start_frame = new_start
        item.timeline_end_frame = new_end
        kept_items.append(item)

    timeline.items = kept_items

    # A word wholly inside a freshly cut source span reads as a manual strike —
    # no reason/disfluency, exactly like toggle_word leaves one.
    for span_start, span_end in new_spans:
        for word in timeline.words:
            if word.enabled and word.start_frame >= span_start and word.end_frame <= span_end:
                word.enabled = False

    timeline.recalculate_duration()
    timeline.revision += 1
    return cut_length


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


# ---------------------------------------------------------------------------
# Camera angles (MULTICAM_SYNC_PLAN.md §3-5)
#
# Every source's frames are counted at the TIMELINE fps (the compiler, clip_ops
# and program_time all read source_start_frame that way; the per-clip fps=
# filter resamples the picture), so moving a piece of the cut from the primary
# to a synced angle is a constant frame shift:
#     angle_frame = primary_frame + round(sync_offset_s * tl_fps)
# whatever rate the angle was recorded at.
# ---------------------------------------------------------------------------

_SENTENCE_END = (".", "?", "!", "।", "॥", "…", ";", ":")
RHYTHM_GAP_SECONDS = 0.6


def angle_offset_frames(timeline: Timeline, source: SourceFile) -> int:
    """Frames to add to a primary-source frame to land on the same instant of `source`."""
    fps = timeline.fps_num / max(1, timeline.fps_den)
    return int(round(float(source.sync_offset_s or 0.0) * fps))


def _angle_frames(timeline: Timeline, source: SourceFile) -> int:
    fps = timeline.fps_num / max(1, timeline.fps_den)
    return int(round(float(source.duration_seconds or 0.0) * fps))


def v1_primary_span(timeline: Timeline, item: TimelineItem) -> Tuple[int, int]:
    """A V1 item's [start, end) in PRIMARY-source frames.

    The same as its own source frames, except for a piece that shows a synced
    camera angle, whose frames are on the angle's clock. Everything that maps the
    transcript through V1 (captions, cut_program_range, kept words) reads V1
    through this so an angle piece still lines up with its words."""
    source = timeline.sources.get(item.source_id or "")
    if source is not None and source.synced_to:
        shift = angle_offset_frames(timeline, source)
        return item.source_start_frame - shift, item.source_end_frame - shift
    return item.source_start_frame, item.source_end_frame


def is_angle_switch(timeline: Timeline, before: TimelineItem, after: TimelineItem) -> bool:
    """Two adjacent V1 clips of different sources that are one continuous moment of
    the primary recording: a change of camera, not a jump cut."""
    if not before.source_id or not after.source_id or before.source_id == after.source_id:
        return False
    if before.timeline_end_frame != after.timeline_start_frame:
        return False
    return v1_primary_span(timeline, before)[1] == v1_primary_span(timeline, after)[0]


def _merge_angle_pieces(spans: List[tuple]) -> List[tuple]:
    """Join contiguous (start, end, anchor) pieces of one anchored segment."""
    merged: List[list] = []
    for start, end, anchor in spans:
        if (merged and anchor is not None and merged[-1][2] == anchor
                and merged[-1][1] == start):
            merged[-1][1] = end
        else:
            merged.append([start, end, anchor])
    return [tuple(m) for m in merged]


def angle_sources(timeline: Timeline, primary_source_id: str) -> List[SourceFile]:
    """The camera angles synced to the primary, in the order they were added."""
    return [s for s in timeline.sources.values()
            if s.synced_to == primary_source_id and s.id != primary_source_id]


def _clean_spans(timeline: Timeline, primary_source_id: str) -> List[Dict[str, Any]]:
    spans = []
    for span in timeline.angle_spans or []:
        try:
            start, end, sid = int(span["start"]), int(span["end"]), span.get("source_id")
        except (KeyError, TypeError, ValueError):
            continue
        source = timeline.sources.get(sid or "")
        if end > start and source is not None and sid != primary_source_id and source.synced_to:
            spans.append({"start": start, "end": end, "source_id": sid})
    return sorted(spans, key=lambda s: s["start"])


def _unapply_angle_spans(timeline: Timeline, primary_source_id: str) -> None:
    """Put every auto V1 angle piece back on the primary and rejoin the split segments."""
    fresh: List[TimelineItem] = []
    pending: Optional[TimelineItem] = None
    for item in timeline.items:
        if item.track != "V1" or item.origin != "auto" or item.kind != "media":
            if pending is not None:
                fresh.append(pending)
                pending = None
            fresh.append(item)
            continue
        source = timeline.sources.get(item.source_id or "")
        if source is not None and source.synced_to == primary_source_id:
            start, end = v1_primary_span(timeline, item)
            item.source_id = primary_source_id
            item.source_start_frame, item.source_end_frame = start, end
        if (pending is not None and item.source_id == pending.source_id
                and item.anchor_word_id is not None
                and item.anchor_word_id == pending.anchor_word_id
                and pending.source_end_frame == item.source_start_frame
                and pending.timeline_end_frame == item.timeline_start_frame):
            pending.source_end_frame = item.source_end_frame
            pending.timeline_end_frame = item.timeline_end_frame
            continue
        if pending is not None:
            fresh.append(pending)
        pending = item
    if pending is not None:
        fresh.append(pending)
    timeline.items = fresh


def apply_angle_spans(timeline: Timeline, primary_source_id: str) -> int:
    """Show the chosen camera angle on V1 for every `angle_spans` span.

    Splits the transcript-built (origin "auto") V1 clips of the primary at the
    span edges and repoints the pieces inside a span at the angle, shifted onto
    its clock. A piece the angle did not record (it started late or stopped
    early) stays on the primary. A1 is never touched -- the sound stays
    continuous on the primary (or its audio_override) -- and link_id, anchor
    word, transform and grade ride along on every piece. Idempotent. Returns
    the number of V1 pieces now on an angle.
    """
    spans = _clean_spans(timeline, primary_source_id)
    if not spans:
        return 0
    covers: Dict[str, Tuple[int, int, int]] = {}
    for span in spans:
        source = timeline.sources[span["source_id"]]
        shift = angle_offset_frames(timeline, source)
        # Primary frames the angle has picture for: [-shift, angle_len - shift).
        covers[source.id] = (shift, -shift, _angle_frames(timeline, source) - shift)

    placed = 0
    out: List[TimelineItem] = []
    for item in timeline.items:
        if (item.track != "V1" or item.origin != "auto" or item.kind != "media"
                or item.source_id != primary_source_id
                or item.source_end_frame <= item.source_start_frame):
            out.append(item)
            continue
        s, e = item.source_start_frame, item.source_end_frame
        edges = {s, e}
        for span in spans:
            if span["start"] < e and span["end"] > s:
                _shift, lo, hi = covers[span["source_id"]]
                for edge in (span["start"], span["end"], lo, hi):
                    if s < edge < e:
                        edges.add(edge)
        bounds = sorted(edges)
        pieces: List[list] = []
        for a, b in zip(bounds, bounds[1:]):
            target = None
            for span in spans:
                if span["start"] <= a and b <= span["end"]:
                    _shift, lo, hi = covers[span["source_id"]]
                    if lo <= a and b <= hi:
                        target = span["source_id"]
                    break
            if pieces and pieces[-1][2] == target:
                pieces[-1][1] = b
            else:
                pieces.append([a, b, target])
        if all(target is None for _a, _b, target in pieces):
            out.append(item)
            continue
        for index, (a, b, target) in enumerate(pieces):
            update: Dict[str, Any] = {
                "id": item.id if index == 0 else f"{item.id}_ang{index}",
                "source_start_frame": a,
                "source_end_frame": b,
                "timeline_start_frame": item.timeline_start_frame + (a - s),
                "timeline_end_frame": item.timeline_start_frame + (b - s),
            }
            if index > 0:
                update["transition"] = None   # a switch inside one take is a hard cut
            if target is not None:
                shift = covers[target][0]
                update.update(source_id=target, source_start_frame=a + shift,
                              source_end_frame=b + shift)
                placed += 1
            out.append(item.model_copy(deep=True, update=update))
    timeline.items = out
    return placed


def _reapply_angles(timeline: Timeline, primary_source_id: str) -> None:
    _unapply_angle_spans(timeline, primary_source_id)
    apply_angle_spans(timeline, primary_source_id)
    timeline.recalculate_duration()
    timeline.revision += 1


def _program_to_primary(timeline: Timeline, start_frame: int,
                        end_frame: int) -> List[List[int]]:
    """A programme range as primary-source spans (cut_program_range's mapping).

    Material the cut removed between two consecutive pieces of the range is
    bridged, so one programme range is one span -- the same shape auto_angles
    writes -- and a word restored inside it later shows the same angle."""
    spans: List[List[int]] = []
    for item in sorted([i for i in timeline.items if i.track == "V1" and i.kind == "media"],
                       key=lambda i: i.timeline_start_frame):
        lo = max(item.timeline_start_frame, start_frame)
        hi = min(item.timeline_end_frame, end_frame)
        if hi <= lo:
            continue
        p_start, _p_end = v1_primary_span(timeline, item)
        span = [p_start + (lo - item.timeline_start_frame), p_start + (hi - item.timeline_start_frame)]
        if spans and span[0] >= spans[-1][1]:
            spans[-1][1] = span[1]                 # bridge the removed gap
        else:
            spans.append(span)
    return _merge_regions(spans)


def _write_spans(timeline: Timeline, primary_source_id: str,
                 ranges: List[List[int]], source_id: Optional[str]) -> None:
    """Overwrite `ranges` (primary frames) with `source_id` (None = the primary)."""
    kept: List[Dict[str, Any]] = []
    for span in _clean_spans(timeline, primary_source_id):
        for a, b in _subtract_regions([span["start"], span["end"]], ranges):
            kept.append({"start": a, "end": b, "source_id": span["source_id"]})
    if source_id is not None:
        kept.extend({"start": a, "end": b, "source_id": source_id} for a, b in ranges)
    kept.sort(key=lambda s: s["start"])
    merged: List[Dict[str, Any]] = []
    for span in kept:
        if merged and merged[-1]["source_id"] == span["source_id"] \
                and span["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    timeline.angle_spans = merged


def set_angle(timeline: Timeline, start_frame: int, end_frame: int,
              source_id: Optional[str], primary_source_id: str) -> List[Dict[str, Any]]:
    """Show `source_id` for the PROGRAMME range [start_frame, end_frame).

    The range is mapped through V1 to primary-source spans (like
    cut_program_range) and stored in `angle_spans`, overwriting whatever was
    chosen there before; `source_id` None (or the primary) puts the primary back.
    The angle layer is then laid again over the current cut.
    """
    start_frame, end_frame = int(start_frame), int(end_frame)
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("set_angle: invalid or empty range")
    if source_id == primary_source_id:
        source_id = None
    if source_id is not None:
        source = timeline.sources.get(source_id)
        if source is None:
            raise ValueError(f"set_angle: no source {source_id!r}")
        if source.synced_to != primary_source_id:
            raise ValueError(f"set_angle: {source_id} is not a camera angle synced to the primary")
    ranges = _program_to_primary(timeline, start_frame, end_frame)
    if not ranges:
        raise ValueError("set_angle: the range holds no V1 footage")
    _write_spans(timeline, primary_source_id, ranges, source_id)
    _reapply_angles(timeline, primary_source_id)
    return timeline.angle_spans


def clear_angles(timeline: Timeline, primary_source_id: str) -> None:
    timeline.angle_spans = []
    _reapply_angles(timeline, primary_source_id)


def program_angle_spans(timeline: Timeline) -> List[Dict[str, Any]]:
    """The programme view of the angle choice: one entry per run of V1 on an angle,
    {start, end (programme frames), source_id, source_start, source_end (primary frames)}."""
    out: List[Dict[str, Any]] = []
    v1 = sorted([i for i in timeline.items if i.track == "V1" and i.kind == "media"],
                key=lambda i: i.timeline_start_frame)
    for item in v1:
        source = timeline.sources.get(item.source_id or "")
        if source is None or not source.synced_to:
            continue
        p_start, p_end = v1_primary_span(timeline, item)
        if out and out[-1]["source_id"] == item.source_id \
                and out[-1]["end"] == item.timeline_start_frame:
            out[-1]["end"] = item.timeline_end_frame
            out[-1]["source_end"] = p_end
        else:
            out.append({"start": item.timeline_start_frame, "end": item.timeline_end_frame,
                        "source_id": item.source_id, "source_start": p_start,
                        "source_end": p_end})
    return out


def _covered_by(timeline: Timeline, source: SourceFile, p_start: int, p_end: int) -> bool:
    shift = angle_offset_frames(timeline, source)
    return -shift <= p_start and p_end <= _angle_frames(timeline, source) - shift


def _rhythm_points(timeline: Timeline, v1: List[TimelineItem]) -> List[Tuple[int, bool]]:
    """(primary frame, at a jump cut) of each sentence/segment boundary that plays."""
    fps = timeline.fps_num / max(1, timeline.fps_den)
    gap_frames = int(round(RHYTHM_GAP_SECONDS * fps))
    words = sorted((w for w in timeline.words if w.enabled), key=lambda w: w.start_frame)
    spans = [v1_primary_span(timeline, i) for i in v1]
    points: List[Tuple[int, bool]] = []
    for prev, nxt in zip(words, words[1:]):
        ends_sentence = (prev.text or "").strip().endswith(_SENTENCE_END)
        if not ends_sentence and nxt.start_frame - prev.end_frame < gap_frames:
            continue
        index = next((k for k, (a, b) in enumerate(spans) if a <= nxt.start_frame < b), None)
        if index is None:
            continue
        p_start = spans[index][0]
        if p_start >= prev.end_frame and index > 0:
            points.append((p_start, True))        # the boundary is a jump cut: switch on it
        else:
            points.append((max(p_start, (prev.end_frame + nxt.start_frame) // 2), False))
    return points


def auto_angles(timeline: Timeline, primary_source_id: str, mode: str,
                min_shot_s: float = 2.5, cadence_s: float = 8.0) -> Dict[str, Any]:
    """Choose the angle spans automatically and lay them.

    hide_cuts: every jump cut becomes an angle change (rotating primary, angle 1,
    angle 2, ...), so the head-position snap reads as a camera cut. A cut under a
    B-roll window is already hidden and gets no switch; no shot is shorter than
    `min_shot_s` (a cut that would make one stays a plain jump cut).
    rhythm: switch at sentence/segment boundaries (end punctuation, or a gap of
    0.6 s or more) roughly every `cadence_s`.
    off: clear every span.
    Returns {"angle_spans", "summary": {switches, cuts_hidden, cuts_total,
    cuts_by_angle, cuts_by_broll, mode}}.
    """
    from presentation.placement import broll_windows, jump_cut_times

    mode = (mode or "").strip().lower()
    if mode not in ("hide_cuts", "rhythm", "off"):
        raise ValueError(f"auto_angles: unknown mode {mode!r} (hide_cuts | rhythm | off)")
    fps = timeline.fps_num / max(1, timeline.fps_den)

    # Decide on the cut without any angle on it.
    timeline.angle_spans = []
    _reapply_angles(timeline, primary_source_id)
    cuts = jump_cut_times(timeline)
    windows = broll_windows(timeline)

    def under_broll(t: float) -> bool:
        return any(start + 0.1 <= t <= end - 0.1 for start, end in windows)

    cuts_by_broll = sum(1 for t in cuts if under_broll(t))
    summary: Dict[str, Any] = {"mode": mode, "switches": 0, "cuts_total": len(cuts),
                               "cuts_by_angle": 0, "cuts_by_broll": cuts_by_broll,
                               "cuts_hidden": cuts_by_broll}
    if mode == "off":
        return {"angle_spans": [], "summary": summary}

    angles = angle_sources(timeline, primary_source_id)
    if not angles:
        raise ValueError("auto_angles: no camera angle is synced to the primary")
    v1 = sorted([i for i in timeline.items if i.track == "V1" and i.enabled and i.kind == "media"],
                key=lambda i: i.timeline_start_frame)
    if not v1:
        return {"angle_spans": [], "summary": summary}
    spans_p = [v1_primary_span(timeline, i) for i in v1]

    def prog_s(p_frame: int) -> Optional[float]:
        for (a, b), item in zip(spans_p, v1):
            if a <= p_frame < b:
                return (item.timeline_start_frame + (p_frame - a)) / fps
        return None

    program_start = v1[0].timeline_start_frame / fps
    program_end = v1[-1].timeline_end_frame / fps
    if mode == "hide_cuts":
        candidates = [(spans_p[k + 1][0], True) for k in range(len(v1) - 1)]
    else:
        candidates = _rhythm_points(timeline, v1)
    min_gap = max(min_shot_s, cadence_s) if mode == "rhythm" else min_shot_s

    rotation: List[Optional[SourceFile]] = [None, *angles]
    current, last_switch = 0, program_start
    switches: List[Tuple[int, int, bool]] = []     # (primary frame, rotation index, at a cut)
    last_p_end = spans_p[-1][1]
    for p_frame, at_cut in sorted(set(candidates)):
        t = prog_s(p_frame)
        if t is None or under_broll(t):
            continue
        if t - last_switch < min_gap - 1e-6 or program_end - t < min_shot_s - 1e-6:
            continue
        # The next angle in the rotation that has picture for the shot it would open.
        shot_end = min(last_p_end, p_frame + int(round(min_shot_s * fps)))
        chosen = None
        for step in range(1, len(rotation)):
            index = (current + step) % len(rotation)
            source = rotation[index]
            if source is None or _covered_by(timeline, source, p_frame, shot_end):
                chosen = index
                break
        if chosen is None:
            continue
        switches.append((p_frame, chosen, at_cut))
        current, last_switch = chosen, t

    spans: List[Dict[str, Any]] = []
    for n, (p_frame, index, _at_cut) in enumerate(switches):
        source = rotation[index]
        if source is None:
            continue
        end = switches[n + 1][0] if n + 1 < len(switches) else last_p_end
        if end > p_frame:
            spans.append({"start": p_frame, "end": end, "source_id": source.id})
    timeline.angle_spans = spans
    _reapply_angles(timeline, primary_source_id)

    by_angle = sum(1 for _p, _i, at_cut in switches if at_cut)
    summary.update(switches=len(switches), cuts_by_angle=by_angle,
                   cuts_hidden=by_angle + cuts_by_broll)
    return {"angle_spans": timeline.angle_spans, "summary": summary}
