/**
 * Where a transition can actually land.
 *
 * A transition is not a property of a clip, it is a property of the *junction*
 * between two programme clips — and the renderer only builds junctions on V1, in
 * timeline order, skipping the first clip because nothing precedes it
 * (`_junction_transitions` in backend/render/compiler.py). Everything else — a
 * V2 overlay, a text clip, the first clip on the track — can be given a
 * transition that is stored, saved, and then silently ignored at render.
 *
 * That gap is what made transitions look broken: the panel accepted the click,
 * the timeline said "Now: Dissolve", and the export came out on hard cuts. The
 * rule lives here once, so the panel and the timeline can both say what the
 * renderer is going to do rather than what was merely stored.
 */

export interface ProgrammeClip {
  id: string;
  track: string;
  kind?: string;
  enabled?: boolean;
  timeline_start_frame: number;
  timeline_end_frame: number;
  transition?: { type: string; duration: number } | null;
  children?: ProgrammeClip[];
  /** The id of the block the timeline actually draws. Same as `id` for a plain
   *  clip; for a compound's children it is the group's id, since the lane shows
   *  one block and has nowhere to hang a marker for an interior cut. */
  laneId?: string;
  /** True for a compound's leading child — the only one whose junction is
   *  visible on the lane. */
  leadsLane?: boolean;
}

export interface EffectiveTransition {
  type: string;
  duration: number;
  /** True when the clip overrides the programme default with its own. */
  own: boolean;
}

/**
 * The V1 clips the renderer will concatenate, in timeline order.
 *
 * Compounds are expanded exactly as `flatten_items` does, because the renderer
 * cuts between a group's children as readily as between two loose clips — count
 * a compound as one clip and the junction total, and therefore everything the
 * panel says, comes out wrong.
 */
export function programmeClips(timeline: any): ProgrammeClip[] {
  if (timeline?.tracks?.V1?.hidden) return [];
  const flat: ProgrammeClip[] = [];
  for (const item of (timeline?.items ?? []) as ProgrammeClip[]) {
    if ((item.track || '').toUpperCase() !== 'V1' || item.enabled === false) continue;
    if (item.kind === 'compound') {
      let lead = true;
      for (const child of item.children ?? []) {
        if (child.enabled === false) continue;
        flat.push({
          ...child,
          track: item.track,
          laneId: item.id,
          leadsLane: lead,
          // Children are stored relative to the group.
          timeline_start_frame: item.timeline_start_frame + child.timeline_start_frame,
          timeline_end_frame: item.timeline_start_frame + child.timeline_end_frame,
          transition: child.transition ?? (lead ? item.transition : null),
        });
        lead = false;
      }
      continue;
    }
    if ((item.kind ?? 'media') !== 'media') continue;
    flat.push({ ...item, laneId: item.id, leadsLane: true });
  }
  return flat.sort((a, b) => a.timeline_start_frame - b.timeline_start_frame);
}

/** How many cuts a programme-wide transition would actually be drawn on. */
export function junctionCount(timeline: any): number {
  return Math.max(0, programmeClips(timeline).length - 1);
}

/**
 * The transition the renderer will use at each junction, keyed by the id of the
 * timeline block it leads *into*. A clip's own transition wins over the
 * programme default; a zero duration means a hard cut and produces no entry.
 *
 * Junctions inside a compound are real — the renderer cuts there — but the lane
 * draws the group as one block, so only the leading one gets a marker.
 */
export function effectiveTransitions(timeline: any): Record<string, EffectiveTransition> {
  const fallback = timeline?.default_transition;
  const out: Record<string, EffectiveTransition> = {};
  programmeClips(timeline).forEach((item, index) => {
    if (index === 0) return;                  // nothing precedes the first clip
    if (!item.leadsLane) return;              // an interior cut has nowhere to show
    const chosen = item.transition || fallback;
    if (chosen && chosen.duration > 0) {
      out[item.laneId ?? item.id] = {
        type: chosen.type, duration: chosen.duration, own: !!item.transition,
      };
    }
  });
  return out;
}

/**
 * The longest transition every junction can actually cover.
 *
 * `xfade` needs both sides to span the overlap, so the renderer clamps each
 * junction to `min(before, after) - 0.04` — which is why a 2s dissolve between
 * two short jump cuts comes back as a flicker. Returns Infinity when there is
 * nothing to clamp against.
 */
export function maxUsableDuration(timeline: any, fps: number): number {
  const clips = programmeClips(timeline);
  if (clips.length < 2 || !fps) return Infinity;
  const seconds = (c: ProgrammeClip) =>
    (c.timeline_end_frame - c.timeline_start_frame) / fps;
  let limit = Infinity;
  for (let i = 1; i < clips.length; i++) {
    limit = Math.min(limit, Math.min(seconds(clips[i - 1]), seconds(clips[i])) - 0.04);
  }
  return Math.max(0, limit);
}
