/**
 * Transitions browser.
 *
 * A transition lives on the *junction between two programme clips*, so there are
 * only ever two things it can be attached to: one clip's cut-in, or the
 * programme default that every other cut inherits. That choice is the first
 * control in the panel rather than a hidden consequence of whatever happens to be
 * selected, because "why did that dissolve appear on all forty cuts" is
 * otherwise impossible to answer from the UI.
 *
 * The panel also refuses to lie about what it just did. A timeline with one V1
 * clip has no junctions at all, an overlay clip on V2 never forms one, and a
 * transition longer than the clips it joins gets clamped at render — in every
 * one of those cases the setting used to be accepted, saved, and then quietly
 * dropped by the compiler, which is exactly what "I can't apply transitions"
 * looks like from the outside. `lib/programme.ts` holds the renderer's own rule
 * so the status line here can say what will actually be drawn.
 *
 * The whole xfade set is here, not just the named presets. Fifty-eight names is a
 * wall, so they keep the renderer's own grouping and can be filtered.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { listPresets, setTransition, PresetCatalogue } from '../hooks/api';
import { PresetChips, Section, Slider } from './controls';
import { junctionCount, maxUsableDuration, programmeClips } from '../lib/programme';

// Turning "wipetl" into "Wipe top-left" from the parts the names are built out
// of, rather than a 58-line lookup table that would fall out of step with the
// renderer's catalogue the first time a name is added to it.
const PREFIX: Record<string, string> = {
  wipe: 'Wipe', slide: 'Slide', cover: 'Cover', reveal: 'Reveal', smooth: 'Smooth',
  circle: 'Circle', vert: 'Vertical', horz: 'Horizontal', diag: 'Diagonal',
  rect: 'Rectangle', fade: 'Fade', squeeze: 'Squeeze', zoom: 'Zoom',
  hl: 'Left', hr: 'Right', vu: 'Up', vd: 'Down', h: 'Horizontal',
};
const SUFFIX: Record<string, string> = {
  left: 'left', right: 'right', up: 'up', down: 'down',
  tl: 'top-left', tr: 'top-right', bl: 'bottom-left', br: 'bottom-right',
  open: 'open', close: 'close', crop: 'crop', slice: 'slice', wind: 'wind',
  black: 'to black', white: 'to white', grays: 'to grey',
  fast: 'fast', slow: 'slow', in: 'in', h: 'horizontal', v: 'vertical', blur: 'blur',
};
const PREFIXES = Object.keys(PREFIX).sort((a, b) => b.length - a.length);

export function prettyTransition(name: string): string {
  for (const p of PREFIXES) {
    if (name.startsWith(p) && name.length > p.length) {
      const rest = SUFFIX[name.slice(p.length)];
      if (rest) return `${PREFIX[p]} ${rest}`;
    }
  }
  return name.charAt(0).toUpperCase() + name.slice(1);
}

const DEFAULT_DURATION = 0.5;

export default function TransitionsPanel() {
  const { project, updateProject, selectedClipId, setError } = useProjectStore();
  const [presets, setPresets] = useState<PresetCatalogue | null>(null);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState('');
  // "clip" targets the selected clip's cut-in; "program" targets every other cut.
  const [scope, setScope] = useState<'clip' | 'program'>('clip');
  const timer = useRef<any>(null);

  const projectId = project?.id;
  const timeline = project?.timeline as any;
  const fps = timeline?.fps_num ? timeline.fps_num / (timeline.fps_den || 1) : 30;

  useEffect(() => {
    listPresets().then(setPresets).catch(() => setPresets(null));
  }, []);

  const item = useMemo(
    () => (timeline?.items || []).find((i: any) => i.id === selectedClipId) || null,
    [timeline, selectedClipId],
  );

  // What the renderer will do, not what the timeline happens to store.
  const cuts = useMemo(() => junctionCount(timeline), [timeline]);
  // Matched on the *lane* id: selecting a compound selects the group, whose
  // junction is its leading child's.
  const clipIndex = useMemo(
    () => programmeClips(timeline).findIndex(
      (c) => c.leadsLane && (c.laneId ?? c.id) === selectedClipId),
    [timeline, selectedClipId],
  );

  // Only a V1 programme clip with something before it forms a junction. An
  // overlay, a text clip or the very first clip can be *given* a transition, and
  // the compiler will never look at it.
  const clipTargetable = clipIndex > 0;
  const clipBlockedReason = !item
    ? 'Select a clip on the timeline first'
    : clipIndex === 0
      ? 'This is the first clip on V1 — there is nothing before it to transition from'
      : `Transitions join the programme clips on V1; this one is a ${item.kind === 'media' ? item.track : item.kind} clip`;

  const targetItemId = scope === 'clip' && clipTargetable ? selectedClipId : null;
  const onProgramme = !targetItemId;

  const active = targetItemId ? item?.transition : timeline?.default_transition;
  const activeType: string | null = active?.type ?? null;
  const duration: number = active?.duration ?? DEFAULT_DURATION;

  // A transition cannot be longer than the clips it joins; the renderer clamps
  // it, so a 2s dissolve between two jump cuts arrives as a flicker. Saying so
  // here is the difference between a setting and a surprise.
  const usableLimit = useMemo(() => maxUsableDuration(timeline, fps), [timeline, fps]);
  const clamped = activeType && onProgramme && cuts > 0
    && Number.isFinite(usableLimit) && duration > usableLimit + 0.005;

  const run = useCallback(async (fn: () => Promise<any>) => {
    setBusy(true);
    try {
      const res = await fn();
      if (res?.timeline) updateProject({ timeline: res.timeline });
    } catch (err: any) {
      setError(err.message || 'Could not set the transition');
    } finally {
      setBusy(false);
    }
  }, [updateProject, setError]);

  const apply = useCallback((type: string) => {
    if (!projectId) return;
    // Carry the length already in use, so browsing types does not silently reset
    // a duration the user has just dialled in.
    run(() => setTransition(projectId, targetItemId, {
      updates: { type, duration: duration > 0 ? duration : DEFAULT_DURATION },
    }));
  }, [projectId, targetItemId, duration, run]);

  const applyPreset = useCallback((id: string) => {
    if (projectId) run(() => setTransition(projectId, targetItemId, { preset: id }));
  }, [projectId, targetItemId, run]);

  const setLength = useCallback((value: number) => {
    if (!projectId || !activeType) return;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => run(() => setTransition(
      projectId, targetItemId, { updates: { duration: value } })), 220);
  }, [projectId, targetItemId, activeType, run]);

  const clear = useCallback(() => {
    if (projectId) run(() => setTransition(projectId, targetItemId, { reset: true }));
  }, [projectId, targetItemId, run]);

  // Length is part of a preset's identity, not decoration: "Hard Cut",
  // "Crossfade" and "Dip to Black" are all built on `fade` and differ only in
  // how long they run, so matching on type alone lit up the wrong chip.
  const activePreset = useMemo(() => {
    const list = presets?.transition ?? [];
    if (!activeType) return list.find((p) => (p.duration ?? 0) === 0)?.id ?? null;
    return list.find((p) => (p as any).type === activeType
      && Math.abs((p.duration ?? 0) - duration) < 0.02)?.id ?? null;
  }, [presets, activeType, duration]);

  const catalogue = presets?.transition_catalogue ?? [];
  const total = catalogue.reduce((n, g) => n + g.transitions.length, 0);

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return catalogue;
    return catalogue
      .map((g) => ({
        group: g.group,
        transitions: g.transitions.filter(
          (n) => n.includes(needle)
            || prettyTransition(n).toLowerCase().includes(needle)
            || g.group.toLowerCase().includes(needle)),
      }))
      .filter((g) => g.transitions.length > 0);
  }, [catalogue, query]);

  if (!projectId) {
    return <div className="fx-empty">Open a project to add transitions.</div>;
  }

  // The one case where nothing you pick can possibly show up: a programme that
  // is still a single unbroken clip has no cuts to put a transition on.
  const noCuts = onProgramme && cuts === 0;

  return (
    <div className="fx-panel">
      <div className="fx-scope">
        <button className={`insp-chip ${scope === 'clip' && clipTargetable ? 'active' : ''}`}
          disabled={!clipTargetable}
          title={clipTargetable
            ? 'Applies only where this clip joins the one before it'
            : clipBlockedReason}
          onClick={() => setScope('clip')}>
          Selected clip{clipTargetable ? ` (cut ${clipIndex})` : ''}
        </button>
        <button className={`insp-chip ${onProgramme ? 'active' : ''}`}
          title="The transition every cut uses unless a clip overrides it"
          onClick={() => setScope('program')}>
          Every cut{cuts ? ` (${cuts})` : ''}
        </button>
      </div>

      {noCuts && (
        <div className="fx-warn">
          <strong>This programme has no cuts yet.</strong> A transition joins two
          clips, and V1 holds {programmeClips(timeline).length === 1 ? 'a single clip' : 'nothing to join'} —
          so whatever you pick here is stored but never drawn. Split the clip
          (<strong>✂ Split</strong> on the timeline) or run <strong>Auto Edit</strong> first;
          the choice below then applies to every cut it makes.
        </div>
      )}

      <div className="fx-status">
        <span>
          {activeType
            ? <>Now: <strong>{prettyTransition(activeType)}</strong> · {duration.toFixed(2)}s
              {onProgramme && cuts > 0 && <> — on all {cuts} cut{cuts === 1 ? '' : 's'}</>}</>
            : <>Now: <strong>hard cut</strong> — no transition here.</>}
        </span>
        {activeType && (
          <button className="btn btn-xs" disabled={busy} onClick={clear} title="Back to a hard cut">
            Clear
          </button>
        )}
      </div>

      {activeType && (
        <Slider label="Length" value={duration} min={0.1} max={3.0}
          onChange={setLength} format={(v) => `${v.toFixed(2)}s`}
          hint={clamped
            ? `Too long for the shortest cut here — the render will clamp it to about ${usableLimit.toFixed(2)}s`
            : 'Clamped at render time to what the two clips can actually cover'} />
      )}

      <Section title="Popular">
        {presets
          ? <PresetChips items={presets.transition} active={activePreset} onPick={applyPreset} />
          : <div className="text-xs text-muted">Loading…</div>}
      </Section>

      <Section title={`All transitions${total ? ` (${total})` : ''}`}>
        <input className="fx-search" type="search" placeholder="Filter — wipe, slide, circle…"
          value={query} onChange={(e) => setQuery(e.target.value)} />
        {groups.length === 0 && (
          <div className="text-xs text-muted">Nothing matches that.</div>
        )}
        {groups.map((g) => (
          <div key={g.group} className="fx-group">
            <div className="insp-group-title">{g.group}</div>
            <div className="fx-grid">
              {g.transitions.map((name) => (
                <button key={name} title={name} disabled={busy}
                  className={`fx-tile ${activeType === name ? 'active' : ''}`}
                  onClick={() => apply(name)}>
                  {prettyTransition(name)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </Section>

      <div className="fx-footnote">
        Transitions are drawn by the renderer, so the preview above keeps showing
        hard cuts however many you set. The timeline marks every cut that has one —
        look for the <span className="fx-legend-swatch" /> stripe on a clip's leading
        edge — and <strong>Render</strong> is where you see them move.
      </div>
    </div>
  );
}
