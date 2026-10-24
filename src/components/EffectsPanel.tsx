/**
 * Effects browser: the generated atmosphere layers — rain, glitch, VHS, shake,
 * light leaks and the rest.
 *
 * Everything here is built procedurally by the renderer, so there is no stock
 * footage to source and an effect costs nothing until the render runs.
 *
 * An effect goes on the whole programme, or on one adjustment layer — an
 * adjustment layer being a clip with no picture that treats every layer beneath
 * it, which is how an effect gets windowed to a few seconds instead of running
 * for the length of the video. The panel offers to create that layer rather than
 * making the user find it in the timeline toolbar first.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import {
  addAdjustment, addEffect, listPresets, removeEffect, updateEffect,
  AtmosphereEffect, PresetCatalogue, ProgressBarChapter,
} from '../hooks/api';
import { ColorField, PresetChips, Section, Slider, toHex } from './controls';

/** The three effects that actually read `color`; the rest are grey layers, where
 *  a tint control would be a knob wired to nothing. */
const TINTABLE = new Set(['sunlight', 'light_leak', 'flash']);

/** Effects that read as a hit rather than as weather — better on a short layer. */
const MOMENTARY = new Set(['flash', 'shake', 'glitch']);

const ADJUSTMENT_SECONDS = 2.0;

/** The progress bar's own params, all in `extra` (merged key by key by the backend). */
function ProgressBarEditor({ effect, busy, packChapters, onSet, onSchedule }: {
  effect: AtmosphereEffect;
  busy: boolean;
  packChapters: ProgressBarChapter[] | null;
  onSet: (extra: Record<string, any>) => void;
  onSchedule: (extra: Record<string, any>) => void;
}) {
  const extra = effect.extra || {};
  const chapters: ProgressBarChapter[] = Array.isArray(extra.chapters) ? extra.chapters : [];
  return (
    <div className="fx-progress-editor">
      <label className="fx-progress-row">
        <span>Position</span>
        <select value={extra.position === 'top' ? 'top' : 'bottom'} aria-label="Progress bar position"
          disabled={busy} onChange={(e) => onSet({ position: e.target.value })}>
          <option value="bottom">Bottom</option>
          <option value="top">Top</option>
        </select>
      </label>
      <Slider label="Height" value={Number(extra.height_px ?? 6)} min={2} max={40} step={1}
        hint="Bar thickness in 1080p pixels (scaled with the canvas)"
        onChange={(v) => onSchedule({ height_px: Math.round(v) })}
        format={(v) => `${Math.round(v)} px`} />
      <label className="fx-progress-row">
        <span>Chapter ticks</span>
        <input type="checkbox" checked={extra.ticks !== false} aria-label="Chapter ticks"
          disabled={busy} onChange={(e) => onSet({ ticks: e.target.checked })} />
      </label>
      <label className="fx-progress-row">
        <span>Label</span>
        <select value={extra.label === 'none' ? 'none' : 'name'} aria-label="Progress bar label"
          disabled={busy} onChange={(e) => onSet({ label: e.target.value })}>
          <option value="name">Current chapter's name</option>
          <option value="none">None</option>
        </select>
      </label>
      <div className="fx-progress-chapters text-xs text-muted">
        {chapters.length
          ? `${chapters.length} chapter${chapters.length === 1 ? '' : 's'}: ${chapters.map((c) => c.name).join(' · ')}`
          : 'No chapters: the bar fills without ticks or a label.'}
      </div>
      <button className="btn btn-xs" disabled={busy || !packChapters}
        title={packChapters
          ? 'Copy the chapters the presentation pass wrote for the metadata pack (times already shifted by the cold open)'
          : 'No chapters yet: run the presentation pass with Story structure on and the progress bar enabled'}
        onClick={() => packChapters && onSet({ chapters: packChapters })}>
        Use chapters from the metadata pack
      </button>
    </div>
  );
}

export default function EffectsPanel() {
  const { project, updateProject, selectedClipId, currentTime, setError } = useProjectStore();
  const [presets, setPresets] = useState<PresetCatalogue | null>(null);
  const [busy, setBusy] = useState(false);
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
  // Only an adjustment layer can hold its own effects. Any other selection means
  // the effect belongs to the programme.
  const isAdjustment = item?.kind === 'adjustment';
  const owner: string | null = isAdjustment ? selectedClipId : null;
  const effects: AtmosphereEffect[] =
    (isAdjustment ? item.atmosphere : timeline?.effects) || [];

  const run = useCallback(async (fn: () => Promise<any>) => {
    setBusy(true);
    try {
      const res = await fn();
      if (res?.timeline) updateProject({ timeline: res.timeline });
      return res;
    } catch (err: any) {
      setError(err.message || 'Could not change the effect');
    } finally {
      setBusy(false);
    }
  }, [updateProject, setError]);

  // Sliders fire on every pixel of travel; the timeline round-trip is what makes
  // that expensive, so only the settled value is sent.
  const schedule = useCallback((fn: () => Promise<any>) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => run(fn), 220);
  }, [run]);

  const add = useCallback((updates: Record<string, any>, preset?: string) => {
    if (projectId) run(() => addEffect(projectId, preset, updates, owner));
  }, [projectId, owner, run]);

  const addLayer = useCallback(() => {
    if (!projectId) return;
    run(() => addAdjustment(projectId, {
      timelineStartFrame: Math.max(0, Math.round(currentTime * fps)),
      durationSeconds: ADJUSTMENT_SECONDS,
    }));
  }, [projectId, currentTime, fps, run]);

  const catalogue = presets?.effect_catalogue ?? [];

  // The metadata pack's chapters, as the presentation pass placed them on its own
  // progress bar (programme seconds, already shifted by the cold open). The pack
  // itself (metadata.json) has no route, so this is where the UI can read them.
  const packChapters: ProgressBarChapter[] | null = useMemo(() => {
    const own = ((timeline?.effects || []) as AtmosphereEffect[]).find(
      (e) => e.type === 'progress_bar' && e.origin === 'presentation:progress'
        && Array.isArray(e.extra?.chapters) && e.extra!.chapters.length > 0);
    return own ? (own.extra!.chapters as ProgressBarChapter[]) : null;
  }, [timeline]);

  if (!projectId) {
    return <div className="fx-empty">Open a project to add effects.</div>;
  }

  return (
    <div className="fx-panel">
      <div className="fx-status">
        <span>
          {isAdjustment
            ? <>Adding to the <strong>adjustment layer</strong> on {item.track} — {effects.length} effect{effects.length === 1 ? '' : 's'}.</>
            : <>Adding to the <strong>whole programme</strong> — {effects.length} effect{effects.length === 1 ? '' : 's'}.</>}
        </span>
        {!isAdjustment && (
          <button className="btn btn-xs" disabled={busy} onClick={addLayer}
            title="Drop a short adjustment layer at the playhead, so the next effect only covers those seconds">
            ＋ Layer
          </button>
        )}
      </div>

      <Section title="Presets">
        {presets
          ? <PresetChips items={presets.effect} onPick={(id) => add({}, id)} />
          : <div className="text-xs text-muted">Loading…</div>}
      </Section>

      <Section title={`All effects${catalogue.length ? ` (${catalogue.length})` : ''}`}>
        <div className="fx-cards">
          {catalogue.map((fx) => (
            <button key={fx.id} className="fx-card" disabled={busy}
              onClick={() => add({ type: fx.id, intensity: 0.5, speed: 1.0,
                ...(fx.extra ? { extra: fx.extra } : {}) })}>
              <span className="fx-card-name">{fx.label}</span>
              <span className="fx-card-desc">{fx.description}</span>
              {MOMENTARY.has(fx.id) && (
                <span className="fx-card-tag" title="Reads best on a short adjustment layer">
                  hit
                </span>
              )}
            </button>
          ))}
        </div>
      </Section>

      <Section title={`Applied${effects.length ? ` (${effects.length})` : ''}`}>
        {effects.length === 0 ? (
          <div className="text-xs text-muted">
            Nothing applied here yet — pick one above. Effects stack in order, so a
            grain over a light leak looks different from a light leak over grain.
          </div>
        ) : effects.map((effect, index) => (
          <div key={index} className="insp-effect">
            <div className="insp-effect-head">
              <span className="insp-effect-name">{effect.type.replace(/_/g, ' ')}</span>
              <label className="text-xs text-muted" style={{ display: 'flex', gap: 4 }}>
                <input type="checkbox" aria-label={`${effect.type.replace(/_/g, ' ')} on`} checked={effect.enabled}
                  onChange={(e) => run(() =>
                    updateEffect(projectId, index, { enabled: e.target.checked }, owner))} />
                on
              </label>
              <button className="btn btn-xs btn-danger" title="Remove" aria-label="Remove" disabled={busy}
                onClick={() => run(() => removeEffect(projectId, index, owner))}>✕</button>
            </div>
            <Slider label="Amount" value={effect.intensity} min={0} max={1}
              onChange={(v) => schedule(() => updateEffect(projectId, index, { intensity: v }, owner))}
              format={(v) => `${Math.round(v * 100)}%`} />
            <Slider label="Speed" value={effect.speed} min={0.2} max={3}
              onChange={(v) => schedule(() => updateEffect(projectId, index, { speed: v }, owner))}
              format={(v) => `${v.toFixed(1)}×`} />
            {TINTABLE.has(effect.type) && (
              <ColorField label="Tint" value={toHex(effect.color || '#ffcc88')} allowAlpha={false}
                onChange={(v) => schedule(() => updateEffect(
                  projectId, index, { color: `0x${v.replace('#', '').toUpperCase()}FF` }, owner))} />
            )}
            {effect.type === 'progress_bar' && (
              <ProgressBarEditor effect={effect} busy={busy} packChapters={packChapters}
                onSet={(extra) => run(() => updateEffect(projectId, index, { extra }, owner))}
                onSchedule={(extra) => schedule(() => updateEffect(projectId, index, { extra }, owner))} />
            )}
          </div>
        ))}
      </Section>

      {/* An effect that has been applied and an effect that has been applied and
          is visible are different things here: the preview plays the source file,
          not a compiled programme, so rain lands in the timeline and nowhere the
          eye can find it until the render runs. Saying so is the whole difference
          between "this is broken" and "this is deferred". */}
      <div className="fx-footnote">
        Effects are built by the renderer, so the preview above will not show them.
        Applied ones are listed here and marked <span className="fx-legend-glyph">✦</span> on
        the timeline; <strong>Render</strong> is where you see them.
      </div>
    </div>
  );
}
