/**
 * Regions (VOICE_CHANGER_PLAN.md §2.7): SkinWalker's Editor tab, native. The project's
 * `voice_regions` -- programme seconds, what the timeline shows -- each with its own voice,
 * preset, params and FX and a status. A region comes from the timeline's audio range ("Add from
 * selection" here, or the range band's "Convert voice…"); its voice, preset and params are edited
 * in the table and saved with `PUT /voice/regions` (the backend clamps them to the programme and
 * trims overlaps). "Apply pending" converts every pending region into ONE new version (a region
 * across a cut included), seams crossfaded 15 ms, and puts each region's FX on the A1 rack
 * windowed to it; "Undo last apply" pops that version and sets its regions pending again. An
 * applied region is locked until it is undone: re-converting it would convert the converted voice.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useProjectStore } from '../../hooks/store';
import { EffectDef, routeMissing } from '../../hooks/audioApi';
import {
  PITCH_RANGE, PresetsResponse, RegionsApplyResult, STEPS_RANGE, VoiceEntry, VoiceRegion,
  addRegionFromSelection, applyRegions, choiceLabel, defaultChoice, fileName, formatSeconds,
  presetChoice, putRegions, refreshTimeline, regionsError, reloadRegions, startError,
  syncRegionsToStore, undoRegions, voiceInLibrary,
} from '../../hooks/voiceApi';
import JobProgress, { JobWatch, jobRunning, useJob, watchJob } from './JobProgress';
import { fxSummaryText } from './FxSummary';

const SAVE_DEBOUNCE_MS = 500;

const range = (lo: number, hi: number) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
const PITCHES = range(PITCH_RANGE[0], PITCH_RANGE[1]);
const STEPS = range(STEPS_RANGE[0], STEPS_RANGE[1]);
const signed = (n: number) => (n > 0 ? `+${n}` : String(n));

export default function RegionsTab({ projectId, voices, presets, defs, reloadEpoch, incoming }: {
  projectId: string;
  voices: VoiceEntry[];
  presets: PresetsResponse | null;
  defs: Record<string, EffectDef>;
  /** bumped when the regions changed outside the tab (the timeline's "Convert voice…") */
  reloadEpoch: number;
  /** a message handed to the panel with the request */
  incoming: { note: string; seq: number } | null;
}) {
  const audioSelection = useProjectStore((s) => s.audioSelection);
  const [regions, setRegions] = useState<VoiceRegion[]>([]);
  const regionsRef = useRef<VoiceRegion[]>([]);
  regionsRef.current = regions;
  const [loaded, setLoaded] = useState(false);
  const [missing, setMissing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [adding, setAdding] = useState(false);
  const [undoing, setUndoing] = useState(false);
  const [watch, setWatch] = useState<JobWatch | null>(null);
  const job = useJob(watch);
  const [summary, setSummary] = useState<RegionsApplyResult | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const alive = useRef(true);
  const saveTimer = useRef<number | null>(null);
  const pendingSave = useRef<VoiceRegion[] | null>(null);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  // -- load ---------------------------------------------------------------------------------------
  const load = useCallback(async () => {
    try {
      const list = await reloadRegions(projectId);
      if (!alive.current) return;
      if (!pendingSave.current) setRegions(list);
      setMissing(false);
    } catch (err) {
      if (!alive.current) return;
      if (routeMissing(err)) setMissing(true);
      else setNote(`Could not load the regions: ${regionsError(err)}`);
    } finally {
      if (alive.current) setLoaded(true);
    }
  }, [projectId]);

  useEffect(() => { void load(); }, [load, reloadEpoch]);

  useEffect(() => {
    if (incoming) setNote(incoming.note);
  }, [incoming]);

  // -- save (debounced; flushed before add, apply, undo and when the tab goes) ------------------
  const flush = useCallback(async (): Promise<boolean> => {
    if (saveTimer.current != null) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    const next = pendingSave.current;
    if (!next) return true;
    pendingSave.current = null;
    if (alive.current) setSaving(true);
    try {
      const res = await putRegions(projectId, next);
      const stored = res.regions || [];
      syncRegionsToStore(projectId, stored);
      if (alive.current && !pendingSave.current) setRegions(stored);
      return true;
    } catch (err) {
      if (alive.current) setNote(`Could not save the regions: ${regionsError(err)}`);
      void load();
      return false;
    } finally {
      if (alive.current) setSaving(false);
    }
  }, [projectId, load]);

  const flushRef = useRef(flush);
  flushRef.current = flush;
  useEffect(() => () => { void flushRef.current(); }, []);

  const commit = (next: VoiceRegion[], delay = SAVE_DEBOUNCE_MS) => {
    regionsRef.current = next;
    setRegions(next);
    pendingSave.current = next;
    if (saveTimer.current != null) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => {
      saveTimer.current = null;
      void flush();
    }, delay);
  };

  /** Edit one region; a failed one becomes pending again (its error no longer applies). */
  const edit = (id: string, change: (r: VoiceRegion) => VoiceRegion, delay = 0) => {
    const next = regionsRef.current.map((r) => {
      if (r.id !== id || r.status === 'applied') return r;
      const changed = change(r);
      if (changed.status !== 'failed') return changed;
      const { error: _error, ...rest } = changed;
      return { ...rest, status: 'pending' as const };
    });
    commit(next, delay);
  };

  const withPreset = (r: VoiceRegion, name: string): VoiceRegion => {
    if (!name) return { ...r, preset: null, fx: [] };
    const p = presets?.presets?.[name];
    if (!p) return { ...r, preset: name };
    const c = presetChoice(p);
    return { ...r, preset: name, voice: c.voice || r.voice, params: { ...r.params, ...c.params }, fx: c.fx };
  };

  const remove = (r: VoiceRegion) => {
    if (r.status === 'applied' && !window.confirm(
      'This region is applied: removing it from the list keeps its converted audio and its FX until the apply is undone. Remove it anyway?',
    )) return;
    commit(regionsRef.current.filter((x) => x.id !== r.id), 0);
  };

  // -- add, apply, undo --------------------------------------------------------------------------
  const running = jobRunning(job);
  const pending = regions.filter((r) => r.status === 'pending').length;
  const applied = regions.filter((r) => r.status === 'applied').length;
  const nextChoice = defaultChoice(presets, voices);
  const sel = audioSelection && audioSelection.end_s > audioSelection.start_s ? audioSelection : null;

  const add = async () => {
    const current = useProjectStore.getState().audioSelection;
    if (!current || !(current.end_s > current.start_s)) return;
    setAdding(true);
    setNote(null);
    try {
      if (!(await flush())) return;
      const res = await addRegionFromSelection(projectId, current, defaultChoice(presets, voices));
      if (alive.current) {
        setRegions(res.regions);
        setNote(res.note);
      }
    } catch (err) {
      if (alive.current) setNote(`Could not add the region: ${regionsError(err)}`);
    } finally {
      if (alive.current) setAdding(false);
    }
  };

  const apply = async () => {
    setNote(null);
    setSummary(null);
    if (!(await flush())) return;
    try {
      const { job_id: jobId } = await applyRegions(projectId);
      const w = watchJob(projectId, jobId);
      setWatch(w);
      const pid = projectId;
      void w.done.then(async (final) => {
        // Statuses, the A1 rack and the source's version changed (some may have, even on a failure).
        await refreshTimeline(pid);
        let list: VoiceRegion[] | null = null;
        try {
          list = await reloadRegions(pid);
        } catch {
          /* shown on the next load */
        }
        if (!alive.current) return;
        if (list && !pendingSave.current) setRegions(list);
        if (final.status === 'completed') setSummary((final.result as unknown as RegionsApplyResult) || null);
      });
    } catch (err) {
      setNote(startError(err, 'Apply'));
    }
  };

  const undo = async () => {
    setNote(null);
    setSummary(null);
    if (!(await flush())) return;
    setUndoing(true);
    try {
      const res = await undoRegions(projectId);
      const list = res.regions || [];
      syncRegionsToStore(projectId, list);
      await refreshTimeline(projectId);
      if (alive.current) {
        setRegions(list);
        setNote('Undid the last apply: its regions are pending again; their converted audio and their FX are gone.');
      }
    } catch (err) {
      if (alive.current) setNote(`Undo: ${regionsError(err)}`);
    } finally {
      if (alive.current) setUndoing(false);
    }
  };

  const regionLabel = (id: string) => {
    const k = regions.findIndex((r) => r.id === id);
    const r = regions[k];
    return r ? `#${k + 1} ${formatSeconds(r.start_s)}–${formatSeconds(r.end_s)}` : id;
  };

  if (missing) {
    return (
      <div className="voice-tab">
        <div className="audio-note">
          Voice regions are not available on this backend yet (it predates the voice routes; restart BuzzEdit to load them).
        </div>
      </div>
    );
  }

  return (
    <div className="voice-tab">
      <div className="audio-card">
        <div className="voice-row">
          {sel ? (
            <span className="audio-inline">
              Selection {formatSeconds(sel.start_s)}–{formatSeconds(sel.end_s)} · {(sel.end_s - sel.start_s).toFixed(2)} s
            </span>
          ) : (
            <span className="audio-hint">
              No selection: Shift+drag on an audio lane or the ruler, or I / O at the playhead.
            </span>
          )}
          <span className="voice-spacer" />
          <button type="button" className="audio-btn primary" disabled={!sel || adding || running}
            title="A pending region over the timeline's audio range" onClick={() => void add()}>
            {adding ? 'Adding…' : 'Add from selection'}
          </button>
        </div>
        <div className="audio-hint">
          New regions take {nextChoice ? choiceLabel(nextChoice) : 'the first voice or preset'} (the voice or preset picked
          last); the timeline range's right-click "Convert voice…" adds one too.
        </div>
      </div>

      {!loaded ? (
        <div className="audio-hint">Loading the regions…</div>
      ) : regions.length === 0 ? (
        <div className="audio-hint">No voice regions yet.</div>
      ) : (
        <div className="voice-table-wrap">
          <table className="voice-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Start</th>
                <th>End</th>
                <th>Voice</th>
                <th>Preset</th>
                <th title="Match my pitch contour">Match</th>
                <th title="Pitch shift, semitones">Pitch</th>
                <th title="Quality steps">Steps</th>
                <th title="Rack effects after the conversion">FX</th>
                <th>Status</th>
                <th aria-label="Remove" />
              </tr>
            </thead>
            <tbody>
              {regions.map((r, i) => {
                const locked = r.status === 'applied' || running;
                const pitch = Number(r.params?.pitch_shift) || 0;
                const steps = Number(r.params?.steps) || 25;
                const lockHint = r.status === 'applied' ? 'Applied: undo the last apply to edit it' : undefined;
                return (
                  <tr key={r.id} className={`voice-region ${r.status}`} title={lockHint}>
                    <td className="voice-num">{i + 1}</td>
                    <td className="voice-time">{formatSeconds(r.start_s)}</td>
                    <td className="voice-time">{formatSeconds(r.end_s)}</td>
                    <td>
                      <select className="audio-select" disabled={locked} value={r.voice || ''}
                        onChange={(e) => {
                          const voice = e.target.value;
                          edit(r.id, (x) => ({ ...x, voice }));
                        }}>
                        <option value="">{r.preset ? "(the preset's)" : '(none)'}</option>
                        {voices.map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
                        {r.voice && !voiceInLibrary(voices, r.voice) && (
                          <option value={r.voice}>{r.voice} (not in the library)</option>
                        )}
                      </select>
                    </td>
                    <td>
                      <select className="audio-select" disabled={locked} value={r.preset || ''}
                        onChange={(e) => {
                          const name = e.target.value;
                          edit(r.id, (x) => withPreset(x, name));
                        }}>
                        <option value="">(none)</option>
                        {Object.keys(presets?.presets || {}).map((name) => <option key={name} value={name}>{name}</option>)}
                        {r.preset && !presets?.presets?.[r.preset] && <option value={r.preset}>{r.preset}</option>}
                      </select>
                    </td>
                    <td className="voice-center">
                      <input type="checkbox" disabled={locked} checked={!!r.params?.match_pitch}
                        onChange={(e) => {
                          const on = e.target.checked;
                          edit(r.id, (x) => ({ ...x, params: { ...x.params, match_pitch: on } }));
                        }} />
                    </td>
                    <td>
                      <select className="audio-select" disabled={locked} value={pitch}
                        onChange={(e) => {
                          const value = Number(e.target.value);
                          edit(r.id, (x) => ({ ...x, params: { ...x.params, pitch_shift: value } }));
                        }}>
                        {!PITCHES.includes(pitch) && <option value={pitch}>{signed(pitch)}</option>}
                        {PITCHES.map((n) => <option key={n} value={n}>{signed(n)}</option>)}
                      </select>
                    </td>
                    <td>
                      <select className="audio-select" disabled={locked} value={steps}
                        onChange={(e) => {
                          const value = Number(e.target.value);
                          edit(r.id, (x) => ({ ...x, params: { ...x.params, steps: value } }));
                        }}>
                        {!STEPS.includes(steps) && <option value={steps}>{steps}</option>}
                        {STEPS.map((n) => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </td>
                    <td className="voice-center" title={fxSummaryText(r.fx, defs)}>{r.fx?.length || 0}</td>
                    <td>
                      <span className={`voice-badge ${r.status}`}
                        title={r.status === 'failed' ? r.error || 'failed' : r.status === 'applied' ? `in ${fileName(r.applied_version || '')}` : 'waits for Apply pending'}>
                        {r.status}
                      </span>
                    </td>
                    <td>
                      <button type="button" className="audio-icon danger" disabled={running}
                        title="Remove the region" onClick={() => remove(r)}>
                        ✕
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {regions.filter((r) => r.status === 'failed' && r.error).map((r) => (
            <div key={r.id} className="text-xs voice-failed">{regionLabel(r.id)} failed: {r.error}</div>
          ))}
        </div>
      )}

      <div className="voice-row">
        <button type="button" className="audio-btn primary" disabled={pending === 0 || running || undoing}
          title="Convert every pending region into one new version" onClick={() => void apply()}>
          {running ? 'Applying…' : `Apply pending${pending ? ` (${pending})` : ''}`}
        </button>
        <button type="button" className="audio-btn" disabled={applied === 0 || running || undoing}
          title="Pop the newest regions version: back to the audio before it" onClick={() => void undo()}>
          {undoing ? 'Undoing…' : 'Undo last apply'}
        </button>
        {saving && <span className="audio-hint">saving…</span>}
      </div>
      <JobProgress watch={watch} label="Apply" />
      {summary && (
        <div className="audio-card">
          <div className="text-xs">
            {summary.applied.length} region{summary.applied.length === 1 ? '' : 's'} converted
            {summary.version ? ` into one new version (${fileName(summary.version)})` : ''}
            {summary.failed.length ? `; ${summary.failed.length} failed and kept the original` : ''}.
          </div>
          {summary.failed.map((f) => (
            <div key={f.id} className="text-xs voice-failed">{regionLabel(f.id)}: {f.error}</div>
          ))}
        </div>
      )}
      <div className="audio-hint">
        Conversions keep the length (length_adjust 1.0), so the cut, the B-roll and every edit stay valid.
        Unmarked audio is never touched, each seam is crossfaded 15 ms, a region's FX go on the A1 rack
        windowed to it, and a region that fails keeps the original.
      </div>
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}
