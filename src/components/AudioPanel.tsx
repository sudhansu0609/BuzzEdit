/**
 * Audio panel -- Audition's effects workflow inside BuzzEdit (AUDIO_SUITE_PLAN.md §2.4, brief A4
 * part a). Left, the Effects menu tree; centre, the Effects Rack of the scope (Programme = the
 * A1 rack, Clip = the selected A2+ item, Master = after the mix); right, the parameters
 * generated from the catalogue, or a process effect's card. Above them the transport and meters
 * of the live engine (brief L, `src/audio/LiveEngine.ts`), so every knob is heard as it moves:
 *
 *   - a parameter whose ffmpeg options take runtime commands goes out as `set`;
 *   - a power button goes out as `bypass` (A/B bypasses the whole rack);
 *   - add, remove, reorder, a window and every other parameter send the whole rack
 *     (`replaceRack`, at most one per 100 ms), which restarts the stream at the playhead.
 *
 * Each edit is also saved with `PUT /api/audio/{project}/rack` 500 ms after the last one, and
 * the store's timeline is patched to match, so the editor's autosave (which writes the whole
 * timeline back) never restores an older rack. Every route may be missing on an older backend:
 * the panel then says so and keeps working with what it has.
 *
 * Part b: the range is the store's `audioSelection`, in programme seconds -- what the timeline
 * shows; the timeline's Shift+drag, I / O and Range toggle set it -- shown under the transport
 * with a Clear. The editor column's "Essential Sound" tab is Audition's Dialogue card: each of
 * its controls owns one effect of the scope's rack, tagged `preset: "essential:<control>"`, and
 * edits it through the same live and save path as the rack.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { getTimeline } from '../hooks/api';
import { LiveEngine, type MeterReading } from '../audio/LiveEngine';
import {
  AnalyzeResult, AudioEffect, Catalogue, EffectDef, RackPresets, Signal, UserPresets,
  analyzeRange, applyRackPreset, audioFileUrl, cloneValue, defaultParams, errorText, getEffects,
  getPresets, getProcesses, getRackPresets, getRacks, newEffectId, putRack, routeMissing,
  saveRackPreset,
} from '../hooks/audioApi';
import { Section } from './controls';
import EffectTree from './audio/EffectTree';
import RackList from './audio/RackList';
import ParamEditor from './audio/ParamEditor';
import Transport, { LiveStatus, PlayState, TransportRange, formatTime } from './audio/Transport';
import Meters from './audio/Meters';
import ProcessCard from './audio/ProcessCard';
import EssentialSound, { ESSENTIAL_TAG } from './audio/EssentialSound';
import '../styles/audio.css';

type ScopeTab = 'A1' | 'clip' | 'MASTER';

const PUT_DEBOUNCE_MS = 500;
const RACK_THROTTLE_MS = 100;
const CONNECTION_LOST = new Set(['live audio connection closed', 'live audio connection failed']);
const MISSING = 'not available on this backend yet (it predates the audio routes; restart BuzzEdit to load them)';

const SCOPE_HINT: Record<ScopeTab, string> = {
  A1: 'The A1 voice: after the voice chain, before the mix',
  clip: "This clip's own audio, before its volume and fades",
  MASTER: 'The whole mix, before the loudness stage',
};

function normalizeRack(effects: unknown): AudioEffect[] {
  if (!Array.isArray(effects)) return [];
  return effects
    .filter((e) => e && typeof e === 'object' && typeof e.id === 'string' && typeof e.effect === 'string')
    .map((e) => ({
      id: e.id,
      effect: e.effect,
      params: e.params && typeof e.params === 'object' ? e.params : {},
      enabled: e.enabled !== false,
      start_s: typeof e.start_s === 'number' ? e.start_s : null,
      end_s: typeof e.end_s === 'number' ? e.end_s : null,
      preset: typeof e.preset === 'string' ? e.preset : null,
    }));
}

const allIds = (racks: Record<string, AudioEffect[]>) => Object.values(racks).flat().map((e) => e.id);

/** An Essential Sound effect keeps its tag through edits in the rack, so its control goes on
 *  updating that one effect instead of adding another. */
const keepTag = (fx: AudioEffect, next: string | null) =>
  (fx.preset && fx.preset.startsWith(ESSENTIAL_TAG) ? fx.preset : next);

/** Mirror a saved rack into the store's timeline (autosave writes the whole timeline back). */
function patchTimeline(projectId: string, key: string, effects: AudioEffect[], revision?: number) {
  const state = useProjectStore.getState();
  const tl = state.project?.id === projectId ? (state.project.timeline as any) : null;
  if (!tl) return;
  let next: any;
  if (key.startsWith('item:')) {
    const id = key.slice(5);
    const walk = (items: any[]): any[] => items.map((i) => (i?.id === id
      ? { ...i, audio_effects: effects }
      : Array.isArray(i?.children) ? { ...i, children: walk(i.children) } : i));
    next = { ...tl, items: walk(tl.items || []) };
  } else {
    next = { ...tl, audio_racks: { ...(tl.audio_racks || {}), [key]: effects } };
  }
  if (typeof revision === 'number') next.revision = revision;
  state.updateProject({ timeline: next });
}

function Stat({ label, value, unit }: { label: string; value: number | null | undefined; unit: string }) {
  return (
    <div className="audio-stat">
      <span className="audio-stat-label">{label}</span>
      <span className="audio-stat-value">
        {typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '–'}
      </span>
      <span className="audio-unit">{unit}</span>
    </div>
  );
}

/** Whole-rack presets: "Load rack preset" replaces the scope's rack with a saved one (the
 *  backend gives every effect a fresh id); "Save rack as preset…" stores this rack in order. */
function RackPresetBar({ names, canSave, canLoad, onSave, onLoad }: {
  names: string[];
  canSave: boolean;
  canLoad: boolean;
  onSave: (name: string) => Promise<void>;
  onLoad: (name: string) => Promise<void>;
}) {
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const run = async (task: () => Promise<void>, what: string): Promise<boolean> => {
    setBusy(true);
    setNote(null);
    try {
      await task();
      return true;
    } catch (err) {
      setNote(routeMissing(err) ? `Rack presets are ${MISSING}.` : `${what}: ${errorText(err)}`);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    const preset = name.trim();
    if (!preset || preset.includes('/')) {
      setNote('Give the rack preset a name (no "/").');
      return;
    }
    if (await run(() => onSave(preset), 'Not saved')) {
      setSaving(false);
      setName('');
    }
  };

  return (
    <div className="audio-presets">
      <div className="audio-presets-row">
        <select className="audio-select" value="" disabled={busy || !canLoad || names.length === 0}
          title="Replace this rack with a saved rack preset"
          onChange={(e) => {
            const picked = e.target.value;
            if (picked) void run(() => onLoad(picked), 'Not loaded');
          }}>
          <option value="">{names.length ? 'Load rack preset…' : 'No rack presets yet'}</option>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <button className="audio-btn" disabled={busy || !canSave}
          title="Save this whole rack, in order, as a preset any scope or project can load"
          onClick={() => { setSaving((s) => !s); setNote(null); }}>
          Save rack as preset…
        </button>
      </div>
      {saving && (
        <div className="audio-presets-row">
          <input className="audio-text" placeholder="Rack preset name" value={name} autoFocus
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void save();
              if (e.key === 'Escape') setSaving(false);
            }} />
          <button className="audio-btn primary" disabled={busy || !name.trim()} onClick={() => void save()}>
            Save
          </button>
        </div>
      )}
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}

export default function AudioPanel() {
  const project = useProjectStore((s) => s.project);
  const selectedClipId = useProjectStore((s) => s.selectedClipId);
  const audioSelection = useProjectStore((s) => s.audioSelection);
  const setAudioSelection = useProjectStore((s) => s.setAudioSelection);

  const projectId = project?.id ?? null;
  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;
  const timeline = project?.timeline as any;
  const fps = timeline?.fps_num ? timeline.fps_num / (timeline.fps_den || 1) : 30;

  // -- catalogue ------------------------------------------------------------------------------
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [catalogueNote, setCatalogueNote] = useState<string | null>(null);
  const [processes, setProcesses] = useState<EffectDef[]>([]);
  const [processNote, setProcessNote] = useState<string | null>(null);
  const [userPresets, setUserPresets] = useState<UserPresets>({});
  const [rackPresets, setRackPresets] = useState<RackPresets>({});

  useEffect(() => {
    let alive = true;
    getEffects()
      .then((c) => {
        if (!alive) return;
        setCatalogue({ effects: c.effects || [], categories: c.categories || [] });
        setCatalogueNote(null);
      })
      .catch((err) => {
        if (alive) setCatalogueNote(routeMissing(err) ? `The effect catalogue is ${MISSING}.` : `Could not load the effects: ${errorText(err)}`);
      });
    getProcesses()
      .then((list) => { if (alive) { setProcesses(list); setProcessNote(null); } })
      .catch((err) => {
        if (alive) setProcessNote(routeMissing(err) ? `Process effects are ${MISSING}.` : `Could not load the process effects: ${errorText(err)}`);
      });
    getPresets()
      .then((r) => { if (alive) setUserPresets(r.presets || {}); })
      .catch(() => { /* the preset menu reports it when it is used */ });
    getRackPresets()
      .then((r) => { if (alive) setRackPresets(r.presets || {}); })
      .catch(() => { /* the rack preset bar reports it when it is used */ });
    return () => { alive = false; };
  }, []);

  const defs = useMemo(() => {
    const out: Record<string, EffectDef> = {};
    for (const def of catalogue?.effects || []) out[def.id] = def;
    return out;
  }, [catalogue]);
  const processDefs = useMemo(() => {
    const out: Record<string, EffectDef> = {};
    for (const def of processes) out[def.id] = def;
    return out;
  }, [processes]);

  // -- scope and range ------------------------------------------------------------------------
  const [tab, setTab] = useState<ScopeTab>('A1');
  const item = useMemo(
    () => (timeline?.items || []).find((i: any) => i.id === selectedClipId) || null,
    [timeline, selectedClipId],
  );
  const itemTrack = String(item?.track || '').toUpperCase();
  // A1 items are rebuilt from the transcript, so only A2+ clips keep a rack of their own.
  const clipEligible = !!item && (!item.kind || item.kind === 'media') && /^A\d+$/.test(itemTrack) && itemTrack !== 'A1';
  const scopeKey = tab === 'clip' ? (clipEligible ? `item:${item.id}` : 'A1') : tab;
  const scopeKeyRef = useRef(scopeKey);
  scopeKeyRef.current = scopeKey;
  const isClipScope = scopeKey.startsWith('item:');

  useEffect(() => {
    if (tab === 'clip' && !clipEligible) setTab('A1');
  }, [tab, clipEligible]);

  const clip = useMemo(() => {
    if (!clipEligible || !item) return null;
    const a = Number(item.source_start_frame) || 0;
    const b = Number(item.source_end_frame) || 0;
    return {
      sourceId: item.source_id ? String(item.source_id) : undefined,
      offset: a / fps,
      length: b > a ? (b - a) / fps : null,
    };
  }, [clipEligible, item, fps]);

  // The programme and master scopes run in programme seconds: what the timeline shows (the
  // backend maps them onto the recording across cuts). Their full length is the programme's,
  // the cut: the timeline's last end frame (as the timeline draws it), else its
  // duration_frames -- never the recording's duration.
  const programmeDuration = useMemo(() => {
    const items: any[] = Array.isArray(timeline?.items) ? timeline.items : [];
    const end = items.reduce((m: number, i: any) => Math.max(m, Number(i?.timeline_end_frame) || 0), 0);
    const frames = end > 0 ? end : Number(timeline?.duration_frames);
    return Number.isFinite(frames) && frames > 0 ? frames / fps : null;
  }, [timeline, fps]);

  // What the transport loops and what analysis and the processes use: the selection (programme
  // seconds, set on the timeline; a clip scope reads it clip-relative), else the selected clip
  // (an A1/V1 clip's span is its place on the timeline, in programme seconds), else the whole
  // programme.
  const range: TransportRange | null = useMemo(() => {
    if (audioSelection && audioSelection.end_s > audioSelection.start_s) {
      return { start_s: audioSelection.start_s, end_s: audioSelection.end_s, origin: 'selection' };
    }
    if (isClipScope && clip?.length) return { start_s: 0, end_s: clip.length, origin: 'clip' };
    if (!isClipScope && item && (itemTrack === 'A1' || itemTrack === 'V1')) {
      const a = Number(item.timeline_start_frame) / fps;
      const b = Number(item.timeline_end_frame) / fps;
      if (Number.isFinite(a) && Number.isFinite(b) && b > a) return { start_s: a, end_s: b, origin: 'clip' };
    }
    if (programmeDuration) return { start_s: 0, end_s: programmeDuration, origin: 'programme' };
    return null;
  }, [audioSelection, isClipScope, clip, item, itemTrack, fps, programmeDuration]);
  const rangeRef = useRef(range);
  rangeRef.current = range;

  // Processes and analysis take programme seconds for the programme and master scopes (the
  // backend maps them onto the recording); a clip scope's seconds are offset by its in-point.
  const offset = isClipScope && clip ? clip.offset : 0;
  const sourceId = isClipScope ? clip?.sourceId : undefined;
  const sourceRange = range && range.origin !== 'programme'
    ? { start_s: offset + range.start_s, end_s: offset + range.end_s } : null;
  const analysisRange = sourceRange ?? (programmeDuration ? { start_s: 0, end_s: programmeDuration } : null);
  const sourceLabel = isClipScope ? `Source of the ${itemTrack} clip` : 'Programme recording';
  const scopeLabel = isClipScope ? `Clip on ${itemTrack}` : scopeKey === 'MASTER' ? 'Master' : 'Programme (A1)';

  // -- racks ----------------------------------------------------------------------------------
  const [racks, setRacks] = useState<Record<string, AudioEffect[]>>({});
  const racksRef = useRef(racks);
  const [racksReady, setRacksReady] = useState(false);
  const [rackNote, setRackNote] = useState<string | null>(null);
  const [saveNote, setSaveNote] = useState<string | null>(null);
  const putTimers = useRef(new Map<string, { timer: ReturnType<typeof setTimeout>; projectId: string }>());

  const savePut = useCallback((pid: string, key: string) => {
    const pending = putTimers.current.get(key);
    if (pending) {
      clearTimeout(pending.timer);
      putTimers.current.delete(key);
    }
    const effects = cloneValue(racksRef.current[key] || []);
    putRack(pid, key, effects)
      .then((res) => {
        setSaveNote(null);
        patchTimeline(pid, key, normalizeRack(res.effects ?? effects), res.revision);
      })
      .catch((err) => setSaveNote(routeMissing(err)
        ? `Saving racks is ${MISSING}: edits are heard but not kept.`
        : `Rack not saved: ${errorText(err)}`));
  }, []);

  const schedulePut = useCallback((key: string) => {
    const pid = projectIdRef.current;
    if (!pid) return;
    const pending = putTimers.current.get(key);
    if (pending) clearTimeout(pending.timer);
    putTimers.current.set(key, { projectId: pid, timer: setTimeout(() => savePut(pid, key), PUT_DEBOUNCE_MS) });
  }, [savePut]);

  const flushPuts = useCallback(() => {
    for (const [key, pending] of Array.from(putTimers.current.entries())) savePut(pending.projectId, key);
  }, [savePut]);

  useEffect(() => {
    if (!projectId) return;
    let alive = true;
    setRacksReady(false);
    getRacks(projectId)
      .then((res) => {
        if (!alive) return;
        const next: Record<string, AudioEffect[]> = {};
        for (const [key, effects] of Object.entries(res.racks || {})) next[key] = normalizeRack(effects);
        for (const [id, effects] of Object.entries(res.items || {})) next[`item:${id}`] = normalizeRack(effects);
        racksRef.current = next;
        setRacks(next);
        setRackNote(null);
      })
      .catch((err) => {
        if (!alive) return;
        racksRef.current = {};
        setRacks({});
        setRackNote(routeMissing(err) ? `Effect racks are ${MISSING}.` : `Could not load the racks: ${errorText(err)}`);
      })
      .finally(() => { if (alive) setRacksReady(true); });
    return () => {
      alive = false;
      flushPuts(); // pending edits go to the project they were made in
    };
  }, [projectId, flushPuts]);

  // -- live engine ----------------------------------------------------------------------------
  const engineRef = useRef<LiveEngine | null>(null);
  const chain = useRef<Promise<unknown>>(Promise.resolve());
  const serverEnabled = useRef(new Map<string, boolean>()); // `enabled` as the server last got it
  const rackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rackLast = useRef(0);
  const posRef = useRef(0);
  const abRef = useRef(false);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>('idle');
  const [liveNote, setLiveNote] = useState<string | null>(null);
  const [liveEpoch, setLiveEpoch] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loopOn, setLoopOn] = useState(false);
  const [ab, setAbState] = useState(false);
  const position = useMemo(() => new Signal<PlayState>(), []);
  const meters = useMemo(() => new Signal<MeterReading>(), []);

  const setAb = (value: boolean) => {
    abRef.current = value;
    setAbState(value);
  };

  /** Connects and disconnects run one after another, never interleaved. */
  const enqueue = (task: () => Promise<void> | void) => {
    const next = chain.current.then(task).catch(() => undefined);
    chain.current = next;
    return next;
  };

  const engine = (): LiveEngine => {
    if (!engineRef.current) {
      const live = new LiveEngine();
      live.onState = (t, isPlaying) => {
        posRef.current = t;
        position.emit({ t, playing: isPlaying });
        setPlaying(isPlaying);
      };
      live.onMeters = (m) => meters.emit(m);
      live.onError = (message) => {
        setLiveNote(message);
        if (CONNECTION_LOST.has(message)) {
          setLiveStatus('off');
          setPlaying(false);
        }
      };
      engineRef.current = live;
    }
    return engineRef.current;
  };

  const remember = (rack: AudioEffect[]) => {
    serverEnabled.current = new Map(rack.map((fx) => [fx.id, fx.enabled]));
  };

  useEffect(() => {
    if (!projectId || !racksReady) return;
    let cancelled = false;
    const key = scopeKey;
    setLiveStatus('connecting');
    setLiveNote(null);
    setPlaying(false);
    setLoopOn(false);
    setAb(false);
    posRef.current = 0;
    position.emit({ t: 0, playing: false });
    void enqueue(async () => {
      if (cancelled) return;
      const rack = cloneValue(racksRef.current[key] || []);
      try {
        await engine().connect(projectId, key, rack);
        if (cancelled) return;
        remember(rack);
        setLiveStatus('live');
        // edited while the socket was opening: the engine still has the older rack
        if (JSON.stringify(racksRef.current[key] || []) !== JSON.stringify(rack)) flushRack();
      } catch (err) {
        if (cancelled) return;
        setLiveStatus('off');
        setLiveNote(`Live preview unavailable: ${errorText(err)}`);
      }
    });
    return () => {
      cancelled = true;
      if (rackTimer.current) {
        clearTimeout(rackTimer.current);
        rackTimer.current = null;
      }
      void enqueue(() => engineRef.current?.disconnect());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, racksReady, scopeKey, liveEpoch]);

  const flushRack = () => {
    rackTimer.current = null;
    rackLast.current = performance.now();
    const live = engineRef.current;
    if (!live) return;
    const rack = cloneValue(racksRef.current[scopeKeyRef.current] || []);
    live.replaceRack(rack);
    remember(rack);
    if (abRef.current) rack.forEach((fx) => live.bypass(fx.id, true)); // B stays B
  };

  /** The whole rack to the engine (it restarts at the playhead), at most once per 100 ms. */
  const sendRack = () => {
    if (rackTimer.current) return; // the pending send reads the latest rack
    const wait = rackLast.current + RACK_THROTTLE_MS - performance.now();
    if (wait <= 0) flushRack();
    else rackTimer.current = setTimeout(flushRack, wait);
  };

  // -- rack edits -----------------------------------------------------------------------------
  const [selectedFx, setSelectedFx] = useState<string | null>(null);
  const [processId, setProcessId] = useState<string | null>(null);
  const [editorTab, setEditorTab] = useState<'effect' | 'essential'>('effect');

  const currentRack = () => racksRef.current[scopeKeyRef.current] || [];

  const commitRack = (next: AudioEffect[]) => {
    const key = scopeKeyRef.current;
    racksRef.current = { ...racksRef.current, [key]: next };
    setRacks(racksRef.current);
    schedulePut(key);
  };

  const addEffect = (def: EffectDef) => {
    const fx: AudioEffect = {
      id: newEffectId(allIds(racksRef.current)),
      effect: def.id,
      params: defaultParams(def),
      enabled: true,
      start_s: null,
      end_s: null,
      preset: null,
    };
    commitRack([...currentRack(), fx]);
    setSelectedFx(fx.id);
    setProcessId(null);
    setEditorTab('effect');
    sendRack();
  };

  const changeParam = (id: string, key: string, value: unknown) => {
    const rack = currentRack();
    const fx = rack.find((e) => e.id === id);
    if (!fx) return;
    commitRack(rack.map((e) => (e.id === id ? { ...e, params: { ...e.params, [key]: value }, preset: keepTag(e, null) } : e)));
    const param = (defs[fx.effect]?.params || []).find((p) => p.key === key);
    const live = engineRef.current;
    if (!live) return;
    // a pending rack already carries the value, and the engine may not know a new effect yet
    if (param?.live && !rackTimer.current && serverEnabled.current.has(id)) live.set(id, key, value);
    else sendRack();
  };

  const toggleEnabled = (id: string) => {
    const rack = currentRack();
    const fx = rack.find((e) => e.id === id);
    if (!fx) return;
    const on = !fx.enabled;
    commitRack(rack.map((e) => (e.id === id ? { ...e, enabled: on } : e)));
    const live = engineRef.current;
    if (!live || abRef.current) return; // during B everything stays bypassed until A
    const known = serverEnabled.current.get(id);
    if (rackTimer.current || known === undefined) {
      sendRack(); // the engine has not got this effect yet: the rack carries the switch
      return;
    }
    live.bypass(id, !on);
    // switched off when the engine got the rack: it is not in the running graph yet
    if (on && known === false) sendRack();
  };

  const moveEffect = (from: number, to: number) => {
    const rack = [...currentRack()];
    if (from === to || from < 0 || to < 0 || from >= rack.length || to >= rack.length) return;
    const [fx] = rack.splice(from, 1);
    rack.splice(to, 0, fx);
    commitRack(rack);
    sendRack();
  };

  const removeEffect = (id: string) => {
    commitRack(currentRack().filter((e) => e.id !== id));
    if (selectedFx === id) setSelectedFx(null);
    sendRack();
  };

  const toggleWindow = (id: string) => {
    const rack = currentRack();
    const fx = rack.find((e) => e.id === id);
    if (!fx) return;
    const windowed = fx.start_s != null || fx.end_s != null;
    const sel = useProjectStore.getState().audioSelection;
    const span = sel && sel.end_s > sel.start_s ? sel : null;
    if (!windowed && !span) return;
    const round = (v: number) => Math.round(v * 1000) / 1000;
    commitRack(rack.map((e) => {
      if (e.id !== id) return e;
      return windowed || !span
        ? { ...e, start_s: null, end_s: null }
        : { ...e, start_s: round(span.start_s), end_s: round(span.end_s) };
    }));
    sendRack();
  };

  const applyPreset = (id: string, name: string | null, values: Record<string, unknown>) => {
    const rack = currentRack();
    const fx = rack.find((e) => e.id === id);
    const def = fx ? defs[fx.effect] : undefined;
    if (!fx || !def) return;
    const params = { ...defaultParams(def), ...cloneValue(values) };
    commitRack(rack.map((e) => (e.id === id ? { ...e, params, preset: keepTag(e, name) } : e)));
    const changed = (def.params || []).filter(
      (p) => JSON.stringify(params[p.key]) !== JSON.stringify(fx.params[p.key] ?? p.default));
    const live = engineRef.current;
    if (!live || changed.length === 0) return;
    if (!rackTimer.current && serverEnabled.current.has(id) && changed.every((p) => p.live)) {
      changed.forEach((p) => live.set(id, p.key, params[p.key]));
    } else sendRack();
  };

  // -- rack presets ---------------------------------------------------------------------------

  const saveRackAsPreset = async (name: string) => {
    const res = await saveRackPreset(name, cloneValue(currentRack()));
    setRackPresets(res.presets || {});
  };

  /** The backend replaces the scope's rack (fresh ids); the panel, the store and the engine follow. */
  const loadRackPreset = async (name: string) => {
    const pid = projectIdRef.current;
    if (!pid) return;
    const key = scopeKeyRef.current;
    const pending = putTimers.current.get(key);
    if (pending) {
      // a pending save of the rack being replaced must not land on top of the preset
      clearTimeout(pending.timer);
      putTimers.current.delete(key);
    }
    const res = await applyRackPreset(pid, key, name);
    const effects = normalizeRack(res.effects);
    racksRef.current = { ...racksRef.current, [key]: effects };
    setRacks(racksRef.current);
    patchTimeline(pid, key, effects, res.revision);
    setSaveNote(null);
    setSelectedFx(null);
    if (scopeKeyRef.current === key) sendRack();
  };

  // -- Essential Sound ------------------------------------------------------------------------
  // Each control owns one effect of the scope's rack, found by its tag, and edits it through
  // the same live and save path as the rack's own edits.

  const setEssential = (control: string, effectId: string, values: Record<string, unknown>) => {
    const def = defs[effectId];
    if (!def) return;
    const tag = `${ESSENTIAL_TAG}${control}`;
    const rack = currentRack();
    const fx = rack.find((e) => e.preset === tag);
    if (!fx || fx.effect !== effectId) {
      // added, or the control changed engine (Reduce Noise): the engine gets the whole rack
      const next: AudioEffect = {
        id: newEffectId(allIds(racksRef.current)),
        effect: effectId,
        params: { ...defaultParams(def), ...cloneValue(values) },
        enabled: true,
        start_s: null,
        end_s: null,
        preset: tag,
      };
      commitRack(fx ? rack.map((e) => (e.id === fx.id ? next : e)) : [...rack, next]);
      if (fx && selectedFx === fx.id) setSelectedFx(next.id);
      sendRack();
      return;
    }
    const params = { ...fx.params, ...cloneValue(values) };
    const changed = (def.params || []).filter(
      (p) => JSON.stringify(params[p.key]) !== JSON.stringify(fx.params[p.key] ?? p.default));
    if (changed.length === 0 && fx.enabled) return;
    commitRack(rack.map((e) => (e.id === fx.id ? { ...e, params, enabled: true } : e)));
    const live = engineRef.current;
    if (!live) return;
    // in the running graph (sent switched on, no rack pending): live params go out as `set`
    const inGraph = !rackTimer.current && serverEnabled.current.get(fx.id) === true;
    if (inGraph && changed.every((p) => p.live)) {
      changed.forEach((p) => live.set(fx.id, p.key, params[p.key]));
      if (!fx.enabled && !abRef.current) live.bypass(fx.id, false); // during B it stays bypassed
    } else sendRack();
  };

  /** A control's tick: on adds its effect (or switches it back on), off bypasses it. */
  const switchEssential = (control: string, effectId: string, on: boolean, values: Record<string, unknown>) => {
    const fx = currentRack().find((e) => e.preset === `${ESSENTIAL_TAG}${control}`);
    if (!fx) {
      if (on) setEssential(control, effectId, values);
      return;
    }
    if (fx.enabled !== on) toggleEnabled(fx.id);
  };

  // -- transport ------------------------------------------------------------------------------
  const togglePlay = () => {
    const live = engineRef.current;
    if (!live || liveStatus !== 'live') return;
    if (playing) {
      live.pause();
      setPlaying(false);
      return;
    }
    const r = rangeRef.current;
    if (r && r.origin !== 'programme' && !loopOn && (posRef.current < r.start_s || posRef.current >= r.end_s - 0.05)) {
      live.seek(r.start_s);
    }
    live.play().catch((err) => setLiveNote(`Could not start playback: ${errorText(err)}`));
    setPlaying(true);
  };

  const toggleLoop = () => {
    const live = engineRef.current;
    if (!live) return;
    if (loopOn) {
      live.loopOff();
      setLoopOn(false);
      return;
    }
    const r = rangeRef.current;
    if (!r) return;
    live.loop(r.start_s, r.end_s);
    setLoopOn(true);
  };

  // A new range while looping: loop that one instead.
  const loopKey = range ? `${range.start_s}:${range.end_s}` : '';
  useEffect(() => {
    if (!loopOn) return;
    const live = engineRef.current;
    const r = rangeRef.current;
    if (!live) return;
    if (r) live.loop(r.start_s, r.end_s);
    else {
      live.loopOff();
      setLoopOn(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loopKey]);

  const toggleAB = () => {
    const live = engineRef.current;
    if (!live) return;
    const rack = currentRack();
    if (!abRef.current) {
      setAb(true);
      rack.forEach((fx) => { if (serverEnabled.current.has(fx.id)) live.bypass(fx.id, true); });
      // effects the engine has not got yet arrive with the next rack, which re-bypasses all of them
      if (rack.some((fx) => !serverEnabled.current.has(fx.id))) sendRack();
      return;
    }
    setAb(false);
    let needRack = false;
    rack.forEach((fx) => {
      if (!fx.enabled) return; // switched off: stays bypassed
      const known = serverEnabled.current.get(fx.id);
      if (known === undefined) {
        needRack = true;
        return;
      }
      live.bypass(fx.id, false);
      if (known === false) needRack = true;
    });
    if (needRack) sendRack();
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== ' ' && event.code !== 'Space') return;
    const target = event.target as HTMLElement;
    const tag = target.tagName;
    const type = tag === 'INPUT' ? ((target as HTMLInputElement).type || 'text') : '';
    if (tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable
      || (tag === 'INPUT' && type !== 'range' && type !== 'button')) return;
    event.preventDefault();
    event.stopPropagation(); // the editor's own Space (timeline play) must not fire too
    if (!event.repeat) togglePlay();
  };

  // -- processes ------------------------------------------------------------------------------
  const onSourceChanged = useCallback(() => {
    const pid = projectIdRef.current;
    if (pid) {
      getTimeline(pid)
        .then((tl) => {
          const state = useProjectStore.getState();
          if (tl && state.project?.id === pid) state.updateProject({ timeline: tl });
        })
        .catch(() => { /* the next timeline load picks it up */ });
    }
    setLiveEpoch((n) => n + 1); // the live engine reads the new version when it reconnects
  }, []);

  // -- analysis -------------------------------------------------------------------------------
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [spectrumUrl, setSpectrumUrl] = useState<string | null>(null);
  const [analysisNote, setAnalysisNote] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const analyze = async () => {
    if (!projectId || !analysisRange) return;
    setAnalyzing(true);
    setAnalysisNote(null);
    try {
      const res = await analyzeRange(projectId, {
        source_id: sourceId, start_s: analysisRange.start_s, end_s: analysisRange.end_s,
      });
      setAnalysis(res);
      setSpectrumUrl(res.spectrum_png ? audioFileUrl(res.spectrum_png) : null);
    } catch (err) {
      setAnalysisNote(routeMissing(err) ? `Analysis is ${MISSING}.` : `Analysis failed: ${errorText(err)}`);
    } finally {
      setAnalyzing(false);
    }
  };

  if (!projectId) {
    return <div className="fx-empty">Open a project to work on its sound.</div>;
  }

  const rack = racks[scopeKey] || [];
  const selected = rack.find((e) => e.id === selectedFx) || null;
  const selectedDef = selected ? defs[selected.effect] : undefined;
  const processDef = processId ? processDefs[processId] : undefined;
  // Essential Sound's Reduce Reverb: DeepFilterNet as a process on the range.
  const aiDef = processDefs.noise_reduction_ai;
  const reduceReverb = aiDef ? (
    <ProcessCard key={`essential:${aiDef.id}:${scopeKey}`} def={aiDef} projectId={projectId}
      sourceId={sourceId} sourceLabel={sourceLabel} range={sourceRange}
      playhead={() => offset + posRef.current}
      userPresets={userPresets} onPresetsChange={setUserPresets}
      onBeforePreview={() => {
        engineRef.current?.pause();
        setPlaying(false);
      }}
      onSourceChanged={onSourceChanged} />
  ) : null;

  return (
    <div className="audio-panel" tabIndex={0} onKeyDown={onKeyDown}>
      <div className="fx-scope audio-scopes">
        <button className={`insp-chip${tab === 'A1' ? ' active' : ''}`} title={SCOPE_HINT.A1}
          onClick={() => setTab('A1')}>
          Programme (A1)
        </button>
        <button className={`insp-chip${tab === 'clip' ? ' active' : ''}`} disabled={!clipEligible}
          title={clipEligible ? SCOPE_HINT.clip : 'Select a clip on A2 or below to give it effects of its own'}
          onClick={() => setTab('clip')}>
          Clip{clipEligible ? ` · ${itemTrack}` : ''}
        </button>
        <button className={`insp-chip${tab === 'MASTER' ? ' active' : ''}`} title={SCOPE_HINT.MASTER}
          onClick={() => setTab('MASTER')}>
          Master
        </button>
      </div>
      <div className="audio-scope-hint">{SCOPE_HINT[isClipScope ? 'clip' : scopeKey === 'MASTER' ? 'MASTER' : 'A1']}</div>

      <Transport position={position} playing={playing} loopOn={loopOn} ab={ab} range={range}
        status={liveStatus} note={liveNote}
        onPlayPause={togglePlay} onLoop={toggleLoop} onAB={toggleAB}
        onReconnect={() => setLiveEpoch((n) => n + 1)} />
      <div className="audio-analyze-row" title="The timeline's audio range, in programme seconds">
        {audioSelection ? (
          <>
            <span className="audio-inline">
              Selection {formatTime(audioSelection.start_s)}–{formatTime(audioSelection.end_s)}
              {' · '}{(audioSelection.end_s - audioSelection.start_s).toFixed(2)} s
            </span>
            <button className="audio-btn" title="Clear the selection (Esc on the timeline)"
              onClick={() => setAudioSelection(null)}>
              Clear
            </button>
          </>
        ) : (
          <span className="audio-hint">
            No selection: Shift+drag on an audio lane or the timeline ruler, I / O at the playhead,
            or the timeline's Range toggle.
          </span>
        )}
      </div>
      <Meters signal={meters} />

      {[rackNote, saveNote].filter(Boolean).map((note) => (
        <div key={note as string} className="audio-note">{note}</div>
      ))}

      <div className="audio-body">
        <div className="audio-col audio-col-tree">
          <div className="audio-col-title">Effects</div>
          <EffectTree categories={catalogue?.categories || []} effects={catalogue?.effects || []}
            processes={processes} selectedProcess={processId} canAdd={racksReady}
            notes={[catalogueNote, processNote]}
            onAdd={addEffect}
            onPickProcess={(def) => { setProcessId(def.id); setSelectedFx(null); setEditorTab('effect'); }} />
        </div>
        <div className="audio-col audio-col-rack">
          <RackPresetBar names={Object.keys(rackPresets)} canSave={racksReady && rack.length > 0}
            canLoad={racksReady} onSave={saveRackAsPreset} onLoad={loadRackPreset} />
          <RackList rack={rack} defs={defs} selectedId={selectedFx} selection={audioSelection}
            userPresets={userPresets} scopeLabel={scopeLabel}
            onSelect={(id) => { setSelectedFx(id); setProcessId(null); setEditorTab('effect'); }}
            onToggle={toggleEnabled} onMove={moveEffect} onRemove={removeEffect} onWindow={toggleWindow}
            onApplyPreset={applyPreset} onPresetsChange={setUserPresets} />
        </div>
        <div className="audio-col audio-col-editor">
          <div className="fx-scope">
            <button className={`insp-chip${editorTab === 'effect' ? ' active' : ''}`}
              title="The selected rack effect's parameters, or a process effect's card"
              onClick={() => setEditorTab('effect')}>
              Effect
            </button>
            <button className={`insp-chip${editorTab === 'essential' ? ' active' : ''}`}
              title="Audition's Essential Sound: quick dialogue controls that write into this rack"
              onClick={() => setEditorTab('essential')}>
              Essential Sound
            </button>
          </div>
          {editorTab === 'essential' ? (
            <EssentialSound rack={rack} defs={defs} scopeLabel={scopeLabel} canEdit={racksReady}
              onSet={setEssential} onSwitch={switchEssential} reduceReverb={reduceReverb} />
          ) : processDef ? (
            <ProcessCard key={`${processDef.id}:${scopeKey}`} def={processDef} projectId={projectId}
              sourceId={sourceId} sourceLabel={sourceLabel} range={sourceRange}
              playhead={() => offset + posRef.current}
              userPresets={userPresets} onPresetsChange={setUserPresets}
              onBeforePreview={() => {
                engineRef.current?.pause();
                setPlaying(false);
              }}
              onSourceChanged={onSourceChanged} />
          ) : selected && selectedDef ? (
            <div className="audio-editor">
              <div className="audio-col-title">
                {selectedDef.name}{selected.enabled ? '' : ' (bypassed)'}
              </div>
              <div className="text-xs text-muted">{selectedDef.description}</div>
              {!selectedDef.windowable && (
                <div className="text-xs text-muted">Covers the whole scope: it cannot be windowed.</div>
              )}
              <ParamEditor def={selectedDef} values={selected.params}
                onChange={(key, value) => changeParam(selected.id, key, value)} />
            </div>
          ) : (
            <div className="audio-editor-empty text-xs text-muted">
              Pick an effect in the rack to set it, or a process effect in the tree.
            </div>
          )}
        </div>
      </div>

      <Section title="Analyze" defaultOpen={false}>
        <div className="audio-analyze-row">
          <button className="audio-btn primary" disabled={analyzing || !analysisRange} onClick={() => void analyze()}>
            {analyzing ? 'Analyzing…' : 'Analyze selection'}
          </button>
          <span className="text-xs text-muted">
            {analysisRange
              ? `${sourceLabel}, ${formatTime(analysisRange.start_s)}–${formatTime(analysisRange.end_s)}`
              : 'No range to analyze'}
          </span>
        </div>
        {analysis && (
          <div className="audio-analysis">
            <div className="audio-stats">
              <Stat label="Integrated" value={analysis.integrated_lufs} unit="LUFS" />
              <Stat label="Loudness range" value={analysis.lra_lu} unit="LU" />
              <Stat label="True peak" value={analysis.true_peak_dbtp} unit="dBTP" />
              <Stat label="Peak" value={analysis.peak_dbfs} unit="dBFS" />
              <Stat label="RMS" value={analysis.rms_dbfs} unit="dBFS" />
              <Stat label="Noise floor" value={analysis.noise_floor_dbfs} unit="dBFS" />
            </div>
            {spectrumUrl && <img className="audio-spectrum" src={spectrumUrl} alt="Spectrum of the analysed range" />}
          </div>
        )}
        {analysisNote && <div className="audio-note">{analysisNote}</div>}
      </Section>
    </div>
  );
}
