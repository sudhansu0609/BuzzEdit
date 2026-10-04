import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { create } from 'zustand';
import { useProjectStore } from '../hooks/store';
import {
  useElectron, trimClip, addMedia, getWaveform,
  detachAudio, compoundClips, uncompoundClip, addTextClip, addAdjustment,
  setTrackFlags, deleteTrack, TrackState, WaveformData,
  getFilmstrip, filmstripImageUrl, FilmstripData,
  cutProgramRange, restoreProgramCuts,
  splitClips, deleteClips, moveClips, trimClips, linkClips, unlinkClips, pasteClips, closeGap,
  setClipFlags, setTransform,
} from '../hooks/api';
import { MEDIA_DND_TYPE } from './MediaPool';
import { COMMANDS, useCommandStore } from '../hooks/commands';
import { chordLabel } from '../hooks/shortcuts';
import { useTimelineHistory } from '../hooks/history';
import { effectiveTransitions } from '../lib/programme';
import { prettyTransition } from './TransitionsPanel';
import ContextMenu, { MenuEntry } from './ContextMenu';

// Build a filled mirrored-waveform SVG path for a source-time slice of peaks.
function buildWavePath(peaks: number[], pps: number, startSec: number, endSec: number, width: number, height: number): string {
  const startIdx = Math.max(0, Math.floor(startSec * pps));
  const endIdx = Math.min(peaks.length, Math.ceil(endSec * pps));
  const slice = peaks.slice(startIdx, endIdx);
  if (slice.length === 0 || width < 2) return '';
  const n = Math.max(1, Math.min(slice.length, Math.floor(width / 2)));
  const step = slice.length / n;
  const mid = height / 2;
  let top = '';
  let bottom = '';
  for (let i = 0; i < n; i++) {
    let peak = 0;
    for (let j = Math.floor(i * step); j < Math.floor((i + 1) * step); j++) peak = Math.max(peak, slice[j] || 0);
    const x = n > 1 ? (i / (n - 1)) * width : 0;
    const y = peak * mid * 0.9;
    top += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${(mid - y).toFixed(1)} `;
    bottom = `L${x.toFixed(1)},${(mid + y).toFixed(1)} ` + bottom;
  }
  return top + bottom + 'Z';
}

function ClipWaveform({ wave, sourceStartSec, sourceEndSec, width, height }: {
  wave: WaveformData; sourceStartSec: number; sourceEndSec: number; width: number; height: number;
}) {
  const d = useMemo(
    () => buildWavePath(wave.peaks, wave.points_per_second, sourceStartSec, sourceEndSec, width, height),
    [wave, sourceStartSec, sourceEndSec, width, height],
  );
  if (!d) return null;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      <path d={d} fill="rgba(255,255,255,0.45)" />
    </svg>
  );
}

// A strip covers the whole source, so the slice under a clip is found by scaling
// it to the clip's pixels-per-second and shifting it left by the trim-in point.
// Left-edge alignment is therefore always exact; only the rate degrades when the
// clamp below kicks in.
//
// Chromium refuses to paint a background scaled past roughly 16k px, so a very
// short clip cut from very long footage — which would demand a strip hundreds of
// thousands of pixels wide — is capped. Past that point the frames advance
// slower than real time, which at a few pixels per frame reads as "roughly this
// part of the shot", and that is all the strip is for.
const MAX_STRIP_PX = 16000;

function ClipFilmstrip({ strip, projectId, sourceId, sourceStartSec, sourceEndSec, width }: {
  strip: FilmstripData; projectId: string; sourceId: string;
  sourceStartSec: number; sourceEndSec: number; width: number;
}) {
  const style = useMemo<React.CSSProperties>(() => {
    const url = `url("${filmstripImageUrl(projectId, sourceId)}")`;
    // A still has no timeline of its own: repeat the one frame across the clip.
    if (strip.columns <= 1 || strip.duration <= 0) {
      return { backgroundImage: url, backgroundSize: 'auto 100%', backgroundRepeat: 'repeat-x' };
    }
    const visibleSpan = Math.max(0.001, sourceEndSec - sourceStartSec);
    const stripWidth = Math.min((strip.duration / visibleSpan) * width, MAX_STRIP_PX);
    return {
      backgroundImage: url,
      backgroundSize: `${stripWidth.toFixed(1)}px 100%`,
      backgroundPosition: `${(-(sourceStartSec / strip.duration) * stripWidth).toFixed(1)}px 0`,
      backgroundRepeat: 'no-repeat',
    };
  }, [strip, projectId, sourceId, sourceStartSec, sourceEndSec, width]);

  return (
    <div aria-hidden style={{
      position: 'absolute', inset: 0, borderRadius: 3, overflow: 'hidden',
      pointerEvents: 'none', opacity: 0.9, ...style,
    }} />
  );
}

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 5;
const LANE_HEIGHT = 46;
const HANDLE_PX = 8;

/** How close (in pixels) a dragged edge must come to another edge to snap to it. */
const SNAP_PX = 8;
/** Movement (px) before a press on a clip becomes a drag rather than a click. */
const DRAG_THRESHOLD_PX = 3;

interface TItem {
  id: string;
  track: string;
  source_id: string;
  source_start_frame: number;
  source_end_frame: number;
  timeline_start_frame: number;
  timeline_end_frame: number;
  enabled: boolean;
  origin?: string;
  locked?: boolean;
  kind?: 'media' | 'text' | 'compound' | 'adjustment';
  label?: string | null;
  mute?: boolean;
  text?: { content?: string } | null;
  children?: TItem[];
  transform?: unknown;
  color?: unknown;
  atmosphere?: unknown[];
  audio_fade_in?: number;
  audio_fade_out?: number;
  duck?: number;
  loop?: boolean;
  /** Shared by the picture and sound halves of one piece of media. */
  link_id?: string | null;
}
interface TTimeline {
  fps_num: number;
  fps_den: number;
  items: TItem[];
  sources: Record<string, { kind?: string; path?: string; duration_seconds?: number }>;
  duration_frames?: number;
  program_offset_frames?: number;
  tracks?: Record<string, TrackState>;
  extra_tracks?: string[];
  /** Programme-frame ranges ripple-cut out of the auto-edited V1/A1 spine. */
  manual_cuts?: unknown[];
}

type DragMode = 'move' | 'trim-start' | 'trim-end';
interface DragState {
  mode: DragMode;
  /** The clip under the pointer. */
  id: string;
  /** Every clip that moves (or trims) with it: the selection and linked partners. */
  ids: string[];
  orig: Record<string, { start: number; end: number; track: string }>;
  startX: number;
  startY: number;
  /** Frames moved so far, after snapping and clamping. */
  delta: number;
  /** Lane the grabbed clip is over (move only). */
  previewTrack: string;
  /** A lane change is only offered when one clip of that kind is moving. */
  laneChange: boolean;
  /** Bounds for `delta`, worked out once at the start of the gesture. */
  minDelta: number;
  maxDelta: number;
  /** False for an auto (ripple-cut) clip: trim edges shrink-only and land on
   *  program/cut rather than clip/trim on release. */
  editable: boolean;
  /** Alt held: move or trim this clip alone, ignoring its link. */
  unlinked: boolean;
  /** Past the click threshold — until then this is a click, not a drag. */
  moved: boolean;
  /** Selection to apply if the press turns out to be a plain click. */
  clickSelect: string[] | null;
  snapFrame: number | null;
}

interface Marquee {
  x0: number; y0: number; x1: number; y1: number;
  additive: boolean;
  base: string[];
  moved: boolean;
  track: string | null;
}

interface MenuState { x: number; y: number; items: MenuEntry[] }

/** Clips copied with Ctrl+C. Kept in memory — clips are not text, so the OS
 *  clipboard is the wrong place — and shared by every timeline in the session. */
const useClipboard = create<{ items: TItem[] | null; set: (items: TItem[] | null) => void }>((set) => ({
  items: null,
  set: (items) => set({ items }),
}));

const SNAP_STORAGE_KEY = 'buzzedit.timelineSnap';

function loadSnap(): boolean {
  try {
    return localStorage.getItem(SNAP_STORAGE_KEY) !== 'off';
  } catch {
    return true;
  }
}

function saveSnap(on: boolean) {
  try {
    localStorage.setItem(SNAP_STORAGE_KEY, on ? 'on' : 'off');
  } catch {
    /* ignore */
  }
}

/** The backend's `{"detail": "..."}` instead of the raw `API 400: {...}`. */
function cleanError(err: any, fallback: string): string {
  const msg = String(err?.message || '');
  const m = msg.match(/^API \d+: ([\s\S]*)$/);
  if (m) {
    try {
      const body = JSON.parse(m[1]);
      if (typeof body.detail === 'string') return body.detail;
    } catch {
      /* not JSON */
    }
  }
  return msg || fallback;
}

/** Overlapping or touching [start, end) spans merged into one. */
function mergeSpans(spans: Array<[number, number]>): Array<[number, number]> {
  const sorted = [...spans].sort((a, b) => a[0] - b[0]);
  const out: Array<[number, number]> = [];
  for (const [s, e] of sorted) {
    const last = out[out.length - 1];
    if (last && s <= last[1]) last[1] = Math.max(last[1], e);
    else out.push([s, e]);
  }
  return out;
}

const COMMAND_LABELS: Record<string, string> = Object.fromEntries(COMMANDS.map(c => [c.id, c.label]));

/** Every command the timeline answers while it is mounted. */
const TIMELINE_COMMAND_IDS = [
  'edit.cut', 'edit.copy', 'edit.paste', 'edit.pasteInsert', 'edit.duplicate',
  'edit.deleteClip', 'edit.rippleDelete', 'edit.selectAll', 'edit.deselectAll',
  'edit.splitAtPlayhead', 'edit.markIn', 'edit.markOut',
  'timeline.trimStart', 'timeline.trimEnd', 'timeline.rippleTrimStart', 'timeline.rippleTrimEnd',
  'timeline.closeGap', 'timeline.selectForward', 'timeline.link', 'timeline.detachAudio',
  'timeline.compound', 'timeline.uncompound', 'timeline.toggleEnabled', 'timeline.toggleMute',
  'timeline.toggleLock', 'timeline.rename', 'timeline.rotateCW', 'timeline.rotateCCW',
  'timeline.flipH', 'timeline.flipV', 'timeline.resetTransform',
  'timeline.nudgeLeft', 'timeline.nudgeRight', 'timeline.toggleSnap',
  'playback.prevFrame', 'playback.nextFrame', 'playback.back1s', 'playback.fwd1s',
  'playback.prevEdit', 'playback.nextEdit', 'playback.start', 'playback.end',
  'view.zoomIn', 'view.zoomOut', 'view.zoomFit',
] as const;
type TimelineCommandId = typeof TIMELINE_COMMAND_IDS[number];

type FollowMode = 'off' | 'left' | 'center' | 'right';
const FOLLOW_STORAGE_KEY = 'buzzedit.timelineFollow';

function loadFollowMode(): FollowMode {
  try {
    const raw = localStorage.getItem(FOLLOW_STORAGE_KEY);
    if (raw === 'off' || raw === 'left' || raw === 'center' || raw === 'right') return raw;
  } catch {
    /* a full or disabled localStorage must not break the editor */
  }
  return 'center';
}

function saveFollowMode(mode: FollowMode) {
  try {
    localStorage.setItem(FOLLOW_STORAGE_KEY, mode);
  } catch {
    /* ignore */
  }
}

const trackNum = (t: string) => parseInt(t.replace(/\D/g, '') || '0', 10);
const trackKind = (t: string): 'A' | 'T' | 'V' => {
  const u = t.toUpperCase();
  if (u.startsWith('A')) return 'A';
  if (u.startsWith('T')) return 'T';
  return 'V';
};

// Lane colours by kind, mirroring the inspector's grouping: text on top, then
// picture, then sound.
const LANE_TINT: Record<string, string> = {
  T: 'rgba(168,85,247,0.05)',
  A: 'rgba(59,130,246,0.04)',
  V: 'transparent',
};

const clipFill = (item: TItem, kind: 'A' | 'T' | 'V', editable: boolean) => {
  if (!editable) return '#444';
  if (item.kind === 'adjustment') return '#b45309';
  if (item.kind === 'compound') return '#7c3aed';
  if (item.kind === 'text') return '#9333ea';
  return kind === 'A' ? '#1e6f5c' : '#2a5c9a';
};

const clipLabel = (item: TItem, mediaKind: string, laneKind: 'A' | 'T' | 'V') => {
  if (item.kind === 'text') return `T ${item.text?.content || item.label || ''}`.slice(0, 40);
  if (item.kind === 'compound') return `▣ ${item.label || 'Compound'}`;
  if (item.kind === 'adjustment') return `◐ ${item.label || 'Adjustment'}`;
  // On an audio lane only the sound is used, even when the source is a video file.
  const icon = laneKind === 'A' || mediaKind === 'audio' ? '🔊' : mediaKind === 'image' ? '🖼' : '🎞';
  return `${icon} ${item.label || item.id.slice(-4)}`;
};

// An AI-managed V1/A1 clip: not draggable, but its edges can still ripple-cut
// the programme and it can still be styled from the Inspector.
const isAutoCut = (item: TItem) => item.origin === 'auto' && !item.locked;

interface DressingChip { key: string; label: string; title: string }

/** What render-time dressing is on this clip, distilled into the small chips
 *  drawn on the lane — the only place that state is otherwise visible before
 *  export. */
function dressingChips(item: TItem): DressingChip[] {
  const chips: DressingChip[] = [];
  const t = item.transform as Record<string, any> | undefined;
  if (t && (t.scale_end != null || (typeof t.scale === 'number' && t.scale !== 1))) {
    const from = typeof t.scale === 'number' ? t.scale : 1;
    const to = t.scale_end != null ? t.scale_end : from;
    chips.push({ key: 'zoom', label: 'zoom', title: `zoom ${from.toFixed(2)}→${to.toFixed(2)}` });
  }
  const c = item.color as Record<string, any> | undefined;
  const fadeIn = (c?.fade_in ?? 0) > 0 || (item.audio_fade_in ?? 0) > 0;
  const fadeOut = (c?.fade_out ?? 0) > 0 || (item.audio_fade_out ?? 0) > 0;
  if (fadeIn || fadeOut) {
    chips.push({
      key: 'fade', label: 'fade',
      title: `fade ${[fadeIn ? 'in' : '', fadeOut ? 'out' : ''].filter(Boolean).join(' / ')}`,
    });
  }
  const gradedFields = ['exposure', 'brightness', 'contrast', 'gamma', 'temperature', 'tint',
    'shadows', 'highlights', 'saturation', 'vibrance', 'hue', 'lut_file', 'sharpen', 'denoise', 'vignette'];
  const isGraded = !!c && (!!c.preset || gradedFields.some((f) => !!c[f]));
  if (isGraded) {
    chips.push({ key: 'grade', label: 'grade', title: c!.preset ? `grade: ${c!.preset}` : 'grade: custom' });
  }
  (item.atmosphere as Array<Record<string, any>> | undefined)?.forEach((fx, idx) => {
    if (!fx || fx.enabled === false) return;
    const label = String(fx.type || 'fx').replace(/_/g, ' ');
    chips.push({ key: `fx-${idx}`, label, title: `effect: ${label}` });
  });
  return chips;
}

export default function Timeline() {
  const containerRef = useRef<HTMLDivElement>(null);
  const lanesRef = useRef<HTMLDivElement>(null);
  const {
    project, updateProject, currentTime, setCurrentTime, selectedClipId, selectedClipIds,
    setSelection, zoom, setZoom, setError, isPlaying,
    setScrubbing: setStoreScrubbing,
  } = useProjectStore();
  const { pickFile } = useElectron();
  const clipboardItems = useClipboard((s) => s.items);
  const setClipboard = useClipboard((s) => s.set);
  const canUndo = useTimelineHistory((s) => s.past.length > 0);
  const canRedo = useTimelineHistory((s) => s.future.length > 0);
  const undo = useTimelineHistory((s) => s.undo);
  const redo = useTimelineHistory((s) => s.redo);
  // Re-render on rebinding so the chords printed in the menus stay true.
  useCommandStore((s) => s.keymap);

  const [drag, setDrag] = useState<DragState | null>(null);
  const dragRef = useRef<DragState | null>(null);
  dragRef.current = drag;
  const [marquee, setMarquee] = useState<Marquee | null>(null);
  const marqueeRef = useRef<Marquee | null>(null);
  marqueeRef.current = marquee;
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [rename, setRename] = useState<{ ids: string[]; value: string } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [snap, setSnapState] = useState<boolean>(loadSnap);
  const snapRef = useRef(snap);
  snapRef.current = snap;
  const snapCandidatesRef = useRef<number[]>([]);
  const anchorRef = useRef<string | null>(null);
  const [localEmptyTracks, setLocalEmptyTracks] = useState<string[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [scrubbing, setScrubbing] = useState(false);
  const [waveforms, setWaveforms] = useState<Record<string, WaveformData>>({});
  const [filmstrips, setFilmstrips] = useState<Record<string, FilmstripData | null>>({});
  const [followMode, setFollowModeState] = useState<FollowMode>(loadFollowMode);
  const [markIn, setMarkIn] = useState<number | null>(null);
  const [markOut, setMarkOut] = useState<number | null>(null);
  const programmaticScrollRef = useRef(false);
  const lastManualScrollAtRef = useRef(0);

  const setFollowMode = useCallback((mode: FollowMode) => {
    setFollowModeState(mode);
    saveFollowMode(mode);
  }, []);

  const tl = project?.timeline as TTimeline | undefined;
  const projectId = project?.id;
  const fps = tl && tl.fps_num ? tl.fps_num / (tl.fps_den || 1) : 30;
  const pixelsPerSecond = 50 * zoom;
  const frameToPx = (f: number) => (f / fps) * pixelsPerSecond;
  const pxToFrame = (px: number) => Math.round((px / pixelsPerSecond) * fps);
  const playFrame = Math.round(currentTime * fps);

  const items = tl?.items ?? [];

  // Transitions and programme effects had no representation on the lanes at all,
  // which is why applying one looked like nothing had happened: the panel said
  // "Dissolve", the preview plays the source file, and the timeline was silent.
  // These two are the edit's own record of what the render will add.
  const junctions = useMemo(() => effectiveTransitions(tl), [tl]);
  const programmeEffects: any[] = (tl as any)?.effects ?? [];
  const programmeTransition = (tl as any)?.default_transition ?? null;

  const maxEndFrame = items.reduce((m, i) => Math.max(m, i.timeline_end_frame), 0);
  const durationSec = Math.max(maxEndFrame / fps, currentTime + 5, 30);
  const trackWidth = durationSec * pixelsPerSecond;

  // Ordered tracks: text on top, then video (highest first), then audio (A1 first),
  // plus any empty lanes the user has reserved.
  const trackSet = new Set<string>([
    ...items.map(i => i.track).filter(t => t.toUpperCase() !== 'CAP'),
    ...(tl?.extra_tracks ?? []),
    ...localEmptyTracks,
  ]);
  const textTracks = [...trackSet].filter(t => trackKind(t) === 'T').sort((a, b) => trackNum(a) - trackNum(b));
  const videoTracks = [...trackSet].filter(t => trackKind(t) === 'V').sort((a, b) => trackNum(b) - trackNum(a));
  const audioTracks = [...trackSet].filter(t => trackKind(t) === 'A').sort((a, b) => trackNum(a) - trackNum(b));
  // Captions used to be filtered off the lanes entirely, which hid every effect
  // of the dressing pass on them. They get one lane of their own, always first,
  // named for what they are rather than the internal track id.
  const capTrack = items.find(i => i.track.toUpperCase() === 'CAP')?.track;
  const orderedTracks = [...(capTrack ? [capTrack] : []), ...textTracks, ...videoTracks, ...audioTracks];
  if (orderedTracks.length === 0) orderedTracks.push('V1', 'A1');

  const orderedRef = useRef(orderedTracks);
  orderedRef.current = orderedTracks;

  const itemById = useMemo(() => {
    const map: Record<string, TItem> = {};
    for (const i of items) map[i.id] = i;
    return map;
  }, [items]);

  /** Clips linked to this one (its own sound, or its picture). */
  const partnerIds = useCallback((id: string): string[] => {
    const it = itemById[id];
    if (!it?.link_id) return [];
    return items.filter(i => i.link_id === it.link_id && i.id !== id).map(i => i.id);
  }, [items, itemById]);

  const withPartners = useCallback((ids: string[]): string[] => {
    const out: string[] = [];
    for (const id of ids) {
      for (const x of [id, ...partnerIds(id)]) if (!out.includes(x)) out.push(x);
    }
    return out;
  }, [partnerIds]);

  const trackFlags = useCallback((track: string): TrackState =>
    tl?.tracks?.[track] ?? { hidden: false, locked: false, muted: false },
  [tl]);

  const isEditable = useCallback((item: TItem) =>
    item.origin !== 'auto' && !item.locked && !trackFlags(item.track).locked,
  [trackFlags]);

  /** A lane a moving clip may land on: same kind, not captions, not locked,
   *  and not a transcript-built spine. */
  const laneAccepts = useCallback((target: string, fromTrack: string) => {
    if (trackKind(target) !== trackKind(fromTrack) || target.toUpperCase() === 'CAP') return false;
    if (trackFlags(target).locked) return false;
    return !items.some(i => i.track === target && i.origin === 'auto');
  }, [items, trackFlags]);
  const laneAcceptsRef = useRef(laneAccepts);
  laneAcceptsRef.current = laneAccepts;

  /** Selection as clips: stale ids dropped, linked partners included. */
  const selection = useMemo(() => selectedClipIds.filter(id => itemById[id]), [selectedClipIds, itemById]);
  const selectedItems = useMemo(
    () => withPartners(selection).map(id => itemById[id]).filter((i): i is TItem => !!i),
    [selection, withPartners, itemById],
  );
  const primaryItem = selectedClipId ? itemById[selectedClipId] : undefined;

  const under = (i: TItem, f: number) => i.timeline_start_frame < f && f < i.timeline_end_frame;

  /** Frames of source a media clip can reach, or null when it has no limit. */
  const sourceFrames = (i: TItem): number | null => {
    if ((i.kind ?? 'media') !== 'media') return null;
    const src = tl?.sources?.[i.source_id];
    if (!src || src.kind === 'image' || !src.duration_seconds) return null;
    return Math.round(src.duration_seconds * fps);
  };

  const noticeTimer = useRef<number | null>(null);
  const flash = useCallback((msg: string) => {
    setNotice(msg);
    if (noticeTimer.current) window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setNotice(null), 2600);
  }, []);

  const refreshTimeline = useCallback((newTimeline: any) => {
    if (newTimeline) updateProject({ timeline: newTimeline });
  }, [updateProject]);

  // Edits run one after another: two quick keypresses must not race each other
  // to the server and land in the wrong order.
  const opQueueRef = useRef<Promise<unknown>>(Promise.resolve());
  const runOp = useCallback(<T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    const next = opQueueRef.current.then(async () => {
      setBusy(true);
      try {
        const res: any = await fn();
        if (res?.timeline) refreshTimeline(res.timeline);
        return res as T;
      } catch (err: any) {
        setError(cleanError(err, 'Timeline operation failed'));
        return undefined;
      } finally {
        setBusy(false);
      }
    });
    opQueueRef.current = next;
    return next;
  }, [refreshTimeline, setError]);

  /** Run several calls as one edit; the last response carries the timeline. */
  const runEach = useCallback(<T,>(list: T[], call: (x: T) => Promise<any>) =>
    runOp(async () => {
      let res: any;
      for (const x of list) res = await call(x);
      return res;
    }), [runOp]);

  // Fetch audio waveforms for any source used on an audio lane.
  const waveReqRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!projectId) return;
    const audioSources = Array.from(new Set(
      items.filter(i => trackKind(i.track) === 'A').map(i => i.source_id)
    ));
    audioSources.forEach(sid => {
      if (waveforms[sid] || waveReqRef.current.has(sid)) return;
      waveReqRef.current.add(sid);
      getWaveform(projectId, sid)
        .then(w => setWaveforms(prev => ({ ...prev, [sid]: w })))
        .catch(() => { waveReqRef.current.delete(sid); });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, items]);

  // Fetch a filmstrip for every source shown on a video lane. Sequential on
  // purpose: each one is an ffmpeg pass over the whole source, and firing a
  // dozen at once on a timeline full of B-roll would starve the render queue for
  // a picture nobody is waiting on. They fill in as they arrive.
  const stripReqRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!projectId) return;
    const videoSources = Array.from(new Set(
      items
        .filter(i => trackKind(i.track) === 'V' && (i.kind ?? 'media') === 'media')
        .map(i => i.source_id)
        .filter(sid => (tl?.sources?.[sid]?.kind ?? 'video') !== 'audio')
    ));
    let cancelled = false;
    (async () => {
      for (const sid of videoSources) {
        if (cancelled) return;
        if (stripReqRef.current.has(sid)) continue;
        stripReqRef.current.add(sid);
        // The first request can land while the backend is still coming up, and
        // one silent failure would otherwise leave that clip a blank block for
        // the rest of the session — nothing re-runs this unless the timeline
        // itself changes.
        let strip = null;
        for (let attempt = 0; attempt < 3 && !cancelled; attempt++) {
          try {
            strip = await getFilmstrip(projectId, sid);
            break;
          } catch {
            await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
          }
        }
        // Commit before checking `cancelled`. This effect re-runs whenever the
        // timeline object is replaced, and dropping an answered request on the
        // way out left that source marked as already-fetched with nothing
        // stored — one clip stuck as a blank block for the rest of the session.
        if (strip) {
          // columns 0 means "no picture in this source" — remember the null so
          // we stop asking, and let the clip keep its plain block.
          setFilmstrips(prev => ({ ...prev, [sid]: strip!.columns > 0 ? strip : null }));
        } else {
          stripReqRef.current.delete(sid);   // a later pass may still get it
        }
        if (cancelled) return;
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, items]);

  // --- Selection -------------------------------------------------------------
  // Clicking a clip selects it with its linked partner (the video and its own
  // sound are one clip); Alt picks the one half alone. Ctrl toggles, Shift
  // extends from the last clicked clip, dragging on empty lane draws a box.

  const selectGroup = useCallback((item: TItem, alone: boolean) => {
    setSelection(alone ? [item.id] : [item.id, ...partnerIds(item.id)], item.id);
    anchorRef.current = item.id;
  }, [partnerIds, setSelection]);

  const toggleClip = useCallback((item: TItem, alone: boolean) => {
    const group = alone ? [item.id] : [item.id, ...partnerIds(item.id)];
    const has = selectedClipIds.includes(item.id);
    const next = has
      ? selectedClipIds.filter(id => !group.includes(id))
      : [...selectedClipIds, ...group];
    setSelection(next, has ? undefined : item.id);
    anchorRef.current = item.id;
  }, [partnerIds, selectedClipIds, setSelection]);

  const rangeSelect = useCallback((item: TItem) => {
    const anchor = anchorRef.current ? itemById[anchorRef.current] : undefined;
    if (!anchor) { selectGroup(item, false); return; }
    const lo = Math.min(anchor.timeline_start_frame, item.timeline_start_frame);
    const hi = Math.max(anchor.timeline_end_frame, item.timeline_end_frame);
    const lanes = orderedRef.current;
    const ia = lanes.indexOf(anchor.track);
    const ib = lanes.indexOf(item.track);
    const span = new Set(lanes.slice(Math.min(ia, ib), Math.max(ia, ib) + 1));
    const hits = items
      .filter(i => span.has(i.track) && i.timeline_start_frame < hi && i.timeline_end_frame > lo)
      .map(i => i.id);
    setSelection(withPartners(hits), item.id);
  }, [itemById, items, selectGroup, setSelection, withPartners]);

  const buildSnapCandidates = (exclude: Set<string>): number[] => {
    const c = new Set<number>([0, playFrame]);
    if (markIn != null) c.add(Math.round(markIn * fps));
    if (markOut != null) c.add(Math.round(markOut * fps));
    for (const i of items) {
      if (exclude.has(i.id)) continue;
      c.add(i.timeline_start_frame);
      c.add(i.timeline_end_frame);
    }
    return [...c];
  };

  // --- Drag: move / trim -----------------------------------------------------

  const startDrag = (e: React.MouseEvent, item: TItem, mode: DragMode) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    e.preventDefault();
    try { window.getSelection()?.removeAllRanges(); } catch { /* ignore */ }
    if (mode === 'move' && (e.ctrlKey || e.metaKey)) { toggleClip(item, e.altKey); return; }
    if (mode === 'move' && e.shiftKey) { rangeSelect(item); return; }

    const group = e.altKey ? [item.id] : [item.id, ...partnerIds(item.id)];
    let selected = selection;
    let clickSelect: string[] | null = null;
    if (!selected.includes(item.id)) {
      selected = group;
      setSelection(group, item.id);
    } else {
      // Already selected: keep the selection so the whole lot can be dragged,
      // and narrow it to this clip only if the press turns out to be a click.
      clickSelect = group;
      setSelection(selected, item.id);
    }
    anchorRef.current = item.id;

    const editable = isEditable(item);
    const orig: DragState['orig'] = {};
    const record = (i: TItem) => {
      orig[i.id] = { start: i.timeline_start_frame, end: i.timeline_end_frame, track: i.track };
    };
    let ids: string[];
    let minDelta = -Infinity;
    let maxDelta = Infinity;
    let laneChange = false;

    if (mode === 'move') {
      if (!editable) {
        // AI-managed or locked: it can be selected, not dragged.
        if (clickSelect) setSelection(clickSelect, item.id);
        return;
      }
      const pool = e.altKey ? selected : withPartners(selected);
      const moving = pool.map(id => itemById[id]).filter((i): i is TItem => !!i && isEditable(i));
      ids = moving.map(i => i.id);
      moving.forEach(record);
      minDelta = -Math.min(...moving.map(i => i.timeline_start_frame));
      laneChange = moving.filter(i => trackKind(i.track) === trackKind(item.track)).length === 1;
    } else {
      // Auto clips can't be moved, but their edges can still ripple-cut the
      // programme, so a trim (not a move) is allowed through.
      if (!editable && !isAutoCut(item)) return;
      const edgeOf = (i: TItem) => (mode === 'trim-start' ? i.timeline_start_frame : i.timeline_end_frame);
      const members = [item, ...(editable && !e.altKey
        ? partnerIds(item.id).map(id => itemById[id])
          .filter((p): p is TItem => !!p && edgeOf(p) === edgeOf(item) && isEditable(p))
        : [])];
      ids = members.map(i => i.id);
      members.forEach(record);
      for (const m of members) {
        const srcFrames = sourceFrames(m);
        if (mode === 'trim-start') {
          maxDelta = Math.min(maxDelta, m.timeline_end_frame - 1 - m.timeline_start_frame);
          // An auto (ripple-cut) clip has no slack before its start: it only shrinks.
          minDelta = Math.max(minDelta, editable ? -m.timeline_start_frame : 0);
          if (editable && srcFrames != null) minDelta = Math.max(minDelta, -m.source_start_frame);
        } else {
          minDelta = Math.max(minDelta, m.timeline_start_frame + 1 - m.timeline_end_frame);
          if (!editable) maxDelta = Math.min(maxDelta, 0);
          if (editable && srcFrames != null) maxDelta = Math.min(maxDelta, srcFrames - m.source_end_frame);
        }
      }
    }
    snapCandidatesRef.current = buildSnapCandidates(new Set(ids));
    setDrag({
      mode, id: item.id, ids, orig, startX: e.clientX, startY: e.clientY, delta: 0,
      previewTrack: item.track, laneChange, minDelta, maxDelta, editable,
      unlinked: e.altKey, moved: false, clickSelect, snapFrame: null,
    });
  };

  useEffect(() => {
    if (!drag) return;

    const onMove = (e: MouseEvent) => {
      setDrag(prev => {
        if (!prev) return prev;
        const dx = e.clientX - prev.startX;
        const dy = e.clientY - prev.startY;
        if (!prev.moved && Math.abs(dx) < DRAG_THRESHOLD_PX && Math.abs(dy) < DRAG_THRESHOLD_PX) return prev;
        const g = prev.orig[prev.id];
        const edges = prev.mode === 'move' ? [g.start, g.end] : prev.mode === 'trim-start' ? [g.start] : [g.end];
        let delta = pxToFrame(dx);
        let snapFrame: number | null = null;
        if (snapRef.current) {
          const threshold = Math.max(1, pxToFrame(SNAP_PX));
          let best: { dist: number; frame: number; delta: number } | null = null;
          for (const edge of edges) {
            for (const c of snapCandidatesRef.current) {
              const dist = Math.abs(edge + delta - c);
              if (dist <= threshold && (!best || dist < best.dist)) best = { dist, frame: c, delta: c - edge };
            }
          }
          if (best) { delta = best.delta; snapFrame = best.frame; }
        }
        delta = Math.min(prev.maxDelta, Math.max(prev.minDelta, delta));
        if (snapFrame != null && !edges.some(edge => edge + delta === snapFrame)) snapFrame = null;

        let previewTrack = prev.previewTrack;
        if (prev.mode === 'move' && prev.laneChange) {
          const rect = lanesRef.current?.getBoundingClientRect();
          if (rect) {
            const target = orderedRef.current[Math.floor((e.clientY - rect.top) / LANE_HEIGHT)];
            if (target && laneAcceptsRef.current(target, g.track)) previewTrack = target;
          }
        }
        return { ...prev, moved: true, delta, snapFrame, previewTrack };
      });
    };

    const onUp = () => {
      const d = dragRef.current;
      setDrag(null);
      if (!d || !projectId) return;
      if (!d.moved) {
        if (d.clickSelect) setSelection(d.clickSelect, d.id);
        return;
      }
      const o = d.orig[d.id];
      if (d.mode === 'move') {
        const laneChanged = d.previewTrack !== o.track;
        if (d.delta !== 0 || laneChanged) {
          runOp(() => moveClips(projectId, d.ids, d.delta,
            laneChanged ? { [d.id]: d.previewTrack } : undefined, !d.unlinked));
        }
        return;
      }
      if (d.delta === 0) return;
      if (d.editable) {
        const edge = d.mode === 'trim-start' ? 'start' : 'end';
        const frame = edge === 'start' ? o.start + d.delta : o.end + d.delta;
        runOp(() => trimClip(projectId, d.id, edge, frame, false, !d.unlinked));
      } else if (d.mode === 'trim-start') {
        // Trimming an AI-managed clip's edge doesn't move slack in and out —
        // it ripple-cuts the programme range that was dragged away.
        runOp(() => cutProgramRange(projectId, o.start, o.start + d.delta));
      } else {
        runOp(() => cutProgramRange(projectId, o.end + d.delta, o.end));
      }
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drag?.id, drag?.mode, projectId]);

  // --- Box (marquee) selection on empty lane area ---------------------------

  const laneAtY = (clientY: number): string | null => {
    const rect = lanesRef.current?.getBoundingClientRect();
    if (!rect) return null;
    return orderedRef.current[Math.floor((clientY - rect.top) / LANE_HEIGHT)] ?? null;
  };

  const handleLanesMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const rect = lanesRef.current?.getBoundingClientRect();
    if (!rect) return;
    e.preventDefault();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMarquee({
      x0: x, y0: y, x1: x, y1: y, additive: e.ctrlKey || e.metaKey || e.shiftKey,
      base: selection, moved: false, track: laneAtY(e.clientY),
    });
  };

  useEffect(() => {
    if (!marquee) return;
    const onMove = (e: MouseEvent) => {
      const rect = lanesRef.current?.getBoundingClientRect();
      const m = marqueeRef.current;
      if (!rect || !m) return;
      const x1 = e.clientX - rect.left;
      const y1 = e.clientY - rect.top;
      const moved = m.moved || Math.abs(x1 - m.x0) > DRAG_THRESHOLD_PX || Math.abs(y1 - m.y0) > DRAG_THRESHOLD_PX;
      setMarquee({ ...m, x1, y1, moved });
      if (!moved) return;
      const left = Math.min(m.x0, x1);
      const right = Math.max(m.x0, x1);
      const top = Math.min(m.y0, y1);
      const bottom = Math.max(m.y0, y1);
      const lanes = orderedRef.current.filter((_, idx) => idx * LANE_HEIGHT < bottom && (idx + 1) * LANE_HEIGHT > top);
      const hits = items
        .filter(i => lanes.includes(i.track)
          && frameToPx(i.timeline_start_frame) < right && frameToPx(i.timeline_end_frame) > left)
        .map(i => i.id);
      const ids = withPartners(hits);
      setSelection(m.additive ? [...m.base, ...ids] : ids);
    };
    const onUp = () => {
      const m = marqueeRef.current;
      setMarquee(null);
      if (m && !m.moved) {
        // A plain click on empty lane: clear the selection, pick the layer.
        if (!m.additive) setSelection([]);
        if (m.track) setSelectedTrack(m.track);
      }
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [marquee != null]);

  // --- Adding things ---------------------------------------------------------

  /** Select what an add/paste just placed, so the next keypress acts on it. */
  const selectPlaced = (res: any) => {
    if (res?.item_ids?.length) setSelection(res.item_ids);
  };

  const handleAddMedia = async (track?: string, frame?: number) => {
    if (!projectId) return;
    const filePath = await pickFile([
      { name: 'Media', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'aac', 'm4a', 'flac', 'jpg', 'jpeg', 'png', 'webp'] },
    ]);
    if (!filePath) return;
    // The backend puts the first clip of a timeline at 0:00, keeps a video's
    // sound linked beside it, and never stacks a clip on top of another.
    const res = await runOp(() => addMedia(projectId, filePath, { track, timelineStartFrame: frame ?? playFrame }));
    if (track) setLocalEmptyTracks(prev => prev.filter(t => t !== track));
    selectPlaced(res);
  };

  const handleAddText = async (track?: string, frame?: number) => {
    if (!projectId) return;
    const res = await runOp(() => addTextClip(projectId, {
      content: 'New text',
      track,
      timelineStartFrame: frame ?? playFrame,
      durationSeconds: 3,
      preset: 'title_bold',
    }));
    if (track) setLocalEmptyTracks(prev => prev.filter(t => t !== track));
    const added = (res?.timeline?.items ?? []).filter((i: TItem) => i.kind === 'text');
    if (added.length) setSelection([added[added.length - 1].id]);
  };

  const handleAddAdjustment = async (track?: string, frame?: number) => {
    if (!projectId) return;
    const res = await runOp(() => addAdjustment(projectId, {
      track,
      timelineStartFrame: frame ?? playFrame,
      durationSeconds: 5,
    }));
    if (track) setLocalEmptyTracks(prev => prev.filter(t => t !== track));
    // Select it straight away: an adjustment layer does nothing until it is
    // given a grade, and the Inspector is where that happens.
    const added = (res?.timeline?.items ?? []).filter((i: TItem) => i.kind === 'adjustment');
    if (added.length) setSelection([added[added.length - 1].id]);
  };

  const toggleTrack = (track: string, flag: 'hidden' | 'locked' | 'muted') => {
    if (!projectId) return;
    const current = trackFlags(track);
    runOp(() => setTrackFlags(projectId, track, { [flag]: !current[flag] }));
  };

  const handleDeleteTrack = async (track: string) => {
    if (!projectId) return;
    const count = items.filter(i => i.track === track).length;
    if (count > 0 && !window.confirm(
      `Delete ${track} and its ${count} clip${count === 1 ? '' : 's'}?`)) return;
    await runOp(() => deleteTrack(projectId, track));
    setLocalEmptyTracks(prev => prev.filter(t => t !== track));
    if (selectedTrack === track) setSelectedTrack(null);
  };

  const handleAddTrack = (kind: 'V' | 'A' | 'T') => {
    // Spawn a distinct empty lane immediately. Tracks are implicit on the backend
    // (created when the first clip lands), so we just reserve the next name in the UI.
    const existing = orderedRef.current.filter(t => trackKind(t) === kind);
    const maxNum = existing.reduce((m, t) => Math.max(m, trackNum(t)), 0);
    const name = `${kind}${maxNum + 1}`;
    setLocalEmptyTracks(prev => (prev.includes(name) ? prev : [...prev, name]));
  };

  // Drop a media-pool item onto a track at the drop position.
  const handleLaneDrop = async (e: React.DragEvent, track: string) => {
    e.preventDefault();
    const raw = e.dataTransfer.getData(MEDIA_DND_TYPE);
    if (!raw || !projectId) return;
    let entry: any;
    try { entry = JSON.parse(raw); } catch { return; }
    // A video dropped on an audio lane (or vice versa) still goes in: the
    // backend picks a lane of the right kind instead of the drop being refused.
    const wantKind = entry.kind === 'audio' ? 'A' : 'V';
    const target = trackKind(track) === wantKind && track.toUpperCase() !== 'CAP' ? track : undefined;
    const rect = lanesRef.current?.getBoundingClientRect();
    const startFrame = Math.max(0, pxToFrame(rect ? e.clientX - rect.left : 0));
    const res = await runOp(() => addMedia(projectId, entry.path, { track: target, timelineStartFrame: startFrame }));
    if (target) setLocalEmptyTracks(prev => prev.filter(t => t !== target));
    selectPlaced(res);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom + delta)));
  };

  // --- Clipboard -------------------------------------------------------------

  const handleCopy = (): boolean => {
    if (!selectedItems.length) return false;
    setClipboard(JSON.parse(JSON.stringify(selectedItems)));
    flash(`Copied ${selectedItems.length} clip${selectedItems.length === 1 ? '' : 's'}`);
    return true;
  };

  const handlePaste = async (insert: boolean, atFrame?: number) => {
    const clips = useClipboard.getState().items;
    if (!clips?.length) { flash('Nothing to paste — copy a clip first (Ctrl+C)'); return; }
    if (!projectId) return;
    const res = await runOp(() => pasteClips(projectId, clips, atFrame ?? playFrame, insert));
    selectPlaced(res);
  };

  const handleDuplicate = async () => {
    if (!projectId || !selectedItems.length) return;
    const end = Math.max(...selectedItems.map(i => i.timeline_end_frame));
    const res = await runOp(() => pasteClips(projectId, JSON.parse(JSON.stringify(selectedItems)), end));
    selectPlaced(res);
  };

  // --- Delete / ripple delete ------------------------------------------------

  const handleDelete = (ripple: boolean) => {
    if (!projectId || !selectedItems.length) return;
    const manual = selectedItems.filter(isEditable);
    // AI-managed spine clips go by ripple-cutting their span out of the
    // programme — latest first, so each cut leaves the earlier spans in place.
    const spans = mergeSpans(selectedItems.filter(isAutoCut)
      .map(i => [i.timeline_start_frame, i.timeline_end_frame] as [number, number]))
      .sort((a, b) => b[0] - a[0]);
    if (!manual.length && !spans.length) {
      flash('Locked clips can\'t be deleted — unlock them first');
      return;
    }
    runOp(async () => {
      let res: any;
      if (manual.length) res = await deleteClips(projectId, manual.map(i => i.id), ripple);
      for (const [s, e] of spans) res = await cutProgramRange(projectId, s, e);
      return res;
    });
    setSelection([]);
  };

  const handleCut = () => {
    if (handleCopy()) handleDelete(false);
  };

  // --- Split / trim to playhead ----------------------------------------------

  const handleSplit = () => {
    if (!projectId) return;
    const f = playFrame;
    // With nothing selected, split whatever the playhead crosses on every lane.
    const pool = selection.length ? selectedItems : items;
    const targets = pool.filter(i => under(i, f) && isEditable(i));
    if (!targets.length) {
      flash(pool.some(i => under(i, f) && isAutoCut(i))
        ? 'AI-managed clips are cut with Delete, Q / W, or from the transcript'
        : 'Put the playhead inside a clip to split it');
      return;
    }
    runOp(() => splitClips(projectId, targets.map(i => i.id), f));
  };

  const handleTrim = (edge: 'start' | 'end', ripple: boolean) => {
    if (!projectId) return;
    const f = playFrame;
    // Nothing selected: the programme clip under the playhead.
    const pool = selection.length ? selectedItems : items.filter(i => i.track === 'V1');
    const hits = pool.filter(i => under(i, f));
    const manual = hits.filter(isEditable);
    if (manual.length) {
      runOp(() => trimClips(projectId, manual.map(i => i.id), edge, f, ripple));
      return;
    }
    const auto = hits.find(i => isAutoCut(i) && i.track === 'V1') ?? hits.find(isAutoCut);
    if (auto) {
      // The AI spine always ripples: trimming it cuts that stretch of programme.
      const [s, e] = edge === 'start' ? [auto.timeline_start_frame, f] : [f, auto.timeline_end_frame];
      runOp(() => cutProgramRange(projectId, s, e));
      return;
    }
    flash('Put the playhead inside the selected clip to trim it');
  };

  const handleCloseGap = (track?: string, frame?: number) => {
    if (!projectId) return;
    const lane = track ?? selectedTrack ?? primaryItem?.track;
    if (!lane) { flash('Click a track first, then close the gap under the playhead'); return; }
    runOp(() => closeGap(projectId, lane, frame ?? playFrame));
  };

  const handleSelectForward = (frame?: number, track?: string) => {
    const from = frame ?? playFrame;
    const ids = items
      .filter(i => i.timeline_start_frame >= from && (!track || i.track === track))
      .map(i => i.id);
    setSelection(withPartners(ids));
    if (!ids.length) flash('No clips after this point');
  };

  // --- Link / detach / compound ----------------------------------------------

  const isLinked = (i: TItem) => !!i.link_id && partnerIds(i.id).length > 0;
  const audioPartner = (i: TItem) => partnerIds(i.id).map(id => itemById[id]).find(p => p && trackKind(p.track) === 'A');

  const handleLinkToggle = () => {
    if (!projectId) return;
    if (selectedItems.some(isLinked)) {
      runOp(() => unlinkClips(projectId, selectedItems.map(i => i.id)));
      flash('Unlinked — each clip now edits on its own');
    } else if (selectedItems.filter(isEditable).length >= 2) {
      runOp(() => linkClips(projectId, selectedItems.filter(isEditable).map(i => i.id)));
      flash('Linked — they now move, trim and delete together');
    } else {
      flash('Select two or more clips to link them');
    }
  };

  // Detaching needs an editable picture clip whose source actually carries sound.
  // Imported video already arrives with its sound on its own (linked) lane, so
  // there "detach" means breaking that link.
  const canDetach = (i?: TItem) => !!i
    && (i.kind ?? 'media') === 'media'
    && trackKind(i.track) === 'V'
    && isEditable(i)
    && tl?.sources?.[i.source_id]?.kind !== 'image'
    && (!!audioPartner(i) || !i.mute);

  const handleDetachAudio = () => {
    if (!projectId) return;
    const item = primaryItem && trackKind(primaryItem.track) === 'V'
      ? primaryItem : selectedItems.find(i => trackKind(i.track) === 'V');
    if (!canDetach(item)) { flash('Select a video clip that has sound'); return; }
    if (audioPartner(item!)) {
      runOp(() => unlinkClips(projectId, [item!.id]));
      flash('Audio detached — it now edits on its own');
    } else {
      runOp(() => detachAudio(projectId, item!.id));
    }
  };

  const handleCompound = () => {
    if (!projectId) return;
    const ids = selectedItems.filter(i => isEditable(i) && i.kind !== 'compound').map(i => i.id);
    if (ids.length < 2) { flash('Select two or more clips to make a compound clip'); return; }
    runOp(() => compoundClips(projectId, ids));
    setSelection([]);
  };

  const handleUncompound = () => {
    if (!projectId) return;
    const target = selectedItems.find(i => i.kind === 'compound' && isEditable(i));
    if (!target) { flash('Select a compound clip to break apart'); return; }
    runOp(() => uncompoundClip(projectId, target.id));
    setSelection([]);
  };

  // --- Clip switches ---------------------------------------------------------

  const handleToggleEnabled = () => {
    if (!projectId || !selectedItems.length) return;
    const enabled = !selectedItems.some(i => i.enabled);
    runEach(selectedItems, i => setClipFlags(projectId, i.id, { enabled }));
  };

  const handleToggleMute = () => {
    if (!projectId) return;
    const audio = selectedItems.filter(i => trackKind(i.track) === 'A');
    if (!audio.length) { flash('No audio in the selection'); return; }
    const mute = !audio.every(i => i.mute);
    runEach(audio, i => setClipFlags(projectId, i.id, { mute }));
  };

  const handleToggleLock = () => {
    if (!projectId || !selectedItems.length) return;
    const locked = !selectedItems.every(i => i.locked);
    runEach(selectedItems, i => setClipFlags(projectId, i.id, { locked }));
  };

  const handleRename = () => {
    const item = primaryItem ?? selectedItems[0];
    if (!item) return;
    setRename({ ids: withPartners([item.id]), value: item.label || '' });
  };

  const submitRename = () => {
    if (!rename || !projectId) return;
    const label = rename.value.trim();
    setRename(null);
    if (!label) return;
    runEach(rename.ids, id => setClipFlags(projectId, id, { label }));
  };

  // --- Transform (rotate / flip) ---------------------------------------------

  const visualItems = () => selectedItems.filter(i =>
    trackKind(i.track) === 'V' && i.track.toUpperCase() !== 'CAP' && !i.locked);

  const handleRotate = (deg: number) => {
    if (!projectId) return;
    const targets = visualItems();
    if (!targets.length) { flash('Select a video or image clip to rotate'); return; }
    runEach(targets, i => {
      const current = Number((i.transform as any)?.rotation ?? 0);
      return setTransform(projectId, i.id, { rotation: (((current + deg) % 360) + 360) % 360 } as any);
    });
  };

  const handleFlip = (axis: 'flip_h' | 'flip_v') => {
    if (!projectId) return;
    const targets = visualItems();
    if (!targets.length) { flash('Select a video or image clip to flip'); return; }
    runEach(targets, i => setTransform(projectId, i.id, { [axis]: !(i.transform as any)?.[axis] } as any));
  };

  const handleResetTransform = () => {
    if (!projectId) return;
    runEach(visualItems(), i => setTransform(projectId, i.id, {}, true));
  };

  const handleNudge = (frames: number) => {
    if (!projectId) return;
    const ids = selectedItems.filter(isEditable).map(i => i.id);
    if (ids.length) runOp(() => moveClips(projectId, ids, frames));
  };

  const toggleSnap = () => {
    setSnapState(prev => {
      saveSnap(!prev);
      return !prev;
    });
  };

  // --- Playhead --------------------------------------------------------------

  const editPoints = useMemo(() => Array.from(new Set([0, ...items.flatMap(i =>
    [i.timeline_start_frame, i.timeline_end_frame])])).sort((a, b) => a - b), [items]);

  const stepFrames = (n: number) => setCurrentTime(Math.max(0, (playFrame + n) / fps));
  const goToEdit = (dir: 1 | -1) => {
    const target = dir > 0
      ? editPoints.find(p => p > playFrame)
      : [...editPoints].reverse().find(p => p < playFrame);
    if (target != null) setCurrentTime(target / fps);
  };

  const zoomFit = () => {
    const el = containerRef.current;
    if (!el) return;
    const seconds = Math.max(1, maxEndFrame / fps);
    setZoom(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, (el.clientWidth - 24) / (seconds * 50))));
  };

  // --- Mark in/out and cutting the auto-edited spine ---
  const handleMarkIn = () => setMarkIn(currentTime);
  const handleMarkOut = () => setMarkOut(currentTime);

  const handleCutRange = async () => {
    if (!projectId || markIn == null || markOut == null) return;
    const startFrame = Math.round(Math.min(markIn, markOut) * fps);
    const endFrame = Math.round(Math.max(markIn, markOut) * fps);
    if (endFrame <= startFrame) return;
    const res = await runOp(() => cutProgramRange(projectId, startFrame, endFrame));
    if (res) { setMarkIn(null); setMarkOut(null); }
  };

  const handleRestoreCuts = () => {
    if (!projectId) return;
    runOp(() => restoreProgramCuts(projectId));
  };

  // --- Command registration ---------------------------------------------------
  // One registration for the timeline's lifetime; each call reaches the
  // handlers of the latest render through the ref, so they always see the
  // current selection and playhead.
  const handlersRef = useRef<Partial<Record<TimelineCommandId, () => void>>>({});
  handlersRef.current = {
    'edit.cut': handleCut,
    'edit.copy': () => { if (!handleCopy()) flash('Select a clip to copy'); },
    'edit.paste': () => handlePaste(false),
    'edit.pasteInsert': () => handlePaste(true),
    'edit.duplicate': handleDuplicate,
    'edit.deleteClip': () => handleDelete(false),
    'edit.rippleDelete': () => handleDelete(true),
    'edit.selectAll': () => setSelection(items.map(i => i.id)),
    'edit.deselectAll': () => setSelection([]),
    'edit.splitAtPlayhead': handleSplit,
    'edit.markIn': handleMarkIn,
    'edit.markOut': handleMarkOut,
    'timeline.trimStart': () => handleTrim('start', false),
    'timeline.trimEnd': () => handleTrim('end', false),
    'timeline.rippleTrimStart': () => handleTrim('start', true),
    'timeline.rippleTrimEnd': () => handleTrim('end', true),
    'timeline.closeGap': () => handleCloseGap(),
    'timeline.selectForward': () => handleSelectForward(),
    'timeline.link': handleLinkToggle,
    'timeline.detachAudio': handleDetachAudio,
    'timeline.compound': handleCompound,
    'timeline.uncompound': handleUncompound,
    'timeline.toggleEnabled': handleToggleEnabled,
    'timeline.toggleMute': handleToggleMute,
    'timeline.toggleLock': handleToggleLock,
    'timeline.rename': handleRename,
    'timeline.rotateCW': () => handleRotate(90),
    'timeline.rotateCCW': () => handleRotate(-90),
    'timeline.flipH': () => handleFlip('flip_h'),
    'timeline.flipV': () => handleFlip('flip_v'),
    'timeline.resetTransform': handleResetTransform,
    'timeline.nudgeLeft': () => handleNudge(-1),
    'timeline.nudgeRight': () => handleNudge(1),
    'timeline.toggleSnap': toggleSnap,
    'playback.prevFrame': () => stepFrames(-1),
    'playback.nextFrame': () => stepFrames(1),
    'playback.back1s': () => stepFrames(-Math.round(fps)),
    'playback.fwd1s': () => stepFrames(Math.round(fps)),
    'playback.prevEdit': () => goToEdit(-1),
    'playback.nextEdit': () => goToEdit(1),
    'playback.start': () => setCurrentTime(0),
    'playback.end': () => setCurrentTime(maxEndFrame / fps),
    'view.zoomIn': () => setZoom(Math.min(MAX_ZOOM, zoom + 0.2)),
    'view.zoomOut': () => setZoom(Math.max(MIN_ZOOM, zoom - 0.2)),
    'view.zoomFit': zoomFit,
  };
  const registerCmd = useCommandStore((s) => s.register);
  const unregisterCmd = useCommandStore((s) => s.unregister);
  useEffect(() => {
    for (const id of TIMELINE_COMMAND_IDS) registerCmd(id, () => handlersRef.current[id]?.());
    return () => { for (const id of TIMELINE_COMMAND_IDS) unregisterCmd(id); };
  }, [registerCmd, unregisterCmd]);

  const scrubToClientX = useCallback((clientX: number) => {
    const rect = lanesRef.current?.getBoundingClientRect();
    if (!rect) return;
    setCurrentTime(Math.max(0, (clientX - rect.left) / pixelsPerSecond));
  }, [pixelsPerSecond, setCurrentTime]);

  const handleScrubStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setScrubbing(true);
    setStoreScrubbing(true);
    scrubToClientX(e.clientX);
  }, [scrubToClientX, setStoreScrubbing]);

  // Continuous scrub while dragging on the ruler / playhead strip.
  useEffect(() => {
    if (!scrubbing) return;
    const onMove = (e: MouseEvent) => scrubToClientX(e.clientX);
    const onUp = () => { setScrubbing(false); setStoreScrubbing(false); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [scrubbing, scrubToClientX, setStoreScrubbing]);

  // A wheel/scrollbar scroll the user made themselves; don't fight it with
  // auto-follow for a little while. Scrolling we did ourselves (below) sets
  // `programmaticScrollRef` first so it isn't mistaken for one.
  const handleContainerScroll = useCallback(() => {
    if (programmaticScrollRef.current) { programmaticScrollRef.current = false; return; }
    lastManualScrollAtRef.current = Date.now();
  }, []);

  // Keep the playhead in view whenever it moves out of it — playback, a scrub
  // dragged past the edge, or a jump from the transcript/keyboard. A playhead
  // that sits still while the user scrolls away does not re-run this effect.
  useEffect(() => {
    if (followMode === 'off') return;
    const el = containerRef.current;
    if (!el) return;
    const manualRecently = !isPlaying && (Date.now() - lastManualScrollAtRef.current < 1500);
    if (manualRecently) return;

    const playheadX = currentTime * pixelsPerSecond;
    const viewWidth = el.clientWidth;
    const scrollLeft = el.scrollLeft;
    const withinView = playheadX >= scrollLeft && playheadX <= scrollLeft + viewWidth;
    const continuousCenter = followMode === 'center' && isPlaying;
    if (!continuousCenter && withinView) return;

    const frac = followMode === 'left' ? 0.1 : followMode === 'right' ? 0.9 : 0.5;
    const target = Math.max(0, playheadX - viewWidth * frac);
    programmaticScrollRef.current = true;
    el.scrollLeft = target;
  }, [currentTime, isPlaying, scrubbing, followMode, pixelsPerSecond]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const frames = Math.floor((seconds % 1) * fps);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
  };

  const getTimeMarkers = () => {
    const markers = [];
    const interval = zoom < 0.5 ? 30 : zoom < 1 ? 10 : zoom < 2 ? 5 : 1;
    for (let t = 0; t <= durationSec; t += interval) markers.push(t);
    return markers;
  };

  // --- Right-click menus -------------------------------------------------------

  const chordOf = (id: string) => chordLabel(useCommandStore.getState().chordFor(id));
  /** A menu row for a command: its label, its live chord, and what it runs. */
  const cmdEntry = (id: string, enabled: boolean, extra: Partial<MenuEntry> = {}): MenuEntry => ({
    label: COMMAND_LABELS[id],
    chord: chordOf(id),
    disabled: !enabled,
    onClick: () => useCommandStore.getState().run(id),
    ...extra,
  });
  const sep: MenuEntry = { separator: true };

  const openClipMenu = (e: React.MouseEvent, item: TItem) => {
    e.preventDefault();
    e.stopPropagation();
    // Right-clicking outside the selection acts on that clip instead.
    let ids = selection;
    if (!ids.includes(item.id)) {
      ids = [item.id, ...partnerIds(item.id)];
      setSelection(ids, item.id);
      anchorRef.current = item.id;
    }
    const all = withPartners(ids).map(id => itemById[id]).filter((i): i is TItem => !!i);
    const editable = all.filter(isEditable);
    const underHead = all.filter(i => under(i, playFrame));
    const canDelete = editable.length > 0 || all.some(isAutoCut);
    const canTrim = underHead.some(i => isEditable(i) || isAutoCut(i));
    const linked = all.some(isLinked);
    const compound = all.find(i => i.kind === 'compound' && isEditable(i));
    const visual = all.filter(i => trackKind(i.track) === 'V' && i.track.toUpperCase() !== 'CAP' && !i.locked);
    const audio = all.filter(i => trackKind(i.track) === 'A');
    const videoItem = all.find(i => trackKind(i.track) === 'V');
    const hasClipboard = !!clipboardItems?.length;
    const allDisabled = all.every(i => !i.enabled);
    const allLocked = all.every(i => i.locked);
    const allMuted = audio.length > 0 && audio.every(i => i.mute);
    const why = isAutoCut(item) ? 'AI-managed clip — edit it from the transcript' : 'Locked';

    setMenu({ x: e.clientX, y: e.clientY, items: [
      cmdEntry('edit.cut', canDelete),
      cmdEntry('edit.copy', all.length > 0),
      cmdEntry('edit.paste', hasClipboard),
      cmdEntry('edit.pasteInsert', hasClipboard),
      cmdEntry('edit.duplicate', all.length > 0),
      sep,
      cmdEntry('edit.deleteClip', canDelete, { danger: true }),
      cmdEntry('edit.rippleDelete', canDelete, { danger: true }),
      sep,
      cmdEntry('edit.splitAtPlayhead', underHead.some(isEditable),
        { hint: underHead.length ? why : 'Move the playhead inside the clip first' }),
      cmdEntry('timeline.trimStart', canTrim, { hint: 'Move the playhead inside the clip first' }),
      cmdEntry('timeline.trimEnd', canTrim, { hint: 'Move the playhead inside the clip first' }),
      cmdEntry('timeline.rippleTrimStart', canTrim, { hint: 'Move the playhead inside the clip first' }),
      cmdEntry('timeline.rippleTrimEnd', canTrim, { hint: 'Move the playhead inside the clip first' }),
      sep,
      cmdEntry('timeline.link', linked || editable.length >= 2, {
        label: linked ? 'Unlink Audio & Video' : 'Link Clips',
        hint: 'Select two or more clips to link them',
      }),
      cmdEntry('timeline.detachAudio', canDetach(videoItem), { hint: 'Needs a video clip with sound' }),
      compound
        ? cmdEntry('timeline.uncompound', true)
        : cmdEntry('timeline.compound', editable.length >= 2, { hint: 'Select two or more clips' }),
      sep,
      {
        label: 'Rotate & Flip', disabled: !visual.length, hint: 'Only video and image clips',
        submenu: [
          cmdEntry('timeline.rotateCW', true),
          cmdEntry('timeline.rotateCCW', true),
          cmdEntry('timeline.flipH', true),
          cmdEntry('timeline.flipV', true),
          sep,
          cmdEntry('timeline.resetTransform', true),
        ],
      },
      cmdEntry('timeline.toggleMute', audio.length > 0, {
        label: allMuted ? 'Unmute Audio' : 'Mute Audio', hint: 'No audio in the selection',
      }),
      cmdEntry('timeline.toggleEnabled', all.length > 0, { label: allDisabled ? 'Enable Clip' : 'Disable Clip' }),
      cmdEntry('timeline.toggleLock', all.length > 0, { label: allLocked ? 'Unlock Clip' : 'Lock Clip' }),
      cmdEntry('timeline.rename', true, { label: 'Rename…' }),
      sep,
      { label: 'Select Clips After This', onClick: () => handleSelectForward(item.timeline_start_frame) },
      {
        label: 'Select All on This Track',
        onClick: () => setSelection(withPartners(items.filter(i => i.track === item.track).map(i => i.id))),
      },
      sep,
      cmdEntry('edit.undo', canUndo),
      cmdEntry('edit.redo', canRedo),
    ] });
  };

  const openLaneMenu = (e: React.MouseEvent, track: string) => {
    e.preventDefault();
    const rect = lanesRef.current?.getBoundingClientRect();
    const frame = rect ? Math.max(0, pxToFrame(e.clientX - rect.left)) : playFrame;
    const onTrack = items.filter(i => i.track === track);
    const inGap = !onTrack.some(i => i.timeline_start_frame <= frame && frame < i.timeline_end_frame)
      && onTrack.some(i => i.timeline_start_frame > frame);
    const kind = trackKind(track);
    const flags = trackFlags(track);
    const isCap = track.toUpperCase() === 'CAP';
    const hasClipboard = !!clipboardItems?.length;
    setSelectedTrack(track);

    setMenu({ x: e.clientX, y: e.clientY, items: [
      { label: 'Paste Here', chord: chordOf('edit.paste'), disabled: !hasClipboard,
        hint: 'Copy a clip first', onClick: () => handlePaste(false, frame) },
      { label: 'Paste Insert Here', chord: chordOf('edit.pasteInsert'), disabled: !hasClipboard,
        hint: 'Copy a clip first', onClick: () => handlePaste(true, frame) },
      sep,
      { label: 'Delete Gap (Ripple)', disabled: !inGap || flags.locked,
        hint: 'Right-click an empty stretch that has a clip after it', onClick: () => handleCloseGap(track, frame) },
      { label: 'Select All Clips After Here', onClick: () => handleSelectForward(frame) },
      { label: 'Select All Clips on This Track', disabled: !onTrack.length,
        onClick: () => setSelection(withPartners(onTrack.map(i => i.id))) },
      sep,
      kind === 'T'
        ? { label: 'Add Text Here', disabled: flags.locked, onClick: () => handleAddText(track, frame) }
        : { label: `Add Media to ${track}…`, disabled: flags.locked || isCap, onClick: () => handleAddMedia(track, frame) },
      ...(kind !== 'T' ? [{ label: 'Add Text Here', onClick: () => handleAddText(undefined, frame) }] : []),
      ...(kind === 'V' && !isCap && track !== 'V1'
        ? [{ label: 'Add Adjustment Layer Here', disabled: flags.locked, onClick: () => handleAddAdjustment(track, frame) }]
        : []),
      sep,
      { label: 'Add Video Track', onClick: () => handleAddTrack('V') },
      { label: 'Add Audio Track', onClick: () => handleAddTrack('A') },
      { label: 'Add Text Track', onClick: () => handleAddTrack('T') },
      sep,
      cmdEntry('edit.undo', canUndo),
      cmdEntry('edit.redo', canRedo),
      sep,
      cmdEntry('timeline.toggleSnap', true, { checked: snap }),
      cmdEntry('view.zoomFit', true),
    ] });
  };

  const openTrackMenu = (e: React.MouseEvent, track: string) => {
    e.preventDefault();
    const flags = trackFlags(track);
    const kind = trackKind(track);
    const isCap = track.toUpperCase() === 'CAP';
    setSelectedTrack(track);
    setMenu({ x: e.clientX, y: e.clientY, items: [
      { label: flags.hidden ? `Show ${track}` : `Hide ${track}`, onClick: () => toggleTrack(track, 'hidden') },
      ...(kind === 'A' ? [{ label: flags.muted ? 'Unmute Track' : 'Mute Track', onClick: () => toggleTrack(track, 'muted') }] : []),
      { label: flags.locked ? 'Unlock Track' : 'Lock Track', onClick: () => toggleTrack(track, 'locked') },
      sep,
      { label: 'Select All Clips on Track',
        onClick: () => setSelection(withPartners(items.filter(i => i.track === track).map(i => i.id))) },
      kind === 'T'
        ? { label: 'Add Text at Playhead', disabled: flags.locked, onClick: () => handleAddText(track) }
        : { label: 'Add Media at Playhead…', disabled: flags.locked || isCap, onClick: () => handleAddMedia(track) },
      sep,
      { label: `Delete ${isCap ? 'Captions' : track}`, danger: true, onClick: () => handleDeleteTrack(track) },
    ] });
  };

  if (!project) return null;

  const canSplitNow = (selection.length ? selectedItems : items).some(i => under(i, playFrame) && isEditable(i));
  const canDeleteNow = selectedItems.some(i => isEditable(i) || isAutoCut(i));
  const detachTarget = primaryItem && trackKind(primaryItem.track) === 'V'
    ? primaryItem : selectedItems.find(i => trackKind(i.track) === 'V');
  const linkedNow = selectedItems.some(isLinked);

  /** Where a clip is drawn: its preview while being dragged, else its own span. */
  const spanOf = (item: TItem): [number, number] => {
    const o = drag?.moved ? drag.orig[item.id] : undefined;
    if (!drag || !o) return [item.timeline_start_frame, item.timeline_end_frame];
    if (drag.mode === 'move') return [o.start + drag.delta, o.end + drag.delta];
    if (drag.mode === 'trim-start') return [o.start + drag.delta, o.end];
    return [o.start, o.end + drag.delta];
  };
  const laneOf = (item: TItem) =>
    (drag?.moved && drag.mode === 'move' && item.id === drag.id ? drag.previewTrack : item.track);

  return (
    <div className="timeline-container multitrack">
      <div className="timeline-header" style={{ gap: 8, flexWrap: 'wrap' }}>
        <div className="timeline-timecode">{formatTime(currentTime)}</div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddMedia()} title="Import media — the first clip starts the timeline; a video's sound comes in linked on an audio track">＋ Media</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddText()} title="Add a text clip at the playhead">＋ Text</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddAdjustment()}
            title="Add an adjustment layer — grades and effects on it apply to every layer below">＋ Adjust</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddTrack('V')} title="Add video track">＋ V</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddTrack('A')} title="Add audio track">＋ A</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddTrack('T')} title="Add text track">＋ T</button>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button className="btn btn-sm" disabled={!canUndo} onClick={() => undo()}
            title={`Undo (${chordOf('edit.undo')})`}>↶</button>
          <button className="btn btn-sm" disabled={!canRedo} onClick={() => redo()}
            title={`Redo (${chordOf('edit.redo')})`}>↷</button>
          <button className="btn btn-sm" disabled={busy || !canSplitNow} onClick={handleSplit}
            title={`Split at the playhead (${chordOf('edit.splitAtPlayhead')})`}>✂ Split</button>
          <button className="btn btn-sm" disabled={busy || !canDetach(detachTarget)} onClick={handleDetachAudio}
            title={`Separate this clip's audio so it edits on its own (${chordOf('timeline.detachAudio')})`}>⇵ Detach Audio</button>
          <button className="btn btn-sm" disabled={busy || (!linkedNow && selectedItems.filter(isEditable).length < 2)}
            onClick={handleLinkToggle}
            title={`${linkedNow ? 'Unlink' : 'Link'} — linked clips move, trim, split and delete together (${chordOf('timeline.link')})`}>
            {linkedNow ? '⛓ Unlink' : '🔗 Link'}
          </button>
          {selectedItems.some(i => i.kind === 'compound') ? (
            <button className="btn btn-sm" disabled={busy} onClick={handleUncompound}
              title="Break this group back into separate clips">▣ Uncompound</button>
          ) : (
            <button className="btn btn-sm" disabled={busy || selectedItems.filter(isEditable).length < 2} onClick={handleCompound}
              title="Ctrl+click or drag a box to select several clips, then group them into one block">▣ Compound</button>
          )}
          <button className="btn btn-sm btn-danger" disabled={busy || !canDeleteNow}
            onClick={() => handleDelete(false)}
            title={primaryItem && isAutoCut(primaryItem)
              ? 'Ripple-cut this clip\'s whole span out of the programme' : `Delete (${chordOf('edit.deleteClip')})`}>
            🗑 Delete
          </button>
          <button className="btn btn-sm btn-danger" disabled={busy || !canDeleteNow}
            onClick={() => handleDelete(true)}
            title={`Delete and close the gap it leaves (${chordOf('edit.rippleDelete')})`}>
            ⇤ Ripple Delete
          </button>
          <button className={`btn btn-sm ${snap ? 'btn-toggle-on' : ''}`} onClick={toggleSnap}
            title={`Snapping ${snap ? 'on' : 'off'} — clips snap to edges and the playhead (${chordOf('timeline.toggleSnap')})`}>
            🧲
          </button>
        </div>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <button className="btn btn-sm" disabled={busy} onClick={handleMarkIn} title="Mark In (I)">⏵| In</button>
          <button className="btn btn-sm" disabled={busy} onClick={handleMarkOut} title="Mark Out (O)">|⏴ Out</button>
          <button className="btn btn-sm" disabled={busy || markIn == null || markOut == null}
            onClick={handleCutRange} title="Ripple-cut the marked range out of the programme">✂ Cut range</button>
          {!!tl?.manual_cuts?.length && (
            <button className="btn btn-sm" disabled={busy} onClick={handleRestoreCuts}
              title="Undo every ripple cut made on the auto-edited program">↺ Restore cuts</button>
          )}
        </div>
        {/* What is dressed on the *programme* — not on any one clip, so there is
            nowhere else on the lanes for it to show. Without this, a rain effect
            and a dissolve on every cut were invisible until the export came back
            looking unexpectedly different. */}
        {(programmeEffects.length > 0 || programmeTransition) && (
          <div className="programme-fx" title="Applied to the whole programme — drawn when you render">
            {programmeEffects.length > 0 && (
              <span className="programme-fx-chip"
                title={programmeEffects.map((e: any) =>
                  `${e.type.replace(/_/g, ' ')}${e.enabled === false ? ' (off)' : ''}`).join(', ')}>
                ✦ {programmeEffects.length} effect{programmeEffects.length === 1 ? '' : 's'}
              </span>
            )}
            {programmeTransition && (
              <span className="programme-fx-chip"
                title={`Every cut without its own transition uses ${prettyTransition(programmeTransition.type)}`}>
                ⋈ {prettyTransition(programmeTransition.type)} at cuts
              </span>
            )}
          </div>
        )}
        {notice && <span className="timeline-notice">{notice}</span>}
        <div className="timeline-follow" title="Keep the playhead in view while playing or scrubbing">
          <label className="text-xs text-muted">Follow:</label>
          <select value={followMode} onChange={(e) => setFollowMode(e.target.value as FollowMode)}>
            <option value="off">Off</option>
            <option value="left">Left</option>
            <option value="center">Center</option>
            <option value="right">Right</option>
          </select>
        </div>
        <div className="timeline-zoom">
          <button className="btn btn-sm" onClick={() => setZoom(Math.max(MIN_ZOOM, zoom - 0.2))}>-</button>
          <span className="text-xs text-muted">{Math.round(zoom * 100)}%</span>
          <button className="btn btn-sm" onClick={() => setZoom(Math.min(MAX_ZOOM, zoom + 0.2))}>+</button>
          <button className="btn btn-sm" onClick={zoomFit} title={`Zoom to fit (${chordOf('view.zoomFit')})`}>⤢</button>
        </div>
        <div className="timeline-duration">{formatTime(durationSec)}</div>
      </div>

      {/* Lanes render even before a timeline exists: a new project has to be able
          to accept media dragged in from the library, and the backend creates the
          timeline on the first drop. */}
      {!tl && (
        <div className="timeline-hint">
          Drag media here from the <strong>Media</strong> tab to start building, or run
          {' '}<strong>Transcribe</strong> to cut the program from your speech.
        </div>
      )}
      <div className="multitrack-body" style={{ display: 'flex', flex: 1, minHeight: 0, overflowX: 'hidden', overflowY: 'auto' }}>
        {/* Track headers */}
        <div className="track-headers">
          <div style={{ height: 22 }} />
          {orderedTracks.map(track => {
            const state = trackFlags(track);
            const selected = selectedTrack === track;
            return (
              <div key={track}
                className={`track-header ${selected ? 'selected' : ''} ${state.hidden ? 'hidden-track' : ''}`}
                style={{ height: LANE_HEIGHT }}
                onClick={() => setSelectedTrack(selected ? null : track)}
                onContextMenu={(e) => openTrackMenu(e, track)}
                title={track.toUpperCase() === 'CAP' ? 'Captions — click to select the layer, right-click for options' : `${track} — click to select the layer, right-click for options`}>
                <span className="track-name">{track.toUpperCase() === 'CAP' ? 'Captions' : track}</span>
                <div className="track-switches">
                  <button className={`track-sw ${state.hidden ? 'on' : ''}`}
                    title={state.hidden ? `${track} is hidden — click to show` : `Hide ${track}`}
                    onClick={(e) => { e.stopPropagation(); toggleTrack(track, 'hidden'); }}>
                    {state.hidden ? '🚫' : '👁'}
                  </button>
                  {trackKind(track) === 'A' && (
                    <button className={`track-sw ${state.muted ? 'on' : ''}`}
                      title={state.muted ? `${track} is muted — click to unmute` : `Mute ${track}`}
                      onClick={(e) => { e.stopPropagation(); toggleTrack(track, 'muted'); }}>
                      {state.muted ? '🔇' : '🔊'}
                    </button>
                  )}
                  <button className={`track-sw ${state.locked ? 'on' : ''}`}
                    title={state.locked ? `${track} is locked — click to unlock` : `Lock ${track}`}
                    onClick={(e) => { e.stopPropagation(); toggleTrack(track, 'locked'); }}>
                    {state.locked ? '🔒' : '🔓'}
                  </button>
                  {track.toUpperCase() !== 'CAP' && (
                    <button className="track-sw"
                      title={trackKind(track) === 'T' ? `Add text to ${track}` : `Add media to ${track}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        trackKind(track) === 'T' ? handleAddText(track) : handleAddMedia(track);
                      }}>＋</button>
                  )}
                  <button className="track-sw danger"
                    title={`Delete ${track} and everything on it`}
                    onClick={(e) => { e.stopPropagation(); handleDeleteTrack(track); }}>🗑</button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Scrollable lanes */}
        <div ref={containerRef} className="timeline-scroll" style={{ flex: 1, overflowX: 'auto', overflowY: 'hidden' }}
          onWheel={handleWheel} onScroll={handleContainerScroll}>
          <div style={{ width: `${trackWidth}px`, position: 'relative' }}>
            {/* Ruler (click or drag to scrub) */}
            <div className="timeline-ruler" style={{ height: 22, position: 'relative', cursor: 'ew-resize', background: 'var(--surface-2,#161616)', userSelect: 'none' }} onMouseDown={handleScrubStart}>
              {getTimeMarkers().map(t => (
                <div key={t} className="timeline-marker" style={{ left: `${t * pixelsPerSecond}px`, position: 'absolute' }}>
                  <span className="marker-label">{formatTime(t)}</span>
                </div>
              ))}
            </div>

            {/* Lanes — press on empty space and drag to box-select */}
            <div ref={lanesRef} style={{ position: 'relative' }} onMouseDown={handleLanesMouseDown}>
              {orderedTracks.map(track => {
                const laneState = trackFlags(track);
                const laneSelected = selectedTrack === track;
                return (
                  <div key={track}
                    className={`timeline-lane ${laneSelected ? 'selected' : ''} ${laneState.hidden ? 'hidden-track' : ''} ${laneState.locked ? 'locked-track' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
                    onDrop={(e) => handleLaneDrop(e, track)}
                    onContextMenu={(e) => openLaneMenu(e, track)}
                    style={{
                      height: LANE_HEIGHT, position: 'relative',
                      borderBottom: '1px solid var(--border,#222)',
                      // background-color, not the shorthand: the shorthand would wipe
                      // out the diagonal hatching a locked lane gets from the stylesheet.
                      backgroundColor: laneSelected ? 'rgba(59,130,246,0.10)' : LANE_TINT[trackKind(track)],
                    }}>
                    {items.filter(i => laneOf(i) === track).map(item => {
                      const [startF, endF] = spanOf(item);
                      const editable = isEditable(item);
                      const autoTrim = isAutoCut(item);
                      const canTrim = editable || autoTrim;
                      const laneKind = trackKind(track);
                      const kind = tl?.sources?.[item.source_id]?.kind || (laneKind === 'A' ? 'audio' : 'video');
                      const selected = selection.includes(item.id);
                      const primary = item.id === selectedClipId;
                      const linked = isLinked(item);
                      const chips = dressingChips(item);
                      const width = Math.max(2, frameToPx(endF - startF));
                      return (
                        <div key={item.id}
                          className={`timeline-clip ${selected ? 'selected' : ''} ${editable ? '' : 'locked'} ${drag?.moved && drag.ids.includes(item.id) ? 'dragging' : ''}`}
                          onMouseDown={(e) => startDrag(e, item, 'move')}
                          onContextMenu={(e) => openClipMenu(e, item)}
                          title={editable
                            ? `Drag to move${linked ? ' (with its linked audio/video — Alt-drag moves it alone)' : ''}, edges to trim · Ctrl+click / Shift+click to select several · right-click for more`
                            : autoTrim
                              ? 'AI-managed clip — drag an edge or press Delete to ripple-cut the programme (edit words in the transcript for word-level control) — right-click for more'
                              : 'Locked clip — right-click to unlock; styling still works in the Inspector'}
                          style={{
                            position: 'absolute', left: `${frameToPx(startF)}px`, width: `${width}px`,
                            top: 3, height: LANE_HEIGHT - 8, borderRadius: 4, boxSizing: 'border-box',
                            background: clipFill(item, laneKind, editable),
                            border: primary ? '2px solid #fff'
                              : selected ? '2px solid var(--accent,#3b82f6)' : '1px solid rgba(0,0,0,0.4)',
                            color: '#fff', fontSize: 11, overflow: 'hidden', whiteSpace: 'nowrap',
                            cursor: editable ? 'grab' : 'not-allowed', opacity: item.enabled ? 1 : 0.4,
                            display: 'flex', alignItems: 'center', padding: '0 6px', userSelect: 'none',
                          }}>
                          {laneKind === 'A' && waveforms[item.source_id] && (
                            <ClipWaveform
                              wave={waveforms[item.source_id]}
                              sourceStartSec={item.source_start_frame / fps}
                              sourceEndSec={item.source_end_frame / fps}
                              width={width}
                              height={LANE_HEIGHT - 8}
                            />
                          )}
                          {laneKind === 'V' && projectId && filmstrips[item.source_id] && (
                            <ClipFilmstrip
                              strip={filmstrips[item.source_id]!}
                              projectId={projectId}
                              sourceId={item.source_id}
                              sourceStartSec={item.source_start_frame / fps}
                              sourceEndSec={item.source_end_frame / fps}
                              width={width}
                            />
                          )}
                          {/* The transition into this clip, drawn at its true
                              length so a 2s dissolve looks like one. Sits under
                              the trim handle and takes no pointer events, so it
                              cannot get in the way of an edit. */}
                          {junctions[item.id] && (
                            <span className={`clip-transition ${junctions[item.id].own ? 'own' : ''}`}
                              style={{ width: Math.max(6, frameToPx(junctions[item.id].duration * fps)) }}
                              title={`${prettyTransition(junctions[item.id].type)} · ${junctions[item.id].duration.toFixed(2)}s`
                                + (junctions[item.id].own ? ' (set on this clip)' : ' (programme default)')} />
                          )}
                          {canTrim && (
                            <div onMouseDown={(e) => startDrag(e, item, 'trim-start')}
                              style={{ position: 'absolute', left: 0, top: 0, width: HANDLE_PX, height: '100%', cursor: 'ew-resize', background: 'rgba(255,255,255,0.25)', zIndex: 2 }} />
                          )}
                          {/* The label sits over the frames now, so it needs its own contrast. */}
                          <span style={{ pointerEvents: 'none', paddingLeft: canTrim ? HANDLE_PX : 0, position: 'relative', zIndex: 1, textOverflow: 'ellipsis', overflow: 'hidden', textShadow: '0 1px 2px rgba(0,0,0,0.85)' }}>
                            {linked && <span className="clip-link" title="Linked">🔗</span>}
                            {clipLabel(item, kind, laneKind)}
                          </span>
                          {/* Chips for dressing that is otherwise invisible on the lane. */}
                          {(chips.length > 0 || item.mute) && (
                            <span className="clip-chips" style={{ position: 'absolute', top: 1, right: canTrim ? HANDLE_PX + 2 : 2, zIndex: 3, pointerEvents: 'none' }}>
                              {chips.map(c => (
                                <span key={c.key} className="clip-chip" title={c.title}>{c.label}</span>
                              ))}
                              {item.mute && <span className="clip-chip" title={laneKind === 'A' ? 'muted' : 'audio detached'}>🔇</span>}
                            </span>
                          )}
                          {canTrim && (
                            <div onMouseDown={(e) => startDrag(e, item, 'trim-end')}
                              style={{ position: 'absolute', right: 0, top: 0, width: HANDLE_PX, height: '100%', cursor: 'ew-resize', background: 'rgba(255,255,255,0.25)' }} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}

              {/* Snap guide — where the dragged edge has locked on. */}
              {drag?.moved && drag.snapFrame != null && (
                <div className="timeline-snap-line" style={{ left: `${frameToPx(drag.snapFrame)}px` }} />
              )}

              {marquee?.moved && (
                <div className="timeline-marquee" style={{
                  left: Math.min(marquee.x0, marquee.x1), top: Math.min(marquee.y0, marquee.y1),
                  width: Math.abs(marquee.x1 - marquee.x0), height: Math.abs(marquee.y1 - marquee.y0),
                }} />
              )}

              {/* Playhead — glide smoothly between coarse timeupdate steps
                  during playback; follow the cursor instantly while scrubbing. */}
              <div className="timeline-playhead"
                style={{
                  position: 'absolute', top: 0, bottom: 0,
                  left: `${currentTime * pixelsPerSecond}px`, width: 2,
                  background: '#ef4444', pointerEvents: 'none',
                  transition: isPlaying && !scrubbing ? 'left 0.25s linear' : 'none',
                }} />
            </div>

            {/* The marked in/out range, translucent, spanning the ruler and lanes.
                Painted last so it tints them rather than being hidden under the
                ruler's own opaque background. */}
            {markIn != null && markOut != null && (
              <div className="mark-range-band" style={{
                position: 'absolute', top: 0, bottom: 0, zIndex: 5, pointerEvents: 'none',
                left: `${Math.min(markIn, markOut) * pixelsPerSecond}px`,
                width: `${Math.max(1, Math.abs(markOut - markIn) * pixelsPerSecond)}px`,
              }} />
            )}
          </div>
        </div>
      </div>

      {menu && <ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => setMenu(null)} />}

      {rename && (
        <div className="timeline-dialog-backdrop" onMouseDown={() => setRename(null)}>
          <div className="timeline-dialog" onMouseDown={(e) => e.stopPropagation()}>
            <div className="timeline-dialog-title">Rename clip</div>
            <input autoFocus value={rename.value}
              onChange={(e) => setRename({ ...rename, value: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitRename();
                if (e.key === 'Escape') setRename(null);
              }} />
            <div className="timeline-dialog-actions">
              <button className="btn btn-sm" onClick={() => setRename(null)}>Cancel</button>
              <button className="btn btn-sm btn-primary" onClick={submitRename}>Rename</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
