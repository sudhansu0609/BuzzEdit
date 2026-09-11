import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { useProjectStore } from '../hooks/store';
import {
  useElectron, splitClip, deleteClip, moveClip, trimClip, addMedia, getWaveform,
  detachAudio, compoundClips, uncompoundClip, addTextClip, addTrack, addAdjustment,
  setTrackFlags, deleteTrack, TrackState, WaveformData,
  getFilmstrip, filmstripImageUrl, FilmstripData,
} from '../hooks/api';
import { MEDIA_DND_TYPE } from './MediaPool';
import { useCommand } from '../hooks/commands';
import { effectiveTransitions } from '../lib/programme';
import { prettyTransition } from './TransitionsPanel';

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
}
interface TTimeline {
  fps_num: number;
  fps_den: number;
  items: TItem[];
  sources: Record<string, { kind?: string; path?: string }>;
  duration_frames?: number;
  program_offset_frames?: number;
  tracks?: Record<string, TrackState>;
  extra_tracks?: string[];
}

type DragMode = 'move' | 'trim-start' | 'trim-end';
interface DragState {
  mode: DragMode;
  id: string;
  startX: number;
  origStart: number;
  origEnd: number;
  origTrack: string;
  previewStart: number;
  previewEnd: number;
  previewTrack: string;
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

export default function Timeline() {
  const containerRef = useRef<HTMLDivElement>(null);
  const lanesRef = useRef<HTMLDivElement>(null);
  const {
    project, updateProject, currentTime, setCurrentTime, selectedClipId, selectedClipIds,
    setSelectedClip, toggleSelectedClip, zoom, setZoom, setError, isPlaying,
    setScrubbing: setStoreScrubbing,
  } = useProjectStore();
  const { pickFile } = useElectron();

  const [drag, setDrag] = useState<DragState | null>(null);
  const dragRef = useRef<DragState | null>(null);
  dragRef.current = drag;
  const [localEmptyTracks, setLocalEmptyTracks] = useState<string[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [scrubbing, setScrubbing] = useState(false);
  const [waveforms, setWaveforms] = useState<Record<string, WaveformData>>({});
  const [filmstrips, setFilmstrips] = useState<Record<string, FilmstripData | null>>({});

  const tl = project?.timeline as TTimeline | undefined;
  const projectId = project?.id;
  const fps = tl && tl.fps_num ? tl.fps_num / (tl.fps_den || 1) : 30;
  const pixelsPerSecond = 50 * zoom;
  const frameToPx = (f: number) => (f / fps) * pixelsPerSecond;
  const pxToFrame = (px: number) => Math.round((px / pixelsPerSecond) * fps);

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
  const orderedTracks = [...textTracks, ...videoTracks, ...audioTracks];
  if (orderedTracks.length === 0) orderedTracks.push('V1', 'A1');

  const orderedRef = useRef(orderedTracks);
  orderedRef.current = orderedTracks;

  const refreshTimeline = useCallback((newTimeline: any) => {
    if (newTimeline) updateProject({ timeline: newTimeline });
  }, [updateProject]);

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

  const runOp = useCallback(async (fn: () => Promise<any>) => {
    setBusy(true);
    try {
      const res = await fn();
      refreshTimeline(res.timeline);
      return res;
    } catch (err: any) {
      setError(err.message || 'Timeline operation failed');
    } finally {
      setBusy(false);
    }
  }, [refreshTimeline, setError]);

  // --- Drag lifecycle (move / trim) via window listeners ---
  useEffect(() => {
    if (!drag) return;

    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - drag.startX;
      const deltaFrames = pxToFrame(dx);
      setDrag(prev => {
        if (!prev) return prev;
        if (prev.mode === 'move') {
          const dur = prev.origEnd - prev.origStart;
          const newStart = Math.max(0, prev.origStart + deltaFrames);
          // Determine target lane from cursor Y.
          let previewTrack = prev.origTrack;
          const rect = lanesRef.current?.getBoundingClientRect();
          if (rect) {
            const laneIndex = Math.floor((e.clientY - rect.top) / LANE_HEIGHT);
            const target = orderedRef.current[laneIndex];
            if (target && trackKind(target) === trackKind(prev.origTrack)) previewTrack = target;
          }
          return { ...prev, previewStart: newStart, previewEnd: newStart + dur, previewTrack };
        } else if (prev.mode === 'trim-start') {
          const newStart = Math.min(Math.max(0, prev.origStart + deltaFrames), prev.origEnd - 1);
          return { ...prev, previewStart: newStart };
        } else {
          const newEnd = Math.max(prev.origStart + 1, prev.origEnd + deltaFrames);
          return { ...prev, previewEnd: newEnd };
        }
      });
    };

    const onUp = () => {
      const d = dragRef.current;
      setDrag(null);
      if (!d || !projectId) return;
      if (d.mode === 'move') {
        const trackChanged = d.previewTrack !== d.origTrack;
        if (d.previewStart !== d.origStart || trackChanged) {
          runOp(() => moveClip(projectId, d.id, d.previewStart, trackChanged ? d.previewTrack : undefined));
        }
      } else if (d.mode === 'trim-start' && d.previewStart !== d.origStart) {
        runOp(() => trimClip(projectId, d.id, 'start', d.previewStart));
      } else if (d.mode === 'trim-end' && d.previewEnd !== d.origEnd) {
        runOp(() => trimClip(projectId, d.id, 'end', d.previewEnd));
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

  const isEditable = (item: TItem) => item.origin !== 'auto' && !item.locked;

  const startDrag = useCallback((e: React.MouseEvent, item: TItem, mode: DragMode) => {
    e.stopPropagation();
    setSelectedClip(item.id);
    if (!isEditable(item)) return; // auto/locked clips are selectable but not editable
    setDrag({
      mode, id: item.id, startX: e.clientX,
      origStart: item.timeline_start_frame, origEnd: item.timeline_end_frame, origTrack: item.track,
      previewStart: item.timeline_start_frame, previewEnd: item.timeline_end_frame, previewTrack: item.track,
    });
  }, [setSelectedClip]);

  // --- Toolbar actions ---
  const handleSplit = useCallback(() => {
    if (!projectId || !selectedClipId) return;
    const frame = Math.round(currentTime * fps);
    runOp(() => splitClip(projectId, selectedClipId, frame));
  }, [projectId, selectedClipId, currentTime, fps, runOp]);

  const handleDelete = useCallback(() => {
    if (!projectId || !selectedClipId) return;
    runOp(() => deleteClip(projectId, selectedClipId));
    setSelectedClip(null);
  }, [projectId, selectedClipId, runOp, setSelectedClip]);

  const handleAddMedia = useCallback(async (track?: string) => {
    if (!projectId) return;
    const filePath = await pickFile([
      { name: 'Media', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'aac', 'm4a', 'flac', 'jpg', 'jpeg', 'png', 'webp'] },
    ]);
    if (!filePath) return;
    const startFrame = Math.round(currentTime * fps);
    await runOp(() => addMedia(projectId, filePath, { track, timelineStartFrame: startFrame }));
    if (track) setLocalEmptyTracks(prev => prev.filter(t => t !== track));
  }, [projectId, pickFile, currentTime, fps, runOp]);

  const handleDetachAudio = useCallback(() => {
    if (!projectId || !selectedClipId) return;
    runOp(() => detachAudio(projectId, selectedClipId));
  }, [projectId, selectedClipId, runOp]);

  const handleCompound = useCallback(() => {
    if (!projectId || selectedClipIds.length < 2) return;
    runOp(() => compoundClips(projectId, selectedClipIds));
    setSelectedClip(null);
  }, [projectId, selectedClipIds, runOp, setSelectedClip]);

  const handleUncompound = useCallback(() => {
    if (!projectId || !selectedClipId) return;
    runOp(() => uncompoundClip(projectId, selectedClipId));
    setSelectedClip(null);
  }, [projectId, selectedClipId, runOp, setSelectedClip]);

  const handleAddText = useCallback(async (track?: string) => {
    if (!projectId) return;
    await runOp(() => addTextClip(projectId, {
      content: 'New text',
      track,
      timelineStartFrame: Math.round(currentTime * fps),
      durationSeconds: 3,
      preset: 'title_bold',
    }));
    if (track) setLocalEmptyTracks(prev => prev.filter(t => t !== track));
  }, [projectId, currentTime, fps, runOp]);

  const handleAddAdjustment = useCallback(async (track?: string) => {
    if (!projectId) return;
    const res = await runOp(() => addAdjustment(projectId, {
      track,
      timelineStartFrame: Math.round(currentTime * fps),
      durationSeconds: 5,
    }));
    if (track) setLocalEmptyTracks(prev => prev.filter(t => t !== track));
    // Select it straight away: an adjustment layer does nothing until it is
    // given a grade, and the Inspector is where that happens.
    const added = (res?.timeline?.items ?? []).filter((i: TItem) => i.kind === 'adjustment');
    if (added.length) setSelectedClip(added[added.length - 1].id);
  }, [projectId, currentTime, fps, runOp, setSelectedClip]);

  const trackFlags = useCallback((track: string): TrackState =>
    tl?.tracks?.[track] ?? { hidden: false, locked: false, muted: false },
  [tl]);

  const toggleTrack = useCallback((track: string, flag: 'hidden' | 'locked' | 'muted') => {
    if (!projectId) return;
    const current = trackFlags(track);
    runOp(() => setTrackFlags(projectId, track, { [flag]: !current[flag] }));
  }, [projectId, trackFlags, runOp]);

  const handleDeleteTrack = useCallback(async (track: string) => {
    if (!projectId) return;
    const count = items.filter(i => i.track === track).length;
    if (count > 0 && !window.confirm(
      `Delete ${track} and its ${count} clip${count === 1 ? '' : 's'}?\n\nThis cannot be undone.`)) return;
    await runOp(() => deleteTrack(projectId, track));
    setLocalEmptyTracks(prev => prev.filter(t => t !== track));
    if (selectedTrack === track) setSelectedTrack(null);
  }, [projectId, items, runOp, selectedTrack]);

  const handleAddTrack = useCallback((kind: 'V' | 'A' | 'T') => {
    // Spawn a distinct empty lane immediately. Tracks are implicit on the backend
    // (created when the first clip lands), so we just reserve the next name in the UI.
    const existing = orderedRef.current.filter(t => trackKind(t) === kind);
    const maxNum = existing.reduce((m, t) => Math.max(m, trackNum(t)), 0);
    const name = `${kind}${maxNum + 1}`;
    setLocalEmptyTracks(prev => (prev.includes(name) ? prev : [...prev, name]));
  }, []);

  // Drop a media-pool item onto a track at the drop position.
  const handleLaneDrop = useCallback(async (e: React.DragEvent, track: string) => {
    e.preventDefault();
    const raw = e.dataTransfer.getData(MEDIA_DND_TYPE);
    if (!raw || !projectId) return;
    let entry: any;
    try { entry = JSON.parse(raw); } catch { return; }

    const wantKind = entry.kind === 'audio' ? 'A' : 'V';
    if (trackKind(track) !== wantKind) {
      setError(`${entry.kind} media can only go on a ${wantKind} track.`);
      return;
    }
    const rect = lanesRef.current?.getBoundingClientRect();
    const x = rect ? e.clientX - rect.left : 0;
    const startFrame = Math.max(0, pxToFrame(x));
    await runOp(() => addMedia(projectId, entry.path, { track, timelineStartFrame: startFrame }));
    setLocalEmptyTracks(prev => prev.filter(t => t !== track));
  }, [projectId, pxToFrame, runOp, setError]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom + delta)));
  }, [zoom, setZoom]);

  // Timeline actions exposed to the menu bar and keyboard shortcuts. Delete used
  // to be a private window listener here; it now goes through the shared command
  // dispatcher (which owns the input-focus guard) so there's a single source of
  // truth for the binding and no double-fire.
  useCommand('edit.deleteClip', handleDelete);
  useCommand('edit.splitAtPlayhead', handleSplit);
  useCommand('view.zoomIn', useCallback(
    () => setZoom(Math.min(MAX_ZOOM, zoom + 0.2)), [zoom, setZoom]));
  useCommand('view.zoomOut', useCallback(
    () => setZoom(Math.max(MIN_ZOOM, zoom - 0.2)), [zoom, setZoom]));

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

  if (!project) return null;

  const selectedItem = items.find(i => i.id === selectedClipId);
  // Detaching needs an editable picture clip whose source actually carries sound
  // and has not been detached already.
  const canDetach = !!selectedItem
    && selectedItem.kind !== 'text' && selectedItem.kind !== 'compound'
    && trackKind(selectedItem.track) === 'V'
    && isEditable(selectedItem)
    && !selectedItem.mute
    && tl?.sources?.[selectedItem.source_id]?.kind !== 'image';

  return (
    <div className="timeline-container multitrack">
      <div className="timeline-header" style={{ gap: 8, flexWrap: 'wrap' }}>
        <div className="timeline-timecode">{formatTime(currentTime)}</div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddMedia()} title="Import media onto a new track">＋ Media</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddText()} title="Add a text clip at the playhead">＋ Text</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddAdjustment()}
            title="Add an adjustment layer — grades and effects on it apply to every layer below">＋ Adjust</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddTrack('V')} title="Add video track">＋ V</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddTrack('A')} title="Add audio track">＋ A</button>
          <button className="btn btn-sm" disabled={busy} onClick={() => handleAddTrack('T')} title="Add text track">＋ T</button>
          <button className="btn btn-sm" disabled={busy || !selectedItem || !isEditable(selectedItem)} onClick={handleSplit} title="Split selected clip at playhead">✂ Split</button>
          <button className="btn btn-sm" disabled={busy || !canDetach} onClick={handleDetachAudio}
            title="Move this clip's audio onto its own track so you can edit it separately">⇵ Detach Audio</button>
          {selectedItem?.kind === 'compound' ? (
            <button className="btn btn-sm" disabled={busy} onClick={handleUncompound}
              title="Break this group back into separate clips">▣ Uncompound</button>
          ) : (
            <button className="btn btn-sm" disabled={busy || selectedClipIds.length < 2} onClick={handleCompound}
              title="Ctrl+click clips to select several, then group them into one block">▣ Compound</button>
          )}
          <button className="btn btn-sm btn-danger" disabled={busy || !selectedItem || !isEditable(selectedItem)} onClick={handleDelete} title="Delete selected clip">🗑 Delete</button>
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
        <div className="timeline-zoom">
          <button className="btn btn-sm" onClick={() => setZoom(Math.max(MIN_ZOOM, zoom - 0.2))}>-</button>
          <span className="text-xs text-muted">{Math.round(zoom * 100)}%</span>
          <button className="btn btn-sm" onClick={() => setZoom(Math.min(MAX_ZOOM, zoom + 0.2))}>+</button>
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
      {(
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
                  title={`${track} — click to select the layer`}>
                  <span className="track-name">{track}</span>
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
                    <button className="track-sw"
                      title={trackKind(track) === 'T' ? `Add text to ${track}` : `Add media to ${track}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        trackKind(track) === 'T' ? handleAddText(track) : handleAddMedia(track);
                      }}>＋</button>
                    <button className="track-sw danger"
                      title={`Delete ${track} and everything on it`}
                      onClick={(e) => { e.stopPropagation(); handleDeleteTrack(track); }}>🗑</button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Scrollable lanes */}
          <div ref={containerRef} className="timeline-scroll" style={{ flex: 1, overflowX: 'auto', overflowY: 'hidden' }} onWheel={handleWheel}>
            <div style={{ width: `${trackWidth}px`, position: 'relative' }}>
              {/* Ruler (click or drag to scrub) */}
              <div className="timeline-ruler" style={{ height: 22, position: 'relative', cursor: 'ew-resize', background: 'var(--surface-2,#161616)', userSelect: 'none' }} onMouseDown={handleScrubStart}>
                {getTimeMarkers().map(t => (
                  <div key={t} className="timeline-marker" style={{ left: `${t * pixelsPerSecond}px`, position: 'absolute' }}>
                    <span className="marker-label">{formatTime(t)}</span>
                  </div>
                ))}
              </div>

              {/* Lanes */}
              <div ref={lanesRef} style={{ position: 'relative' }} onClick={() => setSelectedClip(null)}>
                {orderedTracks.map(track => {
                  const laneState = trackFlags(track);
                  const laneSelected = selectedTrack === track;
                  return (
                  <div key={track}
                    className={`timeline-lane ${laneSelected ? 'selected' : ''} ${laneState.hidden ? 'hidden-track' : ''} ${laneState.locked ? 'locked-track' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
                    onDrop={(e) => handleLaneDrop(e, track)}
                    onClick={() => setSelectedTrack(track)}
                    style={{
                      height: LANE_HEIGHT, position: 'relative',
                      borderBottom: '1px solid var(--border,#222)',
                      // background-color, not the shorthand: the shorthand would wipe
                      // out the diagonal hatching a locked lane gets from the stylesheet.
                      backgroundColor: laneSelected ? 'rgba(59,130,246,0.10)' : LANE_TINT[trackKind(track)],
                    }}>
                    {items.filter(i => {
                      // While moving, render the dragged clip in its preview lane only.
                      if (drag?.id === i.id && drag.mode === 'move') return drag.previewTrack === track;
                      return i.track === track;
                    }).map(item => {
                      const isDragging = drag?.id === item.id;
                      const startF = isDragging ? drag!.previewStart : item.timeline_start_frame;
                      const endF = isDragging ? drag!.previewEnd : item.timeline_end_frame;
                      const editable = isEditable(item);
                      const laneKind = trackKind(track);
                      const kind = tl?.sources?.[item.source_id]?.kind || (laneKind === 'A' ? 'audio' : 'video');
                      const selected = selectedClipIds.includes(item.id);
                      const primary = item.id === selectedClipId;
                      const styled = !!item.transform || !!item.color || !!item.atmosphere?.length;
                      return (
                        <div key={item.id}
                          className={`timeline-clip ${selected ? 'selected' : ''} ${editable ? '' : 'locked'}`}
                          onMouseDown={(e) => startDrag(e, item, 'move')}
                          onClick={(e) => {
                            e.stopPropagation();
                            // Ctrl/Cmd extends the selection, which is how you pick
                            // the several clips that Compound needs.
                            if (e.ctrlKey || e.metaKey) toggleSelectedClip(item.id);
                            else setSelectedClip(item.id);
                          }}
                          title={editable
                            ? 'Drag to move, edges to trim · Ctrl+click to multi-select'
                            : 'AI-managed clip (edit words in the transcript) — styling still works in the Inspector'}
                          style={{
                            position: 'absolute', left: `${frameToPx(startF)}px`, width: `${Math.max(2, frameToPx(endF - startF))}px`,
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
                              width={Math.max(2, frameToPx(endF - startF))}
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
                              width={Math.max(2, frameToPx(endF - startF))}
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
                          {editable && (
                            <div onMouseDown={(e) => startDrag(e, item, 'trim-start')}
                              style={{ position: 'absolute', left: 0, top: 0, width: HANDLE_PX, height: '100%', cursor: 'ew-resize', background: 'rgba(255,255,255,0.25)', zIndex: 2 }} />
                          )}
                          {/* The label sits over the frames now, so it needs its own contrast. */}
                          <span style={{ pointerEvents: 'none', paddingLeft: editable ? HANDLE_PX : 0, position: 'relative', zIndex: 1, textOverflow: 'ellipsis', overflow: 'hidden', textShadow: '0 1px 2px rgba(0,0,0,0.85)' }}>
                            {clipLabel(item, kind, laneKind)}
                          </span>
                          {/* Badges for state that is otherwise invisible on the lane. */}
                          {(styled || item.mute) && (
                            <span style={{ position: 'absolute', top: 1, right: editable ? HANDLE_PX + 2 : 2, fontSize: 9, opacity: 0.85, zIndex: 3, pointerEvents: 'none' }}
                              title={[
                                styled ? `dressed: ${[
                                  item.transform ? 'transform' : '',
                                  item.color ? 'colour' : '',
                                  item.atmosphere?.length ? `${item.atmosphere.length} effect${item.atmosphere.length === 1 ? '' : 's'}` : '',
                                ].filter(Boolean).join(', ')} — applied at render` : '',
                                item.mute ? 'audio detached' : '',
                              ].filter(Boolean).join(' · ')}>
                              {styled ? '✦' : ''}{item.mute ? '🔇' : ''}
                            </span>
                          )}
                          {editable && (
                            <div onMouseDown={(e) => startDrag(e, item, 'trim-end')}
                              style={{ position: 'absolute', right: 0, top: 0, width: HANDLE_PX, height: '100%', cursor: 'ew-resize', background: 'rgba(255,255,255,0.25)' }} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                  );
                })}

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
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
