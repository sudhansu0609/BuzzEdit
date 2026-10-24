import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { useCommand } from '../hooks/commands';
import {
  useElectron, listMedia, importMedia, removeMedia, renameMedia, relinkMedia,
  getMediaWaveform, mediaThumbUrl, mediaStreamUrl, addMedia, getProject,
  startEyeContact, getEyeContactStatus, cancelEyeContact, revertEyeContact, saveEyeContactSetup,
  syncAngle, apiErrorDetail,
  startStabilize, getStabilizeStatus, cancelStabilize, revertStabilize, saveStabilizeSetup,
  MediaEntry, MEDIA_FILTERS, WaveformData, EyeContactJob, EyeContactSetup,
  StabilizeJob, StabilizeSetup, StabilizeReport,
} from '../hooks/api';
import ContextMenu, { MenuEntry } from './ContextMenu';
import EyeContactDialog, { eyeLensSummary } from './EyeContactDialog';
import StabilizeDialog, { stabilizeSummary } from './StabilizeDialog';
import SyncAudioDialog, { fmtDrift, fmtOffset } from './SyncAudioDialog';
import { mediaPoolChanged, openVoicePanel } from '../hooks/voiceApi';

export const MEDIA_DND_TYPE = 'application/x-buzzedit-media';

type KindFilter = 'all' | 'video' | 'audio' | 'image';

/** The backend now tags B-roll/music/cards it added to the library itself;
 *  api.ts's MediaEntry hasn't caught up yet, so extend it locally. */
type LibraryEntry = MediaEntry & { generated?: boolean };

const KIND_ICON: Record<string, string> = { audio: '🎵', image: '🖼', video: '🎞' };

const fmtDuration = (seconds: number) => {
  if (!seconds || seconds < 0) return '';
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m}:${s.toString().padStart(2, '0')}`;
};

const fmtSize = (bytes?: number) => {
  if (!bytes) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
};

/** Mirrored waveform path, so an audio tile shows its shape instead of a generic icon. */
function waveformPath(peaks: number[], width: number, height: number): string {
  if (!peaks.length) return '';
  const columns = Math.max(1, Math.min(peaks.length, Math.floor(width / 2)));
  const step = peaks.length / columns;
  const mid = height / 2;
  let top = '';
  let bottom = '';
  for (let i = 0; i < columns; i++) {
    let peak = 0;
    for (let j = Math.floor(i * step); j < Math.floor((i + 1) * step); j++) {
      peak = Math.max(peak, peaks[j] || 0);
    }
    const x = columns > 1 ? (i / (columns - 1)) * width : 0;
    const y = peak * mid * 0.85;
    top += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${(mid - y).toFixed(1)} `;
    bottom = `L${x.toFixed(1)},${(mid + y).toFixed(1)} ` + bottom;
  }
  return `${top}${bottom}Z`;
}

function AudioArtwork({ wave }: { wave?: WaveformData }) {
  const d = useMemo(() => (wave ? waveformPath(wave.peaks, 200, 70) : ''), [wave]);
  if (!d) return <span style={{ fontSize: 28 }}>🎵</span>;
  return (
    <svg viewBox="0 0 200 70" preserveAspectRatio="none"
      style={{ width: '100%', height: '100%' }}>
      <path d={d} fill="rgba(255,255,255,0.55)" />
    </svg>
  );
}

/**
 * Hover-scrub poster for video tiles: moving across the tile seeks the clip, the
 * way a Filmora/Premiere bin previews footage without opening it.
 */
function VideoPoster({ entry, projectId }: { entry: MediaEntry; projectId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [scrubbing, setScrubbing] = useState(false);

  const onMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    if (!video || !entry.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    video.currentTime = ratio * entry.duration;
  }, [entry.duration]);

  return (
    <div
      style={{ position: 'absolute', inset: 0 }}
      onMouseEnter={() => setScrubbing(true)}
      onMouseLeave={() => setScrubbing(false)}
      onMouseMove={onMove}
    >
      <img
        src={mediaThumbUrl(projectId, entry.id)}
        alt=""
        draggable={false}
        style={{ width: '100%', height: '100%', objectFit: 'cover', display: scrubbing ? 'none' : 'block' }}
        onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = 'hidden'; }}
      />
      {scrubbing && (
        <video
          ref={videoRef}
          src={mediaStreamUrl(entry.path)}
          muted
          preload="metadata"
          draggable={false}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      )}
    </div>
  );
}

export default function MediaPool() {
  const { project, setError, currentTime, updateProject } = useProjectStore();
  const { pickFiles, pickFolders } = useElectron();
  const [media, setMedia] = useState<LibraryEntry[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [filter, setFilter] = useState<KindFilter>('all');
  const [query, setQuery] = useState('');
  const [dropActive, setDropActive] = useState(false);
  const [copyOnImport, setCopyOnImport] = useState(false);
  const [waves, setWaves] = useState<Record<string, WaveformData>>({});
  const [renaming, setRenaming] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  // Eye-contact jobs by media id, while they run on the GPU.
  const [eyeJobs, setEyeJobs] = useState<Record<string, EyeContactJob>>({});
  const [eyeDialog, setEyeDialog] = useState<MediaEntry | null>(null);
  // Right-click on an audio/video tile: "Convert voice…" (the Voice panel's Convert tab).
  const [menu, setMenu] = useState<{ x: number; y: number; items: MenuEntry[] } | null>(null);
  // Multicam: the audio-replace dialog, and the media id whose angle sync is measuring.
  const [audioSync, setAudioSync] = useState<MediaEntry | null>(null);
  const [angleSyncing, setAngleSyncing] = useState<string | null>(null);
  // Stabilization jobs by media id while they run, and the dialog's clip.
  const [stabJobs, setStabJobs] = useState<Record<string, StabilizeJob>>({});
  const [stabDialog, setStabDialog] = useState<MediaEntry | null>(null);

  const projectId = project?.id;
  const fps = (project?.timeline as any)?.fps_num
    ? (project!.timeline as any).fps_num / ((project!.timeline as any).fps_den || 1)
    : 30;

  const refresh = useCallback(async () => {
    if (!projectId) return;
    try {
      const res = await listMedia(projectId);
      setMedia(res.media || []);
      setCounts(res.counts || {});
    } catch (err: any) {
      console.warn('Failed to load media library:', err);
    }
  }, [projectId]);

  useEffect(() => { refresh(); }, [refresh]);
  // A voice conversion or a microphone upload added an item.
  useEffect(() => mediaPoolChanged.subscribe((id) => { if (id === projectId) refresh(); }), [projectId, refresh]);

  // Audio tiles need their peaks before they can draw anything but an icon.
  const waveReq = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!projectId) return;
    media.filter(m => m.kind === 'audio' && !m.missing).forEach((entry) => {
      if (waves[entry.id] || waveReq.current.has(entry.id)) return;
      waveReq.current.add(entry.id);
      getMediaWaveform(projectId, entry.id)
        .then(w => setWaves(prev => ({ ...prev, [entry.id]: w })))
        .catch(() => waveReq.current.delete(entry.id));
    });
  }, [projectId, media, waves]);

  const runImport = useCallback(async (paths: string[]) => {
    if (!projectId || !paths.length) return;
    setBusy(paths.length > 1 ? `Importing ${paths.length} items…` : 'Importing…');
    setNote(null);
    try {
      const res = await importMedia(projectId, paths, copyOnImport);
      setMedia(res.media || []);
      await refresh();
      const parts = [`Added ${res.added.length}`];
      if (res.skipped?.length) parts.push(`skipped ${res.skipped.length}`);
      setNote(parts.join(' · '));
    } catch (err: any) {
      setError(err.message || 'Failed to import media');
    } finally {
      setBusy(null);
    }
  }, [projectId, copyOnImport, refresh, setError]);

  const handleImportFiles = useCallback(async () => {
    const paths = await pickFiles(MEDIA_FILTERS);
    if (!paths.length) {
      if (!window.electronAPI) setError('File picking needs the desktop app.');
      return;
    }
    runImport(paths);
  }, [pickFiles, runImport, setError]);

  const handleImportFolder = useCallback(async () => {
    const paths = await pickFolders();
    if (paths.length) runImport(paths);
  }, [pickFolders, runImport]);

  // File ▸ Import Media… routes here while the media pool is mounted.
  useCommand('file.import', handleImportFiles);

  // Files dragged straight from Explorer/Finder. Electron exposes the real path
  // on the File object; a plain browser does not, so say so rather than failing mute.
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDropActive(false);
    const files = Array.from(e.dataTransfer.files || []) as (File & { path?: string })[];
    if (!files.length) return;
    const paths = files.map(f => f.path).filter((p): p is string => !!p);
    if (!paths.length) {
      setError('Dropping files from the desktop only works in the app window — use ＋ Import here.');
      return;
    }
    runImport(paths);
  }, [runImport, setError]);

  const handleRemove = useCallback(async (entry: MediaEntry) => {
    if (!projectId) return;
    try {
      const res = await removeMedia(projectId, entry.id, !entry.linked);
      setMedia(res.media || []);
      await refresh();
    } catch (err: any) {
      setError(err.message);
    }
  }, [projectId, refresh, setError]);

  const handleRename = useCallback(async (entry: MediaEntry, name: string) => {
    setRenaming(null);
    if (!projectId || !name.trim() || name === entry.name) return;
    try {
      const res = await renameMedia(projectId, entry.id, name.trim());
      setMedia(res.media || []);
    } catch (err: any) {
      setError(err.message);
    }
  }, [projectId, setError]);

  const handleRelink = useCallback(async (entry: MediaEntry) => {
    if (!projectId) return;
    const picked = await pickFiles(MEDIA_FILTERS);
    if (!picked.length) return;
    try {
      const res = await relinkMedia(projectId, entry.id, picked[0]);
      setMedia(res.media || []);
      await refresh();
    } catch (err: any) {
      setError(err.message);
    }
  }, [projectId, pickFiles, refresh, setError]);

  /** The swap changed the project's source and timeline sources server-side: pull them in. */
  const reloadSources = useCallback(async () => {
    if (!projectId) return;
    await refresh();
    try {
      const fresh = await getProject(projectId);
      updateProject({ source_video: fresh.source_video, timeline: fresh.timeline } as any);
    } catch (err) {
      console.warn('Failed to reload the project after an eye-contact swap:', err);
    }
  }, [projectId, refresh, updateProject]);

  const handleEyeContact = useCallback(async (entry: MediaEntry, setup: EyeContactSetup) => {
    setEyeDialog(null);
    if (!projectId || eyeJobs[entry.id]) return;
    try {
      // the speaker setup only: lens marks travel with the run and are stored per recording
      const { lens_marks: _marks, ...speaker } = setup;
      await saveEyeContactSetup(speaker);
      const { job_id } = await startEyeContact(projectId, entry.id, setup);
      setEyeJobs(prev => ({ ...prev, [entry.id]: {
        id: job_id, media_id: entry.id, status: 'queued', progress: 0,
        stage: 'Waiting for the GPU', result: null, error: null,
      } }));
    } catch (err: any) {
      setError(err.message || 'Could not start eye contact correction');
    }
  }, [projectId, eyeJobs, setError]);

  const handleRevertEyeContact = useCallback(async (entry: MediaEntry) => {
    if (!projectId) return;
    try {
      const res = await revertEyeContact(projectId, entry.id);
      setMedia(res.media || []);
      await reloadSources();
      setNote(`${entry.name}: original eyes restored`);
    } catch (err: any) {
      setError(err.message);
    }
  }, [projectId, reloadSources, setError]);

  // Poll running eye-contact jobs; a finished one has already swapped the corrected file in.
  const activeEyeJobs = Object.values(eyeJobs).filter(j => j.status === 'queued' || j.status === 'running');
  const activeEyeKey = activeEyeJobs.map(j => j.id).join(',');
  useEffect(() => {
    if (!activeEyeKey) return;
    const ids = activeEyeKey.split(',');
    const timer = setInterval(async () => {
      for (const jobId of ids) {
        try {
          const next = await getEyeContactStatus(jobId);
          if (next.status === 'completed' || next.status === 'failed' || next.status === 'cancelled') {
            setEyeJobs(prev => {
              const rest = { ...prev };
              delete rest[next.media_id];
              return rest;
            });
            if (next.status === 'completed') {
              await reloadSources();
              const r: (NonNullable<typeof next.result> & { eyes_raised_deg?: number; head_pitch_deg?: number }) | null = next.result;
              setNote(r
                ? `Eye contact fixed: eyes re-aimed ${r.prompter_offset_deg}° toward the lens (${r.median_shift_px}px)${r.eyes_raised_deg ? `, raised ${r.eyes_raised_deg}° for a head bowed ${r.head_pitch_deg}°` : ''}, reading sweep ${r.reading_sweep_deg}° calmed, ${Math.round(r.seconds)}s`
                : 'Eye contact fixed');
            } else if (next.status === 'failed') {
              setError(`Eye contact correction failed: ${next.error}`);
            }
          } else {
            setEyeJobs(prev => ({ ...prev, [next.media_id]: next }));
          }
        } catch (err) {
          console.warn('Eye contact status poll failed:', err);
        }
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [activeEyeKey, reloadSources, setError]);

  const handleStabilize = useCallback(async (entry: MediaEntry, setup: StabilizeSetup) => {
    setStabDialog(null);
    if (!projectId || stabJobs[entry.id]) return;
    try {
      await saveStabilizeSetup(setup);
      const { job_id } = await startStabilize(projectId, entry.id, setup);
      setStabJobs(prev => ({ ...prev, [entry.id]: {
        id: job_id, media_id: entry.id, status: 'queued', progress: 0,
        stage: 'Waiting for the GPU', result: null, error: null,
      } }));
    } catch (err: any) {
      setError(err.message || 'Could not start stabilization');
    }
  }, [projectId, stabJobs, setError]);

  const handleRevertStabilize = useCallback(async (entry: MediaEntry) => {
    if (!projectId) return;
    try {
      const res = await revertStabilize(projectId, entry.id);
      setMedia(res.media || []);
      await reloadSources();
      setNote(`${entry.name}: original (unstabilized) clip restored`);
    } catch (err: any) {
      setError(err.message);
    }
  }, [projectId, reloadSources, setError]);

  // Poll running stabilization jobs; a finished one has already swapped the steady file in.
  const activeStabKey = Object.values(stabJobs)
    .filter(j => j.status === 'queued' || j.status === 'running').map(j => j.id).join(',');
  useEffect(() => {
    if (!activeStabKey) return;
    const ids = activeStabKey.split(',');
    const timer = setInterval(async () => {
      for (const jobId of ids) {
        try {
          const next = await getStabilizeStatus(jobId);
          if (next.status === 'completed' || next.status === 'failed' || next.status === 'cancelled') {
            setStabJobs(prev => {
              const rest = { ...prev };
              delete rest[next.media_id];
              return rest;
            });
            if (next.status === 'completed') {
              await reloadSources();
              const r = next.result as StabilizeReport | null;
              setNote(r ? `Stabilized: ${stabilizeSummary(r)}, ${Math.round(r.seconds)}s` : 'Stabilized');
            } else if (next.status === 'failed') {
              setError(`Stabilization failed: ${next.error}`);
            }
          } else {
            setStabJobs(prev => ({ ...prev, [next.media_id]: next }));
          }
        } catch (err) {
          console.warn('Stabilize status poll failed:', err);
        }
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [activeStabKey, reloadSources, setError]);

  /** Register a library video as a camera angle; the result (or the 422 detail) shows inline. */
  const handleSyncAngle = useCallback(async (entry: MediaEntry) => {
    if (!projectId || angleSyncing) return;
    setAngleSyncing(entry.id);
    setBusy(`Syncing ${entry.name} as a camera angle…`);
    setNote(null);
    try {
      const r = await syncAngle(projectId, entry.id);
      await refresh();
      setNote(`${entry.name}: angle synced · offset ${fmtOffset(r.offset_s)} · confidence `
        + `${r.confidence.toFixed(1)} · ${fmtDrift(r.drift_ppm)}${r.warning ? ` · ⚠ ${r.warning}` : ''}`);
    } catch (err) {
      setNote(`${entry.name}: not synced. ${apiErrorDetail(err)}`);
    } finally {
      setBusy(null);
      setAngleSyncing(null);
    }
  }, [projectId, angleSyncing, refresh]);

  /** Drop straight onto the timeline at the playhead, for people who'd rather click. */
  const handleAddToTimeline = useCallback(async (entry: MediaEntry) => {
    if (!projectId || entry.missing) return;
    setBusy('Adding to timeline…');
    try {
      const res = await addMedia(projectId, entry.path, {
        timelineStartFrame: Math.round(currentTime * fps),
      });
      if (res?.timeline) updateProject({ timeline: res.timeline });
      setNote(`${entry.name} added at the playhead`);
    } catch (err: any) {
      setError(err.message || 'Could not add to the timeline');
    } finally {
      setBusy(null);
    }
  }, [projectId, currentTime, fps, updateProject, setError]);

  const onDragStart = (e: React.DragEvent, entry: MediaEntry) => {
    if (entry.missing) { e.preventDefault(); return; }
    e.dataTransfer.setData(MEDIA_DND_TYPE, JSON.stringify(entry));
    e.dataTransfer.effectAllowed = 'copy';
  };

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return media.filter(m =>
      (filter === 'all' || m.kind === filter) &&
      (!needle || m.name.toLowerCase().includes(needle)));
  }, [media, filter, query]);

  if (!project || !projectId) return null;

  const tabs: { id: KindFilter; label: string }[] = [
    { id: 'all', label: `All (${media.length})` },
    { id: 'video', label: `Video (${counts.video ?? 0})` },
    { id: 'image', label: `Images (${counts.image ?? 0})` },
    { id: 'audio', label: `Music (${counts.audio ?? 0})` },
  ];

  return (
    <div
      className={`media-library ${dropActive ? 'drop-active' : ''}`}
      onDragOver={(e) => {
        // Only light up for files coming from outside the app.
        if (e.dataTransfer.types.includes('Files')) {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'copy';
          setDropActive(true);
        }
      }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDropActive(false); }}
      onDrop={handleDrop}
    >
      <div className="media-head">
        <div>
          <div className="text-sm font-weight-semibold">Media Library</div>
          <div className="text-xs text-muted">
            Drag a tile onto a timeline track, or drop files here from your desktop.
          </div>
        </div>
        <div className="media-head-actions">
          <button className="btn btn-sm btn-primary" disabled={!!busy} onClick={handleImportFiles}>
            ＋ Import
          </button>
          <button className="btn btn-sm" disabled={!!busy} onClick={handleImportFolder} title="Import every media file in a folder">
            📁 Folder
          </button>
        </div>
      </div>

      <div className="media-toolbar">
        <input
          className="media-search"
          type="search"
          placeholder="Search library…"
          aria-label="Search library"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="media-copy-toggle" title="Copy files into the project folder instead of linking them where they are. Linking is instant and uses no extra disk.">
          <input type="checkbox" aria-label="Copy in" checked={copyOnImport} onChange={(e) => setCopyOnImport(e.target.checked)} />
          <span className="text-xs text-muted">Copy in</span>
        </label>
      </div>

      <div className="media-tabs">
        {tabs.map(tab => (
          <button key={tab.id}
            className={`media-tab ${filter === tab.id ? 'active' : ''}`}
            onClick={() => setFilter(tab.id)}>{tab.label}</button>
        ))}
      </div>

      {(busy || note) && (
        <div className="media-status text-xs">{busy || note}</div>
      )}

      {media.length === 0 ? (
        <div className="media-empty" onClick={handleImportFiles}>
          <div style={{ fontSize: 34 }}>🎞️ 🎵 🖼️</div>
          <div className="text-sm">Import video, music and images</div>
          <div className="text-xs">Click here, use ＋ Import, or drop files from your desktop</div>
        </div>
      ) : visible.length === 0 ? (
        <div className="media-empty subtle">
          <div className="text-sm">Nothing matches that filter</div>
          <button className="btn btn-sm" onClick={() => { setQuery(''); setFilter('all'); }}>Clear filters</button>
        </div>
      ) : (
        <div className="media-grid">
          {visible.map((entry) => (
            <div
              key={entry.id}
              draggable={!entry.missing}
              onDragStart={(e) => onDragStart(e, entry)}
              onDoubleClick={() => handleAddToTimeline(entry)}
              onContextMenu={(e) => {
                if (entry.missing || (entry.kind !== 'audio' && entry.kind !== 'video')) return;
                e.preventDefault();
                setMenu({ x: e.clientX, y: e.clientY, items: [
                  { label: 'Convert voice…', onClick: () => openVoicePanel({ tab: 'convert', mediaId: entry.id }) },
                  { separator: true },
                  entry.kind === 'video'
                    ? { label: entry.angle ? 'Re-sync camera angle' : 'Sync as camera angle',
                        disabled: !!angleSyncing || entry.has_audio === false,
                        hint: entry.has_audio === false ? 'Needs a soundtrack to line up' : 'Another sync is running',
                        onClick: () => handleSyncAngle(entry) }
                    : { label: 'Use as programme audio (sync)…', onClick: () => setAudioSync(entry) },
                  ...(entry.kind === 'video' ? [
                    { separator: true },
                    { label: entry.stabilize ? 'Stabilize again…' : 'Stabilize…',
                      disabled: !!stabJobs[entry.id], hint: 'Stabilization is running',
                      onClick: () => setStabDialog(entry) },
                    ...(entry.stabilize ? [{ label: 'Revert stabilization',
                      disabled: !!stabJobs[entry.id], hint: 'Stabilization is running',
                      onClick: () => handleRevertStabilize(entry) }] : []),
                  ] as MenuEntry[] : []),
                ] });
              }}
              className={`media-tile ${entry.missing ? 'missing' : ''}`}
              title={entry.missing
                ? `Missing: ${entry.path}`
                : `${entry.name}\n${entry.path}\nDrag onto a track, or double-click to add at the playhead`}
            >
              <div className={`media-art ${entry.kind}`}>
                {entry.missing ? (
                  <span style={{ fontSize: 24 }}>⚠️</span>
                ) : entry.kind === 'audio' ? (
                  <AudioArtwork wave={waves[entry.id]} />
                ) : entry.kind === 'video' ? (
                  <VideoPoster entry={entry} projectId={projectId} />
                ) : (
                  <img src={mediaThumbUrl(projectId, entry.id)} alt="" draggable={false}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                )}

                <span className="media-kind">{KIND_ICON[entry.kind]}</span>
                {entry.generated && (
                  <span className="media-generated-badge" title="Added automatically from the timeline">
                    generated
                  </span>
                )}
                {entry.duration > 0 && <span className="media-duration">{fmtDuration(entry.duration)}</span>}
                {entry.eye_contact && !eyeJobs[entry.id] && (
                  <span className="media-eye-badge" title={`Eyes re-aimed at the lens.${eyeLensSummary(entry.eye_contact.report) ? ` ${eyeLensSummary(entry.eye_contact.report)}.` : ''} Original: ${entry.eye_contact.original_path}`}>
                    eye contact
                  </span>
                )}
                {entry.angle && (
                  <span className="media-angle-badge"
                    title={`Camera angle of the main recording · offset ${fmtOffset(entry.angle.offset_s)} · confidence ${entry.angle.confidence.toFixed(1)} · ${fmtDrift(entry.angle.drift_ppm)}`}>
                    angle · {fmtOffset(entry.angle.offset_s)}
                  </span>
                )}
                {entry.stabilize && !stabJobs[entry.id] && (
                  <span className="media-stab-badge"
                    title={`Stabilized: ${stabilizeSummary(entry.stabilize.report) || 'done'}. Original: ${entry.stabilize.original_path}`}>
                    stabilized
                  </span>
                )}
                {stabJobs[entry.id] && !eyeJobs[entry.id] && (
                  <div className="media-eye-progress" onClick={(e) => e.stopPropagation()}>
                    <div>{stabJobs[entry.id].stage} {Math.round(stabJobs[entry.id].progress * 100)}%</div>
                    <div className="media-eye-progress-bar">
                      <div style={{ width: `${Math.round(stabJobs[entry.id].progress * 100)}%` }} />
                    </div>
                    <button className="btn btn-xs" onClick={() => cancelStabilize(stabJobs[entry.id].id)}>Cancel</button>
                  </div>
                )}
                {eyeJobs[entry.id] && (
                  <div className="media-eye-progress" onClick={(e) => e.stopPropagation()}>
                    <div>{eyeJobs[entry.id].stage} {Math.round(eyeJobs[entry.id].progress * 100)}%</div>
                    <div className="media-eye-progress-bar">
                      <div style={{ width: `${Math.round(eyeJobs[entry.id].progress * 100)}%` }} />
                    </div>
                    <button className="btn btn-xs" onClick={() => cancelEyeContact(eyeJobs[entry.id].id)}>Cancel</button>
                  </div>
                )}

                <div className="media-actions">
                  <button className="media-action" title="Add at playhead" aria-label="Add at playhead"
                    onClick={(e) => { e.stopPropagation(); handleAddToTimeline(entry); }}>＋</button>
                  <button className="media-action" title="Rename" aria-label="Rename"
                    onClick={(e) => { e.stopPropagation(); setRenaming(entry.id); }}>✎</button>
                  {entry.kind === 'video' && !entry.missing && !eyeJobs[entry.id] && (
                    <button className="media-action"
                      title={entry.eye_contact
                        ? 'Re-run eye contact correction (from the original recording)'
                        : 'Fix eye contact: re-aim teleprompter gaze at the lens (GPU)'}
                      aria-label={entry.eye_contact
                        ? 'Re-run eye contact correction (from the original recording)'
                        : 'Fix eye contact: re-aim teleprompter gaze at the lens (GPU)'}
                      onClick={(e) => { e.stopPropagation(); setEyeDialog(entry); }}>👁</button>
                  )}
                  {entry.eye_contact && !eyeJobs[entry.id] && (
                    <button className="media-action" title="Revert to the original eyes" aria-label="Revert to the original eyes"
                      onClick={(e) => { e.stopPropagation(); handleRevertEyeContact(entry); }}>↺</button>
                  )}
                  {entry.kind === 'video' && !entry.missing && !stabJobs[entry.id] && (
                    <button className="media-action"
                      title={entry.stabilize
                        ? 'Re-run stabilization (from the original recording)'
                        : 'Stabilize: steady handheld shake'}
                      aria-label={entry.stabilize
                        ? 'Re-run stabilization (from the original recording)'
                        : 'Stabilize: steady handheld shake'}
                      onClick={(e) => { e.stopPropagation(); setStabDialog(entry); }}>≋</button>
                  )}
                  <button className="media-action danger" title={entry.linked ? 'Remove from library' : 'Remove and delete the copy'}
                    aria-label={entry.linked ? 'Remove from library' : 'Remove and delete the copy'}
                    onClick={(e) => { e.stopPropagation(); handleRemove(entry); }}>✕</button>
                </div>
              </div>

              <div className="media-meta">
                {renaming === entry.id ? (
                  <input
                    className="media-rename"
                    aria-label={`Rename ${entry.name}`}
                    autoFocus
                    defaultValue={entry.name}
                    onBlur={(e) => handleRename(entry, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                      if (e.key === 'Escape') setRenaming(null);
                    }}
                  />
                ) : (
                  <div className="media-name">{entry.name}</div>
                )}
                <div className="media-sub text-xs text-muted">
                  {entry.missing ? (
                    <button className="btn btn-xs" onClick={(e) => { e.stopPropagation(); handleRelink(entry); }}>
                      Relink…
                    </button>
                  ) : (
                    [entry.width && entry.height ? `${entry.width}×${entry.height}` : null,
                     fmtSize(entry.size_bytes)].filter(Boolean).join(' · ')
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {dropActive && <div className="media-dropzone">Drop media to add it to the library</div>}
      {menu && <ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => setMenu(null)} />}
      {eyeDialog && projectId && (
        <EyeContactDialog entry={eyeDialog} projectId={projectId}
          onRun={(setup) => handleEyeContact(eyeDialog, setup)}
          onClose={() => { setEyeDialog(null); refresh(); }} />
      )}
      {stabDialog && projectId && (
        <StabilizeDialog entry={stabDialog} projectId={projectId}
          onRun={(setup) => handleStabilize(stabDialog, setup)}
          onClose={() => { setStabDialog(null); refresh(); }} />
      )}
      {audioSync && projectId && (
        <SyncAudioDialog entry={audioSync} projectId={projectId} onClose={() => setAudioSync(null)} />
      )}
    </div>
  );
}
