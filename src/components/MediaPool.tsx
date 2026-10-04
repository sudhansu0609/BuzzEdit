import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { useCommand } from '../hooks/commands';
import {
  useElectron, listMedia, importMedia, removeMedia, renameMedia, relinkMedia,
  getMediaWaveform, mediaThumbUrl, mediaStreamUrl, addMedia, getProject,
  startEyeContact, getEyeContactStatus, cancelEyeContact, revertEyeContact,
  getEyeContactSetup, saveEyeContactSetup, startEyeContactPreview,
  MediaEntry, MEDIA_FILTERS, WaveformData, EyeContactJob, EyeContactSetup, EyeContactPreview,
} from '../hooks/api';

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

/**
 * Where the prompter sits relative to the lens decides how far the eyes get re-aimed:
 * reading text 30 cm beside a lens at 1.1 m is a ~15 degree look away. Saved once,
 * pre-filled here every time, editable per run.
 */
const PREVIEW_SECONDS = 10;
/** What changes the eyes; quality only changes the full run's file size. */
const setupKey = ({ quality, ...rest }: EyeContactSetup) => JSON.stringify(rest);
type CompareMode = 'before' | 'after' | 'split';

/**
 * Before/after of a corrected stretch. "After" is the clock and carries the audio;
 * "before" follows it, and lands on the exact same frame whenever playback pauses, so
 * flipping between the two while paused shows only what the correction moved.
 * Click the picture to zoom 3x on that spot (both sides zoom together).
 */
function EyePreviewPlayer({ preview }: { preview: EyeContactPreview }) {
  const before = useRef<HTMLVideoElement>(null);
  const after = useRef<HTMLVideoElement>(null);
  const [mode, setMode] = useState<CompareMode>('after');
  const [zoom, setZoom] = useState<{ x: number; y: number } | null>(null);
  const [playing, setPlaying] = useState(true);
  const [time, setTime] = useState(0);
  const [aspect, setAspect] = useState(16 / 9);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const a = after.current, b = before.current;
      if (a && b) {
        setTime(a.currentTime);
        if (!a.paused && Math.abs(b.currentTime - a.currentTime) > 0.1) b.currentTime = a.currentTime;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const lockstep = () => {
    const a = after.current, b = before.current;
    if (a && b) b.currentTime = a.currentTime;
  };
  const togglePlay = () => {
    const a = after.current, b = before.current;
    if (!a || !b) return;
    if (a.paused) {
      lockstep();
      a.play().catch(() => {});
      b.play().catch(() => {});
      setPlaying(true);
    } else {
      a.pause();
      b.pause();
      lockstep();
      setPlaying(false);
    }
  };
  const seek = (t: number) => {
    if (after.current) after.current.currentTime = t;
    if (before.current) before.current.currentTime = t;
    setTime(t);
  };
  const onPictureClick = (e: React.MouseEvent<HTMLVideoElement>) => {
    if (zoom) { setZoom(null); return; }
    const r = e.currentTarget.getBoundingClientRect();
    setZoom({ x: ((e.clientX - r.left) / r.width) * 100, y: ((e.clientY - r.top) / r.height) * 100 });
  };
  const zoomStyle: React.CSSProperties = zoom
    ? { transform: 'scale(3)', transformOrigin: `${zoom.x}% ${zoom.y}%` } : {};
  const video = (which: 'before' | 'after') => (
    <video
      ref={which === 'before' ? before : after}
      src={mediaStreamUrl(which === 'before' ? preview.before_path : preview.after_path)}
      autoPlay loop playsInline preload="auto" muted={which === 'before'}
      style={zoomStyle}
      onClick={onPictureClick}
      onSeeked={which === 'after' ? () => { if (after.current?.paused) lockstep(); } : undefined}
      onLoadedMetadata={which === 'after'
        ? (e) => setAspect(e.currentTarget.videoWidth / Math.max(1, e.currentTarget.videoHeight)) : undefined}
    />
  );

  return (
    <div className="eye-compare">
      <div className={`eye-compare-stage ${mode === 'split' ? 'split' : ''}`}
        style={{ aspectRatio: mode === 'split' ? `${aspect * 2}` : `${aspect}` }}>
        <div className={`eye-compare-cell ${mode === 'after' ? 'hidden' : ''}`}>
          {video('before')}
          <span className="eye-compare-label">Before</span>
        </div>
        <div className={`eye-compare-cell ${mode === 'before' ? 'hidden' : ''}`}>
          {video('after')}
          <span className="eye-compare-label after">After</span>
        </div>
      </div>
      <div className="eye-compare-controls">
        <button className="btn btn-xs" onClick={togglePlay}>{playing ? '❚❚' : '▶'}</button>
        <input type="range" min={0} max={preview.duration_s} step={0.01} value={time}
          onChange={(e) => seek(Number(e.target.value))} />
        <span className="text-xs text-muted">{time.toFixed(1)}s</span>
        <div className="eye-compare-modes">
          {(['before', 'after', 'split'] as CompareMode[]).map((m) => (
            <button key={m} className={`btn btn-xs ${mode === m ? 'btn-primary' : ''}`}
              onClick={() => setMode(m)}>
              {m === 'split' ? 'Side by side' : m[0].toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <div className="text-xs text-muted">
        {fmtDuration(preview.start_s)}–{fmtDuration(preview.start_s + preview.duration_s)}:
        face found {preview.face_found_pct}%, re-aimed {preview.prompter_offset_deg}°
        ({preview.median_shift_px}px), reading sweep {preview.reading_sweep_deg}°.
        Pause and flip Before/After to compare one frame; click the picture to zoom on the eyes.
      </div>
    </div>
  );
}

function EyeContactDialog({ entry, projectId, onRun, onClose }: {
  entry: MediaEntry;
  projectId: string;
  onRun: (setup: EyeContactSetup) => void;
  onClose: () => void;
}) {
  const [setup, setSetup] = useState<EyeContactSetup | null>(null);
  const maxStart = Math.max(0, Math.floor((entry.duration || 0) - PREVIEW_SECONDS));
  const [startS, setStartS] = useState(() => Math.floor(maxStart / 2));
  const [previewJob, setPreviewJob] = useState<EyeContactJob | null>(null);
  const [preview, setPreview] = useState<{ result: EyeContactPreview; setupKey: string } | null>(null);
  const jobRef = useRef<EyeContactJob | null>(null);
  jobRef.current = previewJob;

  useEffect(() => {
    getEyeContactSetup().then(setSetup).catch(() => setSetup({
      prompter_side: 'left', angle_deg: 0, prompter_cm: 30, camera_cm: 110, steadiness: 0.7, aim_deg: 0,
      pitch_deg: 0, quality: 'high',
    }));
  }, []);

  // a preview still running when the dialog goes away is wasted GPU time
  useEffect(() => () => {
    const j = jobRef.current;
    if (j && (j.status === 'queued' || j.status === 'running')) cancelEyeContact(j.id).catch(() => {});
  }, []);

  const previewSetupKey = useRef('');
  const jobId = previewJob?.id;
  const jobActive = previewJob?.status === 'queued' || previewJob?.status === 'running';
  useEffect(() => {
    if (!jobId || !jobActive) return;
    const timer = setInterval(async () => {
      try {
        const next = await getEyeContactStatus(jobId);
        setPreviewJob(next);
        if (next.status === 'completed' && next.result) {
          setPreview({ result: next.result as EyeContactPreview, setupKey: previewSetupKey.current });
        }
      } catch (err) {
        console.warn('Eye contact preview poll failed:', err);
      }
    }, 600);
    return () => clearInterval(timer);
  }, [jobId, jobActive]);

  if (!setup) return null;

  const runPreview = async () => {
    if (jobRef.current && jobActive) await cancelEyeContact(jobRef.current.id).catch(() => {});
    previewSetupKey.current = setupKey(setup);
    try {
      const { job_id } = await startEyeContactPreview(projectId, entry.id, setup, startS, PREVIEW_SECONDS);
      setPreviewJob({ id: job_id, media_id: entry.id, status: 'queued', progress: 0,
        stage: 'Starting', result: null, error: null });
    } catch (err: any) {
      setPreviewJob({ id: '', media_id: entry.id, status: 'failed', progress: 0, stage: 'Failed',
        result: null, error: err.message || 'Could not start the preview' });
    }
  };
  const stale = preview && preview.setupKey !== setupKey(setup);
  const showPreviewPane = Boolean(preview || jobActive);
  const set = (patch: Partial<EyeContactSetup>) => setSetup({ ...setup, ...patch });
  const fromDistances = Math.round(Math.atan2(setup.prompter_cm, Math.max(setup.camera_cm, 1)) * 180 / Math.PI);
  const angle = setup.prompter_side === 'auto' ? null
    : (setup.angle_deg > 0 ? setup.angle_deg : fromDistances);
  return (
    <div className="eye-dialog-backdrop" onClick={onClose}>
      <div className={`eye-dialog ${showPreviewPane ? 'wide' : ''}`} onClick={(e) => e.stopPropagation()}>
        <div className="eye-dialog-settings">
        <div className="eye-dialog-title">Fix eye contact: {entry.name}</div>
        <label>
          Prompter is on your
          <select value={setup.prompter_side}
            onChange={(e) => set({ prompter_side: e.target.value as EyeContactSetup['prompter_side'] })}>
            <option value="left">left of the lens</option>
            <option value="right">right of the lens</option>
            <option value="auto">guess from my eyes</option>
          </select>
        </label>
        {setup.prompter_side !== 'auto' && (
          <>
            <label>
              Re-aim angle (degrees): lower it if the eyes overshoot the lens
              <input type="number" min={1} max={45} step={0.5} value={angle ?? fromDistances}
                onChange={(e) => set({ angle_deg: Number(e.target.value) })} />
            </label>
            <div className="text-xs text-muted">or work it out from distances:</div>
            <label>
              Sideways distance, lens to text (cm)
              <input type="number" min={0} max={200} step={1} value={setup.prompter_cm}
                onChange={(e) => set({ prompter_cm: Number(e.target.value), angle_deg: 0 })} />
            </label>
            <label>
              Your distance from the camera (cm)
              <input type="number" min={20} max={1000} step={5} value={setup.camera_cm}
                onChange={(e) => set({ camera_cm: Number(e.target.value), angle_deg: 0 })} />
            </label>
          </>
        )}
        <label>
          Fine aim (degrees, + = toward your left)
          <input type="number" min={-15} max={15} step={0.5} value={setup.aim_deg}
            onChange={(e) => set({ aim_deg: Number(e.target.value) })} />
        </label>
        <label>
          Up / down (degrees, + = look lower, − = look higher)
          <input type="number" min={-10} max={10} step={0.5} value={setup.pitch_deg ?? 0}
            onChange={(e) => set({ pitch_deg: Number(e.target.value) })} />
        </label>
        <label>
          Steadiness {Math.round(setup.steadiness * 100)}%
          <input type="range" min={0} max={1} step={0.05} value={setup.steadiness}
            onChange={(e) => set({ steadiness: Number(e.target.value) })} />
        </label>
        <label>
          Quality
          <select value={setup.quality}
            onChange={(e) => set({ quality: e.target.value as EyeContactSetup['quality'] })}>
            <option value="standard">standard (smaller file)</option>
            <option value="high">high (about the source size)</option>
            <option value="max">max</option>
          </select>
        </label>
        <div className="text-xs text-muted">
          {angle !== null ? `Re-aims your eyes about ${angle}° toward the lens. ` : ''}
          Steadiness calms the left-right reading sweep; 0% keeps it, 100% locks the eyes on the lens.
        </div>
        <div className="eye-preview-row">
          {maxStart > 0 && (
            <label>
              Preview {PREVIEW_SECONDS} s starting at {fmtDuration(startS) || '0:00'}
              <input type="range" min={0} max={maxStart} step={1} value={startS}
                onChange={(e) => setStartS(Number(e.target.value))} />
            </label>
          )}
          <button className="btn btn-sm" onClick={runPreview} disabled={jobActive}
            title="Correct just this stretch on the GPU and compare before/after. Nothing in the project changes.">
            {jobActive ? 'Previewing…' : preview ? 'Preview again' : `Preview ${PREVIEW_SECONDS} s`}
          </button>
          {previewJob?.status === 'failed' && (
            <div className="text-xs eye-preview-error">Preview failed: {previewJob.error}</div>
          )}
        </div>
        <div className="eye-dialog-buttons">
          <button className="btn btn-sm" onClick={onClose}>Cancel</button>
          <button className="btn btn-sm btn-primary" onClick={() => onRun(setup)}>Fix eye contact</button>
        </div>
        </div>
        {showPreviewPane && (
          <div className="eye-dialog-preview">
            {jobActive && previewJob && (
              <div className="media-eye-progress eye-preview-progress">
                <div>{previewJob.stage} {Math.round(previewJob.progress * 100)}%</div>
                <div className="media-eye-progress-bar">
                  <div style={{ width: `${Math.round(previewJob.progress * 100)}%` }} />
                </div>
              </div>
            )}
            {preview && (
              <>
                {stale && !jobActive && (
                  <div className="text-xs eye-preview-stale">Settings changed since this preview: preview again to see them.</div>
                )}
                <EyePreviewPlayer key={preview.result.after_path} preview={preview.result} />
              </>
            )}
          </div>
        )}
      </div>
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
      await saveEyeContactSetup(setup);
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
              const r = next.result;
              setNote(r
                ? `Eye contact fixed: eyes re-aimed ${r.prompter_offset_deg}° toward the lens (${r.median_shift_px}px), reading sweep ${r.reading_sweep_deg}° calmed, ${Math.round(r.seconds)}s`
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
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label className="media-copy-toggle" title="Copy files into the project folder instead of linking them where they are. Linking is instant and uses no extra disk.">
          <input type="checkbox" checked={copyOnImport} onChange={(e) => setCopyOnImport(e.target.checked)} />
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
                  <span className="media-eye-badge" title={`Eyes re-aimed at the lens. Original: ${entry.eye_contact.original_path}`}>
                    eye contact
                  </span>
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
                  <button className="media-action" title="Add at playhead"
                    onClick={(e) => { e.stopPropagation(); handleAddToTimeline(entry); }}>＋</button>
                  <button className="media-action" title="Rename"
                    onClick={(e) => { e.stopPropagation(); setRenaming(entry.id); }}>✎</button>
                  {entry.kind === 'video' && !entry.missing && !eyeJobs[entry.id] && (
                    <button className="media-action"
                      title={entry.eye_contact
                        ? 'Re-run eye contact correction (from the original recording)'
                        : 'Fix eye contact: re-aim teleprompter gaze at the lens (GPU)'}
                      onClick={(e) => { e.stopPropagation(); setEyeDialog(entry); }}>👁</button>
                  )}
                  {entry.eye_contact && !eyeJobs[entry.id] && (
                    <button className="media-action" title="Revert to the original eyes"
                      onClick={(e) => { e.stopPropagation(); handleRevertEyeContact(entry); }}>↺</button>
                  )}
                  <button className="media-action danger" title={entry.linked ? 'Remove from library' : 'Remove and delete the copy'}
                    onClick={(e) => { e.stopPropagation(); handleRemove(entry); }}>✕</button>
                </div>
              </div>

              <div className="media-meta">
                {renaming === entry.id ? (
                  <input
                    className="media-rename"
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
      {eyeDialog && projectId && (
        <EyeContactDialog entry={eyeDialog} projectId={projectId}
          onRun={(setup) => handleEyeContact(eyeDialog, setup)}
          onClose={() => setEyeDialog(null)} />
      )}
    </div>
  );
}
