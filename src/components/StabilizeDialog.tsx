import { useEffect, useRef, useState } from 'react';
import {
  mediaStreamUrl, getStabilizeSetup, getStabilizeStatus, cancelStabilize, stabilizeAvailable,
  startStabilizeAnalyse, startStabilizePreview,
  MediaEntry, StabilizeSetup, StabilizeJob, StabilizeAnalysis, StabilizePreview, StabilizeReport,
} from '../hooks/api';

/** m:ss (h:mm:ss past an hour). */
const fmtClock = (seconds: number) => {
  const total = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m}:${s.toString().padStart(2, '0')}`;
};

/** One line for the library tile after a full run: zoom and the shake left over. */
export function stabilizeSummary(r?: StabilizeReport): string {
  if (!r) return '';
  return `zoom ${r.zoom_pct}%, residual shake ${r.mean_shift_px} px mean (${r.max_shift_px} px peak)`;
}

const PREVIEW_SECONDS = 10;
const DEFAULT_SETUP: StabilizeSetup = {
  strength: 0.6, crop: 'auto', smoothing_s: 1.0, zoom_pct: null, tripod: false, quality: 'high',
};
/** What changes the picture; quality only changes the full run's file size. */
const setupKey = ({ quality, ...rest }: StabilizeSetup) => JSON.stringify(rest);
type CompareMode = 'before' | 'after' | 'split';

const isActive = (j: StabilizeJob | null) => j?.status === 'queued' || j?.status === 'running';

/** Polls one stabilize job while it runs; `onDone` gets the completed job. */
function useStabilizeJob(onDone: (job: StabilizeJob) => void) {
  const [job, setJob] = useState<StabilizeJob | null>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;
  const id = job?.id;
  const active = isActive(job);
  useEffect(() => {
    if (!id || !active) return;
    const timer = setInterval(async () => {
      try {
        const next = await getStabilizeStatus(id);
        setJob(next);
        if (next.status === 'completed') doneRef.current(next);
      } catch (err) {
        console.warn('Stabilize job poll failed:', err);
      }
    }, 700);
    return () => clearInterval(timer);
  }, [id, active]);
  return { job, setJob, active };
}

/** Before/after of a stabilized stretch; "after" is the clock and carries the audio. */
function StabilizePreviewPlayer({ preview }: { preview: StabilizePreview }) {
  const before = useRef<HTMLVideoElement>(null);
  const after = useRef<HTMLVideoElement>(null);
  const [mode, setMode] = useState<CompareMode>('split');
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
  const video = (which: 'before' | 'after') => (
    <video
      ref={which === 'before' ? before : after}
      src={mediaStreamUrl(which === 'before' ? preview.before_path : preview.after_path)}
      autoPlay loop playsInline preload="auto" muted={which === 'before'}
      onClick={togglePlay}
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
        <button className="btn btn-xs" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>{playing ? '❚❚' : '▶'}</button>
        <input type="range" min={0} max={preview.duration_s} step={0.01} value={time}
          aria-label="Preview position"
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
        {fmtClock(preview.start_s)}–{fmtClock(preview.start_s + preview.duration_s)}:
        {' '}{stabilizeSummary(preview)}, {preview.seconds}s to process.
      </div>
    </div>
  );
}

/**
 * Stabilize one library clip. Same shape as the eye-contact dialog: measure, tune, preview
 * a stretch before/after, then run it on the whole clip (MediaPool tracks the full run).
 */
export default function StabilizeDialog({ entry, projectId, onRun, onClose }: {
  entry: MediaEntry;
  projectId: string;
  onRun: (setup: StabilizeSetup) => void;
  onClose: () => void;
}) {
  const [setup, setSetup] = useState<StabilizeSetup | null>(null);
  const [avail, setAvail] = useState<{ available: boolean; reason: string } | null>(null);
  const maxStart = Math.max(0, Math.floor((entry.duration || 0) - PREVIEW_SECONDS));
  const [startS, setStartS] = useState(() => Math.floor(maxStart / 2));
  const [analysis, setAnalysis] = useState<StabilizeAnalysis | null>(null);
  const [preview, setPreview] = useState<{ result: StabilizePreview; setupKey: string } | null>(null);
  const previewSetupKey = useRef('');

  const analyse = useStabilizeJob((j) => setAnalysis(j.result as StabilizeAnalysis));
  const prev = useStabilizeJob((j) => setPreview({ result: j.result as StabilizePreview,
    setupKey: previewSetupKey.current }));
  const prevJobRef = useRef<StabilizeJob | null>(null);
  prevJobRef.current = prev.job;

  useEffect(() => {
    getStabilizeSetup().then(setSetup).catch(() => setSetup(DEFAULT_SETUP));
    stabilizeAvailable().then(setAvail).catch(() => setAvail(null));
  }, []);

  // a preview still running when the dialog goes away is wasted GPU time
  useEffect(() => () => {
    const j = prevJobRef.current;
    if (j && isActive(j)) cancelStabilize(j.id).catch(() => {});
  }, []);

  if (!setup) return null;

  const set = (patch: Partial<StabilizeSetup>) => setSetup({ ...setup, ...patch });
  const unavailable = avail !== null && !avail.available;
  const failed = (err: any, what: string): StabilizeJob => ({
    id: '', media_id: entry.id, status: 'failed', progress: 0, stage: 'Failed',
    result: null, error: err?.message || `Could not start the ${what}`,
  });
  const queued = (id: string, stage: string): StabilizeJob => ({
    id, media_id: entry.id, status: 'queued', progress: 0, stage, result: null, error: null,
  });

  const runAnalyse = async () => {
    try {
      const { job_id } = await startStabilizeAnalyse(projectId, entry.id, setup);
      analyse.setJob(queued(job_id, 'Measuring the shake'));
    } catch (err) {
      analyse.setJob(failed(err, 'measurement'));
    }
  };

  const runPreview = async () => {
    previewSetupKey.current = setupKey(setup);
    try {
      // the backend cancels an older preview of this clip itself
      const { job_id } = await startStabilizePreview(projectId, entry.id, setup, startS, PREVIEW_SECONDS);
      prev.setJob(queued(job_id, 'Waiting for the GPU'));
    } catch (err) {
      prev.setJob(failed(err, 'preview'));
    }
  };

  const stale = preview && preview.setupKey !== setupKey(setup);
  const showPreviewPane = Boolean(preview || prev.active);
  const lastRun = entry.stabilize?.report;
  const progressBlock = (job: StabilizeJob) => (
    <div className="media-eye-progress eye-preview-progress">
      <div>{job.stage} {Math.round(job.progress * 100)}%</div>
      <div className="media-eye-progress-bar">
        <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
      </div>
    </div>
  );

  return (
    <div className="eye-dialog-backdrop" onClick={onClose}>
      <div className={`eye-dialog ${showPreviewPane ? 'wide' : ''}`} onClick={(e) => e.stopPropagation()}>
        <div className="eye-dialog-settings">
          <div className="eye-dialog-title">Stabilize: {entry.name}</div>
          {unavailable && (
            <div className="text-xs eye-preview-error">Stabilization unavailable: {avail?.reason}</div>
          )}
          <div className="eye-section">
            <div className="eye-section-title">Shake</div>
            {analysis ? (
              <span className={`stab-shake-pill ${analysis.shaky ? 'shaky' : 'steady'}`}
                title={`Peak ${analysis.max_shift_px} px over ${analysis.frames} frames (1080p pixels, deliberate pans not counted)`}>
                mean shake {analysis.mean_shift_px} px ({analysis.shaky ? 'shaky' : 'steady'})
              </span>
            ) : (
              <div className="text-xs text-muted">
                {lastRun ? `Last run: ${stabilizeSummary(lastRun)}.` : 'Measure the clip to see whether it needs it.'}
              </div>
            )}
            <button className="btn btn-xs" onClick={runAnalyse} disabled={analyse.active || unavailable}
              title="Run only the motion-detection pass (CPU). It is cached, so a full run afterwards skips it.">
              {analyse.active
                ? `Measuring… ${Math.round((analyse.job?.progress ?? 0) * 100)}%`
                : analysis ? 'Measure again' : 'Measure the shake'}
            </button>
            {analyse.job?.status === 'failed' && (
              <div className="text-xs eye-preview-error">Measurement failed: {analyse.job.error}</div>
            )}
          </div>
          <label>
            Strength {Math.round(setup.strength * 100)}%
            <input type="range" min={0} max={1} step={0.05} value={setup.strength}
              aria-label="Strength"
              onChange={(e) => set({ strength: Number(e.target.value) })} />
          </label>
          <label>
            Smoothing {setup.smoothing_s.toFixed(1)} s
            <input type="range" min={0} max={10} step={0.1} value={setup.smoothing_s}
              aria-label="Smoothing"
              onChange={(e) => set({ smoothing_s: Number(e.target.value) })} />
            <span className="text-xs text-muted">Longer smooths more of the camera path (and zooms in more).</span>
          </label>
          <label>
            Edges
            <select value={setup.crop} aria-label="Crop"
              onChange={(e) => set({ crop: e.target.value as StabilizeSetup['crop'] })}>
              <option value="auto">zoom in to hide the edges (auto)</option>
              <option value="keep">keep the size, fill edges from earlier frames</option>
              <option value="black">keep the size, black edges</option>
            </select>
          </label>
          <label className="stab-check">
            <input type="checkbox" checked={setup.tripod} aria-label="Tripod"
              onChange={(e) => set({ tripod: e.target.checked })} />
            <span>Locked-off camera (tripod): hold every frame to the first</span>
          </label>
          <label>
            Quality
            <select value={setup.quality} aria-label="Quality"
              onChange={(e) => set({ quality: e.target.value as StabilizeSetup['quality'] })}>
              <option value="standard">standard (smaller file)</option>
              <option value="high">high (about the source size)</option>
              <option value="max">max</option>
            </select>
          </label>
          <div className="eye-preview-row">
            {maxStart > 0 && (
              <label>
                Preview {PREVIEW_SECONDS} s starting at {fmtClock(startS)}
                <input type="range" min={0} max={maxStart} step={1} value={startS}
                  aria-label="Preview start time"
                  onChange={(e) => setStartS(Number(e.target.value))} />
              </label>
            )}
            <button className="btn btn-sm" onClick={runPreview} disabled={prev.active || unavailable}
              title="Stabilize just this stretch and compare before/after. Nothing in the project changes.">
              {prev.active ? 'Previewing…' : preview ? 'Preview again' : `Preview ${PREVIEW_SECONDS} s`}
            </button>
            {prev.job?.status === 'failed' && (
              <div className="text-xs eye-preview-error">Preview failed: {prev.job.error}</div>
            )}
          </div>
          <div className="eye-dialog-buttons">
            <button className="btn btn-sm" onClick={onClose}>Cancel</button>
            <button className="btn btn-sm btn-primary" onClick={() => onRun(setup)} disabled={unavailable}
              title={entry.stabilize ? 'Re-run from the original recording' : 'Stabilize the whole clip and swap it in'}>
              Stabilize
            </button>
          </div>
        </div>
        {showPreviewPane && (
          <div className="eye-dialog-preview">
            {prev.active && prev.job && progressBlock(prev.job)}
            {preview && (
              <>
                {stale && !prev.active && (
                  <div className="text-xs eye-preview-stale">Settings changed since this preview: preview again to see them.</div>
                )}
                <StabilizePreviewPlayer key={preview.result.after_path} preview={preview.result} />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
