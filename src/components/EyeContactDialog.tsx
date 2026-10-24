import { useEffect, useRef, useState } from 'react';
import {
  mediaStreamUrl, mediaFileUrl, getEyeContactStatus, cancelEyeContact,
  getEyeContactSetup, startEyeContactPreview,
  MediaEntry, EyeContactJob, EyeContactSetup, EyeContactPreview, EyeContactMeasures, EyeContactReport,
} from '../hooks/api';

/** m:ss (h:mm:ss past an hour); 0 reads 0:00. */
const fmtClock = (seconds: number) => {
  const total = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m}:${s.toString().padStart(2, '0')}`;
};

/** A lens mark to a tenth of a second: 30:51.4. */
const fmtMark = (seconds: number) => {
  const t = Math.max(0, seconds);
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, '0')}`;
};

/** Where the lens came from, in plain words. */
export function lensSourceText(r: EyeContactMeasures, marks?: number): string {
  switch (r.lens_source) {
    case 'marks': return marks ? `From your ${marks} mark${marks === 1 ? '' : 's'}` : 'From your marks';
    case 'measured': return 'Measured from your looks into the camera';
    case 'geometry': return 'No camera looks found: using the prompter geometry';
    default: return '';
  }
}

/** What each eye moved: "R 1.65 px, L 1.50 px" (the shared push when no per-eye numbers). */
export function perEyeText(r: EyeContactMeasures): string {
  const per = r.median_shift_px_per_eye || {};
  const parts = ['R', 'L'].filter((k) => per[k] !== undefined).map((k) => `${k} ${per[k]} px`);
  return parts.length ? parts.join(', ') : `${r.median_shift_px} px`;
}

/** One line for the library tile after a full run: lens source, measured angle, per-eye px. */
export function eyeLensSummary(r?: EyeContactReport): string {
  if (!r || !r.lens_source) return '';
  const source = lensSourceText(r, r.settings?.lens_marks?.length);
  const angle = r.measured_shift_deg !== undefined ? `, ${r.measured_shift_deg}° sideways` : '';
  return `Lens: ${source}${angle}; eyes moved ${perEyeText(r)}`;
}

const PREVIEW_SECONDS = 10;
/** What changes the eyes (lens marks included); quality only changes the full run's file size. */
const setupKey = ({ quality, ...rest }: EyeContactSetup) => JSON.stringify(rest);
type CompareMode = 'before' | 'after' | 'split';

/**
 * Before/after of a corrected stretch. "After" is the clock and carries the audio;
 * "before" follows it, and lands on the exact same frame whenever playback pauses, so
 * flipping between the two while paused shows only what the correction moved.
 * Click the picture to zoom 3x on that spot (both sides zoom together).
 * `timeRef` receives the player's position (seconds into the clip) for "Mark this moment".
 */
function EyePreviewPlayer({ preview, timeRef }: {
  preview: EyeContactPreview;
  timeRef?: React.MutableRefObject<number>;
}) {
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
        if (timeRef) timeRef.current = a.currentTime;
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
        face found {preview.face_found_pct}%, re-aimed {preview.prompter_offset_deg}°
        ({preview.median_shift_px}px), reading sweep {preview.reading_sweep_deg}°.
        Pause and flip Before/After to compare one frame; click the picture to zoom on the eyes.
      </div>
    </div>
  );
}

/**
 * Where the lens is decides how far the eyes get re-aimed. Auto mode (the default) measures
 * it from the speaker's own looks into the camera, or from moments the user marks; the
 * prompter geometry (saved once, pre-filled every time) is the fallback, and manual mode.
 * Lens marks belong to the recording: seeded from the media entry, sent with every preview
 * and run (the backend stores them on the entry), never saved into the speaker setup.
 */
export default function EyeContactDialog({ entry, projectId, onRun, onClose }: {
  entry: MediaEntry;
  projectId: string;
  onRun: (setup: EyeContactSetup) => void;
  onClose: () => void;
}) {
  const [setup, setSetup] = useState<EyeContactSetup | null>(null);
  const maxStart = Math.max(0, Math.floor((entry.duration || 0) - PREVIEW_SECONDS));
  const [startS, setStartS] = useState(() => Math.floor(maxStart / 2));
  const [previewJob, setPreviewJob] = useState<EyeContactJob | null>(null);
  const [preview, setPreview] = useState<{ result: EyeContactPreview; setupKey: string; marks: number } | null>(null);
  const [marks, setMarks] = useState<number[]>(() => [...(entry.eye_contact_lens_marks || [])]);
  // Omitted lens_marks keep the stored ones; send them once known (seeded) or edited.
  const [marksEdited, setMarksEdited] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const playerTime = useRef(0);
  const jobRef = useRef<EyeContactJob | null>(null);
  jobRef.current = previewJob;

  useEffect(() => {
    getEyeContactSetup().then(setSetup).catch(() => setSetup({
      mode: 'auto', prompter_side: 'left', angle_deg: 0, prompter_cm: 30, camera_cm: 110, steadiness: 0.7,
      aim_deg: 0, pitch_deg: 0, auto_pitch: true, quality: 'high',
    }));
  }, []);

  // a preview still running when the dialog goes away is wasted GPU time
  useEffect(() => () => {
    const j = jobRef.current;
    if (j && (j.status === 'queued' || j.status === 'running')) cancelEyeContact(j.id).catch(() => {});
  }, []);

  const previewSetupKey = useRef('');
  const previewMarks = useRef(0);
  const jobId = previewJob?.id;
  const jobActive = previewJob?.status === 'queued' || previewJob?.status === 'running';
  useEffect(() => {
    if (!jobId || !jobActive) return;
    const timer = setInterval(async () => {
      try {
        const next = await getEyeContactStatus(jobId);
        setPreviewJob(next);
        if (next.status === 'completed' && next.result) {
          setPreview({ result: next.result as EyeContactPreview, setupKey: previewSetupKey.current,
            marks: previewMarks.current });
        }
      } catch (err) {
        console.warn('Eye contact preview poll failed:', err);
      }
    }, 600);
    return () => clearInterval(timer);
  }, [jobId, jobActive]);

  if (!setup) return null;

  const sendMarks = marksEdited || entry.eye_contact_lens_marks !== undefined;
  const request = (s: EyeContactSetup): EyeContactSetup => {
    const { lens_marks: _old, ...speaker } = s;
    return sendMarks ? { ...speaker, lens_marks: marks } : speaker;
  };

  const runPreview = async () => {
    if (jobRef.current && jobActive) await cancelEyeContact(jobRef.current.id).catch(() => {});
    const req = request(setup);
    previewSetupKey.current = setupKey(req);
    previewMarks.current = req.lens_marks?.length ?? 0;
    try {
      const { job_id } = await startEyeContactPreview(projectId, entry.id, req, startS, PREVIEW_SECONDS);
      setPreviewJob({ id: job_id, media_id: entry.id, status: 'queued', progress: 0,
        stage: 'Starting', result: null, error: null });
    } catch (err: any) {
      setPreviewJob({ id: '', media_id: entry.id, status: 'failed', progress: 0, stage: 'Failed',
        result: null, error: err.message || 'Could not start the preview' });
    }
  };
  const stale = preview && preview.setupKey !== setupKey(request(setup));
  const showPreviewPane = Boolean(preview || jobActive);
  const set = (patch: Partial<EyeContactSetup>) => setSetup({ ...setup, ...patch });
  const auto = setup.mode !== 'manual';
  const fromDistances = Math.round(Math.atan2(setup.prompter_cm, Math.max(setup.camera_cm, 1)) * 180 / Math.PI);
  const angle = setup.prompter_side === 'auto' ? null
    : (setup.angle_deg > 0 ? setup.angle_deg : fromDistances);

  // The newest numbers: this dialog's preview, else the last full run's report.
  const latest: EyeContactMeasures | undefined = preview?.result ?? entry.eye_contact?.report;
  const latestFromRun = !preview && Boolean(entry.eye_contact?.report);
  const latestMarks = preview ? preview.marks : entry.eye_contact?.report?.settings?.lens_marks?.length;
  const candidates = latest?.camera_look_candidates || [];
  const mirrored = Boolean(latest?.side_check && /mirror/i.test(latest.side_check));

  const isMarked = (t: number) => marks.some((m) => Math.abs(m - t) < 0.3);
  const addMark = (t: number) => {
    const r = Math.round(t * 10) / 10;
    if (isMarked(r)) return;
    setMarks([...marks, r].sort((a, b) => a - b));
    setMarksEdited(true);
  };
  const removeMark = (i: number) => {
    setMarks(marks.filter((_, j) => j !== i));
    setMarksEdited(true);
  };
  const jumpTo = (t: number) => setStartS(Math.min(maxStart, Math.max(0, Math.floor(t))));
  // the player's position while the shown preview starts where the slider is, else the slider
  const markNow = () => addMark(preview && preview.result.start_s === startS
    ? preview.result.start_s + playerTime.current : startS);

  const sideSelect = (
    <label>
      Prompter is on your
      <select value={setup.prompter_side}
        aria-label="Prompter side"
        onChange={(e) => set({ prompter_side: e.target.value as EyeContactSetup['prompter_side'] })}>
        <option value="left">left of the lens</option>
        <option value="right">right of the lens</option>
        <option value="auto">guess from my eyes</option>
      </select>
    </label>
  );
  // manual: a distance replaces a typed angle; auto: the fallback must not clear the override
  const distanceInputs = (resetAngle: boolean) => (
    <>
      <label>
        Sideways distance, lens to text (cm)
        <input type="number" min={0} max={200} step={1} value={setup.prompter_cm}
          aria-label="Sideways distance to text"
          onChange={(e) => set({ prompter_cm: Number(e.target.value), ...(resetAngle ? { angle_deg: 0 } : {}) })} />
      </label>
      <label>
        Your distance from the camera (cm)
        <input type="number" min={20} max={1000} step={5} value={setup.camera_cm}
          aria-label="Distance from camera"
          onChange={(e) => set({ camera_cm: Number(e.target.value), ...(resetAngle ? { angle_deg: 0 } : {}) })} />
      </label>
    </>
  );

  return (
    <div className="eye-dialog-backdrop" onClick={onClose}>
      <div className={`eye-dialog ${showPreviewPane ? 'wide' : ''}`} onClick={(e) => e.stopPropagation()}>
        <div className="eye-dialog-settings">
        <div className="eye-dialog-title">Fix eye contact: {entry.name}</div>
        <div className="eye-mode-toggle" role="group" aria-label="Correction mode">
          <button className={`btn btn-xs ${auto ? 'btn-primary' : ''}`} aria-pressed={auto}
            title="Find the lens from your own looks into the camera (or your marks)"
            onClick={() => set({ mode: 'auto' })}>Measure the lens (auto)</button>
          <button className={`btn btn-xs ${!auto ? 'btn-primary' : ''}`} aria-pressed={!auto}
            title="Re-aim by the prompter geometry you enter"
            onClick={() => set({ mode: 'manual' })}>Prompter geometry (manual)</button>
        </div>
        {auto ? (
          <>
            <div className="eye-section">
              <div className="eye-section-title">Lens</div>
              {latest?.lens_source ? (
                <>
                  <span className={`eye-lens-pill ${latest.lens_source}`}>{lensSourceText(latest, latestMarks)}</span>
                  <div className="text-xs text-muted">
                    {latest.measured_shift_deg !== undefined ? `${latest.measured_shift_deg}° sideways; ` : ''}
                    eyes moved {perEyeText(latest)}{latestFromRun ? ' (last run)' : ''}
                  </div>
                </>
              ) : (
                <div className="text-xs text-muted">Preview a stretch to measure where your lens is.</div>
              )}
            </div>
            <div className="eye-section">
              <div className="eye-section-title">Where you look into the camera</div>
              {candidates.length > 0 ? (
                <div className="eye-chip-row">
                  {candidates.map((t) => (
                    <span className="eye-chip" key={t}>
                      <button className="btn btn-xs" onClick={() => jumpTo(t)} disabled={maxStart === 0}
                        title="Move the preview start here">{fmtClock(t)}</button>
                      <button className="btn btn-xs" onClick={() => addMark(t)} disabled={isMarked(t)}
                        title="Measure the lens from this moment">+ mark</button>
                    </span>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-muted">
                  {latest ? 'No held looks into the camera found.' : 'A preview lists the moments you look into the lens.'}
                </div>
              )}
              <button className="btn btn-xs eye-mark-now" onClick={markNow}
                title="Add the preview player's position (or the preview start) as a moment you look into the lens">
                Mark this moment
              </button>
              {marks.length > 0 && (
                <div className="eye-chip-row">
                  {marks.map((m, i) => (
                    <span className="eye-chip mark" key={m}>
                      {fmtMark(m)}
                      <button className="eye-chip-x" aria-label={`Remove mark at ${fmtMark(m)}`}
                        title="Remove this mark" onClick={() => removeMark(i)}>×</button>
                    </span>
                  ))}
                </div>
              )}
              <div className="text-xs text-muted">
                {marks.length > 0
                  ? 'The lens is measured from these moments.'
                  : 'No marks: the lens is measured from the looks found automatically.'}
              </div>
            </div>
            <label>
              Override the measured angle (0 = use measured)
              <input type="number" min={0} max={45} step={0.5} value={setup.angle_deg}
                aria-label="Override the measured angle"
                onChange={(e) => set({ angle_deg: Number(e.target.value) })} />
            </label>
          </>
        ) : (
          <>
            {sideSelect}
            {mirrored && (
              <div className="text-xs eye-warning">
                The video looks mirrored: the prompter shows up on the other side in the picture, so the push
                direction was taken from the recording rather than from the side set here.
              </div>
            )}
            {setup.prompter_side !== 'auto' && (
              <>
                <label>
                  Re-aim angle (degrees): lower it if the eyes overshoot the lens
                  <input type="number" min={1} max={45} step={0.5} value={angle ?? fromDistances}
                    aria-label="Re-aim angle"
                    onChange={(e) => set({ angle_deg: Number(e.target.value) })} />
                </label>
                <div className="text-xs text-muted">or work it out from distances:</div>
                {distanceInputs(true)}
              </>
            )}
          </>
        )}
        <label>
          Fine aim (degrees, + = toward your left)
          <input type="number" min={-15} max={15} step={0.5} value={setup.aim_deg}
            aria-label="Fine aim"
            onChange={(e) => set({ aim_deg: Number(e.target.value) })} />
        </label>
        <label>
          Up / down (degrees, + = look lower, − = look higher)
          <input type="number" min={-10} max={10} step={0.5} value={setup.pitch_deg ?? 0}
            aria-label="Up / down aim"
            onChange={(e) => set({ pitch_deg: Number(e.target.value) })} />
        </label>
        {auto && (
          <div className="text-xs text-muted">Fine aim and Up / down add on top of the measurement.</div>
        )}
        <label>
          Calm the reading sweep {Math.round(setup.steadiness * 100)}%
          <input type="range" min={0} max={1} step={0.05} value={setup.steadiness}
            aria-label="Calm the reading sweep"
            onChange={(e) => set({ steadiness: Number(e.target.value) })} />
          <span className="text-xs text-muted">
            Removes the left-right creep of reading a line. Natural eye movement always stays.
          </span>
        </label>
        {auto && (
          <details className="eye-geometry">
            <summary>Prompter geometry (fallback)</summary>
            <div className="text-xs text-muted">Used only when the take has no looks into the camera.</div>
            {sideSelect}
            {distanceInputs(false)}
          </details>
        )}
        <label>
          Quality
          <select value={setup.quality}
            aria-label="Quality"
            onChange={(e) => set({ quality: e.target.value as EyeContactSetup['quality'] })}>
            <option value="standard">standard (smaller file)</option>
            <option value="high">high (about the source size)</option>
            <option value="max">max</option>
          </select>
        </label>
        {!auto && angle !== null && (
          <div className="text-xs text-muted">Re-aims your eyes about {angle}° toward the lens.</div>
        )}
        <div className="eye-preview-row">
          {maxStart > 0 && (
            <label>
              Preview {PREVIEW_SECONDS} s starting at {fmtClock(startS)}
              <input type="range" min={0} max={maxStart} step={1} value={startS}
                aria-label="Preview start time"
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
          <button className="btn btn-sm btn-primary" onClick={() => onRun(request(setup))}>Fix eye contact</button>
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
                <EyePreviewPlayer key={preview.result.after_path} preview={preview.result} timeRef={playerTime} />
                {preview.result.contact_sheet ? (
                  <div className="eye-sheet">
                    <div className="text-xs text-muted">Eyes, before over after (click to open full size)</div>
                    <img key={preview.result.contact_sheet} src={mediaFileUrl(preview.result.contact_sheet)}
                      alt="Eye crops, before over after" onClick={() => setSheetOpen(true)} />
                  </div>
                ) : (
                  <div className="text-xs text-muted">No eye contact sheet for this preview.</div>
                )}
              </>
            )}
          </div>
        )}
        {sheetOpen && preview?.result.contact_sheet && (
          <div className="eye-sheet-lightbox" role="dialog" aria-label="Eye contact sheet"
            onClick={(e) => { e.stopPropagation(); setSheetOpen(false); }}>
            <img src={mediaFileUrl(preview.result.contact_sheet)} alt="Eye crops, before over after, full size" />
          </div>
        )}
      </div>
    </div>
  );
}
