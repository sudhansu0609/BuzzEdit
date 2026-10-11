/**
 * A process effect (AUDIO_SUITE_PLAN.md §2.1 "Processes"): rendered into a new version of the
 * source instead of running live, like Audition's "(process)" dialogs. Preview renders the
 * range and plays it; Apply runs the job with its progress, then shows the source's version
 * count and an Undo that pops the newest one. Noise Reduction (process) carries the
 * noise-print card. A job keeps being followed after the card closes, so the panel still
 * hears about the new version.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  EffectDef, UserPresets, VersionsResult, audioFileUrl, cloneValue, defaultParams, errorText,
  getProcessJob, getVersions, previewProcess, routeMissing, startProcess, undoProcess,
} from '../../hooks/audioApi';
import ParamEditor from './ParamEditor';
import PresetMenu from './PresetMenu';
import NoisePrintCard from './NoisePrintCard';

const POLL_MS = 500;
const PREVIEW_MAX_S = 20;
const PREVIEW_DEFAULT_S = 10;

export interface ProcessRange {
  start_s: number;
  end_s: number;
}

interface JobView {
  id: string;
  status: string;
  progress: number;
  message?: string;
}

export default function ProcessCard({
  def, projectId, sourceId, sourceLabel, range, playhead, userPresets,
  onPresetsChange, onBeforePreview, onSourceChanged,
}: {
  def: EffectDef;
  projectId: string;
  /** undefined = the programme source */
  sourceId?: string;
  sourceLabel: string;
  /** programme seconds for the programme and master scopes (the backend maps them onto the
   *  source), the clip's source seconds for a clip; null = the whole file */
  range: ProcessRange | null;
  /** the same seconds at the playhead: where a preview without a range starts */
  playhead: () => number;
  userPresets: UserPresets;
  onPresetsChange: (presets: UserPresets) => void;
  /** stop the live engine so the preview is heard alone */
  onBeforePreview: () => void;
  /** a version was added or undone: refresh the timeline and the live preview */
  onSourceChanged: () => void;
}) {
  const [params, setParams] = useState<Record<string, unknown>>(() => defaultParams(def));
  const [preset, setPreset] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [job, setJob] = useState<JobView | null>(null);
  const [versions, setVersions] = useState<VersionsResult | null>(null);
  const [undoing, setUndoing] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const alive = useRef(true);
  const sourceChanged = useRef(onSourceChanged);
  sourceChanged.current = onSourceChanged;

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const fail = (err: unknown, what: string) => {
    if (!alive.current) return;
    if (routeMissing(err)) setMissing(true);
    else setNote(`${what}: ${errorText(err)}`);
  };

  const loadVersions = useCallback(async () => {
    try {
      const v = await getVersions(projectId, sourceId);
      if (alive.current) setVersions(v);
    } catch (err) {
      if (!alive.current) return;
      if (routeMissing(err)) setMissing(true);
      else setNote(`Versions: ${errorText(err)}`);
    }
  }, [projectId, sourceId]);

  useEffect(() => { void loadVersions(); }, [loadVersions]);

  const preview = async () => {
    onBeforePreview();
    setPreviewing(true);
    setPreviewUrl(null);
    setNote(null);
    const start = range ? range.start_s : Math.max(0, playhead());
    const seconds = range
      ? Math.min(PREVIEW_MAX_S, Math.max(0.5, range.end_s - range.start_s))
      : PREVIEW_DEFAULT_S;
    try {
      const res = await previewProcess(projectId, { source_id: sourceId, effect: def.id, params, start_s: start, seconds });
      if (alive.current) setPreviewUrl(audioFileUrl(res.path));
    } catch (err) {
      fail(err, 'Preview failed');
    } finally {
      if (alive.current) setPreviewing(false);
    }
  };

  // Followed to the end even if the card closes: the new version must reach the timeline.
  const follow = (jobId: string) => {
    const tick = async () => {
      try {
        const j = await getProcessJob(projectId, jobId);
        if (alive.current) setJob({ id: jobId, status: j.status, progress: j.progress ?? 0, message: j.message });
        if (j.status === 'completed') {
          if (alive.current) await loadVersions();
          sourceChanged.current();
          return;
        }
        if (j.status === 'failed') {
          if (alive.current) setNote(`Apply failed: ${j.error || j.message || 'unknown error'}`);
          return;
        }
      } catch (err) {
        fail(err, 'Lost track of the job');
        return;
      }
      window.setTimeout(() => void tick(), POLL_MS);
    };
    window.setTimeout(() => void tick(), POLL_MS);
  };

  const apply = async () => {
    setNote(null);
    try {
      const { job_id: jobId } = await startProcess(projectId, {
        source_id: sourceId, effect: def.id, params,
        start_s: range ? range.start_s : null, end_s: range ? range.end_s : null,
      });
      setJob({ id: jobId, status: 'pending', progress: 0, message: 'queued' });
      follow(jobId);
    } catch (err) {
      fail(err, 'Apply failed');
    }
  };

  const undo = async () => {
    setUndoing(true);
    setNote(null);
    try {
      const r = await undoProcess(projectId, sourceId);
      if (alive.current) {
        setJob(null);
        setNote(r.undone
          ? `Undid ${r.undone.effect}: ${r.history_len} version${r.history_len === 1 ? '' : 's'} left.`
          : 'Nothing to undo.');
      }
      await loadVersions();
      sourceChanged.current();
    } catch (err) {
      fail(err, 'Undo failed');
    } finally {
      if (alive.current) setUndoing(false);
    }
  };

  const running = !!job && (job.status === 'pending' || job.status === 'running');
  const count = versions?.versions?.length ?? 0;
  const percent = Math.round(Math.min(1, Math.max(0, job?.progress ?? 0)) * 100);

  return (
    <div className="audio-card audio-process">
      <div className="audio-card-title">
        {def.name} <span className="audio-badge process">process</span>
      </div>
      <div className="text-xs text-muted">{def.description}</div>
      {missing && <div className="audio-note">Process effects are not available on this backend yet.</div>}
      <PresetMenu def={def} userPresets={userPresets} current={params} active={preset}
        onApply={(name, values) => {
          setParams({ ...defaultParams(def), ...cloneValue(values) });
          setPreset(name);
        }}
        onPresetsChange={onPresetsChange} />
      {def.id === 'noise_reduction' && (
        <NoisePrintCard projectId={projectId} sourceId={sourceId}
          outputNoiseOnly={params.output_noise_only === true} />
      )}
      <ParamEditor def={def} values={params}
        onChange={(key, value) => {
          setParams((current) => ({ ...current, [key]: value }));
          setPreset(null);
        }} />
      <div className="audio-process-range">
        {sourceLabel}: {range ? `${range.start_s.toFixed(2)}–${range.end_s.toFixed(2)} s` : 'the whole file'}
      </div>
      <div className="audio-process-actions">
        <button className="audio-btn" disabled={previewing || running} onClick={() => void preview()}
          title={range && range.end_s - range.start_s > PREVIEW_MAX_S
            ? `Renders the first ${PREVIEW_MAX_S} s of the range and plays it`
            : 'Renders the range and plays it'}>
          {previewing ? 'Rendering…' : 'Preview'}
        </button>
        <button className="audio-btn primary" disabled={running} onClick={() => void apply()}
          title="Write a new version of the source; the original file is never touched">
          {running ? 'Applying…' : 'Apply'}
        </button>
        <span className="audio-versions">Versions: {count}</span>
        <button className="audio-btn" disabled={undoing || running || count === 0} onClick={() => void undo()}
          title="Go back to the version before the newest">
          {undoing ? 'Undoing…' : 'Undo'}
        </button>
      </div>
      {running && (
        <div className="audio-progress">
          <div className="audio-progress-fill" style={{ width: `${percent}%` }} />
          <span className="audio-progress-text">{job?.message || job?.status} · {percent}%</span>
        </div>
      )}
      {job?.status === 'completed' && (
        <div className="text-xs">Applied: version {count} is what plays and renders now.</div>
      )}
      {previewUrl && <audio className="audio-preview" controls autoPlay src={previewUrl} />}
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}
