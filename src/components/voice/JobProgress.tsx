/**
 * A voice job's progress (VOICE_CHANGER_PLAN.md §2.7). Every voice route that does work answers
 * `{job_id}`; the job runs in the process lane's registry and is polled on
 * `GET /api/audio/{project}/process/{job_id}`.
 *
 * `watchJob()` polls once per job however many places show it, and keeps polling to the end even
 * when the tab that started it is gone: what the job's end does (a new media-pool item, a new
 * version on the timeline) is attached to `watch.done` when the job is started, so it happens
 * exactly once. `<JobProgress>` only shows a watch.
 */

import { useEffect, useState } from 'react';
import { AudioApiError, errorText, getProcessJob } from '../../hooks/audioApi';
import type { JobState } from '../../hooks/voiceApi';

const POLL_MS = 600;
/** Poll failures in a row before the job counts as lost (the backend may be busy for a moment). */
const MAX_MISSES = 5;

export interface JobWatch {
  id: string;
  projectId: string;
  current: () => JobState;
  subscribe: (fn: (job: JobState) => void) => () => void;
  /** The final state, completed or failed (a lost job resolves as failed). */
  done: Promise<JobState>;
}

const watches = new Map<string, JobWatch>();

export const jobRunning = (job: JobState | null | undefined) =>
  !!job && job.status !== 'completed' && job.status !== 'failed';

export function watchJob(projectId: string, jobId: string): JobWatch {
  const known = watches.get(jobId);
  if (known) return known;

  const listeners = new Set<(job: JobState) => void>();
  let state: JobState = { id: jobId, status: 'pending', progress: 0, message: 'queued', result: null, error: null };
  let resolve: (job: JobState) => void = () => {};
  const done = new Promise<JobState>((r) => { resolve = r; });
  let misses = 0;

  const tick = async () => {
    try {
      const j = await getProcessJob(projectId, jobId);
      misses = 0;
      state = {
        id: jobId,
        status: j.status || 'running',
        progress: typeof j.progress === 'number' && Number.isFinite(j.progress) ? j.progress : 0,
        message: j.message,
        result: j.result ?? null,
        error: j.error ?? null,
      };
    } catch (err) {
      const gone = err instanceof AudioApiError && (err.status === 404 || err.status === 405);
      misses += 1;
      if (!gone && misses < MAX_MISSES) {
        window.setTimeout(() => void tick(), POLL_MS * 2);
        return;
      }
      state = { ...state, status: 'failed', error: `Lost track of the job: ${errorText(err)}` };
    }
    listeners.forEach((fn) => fn(state));
    if (!jobRunning(state)) {
      watches.delete(jobId);
      resolve(state);
      return;
    }
    window.setTimeout(() => void tick(), POLL_MS);
  };

  const watch: JobWatch = {
    id: jobId,
    projectId,
    current: () => state,
    subscribe: (fn) => {
      listeners.add(fn);
      return () => { listeners.delete(fn); };
    },
    done,
  };
  watches.set(jobId, watch);
  window.setTimeout(() => void tick(), POLL_MS);
  return watch;
}

/** Follow a watch's state in a component. */
export function useJob(watch: JobWatch | null): JobState | null {
  const [job, setJob] = useState<JobState | null>(() => watch?.current() ?? null);
  useEffect(() => {
    if (!watch) {
      setJob(null);
      return undefined;
    }
    setJob(watch.current());
    return watch.subscribe(setJob);
  }, [watch]);
  return job;
}

/** The bar while the job runs, its error when it failed; nothing once it completed (the tab
 *  shows the result). */
export default function JobProgress({ watch, label }: { watch: JobWatch | null; label?: string }) {
  const job = useJob(watch);
  if (!job) return null;
  if (job.status === 'failed') {
    return (
      <div className="audio-note">
        {label ? `${label} failed` : 'Failed'}: {job.error || job.message || 'unknown error'}
      </div>
    );
  }
  if (job.status === 'completed') return null;
  const percent = Math.round(Math.min(1, Math.max(0, job.progress)) * 100);
  return (
    <div className="audio-progress voice-progress" title={job.message || job.status}>
      <div className="audio-progress-fill" style={{ width: `${percent}%` }} />
      <span className="audio-progress-text">
        {label ? `${label} · ` : ''}{job.message || job.status} · {percent}%
      </span>
    </div>
  );
}
