import { useCallback, useEffect, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { getSchedulerStatus, getProject } from '../hooks/api';

// Same cadence JobQueue.tsx already polls the scheduler at.
const POLL_MS = 3000;

/**
 * Follows whatever the scheduler is rendering right now, even when it is not
 * the project open in this window — which is the normal case for an overnight
 * BuzzcafStudio job while the editor is being used for something else.
 *
 * No project open at all: the render is opened automatically rather than
 * leaving the user on the start screen. A project already open that differs
 * from the render: a dismissible-by-navigation banner with an Open button, so
 * switching to watch it is one click, not a hunt through Projects.
 */
export default function StudioJobBanner() {
  const { project, setProject } = useProjectStore();
  const [job, setJob] = useState<any | null>(null);
  const [jobProjectName, setJobProjectName] = useState<string | null>(null);
  const autoOpenedFor = useRef<string | null>(null);
  const nameCache = useRef<Map<string, string>>(new Map());

  const poll = useCallback(async () => {
    try {
      const status = await getSchedulerStatus();
      const currentId = status?.current_job_id;
      const current = currentId
        ? (status.jobs || []).find((j: any) => j.id === currentId)
        : null;
      setJob(current && current.status === 'running' ? current : null);
    } catch {
      // Scheduler unreachable this tick — keep the last known state and retry.
    }
  }, []);

  useEffect(() => {
    poll();
    const interval = setInterval(poll, POLL_MS);
    return () => clearInterval(interval);
  }, [poll]);

  const otherProjectId = job && job.project_id !== project?.id ? job.project_id as string : null;

  // Nothing open here: follow the render rather than sit idle on the start screen.
  useEffect(() => {
    if (!otherProjectId || project) return;
    if (autoOpenedFor.current === otherProjectId) return;
    autoOpenedFor.current = otherProjectId;
    getProject(otherProjectId).then(setProject).catch(() => {
      autoOpenedFor.current = null;
    });
  }, [otherProjectId, project, setProject]);

  // Resolve the render's project name for the banner text (cached — the id
  // does not change while one job runs, so this fires once per job).
  useEffect(() => {
    if (!otherProjectId) { setJobProjectName(null); return; }
    const cached = nameCache.current.get(otherProjectId);
    if (cached) { setJobProjectName(cached); return; }
    getProject(otherProjectId)
      .then((p) => {
        const name = p?.name || otherProjectId;
        nameCache.current.set(otherProjectId, name);
        setJobProjectName(name);
      })
      .catch(() => setJobProjectName(otherProjectId));
  }, [otherProjectId]);

  const openRenderedProject = useCallback(() => {
    if (!otherProjectId) return;
    getProject(otherProjectId).then(setProject).catch(() => {});
  }, [otherProjectId, setProject]);

  // The banner itself only makes sense once a *different* project is already
  // open — with nothing open, the effect above opens the render instead.
  if (!otherProjectId || !project) return null;

  const pct = Math.round((job?.progress ?? 0) * 100);

  return (
    <div className="studio-job-banner">
      <span className="studio-job-banner-dot" />
      <span className="studio-job-banner-text">
        Rendering <strong>{jobProjectName || otherProjectId}</strong> for BuzzcafStudio — {pct}%
      </span>
      <button className="btn btn-sm btn-primary" onClick={openRenderedProject}>Open</button>
    </div>
  );
}
