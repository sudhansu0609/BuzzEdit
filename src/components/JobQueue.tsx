import { useState, useEffect, useCallback } from 'react';
import { useProjectStore } from '../hooks/store';
import {
  getSchedulerStatus,
  enqueueJob,
  cancelSchedulerJob,
  clearCompletedJobs,
  pauseScheduler,
  resumeScheduler,
  getComfyUIWorkflows,
  updateAppSettings,
  type ComfyWorkflowList,
} from '../hooks/api';

export default function JobQueue() {
  const { project, setProcessing, setProgress, setError } = useProjectStore();
  const [queueState, setQueueState] = useState<{
    is_running: boolean;
    is_paused: boolean;
    current_job_id: string | null;
    jobs: any[];
  }>({
    is_running: true,
    is_paused: false,
    current_job_id: null,
    jobs: [],
  });

  const [selectedPriority, setSelectedPriority] = useState<'urgent' | 'normal' | 'low'>('normal');
  const [jobType, setJobType] = useState<'presentation' | 'full_edit'>('presentation');
  // Empty means "start as soon as the queue reaches it". A time means the next
  // occurrence of it, so the GPU work happens while nobody is using the machine.
  const [startAt, setStartAt] = useState('');
  const [showOptions, setShowOptions] = useState(false);
  const [passOptions, setPassOptions] = useState({
    broll: true, broll_video: true, popups: true,
    face_zoom: true, captions: true, thumbnail: true,
  });
  // Which ComfyUI workflow serves each generation role. Fetched lazily the
  // first time the options panel is opened; picking one writes the app setting.
  const [workflows, setWorkflows] = useState<ComfyWorkflowList | null>(null);
  const [savingRole, setSavingRole] = useState<string | null>(null);

  const loadWorkflows = useCallback(async () => {
    try {
      setWorkflows(await getComfyUIWorkflows());
    } catch (err) {
      console.error('Failed to load ComfyUI workflows:', err);
    }
  }, []);

  const handlePickWorkflow = async (role: string, file: string) => {
    if (!workflows) return;
    const key = workflows.settings_key[role];
    if (!key) return;
    setSavingRole(role);
    try {
      // Empty selection clears the setting (role falls back to manifest/default,
      // or nothing at all for video/graphic which have no bundled default).
      await updateAppSettings({ [key]: file || null });
      setWorkflows({
        ...workflows,
        selection: { ...workflows.selection, [role]: file || null },
      });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSavingRole(null);
    }
  };

  const fetchStatus = useCallback(async () => {
    try {
      const res = await getSchedulerStatus();
      setQueueState(res);
    } catch (err: any) {
      console.error('Failed to fetch scheduler queue status:', err);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleEnqueueCurrent = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Enqueueing project for nightly batch execution...');
    try {
      await enqueueJob(project.id, selectedPriority, {
        job_type: jobType,
        start_at: startAt || null,
        settings: jobType === 'presentation' ? passOptions : {},
      });
      await fetchStatus();
      setProgress(1, startAt
        ? `Queued — starts at ${startAt}`
        : 'Project enqueued for overnight execution');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      await cancelSchedulerJob(jobId);
      await fetchStatus();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleClearCompleted = async () => {
    try {
      await clearCompletedJobs();
      await fetchStatus();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const togglePause = async () => {
    try {
      if (queueState.is_paused) {
        await resumeScheduler();
      } else {
        await pauseScheduler();
      }
      await fetchStatus();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return '#3b82f6';
      case 'completed': return '#10b981';
      case 'failed': return '#ef4444';
      case 'cancelled': return '#6b7280';
      default: return '#f59e0b';
    }
  };

  return (
    <div className="job-queue-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px', color: '#fff' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px', background: '#181825', borderRadius: '8px' }}>
        <div>
          <h4 style={{ margin: 0, fontSize: '15px' }}>🌙 Nightly Queue Engine</h4>
          <span style={{ fontSize: '12px', color: '#a6adc8' }}>
            {queueState.jobs.length} jobs in queue | Prevent Sleep Active
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-sm" onClick={togglePause}>
            {queueState.is_paused ? '▶ Resume Queue' : '⏸ Pause Queue'}
          </button>
          <button className="btn btn-sm btn-secondary" onClick={handleClearCompleted}>
            🧹 Clear Finished
          </button>
        </div>
      </div>

      {project && (
        <div style={{ background: '#1e1e2e', padding: '12px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <span style={{ fontSize: '13px', fontWeight: 'bold' }}>Queue Current Project:</span>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            <select
              value={jobType}
              onChange={(e: any) => setJobType(e.target.value)}
              title="A presentation pass generates B-roll, zooms, pop-ups and captions. A full edit just cuts and renders."
              style={{ padding: '8px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
            >
              <option value="presentation">🎬 Presentation pass</option>
              <option value="full_edit">✂ Full edit only</option>
            </select>
            <select
              value={selectedPriority}
              onChange={(e: any) => setSelectedPriority(e.target.value)}
              style={{ padding: '8px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
            >
              <option value="urgent">🔥 Urgent Priority</option>
              <option value="normal">⚡ Normal Priority</option>
              <option value="low">🌙 Low Priority</option>
            </select>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}
              title="Leave empty to start straight away. Set a time and the machine stays idle until then.">
              Start at
              <input
                type="time"
                value={startAt}
                onChange={(e) => setStartAt(e.target.value)}
                style={{ padding: '7px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
              />
            </label>
            {jobType === 'presentation' && (
              <button className="btn btn-sm" onClick={() => {
                setShowOptions(v => !v);
                if (!workflows) loadWorkflows();
              }}>
                {showOptions ? '▾' : '▸'} Options
              </button>
            )}
            <button className="btn btn-primary" style={{ flex: 1, minWidth: 160 }}
              onClick={handleEnqueueCurrent}>
              + Enqueue Overnight Job
            </button>
          </div>

          {jobType === 'presentation' && showOptions && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px 16px', fontSize: 12 }}>
              {([
                ['broll', 'B-roll cutaways'],
                ['broll_video', 'Generated video'],
                ['popups', 'Topic pop-ups'],
                ['face_zoom', 'Punch-ins on my face'],
                ['captions', 'Captions'],
                ['thumbnail', 'Thumbnail'],
              ] as [keyof typeof passOptions, string][]).map(([key, label]) => (
                <label key={key} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <input
                    type="checkbox"
                    checked={passOptions[key]}
                    onChange={(e) => setPassOptions(o => ({ ...o, [key]: e.target.checked }))}
                  />
                  {label}
                </label>
              ))}
            </div>
          )}

          {jobType === 'presentation' && showOptions && (
            <div style={{ borderTop: '1px solid #313244', paddingTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 'bold', color: '#a6adc8' }}>
                Generation workflows
              </span>
              {!workflows ? (
                <span style={{ fontSize: 11, color: '#6c7086' }}>Loading ComfyUI workflows…</span>
              ) : workflows.workflows.length === 0 ? (
                <span style={{ fontSize: 11, color: '#f38ba8' }}>
                  No workflow files found in the <code>workflows/</code> folder.
                </span>
              ) : (
                ([
                  ['broll_image', 'Images (B-roll)', false],
                  ['broll_video', 'Video (B-roll)', true],
                  ['thumbnail', 'Thumbnail', false],
                ] as [string, string, boolean][]).map(([role, label, allowNone]) => {
                  const options = workflows.workflows.filter(w => w.valid && w.roles_ok?.[role]);
                  const current = workflows.selection[role] ?? '';
                  return (
                    <label key={role} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                      <span style={{ minWidth: 120 }}>{label}</span>
                      <select
                        value={current}
                        disabled={savingRole === role}
                        onChange={(e) => handlePickWorkflow(role, e.target.value)}
                        style={{ flex: 1, padding: '6px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
                      >
                        {allowNone && <option value="">— none (skip) —</option>}
                        {options.length === 0 && !allowNone && (
                          <option value="" disabled>No compatible workflow</option>
                        )}
                        {options.map(w => (
                          <option key={w.file} value={w.file}>{w.file}</option>
                        ))}
                      </select>
                    </label>
                  );
                })
              )}
              <span style={{ fontSize: 10, color: '#6c7086' }}>
                Drop a workflow <code>.json</code> into the <code>workflows/</code> folder and it
                appears here — bindings are detected automatically.
              </span>
            </div>
          )}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '400px', overflowY: 'auto' }}>
        {queueState.jobs.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#a6adc8', background: '#181825', borderRadius: '8px' }}>
            No jobs in nightly queue. Add videos before sleep!
          </div>
        ) : (
          queueState.jobs.map((job) => (
            <div key={job.id} style={{ background: '#181825', padding: '12px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 'bold', fontSize: '13px' }}>{job.project_id}</span>
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: '#313244', textTransform: 'uppercase' }}>
                    {job.priority}
                  </span>
                  <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '10px', background: getStatusColor(job.status), color: '#fff', fontWeight: 'bold' }}>
                    {job.status}
                  </span>
                  {job.status === 'pending' && (
                    <button className="btn btn-sm btn-danger" style={{ padding: '2px 6px' }} onClick={() => handleCancel(job.id)}>✕</button>
                  )}
                </div>
              </div>

              {job.status === 'running' && (
                <>
                  {/* A real bar now that stages report where they have got to.
                      It used to pulse indeterminately for the whole run. */}
                  <div style={{ height: '4px', background: '#313244', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.round((job.progress ?? 0) * 100)}%`,
                      height: '100%', background: '#3b82f6', transition: 'width 0.4s',
                    }} />
                  </div>
                  {job.message && (
                    <span style={{ fontSize: '11px', color: '#a6adc8' }}>
                      {job.message} · {Math.round((job.progress ?? 0) * 100)}%
                    </span>
                  )}
                </>
              )}

              {job.status === 'pending' && job.start_at && (
                <span style={{ fontSize: '11px', color: '#a6adc8' }}>
                  ⏰ Starts {new Date(job.start_at).toLocaleString()}
                </span>
              )}

              {job.error && (
                <span style={{ fontSize: '11px', color: '#ef4444' }}>Error: {job.error}</span>
              )}
              {job.result?.output_directory && (
                <span style={{ fontSize: '11px', color: '#10b981' }}>Saved: {job.result.output_directory}</span>
              )}
              {job.result?.report && (
                <span style={{ fontSize: '11px', color: '#a6adc8' }}>
                  {job.result.report.broll_placed} cutaways ·{' '}
                  {job.result.report.zooms_segment + job.result.report.zooms_windowed} zooms ·{' '}
                  {job.result.report.popups_placed} pop-ups ·{' '}
                  {job.result.report.captions} captions
                  {job.result.report.degraded?.length > 0
                    ? ` · degraded: ${job.result.report.degraded.join(', ')}` : ''}
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
