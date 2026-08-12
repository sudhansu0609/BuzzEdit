import { useState, useEffect, useCallback } from 'react';
import { useProjectStore } from '../hooks/store';
import {
  getSchedulerStatus,
  enqueueJob,
  cancelSchedulerJob,
  clearCompletedJobs,
  pauseScheduler,
  resumeScheduler,
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
      await enqueueJob(project.id, selectedPriority);
      await fetchStatus();
      setProgress(1, 'Project enqueued for overnight execution');
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
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <select
              value={selectedPriority}
              onChange={(e: any) => setSelectedPriority(e.target.value)}
              style={{ padding: '8px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
            >
              <option value="urgent">🔥 Urgent Priority</option>
              <option value="normal">⚡ Normal Priority</option>
              <option value="low">🌙 Low Priority</option>
            </select>
            <button className="btn btn-primary" style={{ flex: 1 }} onClick={handleEnqueueCurrent}>
              + Enqueue Overnight Job
            </button>
          </div>
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
                <div style={{ height: '4px', background: '#313244', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{ width: '100%', height: '100%', background: '#3b82f6', animation: 'pulse 1.5s infinite' }} />
                </div>
              )}

              {job.error && (
                <span style={{ fontSize: '11px', color: '#ef4444' }}>Error: {job.error}</span>
              )}
              {job.result?.output_directory && (
                <span style={{ fontSize: '11px', color: '#10b981' }}>Saved: {job.result.output_directory}</span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
