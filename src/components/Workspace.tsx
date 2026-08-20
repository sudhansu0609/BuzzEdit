import { useCallback, useEffect, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { useLayoutStore } from '../hooks/layout';
import { useCommand } from '../hooks/commands';
import { transcribeProject, analyzeProject, renderProject, autoEditProject, getProject, getTranscript, getSegments, getTimeline, autoSaveProject, getRenderStatus } from '../hooks/api';
import PreviewPlayer from './PreviewPlayer';
import Timeline from './Timeline';
import TranscriptEditor from './TranscriptEditor';
import ExportSettings from './ExportSettings';
import MediaPool from './MediaPool';
import InspectorPanel from './InspectorPanel';
import LlmIndicator from './LlmIndicator';
import RenderCompleteModal, { RenderResult } from './RenderCompleteModal';
import DockLayout from './DockLayout';

import AgentPanel from './AgentPanel';
import JobQueue from './JobQueue';

export default function Workspace() {
  const { project, setProject, updateProject, isProcessing, setProcessing, setProgress, setError } = useProjectStore();
  const [renderResult, setRenderResult] = useState<RenderResult | null>(null);
  const { toggleVisible, panels } = useLayoutStore();

  useEffect(() => {
    if (project?.id) {
      loadProjectData(project.id);
    }
  }, [project?.id]);

  const loadProjectData = async (projectId: string) => {
    try {
      const [projectData, transcriptData, segmentsData, timelineData] = await Promise.allSettled([
        getProject(projectId),
        getTranscript(projectId).catch(() => null),
        getSegments(projectId).catch(() => null),
        getTimeline(projectId).catch(() => null),
      ]);

      if (projectData.status === 'fulfilled' && projectData.value) {
        setProject(projectData.value);
      }
      if (transcriptData.status === 'fulfilled' && transcriptData.value) {
        updateProject({ transcript: transcriptData.value.segments });
      }
      if (segmentsData.status === 'fulfilled' && segmentsData.value) {
        updateProject({ detectedSegments: segmentsData.value.segments });
      }
      if (timelineData.status === 'fulfilled' && timelineData.value) {
        updateProject({ timeline: timelineData.value });
      }
    } catch (err: any) {
      console.error('Failed to load project data:', err);
    }
  };

  const handleTranscribe = useCallback(async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Transcribing audio & building EDL...');
    try {
      await transcribeProject(project.id);
      setProgress(1, 'Transcription complete');
      await loadProjectData(project.id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, [project, setProcessing, setProgress, setError]);

  const handleAnalyze = useCallback(async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Analyzing segments...');
    try {
      await analyzeProject(project.id);
      setProgress(1, 'Analysis complete');
      await loadProjectData(project.id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, [project, setProcessing, setProgress, setError]);

  const handleRender = useCallback(async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Rendering video single-pass...');
    try {
      const res: any = await renderProject(project.id);
      setProgress(1, 'Render complete');
      await loadProjectData(project.id);
      if (res?.output_path) setRenderResult({ outputPath: res.output_path });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, [project, setProcessing, setProgress, setError]);

  const aggressiveness = project?.settings?.fumble_aggressiveness ?? 0.5;
  const maxPause = project?.settings?.max_pause_seconds ?? 0.4;

  const saveSetting = useCallback((key: string, value: number) => {
    if (!project) return;
    const newSettings = { ...project.settings, [key]: value };
    updateProject({ settings: newSettings });
    // Persist so the backend auto-edit reads the latest value.
    autoSaveProject(project.id, { settings: newSettings }).catch(() => {});
  }, [project, updateProject]);

  const handleAggressivenessChange = useCallback(
    (value: number) => saveSetting('fumble_aggressiveness', value), [saveSetting]);

  const handleAutoEdit = useCallback(async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Starting auto-edit…');
    try {
      // Make sure the current aggressiveness is on disk before the backend reads it.
      await autoSaveProject(project.id, { settings: project.settings }).catch(() => {});

      // Auto-edit runs as a background job (LLM fluency pass + full re-render can
      // take minutes). Poll it so the progress bar and phase message actually
      // move instead of sitting at 0% until the whole thing finishes.
      const start: any = await autoEditProject(project.id);
      const jobId: string | undefined = start?.job_id;

      let result: any = start;
      if (jobId) {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          await new Promise((r) => setTimeout(r, 1000));
          const job: any = await getRenderStatus(jobId);
          setProgress(job?.progress ?? 0, job?.message || 'Auto-editing…');
          if (job?.status === 'completed') { result = job.result ?? {}; break; }
          if (job?.status === 'failed') {
            throw new Error(job?.error || 'Auto-edit failed on the backend');
          }
        }
      }

      const removed = result?.words_removed;
      const total = result?.words_total;
      const recovered = result?.report?.seconds_recovered;
      setProgress(1, recovered != null
        ? `Auto-edit complete — ${recovered.toFixed(1)}s removed`
        : removed != null ? `Auto-edit complete — removed ${removed}/${total} words` : 'Auto-edit complete');
      await loadProjectData(project.id);
      const outputPath = result?.output_path;
      if (outputPath) {
        setRenderResult({ outputPath, wordsRemoved: removed, wordsTotal: total, report: result?.report ?? null });
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, [project, setProcessing, setProgress, setError, updateProject]);

  // Expose the pipeline actions to the menu bar and keyboard shortcuts.
  useCommand('edit.transcribe', handleTranscribe);
  useCommand('edit.analyze', handleAnalyze);
  useCommand('file.render', handleRender);
  useCommand('edit.autoEdit', handleAutoEdit);

  return (
    <div className="workspace">
      <DockLayout panels={{
        preview: (
          <div className="preview-panel">
            <PreviewPlayer />
          <div className="toolbar">
            <button
              className="btn btn-primary"
              disabled={isProcessing}
              onClick={handleTranscribe}
            >
              Transcribe
            </button>
            <button
              className="btn btn-primary"
              disabled={isProcessing}
              onClick={handleAnalyze}
            >
              Analyze
            </button>
            <button
              className="btn btn-success"
              disabled={isProcessing}
              onClick={handleRender}
            >
              Render
            </button>
            <button
              className="btn btn-primary"
              disabled={isProcessing}
              onClick={handleAutoEdit}
            >
              Auto Edit
            </button>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6 }} title="How aggressively to remove fillers & fumbles">
              <span className="text-xs text-muted">Fumbles</span>
              <input
                type="range" min={0} max={1} step={0.05}
                value={aggressiveness}
                onChange={(e) => handleAggressivenessChange(parseFloat(e.target.value))}
                style={{ width: 90 }}
              />
              <span className="text-xs" style={{ width: 28 }}>{Math.round(aggressiveness * 100)}%</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              title="How long a pause may be before it gets trimmed. Lower is a tighter, more jump-cut edit; higher leaves the delivery breathing.">
              <span className="text-xs text-muted">Pauses</span>
              <input
                type="range" min={0.15} max={1.2} step={0.05}
                value={maxPause}
                onChange={(e) => saveSetting('max_pause_seconds', parseFloat(e.target.value))}
                style={{ width: 90 }}
              />
              <span className="text-xs" style={{ width: 34 }}>{maxPause.toFixed(2)}s</span>
            </label>
            <LlmIndicator />
            <div className="toolbar-spacer" />
            {/* Quick access to the two panels this toolbar's work feeds into.
                Everything else lives in the Layout menu in the header. */}
            <button className="btn btn-sm"
              onClick={() => toggleVisible('transcript', !panels.transcript.visible)}>
              {panels.transcript.visible ? 'Hide' : 'Show'} Transcript
            </button>
            <button className="btn btn-sm"
              onClick={() => toggleVisible('inspector', !panels.inspector.visible)}>
              {panels.inspector.visible ? 'Hide' : 'Show'} Inspector
            </button>
            </div>
          </div>
        ),
        // Every panel is its own dockable window now. The dock decides whether
        // they sit side by side or share a zone as tabs, so nothing here needs
        // to know about the arrangement.
        media: <div className="side-panel docked"><div className="panel-content"><MediaPool /></div></div>,
        transcript: <div className="side-panel docked"><div className="panel-content"><TranscriptEditor /></div></div>,
        inspector: <div className="side-panel docked"><div className="panel-content"><InspectorPanel /></div></div>,
        agents: <div className="side-panel docked"><div className="panel-content"><AgentPanel /></div></div>,
        queue: <div className="side-panel docked"><div className="panel-content"><JobQueue /></div></div>,
        export: <div className="side-panel docked"><div className="panel-content"><ExportSettings /></div></div>,
        timeline: <Timeline />,
      }} />

      {renderResult && (
        <RenderCompleteModal result={renderResult} onClose={() => setRenderResult(null)} />
      )}
    </div>
  );
}
