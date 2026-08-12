import { useCallback, useEffect, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { transcribeProject, analyzeProject, renderProject, autoEditProject, getProject, getTranscript, getSegments, getTimeline } from '../hooks/api';
import PreviewPlayer from './PreviewPlayer';
import Timeline from './Timeline';
import TranscriptEditor from './TranscriptEditor';
import ExportSettings from './ExportSettings';

import AgentPanel from './AgentPanel';
import JobQueue from './JobQueue';

export default function Workspace() {
  const { project, setProject, updateProject, setShowTranscript, showTranscript, showSegments, setShowSegments, isProcessing, setProcessing, setProgress, setError } = useProjectStore();
  const [activeTab, setActiveTab] = useState<'transcript' | 'agents' | 'queue' | 'export'>('transcript');

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
      await renderProject(project.id);
      setProgress(1, 'Render complete');
      await loadProjectData(project.id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, [project, setProcessing, setProgress, setError]);

  const handleAutoEdit = useCallback(async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Auto-editing...');
    try {
      await autoEditProject(project.id);
      setProgress(1, 'Auto-edit complete');
      await loadProjectData(project.id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, [project, setProcessing, setProgress, setError]);

  return (
    <div className="workspace">
      <div className="workspace-main">
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
            <div className="toolbar-spacer" />
            <button className="btn btn-sm" onClick={() => setShowTranscript(!showTranscript)}>
              {showTranscript ? 'Hide' : 'Show'} Panel
            </button>
            <button className="btn btn-sm" onClick={() => setShowSegments(!showSegments)}>
              {showSegments ? 'Hide' : 'Show'} Segments
            </button>
          </div>
        </div>

        <div className={`side-panel ${showTranscript ? 'visible' : 'hidden'}`}>
          <div className="panel-tabs">
            <button
              className={`tab ${activeTab === 'transcript' ? 'active' : ''}`}
              onClick={() => setActiveTab('transcript')}
            >
              Transcript
            </button>
            <button
              className={`tab ${activeTab === 'agents' ? 'active' : ''}`}
              onClick={() => setActiveTab('agents')}
            >
              AI Agents
            </button>
            <button
              className={`tab ${activeTab === 'queue' ? 'active' : ''}`}
              onClick={() => setActiveTab('queue')}
            >
              Nightly Queue
            </button>
            <button
              className={`tab ${activeTab === 'export' ? 'active' : ''}`}
              onClick={() => setActiveTab('export')}
            >
              Export
            </button>
          </div>
          <div className="panel-content">
            {activeTab === 'transcript' && <TranscriptEditor />}
            {activeTab === 'agents' && <AgentPanel />}
            {activeTab === 'queue' && <JobQueue />}
            {activeTab === 'export' && <ExportSettings />}
          </div>
        </div>
      </div>

      <Timeline />
    </div>
  );
}
