import { useEffect, useRef, useCallback } from 'react';
import { useProjectStore, Project } from './store';
import { API_BASE, autoSaveProject, getRecoveryState, getProject } from './api';

const AUTO_SAVE_INTERVAL_MS = 30000;

export function useAutoSave() {
  const project = useProjectStore((s) => s.project);
  const setProject = useProjectStore((s) => s.setProject);
  const setError = useProjectStore((s) => s.setError);
  const lastSaveRef = useRef<string | null>(null);
  const intervalRef = useRef<number | null>(null);

  const saveProject = useCallback(async () => {
    if (!project?.id) return;

    const projectJson = JSON.stringify(project);
    if (projectJson === lastSaveRef.current) {
      return;
    }

    try {
      await autoSaveProject(project.id, {
        clips: project.clips,
        transcript: project.transcript,
        detectedSegments: project.detectedSegments,
        settings: project.settings,
        status: project.status,
        timeline: project.timeline,
      });
      lastSaveRef.current = projectJson;
    } catch (err) {
      console.warn('Auto-save failed:', err);
    }
  }, [project]);

  useEffect(() => {
    if (!project?.id) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    intervalRef.current = window.setInterval(saveProject, AUTO_SAVE_INTERVAL_MS);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [project?.id, saveProject]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      if (project?.id) {
        const projectData = {
          clips: project.clips,
          transcript: project.transcript,
          detectedSegments: project.detectedSegments,
          settings: project.settings,
          status: project.status,
          timeline: project.timeline,
        };
        navigator.sendBeacon(
          `${API_BASE}/api/settings/auto_save`,
          JSON.stringify({ project_id: project.id, project_data: projectData })
        );
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [project]);

  return { saveProject };
}

export function useSessionRecovery() {
  const setProject = useProjectStore((s) => s.setProject);
  const setError = useProjectStore((s) => s.setError);
  const project = useProjectStore((s) => s.project);

  const recoverSession = useCallback(async () => {
    if (project) return;

    try {
      const state = await getRecoveryState();

      if (state.has_recovery && state.recovery_project) {
        const recovered: Project = {
          id: state.recovery_project.id,
          name: state.recovery_project.name,
          sourceVideo: state.recovery_project.source_video,
          source_video: state.recovery_project.source_video,
          timeline: state.recovery_project.timeline,
          clips: state.recovery_project.clips || [],
          transcript: state.recovery_project.transcript,
          detectedSegments: state.recovery_project.detectedSegments || [],
          outputPath: state.recovery_project.outputPath,
          status: state.recovery_project.status || 'draft',
          settings: state.recovery_project.settings || {},
        };
        setProject(recovered);
        return { recovered: true, project: recovered };
      }

      return { recovered: false, recentProjects: state.recent_projects };
    } catch (err: any) {
      console.warn('Session recovery failed:', err);
      return { recovered: false, error: err.message };
    }
  }, [project, setProject]);

  return { recoverSession };
}
