import { useEffect, useState, useCallback } from 'react';
import { useProjectStore } from './store';

declare global {
  interface Window {
    electronAPI?: {
      dialogOpenFile: (opts: any) => Promise<any>;
      dialogSaveFile: (opts: any) => Promise<any>;
      windowMinimize: () => Promise<void>;
      windowMaximize: () => Promise<void>;
      windowClose: () => Promise<void>;
      getApiUrl: () => Promise<string>;
    };
  }
}

const API_BASE = 'http://localhost:8099';

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export async function uploadVideo(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/projects/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function importVideoPath(filePath: string): Promise<any> {
  return api('/api/projects/import_path', {
    method: 'POST',
    body: JSON.stringify({ path: filePath }),
  });
}

export async function transcribeProject(projectId: string): Promise<any> {
  return api('/api/transcription/transcribe', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, model: 'large-v3', language: 'en' }),
  });
}

export async function analyzeProject(projectId: string): Promise<any> {
  return api('/api/analysis/analyze', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, remove_silence: true, remove_fumbles: true }),
  });
}

export async function renderProject(projectId: string): Promise<any> {
  return api('/api/rendering/render', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function autoEditProject(projectId: string): Promise<any> {
  return api(`/api/rendering/${projectId}/auto_edit`, {
    method: 'POST',
  });
}

export async function getProject(projectId: string): Promise<any> {
  return api(`/api/projects/${projectId}`);
}

export async function getTranscript(projectId: string): Promise<any> {
  return api(`/api/transcription/${projectId}/transcript`);
}

export async function getSegments(projectId: string): Promise<any> {
  return api(`/api/analysis/${projectId}/segments`);
}

export async function getTimeline(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}`);
}

export async function generateTimeline(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/generate`, { method: 'POST' });
}

export async function toggleWordApi(projectId: string, wordId: string, enabled: boolean): Promise<any> {
  return api(`/api/timeline/${projectId}/toggle_word`, {
    method: 'POST',
    body: JSON.stringify({ word_id: wordId, enabled }),
  });
}

export async function getRenderStatus(jobId: string): Promise<any> {
  return api(`/api/rendering/status/${jobId}`);
}

export async function getTranscriptionStatus(jobId: string): Promise<any> {
  return api(`/api/transcription/status/${jobId}`);
}

export async function deleteProject(projectId: string): Promise<void> {
  await api(`/api/projects/${projectId}`, { method: 'DELETE' });
}

export async function updateSettings(projectId: string, settings: Record<string, any>): Promise<any> {
  return api(`/api/projects/${projectId}/settings`, {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}

export async function getComfyUIStatus(): Promise<any> {
  return api('/api/comfyui/status');
}

export async function triggerFullEdit(projectId: string, opts?: { generate_broll?: boolean; generate_thumbnail?: boolean; burn_captions?: boolean }): Promise<any> {
  return api('/api/agents/full_edit', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, ...opts }),
  });
}

export async function triggerBRoll(projectId: string): Promise<any> {
  return api('/api/agents/broll', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function triggerThumbnail(projectId: string, title?: string): Promise<any> {
  return api('/api/agents/thumbnail', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, title }),
  });
}

export async function triggerCaptions(projectId: string): Promise<any> {
  return api('/api/agents/captions', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function getSchedulerStatus(): Promise<any> {
  return api('/api/scheduler/jobs');
}

export async function enqueueJob(projectId: string, priority: string = 'normal', opts?: any): Promise<any> {
  return api('/api/scheduler/enqueue', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, priority, ...opts }),
  });
}

export async function cancelSchedulerJob(jobId: string): Promise<any> {
  return api(`/api/scheduler/jobs/${jobId}`, {
    method: 'DELETE',
  });
}

export async function clearCompletedJobs(): Promise<any> {
  return api('/api/scheduler/clear_completed', {
    method: 'POST',
  });
}

export async function pauseScheduler(): Promise<any> {
  return api('/api/scheduler/pause', {
    method: 'POST',
  });
}

export async function resumeScheduler(): Promise<any> {
  return api('/api/scheduler/resume', {
    method: 'POST',
  });
}

export async function getGpuStatus(): Promise<any> {
  return api('/api/system/gpu');
}

export async function getSystemPaths(): Promise<any> {
  return api('/api/system/paths');
}

export async function clearVram(): Promise<any> {
  return api('/api/system/clear_vram', {
    method: 'POST',
  });
}

export async function exportFilmoraXml(projectId: string): Promise<any> {
  return api('/api/advanced/export_filmora', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function applyColorGrading(projectId: string, preset: string): Promise<any> {
  return api('/api/advanced/color_grade', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, preset }),
  });
}

export async function convertToShorts(projectId: string): Promise<any> {
  return api('/api/advanced/shorts', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async <T>(fn: () => Promise<T>): Promise<T | null> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fn();
      return result;
    } catch (err: any) {
      setError(err.message || 'Unknown error');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { loading, error, execute, setError };
}

export function useElectron() {
  const pickFile = useCallback(async (filters?: any) => {
    if (window.electronAPI) {
      const result = await window.electronAPI.dialogOpenFile({
        properties: ['openFile'],
        filters: filters || [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] }],
      });
      return result.filePaths?.[0];
    }
    return null;
  }, []);

  const saveFile = useCallback(async (defaultPath?: string) => {
    if (window.electronAPI) {
      const result = await window.electronAPI.dialogSaveFile({
        defaultPath: defaultPath || 'output.mp4',
        filters: [{ name: 'Video', extensions: ['mp4'] }],
      });
      return result.filePath;
    }
    return null;
  }, []);

  return { pickFile, saveFile };
}
