import { useState, useCallback, useRef, useEffect } from 'react';
import { useProjectStore } from './hooks/store';
import { uploadVideo, importVideoPath, getProject } from './hooks/api';
import { useElectron } from './hooks/api';
import { useAutoSave, useSessionRecovery } from './hooks/useAutoSave';
import { useShortcuts } from './hooks/shortcuts';
import { useCommand } from './hooks/commands';
import Header from './components/Header';
import UploadScreen from './components/UploadScreen';
import Workspace from './components/Workspace';
import ProcessingOverlay from './components/ProcessingOverlay';
import { MEDIA_DND_TYPE } from './components/MediaPool';

// Internal drags (media-pool item -> timeline lane) must not trigger the
// full-screen file-drop overlay, or the lane never receives the drop.
const isInternalDrag = (e: React.DragEvent) =>
  Array.from(e.dataTransfer.types || []).includes(MEDIA_DND_TYPE);

function App() {
  const { project, setProject, setProcessing, setProgress, setError, reset } = useProjectStore();
  const [initializing, setInitializing] = useState(true);

  useAutoSave();
  useShortcuts();
  const { recoverSession } = useSessionRecovery();

  useEffect(() => {
    const init = async () => {
      // The backend is spawned around the same time the window opens, so the
      // first recovery attempt can hit a not-yet-ready server. Retry until the
      // backend responds (result has no `error`) instead of giving up after one
      // failure and stranding the user on the upload screen.
      const MAX_ATTEMPTS = 20;
      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        const result = await recoverSession();
        if (!result?.error) {
          if (result?.recovered) console.log('Session recovered:', result.project?.name);
          setInitializing(false);
          return;
        }
        // Backend not reachable yet — wait and retry.
        await new Promise((r) => setTimeout(r, 1000));
      }
      console.warn('Backend did not become reachable; showing upload screen.');
      setInitializing(false);
    };
    init();
  }, []);
  const { pickFile } = useElectron();
  const [dragOver, setDragOver] = useState(false);
  const dragCounter = useRef(0);

  const handleFile = useCallback(async (file: File) => {
    setProcessing(true);
    setProgress(0, 'Loading video...');
    try {
      let result: any;
      const localPath = (file as any).path;
      if (localPath && typeof localPath === 'string' && localPath.length > 0) {
        try {
          result = await importVideoPath(localPath);
        } catch {
          result = await uploadVideo(file);
        }
      } else {
        result = await uploadVideo(file);
      }
      setProject(result);
    } catch (err: any) {
      console.error('Upload failed:', err);
      setError(err.message || 'Failed to load video file');
    } finally {
      setProcessing(false);
    }
  }, [setProject, setProcessing, setProgress, setError]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    if (isInternalDrag(e)) return;
    e.preventDefault();
    dragCounter.current += 1;
    if (dragCounter.current === 1) {
      setDragOver(true);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    if (isInternalDrag(e)) return;
    e.preventDefault();
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (isInternalDrag(e)) return;
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setDragOver(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    if (isInternalDrag(e)) return;
    e.preventDefault();
    dragCounter.current = 0;
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) await handleFile(file);
  }, [handleFile]);

  const handlePickFile = useCallback(async () => {
    const filePath = await pickFile();
    if (!filePath) return;
    setProcessing(true);
    setProgress(0, 'Importing selected video...');
    try {
      const result = await importVideoPath(filePath);
      setProject(result);
    } catch (err: any) {
      console.error('File import failed:', err);
      setError(err.message || 'Failed to import video file');
    } finally {
      setProcessing(false);
    }
  }, [pickFile, setProject, setProcessing, setProgress, setError]);

  const handleOpenProject = useCallback(async (projectId: string) => {
    setProcessing(true);
    setProgress(0, 'Opening project...');
    try {
      const result = await getProject(projectId);
      setProject(result);
    } catch (err: any) {
      console.error('Open project failed:', err);
      setError(err.message || 'Failed to open project');
    } finally {
      setProcessing(false);
    }
  }, [setProject, setProcessing, setProgress, setError]);

  const handleReset = useCallback(() => {
    reset();
  }, [reset]);

  // File ▸ Open Project… returns to the start screen, which lists recent projects.
  useCommand('file.open', handleReset);

  if (initializing) {
    return (
      <div className="app" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: '#888' }}>
          <div style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Loading...</div>
          <div style={{ fontSize: '0.9rem' }}>Recovering last session</div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="app"
      onDrop={handleDrop}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      {dragOver && <div className="drop-overlay" />}
      <Header onReset={handleReset} />
      {!project ? (
        <UploadScreen onFile={handleFile} onPick={handlePickFile} onOpenProject={handleOpenProject} />
      ) : (
        <Workspace />
      )}
      <ProcessingOverlay />
    </div>
  );
}

export default App;
