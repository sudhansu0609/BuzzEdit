import { useState, useCallback, useRef } from 'react';
import { useProjectStore } from './hooks/store';
import { uploadVideo, importVideoPath } from './hooks/api';
import { useElectron } from './hooks/api';
import Header from './components/Header';
import UploadScreen from './components/UploadScreen';
import Workspace from './components/Workspace';
import ProcessingOverlay from './components/ProcessingOverlay';

function App() {
  const { project, setProject, setProcessing, setProgress, setError, reset } = useProjectStore();
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
    e.preventDefault();
    dragCounter.current += 1;
    if (dragCounter.current === 1) {
      setDragOver(true);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setDragOver(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
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

  const handleReset = useCallback(() => {
    reset();
  }, [reset]);

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
        <UploadScreen onFile={handleFile} onPick={handlePickFile} />
      ) : (
        <Workspace />
      )}
      <ProcessingOverlay />
    </div>
  );
}

export default App;
