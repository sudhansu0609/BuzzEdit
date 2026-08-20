import { useCallback, useState, useRef, useEffect } from 'react';
import { listProjects } from '../hooks/api';

interface Props {
  onFile: (file: File) => void;
  onPick: () => void;
  onOpenProject: (projectId: string) => void;
}

interface RecentProject {
  id: string;
  name: string;
  status: string;
}

export default function UploadScreen({ onFile, onPick, onOpenProject }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [recent, setRecent] = useState<RecentProject[]>([]);

  useEffect(() => {
    let cancelled = false;
    listProjects()
      .then((res) => {
        if (!cancelled) setRecent((res.projects || []).slice(0, 8));
      })
      .catch((err) => console.warn('Failed to load recent projects:', err));
    return () => { cancelled = true; };
  }, []);

  const handleClick = useCallback(() => {
    if (window.electronAPI) {
      onPick();
    } else {
      fileInputRef.current?.click();
    }
  }, [onPick]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  }, [onFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
  }, [onFile]);

  return (
    <div className="upload-screen">
      <div className="upload-container">
        <div className="upload-icon">
          <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
            <rect x="8" y="8" width="64" height="64" rx="12" stroke="#3b82f6" strokeWidth="2" fill="none" />
            <path d="M30 32l16 10-16 10V32z" fill="#3b82f6" />
          </svg>
        </div>
        <h1 className="upload-title">BuzzEdit</h1>
        <p className="upload-subtitle">AI-powered video editing — remove fumbles, silence, and b-roll automatically</p>

        <div
          className={`upload-dropzone ${dragOver ? 'drag-over' : ''}`}
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setDragOver(true); }}
          onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setDragOver(false); }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleChange}
            style={{ display: 'none' }}
          />
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
            <path d="M20 8v20M12 16l8-8 8 8" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <p>Drop video here or click to browse</p>
          <p className="upload-formats">MP4, MOV, AVI, MKV, WebM</p>
        </div>

        {recent.length > 0 && (
          <div className="recent-projects" style={{ marginTop: 28, width: '100%', maxWidth: 520 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <span className="text-sm font-weight-semibold">Recent Projects</span>
              <span className="text-xs text-muted">Reopen without re-importing</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {recent.map((p) => (
                <button
                  key={p.id}
                  className="recent-project-item"
                  onClick={() => onOpenProject(p.id)}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '10px 14px', borderRadius: 8, cursor: 'pointer',
                    background: 'var(--surface-2, #1a1a1a)', border: '1px solid var(--border, #2a2a2a)',
                    color: 'inherit', textAlign: 'left', width: '100%',
                  }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    🎬 {p.name}
                  </span>
                  <span className={`badge badge-${p.status}`} style={{ flexShrink: 0, marginLeft: 12 }}>
                    {p.status}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="upload-features">
          <div className="feature">
            <span className="feature-icon">🎙️</span>
            <span>Auto-transcribe with Whisper</span>
          </div>
          <div className="feature">
            <span className="feature-icon">✂️</span>
            <span>Remove silence & fumbles</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🎬</span>
            <span>Smart transitions</span>
          </div>
        </div>
      </div>
    </div>
  );
}
