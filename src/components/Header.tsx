import { useState } from 'react';
import { useProjectStore } from '../hooks/store';
import SetupWizard from './SetupWizard';

interface Props {
  onReset: () => void;
}

export default function Header({ onReset }: Props) {
  const { project, isProcessing } = useProjectStore();
  const [showWizard, setShowWizard] = useState(false);

  const handleMinimize = async () => {
    if (window.electronAPI) await window.electronAPI.windowMinimize();
  };

  const handleMaximize = async () => {
    if (window.electronAPI) await window.electronAPI.windowMaximize();
  };

  const handleClose = async () => {
    if (window.electronAPI) await window.electronAPI.windowClose();
  };

  return (
    <header className="header">
      <div className="header-left">
        <div className="logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <rect x="2" y="2" width="20" height="20" rx="4" fill="#3b82f6" />
            <path d="M9 7l6 5-6 5V7z" fill="white" />
          </svg>
          <span className="logo-text">Buzzcaf Editor</span>
        </div>
        {project && (
          <div className="header-project">
            <span className="project-name">{project.name}</span>
            <span className={`badge badge-${project.status}`}>{project.status}</span>
            {isProcessing && (
              <span className="processing-indicator animate-pulse">Processing...</span>
            )}
          </div>
        )}
      </div>

      <div className="header-actions" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <button className="btn btn-sm btn-secondary" onClick={() => setShowWizard(true)} title="Diagnostics & Settings">
          ⚙️ Diagnostics
        </button>
        {project && (
          <button className="btn btn-sm" onClick={onReset} title="New Project">
            New Project
          </button>
        )}
      </div>

      <div className="window-controls">
        <button className="window-btn" onClick={handleMinimize}>─</button>
        <button className="window-btn" onClick={handleMaximize}>□</button>
        <button className="window-btn close" onClick={handleClose}>✕</button>
      </div>

      <SetupWizard isOpen={showWizard} onClose={() => setShowWizard(false)} />
    </header>
  );
}

