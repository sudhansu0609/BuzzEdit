import { useState, useCallback } from 'react';
import { useProjectStore } from '../hooks/store';
import { useLayoutStore } from '../hooks/layout';
import { useCommand } from '../hooks/commands';
import { autoSaveProject } from '../hooks/api';
import Preferences, { PrefTab } from './Preferences';
import MenuBar from './MenuBar';
import LayoutMenu from './LayoutMenu';

interface Props {
  onReset: () => void;
}

// Common output resolutions. Value is "WIDTHxHEIGHT" as stored in project.settings.resolution.
const RESOLUTION_PRESETS: { label: string; value: string }[] = [
  { label: '4K (3840×2160)', value: '3840x2160' },
  { label: '1080p (1920×1080)', value: '1920x1080' },
  { label: '720p (1280×720)', value: '1280x720' },
  { label: 'Shorts (1080×1920)', value: '1080x1920' },
  { label: 'Vertical 720 (720×1280)', value: '720x1280' },
];

export default function Header({ onReset }: Props) {
  const { project, isProcessing, updateProject } = useProjectStore();
  const [prefs, setPrefs] = useState<{ open: boolean; tab: PrefTab }>({ open: false, tab: 'general' });
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved'>('idle');

  // Select only stable action refs (and `locked`), so panel drags don't churn
  // the header or re-register every command each pointer move.
  const toggleVisible = useLayoutStore((s) => s.toggleVisible);
  const applyPreset = useLayoutStore((s) => s.applyPreset);
  const setLocked = useLayoutStore((s) => s.setLocked);
  const resetLayout = useLayoutStore((s) => s.resetLayout);
  const locked = useLayoutStore((s) => s.locked);

  const openPrefs = useCallback((tab: PrefTab) => setPrefs({ open: true, tab }), []);

  // --- Menu / shortcut commands owned by the header ---
  useCommand('file.new', onReset);
  useCommand('app.preferences', useCallback(() => openPrefs('general'), [openPrefs]));
  useCommand('help.diagnostics', useCallback(() => openPrefs('diagnostics'), [openPrefs]));
  useCommand('help.shortcuts', useCallback(() => openPrefs('shortcuts'), [openPrefs]));
  useCommand('view.panel.transcript', useCallback(() => toggleVisible('transcript'), [toggleVisible]));
  useCommand('view.panel.inspector', useCallback(() => toggleVisible('inspector'), [toggleVisible]));
  useCommand('view.panel.media', useCallback(() => toggleVisible('media'), [toggleVisible]));
  useCommand('view.panel.agents', useCallback(() => toggleVisible('agents'), [toggleVisible]));
  useCommand('view.panel.queue', useCallback(() => toggleVisible('queue'), [toggleVisible]));
  useCommand('view.panel.export', useCallback(() => toggleVisible('export'), [toggleVisible]));
  useCommand('view.preset.default', useCallback(() => applyPreset('default'), [applyPreset]));
  useCommand('view.preset.edit', useCallback(() => applyPreset('edit'), [applyPreset]));
  useCommand('view.preset.colour', useCallback(() => applyPreset('colour'), [applyPreset]));
  useCommand('view.preset.review', useCallback(() => applyPreset('review'), [applyPreset]));
  useCommand('view.lockLayout', useCallback(() => setLocked(!locked), [setLocked, locked]));
  useCommand('view.resetLayout', useCallback(() => resetLayout(), [resetLayout]));

  const currentResolution = project?.settings?.resolution || '1920x1080';

  const persist = useCallback(async (settingsOverride?: Record<string, any>) => {
    if (!project?.id) return;
    setSaveState('saving');
    try {
      await autoSaveProject(project.id, {
        clips: project.clips,
        transcript: project.transcript,
        detectedSegments: project.detectedSegments,
        settings: settingsOverride ?? project.settings,
        status: project.status,
        timeline: project.timeline,
      });
      setSaveState('saved');
      setTimeout(() => setSaveState('idle'), 2000);
    } catch (err) {
      console.error('Save failed:', err);
      setSaveState('idle');
    }
  }, [project]);

  const handleSave = useCallback(() => { persist(); }, [persist]);
  useCommand('file.save', handleSave);

  const handleResolutionChange = useCallback((value: string) => {
    if (!project) return;
    const newSettings = { ...project.settings, resolution: value };
    updateProject({ settings: newSettings });
    persist(newSettings);
  }, [project, updateProject, persist]);

  // Backend returns snake_case `output_path`; fall back to camelCase for safety.
  const outputPath: string | undefined = (project as any)?.output_path || project?.outputPath;

  const handleOpenOutput = useCallback(() => {
    if (outputPath && window.electronAPI?.openPath) {
      window.electronAPI.openPath(outputPath);
    }
  }, [outputPath]);

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
          <span className="logo-text">BuzzEdit</span>
        </div>
        <MenuBar />
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
        {project && (
          <label style={{ display: 'flex', alignItems: 'center', gap: 4 }} title="Output resolution">
            <span className="text-xs text-muted">Resolution</span>
            <select
              className="btn btn-sm"
              value={currentResolution}
              onChange={(e) => handleResolutionChange(e.target.value)}
              style={{ padding: '4px 8px' }}
            >
              {RESOLUTION_PRESETS.some(p => p.value === currentResolution)
                ? null
                : <option value={currentResolution}>{currentResolution}</option>}
              {RESOLUTION_PRESETS.map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </label>
        )}
        {project && (
          <button
            className="btn btn-sm btn-primary"
            onClick={handleSave}
            disabled={saveState === 'saving'}
            title="Save project"
          >
            {saveState === 'saving' ? 'Saving…' : saveState === 'saved' ? 'Saved ✓' : '💾 Save'}
          </button>
        )}
        {outputPath && (
          <button className="btn btn-sm btn-success" onClick={handleOpenOutput} title={`Open last render: ${outputPath}`}>
            ▶ Open Output
          </button>
        )}
        {project && <LayoutMenu />}
        <button className="btn btn-sm btn-secondary" onClick={() => openPrefs('general')} title="Preferences & Settings">
          ⚙️ Settings
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

      <Preferences
        isOpen={prefs.open}
        initialTab={prefs.tab}
        onClose={() => setPrefs((p) => ({ ...p, open: false }))}
      />
    </header>
  );
}

