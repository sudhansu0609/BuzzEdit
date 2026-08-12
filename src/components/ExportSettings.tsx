import { useCallback, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { useElectron, exportFilmoraXml, applyColorGrading, convertToShorts } from '../hooks/api';

export default function ExportSettings() {
  const { project, updateProject, setProcessing, setProgress, setError } = useProjectStore();
  const { saveFile } = useElectron();
  const [resolution, setResolution] = useState('1080p');
  const [fps, setFps] = useState(30);
  const [codec, setCodec] = useState('h264');
  const [quality, setQuality] = useState('high');
  const [format, setFormat] = useState('mp4');
  const [audioBitrate, setAudioBitrate] = useState('192');
  const [includeTransitions, setIncludeTransitions] = useState(true);
  const [removeSilence, setRemoveSilence] = useState(true);
  const [removeFumbles, setRemoveFumbles] = useState(true);
  const [colorPreset, setColorPreset] = useState('cinematic');
  const [msg, setMsg] = useState<string | null>(null);

  const handleSaveSettings = useCallback(async () => {
    if (!project) return;
    const settings = {
      resolution,
      fps,
      codec,
      quality,
      format,
      audioBitrate,
      includeTransitions,
      removeSilence,
      removeFumbles,
    };
    await updateProject({ settings });
  }, [project, resolution, fps, codec, quality, format, audioBitrate, includeTransitions, removeSilence, removeFumbles, updateProject]);

  const handleExport = useCallback(async () => {
    await handleSaveSettings();
    const outputPath = await saveFile(`output.${format}`);
    if (outputPath) {
      updateProject({ outputPath });
    }
  }, [handleSaveSettings, saveFile, format, updateProject]);

  const handleFilmoraExport = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Generating Filmora XML Timeline...');
    try {
      const res = await exportFilmoraXml(project.id);
      setMsg(`Filmora XML Exported to: ${res.xml_path}`);
      setProgress(1, 'Filmora XML Export Complete');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleApplyColorGrade = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, `Applying ${colorPreset} color grade...`);
    try {
      const res = await applyColorGrading(project.id, colorPreset);
      setMsg(`Color Graded Video Saved: ${res.output_video}`);
      setProgress(1, 'Color Grade Complete');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleConvertToShorts = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Converting to 9:16 Vertical Shorts...');
    try {
      const res = await convertToShorts(project.id);
      setMsg(`9:16 Shorts Video Saved: ${res.output_video}`);
      setProgress(1, 'Shorts Conversion Complete');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const resolutions = ['4k', '2k', '1080p', '720p', '480p'];
  const codecs = ['h264', 'h265', 'vp9', 'av1'];
  const qualities = ['ultra', 'high', 'medium', 'low'];
  const formats = ['mp4', 'mkv', 'webm', 'mov'];
  const audioBitrates = ['320', '256', '192', '128', '96'];

  return (
    <div className="export-settings" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px', color: '#fff' }}>
      {msg && (
        <div style={{ padding: '8px 12px', background: '#313244', color: '#a6e3a1', borderRadius: '6px', fontSize: '12px' }}>
          {msg}
        </div>
      )}

      {/* Filmora XML Section */}
      <div style={{ background: '#1e1e2e', padding: '14px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', color: '#3b82f6' }}>🎬 Filmora NLE Integration</h3>
        <span style={{ fontSize: '12px', color: '#a6adc8' }}>
          Export timeline as a Final Cut Pro XML file for manual polishing in Filmora.
        </span>
        <button className="btn btn-primary" onClick={handleFilmoraExport}>
          📄 Export Filmora Timeline (XML)
        </button>
      </div>

      {/* Color Grading & Shorts Presets */}
      <div style={{ background: '#1e1e2e', padding: '14px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', color: '#f59e0b' }}>🎨 Color Grading & Formats</h3>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select
            value={colorPreset}
            onChange={(e) => setColorPreset(e.target.value)}
            style={{ flex: 1, padding: '8px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
          >
            <option value="cinematic">Cinematic Look</option>
            <option value="vibrant">Vibrant & Punchy</option>
            <option value="warm">Warm Film Mood</option>
            <option value="cool">Cool Tech Vibe</option>
            <option value="dark">Dramatic Dark</option>
          </select>
          <button className="btn btn-secondary" onClick={handleApplyColorGrade}>Apply Grade</button>
        </div>
        <button className="btn btn-secondary" onClick={handleConvertToShorts}>📱 Convert to 9:16 Vertical Shorts</button>
      </div>

      <div className="settings-section">
        <h3 className="section-title">Output Resolution & Codec</h3>
        <div className="setting-row">
          <label className="setting-label">Format</label>
          <select value={format} onChange={(e) => setFormat(e.target.value)}>
            {formats.map(f => <option key={f} value={f}>{f.toUpperCase()}</option>)}
          </select>
        </div>
        <div className="setting-row">
          <label className="setting-label">Resolution</label>
          <select value={resolution} onChange={(e) => setResolution(e.target.value)}>
            {resolutions.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
        <div className="setting-row">
          <label className="setting-label">FPS</label>
          <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
            <option value={24}>24</option>
            <option value={30}>30</option>
            <option value={60}>60</option>
          </select>
        </div>
      </div>

      <div className="export-actions">
        <button className="btn" onClick={handleSaveSettings}>
          Save Settings
        </button>
        <button className="btn btn-success btn-lg" onClick={handleExport}>
          Export Final Video
        </button>
      </div>
    </div>
  );
}

