import { useState, useEffect, useCallback } from 'react';
import { useProjectStore } from '../hooks/store';
import { getComfyUIStatus, triggerFullEdit, triggerBRoll, triggerThumbnail, triggerCaptions } from '../hooks/api';

export default function AgentPanel() {
  const { project, setProcessing, setProgress, setError } = useProjectStore();
  const [comfyStatus, setComfyStatus] = useState<{ connected: boolean; status: string }>({ connected: false, status: 'checking' });
  const [genBroll, setGenBroll] = useState(true);
  const [genThumb, setGenThumb] = useState(true);
  const [burnCaps, setBurnCaps] = useState(true);
  const [customTitle, setCustomTitle] = useState('');

  const checkComfy = useCallback(async () => {
    try {
      const res = await getComfyUIStatus();
      setComfyStatus({ connected: res.connected, status: res.status });
    } catch {
      setComfyStatus({ connected: false, status: 'offline' });
    }
  }, []);

  useEffect(() => {
    checkComfy();
    const interval = setInterval(checkComfy, 10000);
    return () => clearInterval(interval);
  }, [checkComfy]);

  const handleFullAgentEdit = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Executing Full Autonomous AI Edit...');
    try {
      await triggerFullEdit(project.id, {
        generate_broll: genBroll,
        generate_thumbnail: genThumb,
        burn_captions: burnCaps,
      });
      setProgress(1, 'AI Edit Completed successfully!');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleBRollOnly = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Generating AI B-Roll Clips...');
    try {
      await triggerBRoll(project.id);
      setProgress(1, 'B-Roll generation completed');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleThumbnailOnly = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Generating Thumbnail...');
    try {
      await triggerThumbnail(project.id, customTitle || undefined);
      setProgress(1, 'Thumbnail generation completed');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleCaptionsOnly = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Burning styled captions...');
    try {
      await triggerCaptions(project.id);
      setProgress(1, 'Caption burn-in completed');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="agent-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px', color: '#fff' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px', background: '#1e1e2e', borderRadius: '8px' }}>
        <span>ComfyUI Bridge:</span>
        <span style={{
          padding: '4px 10px',
          borderRadius: '12px',
          fontSize: '12px',
          fontWeight: 'bold',
          background: comfyStatus.connected ? '#10b981' : '#ef4444',
          color: '#fff'
        }}>
          {comfyStatus.connected ? 'ONLINE (127.0.0.1:8188)' : 'OFFLINE (Fallback Active)'}
        </span>
      </div>

      <div style={{ background: '#181825', padding: '14px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <h4 style={{ margin: 0, color: '#a6adc8' }}>AI Agent Settings</h4>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="checkbox" checked={genBroll} onChange={(e) => setGenBroll(e.target.checked)} />
          Generate AI B-Roll Overlays
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="checkbox" checked={genThumb} onChange={(e) => setGenThumb(e.target.checked)} />
          Generate YouTube Thumbnail
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="checkbox" checked={burnCaps} onChange={(e) => setBurnCaps(e.target.checked)} />
          Burn-in Styled Captions
        </label>
        <div style={{ marginTop: '8px' }}>
          <label style={{ fontSize: '12px', color: '#bac2de', display: 'block', marginBottom: '4px' }}>Thumbnail Custom Title:</label>
          <input
            type="text"
            placeholder="e.g. EPIC VIDEO EDIT!"
            value={customTitle}
            onChange={(e) => setCustomTitle(e.target.value)}
            style={{ width: '100%', padding: '8px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
          />
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <button
          className="btn btn-primary"
          style={{ width: '100%', padding: '12px', fontWeight: 'bold', fontSize: '14px' }}
          onClick={handleFullAgentEdit}
        >
          🚀 Run Full Autonomous AI Agent Edit
        </button>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={handleBRollOnly}>🎬 B-Roll Only</button>
          <button className="btn btn-secondary" onClick={handleThumbnailOnly}>🖼️ Thumbnail</button>
        </div>
        <button className="btn btn-secondary" onClick={handleCaptionsOnly}>💬 Burn Captions Only</button>
      </div>
    </div>
  );
}
