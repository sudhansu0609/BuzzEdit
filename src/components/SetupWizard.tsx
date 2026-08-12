import { useState, useEffect, useCallback } from 'react';
import { getGpuStatus, getSystemPaths, clearVram } from '../hooks/api';

interface SetupWizardProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SetupWizard({ isOpen, onClose }: SetupWizardProps) {
  const [gpuInfo, setGpuInfo] = useState<any>(null);
  const [pathsInfo, setPathsInfo] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const fetchDiagnostics = useCallback(async () => {
    setLoading(true);
    try {
      const [gpuRes, pathsRes] = await Promise.all([
        getGpuStatus(),
        getSystemPaths(),
      ]);
      setGpuInfo(gpuRes);
      setPathsInfo(pathsRes);
    } catch (err: any) {
      console.error('Failed to fetch diagnostics:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchDiagnostics();
    }
  }, [isOpen, fetchDiagnostics]);

  const handleClearVram = async () => {
    try {
      setMessage('Flushing PyTorch CUDA Cache...');
      const res = await clearVram();
      setGpuInfo(res.gpu);
      setMessage('CUDA VRAM Cache cleared successfully!');
      setTimeout(() => setMessage(null), 3000);
    } catch (err: any) {
      setMessage(`Clear VRAM failed: ${err.message}`);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0, 0, 0, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(4px)'
    }}>
      <div style={{
        background: '#181825',
        border: '1px solid #313244',
        borderRadius: '12px',
        width: '560px',
        maxWidth: '90vw',
        padding: '24px',
        color: '#fff',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #313244', paddingBottom: '12px' }}>
          <h3 style={{ margin: 0, fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            ⚙️ Hardware & Dependency Diagnostics
          </h3>
          <button className="btn btn-sm" onClick={onClose}>✕</button>
        </div>

        {message && (
          <div style={{ padding: '8px 12px', background: '#313244', color: '#a6e3a1', borderRadius: '6px', fontSize: '12px' }}>
            {message}
          </div>
        )}

        {/* GPU VRAM Section */}
        <div style={{ background: '#1e1e2e', padding: '16px', borderRadius: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontWeight: 'bold', fontSize: '14px' }}>
              🎮 GPU: {gpuInfo?.device_name || 'Detecting...'}
            </span>
            <span style={{ fontSize: '12px', color: '#a6adc8' }}>
              {gpuInfo?.free_vram_mb || 0} MB Free / {gpuInfo?.total_vram_mb || 0} MB Total
            </span>
          </div>

          {gpuInfo && gpuInfo.total_vram_mb > 0 && (
            <div style={{ height: '8px', background: '#313244', borderRadius: '4px', overflow: 'hidden', marginBottom: '12px' }}>
              <div style={{
                height: '100%',
                width: `${Math.min(100, (gpuInfo.allocated_vram_mb / gpuInfo.total_vram_mb) * 100)}%`,
                background: '#3b82f6',
                transition: 'width 0.3s ease'
              }} />
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: '#bac2de' }}>CUDA Memory Status: Active</span>
            <button className="btn btn-sm btn-secondary" onClick={handleClearVram}>
              🧹 Flush VRAM Cache
            </button>
          </div>
        </div>

        {/* Dependency Checklist */}
        <div style={{ background: '#1e1e2e', padding: '16px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <h4 style={{ margin: 0, fontSize: '13px', color: '#a6adc8' }}>Dependency & Service Checks</h4>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
            <span>FFmpeg Hardware Encoding (NVENC):</span>
            <span style={{ color: pathsInfo?.ffmpeg_available ? '#10b981' : '#ef4444', fontWeight: 'bold' }}>
              {pathsInfo?.ffmpeg_available ? '✓ AVAILABLE' : '✕ NOT FOUND'}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
            <span>ComfyUI Server Bridge (127.0.0.1:8188):</span>
            <span style={{ color: pathsInfo?.comfyui_connected ? '#10b981' : '#f59e0b', fontWeight: 'bold' }}>
              {pathsInfo?.comfyui_connected ? '✓ ONLINE' : '⚠ OFFLINE (Fallback Active)'}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
            <span>ComfyUI Portable Input Folder:</span>
            <span style={{ color: pathsInfo?.comfyui_input_dir_exists ? '#10b981' : '#ef4444', fontSize: '11px' }}>
              {pathsInfo?.comfyui_input_dir_exists ? '✓ EXISTS' : '✕ MISSING'}
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={fetchDiagnostics} disabled={loading}>
            🔄 Refresh
          </button>
          <button className="btn btn-primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
