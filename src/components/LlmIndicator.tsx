import { useCallback, useEffect, useState } from 'react';
import { getLlmStatus, ensureLlm, LlmStatus } from '../hooks/api';

export default function LlmIndicator() {
  const [status, setStatus] = useState<LlmStatus | null>(null);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async () => {
    try { setStatus(await getLlmStatus()); } catch { /* backend may be starting */ }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 15000);
    return () => window.clearInterval(id);
  }, [refresh]);

  const handleStart = useCallback(async () => {
    setStarting(true);
    try {
      const s = await ensureLlm();
      setStatus(s);
    } catch { /* ignore */ } finally {
      setStarting(false);
    }
  }, []);

  const ready = !!status?.server_up && !!status?.model_loaded;
  const color = ready ? '#22c55e' : status?.server_up ? '#f59e0b' : '#ef4444';
  const label = starting ? 'Starting…'
    : ready ? 'LLM ready'
    : status?.server_up ? 'LLM: no model'
    : 'LLM offline';

  const title = !status?.cli_available && !status?.server_up
    ? 'LM Studio CLI not found — run `lms bootstrap` once to enable auto-start'
    : 'LM Studio powers context-aware fumble removal. Click to start & load the model.';

  return (
    <button
      className="btn btn-sm"
      onClick={handleStart}
      disabled={starting || ready}
      title={title}
      style={{ display: 'flex', alignItems: 'center', gap: 6 }}
    >
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block' }} />
      <span className="text-xs">{label}</span>
    </button>
  );
}
