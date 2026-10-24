import { useState } from 'react';
import { MediaEntry, SyncResult, apiErrorDetail, syncAudio, undoAudioVersion } from '../hooks/api';

/** "+1.75 s" with an explicit sign, for offsets. */
export function fmtOffset(seconds: number): string {
  return `${seconds >= 0 ? '+' : '−'}${Math.abs(seconds).toFixed(2)} s`;
}

/** "drift −9 ppm" — the two clocks' rate difference. */
export function fmtDrift(ppm: number): string {
  return `drift ${ppm >= 0 ? '+' : '−'}${Math.abs(ppm).toFixed(0)} ppm`;
}

type Phase = 'confirm' | 'running' | 'done' | 'undone' | 'error';

/**
 * "Use as programme audio (sync)": an external recorder replaces the camera sound,
 * aligned to the picture (POST /api/timeline/{pid}/sync/audio). The result offers
 * Undo right there (POST /api/audio/{pid}/undo with the primary source).
 */
export default function SyncAudioDialog({ entry, projectId, onClose }: {
  entry: MediaEntry; projectId: string; onClose: () => void;
}) {
  const [phase, setPhase] = useState<Phase>('confirm');
  const [result, setResult] = useState<SyncResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [undoing, setUndoing] = useState(false);

  const run = async () => {
    setPhase('running');
    setError(null);
    try {
      setResult(await syncAudio(projectId, entry.id));
      setPhase('done');
    } catch (err) {
      setError(apiErrorDetail(err));
      setPhase('error');
    }
  };

  const undo = async () => {
    if (!result) return;
    setUndoing(true);
    try {
      await undoAudioVersion(projectId, result.undo?.body.source_id ?? result.source_id);
      setPhase('undone');
    } catch (err) {
      setError(apiErrorDetail(err));
    } finally {
      setUndoing(false);
    }
  };

  const busy = phase === 'running' || undoing;

  return (
    <div className="timeline-dialog-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose(); }}>
      <div className="timeline-dialog sync-dialog" role="dialog" aria-label="Use as programme audio">
        <div className="timeline-dialog-title" title={entry.name}>Use “{entry.name}” as programme audio</div>
        {phase === 'confirm' && (
          <div className="text-xs text-muted">
            The camera audio of the main recording will be replaced by this recording, aligned to
            the picture from the two soundtracks (clock drift corrected). Render, preview and the
            next transcription use it. You can undo it here or from the Audio panel.
          </div>
        )}
        {phase === 'running' && (
          <div className="text-xs"><span className="spinner" style={{ width: 10, height: 10 }} /> Measuring the offset and writing the aligned audio…</div>
        )}
        {(phase === 'done' || phase === 'undone') && result && (
          <div className="sync-result text-xs">
            <div>Offset <b>{fmtOffset(result.offset_s)}</b> · confidence {result.confidence.toFixed(1)} · {fmtDrift(result.drift_ppm)}</div>
            <div className="text-muted">
              {phase === 'undone' ? 'Undone: the camera audio is back.' : 'The programme now plays this recording.'}
            </div>
          </div>
        )}
        {error && <div className="sync-error text-xs">{error}</div>}
        <div className="timeline-dialog-actions">
          {phase === 'confirm' && (
            <>
              <button className="btn btn-sm" onClick={onClose}>Cancel</button>
              <button className="btn btn-sm btn-primary" onClick={run}>Replace audio</button>
            </>
          )}
          {phase === 'error' && (
            <>
              <button className="btn btn-sm" onClick={onClose}>Close</button>
              <button className="btn btn-sm" onClick={run}>Try again</button>
            </>
          )}
          {phase === 'done' && (
            <>
              <button className="btn btn-sm" disabled={undoing} onClick={undo}
                title="Put the camera audio back">{undoing ? 'Undoing…' : 'Undo'}</button>
              <button className="btn btn-sm btn-primary" disabled={undoing} onClick={onClose}>Done</button>
            </>
          )}
          {phase === 'undone' && <button className="btn btn-sm btn-primary" onClick={onClose}>Close</button>}
        </div>
      </div>
    </div>
  );
}
