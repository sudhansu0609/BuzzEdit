import { useCallback, useState } from 'react';
import { reasonLabel } from '../lib/editReasons';

/** The grammar audit of the finished edit — see backend/asr/verify.py. */
export interface EditQuality {
  sentences?: number;
  broken?: number;
  verdict?: string;
  rounds?: Array<{ round: number; sentences: number; broken: number; cut?: number }>;
  repairs?: Array<{ text: string; problem: string; removed: string[] }>;
  issues?: Array<{ text: string; problem: string; repaired?: boolean; removed?: string[] }>;
  cut_words_still_audible?: number;
  cut_words_checked?: number;
}

export interface EditReport {
  words_cut?: number;
  words_total?: number;
  fillers_found?: number;
  seconds_recovered?: number;
  speech_fraction?: number;
  used_audio?: boolean;
  used_llm?: boolean;
  used_fluency?: boolean;
  take_swaps?: number;
  final_read_cuts?: number;
  timing?: { repaired?: number; seconds_recovered?: number; dropped?: number };
  reasons?: Record<string, number>;
  quality?: EditQuality;
}

export interface RenderResult {
  outputPath: string;
  wordsRemoved?: number;
  wordsTotal?: number;
  report?: EditReport | null;
}

/** How the edit scored, as a badge: colour and words for the verdict. */
function verdictBadge(quality: EditQuality): { text: string; color: string } | null {
  const { verdict, sentences } = quality;
  if (!verdict || verdict === 'not_verified') {
    return { text: 'Not checked — the language model did not answer', color: 'var(--warning,#f59e0b)' };
  }
  if (verdict === 'clean') {
    return { text: `Reads as complete speech — all ${sentences ?? 0} sentences check out`,
             color: 'var(--success,#22c55e)' };
  }
  const broken = verdict.split(':')[1] ?? '?';
  return { text: `${broken} of ${sentences ?? 0} sentences still read as unfinished`,
           color: 'var(--warning,#f59e0b)' };
}

interface Props {
  result: RenderResult;
  onClose: () => void;
}

const fileName = (p: string) => p.replace(/\\/g, '/').split('/').pop() || p;

export default function RenderCompleteModal({ result, onClose }: Props) {
  const [msg, setMsg] = useState<string | null>(null);

  const handleOpen = useCallback(async () => {
    if (window.electronAPI?.openPath) {
      const err = await window.electronAPI.openPath(result.outputPath);
      if (err) setMsg(`Could not open file: ${err}`);
    } else {
      setMsg('Opening files requires the desktop app.');
    }
  }, [result.outputPath]);

  const handleReveal = useCallback(async () => {
    if (window.electronAPI?.showItemInFolder) {
      await window.electronAPI.showItemInFolder(result.outputPath);
    } else {
      setMsg('Revealing files requires the desktop app.');
    }
  }, [result.outputPath]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(result.outputPath);
      setMsg('Path copied to clipboard.');
    } catch {
      setMsg('Could not copy path.');
    }
  }, [result.outputPath]);

  return (
    <div
      className="modal-overlay"
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
      }}
    >
      <div
        className="modal-card"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(480px, 92vw)', background: 'var(--surface,#1c1c1c)',
          border: '1px solid var(--border,#333)', borderRadius: 12, padding: 24,
          boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
        }}
      >
        <div style={{ fontSize: 40, textAlign: 'center', marginBottom: 8 }}>✅</div>
        <h2 style={{ margin: '0 0 6px', textAlign: 'center', fontSize: 18 }}>Auto Edit Complete</h2>
        {result.wordsRemoved != null && (
          <p className="text-sm text-muted" style={{ textAlign: 'center', margin: '0 0 12px' }}>
            Removed {result.wordsRemoved}
            {result.wordsTotal != null ? ` of ${result.wordsTotal}` : ''} words (fillers & fumbles).
          </p>
        )}

        {/* What the edit actually did. The old build reported a word count and
            nothing else, which gave no way to tell a working pass from a no-op. */}
        {result.report && (
          <div className="edit-report">
            {(result.report.seconds_recovered ?? 0) > 0 && (
              <div><strong>{result.report.seconds_recovered!.toFixed(1)}s</strong> of dead air and fumbles removed</div>
            )}
            {(result.report.fillers_found ?? 0) > 0 && (
              <div>{result.report.fillers_found} filler sounds found in the audio that the transcript never had</div>
            )}
            {(result.report.timing?.repaired ?? 0) > 0 && (
              <div>
                {result.report.timing!.repaired} word timings corrected against the audio
                {(result.report.timing!.seconds_recovered ?? 0) > 0
                  ? ` (${result.report.timing!.seconds_recovered!.toFixed(1)}s was hidden inside them)` : ''}
              </div>
            )}
            {(result.report.take_swaps ?? 0) > 0 && (
              <div>
                {result.report.take_swaps} sentence{result.report.take_swaps === 1 ? '' : 's'} kept
                from an earlier, cleaner take than the last one
              </div>
            )}
            {result.report.reasons && Object.keys(result.report.reasons).length > 0 && (
              <div className="text-muted">
                {Object.entries(result.report.reasons)
                  .map(([reason, count]) => `${count} ${reasonLabel(reason)}`)
                  .join(' · ')}
              </div>
            )}

            {/* How the finished edit scored. The pipeline reads the result back
                sentence by sentence and repairs what it can; that answer used to
                be computed and thrown away, so there was no way to tell a clean
                edit from one that shipped broken sentences. */}
            {result.report.quality && (() => {
              const quality = result.report!.quality!;
              const badge = verdictBadge(quality);
              const repairs = quality.repairs ?? [];
              return (
                <div className="edit-quality" style={{ marginTop: 8 }}>
                  {badge && <div style={{ color: badge.color }}>{badge.text}</div>}
                  {repairs.length > 0 && (
                    <details style={{ marginTop: 4 }}>
                      <summary className="text-muted" style={{ cursor: 'pointer' }}>
                        {repairs.length} sentence{repairs.length === 1 ? '' : 's'} mended
                      </summary>
                      <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                        {repairs.slice(0, 6).map((r, i) => (
                          <li key={i} className="text-xs text-muted">
                            removed “{r.removed.join(' ')}”
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                  {(quality.cut_words_still_audible ?? 0) > 0 && (
                    <div style={{ color: 'var(--danger,#ef4444)' }}>
                      {quality.cut_words_still_audible} cut word
                      {quality.cut_words_still_audible === 1 ? ' is' : 's are'} still covered by the
                      render and will be heard.
                    </div>
                  )}
                </div>
              );
            })()}

            {result.report.used_audio === false && (
              <div style={{ color: 'var(--warning,#f59e0b)' }}>
                Audio could not be analysed — this was a text-only pass, which finds very little.
              </div>
            )}
            {result.report.used_llm === false && result.report.used_fluency === false && (
              <div style={{ color: 'var(--warning,#f59e0b)' }}>
                The language model was not available, so the edit was made on structure alone
                and was never read back for sense.
              </div>
            )}
          </div>
        )}

        <div
          className="font-mono text-xs"
          title={result.outputPath}
          style={{
            background: 'var(--surface-2,#111)', border: '1px solid var(--border,#2a2a2a)',
            borderRadius: 6, padding: '8px 10px', margin: '0 0 16px',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}
        >
          🎬 {fileName(result.outputPath)}
        </div>

        {msg && <div className="text-xs" style={{ marginBottom: 12, color: 'var(--warning,#f59e0b)' }}>{msg}</div>}

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={handleOpen} style={{ flex: 1 }}>▶ Open Video</button>
          <button className="btn" onClick={handleReveal} style={{ flex: 1 }}>📂 Show in Folder</button>
          <button className="btn btn-sm" onClick={handleCopy} title="Copy full path">Copy Path</button>
        </div>
        <button className="btn btn-sm" onClick={onClose} style={{ width: '100%', marginTop: 10 }}>Close</button>
      </div>
    </div>
  );
}
