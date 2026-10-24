import { useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { AutoAngleMode, apiErrorDetail, autoAngles, clearAngles, useAnglesStore } from '../hooks/api';
import { Section } from './controls';
import { fmtOffset } from './SyncAudioDialog';

const MODES: { id: AutoAngleMode; label: string; hint: string }[] = [
  { id: 'hide_cuts', label: 'Hide jump cuts', hint: 'Change angle on every jump cut, rotating through the angles' },
  { id: 'rhythm', label: 'Rhythm', hint: 'Change angle at a sentence boundary about every N seconds' },
  { id: 'off', label: 'Off', hint: 'Remove every angle switch' },
];

/**
 * Inspector "Camera angles": the synced angles (GET /angles, fetched on open and after
 * every angle action) and Auto multicam (POST /angles/auto). Angles are synced from the
 * media library; switches by hand are on the timeline's clip menu.
 */
export default function CameraAnglesSection() {
  const projectId = useProjectStore((s) => s.project?.id);
  const setError = useProjectStore((s) => s.setError);
  const angles = useAnglesStore((s) => (s.projectId === projectId ? s.state : null));
  const load = useAnglesStore((s) => s.load);
  const [mode, setMode] = useState<AutoAngleMode>('hide_cuts');
  const [minShot, setMinShot] = useState(2.5);
  const [cadence, setCadence] = useState(8);
  const [busy, setBusy] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);

  const list = angles?.angles ?? [];
  const inUse = (angles?.spans?.length ?? 0) > 0;

  const run = async () => {
    if (!projectId) return;
    setBusy(true);
    setSummary(null);
    try {
      const r = await autoAngles(projectId, mode, { min_shot_s: minShot, cadence_s: cadence });
      const s = r.summary;
      setSummary(mode === 'off'
        ? 'Angle switches removed'
        : `${s.switches} switch${s.switches === 1 ? '' : 'es'}, ${s.cuts_hidden} of ${s.cuts_total} cut${s.cuts_total === 1 ? '' : 's'} hidden`);
    } catch (err) {
      setError(apiErrorDetail(err));
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    if (!projectId || !window.confirm('Remove every angle switch? V1 shows the main recording again (undoable).')) return;
    setBusy(true);
    try {
      await clearAngles(projectId);
      setSummary('Angle switches removed');
    } catch (err) {
      setError(apiErrorDetail(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Section title="Camera angles" defaultOpen={false}
      onToggle={(open) => { if (open && projectId) load(projectId); }}>
      {!list.length ? (
        <div className="text-xs text-muted">
          No camera angle yet. In the media library, right-click a second camera's video and choose
          “Sync as camera angle”; it is lined up with the main recording from the two soundtracks.
        </div>
      ) : (
        <>
          <div className="angle-list">
            {list.map((a, i) => (
              <div key={a.source_id} className="angle-row" title={`${a.name} · offset ${fmtOffset(a.offset_s)} · confidence ${a.confidence.toFixed(1)}`}>
                <span className={`angle-row-swatch angle-${(i % 3) + 1}`} />
                <span className="angle-row-name">{i + 1}. {a.name}</span>
                <span className="angle-row-meta">{fmtOffset(a.offset_s)} · {a.confidence.toFixed(1)}</span>
              </div>
            ))}
          </div>
          <div className="text-xs font-weight-semibold">Auto multicam</div>
          <div className="angle-auto-modes">
            {MODES.map((m) => (
              <label key={m.id} title={m.hint}>
                <input type="radio" name="angle-auto-mode" checked={mode === m.id} onChange={() => setMode(m.id)} />
                {m.id === 'rhythm' ? (
                  <span>Rhythm every{' '}
                    <input className="angle-auto-num" type="number" min={2} max={60} step={0.5} value={cadence}
                      aria-label="Cadence seconds"
                      onChange={(e) => setCadence(Math.max(1, Number(e.target.value) || 8))} /> s</span>
                ) : m.label}
              </label>
            ))}
            {mode !== 'off' && (
              <label title="No shot is shorter than this">
                Min shot{' '}
                <input className="angle-auto-num" type="number" min={0.5} max={30} step={0.5} value={minShot}
                  aria-label="Minimum shot seconds"
                  onChange={(e) => setMinShot(Math.max(0.5, Number(e.target.value) || 2.5))} /> s
              </label>
            )}
          </div>
          <div className="insp-actions">
            <button className="btn btn-sm btn-primary" disabled={busy || !projectId} onClick={run}>
              {busy ? 'Working…' : 'Run'}
            </button>
            <button className="btn btn-sm" disabled={busy || !projectId || !inUse} onClick={clear}>Clear angles</button>
          </div>
          {summary && <div className="text-xs text-muted">{summary}</div>}
        </>
      )}
    </Section>
  );
}
