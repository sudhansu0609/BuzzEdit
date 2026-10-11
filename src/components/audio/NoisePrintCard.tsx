/**
 * Audition's Capture Noise Print (AUDIO_SUITE_PLAN.md §2.3): Detect noise finds the quietest
 * stretch between words, the range can be corrected by hand, Capture Noise Print stores it on
 * the project, and the print's mean-dB spectrum is drawn with the band floor that the noise
 * reduction is measured against. One print per project; Noise Reduction (process) uses it.
 */

import { useEffect, useState } from 'react';
import {
  AudioApiError, NoisePrint, captureNoisePrint, detectNoise, errorText, getNoisePrint, routeMissing,
} from '../../hooks/audioApi';

const W = 320;
const H = 120;
const PAD_L = 30;
const PAD_R = 6;
const PAD_T = 6;
const PAD_B = 18;
const DECADES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

const secs = (v: unknown, places = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(places) : '?');

const fmtHz = (f: number) => (f >= 1000 ? `${parseFloat((f / 1000).toFixed(1))}k` : `${Math.round(f)}`);

function PrintCurve({ print }: { print: NoisePrint }) {
  const pts: Array<[number, number]> = [];
  (print.freqs_hz || []).forEach((f, i) => {
    const d = print.mean_db?.[i];
    if (f >= 20 && typeof d === 'number' && Number.isFinite(d)) pts.push([f, d]);
  });
  if (pts.length < 2) return <div className="text-xs text-muted">The print has no spectrum to draw.</div>;

  const fMin = 20;
  const fMax = pts[pts.length - 1][0];
  const floorDb = typeof print.band_floor_db === 'number' && Number.isFinite(print.band_floor_db)
    ? print.band_floor_db : null;
  let lo = Math.min(...pts.map((p) => p[1]));
  let hi = Math.max(...pts.map((p) => p[1]));
  if (floorDb !== null) {
    lo = Math.min(lo, floorDb);
    hi = Math.max(hi, floorDb);
  }
  lo = Math.floor(lo / 10) * 10;
  hi = Math.ceil(hi / 10) * 10;
  if (hi - lo < 20) hi = lo + 20;

  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const x = (f: number) => PAD_L + plotW * (Math.log(f / fMin) / Math.log(fMax / fMin));
  const y = (d: number) => PAD_T + plotH * (1 - (d - lo) / (hi - lo));
  const line = pts.map(([f, d]) => `${x(f).toFixed(1)},${y(d).toFixed(1)}`).join(' ');
  const dbTicks: number[] = [];
  const stepDb = hi - lo > 60 ? 20 : 10;
  for (let d = lo; d <= hi; d += stepDb) dbTicks.push(d);

  return (
    <svg className="audio-print-svg" viewBox={`0 0 ${W} ${H}`} aria-label="Noise print spectrum">
      <rect x={PAD_L} y={PAD_T} width={plotW} height={plotH} className="audio-curve-bg" />
      {DECADES.filter((f) => f >= fMin && f <= fMax).map((f) => (
        <g key={f}>
          <line x1={x(f)} x2={x(f)} y1={PAD_T} y2={PAD_T + plotH} className="audio-grid" />
          <text x={x(f)} y={H - 5} textAnchor="middle" className="audio-axis">{fmtHz(f)}</text>
        </g>
      ))}
      {dbTicks.map((d) => (
        <g key={d}>
          <line x1={PAD_L} x2={PAD_L + plotW} y1={y(d)} y2={y(d)} className="audio-grid" />
          <text x={PAD_L - 3} y={y(d) + 3} textAnchor="end" className="audio-axis">{d}</text>
        </g>
      ))}
      <polyline points={line} className="audio-print-line" />
      {floorDb !== null && (
        <g>
          <line x1={PAD_L} x2={PAD_L + plotW} y1={y(floorDb)} y2={y(floorDb)} className="audio-print-floor" />
          <text x={PAD_L + plotW - 2} y={y(floorDb) - 3} textAnchor="end" className="audio-axis">
            floor {floorDb.toFixed(1)} dB
          </text>
        </g>
      )}
    </svg>
  );
}

export default function NoisePrintCard({ projectId, sourceId, outputNoiseOnly }: {
  projectId: string;
  /** undefined = the programme source */
  sourceId?: string;
  /** the process's "Output Noise Only" switch, for the hint */
  outputNoiseOnly: boolean;
}) {
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [floor, setFloor] = useState<number | null>(null);
  const [print, setPrint] = useState<NoisePrint | null>(null);
  const [busy, setBusy] = useState<'detect' | 'capture' | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    let alive = true;
    getNoisePrint(projectId)
      .then((p) => {
        if (!alive) return;
        setPrint(p);
        setStart((s) => s || (typeof p.start_s === 'number' ? p.start_s.toFixed(3) : ''));
        setEnd((e) => e || (typeof p.end_s === 'number' ? p.end_s.toFixed(3) : ''));
      })
      .catch((err) => {
        if (!alive) return;
        if (routeMissing(err)) setMissing(true);
        else if (!(err instanceof AudioApiError && err.status === 404)) {
          setNote(`Could not read the noise print: ${errorText(err)}`);
        }
      });
    return () => { alive = false; };
  }, [projectId]);

  const fail = (err: unknown, what: string) => {
    if (routeMissing(err)) setMissing(true);
    else setNote(`${what}: ${errorText(err)}`);
  };

  const detect = async () => {
    setBusy('detect');
    setNote(null);
    try {
      const r = await detectNoise(projectId, sourceId);
      setStart(secs(r.start_s, 3));
      setEnd(secs(r.end_s, 3));
      setFloor(typeof r.floor_dbfs === 'number' ? r.floor_dbfs : null);
      setNote(`Quietest stretch between words: ${secs(r.start_s)}–${secs(r.end_s)} s`
        + `${r.method ? ` (${r.method})` : ''}. Adjust it if needed, then Capture Noise Print.`);
    } catch (err) {
      fail(err, 'Detect noise failed');
    } finally {
      setBusy(null);
    }
  };

  const capture = async () => {
    const a = parseFloat(start);
    const b = parseFloat(end);
    if (!(Number.isFinite(a) && Number.isFinite(b) && b > a)) {
      setNote('Enter a start and an end after it (seconds), or press Detect noise.');
      return;
    }
    setBusy('capture');
    setNote(null);
    try {
      const p = await captureNoisePrint(projectId, { source_id: sourceId, start_s: a, end_s: b });
      setPrint(p);
      setNote('Noise print captured.');
    } catch (err) {
      fail(err, 'Capture failed');
    } finally {
      setBusy(null);
    }
  };

  if (missing) {
    return (
      <div className="audio-card audio-noise">
        <div className="audio-card-title">Noise print</div>
        <div className="audio-note">Detect noise and Capture Noise Print are not available on this backend yet.</div>
      </div>
    );
  }

  return (
    <div className="audio-card audio-noise">
      <div className="audio-card-title">Noise print</div>
      <div className="audio-noise-row">
        <button className="audio-btn" disabled={busy !== null} onClick={() => void detect()}>
          {busy === 'detect' ? 'Detecting…' : 'Detect noise'}
        </button>
        <label className="audio-inline">
          From
          <input className="audio-num" type="number" step="0.01" min={0} value={start}
            onChange={(e) => setStart(e.target.value)} />
        </label>
        <label className="audio-inline">
          to
          <input className="audio-num" type="number" step="0.01" min={0} value={end}
            onChange={(e) => setEnd(e.target.value)} />
          s
        </label>
        <button className="audio-btn primary" disabled={busy !== null} onClick={() => void capture()}>
          {busy === 'capture' ? 'Capturing…' : 'Capture Noise Print'}
        </button>
      </div>
      {floor !== null && <div className="text-xs text-muted">Level of that stretch: {floor.toFixed(1)} dBFS</div>}
      {print ? (
        <>
          <div className="text-xs text-muted">
            Print: {secs(print.start_s)}–{secs(print.end_s)} s
            {print.source_id ? ` of ${print.source_id}` : ''}
            {typeof print.floor_dbfs === 'number' ? ` · ${print.floor_dbfs.toFixed(1)} dBFS RMS` : ''}
          </div>
          <PrintCurve print={print} />
        </>
      ) : (
        <div className="text-xs text-muted">No noise print captured for this project yet.</div>
      )}
      <div className="audio-hint">
        {outputNoiseOnly
          ? 'Output Noise Only is on: Preview plays only what will be removed.'
          : 'Switch on Output Noise Only below and press Preview to hear just what will be removed.'}
      </div>
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}
