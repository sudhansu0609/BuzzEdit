/**
 * Meters fed by the live engine (`onMeters`, every 50 ms; computed from the PCM being played in
 * `src/audio/meters.ts`): sample peak and RMS bars in dBFS, momentary (400 ms) and short-term
 * (3 s) loudness in LUFS (ITU-R BS.1770), and a Loudness-Radar-style ring -- one wedge of
 * short-term loudness every half second, a full turn per 30 s, against a -14 LUFS ring.
 * Repaints at most ten times a second; the peak shown is the highest since the last repaint.
 */

import { useEffect, useRef, useState } from 'react';
import type { MeterReading } from '../../audio/LiveEngine';
import type { Signal } from '../../hooks/audioApi';

const FLOOR_DB = -60;
const SPOKES = 60;
const SPOKE_MS = 500;
const PAINT_MS = 100;
const TARGET_LUFS = -14;
const RADAR_FLOOR = -50;
const SCALE = [-60, -40, -20, -12, -6, 0];

const fmtDb = (v: number | undefined) => (v !== undefined && Number.isFinite(v) ? v.toFixed(1) : '-∞');
const level = (db: number, floor: number) =>
  (Number.isFinite(db) ? Math.min(1, Math.max(0, (db - floor) / -floor)) : 0);

function MeterBar({ label, db }: { label: string; db: number }) {
  return (
    <div className="audio-meter">
      <span className="audio-meter-label">{label}</span>
      <div className="audio-meter-track">
        <div className="audio-meter-mask" style={{ left: `${level(db, FLOOR_DB) * 100}%` }} />
      </div>
      <span className="audio-meter-value">{fmtDb(db)}</span>
    </div>
  );
}

function wedge(cx: number, cy: number, r: number, a0: number, a1: number): string {
  const x0 = cx + r * Math.cos(a0);
  const y0 = cy + r * Math.sin(a0);
  const x1 = cx + r * Math.cos(a1);
  const y1 = cy + r * Math.sin(a1);
  return `M${cx},${cy} L${x0.toFixed(2)},${y0.toFixed(2)} A${r.toFixed(2)},${r.toFixed(2)} 0 0 1 ${x1.toFixed(2)},${y1.toFixed(2)} Z`;
}

function Radar({ spokes, cursor, momentary }: { spokes: Array<number | null>; cursor: number; momentary?: number }) {
  const C = 42;
  const R = 38;
  const radius = (lufs: number) => level(lufs, RADAR_FLOOR) * R;
  const step = (2 * Math.PI) / SPOKES;
  const angle = (i: number) => i * step - Math.PI / 2;
  const sweep = angle(cursor);
  return (
    <svg className="audio-radar" viewBox="0 0 84 84" aria-label="Short-term loudness radar">
      <circle cx={C} cy={C} r={R} className="audio-radar-ring" />
      <circle cx={C} cy={C} r={radius(TARGET_LUFS)} className="audio-radar-target" />
      {spokes.map((v, i) => (v === null ? null : (
        <path key={i} d={wedge(C, C, Math.max(0.5, radius(v)), angle(i), angle(i) + step)}
          className={`audio-radar-wedge ${v > TARGET_LUFS + 2 ? 'hot' : v >= TARGET_LUFS - 2 ? 'ok' : 'low'}`} />
      )))}
      <line x1={C} y1={C} x2={C + R * Math.cos(sweep)} y2={C + R * Math.sin(sweep)} className="audio-radar-sweep" />
      <text x={C} y={C + 3} textAnchor="middle" className="audio-radar-text">{fmtDb(momentary)}</text>
    </svg>
  );
}

export default function Meters({ signal }: { signal: Signal<MeterReading> }) {
  const [reading, setReading] = useState<MeterReading | null>(null);
  const spokes = useRef<Array<number | null>>(new Array(SPOKES).fill(null));
  const cursor = useRef(0);
  const lastSpoke = useRef(0);
  const lastPaint = useRef(0);
  const peakSince = useRef(-Infinity);

  useEffect(() => signal.subscribe((m) => {
    const now = performance.now();
    if (m.peak > peakSince.current) peakSince.current = m.peak;
    if (now - lastSpoke.current >= SPOKE_MS) {
      lastSpoke.current = now;
      spokes.current[cursor.current] = Number.isFinite(m.shortTermLufs) ? m.shortTermLufs : null;
      cursor.current = (cursor.current + 1) % SPOKES;
    }
    if (now - lastPaint.current >= PAINT_MS) {
      lastPaint.current = now;
      setReading({ ...m, peak: peakSince.current });
      peakSince.current = -Infinity;
    }
  }), [signal]);

  return (
    <div className="audio-meters">
      <div className="audio-meter-bars">
        <MeterBar label="Peak" db={reading?.peak ?? -Infinity} />
        <MeterBar label="RMS" db={reading?.rms ?? -Infinity} />
        <div className="audio-meter-scale">
          {SCALE.map((db) => (
            <span key={db} style={{ left: `${level(db, FLOOR_DB) * 100}%` }}>{db}</span>
          ))}
        </div>
      </div>
      <div className="audio-lufs">
        <div><span className="audio-lufs-label">Momentary</span><span className="audio-lufs-value">{fmtDb(reading?.momentaryLufs)}</span> LUFS</div>
        <div><span className="audio-lufs-label">Short-term</span><span className="audio-lufs-value">{fmtDb(reading?.shortTermLufs)}</span> LUFS</div>
        <div className="text-xs text-muted">dBFS bars · ring = short-term vs {TARGET_LUFS} LUFS</div>
      </div>
      <Radar spokes={spokes.current} cursor={cursor.current} momentary={reading?.momentaryLufs} />
    </div>
  );
}
