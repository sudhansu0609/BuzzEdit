/**
 * Editor for a "curve" parameter (FFT Filter's response, Dynamics Processing's transfer
 * curve): an SVG whose axes, ranges and units come from the catalogue (`x_min`, `x_max`,
 * `x_unit`, `x_log` for x; `min`, `max`, `unit` for y). Drag a point to move it, click empty
 * space to add one, double-click a point to remove it; the end points keep their x, as in
 * Audition. Changes stream out while dragging, so a live curve is heard as it moves.
 */

import { useRef, useState } from 'react';
import type { CurvePoint, ParamDef } from '../../hooks/audioApi';

const W = 320;
const H = 180;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 22;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;
const HIT_R = 8;
const HZ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

function toPoints(value: unknown): CurvePoint[] | null {
  if (!Array.isArray(value)) return null;
  const points = value
    .filter((p) => Array.isArray(p) && p.length >= 2 && Number.isFinite(Number(p[0])) && Number.isFinite(Number(p[1])))
    .map((p) => [Number(p[0]), Number(p[1])] as CurvePoint)
    .sort((a, b) => a[0] - b[0]);
  return points.length >= 2 ? points : null;
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(parseFloat(v.toFixed(6)));
  return out;
}

const fmtHz = (f: number) => (f >= 1000 ? `${parseFloat((f / 1000).toFixed(1))}k` : `${Math.round(f)}`);

export default function CurveEditor({ def, value, onChange, disabled }: {
  def: ParamDef;
  value: unknown;
  onChange: (points: CurvePoint[]) => void;
  disabled?: boolean;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<number | null>(null);
  const latest = useRef<CurvePoint[]>([]);
  const [hover, setHover] = useState<number | null>(null);

  const xMin = def.x_min ?? 0;
  const xMax = def.x_max ?? 1;
  const log = !!def.x_log && xMin > 0 && xMax > xMin;
  const yMin = def.min ?? -1;
  const yMax = def.max ?? 1;
  const xUnit = def.x_unit ?? '';
  const yUnit = def.unit ?? '';
  const fallback = toPoints(def.default) ?? [[xMin, 0], [xMax, 0]] as CurvePoint[];
  const points = toPoints(value) ?? fallback;
  // While dragging, the points just sent may not have come back as props yet.
  if (drag.current === null) latest.current = points;

  const xToPx = (x: number) => PAD_L + PLOT_W * (log
    ? Math.log(x / xMin) / Math.log(xMax / xMin)
    : (x - xMin) / ((xMax - xMin) || 1));
  const pxToX = (px: number) => {
    const f = Math.min(1, Math.max(0, (px - PAD_L) / PLOT_W));
    return log ? xMin * Math.pow(xMax / xMin, f) : xMin + f * (xMax - xMin);
  };
  const yToPx = (y: number) => PAD_T + PLOT_H * (1 - (y - yMin) / ((yMax - yMin) || 1));
  const pxToY = (py: number) => {
    const f = Math.min(1, Math.max(0, 1 - (py - PAD_T) / PLOT_H));
    return yMin + f * (yMax - yMin);
  };
  const roundX = (x: number) => (log ? Math.round(x) : Math.round(x * 10) / 10);
  const roundY = (y: number) => Math.round(y * 10) / 10;
  const fmtX = (x: number) => (log ? fmtHz(x) : `${parseFloat(x.toFixed(1))}`);

  const emit = (next: CurvePoint[]) => {
    latest.current = next;
    onChange(next);
  };

  const local = (e: { clientX: number; clientY: number }): [number, number] => {
    const rect = svg.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return [0, 0];
    return [(e.clientX - rect.left) * (W / rect.width), (e.clientY - rect.top) * (H / rect.height)];
  };

  const nearest = (pts: CurvePoint[], px: number, py: number) => {
    let hit = -1;
    let best = HIT_R * HIT_R;
    pts.forEach(([x, y], i) => {
      const dx = xToPx(x) - px;
      const dy = yToPx(y) - py;
      const d = dx * dx + dy * dy;
      if (d <= best) { best = d; hit = i; }
    });
    return hit;
  };

  const moveTo = (index: number, px: number, py: number) => {
    const next = latest.current.map((p) => [p[0], p[1]] as CurvePoint);
    if (!next[index]) return;
    const last = next.length - 1;
    let x = next[index][0];
    if (index > 0 && index < last) {
      const eps = log ? 1 : 0.1;
      const lo = next[index - 1][0] + eps;
      const hi = next[index + 1][0] - eps;
      if (hi > lo) x = Math.min(hi, Math.max(lo, roundX(pxToX(px))));
    }
    next[index] = [x, Math.min(yMax, Math.max(yMin, roundY(pxToY(py))))];
    emit(next);
  };

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (disabled || e.button !== 0) return;
    const [px, py] = local(e);
    let hit = nearest(latest.current, px, py);
    if (hit < 0) {
      if (px < PAD_L || px > W - PAD_R || py < PAD_T || py > H - PAD_B) return;
      const x = roundX(pxToX(px));
      if (latest.current.some((p) => p[0] === x)) return;
      const next = [...latest.current.map((p) => [p[0], p[1]] as CurvePoint), [x, roundY(pxToY(py))] as CurvePoint]
        .sort((a, b) => a[0] - b[0]);
      hit = next.findIndex((p) => p[0] === x);
      emit(next);
    }
    drag.current = hit;
    setHover(hit);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const [px, py] = local(e);
    if (drag.current !== null) {
      moveTo(drag.current, px, py);
      return;
    }
    const hit = nearest(points, px, py);
    setHover(hit >= 0 ? hit : null);
  };

  const endDrag = (e: React.PointerEvent<SVGSVGElement>) => {
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const onDoubleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (disabled) return;
    const [px, py] = local(e);
    const hit = nearest(latest.current, px, py);
    if (hit <= 0 || hit >= latest.current.length - 1) return; // the end points stay
    emit(latest.current.filter((_, i) => i !== hit));
    setHover(null);
  };

  const xTicks = log ? HZ_TICKS.filter((f) => f >= xMin && f <= xMax) : niceTicks(xMin, xMax, 5);
  const yTicks = niceTicks(yMin, yMax, 4);
  const line = points.map(([x, y]) => `${xToPx(x).toFixed(1)},${yToPx(y).toFixed(1)}`).join(' ');
  // A transfer curve (same unit and range on both axes) gets the unity line for reference.
  const unity = xUnit === yUnit && xMin === yMin && xMax === yMax;
  const shown = drag.current ?? hover;
  const readout = shown !== null && points[shown]
    ? `${fmtX(points[shown][0])} ${xUnit} → ${points[shown][1].toFixed(1)} ${yUnit}`
    : 'Drag a point · click to add · double-click to remove';

  return (
    <div className="audio-curve">
      <div className="audio-curve-head">
        <span className="insp-label">{def.label}</span>
        <span className="audio-unit">x {xUnit}{log ? ' (log)' : ''} · y {yUnit}</span>
        <button className="audio-btn" disabled={disabled} onClick={() => emit(fallback)}>Reset</button>
      </div>
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} className={`audio-curve-svg${disabled ? ' disabled' : ''}`}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag}
        onPointerCancel={endDrag} onPointerLeave={() => { if (drag.current === null) setHover(null); }}
        onDoubleClick={onDoubleClick}>
        <rect x={PAD_L} y={PAD_T} width={PLOT_W} height={PLOT_H} className="audio-curve-bg" />
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={xToPx(t)} x2={xToPx(t)} y1={PAD_T} y2={PAD_T + PLOT_H} className="audio-grid" />
            <text x={xToPx(t)} y={H - 7} textAnchor="middle" className="audio-axis">{fmtX(t)}</text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD_L} x2={PAD_L + PLOT_W} y1={yToPx(t)} y2={yToPx(t)} className="audio-grid" />
            <text x={PAD_L - 4} y={yToPx(t) + 3} textAnchor="end" className="audio-axis">{t}</text>
          </g>
        ))}
        {unity && (
          <line x1={xToPx(xMin)} y1={yToPx(yMin)} x2={xToPx(xMax)} y2={yToPx(yMax)} className="audio-curve-unity" />
        )}
        <polyline points={line} className="audio-curve-line" />
        {points.map(([x, y], i) => (
          <circle key={i} cx={xToPx(x)} cy={yToPx(y)} r={shown === i ? 5 : 4}
            className={`audio-curve-point${i === 0 || i === points.length - 1 ? ' end' : ''}${shown === i ? ' hot' : ''}`} />
        ))}
      </svg>
      <div className="audio-curve-readout">{readout}</div>
    </div>
  );
}
