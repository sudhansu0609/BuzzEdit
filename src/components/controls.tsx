/**
 * The small form controls the editing panels are built from.
 *
 * These grew up inside InspectorPanel; the Effects and Transitions panels need
 * the same row/slider/section vocabulary, and three copies of a slider that all
 * drift apart is how a panel ends up looking like it belongs to a different app.
 */

import { useState } from 'react';

/** Resolve any CSS colour name to #rrggbb so <input type="color"> can show it. */
export function toHex(value: string): string {
  const base = (value || '#ffffff').split('@')[0].trim();
  if (/^#[0-9a-f]{6}$/i.test(base)) return base.toLowerCase();
  const ctx = document.createElement('canvas').getContext('2d');
  if (!ctx) return '#ffffff';
  ctx.fillStyle = '#ffffff';
  ctx.fillStyle = base;
  return ctx.fillStyle as string;
}

export function alphaOf(value: string): number {
  const parts = (value || '').split('@');
  return parts.length > 1 ? Math.max(0, Math.min(1, parseFloat(parts[1]) || 0)) : 1;
}

export const composeColor = (hex: string, alpha: number) =>
  alpha >= 1 ? hex : `${hex}@${alpha.toFixed(2)}`;

export function Row({ label, children, hint }: {
  label: string; children: React.ReactNode; hint?: string;
}) {
  return (
    <label className="insp-row" title={hint}>
      <span className="insp-label">{label}</span>
      <div className="insp-control">{children}</div>
    </label>
  );
}

export function Slider({ label, value, min, max, step = 0.01, onChange, format, hint }: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; format?: (v: number) => string; hint?: string;
}) {
  return (
    <Row label={label} hint={hint}>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
      <span className="insp-value">{format ? format(value) : value.toFixed(2)}</span>
    </Row>
  );
}

export function ColorField({ label, value, onChange, allowAlpha = true }: {
  label: string; value: string; onChange: (v: string) => void; allowAlpha?: boolean;
}) {
  const hex = toHex(value);
  const alpha = alphaOf(value);
  return (
    <Row label={label}>
      <input type="color" value={hex} onChange={(e) => onChange(composeColor(e.target.value, alpha))} />
      {allowAlpha && (
        <>
          <input type="range" min={0} max={1} step={0.05} value={alpha}
            onChange={(e) => onChange(composeColor(hex, parseFloat(e.target.value)))} />
          <span className="insp-value">{Math.round(alpha * 100)}%</span>
        </>
      )}
    </Row>
  );
}

export function Section({ title, children, defaultOpen = true, action, onToggle }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean; action?: React.ReactNode;
  onToggle?: (open: boolean) => void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="insp-section">
      <div className="insp-section-head">
        <button className="insp-section-toggle" onClick={() => {
          const next = !open;
          setOpen(next);
          onToggle?.(next);
        }}>
          <span className={`insp-caret ${open ? 'open' : ''}`}>▸</span>{title}
        </button>
        {action}
      </div>
      {open && <div className="insp-section-body">{children}</div>}
    </div>
  );
}

export function PresetChips({ items, active, onPick }: {
  items: { id: string; label: string }[]; active?: string | null; onPick: (id: string) => void;
}) {
  return (
    <div className="insp-chips">
      {items.map((p) => (
        <button key={p.id} className={`insp-chip ${active === p.id ? 'active' : ''}`}
          onClick={() => onPick(p.id)}>{p.label}</button>
      ))}
    </div>
  );
}
