/**
 * Table editor for a "bands" parameter (the Multiband Compressor's bands): one row per band,
 * one column per ParamDef in the catalogue's `fields`, each clamped to that field's range.
 */

import { useState } from 'react';
import type { BandValue, ParamDef } from '../../hooks/audioApi';

function toBands(value: unknown): BandValue[] | null {
  if (!Array.isArray(value)) return null;
  return value.filter((b) => b && typeof b === 'object' && !Array.isArray(b)) as BandValue[];
}

const clamp = (n: number, field: ParamDef) =>
  Math.min(field.max ?? Infinity, Math.max(field.min ?? -Infinity, n));

function NumberCell({ field, value, onChange, disabled }: {
  field: ParamDef; value: unknown; onChange: (v: number) => void; disabled?: boolean;
}) {
  const [text, setText] = useState<string | null>(null);
  const fallback = typeof field.default === 'number' ? field.default : 0;
  const v = typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  const commit = () => {
    if (text === null) return;
    const n = parseFloat(text);
    if (Number.isFinite(n)) onChange(clamp(n, field));
    setText(null);
  };
  return (
    <input type="number" className="audio-num" step="any" disabled={disabled}
      min={field.min ?? undefined} max={field.max ?? undefined}
      title={`${field.label}: ${field.min ?? ''}…${field.max ?? ''} ${field.unit}`}
      value={text ?? String(v)}
      onChange={(e) => {
        setText(e.target.value);
        const n = parseFloat(e.target.value);
        if (Number.isFinite(n) && n === clamp(n, field)) onChange(n);
      }}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit();
        if (e.key === 'Escape') setText(null);
      }} />
  );
}

function Cell({ field, value, onChange, disabled }: {
  field: ParamDef; value: unknown; onChange: (v: number | string | boolean) => void; disabled?: boolean;
}) {
  if (field.type === 'bool') {
    const on = typeof value === 'boolean' ? value : !!field.default;
    return <input type="checkbox" checked={on} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />;
  }
  if (field.type === 'choice') {
    const choices = field.choices ?? [];
    return (
      <select className="audio-select" value={String(value ?? field.default ?? '')} disabled={disabled}
        onChange={(e) => {
          const hit = choices.find((c) => String(c) === e.target.value);
          onChange(hit ?? e.target.value);
        }}>
        {choices.map((c) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
      </select>
    );
  }
  return <NumberCell field={field} value={value} onChange={onChange} disabled={disabled} />;
}

export default function BandsEditor({ def, value, onChange, disabled }: {
  def: ParamDef;
  value: unknown;
  onChange: (bands: BandValue[]) => void;
  disabled?: boolean;
}) {
  const fields = def.fields ?? [];
  const bands = toBands(value) ?? toBands(def.default) ?? [];
  const set = (index: number, key: string, v: number | string | boolean) =>
    onChange(bands.map((band, i) => (i === index ? { ...band, [key]: v } : band)));

  return (
    <div className="audio-bands">
      <div className="insp-label">{def.label}</div>
      <table className="audio-bands-table">
        <thead>
          <tr>
            <th>#</th>
            {fields.map((f) => (
              <th key={f.key} title={f.label}>
                {f.label}{f.unit ? <span className="audio-unit"> {f.unit}</span> : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bands.map((band, index) => (
            <tr key={index}>
              <td className="audio-band-index">{index + 1}</td>
              {fields.map((f) => (
                <td key={f.key}>
                  <Cell field={f} value={band[f.key]} disabled={disabled}
                    onChange={(v) => set(index, f.key, v)} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
