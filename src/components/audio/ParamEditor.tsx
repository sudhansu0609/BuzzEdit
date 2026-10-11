/**
 * Parameter editors generated from the catalogue (AUDIO_SUITE_PLAN.md §2.2): every ParamDef
 * becomes a control by its type -- number (slider + numeric entry; wide Hz ranges travel in
 * log), choice, bool, curve (CurveEditor), bands (BandsEditor). Two layouts come from the data
 * alone: a `<x>_on` switch heads the `<x>_…` params that follow it (the Parametric EQ's bands
 * and filters, the Notch Filter's notches), and a run of `band_<n>` numbers is drawn as a
 * fader bank (the graphic EQs), limited to ± half of the effect's `range` when it has one.
 */

import { useMemo, useRef, useState } from 'react';
import type { BandValue, CurvePoint, EffectDef, ParamDef } from '../../hooks/audioApi';
import { Row } from '../controls';
import CurveEditor from './CurveEditor';
import BandsEditor from './BandsEditor';

type OnChange = (key: string, value: unknown) => void;

const LOG_STEPS = 1000;
const FADER_KEY = /^band_\d+$/;
const MIN_FADERS = 5;

export function decimals(step: number | null | undefined): number {
  if (!step || step >= 1) return 0;
  return Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
}

/** Round to the parameter's step and clamp to its range (or to `lo`/`hi`). */
export function snapValue(value: number, def: ParamDef, lo?: number, hi?: number): number {
  const min = lo ?? def.min ?? -Infinity;
  const max = hi ?? def.max ?? Infinity;
  let v = value;
  if (def.step && def.step > 0) v = Math.round(v / def.step) * def.step;
  v = Math.min(max, Math.max(min, v));
  return parseFloat(v.toFixed(decimals(def.step)));
}

const isLog = (def: ParamDef) =>
  def.unit === 'Hz' && (def.min ?? 0) > 0 && (def.max ?? 0) / (def.min as number) >= 20;

const numberOr = (value: unknown, fallback: number) =>
  (typeof value === 'number' && Number.isFinite(value) ? value : fallback);

const liveHint = (def: ParamDef) =>
  (def.live ? `${def.label}: heard live as it moves` : `${def.label}: the preview restarts at the playhead`);

export function NumberParam({ def, value, onChange, label, disabled }: {
  def: ParamDef; value: unknown; onChange: (v: number) => void; label?: string; disabled?: boolean;
}) {
  const min = def.min ?? 0;
  const max = def.max ?? 1;
  const fallback = numberOr(def.default, min);
  const v = numberOr(value, fallback);
  const log = isLog(def);
  const places = decimals(def.step);
  const [text, setText] = useState<string | null>(null);

  const pos = log ? Math.round((Math.log(Math.max(v, min) / min) / Math.log(max / min)) * LOG_STEPS) : v;
  const fromPos = (p: number) => snapValue(log ? min * Math.pow(max / min, p / LOG_STEPS) : p, def);
  const commit = () => {
    if (text === null) return;
    const n = parseFloat(text);
    if (Number.isFinite(n)) onChange(snapValue(n, def));
    setText(null);
  };

  return (
    <Row label={label ?? def.label} hint={liveHint(def)}>
      <input type="range" min={log ? 0 : min} max={log ? LOG_STEPS : max} step="any" value={pos}
        disabled={disabled}
        onChange={(e) => onChange(fromPos(parseFloat(e.target.value)))}
        onDoubleClick={() => onChange(fallback)} />
      <input type="number" className="audio-num" step="any" min={min} max={max} disabled={disabled}
        value={text ?? v.toFixed(places)}
        onChange={(e) => {
          setText(e.target.value);
          const n = parseFloat(e.target.value);
          if (Number.isFinite(n) && n >= min && n <= max) onChange(n);
        }}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit();
          if (e.key === 'Escape') setText(null);
        }} />
      {def.unit && <span className="audio-unit">{def.unit}</span>}
    </Row>
  );
}

export function ChoiceParam({ def, value, onChange, label, disabled }: {
  def: ParamDef; value: unknown; onChange: (v: unknown) => void; label?: string; disabled?: boolean;
}) {
  const choices = def.choices ?? [];
  const current = String(value ?? def.default ?? '');
  return (
    <Row label={label ?? def.label} hint={liveHint(def)}>
      <select value={current} disabled={disabled}
        onChange={(e) => {
          // choices may be numbers (FFT size): hand back the catalogue's own value
          const hit = choices.find((c) => String(c) === e.target.value);
          onChange(hit ?? e.target.value);
        }}>
        {!choices.some((c) => String(c) === current) && <option value={current}>{current}</option>}
        {choices.map((c) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
      </select>
      {def.unit && <span className="audio-unit">{def.unit}</span>}
    </Row>
  );
}

export function BoolParam({ def, value, onChange, label, disabled }: {
  def: ParamDef; value: unknown; onChange: (v: boolean) => void; label?: string; disabled?: boolean;
}) {
  const on = typeof value === 'boolean' ? value : !!def.default;
  return (
    <Row label={label ?? def.label} hint={liveHint(def)}>
      <input type="checkbox" checked={on} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
    </Row>
  );
}

// -- graphic EQ faders ---------------------------------------------------------------------

function Fader({ def, value, lo, hi, onChange, disabled }: {
  def: ParamDef; value: unknown; lo: number; hi: number; onChange: (v: number) => void; disabled?: boolean;
}) {
  const track = useRef<HTMLDivElement>(null);
  const fallback = numberOr(def.default, 0);
  const v = numberOr(value, fallback);
  const span = hi - lo || 1;
  const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
  const frac = clamp01((v - lo) / span);
  const zero = clamp01((0 - lo) / span);
  const nudge = (def.step || 0.1) * 5;

  const fromY = (clientY: number) => {
    const rect = track.current?.getBoundingClientRect();
    if (!rect || rect.height <= 0) return;
    onChange(snapValue(lo + clamp01(1 - (clientY - rect.top) / rect.height) * span, def, lo, hi));
  };

  return (
    <div className={`audio-fader${disabled ? ' disabled' : ''}`} title={`${def.label}: ${v.toFixed(1)} ${def.unit}`}>
      <div className="audio-fader-value">{v > 0 ? '+' : ''}{v.toFixed(1)}</div>
      <div className="audio-fader-track" ref={track} tabIndex={disabled ? -1 : 0}
        onPointerDown={(e) => {
          if (disabled || e.button !== 0) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          fromY(e.clientY);
        }}
        onPointerMove={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) fromY(e.clientY);
        }}
        onDoubleClick={() => { if (!disabled) onChange(snapValue(fallback, def, lo, hi)); }}
        onKeyDown={(e) => {
          if (disabled) return;
          if (e.key === 'ArrowUp') { e.preventDefault(); onChange(snapValue(v + nudge, def, lo, hi)); }
          if (e.key === 'ArrowDown') { e.preventDefault(); onChange(snapValue(v - nudge, def, lo, hi)); }
        }}>
        <div className="audio-fader-zero" style={{ bottom: `${zero * 100}%` }} />
        <div className="audio-fader-fill"
          style={{ bottom: `${Math.min(frac, zero) * 100}%`, height: `${Math.abs(frac - zero) * 100}%` }} />
        <div className="audio-fader-thumb" style={{ bottom: `${frac * 100}%` }} />
      </div>
      <div className="audio-fader-label">{def.label.replace(/ kHz$/, 'k').replace(/ Hz$/, '')}</div>
    </div>
  );
}

function FaderBank({ params, values, limit, onChange, disabled }: {
  params: ParamDef[]; values: Record<string, unknown>; limit: number | null; onChange: OnChange; disabled?: boolean;
}) {
  return (
    <div className="audio-faders">
      {params.map((p) => {
        const lo = Math.max(p.min ?? -24, limit != null ? -limit : -Infinity);
        const hi = Math.min(p.max ?? 24, limit != null ? limit : Infinity);
        return (
          <Fader key={p.key} def={p} value={values[p.key]} lo={lo} hi={hi} disabled={disabled}
            onChange={(v) => onChange(p.key, v)} />
        );
      })}
    </div>
  );
}

// -- layout --------------------------------------------------------------------------------

type Block =
  | { kind: 'one'; param: ParamDef }
  | { kind: 'group'; head: ParamDef; params: ParamDef[] }
  | { kind: 'faders'; params: ParamDef[] };

const isSwitch = (p: ParamDef) => p.type === 'bool' && p.key.endsWith('_on');

export function layoutParams(params: ParamDef[]): Block[] {
  const blocks: Block[] = [];
  let i = 0;
  while (i < params.length) {
    const p = params[i];
    if (p.type === 'number' && FADER_KEY.test(p.key)) {
      let j = i;
      while (j < params.length && params[j].type === 'number' && FADER_KEY.test(params[j].key)) j++;
      const run = params.slice(i, j);
      if (run.length >= MIN_FADERS) blocks.push({ kind: 'faders', params: run });
      else run.forEach((param) => blocks.push({ kind: 'one', param }));
      i = j;
      continue;
    }
    if (isSwitch(p)) {
      const prefix = p.key.slice(0, -2); // band1_on -> band1_
      let j = i + 1;
      while (j < params.length && params[j].key.startsWith(prefix) && !isSwitch(params[j])) j++;
      if (j > i + 1) {
        blocks.push({ kind: 'group', head: p, params: params.slice(i + 1, j) });
        i = j;
        continue;
      }
    }
    blocks.push({ kind: 'one', param: p });
    i++;
  }
  return blocks;
}

/** "Band 1 Frequency" under the "Band 1" switch reads "Frequency". */
function shortLabel(label: string, head: string): string {
  if (!label.startsWith(`${head} `)) return label;
  const rest = label.slice(head.length + 1);
  return rest.charAt(0).toUpperCase() + rest.slice(1);
}

function renderParam(p: ParamDef, values: Record<string, unknown>, onChange: OnChange,
  disabled: boolean | undefined, label?: string) {
  const value = values[p.key] !== undefined ? values[p.key] : p.default;
  switch (p.type) {
    case 'number':
      return <NumberParam key={p.key} def={p} value={value} label={label} disabled={disabled}
        onChange={(v) => onChange(p.key, v)} />;
    case 'choice':
      return <ChoiceParam key={p.key} def={p} value={value} label={label} disabled={disabled}
        onChange={(v) => onChange(p.key, v)} />;
    case 'bool':
      return <BoolParam key={p.key} def={p} value={value} label={label} disabled={disabled}
        onChange={(v) => onChange(p.key, v)} />;
    case 'curve':
      return <CurveEditor key={p.key} def={p} value={value} disabled={disabled}
        onChange={(v: CurvePoint[]) => onChange(p.key, v)} />;
    case 'bands':
      return <BandsEditor key={p.key} def={p} value={value} disabled={disabled}
        onChange={(v: BandValue[]) => onChange(p.key, v)} />;
    default:
      return (
        <div key={p.key} className="audio-note">
          {p.label}: parameter type "{String((p as ParamDef).type)}" has no editor yet.
        </div>
      );
  }
}

export default function ParamEditor({ def, values, onChange, disabled }: {
  def: EffectDef;
  values: Record<string, unknown>;
  onChange: OnChange;
  disabled?: boolean;
}) {
  const blocks = useMemo(() => layoutParams(def.params || []), [def]);
  const rangeDef = (def.params || []).find((p) => p.key === 'range' && p.type === 'number');
  const range = rangeDef ? numberOr(values.range, numberOr(rangeDef.default, 0)) : 0;
  const limit = range > 0 ? range / 2 : null;

  if (!def.params || def.params.length === 0) {
    return <div className="text-xs text-muted">This effect has no settings.</div>;
  }

  return (
    <div className="audio-params">
      {blocks.map((block) => {
        if (block.kind === 'faders') {
          return (
            <FaderBank key={`faders:${block.params[0].key}`} params={block.params} values={values}
              limit={limit} onChange={onChange} disabled={disabled} />
          );
        }
        if (block.kind === 'group') {
          const headValue = values[block.head.key] !== undefined ? values[block.head.key] : block.head.default;
          const on = typeof headValue === 'boolean' ? headValue : !!block.head.default;
          return (
            <div key={`group:${block.head.key}`} className={`audio-param-group${on ? '' : ' off'}`}>
              <label className="audio-group-head" title={liveHint(block.head)}>
                <input type="checkbox" checked={on} disabled={disabled}
                  onChange={(e) => onChange(block.head.key, e.target.checked)} />
                {block.head.label}
              </label>
              {block.params.map((p) => renderParam(p, values, onChange, disabled, shortLabel(p.label, block.head.label)))}
            </div>
          );
        }
        return renderParam(block.param, values, onChange, disabled);
      })}
    </div>
  );
}
