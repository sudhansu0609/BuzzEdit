/**
 * Essential Sound -> Dialogue (AUDIO_SUITE_PLAN.md §2.4, brief A4 part b): Audition's quick card
 * for speech. Every control owns ONE effect in the current scope's rack, tagged
 * `preset: "essential:<control>"`, so moving it again updates that effect instead of adding
 * another; the panel's own rack path does the rest (heard live, saved with the rack). The tick
 * of a control bypasses its effect or switches it back on; the rack's ✕ removes it.
 *
 *   Repair    Reduce Noise (Adaptive or RNNoise, 0-10), Reduce Rumble (high-pass 40-160 Hz),
 *             DeHum (50 / 60 Hz), DeEss (threshold), Reduce Reverb (DeepFilterNet, a process)
 *   Clarity   Dynamics (Speech Volume Leveler amount), EQ (Parametric EQ preset),
 *             Enhance Speech (Vocal Enhancer, Male / Female + amount)
 *   Creative  Reverb (Studio Reverb preset + mix; hidden when the catalogue has no `reverb`)
 *   Clip Volume (Amplify, dB)
 *
 * A control whose effect is not in the catalogue is left out.
 */

import { useState } from 'react';
import { AudioEffect, EffectDef, ParamDef, cloneValue, defaultParams } from '../../hooks/audioApi';
import { Row } from '../controls';
import { ChoiceParam, NumberParam, snapValue } from './ParamEditor';

export const ESSENTIAL_TAG = 'essential:';

type Values = Record<string, unknown>;

/** Reduce Noise's two live engines; the 0-10 amount drives `key` in steps of `perStep`. */
const NOISE_ENGINES = [
  { id: 'adaptive_noise_reduction', label: 'Adaptive', key: 'reduce_by', perStep: 3 }, // 0-30 dB
  { id: 'noise_reduction_rnnoise', label: 'RNNoise', key: 'mix', perStep: 10 }, // 0-100 %
];

const RUMBLE_MIN_HZ = 40;
const RUMBLE_MAX_HZ = 160;
/** Named in the brief; listed first when the catalogue ships them. */
const PREFERRED_EQ = ['Podcast Voice', 'Broadcast', 'Warm', 'Clean'];

const num = (v: unknown, fallback: number) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** The preset whose values (over the defaults) the params hold, ignoring `skip`. */
function presetOf(def: EffectDef, params: Values, skip?: string): string | null {
  const base = defaultParams(def);
  for (const [name, values] of Object.entries(def.presets || {})) {
    const want: Values = { ...base, ...values };
    const same = (def.params || []).every((p) => p.key === skip
      || JSON.stringify(params[p.key] ?? p.default) === JSON.stringify(want[p.key]));
    if (same) return name;
  }
  return null;
}

function presetNames(def: EffectDef, preferred: string[] = []): string[] {
  const names = Object.keys(def.presets || {});
  return [...preferred.filter((n) => names.includes(n)), ...names.filter((n) => !preferred.includes(n))];
}

function Control({ title, hint, fx, disabled, onSwitch, children }: {
  title: string;
  hint: string;
  fx: AudioEffect | null;
  disabled: boolean;
  onSwitch: (on: boolean) => void;
  children: React.ReactNode;
}) {
  const on = !!fx && fx.enabled;
  return (
    <div className={`audio-param-group${on ? '' : ' off'}`}>
      <label className="audio-group-head" title={hint}>
        <input type="checkbox" checked={on} disabled={disabled} onChange={(e) => onSwitch(e.target.checked)} />
        {title}
        {fx && !fx.enabled && <span className="audio-unit">bypassed</span>}
      </label>
      {children}
    </div>
  );
}

function PresetSelect({ names, current, disabled, onPick }: {
  names: string[];
  /** '' = no effect yet, null = values that match no preset */
  current: string | null;
  disabled: boolean;
  onPick: (name: string) => void;
}) {
  const value = current === null ? '__custom__' : current;
  return (
    <Row label="Preset">
      <select className="audio-select" value={value} disabled={disabled || names.length === 0}
        onChange={(e) => { if (names.includes(e.target.value)) onPick(e.target.value); }}>
        {value === '' && <option value="">Choose…</option>}
        {value === '__custom__' && <option value="__custom__">Custom</option>}
        {names.map((n) => <option key={n} value={n}>{n}</option>)}
      </select>
    </Row>
  );
}

export default function EssentialSound({
  rack, defs, scopeLabel, canEdit, onSet, onSwitch, reduceReverb,
}: {
  /** the current scope's rack */
  rack: AudioEffect[];
  defs: Record<string, EffectDef>;
  scopeLabel: string;
  /** false until the racks have loaded */
  canEdit: boolean;
  /** add or update the control's tagged effect with these values, switched on */
  onSet: (control: string, effectId: string, values: Values) => void;
  /** the tick: on adds the effect with `values` (or switches it back on), off bypasses it */
  onSwitch: (control: string, effectId: string, on: boolean, values: Values) => void;
  /** the DeepFilterNet process card for the range; null when that process is not available */
  reduceReverb: React.ReactNode;
}) {
  const [reverbOpen, setReverbOpen] = useState(false);
  const off = !canEdit;
  const tagged = (control: string) => rack.find((e) => e.preset === `${ESSENTIAL_TAG}${control}`) || null;
  const param = (effectId: string, key: string): ParamDef | undefined =>
    (defs[effectId]?.params || []).find((p) => p.key === key);

  if (Object.keys(defs).length === 0) {
    return <div className="audio-note">The effect catalogue has not loaded, so Essential Sound has nothing to drive.</div>;
  }

  // -- Repair -----------------------------------------------------------------------------
  const noiseFx = tagged('noise');
  const engines = NOISE_ENGINES.filter((e) => param(e.id, e.key));
  const engine = engines.find((e) => e.id === noiseFx?.effect) || engines[0];
  let noise: React.ReactNode = null;
  if (engine) {
    const p = param(engine.id, engine.key) as ParamDef;
    const amountDefault = clamp(num(p.default, 0) / engine.perStep, 0, 10);
    const amount = noiseFx && noiseFx.effect === engine.id
      ? clamp(num(noiseFx.params[engine.key], num(p.default, 0)) / engine.perStep, 0, 10)
      : amountDefault;
    const valuesFor = (e: typeof engine, a: number): Values => {
      const target = param(e.id, e.key) as ParamDef;
      return { [e.key]: snapValue(a * e.perStep, target) };
    };
    const amountDef: ParamDef = {
      key: 'amount', label: 'Amount', type: 'number', min: 0, max: 10, step: 0.1,
      default: amountDefault, unit: '', live: p.live,
    };
    const engineDef: ParamDef = {
      key: 'engine', label: 'Engine', type: 'choice', min: null, max: null, step: null,
      default: engines[0].label, unit: '', choices: engines.map((e) => e.label), live: false,
    };
    noise = (
      <Control title="Reduce Noise" fx={noiseFx} disabled={off}
        hint={`Live noise reduction: ${engines.map((e) => defs[e.id].name).join(' or ')}. 0-10 drives `
          + `${engine.key === 'reduce_by' ? 'Reduce Noise By (3 dB a step)' : 'Mix (10 % a step)'}.`}
        onSwitch={(on) => onSwitch('noise', engine.id, on, valuesFor(engine, amount))}>
        {engines.length > 1 && (
          <ChoiceParam def={engineDef} value={engine.label} disabled={off}
            onChange={(label) => {
              const next = engines.find((e) => e.label === label);
              if (next && next.id !== engine.id) onSet('noise', next.id, valuesFor(next, amount));
            }} />
        )}
        <NumberParam def={amountDef} value={amount} disabled={off}
          onChange={(a) => onSet('noise', engine.id, valuesFor(engine, a))} />
      </Control>
    );
  }

  const rumbleFx = tagged('rumble');
  const hp = param('parametric_eq', 'hp_frequency');
  let rumble: React.ReactNode = null;
  if (hp && param('parametric_eq', 'hp_on')) {
    const hz = clamp(num(rumbleFx?.params.hp_frequency, num(hp.default, 80)), RUMBLE_MIN_HZ, RUMBLE_MAX_HZ);
    const hzDef: ParamDef = {
      ...hp, label: 'Frequency', min: RUMBLE_MIN_HZ, max: RUMBLE_MAX_HZ, step: 1,
      default: clamp(num(hp.default, 80), RUMBLE_MIN_HZ, RUMBLE_MAX_HZ),
    };
    rumble = (
      <Control title="Reduce Rumble" fx={rumbleFx} disabled={off}
        hint="A high-pass filter (Parametric Equalizer) under the voice"
        onSwitch={(on) => onSwitch('rumble', 'parametric_eq', on, { hp_on: true, hp_frequency: hz })}>
        <NumberParam def={hzDef} value={hz} disabled={off}
          onChange={(v) => onSet('rumble', 'parametric_eq', { hp_on: true, hp_frequency: v })} />
      </Control>
    );
  }

  const humFx = tagged('dehum');
  const humHz = param('dehummer', 'frequency');
  let dehum: React.ReactNode = null;
  if (humHz) {
    const hz = num(humFx?.params.frequency, num(humHz.default, 50));
    const mainsDef: ParamDef = {
      ...humHz, label: 'Mains', type: 'choice', min: null, max: null, step: null,
      default: 50, choices: [50, 60],
    };
    dehum = (
      <Control title="DeHum" fx={humFx} disabled={off}
        hint="Notches out mains hum and its harmonics (DeHummer): 50 Hz in India and Europe, 60 Hz in the Americas"
        onSwitch={(on) => onSwitch('dehum', 'dehummer', on, { frequency: hz })}>
        <ChoiceParam def={mainsDef} value={hz} disabled={off}
          onChange={(v) => onSet('dehum', 'dehummer', { frequency: num(Number(v), 50) })} />
      </Control>
    );
  }

  const essFx = tagged('deess');
  const essThreshold = param('deesser', 'threshold');
  let deess: React.ReactNode = null;
  if (essThreshold) {
    const t = num(essFx?.params.threshold, num(essThreshold.default, -30));
    deess = (
      <Control title="DeEss" fx={essFx} disabled={off}
        hint="Turns down sibilance (DeEsser): a lower threshold catches more"
        onSwitch={(on) => onSwitch('deess', 'deesser', on, { threshold: t })}>
        <NumberParam def={essThreshold} value={t} disabled={off}
          onChange={(v) => onSet('deess', 'deesser', { threshold: v })} />
      </Control>
    );
  }

  const reverbProcess = (
    <div className="audio-param-group">
      <div className="audio-group-head">
        Reduce Reverb <span className="audio-badge process">process</span>
        <button className="audio-btn" style={{ marginLeft: 'auto' }} onClick={() => setReverbOpen((o) => !o)}>
          {reverbOpen ? 'Hide' : 'Open…'}
        </button>
      </div>
      <div className="audio-hint">
        DeepFilterNet, run as a process on the range its card shows (the selection, else the
        selected clip, else the whole recording). It writes a new version of the source, which
        Undo takes back; it is not heard live -- Preview renders it.
      </div>
      {reverbOpen && (reduceReverb
        || <div className="audio-note">The DeepFilterNet process is not available on this backend.</div>)}
    </div>
  );

  // -- Clarity ----------------------------------------------------------------------------
  const dynFx = tagged('dynamics');
  const leveler = param('speech_volume_leveler', 'amount');
  let dynamics: React.ReactNode = null;
  if (leveler) {
    const a = num(dynFx?.params.amount, num(leveler.default, 50));
    dynamics = (
      <Control title="Dynamics" fx={dynFx} disabled={off}
        hint="Evens out the speech level (Speech Volume Leveler)"
        onSwitch={(on) => onSwitch('dynamics', 'speech_volume_leveler', on, { amount: a })}>
        <NumberParam def={leveler} label="Amount" value={a} disabled={off}
          onChange={(v) => onSet('dynamics', 'speech_volume_leveler', { amount: v })} />
      </Control>
    );
  }

  const eqFx = tagged('eq');
  const peq = defs.parametric_eq;
  let eq: React.ReactNode = null;
  if (peq) {
    const names = presetNames(peq, PREFERRED_EQ);
    const presetValues = (name: string): Values => ({ ...defaultParams(peq), ...cloneValue(peq.presets[name] || {}) });
    eq = (
      <Control title="EQ" fx={eqFx} disabled={off}
        hint="A Parametric Equalizer preset from the catalogue; fine-tune it in the rack"
        onSwitch={(on) => onSwitch('eq', 'parametric_eq', on, names.length ? presetValues(names[0]) : {})}>
        <PresetSelect names={names} disabled={off}
          current={eqFx ? presetOf(peq, eqFx.params) : ''}
          onPick={(name) => onSet('eq', 'parametric_eq', presetValues(name))} />
      </Control>
    );
  }

  const enhanceFx = tagged('enhance');
  const mode = param('vocal_enhancer', 'mode');
  const enhanceAmount = param('vocal_enhancer', 'amount');
  let enhance: React.ReactNode = null;
  if (mode && enhanceAmount) {
    const voices = (mode.choices || []).filter((c) => c === 'Male' || c === 'Female');
    const modeDef: ParamDef = { ...mode, label: 'Voice', choices: voices.length ? voices : mode.choices };
    const m = enhanceFx?.params.mode ?? mode.default;
    const a = num(enhanceFx?.params.amount, num(enhanceAmount.default, 60));
    enhance = (
      <Control title="Enhance Speech" fx={enhanceFx} disabled={off}
        hint="One-knob voice polish (Vocal Enhancer) for a male or female voice"
        onSwitch={(on) => onSwitch('enhance', 'vocal_enhancer', on, { mode: m, amount: a })}>
        <ChoiceParam def={modeDef} value={m} disabled={off}
          onChange={(v) => onSet('enhance', 'vocal_enhancer', { mode: v })} />
        <NumberParam def={enhanceAmount} value={a} disabled={off}
          onChange={(v) => onSet('enhance', 'vocal_enhancer', { amount: v })} />
      </Control>
    );
  }

  // -- Creative ---------------------------------------------------------------------------
  const reverbFx = tagged('reverb');
  const studio = defs.reverb;
  let reverb: React.ReactNode = null;
  if (studio) {
    const mixKey = (studio.params || []).some((p) => p.key === 'mix') ? 'mix' : 'wet';
    const mixParam = param('reverb', mixKey);
    const names = presetNames(studio);
    const presetValues = (name: string): Values => ({ ...defaultParams(studio), ...cloneValue(studio.presets[name] || {}) });
    const mix = mixParam ? num(reverbFx?.params[mixKey], num(mixParam.default, 30)) : 0;
    reverb = (
      <Control title="Reverb" fx={reverbFx} disabled={off}
        hint={`A room around the voice (${studio.name}): pick a preset, then set the mix`}
        onSwitch={(on) => onSwitch('reverb', 'reverb', on, names.length ? presetValues(names[0]) : {})}>
        <PresetSelect names={names} disabled={off}
          current={reverbFx ? presetOf(studio, reverbFx.params, mixKey) : ''}
          onPick={(name) => onSet('reverb', 'reverb', presetValues(name))} />
        {mixParam && (
          <NumberParam def={mixParam} label="Mix" value={mix} disabled={off}
            onChange={(v) => onSet('reverb', 'reverb', { [mixKey]: v })} />
        )}
      </Control>
    );
  }

  // -- Clip Volume --------------------------------------------------------------------------
  const volumeFx = tagged('volume');
  const gain = param('amplify', 'gain');
  let volume: React.ReactNode = null;
  if (gain) {
    const g = num(volumeFx?.params.gain, num(gain.default, 0));
    volume = (
      <Control title="Clip Volume" fx={volumeFx} disabled={off}
        hint="A fixed level change (Amplify) in this scope's rack"
        onSwitch={(on) => onSwitch('volume', 'amplify', on, { gain: g })}>
        <NumberParam def={gain} label="Level" value={g} disabled={off}
          onChange={(v) => onSet('volume', 'amplify', { gain: v })} />
      </Control>
    );
  }

  return (
    <div className="audio-card audio-essential">
      <div className="audio-card-title">
        Essential Sound <span className="audio-unit">Dialogue · {scopeLabel}</span>
      </div>
      <div className="audio-hint">
        Each control adds or updates one effect in this rack (tagged essential:…), heard live
        and saved with the rack. Untick a control to bypass its effect; remove it in the rack.
      </div>
      <div className="audio-col-title">Repair</div>
      {noise}
      {rumble}
      {dehum}
      {deess}
      {reverbProcess}
      {(dynamics || eq || enhance) && <div className="audio-col-title">Clarity</div>}
      {dynamics}
      {eq}
      {enhance}
      {reverb && <div className="audio-col-title">Creative</div>}
      {reverb}
      {volume && <div className="audio-col-title">Clip Volume</div>}
      {volume}
    </div>
  );
}
