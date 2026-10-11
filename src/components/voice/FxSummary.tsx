/**
 * A preset's creature FX (VOICE_CHANGER_PLAN.md §2): live rack effects after the conversion --
 * Pitch Shifter, Distortion, Reverb, Sub-Octave -- named and valued from the effect catalogue.
 * An intent the catalogue has no effect for (SkinWalker's "formant", a missing sub-octave) is
 * listed as not in the rack, so nothing is silently lost.
 */

import type { AudioEffect, EffectDef } from '../../hooks/audioApi';
import type { FxIntent } from '../../hooks/voiceApi';

/** presets.json fx intent -> the rack effect that carries it ('' = none in the catalogue). */
const INTENT_EFFECT: Record<string, string> = {
  pitch_semitones: 'pitch_shifter',
  drive: 'distortion',
  reverb: 'reverb',
  sub_octave: 'sub_octave',
  formant: '',
};

const INTENT_LABEL: Record<string, string> = {
  pitch_semitones: 'pitch',
  drive: 'drive',
  reverb: 'reverb',
  sub_octave: 'sub-octave',
  formant: 'formant',
};

const FALLBACK_NAME: Record<string, string> = {
  pitch_shifter: 'Pitch Shifter',
  distortion: 'Distortion',
  reverb: 'Reverb',
  sub_octave: 'Sub-Octave',
};

function number(value: unknown): string {
  if (typeof value === 'number') return String(Math.round(value * 10) / 10);
  if (typeof value === 'boolean') return value ? 'on' : 'off';
  return String(value);
}

export function effectName(fx: AudioEffect, defs?: Record<string, EffectDef>): string {
  return defs?.[fx.effect]?.name || FALLBACK_NAME[fx.effect] || fx.effect;
}

export function effectParams(fx: AudioEffect, defs?: Record<string, EffectDef>): string {
  const def = defs?.[fx.effect];
  return Object.entries(fx.params || {})
    .map(([key, value]) => {
      const spec = def?.params?.find((p) => p.key === key);
      const unit = spec?.unit ? ` ${spec.unit}` : '';
      return `${spec?.label || key} ${number(value)}${unit}`;
    })
    .join(', ');
}

/** One line, for tooltips: "Pitch Shifter (Semitones -4 st); Reverb (...)". */
export function fxSummaryText(effects: AudioEffect[] | null | undefined, defs?: Record<string, EffectDef>): string {
  if (!effects || effects.length === 0) return 'no FX';
  return effects.map((fx) => {
    const params = effectParams(fx, defs);
    return params ? `${effectName(fx, defs)} (${params})` : effectName(fx, defs);
  }).join('; ');
}

export default function FxSummary({ effects, intent, defs, hint }: {
  effects: AudioEffect[];
  /** the preset's FX intent, to name what the rack could not carry */
  intent?: FxIntent | null;
  defs?: Record<string, EffectDef>;
  hint?: string;
}) {
  const present = new Set(effects.map((fx) => fx.effect));
  const skipped = Object.entries(intent || {}).filter(([key, value]) =>
    typeof value === 'number' && value !== 0 && !present.has(INTENT_EFFECT[key] || `__none_${key}`));
  return (
    <div className="voice-fx">
      <div className="voice-fx-title">
        FX after the conversion{effects.length ? ` · ${effects.length}` : ''}
      </div>
      {effects.length === 0 ? (
        <div className="audio-hint">No FX: the converted voice as it comes.</div>
      ) : (
        <ul className="voice-fx-list">
          {effects.map((fx) => (
            <li key={fx.id}>
              <span className="voice-fx-name">{effectName(fx, defs)}</span>
              <span className="voice-fx-params">{effectParams(fx, defs)}</span>
            </li>
          ))}
        </ul>
      )}
      {skipped.length > 0 && (
        <div className="audio-hint">
          Not in the rack (no such effect in the catalogue yet):{' '}
          {skipped.map(([key, value]) => `${INTENT_LABEL[key] || key} ${number(value)}`).join(', ')}.
        </div>
      )}
      {hint && <div className="audio-hint">{hint}</div>}
    </div>
  );
}
