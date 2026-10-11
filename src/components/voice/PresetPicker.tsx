/**
 * The five presets (`GET /api/audio/voice/presets`: Woman, Child, Demon, Monster, Ghost) as
 * buttons, and `VoiceSettings` -- the voice from the library, a preset, Match my pitch contour,
 * Pitch shift and Quality steps, with the preset's FX summary -- shared by the Convert and Batch
 * tabs. A preset fills the voice, the params and the FX; changing a value afterwards keeps the
 * preset's name (the backend takes the preset's values, then the params, then the voice).
 */

import type { EffectDef } from '../../hooks/audioApi';
import {
  PITCH_RANGE, PresetsResponse, STEPS_RANGE, VoiceChoice, VoiceEntry, VoicePreset, emptyChoice,
  presetChoice, rememberChoice, voiceInLibrary,
} from '../../hooks/voiceApi';
import FxSummary, { fxSummaryText } from './FxSummary';

const signed = (n: number) => (n > 0 ? `+${n}` : String(n));

function presetTitle(p: VoicePreset, defs?: Record<string, EffectDef>): string {
  const params = p.params || {};
  return [
    `${p.name}: voice "${p.voice}"${p.voice_in_library ? '' : ' (not in the library yet)'}`,
    `pitch ${signed(Number(params.pitch_shift) || 0)} st, ${params.steps ?? 25} steps, ${params.match_pitch ? 'matches your pitch contour' : 'free pitch'}`,
    `FX: ${fxSummaryText(p.fx_rack, defs)}`,
  ].join('\n');
}

export default function PresetPicker({ presets, active, defs, disabled, onPick, onCustom }: {
  presets: PresetsResponse | null;
  active: string | null;
  defs?: Record<string, EffectDef>;
  disabled?: boolean;
  onPick: (preset: VoicePreset) => void;
  /** "Custom": no preset, the voice and params as they are */
  onCustom?: () => void;
}) {
  const table = presets?.presets || {};
  const names = Object.keys(table);
  if (names.length === 0) {
    return <div className="audio-hint">No presets: the presets route is not available on this backend.</div>;
  }
  return (
    <div className="voice-presets">
      {names.map((name) => {
        const p = table[name];
        return (
          <button key={name} type="button" disabled={disabled} title={presetTitle(p, defs)}
            className={`voice-preset${active === name ? ' active' : ''}${p.voice_in_library ? '' : ' warn'}`}
            onClick={() => onPick(p)}>
            {name}
          </button>
        );
      })}
      {onCustom && (
        <button type="button" disabled={disabled} className={`voice-preset${active ? '' : ' active'}`}
          title="No preset: the voice and the values below as they are, no FX" onClick={onCustom}>
          Custom
        </button>
      )}
    </div>
  );
}

/** Voice, preset, params and the FX summary of one conversion setting. */
export function VoiceSettings({ choice, onChange, voices, presets, defs, disabled, fxHint }: {
  choice: VoiceChoice | null;
  onChange: (choice: VoiceChoice) => void;
  voices: VoiceEntry[];
  presets: PresetsResponse | null;
  defs?: Record<string, EffectDef>;
  disabled?: boolean;
  fxHint?: string;
}) {
  const c = choice ?? emptyChoice(presets);
  const preset = c.preset ? presets?.presets?.[c.preset] : undefined;
  const voice = c.voice || preset?.voice || '';
  const pitch = Number(c.params.pitch_shift) || 0;
  const steps = Number(c.params.steps) || 25;

  const change = (next: VoiceChoice, remember = false) => {
    if (remember) rememberChoice(next);
    onChange(next);
  };
  const setParam = (key: 'match_pitch' | 'pitch_shift' | 'steps', value: boolean | number) =>
    change({ ...c, params: { ...c.params, [key]: value } });

  return (
    <div className="voice-settings">
      <label className="voice-field">
        <span className="voice-label">Voice</span>
        <select className="audio-select" disabled={disabled} value={c.voice}
          onChange={(e) => change({ ...c, voice: e.target.value }, true)}>
          <option value="">{c.preset ? `(the preset's: ${preset?.voice || 'none'})` : '(choose a voice)'}</option>
          {voices.map((v) => (
            <option key={v.name} value={v.name}>
              {v.name}{v.kind ? ` · ${v.kind}` : ''}{typeof v.seconds === 'number' ? ` · ${v.seconds.toFixed(1)} s` : ''}
            </option>
          ))}
          {c.voice && !voiceInLibrary(voices, c.voice) && (
            <option value={c.voice}>{c.voice} (not in the library)</option>
          )}
        </select>
      </label>
      <div className="voice-field">
        <span className="voice-label">Preset</span>
        <PresetPicker presets={presets} active={c.preset} defs={defs} disabled={disabled}
          onPick={(p) => change(presetChoice(p), true)}
          onCustom={() => change({ ...c, preset: null, fx: [], voice: voice }, true)} />
      </div>
      <label className="voice-check" title="seed-vc's f0 condition: keep your own intonation, shifted into the voice">
        <input type="checkbox" disabled={disabled} checked={!!c.params.match_pitch}
          onChange={(e) => setParam('match_pitch', e.target.checked)} />
        Match my pitch contour
      </label>
      <label className="voice-field" title="Semitones the converted voice is moved (the engine's pitch shift)">
        <span className="voice-label">Pitch shift</span>
        <input className="voice-slider" type="range" disabled={disabled}
          min={PITCH_RANGE[0]} max={PITCH_RANGE[1]} step={1} value={pitch}
          onChange={(e) => setParam('pitch_shift', Number(e.target.value))} />
        <span className="voice-value">{signed(pitch)} st</span>
      </label>
      <label className="voice-field" title="Diffusion steps: more is finer and slower (SkinWalker: 25, creatures 30)">
        <span className="voice-label">Quality steps</span>
        <input className="voice-slider" type="range" disabled={disabled}
          min={STEPS_RANGE[0]} max={STEPS_RANGE[1]} step={1} value={steps}
          onChange={(e) => setParam('steps', Number(e.target.value))} />
        <span className="voice-value">{steps}</span>
      </label>
      {!voice && <div className="audio-note">Pick a voice or a preset.</div>}
      {voice && !voiceInLibrary(voices, voice) && (
        <div className="audio-note">
          The voice "{voice}" is not in the library yet: record or upload it on the Voices tab under
          that name, or pick another voice.
        </div>
      )}
      <FxSummary effects={c.fx} intent={preset?.fx} defs={defs} hint={fxHint} />
    </div>
  );
}
