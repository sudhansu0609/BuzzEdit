/**
 * Presets of one effect: Audition's shipped ones from the catalogue, the creator's own
 * (`data/audio_presets.json` through /api/audio/presets), "Save…" for the current settings and
 * a delete button while one of his own is the active preset. A native <select>, so the list is
 * never clipped by the scrolling rack around it.
 */

import { useState } from 'react';
import {
  EffectDef, UserPresets, deletePreset, errorText, routeMissing, savePreset,
} from '../../hooks/audioApi';

const DEFAULT_KEY = '__default__';

const has = (obj: object, key: string) => Object.prototype.hasOwnProperty.call(obj, key);

export default function PresetMenu({ def, userPresets, current, active, onApply, onPresetsChange }: {
  def: EffectDef;
  userPresets: UserPresets;
  /** the settings "Save…" stores */
  current: Record<string, unknown>;
  /** the effect's `preset` (null once a parameter has moved) */
  active?: string | null;
  /** `name` null = back to the defaults; `params` are merged over the defaults by the caller */
  onApply: (name: string | null, params: Record<string, unknown>) => void;
  onPresetsChange: (presets: UserPresets) => void;
}) {
  const shipped = def.presets || {};
  const mine = userPresets[def.id] || {};
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const activeIsMine = !!active && has(mine, active);
  const activeKey = active ? (activeIsMine ? `u:${active}` : has(shipped, active) ? `b:${active}` : '') : '';

  const pick = (key: string) => {
    setNote(null);
    if (key === DEFAULT_KEY) onApply(null, {});
    else if (key.startsWith('b:')) onApply(key.slice(2), shipped[key.slice(2)] || {});
    else if (key.startsWith('u:')) onApply(key.slice(2), mine[key.slice(2)] || {});
  };

  const fail = (err: unknown, what: string) => setNote(routeMissing(err)
    ? 'Presets are not available on this backend yet.'
    : `${what}: ${errorText(err)}`);

  const save = async () => {
    const preset = name.trim();
    if (!preset || preset.includes('/')) {
      setNote('Give the preset a name (no "/").');
      return;
    }
    setBusy(true);
    setNote(null);
    try {
      const res = await savePreset(def.id, preset, current);
      onPresetsChange(res.presets || {});
      onApply(preset, current);
      setSaving(false);
      setName('');
    } catch (err) {
      fail(err, 'Not saved');
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!active || !activeIsMine) return;
    setBusy(true);
    setNote(null);
    try {
      const res = await deletePreset(def.id, active);
      onPresetsChange(res.presets || {});
    } catch (err) {
      fail(err, 'Not deleted');
    } finally {
      setBusy(false);
    }
  };

  const shippedNames = Object.keys(shipped);
  const myNames = Object.keys(mine);

  return (
    <div className="audio-presets">
      <div className="audio-presets-row">
        <select className="audio-select" value={activeKey} title="Presets"
          onChange={(e) => pick(e.target.value)}>
          <option value="">{active && !activeKey ? active : 'Presets…'}</option>
          <option value={DEFAULT_KEY}>Default settings</option>
          {shippedNames.length > 0 && (
            <optgroup label="Presets">
              {shippedNames.map((n) => <option key={n} value={`b:${n}`}>{n}</option>)}
            </optgroup>
          )}
          {myNames.length > 0 && (
            <optgroup label="Your presets">
              {myNames.map((n) => <option key={n} value={`u:${n}`}>{n}</option>)}
            </optgroup>
          )}
        </select>
        <button className="audio-btn" title="Save the current settings as a preset of your own"
          onClick={() => { setSaving((s) => !s); setNote(null); }}>
          Save…
        </button>
        {activeIsMine && (
          <button className="audio-btn danger" disabled={busy} onClick={remove}
            title={`Delete your preset "${active}"`}>
            ✕
          </button>
        )}
      </div>
      {saving && (
        <div className="audio-presets-row">
          <input className="audio-text" placeholder="Preset name" value={name} autoFocus
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void save();
              if (e.key === 'Escape') setSaving(false);
            }} />
          <button className="audio-btn primary" disabled={busy || !name.trim()} onClick={() => void save()}>
            Save
          </button>
        </div>
      )}
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}
