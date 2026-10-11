/**
 * The Effects Rack of one scope (AUDIO_SUITE_PLAN.md §2.4): the effects in processing order,
 * each with a power button, drag by the grip (or ↑ / ↓) to reorder, "windowed to selection",
 * its presets (on the selected row) and remove. Clicking a name opens its parameters.
 */

import { useState } from 'react';
import { GripVertical, Power } from 'lucide-react';
import type { AudioEffect, EffectDef, UserPresets } from '../../hooks/audioApi';
import PresetMenu from './PresetMenu';

const secs = (v: number | null | undefined) => (v == null ? '…' : v.toFixed(2));

export interface RackSelection {
  start_s: number;
  end_s: number;
}

function windowTitle(def: EffectDef | undefined, fx: AudioEffect, selection: RackSelection | null): string {
  if (!def?.windowable) return 'This effect always covers the whole scope';
  if (fx.start_s != null || fx.end_s != null) {
    return `Windowed to ${secs(fx.start_s)}-${secs(fx.end_s)} s: click to cover the whole scope again`;
  }
  if (selection) {
    return `Window this effect to the selection, ${secs(selection.start_s)}-${secs(selection.end_s)} s`;
  }
  return 'Select a range first to window this effect to it';
}

export default function RackList({
  rack, defs, selectedId, selection, userPresets, scopeLabel,
  onSelect, onToggle, onMove, onRemove, onWindow, onApplyPreset, onPresetsChange,
}: {
  rack: AudioEffect[];
  defs: Record<string, EffectDef>;
  selectedId: string | null;
  /** the store's audioSelection: what "windowed to selection" copies */
  selection: RackSelection | null;
  userPresets: UserPresets;
  scopeLabel: string;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
  onMove: (from: number, to: number) => void;
  onRemove: (id: string) => void;
  onWindow: (id: string) => void;
  onApplyPreset: (id: string, name: string | null, params: Record<string, unknown>) => void;
  onPresetsChange: (presets: UserPresets) => void;
}) {
  const [dragFrom, setDragFrom] = useState<number | null>(null);
  const [dropAt, setDropAt] = useState<number | null>(null);

  const endDrag = () => {
    setDragFrom(null);
    setDropAt(null);
  };

  return (
    <div className="audio-rack">
      <div className="audio-col-title">Effects Rack · {scopeLabel}</div>
      {rack.length === 0 && (
        <div className="text-xs text-muted">
          Empty. Pick an effect in the tree to add it; the rack runs top to bottom.
        </div>
      )}
      {rack.map((fx, index) => {
        const def = defs[fx.effect];
        const windowed = fx.start_s != null || fx.end_s != null;
        const canWindow = !!def?.windowable && (windowed || !!selection);
        const selected = fx.id === selectedId;
        const dropping = dropAt === index && dragFrom !== null && dragFrom !== index;
        return (
          <div key={fx.id}
            className={`audio-rack-item${selected ? ' selected' : ''}${fx.enabled ? '' : ' off'}${dropping ? ' drop' : ''}`}
            onDragOver={(e) => {
              if (dragFrom === null) return;
              e.preventDefault();
              e.dataTransfer.dropEffect = 'move';
              if (dropAt !== index) setDropAt(index);
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (dragFrom !== null && dragFrom !== index) onMove(dragFrom, index);
              endDrag();
            }}>
            <div className="audio-rack-head" draggable
              onDragStart={(e) => {
                setDragFrom(index);
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', fx.id);
              }}
              onDragEnd={endDrag}>
              <span className="audio-grip" title="Drag to reorder"><GripVertical size={13} /></span>
              <button className={`audio-power${fx.enabled ? ' on' : ''}`}
                title={fx.enabled ? 'On: click to bypass' : 'Bypassed: click to switch on'}
                onClick={() => onToggle(fx.id)}>
                <Power size={12} />
              </button>
              <button className="audio-rack-name" onClick={() => onSelect(fx.id)}
                title={def?.description || `Unknown effect ${fx.effect}`}>
                <span className="audio-rack-index">{index + 1}</span>
                <span className="audio-rack-label">{def?.name || fx.effect}</span>
                {fx.preset && <span className="audio-rack-preset">{fx.preset}</span>}
              </button>
              <button className="audio-icon" title="Move up" disabled={index === 0}
                onClick={() => onMove(index, index - 1)}>↑</button>
              <button className="audio-icon" title="Move down" disabled={index === rack.length - 1}
                onClick={() => onMove(index, index + 1)}>↓</button>
              <button className={`audio-icon${windowed ? ' active' : ''}`} disabled={!canWindow}
                title={windowTitle(def, fx, selection)} onClick={() => onWindow(fx.id)}>⇥</button>
              <button className="audio-icon danger" title="Remove from the rack"
                onClick={() => onRemove(fx.id)}>✕</button>
            </div>
            {windowed && (
              <div className="audio-rack-window">Only {secs(fx.start_s)}–{secs(fx.end_s)} s</div>
            )}
            {!def && <div className="audio-note">Not in the effect catalogue: {fx.effect}</div>}
            {selected && def && (
              <PresetMenu def={def} userPresets={userPresets} current={fx.params} active={fx.preset}
                onApply={(name, params) => onApplyPreset(fx.id, name, params)}
                onPresetsChange={onPresetsChange} />
            )}
          </div>
        );
      })}
    </div>
  );
}
