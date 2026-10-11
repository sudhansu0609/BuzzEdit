/**
 * Audition's Effects menu as a tree (AUDIO_SUITE_PLAN.md §2.4): one branch per category, in
 * the catalogue's order. A rack effect ("ffmpeg") is appended to the scope's rack with its
 * defaults on click; a process effect opens its card (Preview / Apply / Undo) instead.
 */

import { useMemo, useState } from 'react';
import type { EffectDef } from '../../hooks/audioApi';

export default function EffectTree({
  categories, effects, processes, onAdd, onPickProcess, selectedProcess, canAdd, notes,
}: {
  categories: string[];
  effects: EffectDef[];
  processes: EffectDef[];
  onAdd: (def: EffectDef) => void;
  onPickProcess: (def: EffectDef) => void;
  selectedProcess: string | null;
  /** false while there is no rack to add to */
  canAdd: boolean;
  notes?: Array<string | null>;
}) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState<Set<string>>(() => new Set());
  const q = query.trim().toLowerCase();

  const groups = useMemo(() => {
    const all = [...effects, ...processes];
    const order = [...categories];
    for (const def of all) if (!order.includes(def.category)) order.push(def.category);
    return order.map((category) => ({
      category,
      items: all.filter((def) => def.category === category
        && (!q || String(def.name || '').toLowerCase().includes(q) || String(def.id).toLowerCase().includes(q))),
    })).filter((group) => group.items.length > 0);
  }, [categories, effects, processes, q]);

  const toggle = (category: string) => setOpen((current) => {
    const next = new Set(current);
    if (next.has(category)) next.delete(category);
    else next.add(category);
    return next;
  });

  return (
    <div className="audio-tree">
      <input className="fx-search" placeholder="Search effects" value={query}
        onChange={(e) => setQuery(e.target.value)} />
      {(notes || []).filter(Boolean).map((note) => (
        <div key={note as string} className="audio-note">{note}</div>
      ))}
      {groups.length === 0 && (
        <div className="text-xs text-muted">{q ? 'No effect matches.' : 'No effects loaded.'}</div>
      )}
      {groups.map(({ category, items }) => {
        const expanded = !!q || open.has(category);
        return (
          <div key={category} className="audio-tree-group">
            <button className="audio-tree-cat" onClick={() => toggle(category)}>
              <span className={`insp-caret ${expanded ? 'open' : ''}`}>▸</span>
              <span className="audio-tree-cat-name">{category}</span>
              <span className="audio-tree-count">{items.length}</span>
            </button>
            {expanded && items.map((def) => {
              const isProcess = def.engine === 'process';
              return (
                <button key={`${def.engine}:${def.id}`}
                  className={`audio-tree-item ${isProcess && selectedProcess === def.id ? 'active' : ''}`}
                  title={def.description}
                  disabled={!isProcess && !canAdd}
                  onClick={() => (isProcess ? onPickProcess(def) : onAdd(def))}>
                  <span className="audio-tree-name">{def.name || def.id}</span>
                  {isProcess
                    ? <span className="audio-badge process">process</span>
                    : <span className="audio-tree-add" aria-hidden>＋</span>}
                </button>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
