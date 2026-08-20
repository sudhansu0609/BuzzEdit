import { useEffect, useRef, useState } from 'react';
import { useLayoutStore, LAYOUT_PRESETS, PANEL_TITLES, PanelId } from '../hooks/layout';

/**
 * Layout control for the header: pick an arrangement, show or hide individual
 * panels, and lock everything so a stray drag cannot disturb it mid-edit.
 */
export default function LayoutMenu() {
  const {
    panels, locked, preset, setLocked, applyPreset, resetLayout, toggleVisible,
  } = useLayoutStore();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    return () => window.removeEventListener('mousedown', onDown);
  }, [open]);

  const hidden = (Object.keys(panels) as PanelId[]).filter((id) => !panels[id].visible);

  return (
    <div className="layout-menu" ref={root}>
      <button
        className={`btn btn-sm ${locked ? 'btn-primary' : ''}`}
        title={locked ? 'Layout is locked — click to unlock' : 'Lock the layout so panels cannot move'}
        onClick={() => setLocked(!locked)}
      >
        {locked ? '🔒' : '🔓'}
      </button>
      <button className="btn btn-sm" onClick={() => setOpen((v) => !v)} title="Workspace layout">
        ▦ Layout{hidden.length > 0 ? ` (${hidden.length} hidden)` : ''}
      </button>

      {open && (
        <div className="layout-dropdown">
          <div className="layout-section-title">Arrangement</div>
          {LAYOUT_PRESETS.map((p) => (
            <button key={p.id}
              className={`layout-item ${preset === p.id ? 'active' : ''}`}
              onClick={() => { applyPreset(p.id); setOpen(false); }}>
              {preset === p.id ? '✓ ' : ''}{p.label}
            </button>
          ))}
          {preset === 'custom' && <div className="layout-note text-xs">Custom arrangement</div>}

          <div className="layout-section-title">Panels</div>
          {(Object.keys(panels) as PanelId[]).map((id) => (
            <label key={id} className="layout-item checkbox">
              <input
                type="checkbox"
                checked={panels[id].visible}
                onChange={(e) => toggleVisible(id, e.target.checked)}
              />
              {PANEL_TITLES[id]}
              <span className="text-xs text-muted" style={{ marginLeft: 'auto' }}>
                {panels[id].placement}
              </span>
            </label>
          ))}

          <div className="layout-section-title">Layout</div>
          <label className="layout-item checkbox">
            <input type="checkbox" checked={locked} onChange={(e) => setLocked(e.target.checked)} />
            Lock layout
          </label>
          <button className="layout-item" onClick={() => { resetLayout(); setOpen(false); }}>
            Reset to default
          </button>
          <div className="layout-note text-xs">
            Drag a panel by its title bar to move it. Locking prevents moving, resizing
            and undocking, but panels can still be shown or hidden.
          </div>
        </div>
      )}
    </div>
  );
}
