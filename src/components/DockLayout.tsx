import { useCallback, useEffect, useRef, useState } from 'react';
import {
  useLayoutStore, DockZone, PanelId, PanelPlacement, PANEL_TABS, PANEL_TITLES, ZoneSizes,
} from '../hooks/layout';

/**
 * Dockable workspace: panels sit in four zones or float above them, splitters
 * resize the zones, and a lock pins the whole arrangement.
 *
 * Dragging is done with pointer events on the title bar rather than HTML5 drag
 * and drop. The editor already uses HTML5 drag for media-pool → timeline drops,
 * and mixing the two makes a panel drag look like a media drop to the lanes
 * underneath. Pointer capture also keeps the drag alive over a <video>-heavy
 * layout, where dragover events go missing.
 *
 * Several panels can share a zone, in which case the zone shows a tab strip and
 * only the active one is mounted. That is what lets every panel dock
 * independently without the workspace collapsing into eight stacked slivers —
 * and it makes the tab bar a property of a *place* rather than of one panel, so
 * any tab can be dragged out into a zone of its own.
 */

const ZONE_LABELS: Record<PanelPlacement, string> = {
  center: 'Centre',
  left: 'Left',
  right: 'Right',
  bottom: 'Bottom',
  floating: 'Floating',
};

const PLACEMENTS: PanelPlacement[] = ['center', 'left', 'right', 'bottom', 'floating'];

/**
 * Start a drag that can re-dock a panel.
 *
 * There is no ghost following the cursor for a docked panel — re-parenting a
 * live <video> mid-gesture stalls playback — so the feedback is the zones
 * lighting up instead. The drop target is resolved on release.
 */
function useDockDrag(id: PanelId) {
  const { locked, setDragging, setPlacement } = useLayoutStore();

  return useCallback((e: React.PointerEvent) => {
    if (locked || e.button !== 0) return;
    const origin = { x: e.clientX, y: e.clientY };
    let started = false;

    const onMove = (ev: PointerEvent) => {
      if (started) return;
      // A few pixels of slop, or every click on a tab reads as a drag and the
      // zones flash on each selection.
      if (Math.hypot(ev.clientX - origin.x, ev.clientY - origin.y) > 5) {
        started = true;
        setDragging(id);
      }
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      if (!started) return;
      setDragging(null);
      const target = document.elementFromPoint(ev.clientX, ev.clientY)
        ?.closest('[data-dock-target]') as HTMLElement | null;
      const zone = target?.dataset.dockTarget as DockZone | undefined;
      // Dropped away from any dock: the panel floats where it was let go.
      setPlacement(id, zone ?? 'floating');
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, [id, locked, setDragging, setPlacement]);
}

/** The move/hide controls for whichever panel is on top of a zone. */
function PanelControls({ id }: { id: PanelId }) {
  const { panels, locked, setPlacement, toggleVisible } = useLayoutStore();
  const [menuOpen, setMenuOpen] = useState(false);
  const panel = panels[id];

  useEffect(() => {
    if (!menuOpen) return;
    const close = () => setMenuOpen(false);
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [menuOpen]);

  return (
    <>
      <button
        className="dock-btn"
        title={locked ? 'Layout is locked' : 'Move this panel'}
        disabled={locked}
        onClick={(e) => { e.stopPropagation(); setMenuOpen((v) => !v); }}
        onPointerDown={(e) => e.stopPropagation()}
        onMouseDown={(e) => e.stopPropagation()}
      >▾</button>
      <button
        className="dock-btn"
        title="Hide this panel"
        onClick={(e) => { e.stopPropagation(); toggleVisible(id, false); }}
        onPointerDown={(e) => e.stopPropagation()}
      >✕</button>

      {menuOpen && (
        <div className="dock-menu" onPointerDown={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}>
          {PLACEMENTS.map((zone) => (
            <button key={zone}
              className={`dock-menu-item ${panel.placement === zone ? 'active' : ''}`}
              onClick={() => { setPlacement(id, zone); setMenuOpen(false); }}>
              {panel.placement === zone ? '✓ ' : ''}{ZONE_LABELS[zone]}
            </button>
          ))}
        </div>
      )}
    </>
  );
}

interface ZonePanelProps {
  zone: DockZone;
  ids: PanelId[];
  content: Record<PanelId, React.ReactNode>;
}

/** One dock zone: a tab strip when it holds several panels, plus the active body. */
function ZonePanel({ zone, ids, content }: ZonePanelProps) {
  const { active, setActive, locked } = useLayoutStore();
  const stored = active[zone];
  const activeId = stored && ids.includes(stored) ? stored : ids[0];
  const tabbed = ids.length > 1;

  return (
    <section className="dock-panel">
      <header className={`dock-panel-bar ${locked ? 'locked' : ''} ${tabbed ? 'tabbed' : ''}`}>
        {tabbed ? (
          <div className="dock-tabs">
            {ids.map((id) => (
              <DockTab key={id} id={id} zone={zone}
                active={id === activeId} onSelect={() => setActive(zone, id)} />
            ))}
          </div>
        ) : (
          <SoloTitle id={activeId} />
        )}
        <PanelControls id={activeId} />
      </header>
      <div className="dock-panel-body">{content[activeId]}</div>
    </section>
  );
}

function SoloTitle({ id }: { id: PanelId }) {
  const { locked } = useLayoutStore();
  const beginDrag = useDockDrag(id);
  return (
    <span
      className="dock-solo-title"
      onPointerDown={beginDrag}
      title={locked ? 'Layout is locked' : 'Drag to move this panel'}
    >
      <span className="dock-grip" aria-hidden>⠿</span>
      {PANEL_TITLES[id]}
    </span>
  );
}

function DockTab({ id, active, onSelect }: {
  id: PanelId; zone: DockZone; active: boolean; onSelect: () => void;
}) {
  const { locked } = useLayoutStore();
  const beginDrag = useDockDrag(id);
  return (
    <button
      className={`dock-tab ${active ? 'active' : ''}`}
      onClick={onSelect}
      onPointerDown={beginDrag}
      title={locked ? PANEL_TITLES[id] : `${PANEL_TITLES[id]} — drag to move`}
    >
      {PANEL_TABS[id]}
    </button>
  );
}

/** A floating panel: drag by the title bar, resize from the bottom-right corner. */
function FloatingPanel({ id, children }: { id: PanelId; children: React.ReactNode }) {
  const { panels, locked, setRect, setPlacement, setDragging, dragging } = useLayoutStore();
  const rect = panels[id].rect;
  // While being dragged the panel must not intercept hit-testing: the drop
  // target is resolved with elementFromPoint, and a panel that follows the
  // cursor is always the topmost element under it, so every drop landed on
  // itself and nothing ever re-docked.
  const isDragging = dragging === id;
  const drag = useRef<{ x: number; y: number; rect: typeof rect; mode: 'move' | 'resize' } | null>(null);

  const begin = useCallback((mode: 'move' | 'resize') => (e: React.PointerEvent) => {
    if (locked) return;
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, rect: { ...rect }, mode };
    if (mode === 'move') setDragging(id);
  }, [id, locked, rect, setDragging]);

  useEffect(() => {
    if (!drag.current) return;
    const onMove = (e: PointerEvent) => {
      const state = drag.current;
      if (!state) return;
      const dx = e.clientX - state.x;
      const dy = e.clientY - state.y;
      if (state.mode === 'move') {
        setRect(id, {
          x: Math.max(0, state.rect.x + dx),
          y: Math.max(0, state.rect.y + dy),
        });
      } else {
        setRect(id, {
          width: Math.max(260, state.rect.width + dx),
          height: Math.max(160, state.rect.height + dy),
        });
      }
    };
    const onUp = (e: PointerEvent) => {
      const wasMoving = drag.current?.mode === 'move';
      drag.current = null;
      setDragging(null);
      if (!wasMoving) return;
      // Dropping over a dock target re-docks the panel.
      const target = document.elementFromPoint(e.clientX, e.clientY)
        ?.closest('[data-dock-target]') as HTMLElement | null;
      const zone = target?.dataset.dockTarget as DockZone | undefined;
      if (zone) setPlacement(id, zone);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  });

  return (
    <div
      className={`dock-floating ${isDragging ? 'dragging' : ''}`}
      style={{ left: rect.x, top: rect.y, width: rect.width, height: rect.height }}
    >
      <section className="dock-panel floating">
        <header className={`dock-panel-bar ${locked ? 'locked' : ''}`}
          onPointerDown={locked ? undefined : begin('move')}>
          <span className="dock-solo-title">
            <span className="dock-grip" aria-hidden>⠿</span>
            {PANEL_TITLES[id]}
          </span>
          <PanelControls id={id} />
        </header>
        <div className="dock-panel-body">{children}</div>
      </section>
      {!locked && (
        <div className="dock-resize" onPointerDown={begin('resize')} title="Resize" />
      )}
    </div>
  );
}

/** Draggable divider between two zones. */
function Splitter({ axis, zone, invert }: {
  axis: 'x' | 'y'; zone: keyof ZoneSizes; invert?: boolean;
}) {
  const { sizes, setZoneSize, locked } = useLayoutStore();
  const start = useRef<{ pos: number; size: number } | null>(null);

  const begin = (e: React.PointerEvent) => {
    if (locked) return;
    e.preventDefault();
    start.current = { pos: axis === 'x' ? e.clientX : e.clientY, size: sizes[zone] };
    const onMove = (ev: PointerEvent) => {
      if (!start.current) return;
      const delta = (axis === 'x' ? ev.clientX : ev.clientY) - start.current.pos;
      setZoneSize(zone, start.current.size + (invert ? -delta : delta));
    };
    const onUp = () => {
      start.current = null;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  if (locked) return <div className={`dock-splitter ${axis} locked`} />;
  return (
    <div className={`dock-splitter ${axis}`} onPointerDown={begin}
      title="Drag to resize" />
  );
}

export interface DockLayoutProps {
  panels: Record<PanelId, React.ReactNode>;
}

export default function DockLayout({ panels: content }: DockLayoutProps) {
  const { panels, sizes, dragging } = useLayoutStore();

  const inZone = (zone: DockZone) =>
    (Object.keys(panels) as PanelId[]).filter(
      (id) => panels[id].placement === zone && panels[id].visible);
  const floating = (Object.keys(panels) as PanelId[]).filter(
    (id) => panels[id].placement === 'floating' && panels[id].visible);

  const left = inZone('left');
  const right = inZone('right');
  const centre = inZone('center');
  const bottom = inZone('bottom');

  const renderZone = (zone: DockZone, ids: PanelId[]) => (
    <div className={`dock-zone dock-zone-${zone} ${dragging ? 'targetable' : ''}`}
      data-dock-target={zone}
      style={
        zone === 'left' ? { width: sizes.left }
          : zone === 'right' ? { width: sizes.right }
            : zone === 'bottom' ? { height: sizes.bottom }
              : undefined
      }>
      {ids.length > 0 && <ZonePanel zone={zone} ids={ids} content={content} />}
      {/* While a panel is in flight, empty zones still have to be droppable. */}
      {dragging && ids.length === 0 && (
        <div className="dock-drop-hint">Drop here to dock {ZONE_LABELS[zone]}</div>
      )}
    </div>
  );

  return (
    <div className="dock-root">
      <div className="dock-row">
        {(left.length > 0 || dragging) && renderZone('left', left)}
        {(left.length > 0 || dragging) && <Splitter axis="x" zone="left" />}

        {renderZone('center', centre)}

        {(right.length > 0 || dragging) && <Splitter axis="x" zone="right" invert />}
        {(right.length > 0 || dragging) && renderZone('right', right)}
      </div>

      {(bottom.length > 0 || dragging) && <Splitter axis="y" zone="bottom" invert />}
      {(bottom.length > 0 || dragging) && renderZone('bottom', bottom)}

      {floating.map((id) => (
        <FloatingPanel key={id} id={id}>{content[id]}</FloatingPanel>
      ))}
    </div>
  );
}
