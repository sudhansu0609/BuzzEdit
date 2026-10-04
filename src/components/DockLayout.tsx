import { useCallback, useEffect, useRef, useState } from 'react';
import {
  useLayoutStore, DockZone, FloatRect, PanelId, PanelPlacement, PANEL_TABS, PANEL_TITLES, ZoneSizes,
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
          <div className="dock-tabs" onWheel={scrollTabs}>
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

/**
 * The tab strip scrolls sideways with its scrollbar hidden, so in a narrow
 * zone the last tabs sat past the right edge where a plain mouse wheel could
 * not reach them. Map the vertical wheel onto the strip.
 */
function scrollTabs(e: React.WheelEvent<HTMLDivElement>) {
  const strip = e.currentTarget;
  if (strip.scrollWidth <= strip.clientWidth) return;
  strip.scrollLeft += e.deltaY || e.deltaX;
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
  const ref = useRef<HTMLButtonElement>(null);
  // Keep the active tab in view when the strip is scrolled or squeezed.
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [active]);
  return (
    <button
      ref={ref}
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
function FloatingPanel({ id, bounds, children }: {
  id: PanelId; bounds: Bounds | null; children: React.ReactNode;
}) {
  const { panels, locked, setRect, setPlacement, setDragging, dragging } = useLayoutStore();
  const rect = panels[id].rect;
  // Draw the panel inside the window even when the stored rect came from a
  // larger window. Only the drawn rect is clamped, so growing the window again
  // puts the panel back where it was left.
  const shown = bounds ? fitRect(rect, bounds) : rect;
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
    drag.current = { x: e.clientX, y: e.clientY, rect: { ...shown }, mode };
    if (mode === 'move') setDragging(id);
  }, [id, locked, shown, setDragging]);

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
      style={{ left: shown.x, top: shown.y, width: shown.width, height: shown.height }}
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

interface Bounds { width: number; height: number }

/** Narrowest the centre (preview) may get before the side zones give way. */
const MIN_CENTRE_W = 320;
/** Shortest the top row may get before the bottom zone gives way. */
const MIN_CENTRE_H = 180;
/** A side or bottom zone is not squeezed below this while there is room. */
const MIN_FITTED = 140;
/** Width of one splitter. */
const SPLITTER = 4;

/**
 * Fit the stored zone sizes into the space the window actually has.
 *
 * Side zones are fixed-width, so in a small or restored window they used to
 * keep their full size and push the right-hand zone off the edge. Here they
 * shrink proportionally until the centre keeps its minimum. The store is left
 * alone, so maximising again restores the layout exactly.
 */
function fitSizes(sizes: ZoneSizes, bounds: Bounds | null,
  has: { left: boolean; right: boolean; bottom: boolean }): ZoneSizes {
  if (!bounds || bounds.width === 0) return sizes;
  let { left, right, bottom } = sizes;
  const l = has.left ? left : 0;
  const r = has.right ? right : 0;
  const splitters = (has.left ? SPLITTER : 0) + (has.right ? SPLITTER : 0);
  const room = bounds.width - splitters - MIN_CENTRE_W;
  if (l + r > room && l + r > 0) {
    const scale = Math.max(0, room) / (l + r);
    left = Math.max(MIN_FITTED, Math.floor(l * scale));
    right = Math.max(MIN_FITTED, Math.floor(r * scale));
    // Even the minimums do not fit: the centre gives up its minimum too,
    // but the sides still never run past the edge.
    const total = (has.left ? left : 0) + (has.right ? right : 0);
    const hardRoom = bounds.width - splitters - 80;
    if (total > hardRoom && total > 0) {
      const squeeze = Math.max(0, hardRoom) / total;
      left = Math.floor(left * squeeze);
      right = Math.floor(right * squeeze);
    }
  }
  if (has.bottom && bounds.height > 0) {
    const roomH = bounds.height - SPLITTER - MIN_CENTRE_H;
    if (bottom > roomH) bottom = Math.max(Math.min(MIN_FITTED, bottom), Math.floor(roomH));
    bottom = Math.min(bottom, Math.max(0, bounds.height - SPLITTER - 60));
  }
  return { left, right, bottom };
}

/** Keep a floating rect inside the dock area, shrinking it if it must. */
function fitRect(rect: FloatRect, bounds: Bounds): FloatRect {
  const width = Math.min(rect.width, Math.max(200, bounds.width));
  const height = Math.min(rect.height, Math.max(120, bounds.height));
  const x = Math.max(0, Math.min(rect.x, bounds.width - width));
  const y = Math.max(0, Math.min(rect.y, bounds.height - height));
  return { x, y, width, height };
}

/** Track an element's size; null until the first measurement. */
function useBounds() {
  const ref = useRef<HTMLDivElement>(null);
  const [bounds, setBounds] = useState<Bounds | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const width = el.clientWidth;
      const height = el.clientHeight;
      setBounds((b) => (b && b.width === width && b.height === height ? b : { width, height }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, bounds] as const;
}

/** Draggable divider between two zones. */
function Splitter({ axis, zone, size, invert }: {
  axis: 'x' | 'y'; zone: keyof ZoneSizes; size: number; invert?: boolean;
}) {
  const { setZoneSize, locked } = useLayoutStore();
  const start = useRef<{ pos: number; size: number } | null>(null);

  const begin = (e: React.PointerEvent) => {
    if (locked) return;
    e.preventDefault();
    // Start from the size actually on screen: in a narrow window the zone is
    // drawn smaller than its stored size, and starting from the stored one
    // would make the first pixel of a drag jump.
    start.current = { pos: axis === 'x' ? e.clientX : e.clientY, size };
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
  const { panels, sizes: stored, dragging } = useLayoutStore();
  const [rootRef, bounds] = useBounds();

  const inZone = (zone: DockZone) =>
    (Object.keys(panels) as PanelId[]).filter(
      (id) => panels[id].placement === zone && panels[id].visible);
  const floating = (Object.keys(panels) as PanelId[]).filter(
    (id) => panels[id].placement === 'floating' && panels[id].visible);

  const left = inZone('left');
  const right = inZone('right');
  const centre = inZone('center');
  const bottom = inZone('bottom');

  const showLeft = left.length > 0 || !!dragging;
  const showRight = right.length > 0 || !!dragging;
  const showBottom = bottom.length > 0 || !!dragging;
  const sizes = fitSizes(stored, bounds, { left: showLeft, right: showRight, bottom: showBottom });

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
    <div className="dock-root" ref={rootRef}>
      <div className="dock-row">
        {showLeft && renderZone('left', left)}
        {showLeft && <Splitter axis="x" zone="left" size={sizes.left} />}

        {renderZone('center', centre)}

        {showRight && <Splitter axis="x" zone="right" size={sizes.right} invert />}
        {showRight && renderZone('right', right)}
      </div>

      {showBottom && <Splitter axis="y" zone="bottom" size={sizes.bottom} invert />}
      {showBottom && renderZone('bottom', bottom)}

      {floating.map((id) => (
        <FloatingPanel key={id} id={id} bounds={bounds}>{content[id]}</FloatingPanel>
      ))}
    </div>
  );
}
