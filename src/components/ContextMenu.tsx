import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export interface MenuEntry {
  label?: string;
  /** Shortcut shown on the right, already formatted. */
  chord?: string;
  disabled?: boolean;
  danger?: boolean;
  /** Draws a tick, for on/off items such as Snapping. */
  checked?: boolean;
  separator?: boolean;
  /** Why it is greyed out — shown as the tooltip. */
  hint?: string;
  onClick?: () => void;
  submenu?: MenuEntry[];
}

interface Props {
  x: number;
  y: number;
  items: MenuEntry[];
  onClose: () => void;
}

/**
 * A right-click menu: portalled to <body> so no panel's overflow clips it,
 * nudged back inside the window when it would spill past an edge, and closed by
 * a click elsewhere, Esc, scrolling or the window losing focus.
 */
export default function ContextMenu({ x, y, items, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: x, top: y });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const left = Math.max(4, Math.min(x, window.innerWidth - r.width - 4));
    const top = Math.max(4, Math.min(y, window.innerHeight - r.height - 4));
    setPos({ left, top });
  }, [x, y]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    const close = () => onClose();
    window.addEventListener('mousedown', onDown, true);
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('blur', close);
    window.addEventListener('resize', close);
    window.addEventListener('wheel', close, { passive: true });
    return () => {
      window.removeEventListener('mousedown', onDown, true);
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('blur', close);
      window.removeEventListener('resize', close);
      window.removeEventListener('wheel', close);
    };
  }, [onClose]);

  return createPortal(
    <div ref={ref} className="ctx-menu" style={{ left: pos.left, top: pos.top }}
      onContextMenu={(e) => e.preventDefault()}>
      <MenuList items={items} onClose={onClose} />
    </div>,
    document.body,
  );
}

function MenuList({ items, onClose }: { items: MenuEntry[]; onClose: () => void }) {
  const [openSub, setOpenSub] = useState<number | null>(null);
  return (
    <>
      {items.map((item, idx) => {
        if (item.separator) return <div key={idx} className="ctx-sep" />;
        const hasSub = !!item.submenu?.length;
        return (
          <div key={idx} className="ctx-row"
            onMouseEnter={() => setOpenSub(hasSub ? idx : null)}>
            <button
              className={`ctx-item ${item.danger ? 'danger' : ''}`}
              disabled={item.disabled}
              title={item.disabled ? item.hint : undefined}
              onClick={() => {
                if (hasSub) { setOpenSub(idx); return; }
                onClose();
                item.onClick?.();
              }}>
              <span className="ctx-check">{item.checked ? '✓' : ''}</span>
              <span className="ctx-label">{item.label}</span>
              <span className="ctx-chord">{hasSub ? '▸' : item.chord}</span>
            </button>
            {hasSub && openSub === idx && !item.disabled && (
              <div className="ctx-menu ctx-submenu">
                <MenuList items={item.submenu!} onClose={onClose} />
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}
