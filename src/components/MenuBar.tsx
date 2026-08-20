import { useEffect, useRef, useState } from 'react';
import { COMMANDS, COMMAND_GROUPS, CommandGroup, useCommandStore } from '../hooks/commands';
import { chordLabel } from '../hooks/shortcuts';

/**
 * The in-app application menu (File / Edit / View / Help). The window is
 * frameless with a custom titlebar, so this replaces the OS menu bar. Every item
 * is driven by the command registry: it shows the command's current (remappable)
 * chord and is disabled when no handler is currently registered for it.
 */
export default function MenuBar() {
  const [open, setOpen] = useState<CommandGroup | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  const run = useCommandStore((s) => s.run);
  const isEnabled = useCommandStore((s) => s.isEnabled);
  const chordFor = useCommandStore((s) => s.chordFor);
  // Subscribe to keymap so chord labels update live after a rebind.
  useCommandStore((s) => s.keymap);
  useCommandStore((s) => s.handlers);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) setOpen(null);
    };
    const onEsc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(null);
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onEsc);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onEsc);
    };
  }, [open]);

  const handleItem = (id: string) => {
    setOpen(null);
    run(id);
  };

  return (
    <div className="menubar" ref={barRef}>
      {COMMAND_GROUPS.map((group) => (
        <div key={group} className="menubar-item">
          <button
            className={`menubar-btn ${open === group ? 'active' : ''}`}
            onClick={() => setOpen(open === group ? null : group)}
            onMouseEnter={() => open && setOpen(group)}
          >
            {group}
          </button>
          {open === group && (
            <div className="menu-dropdown">
              {COMMANDS.filter((c) => c.group === group).map((cmd) => {
                const enabled = isEnabled(cmd.id);
                return (
                  <button
                    key={cmd.id}
                    className="menu-option"
                    disabled={!enabled}
                    onClick={() => handleItem(cmd.id)}
                  >
                    <span className="menu-option-label">{cmd.label}</span>
                    <span className="menu-option-chord">{chordLabel(chordFor(cmd.id))}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
