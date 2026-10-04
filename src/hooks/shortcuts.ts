import { useEffect } from 'react';
import { COMMANDS, useCommandStore } from './commands';

/**
 * Turns keyboard events into canonical chord strings and dispatches them to the
 * matching command. One listener for the whole app, mounted once in App.
 */

// How a physical key reports itself vs. how we want to name it in a chord.
const KEY_ALIASES: Record<string, string> = {
  ' ': 'Space',
  Spacebar: 'Space',
  Esc: 'Escape',
  Del: 'Delete',
  '+': '=', // so Ctrl+= and Ctrl++ (shifted) both read as "Ctrl+=" for zoom-in
};

/** Normalize one key name to its chord token (modifiers handled separately). */
function keyToken(key: string): string {
  if (KEY_ALIASES[key]) return KEY_ALIASES[key];
  // Single printable letters are upper-cased so "s" and "S" are the same chord.
  if (key.length === 1) return key.toUpperCase();
  return key; // Delete, Enter, ArrowLeft, F5, etc. — already canonical
}

/** Build the canonical chord for a keyboard event, e.g. "Ctrl+Shift+E". */
export function eventToChord(e: KeyboardEvent): string {
  const parts: string[] = [];
  if (e.ctrlKey) parts.push('Ctrl');
  if (e.altKey) parts.push('Alt');
  if (e.shiftKey) parts.push('Shift');
  if (e.metaKey) parts.push('Meta');
  const token = keyToken(e.key);
  // A bare modifier press ("Control") is not a chord on its own.
  if (['Control', 'Alt', 'Shift', 'Meta'].includes(e.key)) return parts.join('+');
  parts.push(token);
  return parts.join('+');
}

/** Human-friendly rendering of a chord for menus and the shortcuts editor. */
export function chordLabel(chord?: string): string {
  if (!chord) return '';
  return chord
    .split('+')
    .map((p) => (p === 'Delete' ? 'Del' : p === 'Escape' ? 'Esc' : p))
    .join('+');
}

/** True when the event target is a field we must not steal keys from. A
 *  dropdown counts: its arrow keys pick an option, they don't step frames. */
function inTextField(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

function hasTextSelection(): boolean {
  try {
    return !!window.getSelection()?.toString();
  } catch {
    return false;
  }
}

/**
 * Attach the single global keydown → command dispatcher. Call once, high in the
 * tree. It reads the live keymap on every event, so rebinding a shortcut in
 * Preferences takes effect immediately with no re-mount.
 */
export function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (inTextField(e.target)) return;
      const chord = eventToChord(e);
      if (!chord || ['Ctrl', 'Alt', 'Shift', 'Meta'].includes(chord)) return;

      const { keymap, run, isEnabled } = useCommandStore.getState();
      // Resolve chord -> command id using overrides on top of defaults. An empty
      // override means the command was explicitly unbound, so it never matches.
      // Main chords are matched before extra ones, so a user binding always wins
      // over another command's secondary default.
      const fire = (cmd: (typeof COMMANDS)[number]) => {
        if (!isEnabled(cmd.id)) return false;
        if (cmd.yieldsToTextSelection && hasTextSelection()) return false;
        e.preventDefault();
        run(cmd.id);
        return true;
      };
      for (const cmd of COMMANDS) {
        const bound = cmd.id in keymap ? keymap[cmd.id] : cmd.defaultChord;
        if (bound && bound === chord && fire(cmd)) return;
      }
      for (const cmd of COMMANDS) {
        if (cmd.id in keymap) continue;
        if ((cmd.extraChords ?? []).includes(chord) && fire(cmd)) return;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}
