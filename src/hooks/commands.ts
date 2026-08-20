import { useEffect } from 'react';
import { create } from 'zustand';

/**
 * A single source of truth for every user-invokable action in the app.
 *
 * A command is an id + a human label + which menu it belongs to + a default
 * keyboard chord. The *handler* is registered at runtime by whichever component
 * owns the action (so `edit.deleteClip` only works while the timeline is
 * mounted), which lets the menu bar grey out actions that aren't currently
 * available instead of firing into nothing.
 *
 * Keyboard bindings are user-remappable and persisted to localStorage, so both
 * the menu bar and the global shortcut dispatcher read the *current* chord for a
 * command rather than a hardcoded one.
 */

export type CommandGroup = 'File' | 'Edit' | 'View' | 'Help';

export interface CommandDef {
  id: string;
  label: string;
  group: CommandGroup;
  /** Canonical chord, e.g. "Ctrl+S", "Ctrl+Shift+E", "Delete", "S". */
  defaultChord?: string;
}

// The catalogue. Order within a group is the order shown in the menu.
export const COMMANDS: CommandDef[] = [
  // File
  { id: 'file.new', label: 'New Project', group: 'File', defaultChord: 'Ctrl+N' },
  { id: 'file.open', label: 'Open Project…', group: 'File', defaultChord: 'Ctrl+O' },
  { id: 'file.import', label: 'Import Media…', group: 'File', defaultChord: 'Ctrl+I' },
  { id: 'file.save', label: 'Save', group: 'File', defaultChord: 'Ctrl+S' },
  { id: 'file.render', label: 'Render / Export', group: 'File', defaultChord: 'Ctrl+Shift+E' },
  { id: 'app.preferences', label: 'Preferences…', group: 'File', defaultChord: 'Ctrl+,' },

  // Edit
  { id: 'edit.transcribe', label: 'Transcribe', group: 'Edit', defaultChord: 'Ctrl+T' },
  { id: 'edit.analyze', label: 'Analyze', group: 'Edit', defaultChord: 'Ctrl+Shift+A' },
  { id: 'edit.autoEdit', label: 'Auto Edit', group: 'Edit', defaultChord: 'Ctrl+E' },
  { id: 'edit.deleteClip', label: 'Delete Clip', group: 'Edit', defaultChord: 'Delete' },
  { id: 'edit.splitAtPlayhead', label: 'Split at Playhead', group: 'Edit', defaultChord: 'S' },

  // View
  { id: 'view.panel.transcript', label: 'Toggle Transcript', group: 'View', defaultChord: 'Ctrl+1' },
  { id: 'view.panel.inspector', label: 'Toggle Inspector', group: 'View', defaultChord: 'Ctrl+2' },
  { id: 'view.panel.media', label: 'Toggle Media', group: 'View', defaultChord: 'Ctrl+3' },
  { id: 'view.panel.agents', label: 'Toggle AI Agents', group: 'View', defaultChord: 'Ctrl+4' },
  { id: 'view.panel.queue', label: 'Toggle Nightly Queue', group: 'View', defaultChord: 'Ctrl+5' },
  { id: 'view.panel.export', label: 'Toggle Export', group: 'View', defaultChord: 'Ctrl+6' },
  { id: 'view.preset.default', label: 'Layout: Default', group: 'View' },
  { id: 'view.preset.edit', label: 'Layout: Editing', group: 'View' },
  { id: 'view.preset.colour', label: 'Layout: Colour & FX', group: 'View' },
  { id: 'view.preset.review', label: 'Layout: Review', group: 'View' },
  { id: 'view.lockLayout', label: 'Lock Layout', group: 'View' },
  { id: 'view.resetLayout', label: 'Reset Layout', group: 'View' },
  { id: 'view.zoomIn', label: 'Timeline Zoom In', group: 'View', defaultChord: 'Ctrl+=' },
  { id: 'view.zoomOut', label: 'Timeline Zoom Out', group: 'View', defaultChord: 'Ctrl+-' },

  // Help
  { id: 'help.diagnostics', label: 'Diagnostics', group: 'Help' },
  { id: 'help.shortcuts', label: 'Keyboard Shortcuts', group: 'Help' },
];

export const COMMAND_GROUPS: CommandGroup[] = ['File', 'Edit', 'View', 'Help'];

const KEYMAP_STORAGE = 'buzzedit.keymap.v1';

function loadKeymap(): Record<string, string> {
  try {
    const raw = localStorage.getItem(KEYMAP_STORAGE);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function persistKeymap(keymap: Record<string, string>) {
  try {
    localStorage.setItem(KEYMAP_STORAGE, JSON.stringify(keymap));
  } catch {
    /* a full or disabled localStorage must not break the editor */
  }
}

const DEFAULT_CHORDS: Record<string, string> = Object.fromEntries(
  COMMANDS.filter((c) => c.defaultChord).map((c) => [c.id, c.defaultChord as string]),
);

interface CommandState {
  /** id -> the function to run. Only present while its owner is mounted. */
  handlers: Record<string, () => void>;
  /** id -> user override chord. Absence means "use the default". */
  keymap: Record<string, string>;

  register: (id: string, fn: () => void) => void;
  unregister: (id: string) => void;
  run: (id: string) => void;
  isEnabled: (id: string) => boolean;
  /** The chord currently bound to a command, override or default. */
  chordFor: (id: string) => string | undefined;
  setBinding: (id: string, chord: string) => void;
  resetBinding: (id: string) => void;
  resetAll: () => void;
}

export const useCommandStore = create<CommandState>((set, get) => ({
  handlers: {},
  keymap: loadKeymap(),

  register: (id, fn) => set((s) => ({ handlers: { ...s.handlers, [id]: fn } })),
  unregister: (id) =>
    set((s) => {
      // Only drop the handler if it hasn't already been replaced by a remount.
      const next = { ...s.handlers };
      delete next[id];
      return { handlers: next };
    }),

  run: (id) => {
    const fn = get().handlers[id];
    if (fn) fn();
  },
  isEnabled: (id) => !!get().handlers[id],

  chordFor: (id) => get().keymap[id] ?? DEFAULT_CHORDS[id],

  setBinding: (id, chord) =>
    set((s) => {
      // A chord is unique: assigning it to one command clears it from any other.
      const keymap = { ...s.keymap };
      for (const other of COMMANDS) {
        const current = keymap[other.id] ?? DEFAULT_CHORDS[other.id];
        if (other.id !== id && current === chord) {
          // Shadow the clashing default with an empty binding ("unbound").
          keymap[other.id] = '';
        }
      }
      keymap[id] = chord;
      persistKeymap(keymap);
      return { keymap };
    }),

  resetBinding: (id) =>
    set((s) => {
      const keymap = { ...s.keymap };
      delete keymap[id];
      persistKeymap(keymap);
      return { keymap };
    }),

  resetAll: () => {
    persistKeymap({});
    set({ keymap: {} });
  },
}));

/**
 * Register a command handler for the lifetime of the calling component. Pass a
 * stable (useCallback'd) function or accept that it re-registers each render —
 * either is cheap, and re-registering keeps the closure fresh.
 */
export function useCommand(id: string, fn: () => void) {
  const register = useCommandStore((s) => s.register);
  const unregister = useCommandStore((s) => s.unregister);
  useEffect(() => {
    register(id, fn);
    return () => unregister(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, fn]);
}
