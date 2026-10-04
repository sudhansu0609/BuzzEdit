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

export type CommandGroup = 'File' | 'Edit' | 'Timeline' | 'Playback' | 'View' | 'Help';

export interface CommandDef {
  id: string;
  label: string;
  group: CommandGroup;
  /** Canonical chord, e.g. "Ctrl+S", "Ctrl+Shift+E", "Delete", "S". */
  defaultChord?: string;
  /** Further default chords that also fire it (S beside Ctrl+B for split).
   *  They stop applying once the user rebinds the command. */
  extraChords?: string[];
  /** Clipboard keys: leave them to the browser while the user has text selected,
   *  so copying a sentence out of the transcript still works. */
  yieldsToTextSelection?: boolean;
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
  { id: 'edit.undo', label: 'Undo', group: 'Edit', defaultChord: 'Ctrl+Z' },
  { id: 'edit.redo', label: 'Redo', group: 'Edit', defaultChord: 'Ctrl+Y', extraChords: ['Ctrl+Shift+Z'] },
  { id: 'edit.cut', label: 'Cut', group: 'Edit', defaultChord: 'Ctrl+X', yieldsToTextSelection: true },
  { id: 'edit.copy', label: 'Copy', group: 'Edit', defaultChord: 'Ctrl+C', yieldsToTextSelection: true },
  { id: 'edit.paste', label: 'Paste', group: 'Edit', defaultChord: 'Ctrl+V' },
  { id: 'edit.pasteInsert', label: 'Paste Insert (push clips right)', group: 'Edit', defaultChord: 'Ctrl+Shift+V' },
  { id: 'edit.duplicate', label: 'Duplicate', group: 'Edit', defaultChord: 'Ctrl+D' },
  { id: 'edit.deleteClip', label: 'Delete', group: 'Edit', defaultChord: 'Delete', extraChords: ['Backspace'] },
  { id: 'edit.rippleDelete', label: 'Ripple Delete', group: 'Edit', defaultChord: 'Shift+Delete', extraChords: ['Shift+Backspace'] },
  { id: 'edit.selectAll', label: 'Select All Clips', group: 'Edit', defaultChord: 'Ctrl+A' },
  { id: 'edit.deselectAll', label: 'Deselect All', group: 'Edit', defaultChord: 'Escape' },
  { id: 'edit.transcribe', label: 'Transcribe', group: 'Edit', defaultChord: 'Ctrl+T' },
  { id: 'edit.analyze', label: 'Analyze', group: 'Edit', defaultChord: 'Ctrl+Shift+A' },
  { id: 'edit.autoEdit', label: 'Auto Edit', group: 'Edit', defaultChord: 'Ctrl+E' },

  // Timeline
  { id: 'edit.splitAtPlayhead', label: 'Split at Playhead', group: 'Timeline', defaultChord: 'Ctrl+B', extraChords: ['S'] },
  { id: 'timeline.trimStart', label: 'Trim Start to Playhead', group: 'Timeline', defaultChord: 'Alt+[' },
  { id: 'timeline.trimEnd', label: 'Trim End to Playhead', group: 'Timeline', defaultChord: 'Alt+]' },
  { id: 'timeline.rippleTrimStart', label: 'Ripple Trim Start to Playhead', group: 'Timeline', defaultChord: 'Q' },
  { id: 'timeline.rippleTrimEnd', label: 'Ripple Trim End to Playhead', group: 'Timeline', defaultChord: 'W' },
  { id: 'timeline.closeGap', label: 'Close Gap at Playhead', group: 'Timeline' },
  { id: 'timeline.selectForward', label: 'Select Clips After Playhead', group: 'Timeline' },
  { id: 'timeline.link', label: 'Link / Unlink Audio & Video', group: 'Timeline', defaultChord: 'Ctrl+L' },
  { id: 'timeline.detachAudio', label: 'Detach Audio', group: 'Timeline', defaultChord: 'Ctrl+Alt+D' },
  { id: 'timeline.compound', label: 'Create Compound Clip', group: 'Timeline', defaultChord: 'Alt+G' },
  { id: 'timeline.uncompound', label: 'Break Apart Compound Clip', group: 'Timeline', defaultChord: 'Alt+Shift+G' },
  { id: 'timeline.toggleEnabled', label: 'Enable / Disable Clip', group: 'Timeline', defaultChord: 'Shift+E' },
  { id: 'timeline.toggleMute', label: 'Mute / Unmute Clip Audio', group: 'Timeline', defaultChord: 'Ctrl+Shift+M' },
  { id: 'timeline.toggleLock', label: 'Lock / Unlock Clip', group: 'Timeline' },
  { id: 'timeline.rename', label: 'Rename Clip', group: 'Timeline', defaultChord: 'F2' },
  { id: 'timeline.rotateCW', label: 'Rotate 90° Clockwise', group: 'Timeline' },
  { id: 'timeline.rotateCCW', label: 'Rotate 90° Counter-clockwise', group: 'Timeline' },
  { id: 'timeline.flipH', label: 'Flip Horizontal', group: 'Timeline' },
  { id: 'timeline.flipV', label: 'Flip Vertical', group: 'Timeline' },
  { id: 'timeline.resetTransform', label: 'Reset Transform', group: 'Timeline' },
  { id: 'timeline.nudgeLeft', label: 'Nudge Clip Left 1 Frame', group: 'Timeline', defaultChord: 'Alt+ArrowLeft' },
  { id: 'timeline.nudgeRight', label: 'Nudge Clip Right 1 Frame', group: 'Timeline', defaultChord: 'Alt+ArrowRight' },
  { id: 'timeline.toggleSnap', label: 'Snapping', group: 'Timeline', defaultChord: 'N' },
  { id: 'edit.markIn', label: 'Mark In', group: 'Timeline', defaultChord: 'I' },
  { id: 'edit.markOut', label: 'Mark Out', group: 'Timeline', defaultChord: 'O' },

  // Playback
  { id: 'playback.playPause', label: 'Play / Pause', group: 'Playback', defaultChord: 'Space' },
  { id: 'playback.prevFrame', label: 'Previous Frame', group: 'Playback', defaultChord: 'ArrowLeft' },
  { id: 'playback.nextFrame', label: 'Next Frame', group: 'Playback', defaultChord: 'ArrowRight' },
  { id: 'playback.back1s', label: 'Back 1 Second', group: 'Playback', defaultChord: 'Shift+ArrowLeft' },
  { id: 'playback.fwd1s', label: 'Forward 1 Second', group: 'Playback', defaultChord: 'Shift+ArrowRight' },
  { id: 'playback.prevEdit', label: 'Previous Edit Point', group: 'Playback', defaultChord: 'ArrowUp' },
  { id: 'playback.nextEdit', label: 'Next Edit Point', group: 'Playback', defaultChord: 'ArrowDown' },
  { id: 'playback.start', label: 'Go to Start', group: 'Playback', defaultChord: 'Home' },
  { id: 'playback.end', label: 'Go to End', group: 'Playback', defaultChord: 'End' },

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
  { id: 'view.zoomFit', label: 'Zoom Timeline to Fit', group: 'View', defaultChord: 'Shift+Z' },

  // Help
  { id: 'help.diagnostics', label: 'Diagnostics', group: 'Help' },
  { id: 'help.shortcuts', label: 'Keyboard Shortcuts', group: 'Help' },
];

export const COMMAND_GROUPS: CommandGroup[] = ['File', 'Edit', 'Timeline', 'Playback', 'View', 'Help'];

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
        if (other.id === id) continue;
        const current = keymap[other.id] ?? DEFAULT_CHORDS[other.id];
        const extraClash = !(other.id in keymap) && (other.extraChords ?? []).includes(chord);
        if (current === chord || extraClash) {
          // Pin the other command to what it keeps — its main chord, unless that
          // was the clash ("unbound"). Any override also retires extra chords.
          keymap[other.id] = current === chord ? '' : current ?? '';
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
