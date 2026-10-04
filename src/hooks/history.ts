import { useEffect } from 'react';
import { create } from 'zustand';
import { useProjectStore } from './store';
import { replaceTimeline } from './api';
import { useCommand } from './commands';

/**
 * Undo / redo for the timeline.
 *
 * Every edit — a split on the lanes, a grade in the Inspector, a word struck in
 * the transcript — ends with the backend handing back a new timeline that lands
 * in the project store. Watching the store for that, rather than wrapping each
 * call site, means nothing that changes the edit can forget to be undoable.
 * Undo puts the previous snapshot back on the server (`/replace`) so the next
 * render and the next edit both start from it.
 */

/** Snapshots kept. A long transcript makes each one large, so not unbounded. */
const LIMIT = 60;
/** Changes closer together than this are one step: a slider drag or a burst of
 *  per-clip calls behind one menu command undo together. */
const COALESCE_MS = 600;

interface HistoryState {
  projectId: string | null;
  past: any[];
  future: any[];
  lastPushAt: number;
  /** Set while undo/redo is writing, so its own store update isn't recorded. */
  applying: boolean;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
}

export const useTimelineHistory = create<HistoryState>((set, get) => ({
  projectId: null,
  past: [],
  future: [],
  lastPushAt: 0,
  applying: false,

  undo: async () => {
    const { past, future, applying } = get();
    const project = useProjectStore.getState().project;
    if (applying || !past.length || !project) return;
    const target = past[past.length - 1];
    set({ applying: true });
    try {
      const res = await replaceTimeline(project.id, target);
      set({ past: past.slice(0, -1), future: [...future, project.timeline] });
      useProjectStore.getState().updateProject({ timeline: res.timeline });
    } catch (err: any) {
      useProjectStore.getState().setError(err.message || 'Undo failed');
    } finally {
      set({ applying: false, lastPushAt: 0 });
    }
  },

  redo: async () => {
    const { past, future, applying } = get();
    const project = useProjectStore.getState().project;
    if (applying || !future.length || !project) return;
    const target = future[future.length - 1];
    set({ applying: true });
    try {
      const res = await replaceTimeline(project.id, target);
      set({ future: future.slice(0, -1), past: [...past, project.timeline].slice(-LIMIT) });
      useProjectStore.getState().updateProject({ timeline: res.timeline });
    } catch (err: any) {
      useProjectStore.getState().setError(err.message || 'Redo failed');
    } finally {
      set({ applying: false, lastPushAt: 0 });
    }
  },
}));

/** Record timeline changes and expose Undo / Redo as commands. Mount once. */
export function useTimelineHistoryTracking() {
  useEffect(() => useProjectStore.subscribe((state, prev) => {
    const history = useTimelineHistory.getState();
    const id = state.project?.id ?? null;
    if (id !== history.projectId) {
      useTimelineHistory.setState({ projectId: id, past: [], future: [], lastPushAt: 0 });
      return;
    }
    const before = prev.project?.timeline;
    const after = state.project?.timeline;
    if (!before || before === after || history.applying) return;
    // A reload of the same edit (same revision) is not a change.
    if (after && before.revision != null && before.revision === after.revision) return;

    const now = Date.now();
    if (now - history.lastPushAt < COALESCE_MS) {
      useTimelineHistory.setState({ future: [], lastPushAt: now });
      return;
    }
    useTimelineHistory.setState({
      past: [...history.past, before].slice(-LIMIT),
      future: [],
      lastPushAt: now,
    });
  }), []);

  const undo = useTimelineHistory((s) => s.undo);
  const redo = useTimelineHistory((s) => s.redo);
  useCommand('edit.undo', undo);
  useCommand('edit.redo', redo);
}
