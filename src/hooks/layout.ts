import { create } from 'zustand';

/**
 * Workspace layout: where each panel lives, how big the docks are, and whether
 * the whole arrangement is pinned.
 *
 * Panels are either docked in one of four zones or floating free above them.
 * Layout is deliberately kept in localStorage rather than the project file — it
 * describes how *this machine* is set up, not the edit, so it should not travel
 * with a project or be undone by opening one.
 */

/**
 * Every panel docks on its own. They used to be three — Preview, Timeline, and a
 * single "Tools" panel holding a fixed tab bar — which meant the transcript and
 * the inspector could never be on screen at the same time however the workspace
 * was arranged. Splitting them makes the tab bar a *property of a dock zone*
 * rather than of one panel: several panels sharing a zone are tabbed, and pulling
 * one out into its own zone is just a drag.
 */
export type PanelId =
  | 'preview' | 'media' | 'transcript' | 'inspector'
  | 'effects' | 'transitions'
  | 'timeline' | 'agents' | 'queue' | 'export';
export type DockZone = 'center' | 'left' | 'right' | 'bottom';
export type PanelPlacement = DockZone | 'floating';

export const DOCK_ZONES: DockZone[] = ['center', 'left', 'right', 'bottom'];

export interface FloatRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PanelConfig {
  placement: PanelPlacement;
  visible: boolean;
  /** Remembered even while docked, so undocking returns it where it was. */
  rect: FloatRect;
}

export interface ZoneSizes {
  left: number;
  right: number;
  bottom: number;
}

export const PANEL_TITLES: Record<PanelId, string> = {
  preview: 'Preview',
  media: 'Media',
  transcript: 'Transcript',
  inspector: 'Inspector',
  effects: 'Effects',
  transitions: 'Transitions',
  timeline: 'Timeline',
  agents: 'AI Agents',
  queue: 'Nightly Queue',
  export: 'Export',
};

/** Short labels for the tab strip, where space is tight. */
export const PANEL_TABS: Record<PanelId, string> = {
  preview: 'Preview',
  media: 'Media',
  transcript: 'Transcript',
  inspector: 'Inspect',
  effects: 'Effects',
  transitions: 'Transitions',
  timeline: 'Timeline',
  agents: 'Agents',
  queue: 'Queue',
  export: 'Export',
};

const MIN_ZONE = 180;
const MAX_ZONE = 900;
const STORAGE_KEY = 'buzzedit.layout.v1';
// The editor used to be called BuzzcafEditor. A layout saved under the old key is
// still the arrangement this user chose, so carry it over once rather than
// dropping them back to the default the first time they open the renamed app.
const LEGACY_STORAGE_KEY = 'buzzcaf.layout.v1';

export interface LayoutPreset {
  id: string;
  label: string;
  panels: Record<PanelId, PanelConfig>;
  sizes: ZoneSizes;
}

const defaultRect = (index: number): FloatRect => ({
  x: 140 + index * 36,
  y: 120 + index * 36,
  width: 520,
  height: 360,
});

/**
 * The Media panel lives on the left, beside the preview, in every arrangement.
 * It is the one panel that is a *source* rather than a view of the edit — you
 * drag from it onto the timeline — so it has to stay reachable no matter what
 * else is on screen. Presets may resize the left zone but never move Media out
 * of it.
 */
const MEDIA_HOME: DockZone = 'left';

function makePanels(overrides: Partial<Record<PanelId, Partial<PanelConfig>>> = {}) {
  const base: Record<PanelId, PanelConfig> = {
    preview: { placement: 'center', visible: true, rect: defaultRect(0) },
    media: { placement: MEDIA_HOME, visible: true, rect: defaultRect(1) },
    transcript: { placement: 'right', visible: true, rect: defaultRect(2) },
    inspector: { placement: 'right', visible: true, rect: defaultRect(3) },
    // Tabbed beside the Inspector: they act on the same selection, and a cut is
    // usually being dressed at the same time it is being graded.
    effects: { placement: 'right', visible: true, rect: defaultRect(4) },
    transitions: { placement: 'right', visible: true, rect: defaultRect(5) },
    timeline: { placement: 'bottom', visible: true, rect: defaultRect(6) },
    agents: { placement: 'right', visible: true, rect: defaultRect(7) },
    queue: { placement: 'right', visible: false, rect: defaultRect(8) },
    export: { placement: 'right', visible: true, rect: defaultRect(9) },
  };
  (Object.keys(overrides) as PanelId[]).forEach((id) => {
    base[id] = { ...base[id], ...overrides[id] };
  });
  return base;
}

export const LAYOUT_PRESETS: LayoutPreset[] = [
  {
    id: 'default',
    label: 'Default',
    panels: makePanels(),
    sizes: { left: 300, right: 400, bottom: 300 },
  },
  {
    id: 'edit',
    label: 'Editing',
    // The transcript beside the picture and a tall timeline: the arrangement for
    // cutting, where the words are what you are working on.
    panels: makePanels({
      queue: { visible: false }, agents: { visible: false },
      effects: { visible: false }, transitions: { visible: false },
    }),
    sizes: { left: 280, right: 420, bottom: 440 },
  },
  {
    id: 'colour',
    label: 'Colour & FX',
    // Big picture, wide inspector on its own, timeline out of the way. Grading
    // and keying are judged by eye, so the preview gets the room — and this is
    // the arrangement where the effects and transitions browsers belong.
    panels: makePanels({
      transcript: { visible: false },
      agents: { visible: false },
      export: { visible: false },
    }),
    sizes: { left: 250, right: 500, bottom: 170 },
  },
  {
    id: 'review',
    label: 'Review',
    // Just the picture and the timeline, with the media library still to hand.
    panels: makePanels({
      transcript: { visible: false },
      inspector: { visible: false },
      effects: { visible: false },
      transitions: { visible: false },
      agents: { visible: false },
      export: { visible: false },
    }),
    sizes: { left: 260, right: 400, bottom: 260 },
  },
];

export type ZoneTabs = Partial<Record<DockZone, PanelId>>;

interface LayoutState {
  panels: Record<PanelId, PanelConfig>;
  sizes: ZoneSizes;
  locked: boolean;
  preset: string;
  /** Which panel is on top in each zone. Zones stack their panels as tabs. */
  active: ZoneTabs;
  /** Panel currently being dragged, so the dock targets can light up. */
  dragging: PanelId | null;

  setPlacement: (id: PanelId, placement: PanelPlacement) => void;
  toggleVisible: (id: PanelId, visible?: boolean) => void;
  setRect: (id: PanelId, rect: Partial<FloatRect>) => void;
  setZoneSize: (zone: keyof ZoneSizes, size: number) => void;
  setLocked: (locked: boolean) => void;
  setDragging: (id: PanelId | null) => void;
  setActive: (zone: DockZone, id: PanelId) => void;
  applyPreset: (presetId: string) => void;
  resetLayout: () => void;
}

type Persisted = Pick<LayoutState, 'panels' | 'sizes' | 'locked' | 'preset' | 'active'>;

// Panel ids that existed in the three-panel build. A stored layout naming them
// would otherwise leave the new panels at their defaults *and* keep a dead entry.
const RETIRED_PANELS = new Set(['tools']);

function load(): Persisted {
  const fallback: Persisted = {
    panels: LAYOUT_PRESETS[0].panels,
    sizes: LAYOUT_PRESETS[0].sizes,
    locked: false,
    preset: 'default',
    active: {},
  };
  try {
    let raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      raw = localStorage.getItem(LEGACY_STORAGE_KEY);
      if (raw) localStorage.setItem(STORAGE_KEY, raw);
    }
    if (!raw) return fallback;
    const saved = JSON.parse(raw);
    const storedPanels = { ...(saved.panels ?? {}) };
    RETIRED_PANELS.forEach((id) => delete storedPanels[id]);
    // Merge rather than replace: a stored layout from an older build may not
    // mention a panel that exists now, and a missing panel would render nothing.
    return {
      panels: makePanels(storedPanels),
      sizes: { ...fallback.sizes, ...(saved.sizes ?? {}) },
      locked: !!saved.locked,
      preset: saved.preset ?? 'custom',
      active: saved.active ?? {},
    };
  } catch {
    return fallback;
  }
}

function persist(state: LayoutState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      panels: state.panels,
      sizes: state.sizes,
      locked: state.locked,
      preset: state.preset,
      active: state.active,
    }));
  } catch {
    /* a full or disabled localStorage must not break the editor */
  }
}

const clampZone = (size: number) => Math.max(MIN_ZONE, Math.min(MAX_ZONE, Math.round(size)));

export const useLayoutStore = create<LayoutState>((set, get) => {
  const saved = load();

  const commit = (partial: Partial<LayoutState>) => {
    set(partial as LayoutState);
    persist(get());
  };

  return {
    ...saved,
    dragging: null,

    setPlacement: (id, placement) => {
      if (get().locked) return;
      commit({
        panels: { ...get().panels, [id]: { ...get().panels[id], placement, visible: true } },
        // A panel moved into a zone becomes the one you are looking at there.
        // Docking it behind an existing tab looks like the drag did nothing.
        active: placement === 'floating'
          ? get().active
          : { ...get().active, [placement]: id },
        preset: 'custom',
      });
    },

    toggleVisible: (id, visible) => {
      const current = get().panels[id];
      const shown = visible ?? !current.visible;
      const active = { ...get().active };
      if (shown && current.placement !== 'floating') {
        active[current.placement] = id;
      } else if (!shown && active[current.placement as DockZone] === id) {
        // Its zone has to fall back to something, or it renders an empty body
        // with a tab strip that has no selection.
        delete active[current.placement as DockZone];
      }
      commit({
        panels: { ...get().panels, [id]: { ...current, visible: shown } },
        active,
        preset: 'custom',
      });
    },

    setActive: (zone, id) => commit({ active: { ...get().active, [zone]: id } }),

    setRect: (id, rect) => {
      if (get().locked) return;
      const current = get().panels[id];
      commit({
        panels: { ...get().panels, [id]: { ...current, rect: { ...current.rect, ...rect } } },
        preset: 'custom',
      });
    },

    setZoneSize: (zone, size) => {
      if (get().locked) return;
      commit({ sizes: { ...get().sizes, [zone]: clampZone(size) }, preset: 'custom' });
    },

    setLocked: (locked) => commit({ locked }),

    setDragging: (id) => set({ dragging: id } as LayoutState),

    applyPreset: (presetId) => {
      const preset = LAYOUT_PRESETS.find((p) => p.id === presetId);
      if (!preset) return;
      // A preset repositions panels, so it has to override the lock rather than
      // be silently ignored — the user picked it deliberately.
      commit({
        panels: JSON.parse(JSON.stringify(preset.panels)),
        sizes: { ...preset.sizes },
        active: {},
        preset: preset.id,
      });
    },

    resetLayout: () => get().applyPreset('default'),
  };
});
