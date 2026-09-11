import { create } from 'zustand';

/**
 * UI appearance: theme, accent colour and density. Like the dock layout, this
 * describes how *this machine* is set up rather than any project, so it lives in
 * localStorage and is applied to the document root as CSS variables /
 * data-attributes that styles.css keys off.
 */

export type ThemeId =
  | 'dark' | 'midnight' | 'graphite' | 'nord' | 'dracula' | 'forest' | 'ember'
  | 'light' | 'paper' | 'contrast';
export type DensityId = 'comfortable' | 'compact';

export interface Appearance {
  accent: string;   // hex, e.g. "#3b82f6"
  theme: ThemeId;
  density: DensityId;
  /** While true, changing theme moves the accent to that theme's own colour.
   *  Picking a swatch turns it off, so a deliberate accent is never overwritten. */
  accentFollowsTheme: boolean;
}

export interface ThemeDef {
  id: ThemeId;
  label: string;
  /** One line in the picker: what this theme is *for*, not what it looks like. */
  note: string;
  mode: 'dark' | 'light';
  /** The accent this palette was drawn around. */
  accent: string;
  /** Three swatches for the picker card: page, panel, text. */
  preview: [string, string, string];
}

/**
 * Every theme, in picker order. The colours here are only the *preview* — the
 * palette itself lives in styles.css under `:root[data-theme="..."]`, so the
 * whole UI stays themed by CSS variables and nothing re-renders on a switch.
 * Keep the two in step when adding one.
 */
export const THEMES: ThemeDef[] = [
  { id: 'dark', label: 'Charcoal', mode: 'dark', accent: '#3b82f6',
    note: 'The default — neutral grey, nothing competing with the picture',
    preview: ['#0f0f0f', '#252525', '#e0e0e0'] },
  { id: 'midnight', label: 'Midnight', mode: 'dark', accent: '#6366f1',
    note: 'Deep navy, low glare for long night sessions',
    preview: ['#070912', '#161a2e', '#dfe3f5'] },
  { id: 'graphite', label: 'Graphite', mode: 'dark', accent: '#f97316',
    note: 'A lighter warm grey — easier in a bright room than Charcoal',
    preview: ['#17181a', '#292c30', '#e6e7e9'] },
  { id: 'nord', label: 'Nord', mode: 'dark', accent: '#88c0d0',
    note: 'Cool arctic blue-grey, soft contrast',
    preview: ['#242933', '#3b4252', '#eceff4'] },
  { id: 'dracula', label: 'Dracula', mode: 'dark', accent: '#bd93f9',
    note: 'Warm purple dark, high colour separation',
    preview: ['#1e1f29', '#343746', '#f8f8f2'] },
  { id: 'forest', label: 'Forest', mode: 'dark', accent: '#34d399',
    note: 'Green-black — keeps a warm grade looking warm',
    preview: ['#0d1411', '#1a2721', '#e3ece6'] },
  { id: 'ember', label: 'Ember', mode: 'dark', accent: '#fb7185',
    note: 'Warm brown-black, the least blue light of the dark set',
    preview: ['#14100e', '#29211d', '#f0e6e0'] },
  { id: 'light', label: 'Daylight', mode: 'light', accent: '#2563eb',
    note: 'Clean white and cool grey for a lit room',
    preview: ['#f4f5f7', '#eceef1', '#1c1e26'] },
  { id: 'paper', label: 'Paper', mode: 'light', accent: '#b45309',
    note: 'Warm off-white — light without the glare of pure white',
    preview: ['#f6f1e7', '#ece5d8', '#2a2620'] },
  { id: 'contrast', label: 'High Contrast', mode: 'dark', accent: '#22d3ee',
    note: 'Pure black, bright text and hard borders, for accessibility',
    preview: ['#000000', '#141414', '#ffffff'] },
];

export const THEME_BY_ID: Record<ThemeId, ThemeDef> =
  Object.fromEntries(THEMES.map((t) => [t.id, t])) as Record<ThemeId, ThemeDef>;

export const ACCENT_SWATCHES: { label: string; value: string }[] = [
  { label: 'Blue', value: '#3b82f6' },
  { label: 'Violet', value: '#8b5cf6' },
  { label: 'Emerald', value: '#10b981' },
  { label: 'Rose', value: '#f43f5e' },
  { label: 'Amber', value: '#f59e0b' },
  { label: 'Cyan', value: '#06b6d4' },
];

const STORAGE_KEY = 'buzzedit.appearance.v1';

const DEFAULTS: Appearance = {
  accent: '#3b82f6', theme: 'dark', density: 'comfortable', accentFollowsTheme: true,
};

function load(): Appearance {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const saved: Appearance = raw ? { ...DEFAULTS, ...JSON.parse(raw) } : DEFAULTS;
    // A theme id saved by a build that had fewer of them, or a hand-edited
    // value, must not leave the root with a data-theme no stylesheet answers to.
    if (!THEME_BY_ID[saved.theme]) saved.theme = DEFAULTS.theme;
    return saved;
  } catch {
    return DEFAULTS;
  }
}

/** Darken/lighten a hex colour by `amt` (-1..1) for the hover/glow variants. */
function shade(hex: string, amt: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  const r = clamp((n >> 16) + 255 * amt);
  const g = clamp(((n >> 8) & 0xff) + 255 * amt);
  const b = clamp((n & 0xff) + 255 * amt);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}

function rgba(hex: string, a: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 0xff}, ${n & 0xff}, ${a})`;
}

/** Perceived brightness (0..1), so a filled accent knows whether to carry white
 *  or black text. Amber on Paper was white-on-yellow and unreadable. */
function luminance(hex: string): number {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return 0;
  const n = parseInt(m[1], 16);
  return (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 0xff) + 0.114 * (n & 0xff)) / 255;
}

/** Write the appearance onto the document root so the whole UI reacts. */
export function applyAppearance(a: Appearance) {
  const root = document.documentElement;
  root.style.setProperty('--accent', a.accent);
  root.style.setProperty('--accent-hover', shade(a.accent, -0.12));
  root.style.setProperty('--accent-glow', rgba(a.accent, 0.3));
  root.style.setProperty('--accent-soft', rgba(a.accent, 0.14));
  root.style.setProperty('--accent-ink', luminance(a.accent) > 0.62 ? '#101114' : '#ffffff');
  root.setAttribute('data-theme', a.theme);
  root.setAttribute('data-density', a.density);
  // Components that need to know only light-vs-dark (the preview letterbox, the
  // timeline lanes) key off this instead of listing every theme id.
  root.setAttribute('data-mode', THEME_BY_ID[a.theme]?.mode ?? 'dark');
}

interface AppearanceState extends Appearance {
  set: (updates: Partial<Appearance>) => void;
  reset: () => void;
}

export const useAppearanceStore = create<AppearanceState>((set, get) => {
  const initial = load();
  // Apply immediately on store creation so the first paint is already themed.
  applyAppearance(initial);

  const commit = (next: Appearance) => {
    applyAppearance(next);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* ignore storage failures */
    }
    set(next as AppearanceState);
  };

  return {
    ...initial,
    set: (updates) => {
      const next: Appearance = { ...pick(get()), ...updates };
      // Choosing a colour by hand pins it; choosing a theme while it is not
      // pinned moves the accent to the one that palette was drawn around, which
      // is the only way Dracula does not arrive wearing the default blue.
      if (updates.accent !== undefined && updates.accentFollowsTheme === undefined) {
        next.accentFollowsTheme = false;
      }
      if (updates.theme !== undefined && next.accentFollowsTheme) {
        next.accent = THEME_BY_ID[next.theme]?.accent ?? next.accent;
      }
      if (updates.accentFollowsTheme === true && updates.accent === undefined) {
        next.accent = THEME_BY_ID[next.theme]?.accent ?? next.accent;
      }
      commit(next);
    },
    reset: () => commit(DEFAULTS),
  };
});

function pick(s: AppearanceState): Appearance {
  return {
    accent: s.accent, theme: s.theme, density: s.density,
    accentFollowsTheme: s.accentFollowsTheme,
  };
}
