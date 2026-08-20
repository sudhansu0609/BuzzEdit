import { create } from 'zustand';

/**
 * UI appearance: accent colour, theme and density. Like the dock layout, this
 * describes how *this machine* is set up rather than any project, so it lives in
 * localStorage and is applied to the document root as CSS variables /
 * data-attributes that styles.css keys off.
 */

export type ThemeId = 'dark' | 'midnight' | 'light';
export type DensityId = 'comfortable' | 'compact';

export interface Appearance {
  accent: string;   // hex, e.g. "#3b82f6"
  theme: ThemeId;
  density: DensityId;
}

export const ACCENT_SWATCHES: { label: string; value: string }[] = [
  { label: 'Blue', value: '#3b82f6' },
  { label: 'Violet', value: '#8b5cf6' },
  { label: 'Emerald', value: '#10b981' },
  { label: 'Rose', value: '#f43f5e' },
  { label: 'Amber', value: '#f59e0b' },
  { label: 'Cyan', value: '#06b6d4' },
];

const STORAGE_KEY = 'buzzedit.appearance.v1';

const DEFAULTS: Appearance = { accent: '#3b82f6', theme: 'dark', density: 'comfortable' };

function load(): Appearance {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : DEFAULTS;
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

/** Write the appearance onto the document root so the whole UI reacts. */
export function applyAppearance(a: Appearance) {
  const root = document.documentElement;
  root.style.setProperty('--accent', a.accent);
  root.style.setProperty('--accent-hover', shade(a.accent, -0.12));
  root.style.setProperty('--accent-glow', rgba(a.accent, 0.3));
  root.setAttribute('data-theme', a.theme);
  root.setAttribute('data-density', a.density);
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
    set: (updates) => commit({ ...pick(get()), ...updates }),
    reset: () => commit(DEFAULTS),
  };
});

function pick(s: AppearanceState): Appearance {
  return { accent: s.accent, theme: s.theme, density: s.density };
}
