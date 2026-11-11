// The one palette every composition takes as props.theme (MOTION_GRAPHICS_PLAN.md
// Architecture). Built on the Python side (presentation/motion.py::theme_for) from
// the same tables maps.py/charts.py/art_style.py already use for the PIL
// renderers, so a motion composition and its PIL fallback always agree on
// colour. Every field is optional here only so a hand-written sample props
// file (gallery, tests) does not have to fill in every one.
export type Theme = {
  bg: string;
  panel: string;
  fg: string;
  muted: string;
  accent: string;
  accent2: string;
  water: string;
  land: string;
  border: string;
  fontHead: string;
  fontBody: string;
  paper: boolean;
  grain: boolean;
  noir: boolean;
};

export const DEFAULT_THEME: Theme = {
  bg: "#0C0D10",
  panel: "#16181D",
  fg: "#F3EAD3",
  muted: "#8C887E",
  accent: "#E0393E",
  accent2: "#2E6FAB",
  water: "#0B0E12",
  land: "#1A1D22",
  border: "#3E444C",
  fontHead: "Poppins",
  fontBody: "Poppins",
  paper: false,
  grain: false,
  noir: true,
};

export function withTheme(theme?: Partial<Theme>): Theme {
  return { ...DEFAULT_THEME, ...(theme || {}) };
}
