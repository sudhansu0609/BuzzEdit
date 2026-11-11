// @font-face rules for the font files motion/scripts/copy_fonts.mjs copies
// into public/fonts/ on every `npm install` (MOTION_GRAPHICS_PLAN.md
// Architecture: "Fonts are files in public/fonts/ ... loaded with
// staticFile"). Poppins covers English/Hinglish headings; Noto Sans
// Devanagari covers Hindi script -- both brought in as ordinary npm
// dependencies (@fontsource/*) rather than downloaded by hand, so a plain
// `npm install` always leaves the right files in place.
import { staticFile } from "remotion";

type FontFile = { family: string; weight: number; file: string };

const FILES: FontFile[] = [
  { family: "Poppins", weight: 400, file: "poppins-latin-400-normal.woff2" },
  { family: "Poppins", weight: 600, file: "poppins-latin-600-normal.woff2" },
  { family: "Poppins", weight: 700, file: "poppins-latin-700-normal.woff2" },
  { family: "Noto Sans Devanagari", weight: 400, file: "noto-sans-devanagari-devanagari-400-normal.woff2" },
  { family: "Noto Sans Devanagari", weight: 600, file: "noto-sans-devanagari-devanagari-600-normal.woff2" },
  { family: "Noto Sans Devanagari", weight: 700, file: "noto-sans-devanagari-devanagari-700-normal.woff2" },
  { family: "Patrick Hand", weight: 400, file: "patrick-hand-latin-400-normal.woff2" },
];

export function fontFaceCss(): string {
  return FILES.map(({ family, weight, file }) => `
@font-face {
  font-family: '${family}';
  src: url('${staticFile(`fonts/${file}`)}') format('woff2');
  font-weight: ${weight};
  font-style: normal;
  font-display: block;
}`).join("\n");
}

// A stack that falls back from a themed heading/body font to Poppins, then to
// Noto Sans Devanagari so Hindi (Devanagari) words in an otherwise-Latin
// string still render instead of showing tofu boxes.
export function fontStack(preferred: string): string {
  const names = Array.from(new Set([preferred, "Poppins", "Noto Sans Devanagari"]));
  return names.map((n) => `'${n}'`).join(", ") + ", sans-serif";
}

// StickScene's caption (MOTION_GRAPHICS_PLAN.md M7 review finding 6): a
// handwriting face first, falling back through the normal stack so a
// Devanagari caption still renders instead of tofu boxes -- Patrick Hand
// itself has no Devanagari glyphs.
export function handFontStack(): string {
  return `'Patrick Hand', ${fontStack("Poppins")}`;
}
