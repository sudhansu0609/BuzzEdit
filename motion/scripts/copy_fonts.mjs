// Copies the woff2 files KineticText needs out of the @fontsource packages
// (installed as ordinary npm dependencies) into public/fonts/, where
// Remotion's staticFile() can serve them to the headless browser. Run as
// package.json's "postinstall", so a plain `npm install` always leaves
// public/fonts/ in place -- no manual font download.
import { mkdirSync, copyFileSync, readdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const outDir = join(root, "public", "fonts");
mkdirSync(outDir, { recursive: true });

// [package, subset-weight-style patterns we want, destination name prefix]
const WANTED = [
  { pkg: "@fontsource/poppins", prefix: "poppins", subsets: ["latin"], weights: ["400", "600", "700"] },
  { pkg: "@fontsource/noto-sans-devanagari", prefix: "noto-sans-devanagari", subsets: ["devanagari"],
    weights: ["400", "600", "700"] },
  { pkg: "@fontsource/patrick-hand", prefix: "patrick-hand", subsets: ["latin"], weights: ["400"] },
];

let copied = 0;
for (const { pkg, prefix, subsets, weights } of WANTED) {
  const filesDir = join(root, "node_modules", pkg, "files");
  if (!existsSync(filesDir)) {
    console.warn(`[copy_fonts] ${pkg}/files not found -- skipping (run npm install first)`);
    continue;
  }
  for (const name of readdirSync(filesDir)) {
    if (!name.endsWith(".woff2")) continue;
    const subsetMatch = subsets.some((s) => name.startsWith(`${prefix}-${s}-`));
    const weightMatch = weights.some((w) => name.includes(`-${w}-normal`));
    if (!subsetMatch || !weightMatch) continue;
    copyFileSync(join(filesDir, name), join(outDir, name));
    copied++;
  }
}
console.log(`[copy_fonts] ${copied} font file(s) copied into motion/public/fonts/`);
