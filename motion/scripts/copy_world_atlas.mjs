// Copies the world outline MotionMap draws its globe from out of the
// world-atlas npm package into public/world/, where Remotion's staticFile()
// can serve it to the headless browser (MOTION_GRAPHICS_PLAN.md M2: "the
// world outline from the world-atlas 110m file in public/"). Run as
// package.json's "postinstall" alongside copy_fonts.mjs, so a plain
// `npm install` always leaves public/world/ in place -- no manual download.
import { mkdirSync, copyFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const outDir = join(root, "public", "world");
mkdirSync(outDir, { recursive: true });

const source = join(root, "node_modules", "world-atlas", "land-110m.json");
if (!existsSync(source)) {
  console.warn("[copy_world_atlas] world-atlas/land-110m.json not found -- skipping (run npm install first)");
} else {
  copyFileSync(source, join(outDir, "land-110m.json"));
  console.log("[copy_world_atlas] land-110m.json copied into motion/public/world/");
}
