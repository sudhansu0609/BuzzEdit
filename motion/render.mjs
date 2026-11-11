#!/usr/bin/env node
// The Node side of the motion-template engine (MOTION_GRAPHICS_PLAN.md M1).
//
//   node render.mjs jobs.json results.json
//   node render.mjs --selftest
//
// Bundles src/ once (cached under .bundle/<hash>/, keyed by a hash of every
// file in src/ plus package-lock.json -- a change to either invalidates the
// cache), ensures a headless browser is present, then renders each job in
// `jobs.json` in turn: `selectComposition` resolves that job's own
// width/height/fps/duration (via Root.tsx's calculateMetadata), `renderMedia`
// renders it. Opaque jobs encode H.264 MP4 (yuv420p, crf 18); transparent jobs
// encode ProRes 4444 MOV (yuva444p10le). One result line is written per job to
// `results.json`: {id, ok, out, error, seconds}.
//
// `--selftest` asks the bundle for every registered composition
// (getCompositions) and renders 1 second of each at 320x180 -- a smoke test
// that stays correct as M2/M3 add compositions, with no list to keep in sync
// here.
//
//   node render.mjs --dump out.json
//
// `--dump` (M11 Motion Gallery, MOTION_GRAPHICS_PLAN.md) writes every
// registered composition's id, its own defaultProps (as set on registry.ts)
// and its own default width/height/fps/durationInFrames to `out.json` --
// `getCompositions` already resolves all of that from the bundle, so the
// Python side never hand-copies the registry.
import { bundle } from "@remotion/bundler";
import { ensureBrowser, getCompositions, renderMedia, selectComposition } from "@remotion/renderer";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const ENTRY = join(ROOT, "src", "index.ts");
const BUNDLE_CACHE_DIR = join(ROOT, ".bundle");

function hashSrc() {
  const hash = createHash("sha1");
  const files = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      const st = statSync(p);
      if (st.isDirectory()) walk(p);
      else files.push(p);
    }
  };
  walk(join(ROOT, "src"));
  files.sort();
  for (const f of files) {
    hash.update(relative(ROOT, f));
    hash.update(readFileSync(f));
  }
  const lock = join(ROOT, "package-lock.json");
  if (existsSync(lock)) hash.update(readFileSync(lock));
  return hash.digest("hex").slice(0, 16);
}

async function getServeUrl() {
  const key = hashSrc();
  const dir = join(BUNDLE_CACHE_DIR, key);
  const marker = join(dir, ".ok");
  if (existsSync(marker)) return dir;
  mkdirSync(BUNDLE_CACHE_DIR, { recursive: true });
  const bundled = await bundle({ entryPoint: ENTRY, outDir: dir, onProgress: () => {} });
  writeFileSync(marker, new Date().toISOString());
  return bundled;
}

function secondsSince(t0) {
  return Math.round((Date.now() - t0) / 100) / 10;
}

async function renderJob(serveUrl, job, glOverride) {
  const t0 = Date.now();
  // M6 parallax (WebGL, @remotion/three) sets `job.gl` ("angle" to start);
  // every other job leaves it unset, so chromiumOptions stays undefined for
  // them exactly as before -- this must never change their own render.
  const gl = glOverride !== undefined ? glOverride : job.gl;
  const chromiumOptions = gl ? { gl } : undefined;
  try {
    const inputProps = {
      ...(job.props || {}),
      width: job.width,
      height: job.height,
      fps: job.fps,
      durationInFrames: job.frames,
    };
    const composition = await selectComposition({ serveUrl, id: job.composition, inputProps, chromiumOptions });
    mkdirSync(dirname(job.out), { recursive: true });
    const concurrency = job.concurrency || 1;
    if (job.transparent) {
      await renderMedia({
        serveUrl,
        composition,
        inputProps,
        codec: "prores",
        proResProfile: "4444",
        pixelFormat: "yuva444p10le",
        // Alpha survives only through PNG intermediate frames; the default
        // (jpeg) has no alpha channel, and "yuva444p10le but not PNG" fails.
        imageFormat: "png",
        outputLocation: job.out,
        overwrite: true,
        concurrency,
        chromiumOptions,
      });
    } else {
      await renderMedia({
        serveUrl,
        composition,
        inputProps,
        codec: "h264",
        crf: 18,
        pixelFormat: "yuv420p",
        outputLocation: job.out,
        overwrite: true,
        concurrency,
        chromiumOptions,
      });
    }
    return { id: job.id, ok: true, out: job.out, error: null, seconds: secondsSince(t0) };
  } catch (e) {
    // A WebGL job's first attempt is "angle" (hardware-ish ANGLE translation);
    // headless Chromium without a GPU context for it fails to create a WebGL
    // context, not to render -- one retry with "swangle" (SwiftShader,
    // software) covers that, same job, same output path.
    if (job.gl && gl === "angle" && glOverride === undefined) {
      return renderJob(serveUrl, job, "swangle");
    }
    return { id: job.id, ok: false, out: job.out, error: String((e && e.message) || e), seconds: secondsSince(t0) };
  }
}

async function render(jobsPath, resultsPath) {
  const jobs = JSON.parse(readFileSync(jobsPath, "utf-8"));
  await ensureBrowser();
  const serveUrl = await getServeUrl();
  const results = [];
  for (const job of jobs) {
    results.push(await renderJob(serveUrl, job));
  }
  writeFileSync(resultsPath, JSON.stringify(results, null, 2));
}

async function selftest() {
  await ensureBrowser();
  const serveUrl = await getServeUrl();
  const comps = await getCompositions({ serveUrl });
  const tmpDir = join(ROOT, ".selftest");
  rmSync(tmpDir, { recursive: true, force: true });
  mkdirSync(tmpDir, { recursive: true });
  let ok = comps.length > 0;
  if (!ok) console.log("[selftest] no compositions registered");
  for (const comp of comps) {
    const fps = comp.fps || 30;
    const job = {
      id: comp.id,
      composition: comp.id,
      props: {},
      width: 320,
      height: 180,
      fps,
      frames: Math.max(1, Math.round(fps)),
      out: join(tmpDir, `${comp.id}.mp4`),
      transparent: false,
      concurrency: 1,
      // Only the WebGL composition (M6) needs a gl renderer at all; every
      // other composition's selftest job is unaffected (gl stays undefined).
      gl: comp.id === "Parallax" ? "angle" : undefined,
    };
    const result = await renderJob(serveUrl, job);
    console.log(`[selftest] ${comp.id}: ${result.ok ? "ok" : `FAIL ${result.error}`} (${result.seconds}s)`);
    if (!result.ok) ok = false;
  }
  console.log(ok ? "SELFTEST PASSED" : "SELFTEST FAILED");
  process.exit(ok ? 0 : 1);
}

async function dumpCompositions(outPath) {
  await ensureBrowser();
  const serveUrl = await getServeUrl();
  const comps = await getCompositions({ serveUrl });
  const out = comps.map((c) => ({
    id: c.id,
    defaultProps: c.defaultProps || {},
    width: c.width,
    height: c.height,
    fps: c.fps,
    durationInFrames: c.durationInFrames,
  }));
  writeFileSync(outPath, JSON.stringify(out, null, 2));
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes("--selftest")) {
    await selftest();
    return;
  }
  const dumpIdx = args.indexOf("--dump");
  if (dumpIdx !== -1) {
    const outPath = args[dumpIdx + 1];
    if (!outPath) {
      console.error("usage: node render.mjs --dump out.json");
      process.exit(2);
      return;
    }
    await dumpCompositions(outPath);
    return;
  }
  const [jobsPath, resultsPath] = args;
  if (!jobsPath || !resultsPath) {
    console.error("usage: node render.mjs jobs.json results.json | node render.mjs --selftest");
    process.exit(2);
    return;
  }
  await render(jobsPath, resultsPath);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
