// Deciding whether a ComfyUI exit looks like a broken Python environment.
//
// Split out of main.cjs so it can be unit-tested without spawning anything
// (no `electron` import here). The ComfyUI 'close' handler calls
// `looksLikeBrokenEnv()` on the process's last ~60 stderr lines and only
// trusts a match within `diedAtStartup()` of the spawn — a broken environment
// dies at startup, not twenty minutes into a render.
//
// Every normal ComfyUI start prints lines that a naive
// ModuleNotFoundError/ImportError scan would wrongly call a broken core
// environment: an optional accelerator backend (triton) or a custom node's
// own optional dependency failing to import is reported and skipped, not
// fatal. Those are dropped first:
//
//   Found comfy_kitchen backend triton: {... 'unavailable_reason':
//     "ImportError: No module named 'triton'" ...}
//   Cannot import C:\...\custom_nodes\X module for custom nodes: No module
//     named 'loguru'
//   (IMPORT FAILED): ...custom_nodes\X
//   Traceback (most recent call last):
//     File "...\custom_nodes\X\__init__.py", line 3, in <module>
//       import loguru
//   ModuleNotFoundError: No module named 'loguru'
//
// The last shape — a multi-line traceback — only names `custom_nodes` on its
// `File "..."` frame, not on its final exception line, so filtering line by
// line would still flag it. Matching lines are instead dropped as whole
// traceback blocks (header, every indented frame, and the exception line that
// ends it).

// A line (or a frame inside a traceback) that marks the whole block as an
// optional/custom-node dependency ComfyUI itself chose to skip — never the
// core environment the launcher's own repair script cares about.
const NOISE_LINE = /custom_nodes|for custom nodes|IMPORT FAILED|comfy_kitchen backend|unavailable_reason/i;

// A genuine broken-environment signature: the base interpreter cannot import
// something ComfyUI's core (not a custom node) needs.
const CRASH_PATTERN = /ModuleNotFoundError|ImportError|DLL load failed|No module named|torch.*not compiled with CUDA/i;

const TRACEBACK_HEADER = /^\s*Traceback \(most recent call last\):\s*$/;

/** Drops noise lines and whole noisy traceback blocks from `text`. */
function stripNoise(text) {
  const lines = String(text || '').split(/\r?\n/);
  const kept = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (TRACEBACK_HEADER.test(line)) {
      // A traceback: its header, every indented frame line, and the one
      // non-indented exception line that ends it — kept or dropped together.
      const block = [line];
      let j = i + 1;
      while (j < lines.length && /^\s+/.test(lines[j])) {
        block.push(lines[j]);
        j += 1;
      }
      if (j < lines.length) {
        block.push(lines[j]);
        j += 1;
      }
      if (!block.some((l) => NOISE_LINE.test(l))) kept.push(...block);
      i = j;
      continue;
    }
    if (!NOISE_LINE.test(line)) kept.push(line);
    i += 1;
  }
  return kept.join('\n');
}

/** True when `stderrText` looks like a broken core Python environment, once
 * the harmless custom-node/optional-backend noise has been dropped. */
function looksLikeBrokenEnv(stderrText) {
  return CRASH_PATTERN.test(stripNoise(stderrText));
}

/** Whether an exit this soon after spawning should even be considered a
 * possible environment failure — a broken environment dies at startup, so an
 * exit long after that is an ordinary crash (or a sleep/wake, or a kill). */
function diedAtStartup(spawnedAt, exitedAt = Date.now(), thresholdMs = 180000) {
  if (!spawnedAt) return true;
  return exitedAt - spawnedAt < thresholdMs;
}

module.exports = {
  NOISE_LINE,
  CRASH_PATTERN,
  stripNoise,
  looksLikeBrokenEnv,
  diedAtStartup,
};
