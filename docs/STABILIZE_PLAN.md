# Video stabilization (handheld clips)

Design
- A cached pre-process on one media-library file, the same shape as eye contact: a job renders `<stem>_stable.mp4` beside the source, then swaps it in everywhere (library entry, timeline sources, `source_video`); the entry keeps `stabilize: {original_path, output_path, report}` so the UI can badge it and `revert` swaps back. The original is never overwritten.
- Two ffmpeg passes with libvidstab: `vidstabdetect` (CPU, writes an ASCII .trf, cached per source file + shakiness + tripod under `data/cache/stabilize/<sha>.trf`) then `vidstabtransform` + NVENC HEVC encode (constant QP: standard 22 / high 18 / max 14), audio copied, source timestamps passed through, rotation tag and colour tags kept (`-noautorotate`, `-map_metadata 0`).
- Settings: strength -> shakiness 1-10; smoothing_s -> smoothing frames; crop auto (optimal static zoom) / keep / black; optional fixed zoom_pct; tripod (locked camera); quality. `unsharp` only when the zoom exceeds 3 %.
- Shake measurement comes from the .trf: per frame the median local-motion vector (dx, dy), minus its 1 s moving average (so a deliberate pan is not counted), scaled to 1080p; mean > 1.5 px means "shaky". `analyse` runs only the detect pass so the UI can suggest stabilizing.
- Progress: 40 % detect / 60 % transform, from ffmpeg `-progress pipe:1`; cancel kills the ffmpeg process tree.
- API under `/api/stabilize`: available, settings (GET/PUT), start, analyse, preview (10 s before/after), status, cancel, revert.

## Status

| # | Deliverable | Status |
|---|---|---|
| 1 | `backend/stabilize/__init__.py` (Settings, filters, stabilize_video, analyse, trf parsing) | done |
| 2 | `backend/routes/stabilize.py` + main.py registration | done |
| 3 | `backend/tests/test_stabilize.py` | done: 27 passed |
| 4 | Real check on synthetic shaky clip + tripod excerpt | done: shaky 8.18 -> 3.77 px mean (smoothing_s 0.5; default raised to 1.0 afterwards, zoom 2.0 %, 31.7 s for 20 s 1080p); smoothing_s=1.0 -> 0.59 px, zoom 3.7 %; tripod excerpt 0.0 px, shaky=False |
| 5 | This plan doc | done |

## Frontend

| # | Deliverable | User action -> request | Status |
|---|---|---|---|
| F1 | `src/hooks/api.ts` stabilize types + functions | `stabilizeAvailable` GET /available, `get/saveStabilizeSetup` GET/PUT /settings, `startStabilize` POST /projects/{pid}/media/{mid}, `startStabilizeAnalyse` POST .../analyse, `startStabilizePreview` POST .../preview, `getStabilizeStatus` GET /status/{job}, `cancelStabilize` POST /cancel/{job}, `revertStabilize` POST .../revert; `MediaEntry.stabilize` | done (tsc clean) |
| F2 | `src/components/StabilizeDialog.tsx` + `MediaPool.tsx` wiring | see rows F2a-F2h | done (tsc clean) |
| F2a | Open the dialog: tile action `≋` or right-click > "Stabilize…" | GET /api/stabilize/settings, GET /api/stabilize/available (unavailable: reason shown, buttons disabled) | done |
| F2b | "Measure the shake" | POST /api/stabilize/projects/{pid}/media/{mid}/analyse body=settings, then GET /status/{job} every 0.7 s; shows "mean shake N px (shaky / steady)", peak + frames in the tooltip | done |
| F2c | Strength slider, Smoothing slider, Edges (crop auto/keep/black) select, Tripod checkbox, Quality select | local only; sent with preview/run; a changed setting marks the shown preview stale (quality excluded) | done |
| F2d | "Preview 10 s" (start slider) | POST .../preview body=settings+{start_s, duration_s: 10}, poll /status; before/after/side-by-side player streams before_path/after_path via /api/media/stream; closing the dialog mid-preview POSTs /cancel/{job} | done |
| F2e | "Stabilize" (full run) | PUT /api/stabilize/settings (saved setup), POST /api/stabilize/projects/{pid}/media/{mid} body=settings, poll /status every 1.5 s; tile shows stage + % + Cancel (POST /cancel/{job}); on completion reloads library + project sources and notes zoom/residual px | done |
| F2f | Badge "stabilized" on the tile | from `entry.stabilize`; tooltip: zoom %, residual mean/peak px, original path | done |
| F2g | Right-click > "Revert stabilization" (only when stabilized) | POST .../revert, then reload library + project sources | done |
| F2h | Styles | `.media-stab-badge`, `.stab-shake-pill`, `.stab-check`; dialog reuses `.eye-dialog` (stacks under 900 px, max-width 100vw-32px) | done |
| F5 | `npx tsc --noEmit`, `npx vite build` | | done: both clean |
