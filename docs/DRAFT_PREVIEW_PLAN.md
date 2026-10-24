# Draft preview (fast dressed FX proxy)

The FX proxy used to compile the whole programme at source size in one graph, scale it to 540p at the end and encode it with libx264. On a long dressed episode that is the slow single-graph case described in `render/chunked.py`.
Now the proxy is a **draft**: `render/preview.py:draft_timeline` composes it small from the start, at 540p with text scaled to match, using the same helper the real render uses (`render/runner.py:compose_canvas`).
It also swaps generated video B-roll for the still each clip was made from (`still_for_video`: the matching `beat_id` sidecar in `assets/generated`) and drops the noise atmosphere layers (grain, vhs, wind, rain, snow, fog, lightning, light_leak, sunlight).
Cheap hit effects (flash, shutter, shake, glitch, …) and the colour grade stay.
`_run_proxy` renders the draft through `render_timeline_async(prerender=False, chunked=True, workers=2, encoder_profile="draft")`. That is NVENC p1 / cq 32 / GOP 15 / aac 96k, with a libx264 ultrafast crf 30 fallback.
Ken Burns zooms still animate, but zoompan supersampling is forced to 1x through the `render.effects.ZOOM_SS_OVERRIDE` context variable.
A windowed draft (`start_s`/`end_s`) is first cut out of the timeline (`window_timeline`: `chunked.sub_timeline` for the picture plus trimmed audio items, with window edges snapped outward to frames nothing animated crosses), so a late window costs only its own length. It is stored next to the full draft as `<id>_fx_<start>-<end>.mp4`, and status reports `offset_s` (the real, snapped start).
The full-quality render's own `window=` path (output-side `-ss`, single graph) is unchanged, and so are the "final" encoder flags.
The old single-graph proxy is kept as `_run_proxy_legacy` (`legacy=True`) for before/after timing only.

## Status

| # | Deliverable | Status |
|---|---|---|
| 1 | `draft_timeline` + `still_for_video` | done. On 5fbc856f, 13 of the 25 V2+ video items are swapped to stills. The rest are not ComfyUI clips (recording composites, freezes). |
| 2 | `_run_proxy` via `render_timeline_async`, draft encoder profile, 2 workers, 1x zoompan | done. The "final" flags are byte-for-byte unchanged (tested). Drafting happens before windowing, so the noise-only mood layers no longer pin the window edges. |
| 3 | Window support (`start_s`/`end_s`, `proxy_path(id, window)`, `offset_s`, `seconds`, `realtime_x`) | done. Status also reports `full_exists`, `windows`, `progress` and `draft`, and a `.json` sidecar keeps `offset_s` and the timing across restarts. The full render's own `window=` path is unchanged. |
| 4 | Timing on 5fbc856f (legacy vs draft full vs 120 s window from 300 s) | done, targets missed. Programme is 476.4 s. Legacy took 1504.9 s (0.32x realtime). The full draft took 150.7 s (3.16x, 11 chunks, 2 workers): 10x faster, but short of the 5x target by about 37% (it needs about 95 s). The 120 s window from 300 s (snapped to 294.92-420.0 s, 125.1 s) took 63.5 s against the 30 s target (2.1x over). What dominates: each chunk decodes the single V1 recording from t=0 (`[0:v]trim=start=...` on a shared input, because there are fewer than 12 V1 segments), so late chunks pay for every earlier second. The last window chunk took 33.2 s with the shared decode and 8.0 s with an input-side `-ss` (4.1x faster). The fix belongs in `render/compiler.py`, which is not in this plan. The GPU was also about 93% busy with other apps (Blender, LM Studio) during every run. |
| 5 | Visual check of 4 frames | done, one layer missing. Frames at 61.3, 82.5, 116.8 and 128.1 s of the full draft (960x540): captions (Devanagari and the English second line), B-roll stills, cards, aspect bars and grade are present and legible. **Missing: the text-fx layers (knockout "TRAP" at 115.9-118.1 s, behind_head).** They are present in the legacy single-graph proxy. Cause, not draft-specific: `chunked.sub_timeline` shifts items into chunk-local time but leaves `AtmosphereEffect.extra.start_s/end_s` in absolute programme time, so any text-fx outside the first chunk never fires. This hits full chunked renders too. The card's tofu glyph at 61.3 s is in the legacy render as well, so it is not a draft issue. The windowed draft matches the full draft at `offset_s` (PSNR 41 dB, against 11 dB 2 s off). |
| 6 | Tests | done. `tests/test_preview_draft.py`: 11 pass. |
| 7 | This doc | done |

## Frontend

| # | Deliverable | User action -> request | Status |
|---|---|---|---|
| F1 | `src/hooks/api.ts` ProxyStatus draft fields + window params | `getPreviewProxy(pid, window?)` GET /api/timeline/{pid}/preview/proxy?start_s=&end_s=; `buildPreviewProxy(pid, 540, window?)` POST {height, start_s?, end_s?}; ProxyStatus gains draft, window, offset_s, seconds, realtime_x, progress, full_exists, windows | done (tsc clean) |
| F3 | `src/components/PreviewPlayer.tsx` windowed draft + timing | see rows F3a-F3f | done (tsc clean) |
| F3a | FX mode, "Build moving preview" / "Rebuild" on the full draft (unchanged behaviour) | POST /api/timeline/{pid}/preview/proxy {height: 540}, then GET .../preview/proxy every 2.5 s while building | done |
| F3b | FX mode, "Draft around playhead (2 min)" | POST .../preview/proxy {height: 540, start_s: max(0, programme_s - 60), end_s: min(programme_s + 60, programme length)} (0.1 s rounding), then GET .../preview/proxy?start_s=&end_s= (same values) every 2.5 s while building | done |
| F3c | Playing the window draft | video time + `offset_s` = programme time: timeupdate -> store (source time), playhead -> video seek (programme - offset_s), seek-bar click inside the window seeks the draft; outside the window the stage shows the exact still with an "outside draft" pill | done |
| F3d | Build progress + timing | building pill "Drafting m:ss-m:ss... N% · Ns" from `progress`/`elapsed`; ready pill "drafted in Ns (Nx realtime)" from `seconds`/`realtime_x` | done |
| F3e | Stale pill | `stale` from the status GET made with the window's own start_s/end_s (so the window's revision decides); re-fetched on every timeline revision; "Rebuild" re-drafts the same window | done |
| F3f | "Full draft" (shown while a window is followed and `full_exists`) | switches back: GET .../preview/proxy (no window) | done |
| F5 | `npx tsc --noEmit`, `npx vite build` | | done: both clean |
