# Progress bar with chapter ticks

An on-screen programme progress bar: atmosphere effect type `progress_bar` on `timeline.effects`, which the user can add by hand from the effects browser or the presentation pass can add with origin `presentation:progress`.
- **Fill**: a full-width colour box slid in from the left by `overlay` (`x = -W + W*(t+offset)/T`). drawbox geometry is evaluated only once in this ffmpeg 8.1 build, so drawbox draws only the static parts: the track and the ticks.
- **Ticks**: one static 2-3 px drawbox at each chapter's x (the chapter at 0 s gets no tick). **Label**: drawtext with a pinned font (Nirmala UI for Devanagari, otherwise Montserrat, then Arial), one per chapter, `enable=between(t,at,next)`. It fades in over 1.2 s, then stays at 70% opacity.
- **T and offset**: T is the programme length (`build_effect`'s `duration` minus the compiler's 1 s headroom). Chunked renders rebase each chunk to t=0, so `chunked.sub_timeline` sets `extra._offset_s` (the chunk start) and `extra._programme_s` (the full length). Every expression then uses `t+offset` and the full T.
- **Stage**: compiler step 7b, together with the other programme effects. It sits after B-roll, cards and the master grade, and before the letterbox bars (7c) and the burnt-in text and captions (8). Only the bars and the captions draw over it, and the director's placement rule keeps it clear of the captions.
- **Director rule**: `progress_bar` must be true. With style `auto` the bar appears only for explainer genres (tech, science_education, finance, news, documentary, geopolitics, health_fitness; not `general`, the detection fallback); `on` forces it on and `off` turns it off. Chapters come from `chapter_seconds(topics, cold-open offset)`, the same offset the metadata reshift uses.
- **Placement vs captions** (1080p px; the bottom edge is the picture's, above any letterbox bar): no captions, or captions in the top half -> bar at the bottom edge with its label. Captions in the bottom half with room under them -> 6 px at the very bottom edge, label hidden. Captions reaching the bottom edge -> bar above the caption band (margin from the band's top + 8 px), label shown.

| # | Deliverable | Status |
|---|---|---|
| 1 | `progress_bar` effect in render/atmosphere.py (+ chunk offset in chunked.py, assets_dir in compiler 7b) | done: `_progress_bar` + `programme_s` kwarg; chunk offsets in sub_timeline (programme effects + adjustment copies); compiler 7b passes assets_dir |
| 2 | Catalogue entry in `available()` with default extra | done: "Progress bar" entry carries `extra` = PROGRESS_BAR_DEFAULTS |
| 3 | Presentation settings/director/report | done: settings `progress_bar`/`progress_bar_style`, `_apply_progress_bar` (origin presentation:progress) after cold open, `structure.chapter_seconds`, report `progress_bar` |
| 4 | Draft/proxy compatibility (not in any noise list) | done: `progress_bar` is not in preview.NOISE_EFFECTS (grain, vhs, wind, rain, snow, fog, lightning, light_leak, sunlight), so drafts keep it; preview.py untouched |
| 5 | Tests | done: backend/tests/test_progress_bar.py (17 tests); test_effects, test_presentation, test_structure, test_render_chunked and test_compiler all pass |
| 6 | Real check on 5fbc856f (40 s window) | done: 40 s window, chapters 0/15/30 s (T = 476.36 s), placement rule gave margin 232 (captions reach the letterboxed bottom), so the bar sits at y=842. Fill measured about 20 / 61 / 141 px at 5 / 16 / 35 s (expected W*t/T). Ticks at x≈60 and 121. The label changes Intro -> पहली रात (shaped correctly by Nirmala) -> The Third Night. No overlap with the captions (856-1023) or the chapter cards. The Devanagari label is faint at 70% over this footage, which renders as heavy noise with or without the bar (that noise was there before this change) |
| 7 | This plan doc | done |
| 8 | (extra) Chunk rebase of text-fx `extra` start_s/end_s in chunked.sub_timeline | done: programme effects and item/adjustment atmosphere windows are shifted by -a/fps, and windows outside the chunk are disabled. `matte_offset` (a seek into the matte file) is kept. Regression test `test_sub_timeline_rebases_text_fx_windows_into_the_chunk` |

## Frontend

| # | Deliverable | User action -> request | Status |
|---|---|---|---|
| F1 | `src/hooks/api.ts` effect `extra`/`origin` + catalogue `extra` | `AtmosphereEffect.extra/origin`, `ProgressBarExtra`, `ProgressBarChapter`, `effect_catalogue[].extra` (types only; requests via existing addEffect/updateEffect) | done (tsc clean) |
| F4 | `src/components/EffectsPanel.tsx` progress-bar editor; `src/components/AgentPanel.tsx` presentation toggles | see rows F4a-F4e | done (tsc clean) |
| F4a | Effects browser, click "Progress bar" card | POST /api/timeline/{pid}/effects/add {updates: {type: "progress_bar", intensity: 0.5, speed: 1, extra: <catalogue extra = PROGRESS_BAR_DEFAULTS>}, item_id} (any catalogue entry with `extra` now sends it) | done |
| F4b | Applied progress_bar: Position select (bottom/top), Ticks checkbox, Label select (name/none) | POST .../effects/update {index, updates: {extra: {position \| ticks \| label}}, item_id} (backend merges `extra` key by key) | done |
| F4c | Applied progress_bar: Height slider (2-40 px at 1080p) | same, {extra: {height_px}}, debounced 220 ms like the other sliders | done |
| F4d | "Use chapters from the metadata pack" | POST .../effects/update {extra: {chapters: [{at_s, name}]}} copied from the presentation pass's own progress bar on `timeline.effects` (origin `presentation:progress`, built by `structure.chapter_seconds`, the metadata pack's chapters shifted by the cold open). Disabled with a hint when no such bar exists. Open issue: `metadata.json` itself has no route, and the report only carries `progress_bar: {chapters: n}` | done (see open issue) |
| F4e | AI Edit panel > Dressing: "Progress bar with chapter ticks" checkbox + "Progress bar" Auto / Always / Never select | sent with the pass: POST /api/presentation/{pid}/start {settings: {..., progress_bar, progress_bar_style: "auto"\|"on"\|"off"}} | done |
| F5 | `npx tsc --noEmit`, `npx vite build` | | done: both clean |
