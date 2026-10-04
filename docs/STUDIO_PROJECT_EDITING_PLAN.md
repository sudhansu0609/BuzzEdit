# Studio projects: fully editable in BuzzEdit

Goal: a video produced by BuzzcafStudio (import_path → presentation job) opens in
BuzzEdit as a normal project where everything the backend did is visible and
editable as if done by hand.

Already true (no work): Studio creates the project via `/api/projects/import_path`,
the presentation pass saves `timeline` before rendering (`director.py:777`), and
the start screen lists it. Preview "FX" mode shows the dressed programme.

## Gaps and fixes

| # | Gap | Fix | Owner |
|---|-----|-----|-------|
| B1 | V1/A1 are `origin="auto"` → no drag/trim/delete, so no extra cuts | Persistent `Timeline.manual_cuts` (source-frame spans) applied after every `rebuild_primary_tracks`; `cut_program_range` ripple-cuts a programme range on V1/A1 and ripples overlays; re-enabling a word un-cuts its span | backend |
| B2 | No find/replace of transcript words | `POST /api/transcription/{id}/replace_words` updates timeline words, transcript segments and caption text | backend |
| B3 | Knockout overlay (video-filled text) not editable | `update_effect` merges `extra`; knockout honours `extra.plate_color`, `extra.font_size` | backend |
| B4 | No audio treatment API | `GET /api/timeline/{id}/audio_master` (+ preset lists), `POST .../audio_master` | backend |
| B5 | Generated media (B-roll, music, cards) missing from library | `list_media` syncs `timeline.sources` into `media_pool` (persisted, `generated: true`) | backend |
| F1 | Timeline doesn't follow playhead | Follow mode off/left/center/right (persisted in localStorage), page/continuous scroll | frontend-timeline |
| F2 | Can't cut V1/A1 | Trim handles + Delete on auto clips call program/cut; Mark In/Out (I/O) + "Cut range" button; "Restore cuts" | frontend-timeline |
| F3 | Transcript search/replace | Search box with match highlighting + next/prev, Replace / Replace all | frontend-timeline |
| F4 | Dressing not visible on lanes | Clip chips for zoom / fade / grade / effects; show CAP lane | frontend-timeline |
| F5 | Audio settings | Inspector "Audio" section on A-track clips: clip gain/fades/duck/loop/mute + programme voice chain (clean, denoise, de-ess, compress, EQ preset incl. rap_vocal, saturation, reverb, normalise LUFS) | frontend-inspector |
| F6 | Knockout/text-fx editing | Per-effect fields for text, colour, plate colour, size | frontend-inspector |
| F7 | Library tabs | Video / Images / Music tabs; "generated" badge | frontend-inspector |

## Status

| Item | Status |
|------|--------|
| B1 | done |
| B2 | done |
| B3 | done |
| B4 | done |
| B5 | done |
| F1 | done |
| F2 | done |
| F3 | done |
| F4 | done |
| F5 | done |
| F6 | done |
| F7 | done |
| Review | done (fixes: native-script replace, replace-all scope, audio debounce batching, V1 gain routing, follow on any playhead jump, AudioMaster type) |
