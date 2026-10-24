# Multicam angles + external audio sync (backend)

## Design

- `backend/sync/audio_sync.py` measures how two recordings of one event line up. Convention:
  `other_time = ref_time + offset_s`. It works coarse to fine. First an onset envelope
  (|x| -> 20 Hz low-pass -> 100 Hz) is FFT cross-correlated over +-max_offset_s. Then the lag
  is refined on 8 kHz PCM within +-60 ms. The confidence is the main peak divided by the
  second-highest distinct peak; below 1.5 the match is unreliable. Drift is measured from the
  lag on a window near the start and one near the end of the overlap.
- External recorder audio does not get a new mechanism. It reuses the audio suite's version
  stack: an aligned 48 kHz WAV, exactly as long as the primary source and drift-corrected with
  `atempo`, is pushed on `audio_history` and becomes `audio_override`. `POST
  /api/audio/{pid}/undo` reverts it. Transcription prefers the override when one is set.
- A camera angle is a `SourceFile` with `synced_to` / `sync_offset_s` / `sync_confidence`.
  The edit decision is `Timeline.angle_spans`, stored in PRIMARY-SOURCE frames like
  `manual_cuts`, so it survives every word-toggle rebuild. `rebuild_primary_tracks` runs
  `apply_angle_spans` last. It splits V1 at span edges and repoints the V1 pieces inside a
  span to the angle: `angle_frame = primary_frame + round(sync_offset_s * tl_fps)`. Every
  source's frames are counted at the TIMELINE fps, and the compiler's per-clip `fps=`
  resamples the picture. A1 is never touched, so the sound stays continuous and link_ids stay.
- Code that maps V1 back to the transcript uses `ops.v1_primary_span` (it converts angle
  pieces back to primary frames): `cut_program_range`, the captions/presentation segment map,
  kept_words and `jump_cut_times`. A join between two angles that is continuous in primary
  time is an angle switch, not a jump cut.
- The compiler picks the canvas from the first V1 item. When that item is an angle, it now
  uses the angle's `synced_to` source (else the timeline size), so a programme that opens on
  cam B still renders at cam A's size.
- Auto multicam (`ops.auto_angles`): `hide_cuts` puts an angle change on every jump cut
  (A/B/... rotation), with a minimum shot length and no switch under B-roll. `rhythm` switches
  at sentence boundaries about every `cadence_s`. `off` clears.

## API (all under `/api/timeline/{project_id}`)

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/sync/audio` | `{media_id, retranscribe?: false}` | `{offset_s, confidence, drift_ppm, method, version_path, source_id, retranscribe_hint}`. 422 if confidence < 1.5 |
| POST | `/angles/sync` | `{media_id}` | `{source_id, offset_s, confidence, drift_ppm, method, angles, timeline}`. 422 if unreliable |
| GET | `/angles` | - | `{primary_source_id, angles:[{source_id, media_id, name, offset_s, confidence, width, height, fps}], spans:[{start, end, source_id, source_start, source_end}]}` (start/end are programme frames, source_start/source_end are the stored primary frames) |
| POST | `/angles/set` | `{start_frame, end_frame, source_id or null}` (programme frames) | `{status, angle_spans, timeline}` |
| DELETE | `/angles` | - | `{status, angle_spans: [], timeline}` |
| POST | `/angles/auto` | `{mode: "hide_cuts" or "rhythm" or "off", min_shot_s?: 2.5, cadence_s?: 8.0}` | `{status, angle_spans, summary:{switches, cuts_hidden, cuts_total, ...}, timeline}` |

Undo for the angle routes works like every other timeline edit: the client keeps the snapshot
and puts it back with `POST /replace`. Undo for `/sync/audio` is `POST /api/audio/{pid}/undo`.

## Status

| # | Deliverable | Status |
|---|---|---|
| 1 | `backend/sync/audio_sync.py` (decode_mono, find_offset, drift) | done. The coarse onset-envelope xcorr is followed by a per-window local envelope lag (+-1 s, so a drifting long take stays inside the +-60 ms PCM search) and a PCM refine on up to 24 windows of <=10 s, with a robust line fit for drift. Synthetic: offset 3.37013 (conf 5.97), reverse -3.37013, unrelated conf 1.10, atempo 1.0005 -534 ppm on a 90 s take. On 20 s, atempo's WSOLA steps alone wander 378..584 ppm, so the drift test uses 90 s. |
| 2 | External audio as programme sound (`/sync/audio`, transcription prefers override) | done. The route reuses `processes._source_lock/_next_version/_save_source`, so the version lands in `audio/versions/<n>_external_audio_sync.wav` and is reverted by `/api/audio/{pid}/undo` (tested). `routes/transcription.py::_transcription_audio` feeds the override to both `extract_audio` call sites. `retranscribe: true` only returns a hint; nothing is started. |
| 3 | Camera angle schema + `/angles/sync` + `GET /angles` | done. `SourceFile.synced_to/sync_offset_s/sync_confidence` and `Timeline.angle_spans` all have defaults, so old JSON loads. Source registration is factored out of add_media as `_register_source`. The pool entry gets `angle: {synced_to, offset_s, confidence, source_id, drift_ppm}`. |
| 4 | `apply_angle_spans` in rebuild, `set_angle`, `/angles/set`, `DELETE /angles` | done. apply runs last in `rebuild_primary_tracks` and again at the end of `trim_source_regions`, which now reads V1 in primary frames, so manual cuts and cut-verify trims stay angle-safe. It acts only on origin `auto` V1. `set_angle` maps programme to primary frames (removed gaps are bridged into one span) and re-lays the angle layer (unapply + apply) instead of a full word rebuild, so a drag-built V1 is never wiped. Mapping fixed for angle pieces: captions `_program_segments`, `authoring._primary_source_id`, kept_words, `cut_program_range`, `jump_cut_times`. Compiler: the canvas comes from the angle's `synced_to` source, and the default transition is skipped at a pure angle switch. preview/frame checked inside a span: `compose_frame` returns cam B's frame on a 640x360 (16:9) canvas; preview/proxy goes through the same compiler. |
| 5 | `auto_angles` + `/angles/auto` + `angle_switches` in jump_cut_coverage | done. Rotation is primary, angle1, angle2, ... A cut under B-roll is counted in `cuts_by_broll` and gets no switch. `angle_switches` is added only when `angle_spans` is non-empty, so the existing dict-equality test still holds. |
| 6 | Tests (test_audio_sync.py, test_angles.py) + existing suites | done. test_audio_sync 7 passed, test_angles 13 passed. Existing timeline/editing/clip_ops/render_chunked/cmdline/prerender/kept_words/presentation/audio_process suites: 220 passed together with the new ones. cut_verify/program_time/transcribe_checkpoint/verify_presentation/clip_variety: 58 passed. |
| 7 | End-to-end on real footage (scratchpad) | done. Media (scratchpad `multicam/`): `ffmpeg -ss 1380 -t 60 -i 674aefab_source.mp4 -c copy camA.mp4`; `ffmpeg -ss 1382.3 -t 57.7 -i 674aefab_source.mp4 -vf scale=1280:720,hflip -af lowpass=f=3000,volume=-8dB -c:v h264_nvenc -cq 23 -c:a aac -b:a 160k camB.mp4`; `ffmpeg -ss 1378.25 -t 63 -i 674aefab_source.mp4 -vn -af highpass=f=80 -ar 48000 -c:a pcm_s16le recorder.wav`. Run: `cd backend && ../.venv/Scripts/python.exe <scratchpad>/multicam/e2e_multicam.py <scratchpad>/multicam`. Measured (convention other = ref + offset): cam B **-2.29992 s** (conf 15.86, drift 0.0 ppm); recorder **+1.74918 s** (conf 18.63, -9.5 ppm). The brief's expected 2.300 / -1.750 have the opposite sign: cam B starts later, so under the stated convention its offset is negative. Both find_offset calls took 0.44 s together. The aligned recorder WAV against cam A measures -0.00001 s. Timeline: keep_full_source cam A, cut_program_range to 20 s, set_angle programme [3 s, 8 s) and [12 s, 17 s) to cam B, giving angle pieces camB 18..143 and 243..368 (= primary - 57 frames). Render (render_timeline_async, 4.2 s): `multicam_e2e.mp4` is h264 1920x1080 (cam A's canvas) + aac, 20.000 s, and decodes with no errors. Frame check (192x108 gray correlation vs cam A at the same t): outside the spans corr 1.0 and mirrored 0.09-0.10; inside the spans (5.5 s, 14.5 s) corr 0.10 and mirrored 1.0, so the angle is frame-accurate and mirrored. Programme audio vs cam A: -0.00018 s, conf 10.6. Long-file check: the 42-min source vs recorder.wav gives -1378.2508 s (expected -1378.25) in 4.0 s. |
| 8 | This plan doc | done (design + API written; Status kept current) |

## Frontend

All calls live in `src/hooks/api.ts` (multicam section). Responses that carry `timeline` are
pushed into the project store, so they render and are undoable through `hooks/history.ts`
(snapshot + `POST /replace`). `useAnglesStore` keeps the last `GET /angles` for the timeline
chips and the Inspector section; every angle call refreshes it.

| User action | Where | Request |
|---|---|---|
| Right-click a video tile > "Sync as camera angle" | Media library | `POST /api/timeline/{pid}/angles/sync {media_id}` |
| Right-click an audio tile > "Use as programme audio (sync)" > Replace audio | Media library dialog | `POST /api/timeline/{pid}/sync/audio {media_id}`, then `GET /api/timeline/{pid}` |
| "Undo" in the programme-audio result | Media library dialog | `POST /api/audio/{pid}/undo {source_id}`, then `GET /api/timeline/{pid}` |
| Right-click a V1 clip / selection / In-Out range > Switch to > Primary or an angle | Timeline | `POST /api/timeline/{pid}/angles/set {start_frame, end_frame, source_id or null}` |
| Right-click > "Clear all angles" (confirm) | Timeline | `DELETE /api/timeline/{pid}/angles` |
| Open "Camera angles" section | Inspector | `GET /api/timeline/{pid}/angles` |
| Auto multicam > Run | Inspector | `POST /api/timeline/{pid}/angles/auto {mode, min_shot_s, cadence_s}` |
| "Clear angles" (confirm) | Inspector | `DELETE /api/timeline/{pid}/angles` |

### Frontend status

| # | Deliverable | Status |
|---|---|---|
| F1 | api.ts types + calls (AngleInfo, AngleSpan, AnglesState, SyncResult; syncAudio, syncAngle, getAngles, setAngle, clearAngles, autoAngles, undoAudioVersion) | done (tsc clean) |
| F2 | Media library actions + angle badge | done. Video tile menu "Sync as camera angle" (or "Re-sync"), result or 422 detail in the library status line; audio tile menu "Use as programme audio (sync)…" opens `SyncAudioDialog` (confirm, result with offset/confidence/drift, Undo). Badge `.media-angle-badge` "angle · +2.30 s", confidence + drift in its tooltip. |
| F3 | Timeline angle tint, chip, Switch to / Clear all angles | done. V1 pieces on an angle get `.clip-angle-tint angle-N` (3 hues in `--angle-1..3`) and a `.clip-chip.angle` with the angle name (tooltip: number, name, offset). Clip menu (V1 clip, multi-selection, or marked In/Out): "Switch to (the selection / the In/Out range)" submenu Primary + angles, and "Clear all angles" (confirm). Both run through `runOp`, so they queue and undo like other edits. |
| F4 | Inspector "Camera angles" section | done. `CameraAnglesSection.tsx`, mounted after "Cinematic framing" (collapsed; GET /angles on open, and the shared store refreshes after every angle call). Empty state points to the library action. Lists angles (swatch, name, offset, confidence); Auto multicam radio Hide jump cuts / Rhythm every N s / Off, Min shot s, Run shows "12 switches, 9 of 14 cuts hidden"; Clear angles (confirm). |
| F5 | Preview player angle indicator (cut mode) | done. In Cut mode, when `currentTime` (primary seconds) falls in a stored `angle_spans` entry, a `.preview-angle-pill` "angle: <file>" shows top-right; tooltip says to use FX to see it. No video switching. |
| F6 | Responsive check (ellipsis, dialog width) | done. Library badge: absolute, `max-width: calc(100% - 26px)` + ellipsis. Clip angle chip: `max-width: 96px` + ellipsis inside the absolute chip row (the clip keeps `overflow: hidden`). Dialog: `min(420px, 100vw - 32px)`, title ellipsis, result text wraps. Inspector rows: `min-width: 0` + ellipsis. Preview pill: kept clear of the mode buttons. Not checked in a running window (Electron not run). |
| F7 | tsc + vite build | done. `npx tsc --noEmit` clean; `npx vite build` clean (only the existing >500 kB chunk-size notice). |

### Frontend open issues

- The UI does not track a running render, so the sync actions are not disabled during one.
  `/sync/audio` refuses on the server while a render runs, and the dialog shows that detail.
- `/sync/audio` and its undo re-read the timeline (`audio_override` lives on the source), and
  timeline history records that as a step. A timeline Undo past it `/replace`s a snapshot
  without the override, which silences the external audio without popping `audio_history`.
  The Audio panel's processes have the same interaction.
- A timeline Undo past `/angles/sync` drops the angle source from the timeline, but the
  library entry keeps its `angle` badge until the angle is synced again.
- "Switch to" uses timeline frames as programme frames, the same convention as "Cut range".

## Open issues

- Offset sign: the brief's E2E expectations (cam B +2.300, recorder -1.750) contradict its own
  convention (`other = ref + offset`) and the frame formula. The code follows the convention:
  cam B -2.300, recorder +1.750.
- `/transcribe` and `/recut` build a brand-new Timeline. That drops `audio_override`/history (as
  it already did) and now also the angle sources and `angle_spans`. They are not carried over.
- An angle's picture is placed with one offset. Clock drift on video is reported (`warning` past
  20 ppm) but not corrected.
- V1 readers not yet angle-aware: presentation/facezoom.py, presentation/structure.py
  (`ProgramSegment.source_start_frame` from program.py), presentation/shorts.py,
  asr/cut_verify.py. The PiP in composite.py/graphics.py copies the V1 segment, so it shows
  the angle.
- The frontend live engine must play V1 items whose source is an angle (that is the frontend brief).
- On overlaps under about 60 s the drift estimate has +-10..30 ppm of noise, which can trigger
  a sub-millisecond atempo correction. It is harmless.
