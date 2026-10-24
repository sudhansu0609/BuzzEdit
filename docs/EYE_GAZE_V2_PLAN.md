# Eye gaze v2: land the eyes ON the lens, and keep them alive

Backend only (`backend/eyecontact/`, `backend/routes/eyecontact.py`). The dialog is a
separate brief; every new API field is optional.

## Diagnosis (674aefab recording, 2026-10-24)

- 1080p, 25 fps, 42 min, **mirrored** video, prompter on the speaker's left, 16 cm beside
  the lens, camera ~110-140 cm away: physically ~7-8 deg between reading and the lens.
- Auto mode measured only 2.4 deg and moved the irises a median 0.7 px (eye 33 px wide).
  The "after" frames were indistinguishable from "before".
- The iris landmarks are accurate (pixel centroid agrees to 0.002 eye widths). The
  measurement was of the wrong thing: it looked at the eyes alone, but the head carries
  part of the gaze (reading yaw -2.8 deg, pitch +7.4 deg), and the "camera looks" it picked
  were frames where the eyes drift to the middle of the opening while the head stays turned
  (yaw -4.6 deg). Combined gaze (eyes + IRIS_SWING * sin(yaw)) is bimodal with the reading
  and lens modes ~0.05 eye widths (~8 deg) apart -- matching the geometry.
- One trusted "camera look" (t=131 s) was the speaker leaning into the camera with the face
  half out of frame. Such frames must never calibrate anything.
- In auto mode `angle_deg` and `aim_deg` were ignored and `pitch_deg` only partly applied:
  the creator's 17 deg / -11.5 deg / -6 deg / 0.95 settings changed nothing.
- "The subject's left is the same whether the video is mirrored" was false: landmarks 33/263
  are assigned by image position, so a mirrored video flips what `prompter_side` means.
- `steadiness` damped everything, including the fast natural movement: at 0.95 a stare.

## Design

1. **Head-coordinate gaze.** Per frame: head yaw (face normal along the eye axis) and pitch.
   Combined gaze = eye gaze + IRIS_SWING * sin(yaw) sideways, eye_t + IRIS_SWING * sin(pitch)
   vertically. Camera looks are holds (>= 0.4 s of the 0.4 s-smoothed gaze) in the mode of
   combined gaze nearest zero (distinct from the reading mode), gated: eye width within
   +-20 % of its rolling median, all face landmarks inside the frame, head not dropped more
   than HEAD_PITCH_AWAY_DEG below its reading pitch. The iris target per frame is the
   combined lens minus that frame's head part, so the eyes compensate a turned or bowed
   head. per_eye_limits keeps its balance; each eye's limit moves with the head and the
   user's nudges. Geometry fallback stays when no camera looks are found.
2. **User knobs in auto mode.** `aim_deg`, `pitch_deg` add on top of the measured target;
   `angle_deg > 0` overrides the measured sideways magnitude (sign measured). New
   `lens_marks` (seconds): the lens is measured from +-0.3 s around each mark; stored per
   media entry (`eye_contact_lens_marks`). Reports `lens_source` (marks / measured /
   geometry) and up to 3 `camera_look_candidates` (seconds). Manual mode takes the push
   direction from the measured reading offset and reports `side_check` when the configured
   side disagrees (mirrored video).
3. **Lively, not locked.** The reading residual is split at ~1.2 Hz: the slow line sweep is
   removed by `steadiness`; the fast part (micro-saccades, fixational jitter) always passes
   at full amplitude, inside AUTO_MAX_SHIFT and the per-eye limits. Blinks bridged, away
   gate kept.
4. **Contact sheet.** `eyecontact.contact_sheet(src, dst, plan, out)`: 5x eye crops, before
   over after, 6 reading + 3 camera-look frames, labelled; `<output>_eyes.jpg` for every run
   and preview, returned as `contact_sheet`. CPU (cv2) only.

## Status

| # | Deliverable | Status |
|---|---|---|
| 1 | Head-coordinate gaze model, gated camera-look holds, head compensation | done (plan.py). 674aefab plan-only: measured_shift_deg 7.85, median shift 1.59 px (R 1.65, L 1.50) at 33.2 px eyes, lens_source measured, 537 camera-look frames, t=131 s lean-in excluded, side_check "video looks mirrored". Vertical: head lifts 7.2 deg for the lens, eyes raised 8.7 deg (1.77 px) |
| 2 | aim/pitch/angle in auto mode, lens_marks, lens_source, candidates, routes | done. 674aefab: aim +3 -> 2.20 px, pitch +3 -> vertical -5.7 deg, angle 10 -> 2.02 px, marks [104.5, 138.6, 1851.4] -> lens_source marks, 6.9 deg. Routes: `lens_marks` on EyeContactSetup, stored on the media entry as `eye_contact_lens_marks`; analyse/preview/run report `lens_source`, `camera_look_candidates`, `side_check`; preview uses the whole recording's lens + candidates (`eyecontact.lens_info`) |
| 3 | Sweep removal (slow < 1.2 Hz) instead of damping everything | done (plan.sweep: snap-back steps > SNAP_MIN kept sharp + low-pass below SWEEP_HZ; only that is removed by steadiness; fast part passes, per-eye limits applied with the fast part taken out) |
| 4 | Contact sheet | done (eyecontact/sheet.py, `contact_sheet` in every run/preview report; `<output>_eyes.jpg`; preview pruning includes the sheets). A clip measured on the whole recording uses its left-alone frames as its camera looks; a pure-reading clip has no CAM row |
| 5 | Verification on 674aefab (plan numbers, 10 s preview, 60 s excerpt) | done. (a) plan-only: 7.85 deg, 1.59 px, vertical raise 8.7 deg (1.77 px). (b) 10 s preview at 1391 s: 250 frames in 3.8 s, 100 % corrected, 1.71 px (R 1.65, L 1.70), eyes raised 5.3 deg; sheet C:\Users\singh\AppData\Local\Temp\claude\B--youtubeProjects-Buzzcaf-Media-BuzzEdit\30e7f539-3973-43d8-a67b-68a3345bad3a\scratchpad\eye\preview_1391_after_eyes.jpg. (c) 60 s excerpt at 1380 s: 1500 frames in 19.6 s, 1.64 px, raised 6.4 deg; sheet C:\Users\singh\AppData\Local\Temp\claude\B--youtubeProjects-Buzzcaf-Media-BuzzEdit\30e7f539-3973-43d8-a67b-68a3345bad3a\scratchpad\eye\excerpt_1380_after_eyes.jpg. Extra: preview at 99.5 s (camera looks) is left 100 % alone; sheet C:\Users\singh\AppData\Local\Temp\claude\B--youtubeProjects-Buzzcaf-Media-BuzzEdit\30e7f539-3973-43d8-a67b-68a3345bad3a\scratchpad\eye\preview_0995_after_eyes.jpg. Seen: in the reading crops the image-left iris sits clearly left of the opening's middle before and moves ~1.5-2 px right toward it after -- where the camera-look irises sit -- with the image-right iris moved alike; whites look natural, no smear or fold; consecutive reading frames still differ (frame-to-frame 0.07 px median, high-passed movement 0.09 px std vs 0.15 in, the difference being the removed snap-backs; slow sweep 0.42 -> 0.13 px std at steadiness 0.7). The shift is real but modest at 33 px eyes. Fixed during verification: a camera-look-heavy clip invented a phantom reading position and lowered real camera looks 1.5 px; the reading split is now anchored on the measured reading position |
| 6 | Tests | done: 35 passed (29 existing + 6 new: head-yaw compensation, outlier width / out-of-frame looks, lens_marks override, aim/pitch/angle in auto, fast movement kept while the sweep goes, contact sheet). One existing assertion changed: prompter_side="right" on a left-reading take now pushes the measured way and reports side_check "video looks mirrored" |
| 7 | This doc + module docstrings | done (plan.py header: head-coordinate model, camera-look gates, mirroring note; Settings: steadiness, aim/pitch/angle, lens_marks, head_depth unused) |

## Open issues

- The creator's saved setup (`backend/data/app_settings.json` -> eye_contact: angle 17, pitch -6,
  steadiness 0.95) now takes effect in auto mode: angle 17 sizes the sideways push at ~3.4 px
  (vs the measured 1.6 px) and pitch -6 on top of the measured 8.7 deg raise hits the 10 deg cap.
  Worth resetting angle to 0 / pitch to 0 before the next full run.
- The vertical correction on 674aefab is a raise of 8.7 deg (1.8 px) because the head lifts ~7 deg
  for camera looks; it is inside MAX_PITCH_DEG and looked clean in the sheets, but it is the
  bigger of the two moves.
- Auto mode without camera looks still falls back to the symmetric guess (combined gaze 0), as
  before, not to the prompter geometry; it is reported as lens_source "geometry".
- A clip's own `camera_look_candidates` (CLI, direct calls) are in clip seconds; the preview
  route replaces them with the whole recording's.
- The frontend dialog (separate brief) still has to show lens_source / side_check / candidates
  / contact_sheet and send lens_marks.

## Frontend

The eye-contact dialog (`src/components/EyeContactDialog.tsx`, extracted from `MediaPool.tsx`),
`src/hooks/api.ts` types, `src/styles.css`. Backend unchanged.

### Status

| # | Deliverable | Status |
|---|---|---|
| F1 | Types + api.ts (lens_marks, result fields, `mediaFileUrl`) | done. `EyeContactSetup` gains `mode`, `auto_pitch?`, `lens_marks?`; new `EyeContactMeasures` (lens_source, camera_look_candidates, side_check, measured_shift_deg, median_shift_px_per_eye, eyes_raised_deg, ...) shared by `EyeContactReport` (+ contact_sheet) and new `EyeContactAnalysis`; `MediaEntry.eye_contact_lens_marks?`; `mediaFileUrl` = /api/media/stream (download only transcodes audio); `saveEyeContactSetup` strips lens_marks before the PUT |
| F2 | Dialog layout for auto mode (lens block, camera looks + marks, mode toggle, geometry fallback, override angle, sweep label, mirrored warning) | done. Dialog + player moved to `src/components/EyeContactDialog.tsx` (MediaPool.tsx 864 -> 562 lines). Mode toggle (auto/manual). Auto: Lens pill (marks / measured / geometry in plain words) + `measured_shift_deg`° + per-eye px from the latest preview, else the last run's report ("(last run)"); camera-look candidates as mm:ss buttons (move the preview start slider) + "+ mark"; "Mark this moment" (player position when the shown preview starts at the slider, else the slider); mark chips with remove; "Override the measured angle (0 = use measured)"; "add on top of the measurement" line; side + distances under a collapsed "Prompter geometry (fallback)" details (distances do not clear the override there). Manual: as before + mirrored warning when `side_check` matches /mirror/. Steadiness -> "Calm the reading sweep" + help text |
| F3 | Contact sheet strip + lightbox; tile badge tooltip after a run | done. Under the before/after player: "Eyes, before over after" full-width `<img>` of `contact_sheet` via `mediaFileUrl` (new path per preview, so it refreshes); click opens a fixed lightbox at natural size (scrolls; click closes). Tile badge title = "Eyes re-aimed at the lens. Lens: <source>, <measured_shift_deg>° sideways; eyes moved R x px, L y px. Original: ..." (`eyeLensSummary`, marks count from report.settings.lens_marks); the badge lost `pointer-events: none` so the tooltip can show at all |
| F4 | Polling / cancel / run / revert / saved setup round trip; lens_marks in setupKey, never saved | done. Polling, unmount-cancel, run, revert unchanged. Marks seeded from `entry.eye_contact_lens_marks`; `lens_marks` is sent only once seeded or edited (omitting keeps the stored marks, so a stale entry can never clear them with []); `setupKey` hashes the request incl. lens_marks (a mark makes the preview stale). lens_marks stripped twice before the PUT (handleEyeContact + `saveEyeContactSetup`); the backend pops it too. Closing the dialog refreshes the library so the entry picks up marks the backend stored |
| F5 | Responsive fit 640-1366 px | done (CSS reasoning, app not run). 1366: dialog 1100 = 308 settings + 16 gap + 744 preview + 32 padding. 901-1365: dialog = 100vw-32, preview >= 513 px. <= 900 px (new media query): settings stacked over the preview at full width (>= 576 px at the 640 floor), preview `flex: none`. Settings column: chips/mode toggle `flex-wrap`, buttons in the toggle `white-space: normal`, pill wraps, numbers `overflow-wrap: anywhere`; `.eye-compare-controls` now wraps. Only the lightbox (natural size by design) can scroll sideways, inside itself |
| F6 | tsc + vite build + action walkthrough | done. `npx tsc --noEmit` clean; `npx vite build` clean (only the existing >500 kB chunk-size notice). Walkthrough below |
| F7 | This section | done |

### Action walkthrough (what each user action sends)

`setup` = mode, prompter_side, angle_deg, prompter_cm, camera_cm, steadiness, aim_deg, pitch_deg,
auto_pitch, quality (as returned by GET settings, edited in the dialog). `lens_marks` = the dialog's
marks (sorted, 0.1 s), included only once seeded from the entry or edited; omitted = backend keeps
the stored ones.

| User action | Request(s) |
|---|---|
| Open the dialog (eye button on a video tile) | GET /api/eyecontact/settings. Marks seeded from the listed entry's `eye_contact_lens_marks` (no request) |
| Mode toggle, override angle, fine aim, up/down, calm sweep, geometry fallback inputs, quality | none (local); sent with the next preview/run as mode / angle_deg / aim_deg / pitch_deg / steadiness / prompter_side, prompter_cm, camera_cm / quality. Anything but quality marks the shown preview stale |
| Move the preview start slider | none (local `start_s`) |
| Preview 10 s / Preview again | (if a preview is running) POST /api/eyecontact/cancel/{job}; POST /api/eyecontact/projects/{pid}/media/{mid}/preview {...setup, [lens_marks], start_s, duration_s: 10}; GET /api/eyecontact/status/{job} every 600 ms until done |
| Preview finishes | GET /api/media/stream?path=before_path and ?path=after_path (player), ?path=contact_sheet (sheet). Lens pill / angle / per-eye px / candidates / side_check read from the result |
| Candidate mm:ss button | none: preview start slider = floor(t), clamped to the clip |
| "+ mark" on a candidate / "Mark this moment" / x on a mark | none: local marks change (dedup within 0.3 s), preview marked stale; sent as `lens_marks` with the next preview/run. "Mark this moment" = preview start_s + player position when the shown preview starts at the slider, else the slider value |
| Click the contact sheet | none (lightbox reuses the stream URL); click closes |
| Fix eye contact | PUT /api/eyecontact/settings {setup} (lens_marks stripped); POST /api/eyecontact/projects/{pid}/media/{mid} {...setup, [lens_marks]}; tile polls GET /api/eyecontact/status/{job}; on completion GET /api/projects/{pid}/media + GET project (reloadSources) |
| Cancel / backdrop click (dialog) | POST /api/eyecontact/cancel/{job} if a preview is still running; GET /api/projects/{pid}/media (picks up stored marks) |
| Tile Cancel during a run | POST /api/eyecontact/cancel/{job} |
| Revert (tile) | POST /api/eyecontact/projects/{pid}/media/{mid}/revert |
| Hover the "eye contact" badge | none: tooltip from entry.eye_contact.report (lens_source, settings.lens_marks length, measured_shift_deg, median_shift_px_per_eye) |

### Frontend open issues

- /api/media/stream labels every file `video/mp4`; Chromium's `<img>` decodes the JPEG by
  sniffing, so the sheet shows, but a proper `image/jpeg` content type would be a backend fix.
- No Analyse button: the lens pill and candidates appear after the first preview (or from the last
  run's report). `EyeContactAnalysis` is typed; no `startEyeContactAnalyse` call was added.
- A preview's own report carries `settings.lens_marks = []` (backend design), so the "From your N
  marks" count for a preview is the dialog's mark count at request time.
- Not exercised in the running app (Electron not run, no frontend tests).
