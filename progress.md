# BuzzEdit - Implementation Progress

## PHASE 1: Core Video Pipeline (Week 1-2)
**Goal: Upload video → Auto-remove fumbles/silence → Export clean cut**

### 1.1 Backend Setup
- [x] Create FastAPI backend on port 8099
- [x] Set up CORS, Pydantic models, project structure
- [x] Implement FFmpeg wrapper with NVENC hardware encoding
- [x] File upload and project management endpoints
- [x] Health endpoint
- [x] Transcription routes
- [x] Analysis routes
- [x] Rendering routes

### 1.2 Audio Analysis & Transcription
- [x] Integrate Whisper large-v3
- [x] Transcribe video → timestamped SRT/JSON
- [x] Detect silence gaps (>1.5s pause = potential cut)
- [x] Detect fumbles using speech pattern analysis
- [x] Mark problematic segments with timestamps

### 1.3 Smart Cutting
- [x] Auto-cut silence segments (configurable threshold)
- [x] Auto-cut fumble segments (mark for review)
- [x] Scene detection using PySceneDetect
- [x] Generate timeline of "good" vs "cut" segments
- [x] Preview cuts before final export

### 1.4 Transitions
- [x] Crossfade between scenes (configurable duration: 0.5s-2s)
- [x] Zoom-in/zoom-out transitions (Ken Burns effect)
- [x] Dissolve transitions
- [x] Auto-select transition based on scene type
- [x] FFmpeg filter chain for transition composition

### 1.5 Frontend - Core UI
- [x] Electron window with React frontend
- [x] Drag-and-drop video input
- [x] Timeline visualization
- [x] Preview player with seek
- [x] Transcript editor
- [x] "Auto-Edit" button to trigger full pipeline
- [x] Export settings (resolution, bitrate, codec)

- [x] Verified and improved Phase 1 pipeline (Fixed FFmpeg xfade offset math, normalized Whisper word timestamps, fixed health check & portable python paths, created automated backend verification test `test_phase1.py`).

---

## PHASE 2: ComfyUI Integration & AI Agents (Week 3-4)
**Goal: AI-generated B-roll, thumbnails, overlays generated from ComfyUI**

### 2.1 ComfyUI Bridge
- [x] HTTP & WebSocket client (`comfyui_bridge/client.py`) connecting to ComfyUI (localhost:8188)
- [x] API workflow loader (`comfyui_bridge/workflow_loader.py`) for dynamic prompt/seed injection
- [x] Async queue manager (`comfyui_bridge/queue_manager.py`) with poll status & output extraction
- [x] Default workflow templates (`workflows/broll_generation.json`, `workflows/thumbnail.json`)

### 2.2 B-Roll Generation Agent
- [x] Extract visual concepts from transcript segments (`agents/broll_agent.py`)
- [x] Generate B-roll images/video via ComfyUI with styled color-card fallback
- [x] Render B-roll overlay video clips for project timeline

### 2.3 Thumbnail Agent
- [x] Extract keyframe from video at optimal speech segment (`agents/thumbnail_agent.py`)
- [x] Generate YouTube thumbnail via ComfyUI (or fallback contrast/overlay composition with custom title text)

### 2.4 Caption Agent
- [x] Generate styled burned-in captions from transcript (`agents/caption_agent.py`)
- [x] Support SRT subtitles and FFmpeg drawtext filter chains

### 2.5 Edit Agent (Main Orchestrator)
- [x] Autonomous agent (`agents/edit_agent.py`) driving end-to-end edit flow (Cut -> B-Roll -> Captions -> Thumbnail -> Export)
- [x] API endpoints (`routes/agents.py` & `routes/comfyui.py`) and UI controls (`AgentPanel.tsx`)

---

## PHASE 3: Nightly Automation (Week 5)
**Goal: Queue jobs at night → wake up to finished videos**

### 3.1 Job Scheduler & Queue Engine
- [x] Async sequential GPU job queue manager (`agents/scheduler.py`) with priority levels (`urgent`, `normal`, `low`)
- [x] Persistent queue state (`data/jobs.json`)
- [x] Date-organized output folder structure (`data/output/YYYY-MM-DD/{project_name}/`)

### 3.2 Power Management & Prevent Sleep
- [x] Windows API wrapper (`utils/power.py`) using `SetThreadExecutionState` to prevent system sleep while jobs are processing
- [x] Automatic sleep lock activation and release when queue drains

### 3.3 Dashboard UI & Controls
- [x] React Nightly Queue panel (`src/components/JobQueue.tsx`)
- [x] Queue current video with priority selector, pause/resume queue, cancel job, clear completed
- [x] Automated backend verification test suite (`test_phase3.py`)

---

## PHASE 4: Advanced Features & Filmora Integration (Week 6-7)
**Goal: Polish, multi-video projects, color grading, Filmora Export**

### 4.1 Filmora Timeline Export
- [x] Final Cut Pro XML (`.xml`) timeline exporter (`video_pipeline/filmora_exporter.py`)
- [x] Export AI-edited cuts and audio tracks for direct import into Filmora

### 4.2 Color Grading & Audio Filters
- [x] Cinematic, Vibrant, Warm, Cool, and Dark color grading filters (`video_pipeline/filters.py`)
- [x] 9:16 Vertical Shorts/Reels smart cropping converter
- [x] Background music vocal ducking (`sidechaincompress`)

### 4.3 Presets & UI Controls
- [x] Pre-defined editing templates (`video_pipeline/templates.py`)
- [x] Export to Filmora XML, Color Grading, and 9:16 Shorts converter in `ExportSettings.tsx`
- [x] Automated backend verification test suite (`test_phase4.py`)

---

## PHASE 5: Production Hardening (Week 8)
**Goal: Stability, packaging, deployment**

### 5.1 VRAM Protection & GPU Diagnostics
- [x] PyTorch CUDA memory tracking and VRAM cache flusher (`utils/gpu_utils.py`)
- [x] OOM exception inspector for automatic resolution fallback
- [x] Real-time GPU VRAM and system paths diagnostic router (`routes/system.py`)

### 5.2 Diagnostics & Setup Wizard
- [x] Interactive Diagnostics & Setup Wizard modal (`src/components/SetupWizard.tsx`)
- [x] Diagnostic button in header (`src/components/Header.tsx`)
- [x] Automated backend verification test suite (`test_phase5.py`)

---

---

## PHASE 6: Creative Toolset (Text, Geometry, Colour, Grouping)
**Goal: real NLE creative controls, all compiled into the same single FFmpeg pass**

Note on prior state: `TimelineEffect` existed in the schema but the compiler ignored it,
and `routes/advanced.py` returned `{"status": "completed"}` without running FFmpeg. Both
gaps are closed — effects now compile, and the advanced stubs are superseded by the
per-clip/master pipeline below.

### 6.1 Clip Geometry — crop, pan, zoom
- [x] `Transform` model on every clip: crop (per edge), scale, position, rotation, opacity
- [x] Static path scales from the *source* (`scale`→`pad`→`crop`) so zooming a 4K clip stays sharp
- [x] Animated Ken Burns path via `zoompan`, pre-scaling to the smaller keyframe so
      zoom-out works despite zoompan's `z >= 1` clamp (`render/effects.py`)
- [x] Overlay clips: fitted + centred by default, animated pan via time-varying `overlay` x/y
- [x] Stills get their zoompan frames retimed and PTS-shifted to the clip's position
- [x] Master transform on `Timeline`, applied after the V1 concat and surviving word rebuilds

### 6.2 Colour Correction
- [x] `ColorGrade` model: brightness, contrast, saturation, gamma, temperature, hue,
      sharpen, vignette, fade in/out — neutral values compile to no filter at all
- [x] 11 presets (`timeline/presets.py`); a preset seeds values, then stays user-editable
- [x] Per-clip grades plus a master grade applied after overlays (so B-roll grades too)

### 6.3 Text & Fonts
- [x] Text clips as first-class timeline items (`kind="text"`) on T-tracks
- [x] System font discovery by parsing each file's `name` table — 309 families found on
      this machine, grouped into regular/bold/italic (`utils/fonts.py`, no new dependency)
- [x] Full `TextStyle`: font, size, fill, outline, shadow, background box, alignment,
      position, line spacing, and fade/pop/slide-up animations
- [x] Content written to UTF-8 sidecars and referenced via `textfile=`, so Hinglish and
      Devanagari transcript text needs no filtergraph escaping (`render/text.py`)
- [x] 9 text presets (Bold Title, YouTube Pop, Lower Third, Neon Glow, Quote Card, …)

### 6.4 Caption & Intro Presets
- [x] 5 caption presets; cards built from the *enabled* words and mapped through the V1
      cuts, so captions track the edit instead of drifting with every removed filler
- [x] 6 intro presets; card-backed intros push the program behind a generated colour+silence
      segment (`program_offset_frames`) that survives transcript rebuilds
- [x] Generated items tagged by `origin`, so regenerating replaces only the last batch

### 6.5 Detach Audio & Compound Clips
- [x] Detach audio splits a video clip's sound onto its own A-track (V-tracks stay picture-only)
- [x] Compound clips group items into one block; children stored relative to the parent so
      move/trim/split slide them together, flattened at render time
- [x] Ctrl+click multi-select in the timeline drives compounding

### 6.6 UI
- [x] Inspector panel (`InspectorPanel.tsx`) — Transform, Colour, Text, Captions, Intro
- [x] Timeline: text lanes, ＋Text / ＋T, Detach Audio, Compound/Uncompound, effect badges
- [x] New routes: `/api/presets/*` plus timeline transform/colour/text/captions/intro endpoints

### 6.7 Verification
- [x] 82 backend tests pass (55 new, covering effects, text, compound, captions, intros)
- [x] End-to-end FFmpeg render exercising every feature, checked frame-by-frame
- [x] All new HTTP endpoints driven against a live backend, ending in a real render
- [x] UI smoke-tested in the built bundle against a real project

---

## PHASE 7: Media Library
**Goal: a Filmora-style bin — import video/music/images, then drag them to the timeline**

### 7.1 Bugs found in the old media pool
- [x] Pool entries created before thumbnails existed had no `thumb` key, so
      `/media/{id}/thumb` returned 404 forever and every tile rendered blank. Posters are
      now addressed by convention (`{project}_{media}_thumb.jpg`) and generated on first
      request, so old entries heal themselves
- [x] Thumbnail generation had no `-update 1`, and a clip shorter than the 1s seek point
      produced nothing; both fixed, with a frame-one fallback
- [x] Tile name/metadata row was clipped away entirely — the grid is a scrolling flex child,
      so `auto` rows were squeezed to fit instead of scrolling (`grid-auto-rows: max-content`)
- [x] Dropping media on a project that had never been transcribed failed: no timeline
      existed and the timeline panel drew no lanes to drop onto

### 7.2 Import
- [x] Files **and folders** (walked recursively, depth- and count-capped)
- [x] **Links in place by default** instead of copying — importing a 300 MB clip is instant
      and costs no disk; `copy: true` still available for material that must travel
- [x] Duplicate detection by source path; unreadable files are skipped, not fatal
- [x] OS drag-and-drop from Explorer/Finder straight onto the panel
- [x] A new project seeds the library with its own source video

### 7.3 Browsing
- [x] Search box and All/Video/Music/Images filter tabs with live counts
- [x] Video tiles hover-scrub (move across a tile to seek it), audio tiles draw their waveform
- [x] Duration, resolution and file size on every tile; rename, remove, add-at-playhead
- [x] Missing files flagged with a Relink action; removing a linked entry never deletes
      the user's own file

### 7.4 Onto the timeline
- [x] Drag a tile to any track, or double-click to drop it at the playhead
- [x] Timeline is created on demand, so a project can be built without transcribing at all
- [x] First video becomes the program (V1) and its audio is paired onto A1; later drops
      stack onto overlay tracks
- [x] Timeline sources reference the library file instead of making a third copy, and
      re-adding a file reuses its existing source so the render graph opens it once

### 7.5 Verification
- [x] 119 backend tests pass (37 new: media pool units + route-level tests driving the real app)
- [x] Live API run of the whole library flow, and a project built only by dragging media in
      that renders end to end
- [x] Media Library verified in the built UI against a real project

---

## PHASE 8: Reference-Video Style Matching
**Goal: point the tool at a video you like and reproduce how it was edited**

Scope note: the effects themselves cannot be recovered — a rendered file carries the
result, not the effect stack, keyframes or fonts. What is built is a *measured profile*
of the editing grammar, with a stated confidence per signal, applied to the user's own
footage through the existing EDL.

### 8.1 Measurement (`backend/style/`)
- [x] `frames.py` — one ffmpeg pass into a raw pipe at reduced size/rate; a 20-minute
      reference profiles in seconds. Lowers the rate rather than truncating, so a profile
      always describes the whole edit and not just its intro
- [x] `shots.py` — cut / dissolve / fade classification off a colour difference curve,
      with a **locally adaptive threshold** so calm and handheld passages of the same
      video are both judged fairly
- [x] `motion.py` — per-shot zoom and pan via RANSAC similarity fits, composed across
      the shot
- [x] `color.py` — exposure / contrast / saturation / cast statistics and a fit onto the
      existing `ColorGrade`, so a matched look stays editable in the Inspector
- [x] `rhythm.py` — spectral-flux onset envelope, autocorrelation tempo with octave
      correction, and cut-to-beat alignment
- [x] `text_regions.py` — caption band geometry (position, size, box) with no OCR
- [x] `profile.py` — the `StyleProfile` model, orchestration, and app-wide storage in
      `data/styles/` so one reference serves every project

### 8.2 Bugs found and fixed along the way
- [x] Motion originally integrated a per-frame optical-flow rate over each shot; noise
      accumulated instead of cancelling and **static footage measured 0.156 of false zoom**.
      Replaced with composed per-step RANSAC fits — static now measures exactly zero
- [x] Cut detection ran on luma only, so a cut between two equally bright shots was
      invisible. Now differences all three channels
- [x] Tempo estimation returned **half the true BPM** (120 read as 60) — the classic
      autocorrelation octave error. Fixed with harmonic candidates and a tempo prior
- [x] Beat alignment compared cuts to an absolute beat grid, so a 2% tempo error walked
      a full beat out of step and scored a perfectly beat-cut montage near zero. Now
      measured on intervals, which stays local
- [x] Caption detection measured mean gradient energy and flagged any hard-edged graphic
      as text; now counts stroke crossings and bounds band thickness and horizontal extent
- [x] `video_pipeline/analyzer.py` imported `scene_detect` (the package is `scenedetect`),
      so scene detection had **always** silently fallen back to a 0.5 fps frame-diff scan
      that cannot resolve a shot shorter than two seconds. Now uses the style analyser
- [x] A source with no audio stream still got A1 items, and `[n:a]` made ffmpeg reject the
      whole graph. `has_audio` is now probed and honoured, and render failures return the
      reason instead of a bare 500

### 8.3 Application (`style/apply.py`)
- [x] Look -> `master_color`, fitted from the project's *own* footage to the reference's
      statistics (the same target needs opposite corrections depending on your source)
- [x] Pace + movement -> punch-ins and pull-backs placed at the reference's cuts-per-minute
      and zoom depth. Single-camera footage has no angles to cut between, so the felt
      rhythm is produced the way editors actually produce it
- [x] Caption geometry -> stored overrides the caption generator picks up
- [x] Transitions -> programme-level fades only; a dissolve at every cut needs `xfade`,
      which the concat programme path cannot express
- [x] Deterministic: re-applying a profile reproduces the same edit

### 8.4 UI & API
- [x] `routes/style.py` — analyze / list / get / apply / delete, analysis off the event loop
- [x] "Match a reference video" in the Inspector: measured summary with per-signal
      confidence badges, per-part apply toggles, and a plain-English report of what changed

### 8.5 Verification
- [x] 151 backend tests (32 new) against ffmpeg-generated references with known ground
      truth: 8 shots at 1.5s recovered exactly, a 30% push recovered as 0.28, 120 BPM
      recovered as 119.5, caption band at 82% height recovered as 82%
- [x] False-positive coverage is explicit — static footage, graphics and texture must
      report *no* zoom and *no* captions, since a wrong profile actively damages an edit
- [x] Live API run ending in a real render; apply verified in the built UI

---

## PHASE 9: Auto-Edit Rebuilt (audio-driven)
**Goal: make auto-edit actually cut the fumbles and dead air**

### 9.1 Why the old one did nothing — measured, not guessed
On the real 195s project in `data/projects/7322a87c`:
- **4 of 351 words** were flagged. Whisper is trained to emit clean readable text and
  deletes "um", "uh", stutters and false starts *before* we see them, so matching filler
  vocabulary against the transcript can only ever find what the ASR left behind.
- **23 words claimed over a second each; one claimed 10.13s.** 76 seconds — 39% of the
  video — sat sealed inside "words". Dead air removal worked on the gaps *between* words,
  so none of it was reachable.
- **The timeline had been built from the English translation.** The ASR produced 485
  Hinglish words; the timeline held 351 English ones ("Has it ever happened to you…").
  Whisper's language auto-detection is unstable on code-switched speech, and a run that
  lands on `en` paraphrases rather than transcribes — so cut points had no relationship
  to the audio at all.

### 9.2 The new approach: decide from the audio, use text only where text is better
- [x] `asr/vad.py` rewritten — speech/silence with a threshold derived from each file's
      own noise floor and speech peak. The old fixed -35 dB read a quiet recording as
      pure silence and a loud one as pure speech
- [x] `asr/align.py` — clamps word timings onto the speech they actually cover, exposing
      the air hidden inside them; finds speech regions no word claims, which is exactly
      where the fillers Whisper deleted still live (bounded at 0.18–1.2s: longer is
      speech the ASR missed, and cutting it would delete the user's content)
- [x] `asr/fumble_engine.py` — the planner. Detected fillers are inserted as ordinary
      disabled words, so they cut, show up in the transcript panel, and can be overruled
- [x] `timeline/ops.py` — the rebuild intersects kept words with the speech map, so
      silence *inside* a word is cut too, and trims a long pause to a padding beat rather
      than deleting every gap over 0.17s, which is what made the old output breathless
- [x] ASR config fixed, measured three ways: forcing the language and disabling
      `condition_on_previous_text` gave 435 words / 0.51s p90 / 13 over-long, against
      401 / 0.72s / 27 before. `vad_filter` is off — the planner does its own VAD, and
      Whisper's remapping is where the absorbed timestamps came from
- [x] Detected language persisted to project settings and reused, so a project can never
      silently flip to an English paraphrase between runs

### 9.3 Result on the real project
| | old | new |
|---|---|---|
| removed | 26.9s (14%) | **40.4s (21%)** |
| silence left in the cut | 36.2s | **17.7s** (≈ the intended padding) |
| fillers found | 0 | **23** |
| word timings corrected | 0 | **53** (12.5s recovered) |

Rendered output verified at 155.3s, matching the timeline exactly.

### 9.4 UI
- [x] "Pauses" slider next to Fumbles — how long a pause may be before it is trimmed
- [x] The completion dialog now reports what was removed and why, instead of a bare word
      count that could not distinguish a working pass from a no-op

### 9.5 Verification
- [x] 174 backend tests (18 new) including quiet-recording VAD, timing repair, the
      filler/content boundary, and silence hidden inside a word
- [x] Live `auto_edit` run against the real project, output rendered and measured

---

## PHASE 10: Retake / False-Start Collapsing
**Goal: "I started that sentence four times" becomes one clean take**

The shape, verbatim from the user's recording:

    dosto kya ap · dosto kya · dosto kya a · dosto · dosto kya apko pata hai india…

The old detector cut **one word out of nine** here. It looked for exact adjacent
n-gram repeats, but a restart is a *prefix* of the good take at varying lengths with
clipped words ("a" for "apko"), which exact comparison never matches.

### 10.1 Why not grammar checking
It was the obvious idea and it does not work: there is no dependable parser for
romanised Hinglish code-switching; ASR text is already noisy, so ungrammaticality is
not evidence of a fumble; and grammar cannot answer the actual question, which is
*which attempt to keep*.

### 10.2 What replaced it (`asr/retakes.py`)
Speech research models a repair as `reparandum → (interregnum) → repair`: an abandoned
attempt, an optional "uh", then a version that starts the same way. That leaves a
detectable "rough copy" — needs no grammar and no vocabulary, so it works the same in
Hindi, English or a mix.
- [x] Fuzzy token matching: prefixes ("prob"/"probably"), and edit distance for the
      ASR spelling the same sound differently between takes
- [x] Greedy left-to-right, keeping the **last** complete take, so a pile-up of four
      run-ups collapses in one pass
- [x] Confidence, not a boolean. Structure alone cannot separate a restart from an
      idiom, so weak signals become LLM candidates instead of edits

### 10.3 Calibrated against real over-cuts
- A **two-word** match is not enough: "it is what it is" is a rough copy by every
  structural measure
- A **three-word** match is not enough either — "unakaa naam thaa" ("his name was")
  recurs naturally, and an earlier build used it to **delete the speaker's own
  subject, Thomas Gilovich, from the video**. Now referred to the LLM
- Confidence is raised by real evidence: a chain of consecutive restarts (floundering
  repeats, idiom does not), a hesitation pause, a word clipped mid-syllable, and an
  exact double with nothing left over
- Single-word matches count only back-to-back; Hindi is full of "hai … hai"
- The abandoned tail allowance scales with match length — speakers routinely get most
  of a sentence out before restarting (one real case ran 16 words), and a fixed 3-word
  cap missed a **seven-word verbatim match**
- [x] LLM prompt rewritten to spell out restart-versus-deliberate-repetition and the
      "keep the last, most complete take" rule

### 10.4 Result on the real 195s video
| | original | after Phase 9 | after Phase 10 |
|---|---|---|---|
| removed | 26.9s (14%) | 56.3s (29%) | **76.4s (39%)** |
| retake words cut | 0 | 0 | **130** |

The speaker restarts *"aapake saath kabhee aisaa huaa hai ki aap kisee"* five times and
*"ek psychological experiment kiyaa gayaa thaa"* four times; all collapse to the final take.

**Known limit:** when the ASR romanises the same speech differently across takes
("cornwall ooniversity" vs "kaॉnvel yoonivarsitee"), the tokens do not match and the
repeat survives structural detection. The LLM adjudication layer catches these
semantically when LM Studio is running.

### 10.5 Verification
- [x] 199 backend tests (23 new). Half the retake tests are negative — the dangerous
      failure is deleting a phrase the speaker repeated on purpose, not missing a fumble

---

## PHASE 11: Transitions, Atmosphere Effects & Cinematic Framing

### 11.1 Transitions (`render/transitions.py`)
- [x] All **58 xfade transitions** this FFmpeg build supports, grouped for the UI
      (Dissolve / Wipe / Slide / Smooth / Shape / Motion), plus 14 named presets
- [x] Set per cut or as a programme default; a per-clip transition overrides it
- [x] The programme is assembled as a **mix of concat and xfade** — runs of hard cuts
      concatenate, and only real transitions overlap
- [x] Audio `acrossfade`s at exactly the same junctions
- [x] Durations clamped to what the two clips can cover: xfade needs both sides to
      span the overlap, and asking for more produces a broken graph, not a shorter fade

### 11.2 Atmosphere effects (`render/atmosphere.py`)
Rain · Snow · Lightning · Sunlight · Light Leak · Fog · Wind · Film Grain — all
generated procedurally from lavfi sources, so they need no stock footage and work at
any resolution. 11 presets (Light/Heavy Rain, Snowfall, Lightning Storm, Golden Hour,
Sun Flare, Warm/Cool Light Leak, Fog, Wind, Film Grain), each with intensity, speed
and — where it applies — a tint.

### 11.3 Cinematic framing
- [x] 2.39:1, 21:9, 1.85:1, 4:3 and 1:1, matted on without changing export resolution
- [x] Note: scope bars on a *vertical* source leave a very thin band. That is the
      correct maths, not a bug — 1080/2.39 is 452px of a 1920-tall frame

### 11.4 Four bugs found by looking at the output rather than trusting the graph
- **Everything turned magenta.** `screen` is an RGB operation; run over YUV chroma —
  signed offsets around 128 — it maps two neutral greys to 191. Every composite now
  converts to `gbrp` first
- **Audio drifted 1.2s from picture.** The atmosphere layer is generated longer than
  the programme, and `blend` pads its *other* input to match, silently undoing every
  transition's shortening. Fixed with `shortest=1` and the post-transition duration
- **`gradients` colours were ignored.** 6-digit hex parses with alpha 0 and yields a
  transparent layer; 8-digit (`0xFF7A2AFF`) is required
- **Rain rendered empty.** `noise` sits around its base value, so on black it never
  exceeds ~148 and a 230 threshold produced nothing. Rain and snow start from grey

### 11.5 Verification
- [x] 223 backend tests (24 new), including an end-to-end render asserting picture and
      sound come out the same length
- [x] Every effect prototyped and inspected frame by frame before being implemented
- [x] Live API run and a render on the user's own footage

---

## PHASE 12: Dockable Workspace

### 12.1 What it does
- [x] Three panels — **Preview**, **Tools**, **Timeline** — each dockable to the
      centre, left, right or bottom zone, floated free, or hidden
- [x] **Drag by the title bar** to move a panel; dock zones light up while dragging
      and a drop re-docks it. Floating panels also drag and resize freely
- [x] **Splitters** between zones, with min/max clamps
- [x] **Lock** (header padlock) pins the arrangement: no moving, resizing, undocking
      or splitter drags. Showing and hiding panels still works, since that is a
      change of what you are looking at rather than of the layout
- [x] Four presets — Default, Editing (tools left, tall timeline), Colour & FX (wide
      inspector, short timeline), Review (picture and timeline only) — plus Reset
- [x] Everything persists to `localStorage`, deliberately *not* the project file:
      the layout describes this machine's setup, so it should not travel with a
      project or be undone by opening one

### 12.2 Implementation notes
- Dragging uses **pointer events, not HTML5 drag-and-drop**. The editor already uses
  HTML5 drag for media-pool → timeline drops, and mixing the two makes a panel drag
  register as a media drop on the lanes underneath
- A dragged floating panel gets `pointer-events: none`. Drop targets are resolved
  with `elementFromPoint`, and a panel that follows the cursor is always the topmost
  element under it — so before this every drop landed on the panel itself and
  nothing ever re-docked
- Stored layouts are **merged** with the defaults on load, so a layout saved by an
  older build that predates a panel still renders that panel instead of dropping it

### 12.3 Verification
Driven in the built UI: preset switching, undock to floating, drag, drop-to-redock,
splitter resize, lock blocking resize while still allowing hide, and persistence
across a reload. 223 backend tests still pass.

---

## PHASE 13: Layer Management & Flip

### 13.1 What it does
- [x] Every lane now has a **header** in a gutter down the left: the track's name and
      five switches — hide (👁), mute (🔊, audio lanes only), lock (🔓), add (＋)
      and delete (🗑)
- [x] **Selecting a layer** highlights both the header (blue tint + accent bar) and the
      whole lane; clicking the header again clears it
- [x] **Hide** drops the lane out of the render entirely — an overlay stops being
      composited, text stops being drawn — and dims it on the timeline
- [x] **Mute** keeps an audio lane visible but out of the mix. Muting the last mixed
      lane removes the `amix` rather than mixing one input
- [x] **Lock** refuses every edit on that lane — split, move, trim, delete, dropping
      new media, and effect changes — and hatches the lane so it reads as locked.
      Deleting a locked track is refused too; you have to unlock it first
- [x] **Delete layer** removes the lane and every clip on it
- [x] **Flip horizontal / vertical** in the Inspector's Transform section

### 13.2 Implementation notes
- The switches live on the **timeline** (`Timeline.tracks: Dict[str, TrackState]`),
  not on the clips. A lane is a property of the arrangement — an empty lane still has
  to be able to be muted, and a flag on the clips would be lost the moment the lane
  was cleared
- `extra_tracks` remembers lanes that hold no clips, so a reserved or flagged lane
  survives a reload instead of vanishing when the last clip leaves it
- Track lock is enforced in `clip_ops`, not in the UI: `_require_editable` takes the
  timeline and raises, so the API refuses a locked edit with a 400 whatever the
  caller is. The greying-out in the UI is a courtesy on top of that
- **Flip runs after the crop.** Crop coordinates then still refer to the original
  frame, so cropping the left and flipping keeps the crop on what was the left —
  which is what two independent controls should do
- The lane's inline style sets `background-color`, not the `background` shorthand.
  The shorthand resets `background-image` and so silently erased the locked lane's
  hatching

### 13.3 Verification
- [x] 235 backend tests (12 new): flag round-trip and persistence, locked-track
      refusals through the real API, delete-with-clips, hidden/muted lanes dropping
      out of the compiled filtergraph, and flip's position in the geometry chain
- [x] Driven in the built UI on the user's own project: select, mute, lock (hatching
      confirmed), and flip — each checked against what the backend actually stored,
      and the compiled graph confirmed to carry `hflip`. All test changes reverted

---

## PHASE 14: Adjustment Layers

### 14.1 What it does
- [x] **＋ Adjust** drops an adjustment layer at the playhead — a clip with no
      picture of its own that treats every layer *below* it, for as long as it runs
- [x] It carries the **grade**, the **transform** and its own **atmosphere effects**,
      all from the panels that already existed. Selecting one re-points the
      Inspector at it and says so in plain words
- [x] **Where it sits is what it reaches.** An adjustment on V3 treats V1 and V2 and
      leaves V4 alone, because it is applied at its own height in the compositing
      stack rather than at the end
- [x] **How long it runs is how long the look lasts** — trim or move it like any clip
- [x] **Strength** (the transform's opacity) blends the treatment back over the
      untouched picture, so a grade can be applied at 40%
- [x] Effects can now be attached to one adjustment layer instead of the whole
      programme, which is what makes "rain over this section only" possible

### 14.2 Implementation notes
- The picture is **split in two**: one copy goes through the layer's chain and is
  overlaid back onto the untouched copy with `enable='between(t,…)'`. Windowing this
  way rather than hanging `enable=` off each filter keeps one mechanism for all
  three — not every filter supports the timeline feature, and `zoompan` and the
  blends behind the atmosphere effects certainly do not
- An adjustment that adjusts nothing emits **no filters at all**, so an empty layer
  costs nothing
- `build_canvas_transform` gained `frame_offset`. An adjustment's chain runs over the
  whole programme, so an animated move has to be counted from the frame the clip
  starts at and held still either side — `clip((on-N)/span,0,1)` instead of `on/span`
- Effects are addressed by `item_id`: absent means the programme, present means that
  adjustment layer. Anything that is not an adjustment is refused rather than
  silently accepting an effect nothing would ever render

### 14.3 Verification
- [x] 249 backend tests (14 new), including an **end-to-end render measured with
      `signalstats`**: saturation inside the layer's window drops to 1.0 while the
      frames either side stay above 100 — and the assertion was deliberately
      inverted once to prove it bites
- [x] Stacking order asserted against the compiled graph: the low layer's grade lands
      before the B-roll composite, the high layer's after it
- [x] Driven in the built UI on the user's own project — add, grade with a preset,
      attach rain, confirm the programme's own effect list stayed empty — then a
      6-second render of the real footage inspected frame by frame. Test layer removed

---

## FIX: repeats, stumbles and choppiness (2026-08-13)

User watched the render and named three faults. Measured before -> after:

| | before | after |
|---|---|---|
| 4-word phrase repeats | 1 | **0** |
| Cuts in the programme | 35 | 32, all softened |
| Segments under 0.5s | 4 | 3 (shortest 0.40s, was 0.27s) |
| Long fumbles still audible | 0 | 0 |

- **Repeats:** the model reads a phrase said twice as rhetoric and keeps both.
  In an unscripted monologue a 4+ word run coming back within a breath is the
  speaker repeating themselves, so `_long_repeats` now overrides the model on
  exactly that shape (verbatim, >=4 words, within 25 words) and drops the first
  copy. Short echoes and spaced restatements are still left alone.
- **Stumbles:** the second fluency pass may now cut a short run (<=4 words)
  anywhere, not only words that repeat. That restriction existed because the
  pass had cut the greeting off the top of the video; the length limit protects
  that case while allowing "dosto ... aisaa [ha aap vah] hai ki ..." to go.
- **Choppiness:** a cut costs a visible jump, so it now has to be worth one.
  `Timeline.min_removal_seconds` (0.25s) refuses to cut a removal too short to
  hear, and `min_segment_seconds` (0.35s) drops a kept sliver that sits between
  two removals — on screen that is a flash of a different head position, and
  dropping it merges two cuts into one. On top of that, auto-edit sets a 0.12s
  dissolve as the default transition: far too short to read as a dissolve, long
  enough that the join stops snapping. Turn it off with `hard_cuts` in project
  settings, or per-clip in the Inspector.

**Known limit, not solved:** on this recording the final take of the opening is
itself garbled ("kabhee aisaa **ha aap vah** hai ki"), and the only clean reading
of that phrase lives in an *earlier* attempt. Fixing it needs either a
take-to-take join (removed deliberately, because it sounded like two takes) or
dropping the greeting. That is an editorial choice, not a bug.

---

## FIX: switched off the portable ComfyUI (2026-08-13)

The "procedure entry point could not be located in the dynamic link library"
dialog came from **ComfyUI portable**, which is a cu118 build. Everything else on
this machine is cu12 — the app's own venv, and the ComfyUI Desktop install. Now
on the Desktop one:

    code   C:\Users\singh\ComfyUI-Installs\ComfyUI\ComfyUI   (main.py)
    data   C:\Users\singh\Documents\ComfyUI                  (models/input/output)
    python C:\Users\singh\Documents\ComfyUI\.venv            (torch 2.10 + cu128)

- `START_APP.bat` / `.ps1` launch it as `python main.py --base-directory <data>`.
  `PYTHONIOENCODING=utf-8` is set first: ComfyUI logs emoji and on the console's
  cp1252 codepage the *logging call itself* throws and kills the process.
- The `start` line no longer nests quotes inside `cmd /c` — `start /d <dir>` does
  the same job without the quoting minefield. Verified: 8188 answers 200.
- `config.py` derives `COMFYUI_INPUT_DIR`/`OUTPUT_DIR` from `COMFYUI_BASE_DIR`,
  which reads an env var so another machine needs no code change.
- **`faster_whisper_engine` was injecting the portable's cu118 torch DLLs into
  our own process** — unconditionally, even after the venv's cu12 libraries had
  already been registered. Two CUDA generations on one DLL search path is exactly
  how that entry-point error is produced. It now falls back only when this venv
  supplies nothing, and reads the path from `CUDA_DLL_DIR` instead of hard-coding
  a machine.

---

## FIX: "app not starting" / entry point could not be found (2026-08-13)

Two faults, one of them long-standing and one mine.

**1. Electron polled `localhost`, the backend binds `127.0.0.1`.** On Windows
`localhost` resolves to `::1` first, so every health poll from the Electron main
process was refused before it reached a backend that had started perfectly well.
Symptoms: "Backend failed to start" printed over a working backend, and
`isBackendRunning()` always answering false — so each launch spawned a **second**
backend to fight the first for port 8099. Two Python processes racing to load the
same CUDA DLLs is where a "procedure entry point could not be located" dialog
comes from. Chromium hid the same bug in the renderer by retrying over IPv4
itself, which is why the window still worked while the console screamed.

**2. `/api/health` imported `faster_whisper`** (my regression from the earlier
whisper-probe fix). That import pulls in CTranslate2 and the CUDA runtime;
Electron allows each health call two seconds, and a cold import blows through
that every time. Availability is now checked with `importlib.util.find_spec`,
which reads packaging metadata, loads no DLLs and cannot hang: 0.23s per call.

Verified by launching exactly as `START_APP.bat` does: "Backend started
successfully", one backend process, one CUDA registration.

---

## PHASE 15: The LLM writes the edit (2026-08-13)

The user asked why grammar was never used: "get the transcript, ask the LLM for
the proper grammatically correct, most continuous sentence, and then based on
that make the edits and cuts". They were right, and the earlier note rejecting
"grammar checking" does not cover it — that rejected rule-based *parsers*
(no dependable one exists for romanized Hinglish). Using the model as the
fluency judge is a different mechanism, and it is the only component that can
answer the question that actually matters: **is what is left a sentence?**

The evidence that structure alone cannot: the speaker made five attempts at the
opening. Structure cut attempts 1-4 correctly, then cut *inside* attempt 5 and
left `aap kisee kamare [men gae hoon jahaan par bahut saare log hai phrikteev
hol] men gae ho phir baahar gayaa hai` — a Frankenstein sentence nobody said.

### 15.1 How it works (`asr/fluency.py`)
The model is handed the spoken text (with `|` where the speaker paused) and asked
for the transcript as it should sound. Its answer is aligned back onto the real
tokens with `difflib`, and **only `delete` opcodes cut**:
`replace` keeps the original (a reworded span is a guess), `insert` is ignored
(there is no video for words nobody said). So a model that paraphrases,
translates or hallucinates *cannot delete anything* — it can only fail to delete.

Trust limits per window, any of which discards it and leaves structure in charge:
≥35% of the window echoed verbatim, ≤75% deleted, no single removal over 40% of
the window. A discarded window is **absent** from the result rather than reported
as "keep everything" — reporting it as keeps silently resurrected 81 structural
cuts.

### 15.2 The three findings that made it work
- **Model size is the whole game.** A 7.5B model echoed the transcript back
  unchanged (2 cuts in a 240-word window). A 26B-A4B removed 101 and collapsed
  the pile-up correctly. The default model is now the 26B and `ensure_ready` will
  *load* it rather than settling for whatever is already resident.
- **Alignment must anchor from the RIGHT.** Every attempt opens with the same
  words, so the model's answer matches all five equally; a left-anchored diff
  kept attempt one's opening (1.5s) and spliced it onto attempt five's
  continuation (17.2s). The text read perfectly and the audio jumped 15 seconds
  mid-sentence.
- **Text-optimal is not always speakable.** The model's best wording took the
  opening from attempt 3 and the ending from attempt 5, because neither is clean
  alone. `snap_runs_to_the_final_take` moves each surviving block onto the take
  that follows it when the words just before that take are a rough copy — so the
  edit plays from one continuous stretch of speech, and the second pass tidies up
  *inside* it. The second pass is restricted to words that repeat nearby: given
  free rein it re-edited the sentence and cut the greeting off the video.

### 15.3 Measured on the real 195.6s recording
Five opening attempts → one continuous take, greeting intact. Cut words still
audible in the render: **0**. A/V sync within 3ms. 105.3s programme.
288 backend tests (24 new, most of them about what happens when the model
misbehaves). The one repetition that survives — `sab aapako jaz kar rahe hain`
twice — is a rhetorical restatement the model deliberately kept on both passes.

---

## FIX: seamless auto-edit (2026-08-13, second pass)

Reported: "slightly better but the edit does not feel seamless and it still has
repeating words or phrases in form of fumbles". Measured against the render, not
the transcript, which is what exposed the core bug.

**1. Cut words were still IN the render.** The transcript said cut; the audio
played them anyway. The segment rebuild merges kept pieces across any gap shorter
than the 0.4s natural-pause threshold — and a removed filler or stutter *is* such
a gap. Measured: 13 "cut" words, five of them "[uh]" fillers, still audible in the
finished video. The rebuild now distinguishes the two kinds of gap: a pause is
bridged, a gap containing removed speech never is (removals under ~2 frames stay,
deliberately — a 66ms cut is an inaudible pop and a visible video jump).

**2. Padding replayed what was just cut.** The 0.12s breath either side of a cut
was allowed to expand back into the removed span, re-covering most of a short
fumble. Padding now stops at removed speech; it still breathes into silence.

**3. Cuts land on the quietest nearby moment.** The audio pass now stores the
VAD's 20ms loudness envelope on the timeline (`energy_envelope`), and the rebuild
snaps every cut edge ±2 frames to the deepest energy dip — the articulation
boundary — instead of trusting the ASR's ±50ms word edge. It only moves for a
real dip (≥2dB), so flat energy leaves the edge alone.

**4. Every audio join is declicked.** Hard concat at an arbitrary sample is an
audible tick at every cut. Each A1 segment now carries 8ms triangular edge fades
— far below the threshold of sounding like a fade, but the click is gone.

**5. Language detection is pooled.** Whisper auto-detect reads the first 30s
only; one atypical opening mislabels the file, and an `en` label over Hindi
*paraphrases* the audio into English — the current transcript was exactly that,
so the repeats in the audio were invisible to any text detector and every word
timing was a guess. Detection now samples three windows across the file and, when
torn between English and an Indic language, takes the Indic side (a `hi` pass
still transcribes English words; an `en` pass destroys Hindi ones).

**6. Romanization variance folded.** "woh/vo", "kyaa/kya", "dostho/dosto" now
match in the retake detector via a coarse phonetic fold (secondary check, 0.85).

**7. The retake matcher grew up.** Measured misses on the re-transcribed
recording, each with a rule that closes it:
- The takes of a doubled sentence sat 31 words apart; lookahead was 30. Now 45 —
  the abandoned-tail allowance is the real guard, not the window.
- A second retake pass runs after the LLM cuts land: removing an *inner*
  flounder shortens the distance to an outer retry, so the pass converges.
- Retries reformulate ("ek experiment kiyaa thaa" → "ek saaikalaॉjikal
  eksaperiment kiyaa gayaa thaa"): the run matcher now tolerates single-word
  insertions/deletions/substitutions, rationed, quality-discounted, and stopped
  dead at any recurrence of the anchor word (an anchor recurrence is the next
  attempt's start; runs that crossed one out-scored the true repair and left
  stray fragments). Candidates are selected by evidence = length × quality.
- A speaker who flubs on camera stops, composes for 3-6 seconds, and restarts.
  Long clean silences no longer end the search; they raise the evidence bar
  (4+ matched words) and add confidence when met.
- Cuts create new stutters ("men [hol] men" → "men men"): a final duplicate
  sweep runs after every pass.
- The energy-dip snap is bounded by removed speech (one frame of grace), or it
  re-covered short fumbles from both sides of the gap.

**Measured on the real 195.6s recording, end to end** (re-transcribed as `hi`,
LLM adjudication live): cut words still audible in the render 13 → **0**;
3-gram near-repeats 14 → **0**; adjacent doubles **0**; picture and sound within
1.3ms; programme 100.9s. 267 backend tests pass (14 new).

---

## FIX: repeats surviving auto-edit (2026-08-13)

Reported as "the auto edit is not working well, same repeated words as fumbles are
still present". Measured on the real project rather than read off the code, and it
was three separate faults stacked on top of each other.

**1. The matcher was blind past the first filler.** A repair is
`reparandum → interregnum → repair`, and the interregnum is made of the words
already marked for cutting. Walking the raw word list, "you must feel that everyone
is watching you · uh · you must feel that everyone is watching you" matched *one*
word and the whole attempt was kept — the sentence played twice in the finished cut.
Adjacency is now judged on the words that survive, in both the retake matcher and
the stutter rule. A gap holding an editing term may also run to 6s instead of 2.5s:
a silence the speaker spent floundering in is the opposite of a sentence boundary.

**2. That fix over-reached, so it is fenced.** Seeing through a filler puts two
ordinary sentences within reach of each other — "they didn't notice it · uh · very
few people noticed it" is a two-word rough copy where both halves are meant. A match
that had to bridge a filler now needs three words to count.

**3. The LLM adjudication layer had never run.** Every auto-edit reported
`used_llm: False` with nothing in the log. Two silent 400s: readiness was judged on
`/v1/models`, which lists models that are merely *downloaded*, so the configured
`google/gemma-4-12b` — which cannot load on this machine — looked ready; and this LM
Studio build rejects `response_format: json_object`. Both were swallowed by a bare
`if status != 200: return None`. Readiness now comes from `/api/v0/models` (real
`state`), `ensure_ready` returns the model id that actually serves and falls back to
one that loads, requests use `json_schema` with a plain-text retry, and every non-200
is logged. Also fixed: `/api/health` probed `whisper`, which is not a dependency, so
a working install always reported transcription unavailable.

**Measured on `data/projects/7322a87c` (195.6s):** the doubled sentence is gone, the
genuine repeat ("very few people noticed it") is kept, and a 3-gram scan of the cut
finds no surviving fumble repeats. Deterministic core: 99 words (75 retake, 23
filler, 1 stutter). 254 backend tests, 5 new.

**Note for the user:** this project has `fumble_aggressiveness: 1.0`, at which every
ambiguous candidate is cut — that is what frays sentences into "your shoes broke or
that". At 0.5 the same fumbles go (99 words) and the sentences stay whole.

---

## RENAME: BuzzcafEditor -> BuzzEdit (2026-08-13)

The editor is now **BuzzEdit**. The parent folder `Buzzcaf Media` is the company and
was deliberately left alone; only the editor's own name changed.

- Folder: `B:\youtubeProjects\Buzzcaf Media\BuzzEdit`
- Package `buzzedit`, appId `com.buzzedit.app`, product name / window title / tray /
  logo `BuzzEdit`, API title `BuzzEdit API`
- Identifiers: ComfyUI client `buzzedit_client`, output prefixes `buzzedit_broll` and
  `buzzedit_thumbnail` (matched in `workflows/*.json`), drag type
  `application/x-buzzedit-media`, layout key `buzzedit.layout.v1`
- `layout.ts` reads the old `buzzcaf.layout.v1` key once and copies it forward, so a
  saved dock layout is not thrown away by the rename
- Absolute paths stored inside `data/projects/*.json` were rewritten, so existing
  projects still find their footage. All 9 stored media paths verified to resolve

Windows will not rename a directory that a running process is sitting in, so the move
was done by relocating the contents; an empty `BuzzcafEditor` folder is left behind
until whatever holds it (this editor session / an Explorer window) closes.

---

## STATUS: ALL PHASES COMPLETE (Phase 1-14)
## LAST UPDATED: 2026-08-13