# BuzzEdit - AI-Powered Automated Video Editor
## Full Implementation Plan
## Created: 2026-08-10

---

## WHY NOT A FILMORA EXTENSION?

**Filmora has no public plugin/extension API.** Attempting to reverse-engineer its SDK is fragile, breaks on updates, and cannot access system resources (ComfyUI, GPU, file system) needed for AI processing.

**Decision: Build a standalone Electron desktop app** that acts as your personal video editing workstation. It will:
- Accept raw video input
- Process everything automatically (transcription, silencing, cuts, transitions, AI-generated B-roll)
- Export finished video OR send timeline to Filmora for manual polish
- Integrate directly with your existing ComfyUI setup for overnight AI generation

---

## YOUR HARDWARE & ASSETS (Inventory)

### System
| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5060 Ti (16GB VRAM, CUDA 13.2) |
| CPU | Intel i7-14700K (20 cores / 28 threads) |
| RAM | 32GB DDR5 4800MHz |
| Disk (B:) | ~81GB free of ~1TB |
| FFmpeg | 8.1 with NVENC/NVDEC hardware acceleration |

### Installed AI Models (B: portable + C:\Documents\ComfyUI)
| Capability | Models Available |
|-----------|-----------------|
| **Video Generation** | Wan2.1 T2V 1.3B, Wan2.2 I2V 5B/14B, LTX-Video 2B |
| **Image Generation** | Flux.1 Dev (fp8), Z-Image Turbo, Juggernaut XL v9, DreamShaper XL Turbo, RealVis XL v4, HiDream I1 |
| **Image Editing** | Qwen Image Edit 2509, USO Flux Projector |
| **LoRAs** | ~50 LoRAs (Khayal3Baje, podcast3baje, Wan2.2 Lightx2V, realisms, styles) |
| **Speech-to-Text** | Whisper large-v3, tiny.en, faster-whisper custom nodes |
| **Face Tools** | InsightFace, ReActor, GFPGAN, CodeFormer |
| **Background Removal** | BiRefNet (7 models), RMBG-1.4 |
| **Upscaling** | 4x-UltraSharp, RealESRGAN x4plus |
| **ComfyUI Workflows** | 32 saved workflows (video, image, transcribe, upscale, faceswap, B-roll) |

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                    BUZZEDIT (Electron)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  UI Layer   │  │  Agent Layer │  │   Pipeline Engine      │  │
│  │  (React)    │  │  (Python)    │  │   (Python + FFmpeg)    │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬─────────────┘  │
│         │                │                      │                │
│         ▼                ▼                      ▼                │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              FastAPI Backend (localhost:8099)             │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │    │
│  │  │ Video    │ │ Audio    │ │ AI       │ │ ComfyUI    │  │    │
│  │  │ Pipeline │ │ Pipeline │ │ Agents   │ │ Bridge     │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## PROJECT STRUCTURE

```
BuzzEdit/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── electron/
│   ├── main.cjs              # Electron main process
│   └── preload.cjs           # IPC bridge
├── src/                      # React + TypeScript frontend
│   ├── main.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── components/
│   │   ├── Timeline.tsx      # Video timeline editor
│   │   ├── MediaLibrary.tsx  # Input video browsing
│   │   ├── PreviewPlayer.tsx # Video preview
│   │   ├── AgentPanel.tsx    # AI agent controls
│   │   ├── SettingsPanel.tsx # ComfyUI paths, model selection
│   │   ├── JobQueue.tsx      # Nightly job dashboard
│   │   └── TranscriptEditor.tsx  # Edit extracted transcript
│   ├── hooks/
│   │   ├── useVideoPipeline.ts
│   │   ├── useAgent.ts
│   │   └── useJobQueue.ts
│   └── api.ts                # Centralized API client
├── backend/                  # Python FastAPI backend
│   ├── main.py               # FastAPI app, routes
│   ├── video_pipeline/
│   │   ├── __init__.py
│   │   ├── analyzer.py       # Scene detection, silence detection
│   │   ├── cutter.py         # FFmpeg-based cutting
│   │   ├── transitions.py    # Crossfade, dissolve, zoom transitions
│   │   ├── compositor.py     # Layer compositing, B-roll overlay
│   │   └── exporter.py       # Final render with NVENC
│   ├── audio_pipeline/
│   │   ├── __init__.py
│   │   ├── transcriber.py    # Whisper STT (large-v3)
│   │   ├── silence_detector.py  # Detect pauses/fumbles
│   │   ├── noise_reducer.py  # Background noise removal
│   │   └── tts.py            # AI voice generation (edge-tts / Kokoro)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── edit_agent.py     # Main editing decision agent
│   │   ├── broll_agent.py    # B-roll generation agent
│   │   ├── thumbnail_agent.py # Thumbnail generation agent
│   │   ├── caption_agent.py  # Auto-caption generation
│   │   └── scheduler.py      # Nightly cron scheduler
│   ├── comfyui_bridge/
│   │   ├── __init__.py
│   │   ├── client.py         # WebSocket client for ComfyUI
│   │   ├── workflow_loader.py # Load/modify saved workflows
│   │   └── queue_manager.py  # Queue management, progress tracking
│   └── utils/
│       ├── ffmpeg.py         # FFmpeg wrapper with NVENC
│       ├── file_watcher.py   # Watch output directory
│       └── logger.py         # Structured logging
├── data/
│   ├── jobs.json             # Nightly job queue
│   ├── settings.json         # User preferences
│   └── projects/             # Per-project state
├── workflows/                # ComfyUI workflow templates
│   ├── broll_generation.json
│   ├── thumbnail.json
│   ├── scene_overlay.json
│   └── upscale.json
├── requirements.txt
├── start_editor.bat
└── README.md
```

---

## IMPLEMENTATION PHASES

### PHASE 1: Core Video Pipeline (Week 1-2)
**Goal: Upload video → Auto-remove fumbles/silence → Export clean cut**

#### 1.1 Backend Setup
- [ ] Create FastAPI backend on port 8099
- [ ] Set up CORS, Pydantic models, project structure
- [ ] Implement FFmpeg wrapper with NVENC hardware encoding
- [ ] File upload and project management endpoints

#### 1.2 Audio Analysis & Transcription
- [ ] Integrate Whisper large-v3 (already installed at C:\Documents\ComfyUI\models\stt\whisper\)
- [ ] Transcribe video → timestamped SRT/JSON
- [ ] Detect silence gaps (>1.5s pause = potential cut)
- [ ] Detect fumbles using speech pattern analysis:
  - Repeated words ("um", "uh", "like", "so", "you know")
  - Sentence fragments followed by restarts
  - Long pauses mid-sentence
  - Self-corrections ("I went to... I mean I drove to")
- [ ] Mark problematic segments with timestamps

#### 1.3 Smart Cutting
- [ ] Auto-cut silence segments (configurable threshold)
- [ ] Auto-cut fumble segments (mark for review)
- [ ] Scene detection using PySceneDetect (cut on visual scene changes)
- [ ] Generate timeline of "good" vs "cut" segments
- [ ] Preview cuts before final export

#### 1.4 Transitions
- [ ] Crossfade between scenes (configurable duration: 0.5s-2s)
- [ ] Zoom-in/zoom-out transitions (Ken Burns effect)
- [ ] Dissolve transitions
- [ ] Auto-select transition based on scene type
- [ ] FFmpeg filter chain for transition composition

#### 1.5 Frontend - Core UI
- [ ] Electron window with React frontend
- [ ] Drag-and-drop video input
- [ ] Timeline visualization (tracks: video, audio, B-roll, captions)
- [ ] Preview player with seek
- [ ] Transcript editor (view/edit extracted transcript)
- [ ] "Auto-Edit" button to trigger full pipeline
- [ ] Export settings (resolution, bitrate, codec)

**Deliverable:** Drop a video in → get back a clean cut with transitions.

---

### PHASE 2: ComfyUI Integration & AI Agents (Week 3-4)
**Goal: AI-generated B-roll, thumbnails, overlays generated from ComfyUI**

#### 2.1 ComfyUI Bridge
- [ ] WebSocket client to connect to ComfyUI (localhost:8188)
- [ ] Load existing workflows from C:\Documents\ComfyUI\user\default\workflows\
- [ ] Modify workflow inputs dynamically (change prompts, swap images)
- [ ] Queue management: submit jobs, track progress, retrieve outputs
- [ ] Handle ComfyUI startup/shutdown gracefully

#### 2.2 B-Roll Generation Agent
- [ ] Analyze transcript → extract key topics/scenes
- [ ] Generate image prompts from transcript segments
- [ ] Queue image generation via ComfyUI:
  - **Flux.1 Dev** for photorealistic B-roll
  - **Z-Image Turbo** for fast stylized frames
  - **Wan2.1 T2V 1.3B** for short video clips (4-8 seconds)
- [ ] Insert generated B-roll at appropriate timeline positions
- [ ] Apply crossfade transitions to B-roll segments

#### 2.3 Thumbnail Agent
- [ ] Extract best frame from video (highest visual quality)
- [ ] Generate YouTube thumbnail via ComfyUI:
  - Face swap / enhancement (ReActor + GFPGAN)
  - Text overlay with title
  - Background blur + vibrant color grading
- [ ] Use existing workflows: `flux_dev_fp8_scaledcheckpoint.json`, `faceswap1.json`

#### 2.4 Caption Agent
- [ ] Generate burned-in captions from transcript
- [ ] Style: position, font, color, background, animation
- [ ] Support multi-language captions
- [ ] FFmpeg drawtext filter for caption rendering

#### 2.5 Edit Agent (Main Orchestrator)
- [ ] LLM-powered agent that makes editing decisions:
  - Which segments to cut
  - Where to insert B-roll
  - What transitions to use
  - Pacing and rhythm decisions
- [ ] Use local LLM (Ollama/LM Studio) or cloud (Gemini)
- [ ] Configurable "aggressiveness" of auto-edits

**Deliverable:** Video goes in → AI generates B-roll, captions, thumbnail → edited video comes out.

---

### PHASE 3: Nightly Automation (Week 5)
**Goal: Queue jobs at night → wake up to finished videos**

#### 3.1 Job Scheduler
- [ ] Cron-based scheduler in Python (APScheduler)
- [ ] Job types:
  - **Full Edit:** Transcribe → cut → B-roll → transitions → export
  - **B-Roll Generation:** Generate images/videos for upcoming scripts
  - **Thumbnail Batch:** Generate thumbnails for queued videos
  - **Upscale Batch:** Upscale existing videos to 4K
- [ ] Job queue persisted to `data/jobs.json`
- [ ] Priority system: urgent jobs first

#### 3.2 Nightly Workflow
- [ ] User queues videos before sleep
- [ ] App runs in background (minimized to tray)
- [ ] Processes jobs sequentially (GPU can only handle 1 at a time)
- [ ] Progress tracking per job
- [ ] Notification on completion (Windows toast notification)
- [ ] Power management: prevent sleep during processing

#### 3.3 Output Management
- [ ] Organized output directory: `output/YYYY-MM-DD/`
- [ ] Each job creates a project folder with:
  - Final video
  - Thumbnail
  - Transcript (SRT)
  - Generated B-roll assets
  - Edit log (what was cut, why)

**Deliverable:** Queue 3 videos at 11 PM → wake up to 3 edited videos at 7 AM.

---

### PHASE 4: Advanced Features (Week 6-7)
**Goal: Polish, multi-video projects, color grading**

#### 4.1 Multi-Source Editing
- [ ] Combine multiple video clips into one project
- [ ] Screen recording + webcam picture-in-picture
- [ ] Green screen keying (BiRefNet background removal)
- [ ] Audio mixing: music, voiceover, sound effects

#### 4.2 Color Grading
- [ ] Auto color correction (white balance, exposure)
- [ ] LUT application (cinematic looks)
- [ ] Scene-matching (consistent color across clips)

#### 4.3 Audio Enhancement
- [ ] Noise reduction (already have models)
- [ ] Voice isolation / dereverb
- [ ] Auto-leveling (consistent volume)
- [ ] Background music ducking (lower music when voice speaks)

#### 4.4 Template System
- [ ] Pre-built editing templates:
  - "YouTube Vlog" style
  - "Tutorial" style
  - "Shorts/Reels" (9:16 vertical)
  - "Podcast" style
- [ ] Each template defines: transitions, captions, B-roll style, color grade

#### 4.5 Filmora Export
- [ ] Generate an XML/DXF timeline file
- [ ] Import into Filmora for manual fine-tuning
- [ ] This bridges our AI pipeline with your existing Filmora workflow

**Deliverable:** Professional-grade edits with templates and Filmora integration.

---

### PHASE 5: Production Hardening (Week 8)
**Goal: Stability, packaging, deployment**

#### 5.1 Error Handling
- [ ] GPU OOM detection → auto-reduce resolution/retry
- [ ] ComfyUI crash recovery → restart server
- [ ] FFmpeg failure handling → resume from last good frame
- [ ] Job retry with exponential backoff

#### 5.2 Performance
- [ ] GPU memory management (clear VRAM between jobs)
- [ ] Parallel processing where possible (transcribe while cutting)
- [ ] Progress estimation (ETA per job)
- [ ] VRAM monitoring dashboard

#### 5.3 Packaging
- [ ] Electron Builder for Windows .exe
- [ ] Auto-update mechanism
- [ ] Installer with dependency check (Python, FFmpeg, ComfyUI)
- [ ] First-run setup wizard (detect ComfyUI, configure paths)

---

## TECHNICAL DECISIONS

### Why FastAPI + Electron (not just Electron)?
- Python ecosystem for video/audio processing is mature (OpenCV, MoviePy, Whisper)
- FFmpeg integration is cleaner from Python
- ComfyUI is a Python service we need to talk to
- Electron handles the UI only (lightweight)

### Why Whisper large-v3 (not faster-whisper)?
- You have large-v3 installed and it gives best accuracy
- Your CPU (i7-14700K) handles it well
- Can switch to faster-whisper for speed if needed

### Why integrate ComfyUI instead of calling models directly?
- You have 32 workflows already configured
- ComfyUI handles VRAM management, model loading
- WebSocket API is well-documented
- No need to reinstantiate model loading logic

### NVENC for encoding
- Your RTX 5060 Ti supports NVENC
- 10x faster than CPU encoding
- Quality is near-identical to x264 at same bitrate

---

## DEPENDENCIES

### Python (backend/requirements.txt)
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
whisper>=1.0.0
opencv-python>=4.8.0
moviepy>=1.0.3
pyscenedetect[opencv]>=0.6.0
numpy>=1.24.0
apscheduler>=3.10.0
websockets>=12.0
requests>=2.31.0
python-multipart>=0.0.6
edge-tts>=6.1.0
soundfile>=0.12.1
pydub>=0.25.1
```

### Node.js (package.json)
```
react: ^18.2.0
react-dom: ^18.2.0
electron: ^28.0.0
electron-builder: ^24.0.0
lucide-react: ^0.294.0
concurrently: ^8.2.2
vite: ^4.5.0
@vitejs/plugin-react: ^4.2.0
typescript: ^5.2.2
```

---

## ESTIMATED TIMELINE

| Phase | Duration | Deliverable |
|-------|----------|------------|
| Phase 1 | Week 1-2 | Auto-cut fumbles + transitions |
| Phase 2 | Week 3-4 | AI B-roll + ComfyUI integration |
| Phase 3 | Week 5 | Nightly automation |
| Phase 4 | Week 6-7 | Templates, color, audio polish |
| Phase 5 | Week 8 | Packaging, stability |

**Total: ~8 weeks for full production-ready app**

---

## STATUS: NOT STARTED
## LAST UPDATED: 2026-08-10

lan: Replace IMPLEMENTATION_PLAN.md with a corrected, comprehensive IMPLEMENTATION2_PLAN.md
Context
IMPLEMENTATION_PLAN.md (dated 2026-08-10) says "STATUS: NOT STARTED". progress.md (2026-08-12) says "ALL PHASES COMPLETE". Neither is true, and the plan's factual premises are wrong.

A three-way audit (backend, environment, frontend) plus an architecture review established ground truth:

The plan's stated facts are wrong.

It routes everything through C:\Documents\ComfyUI. That path does not exist. Real location: B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI. (The bad path originates from a stale entry in .claude/settings.local.json.)
It claims Whisper lives under ComfyUI's models dir. It's at C:\Users\singh\.cache\whisper\large-v3.pt.
It claims 32 saved ComfyUI workflows. There are 2, both stock samples.
It claims ~81 GB free on B:. Actual: 90 GB. Models on disk total ~318 GB.
Dependency list is stale/wrong: whisper>=1.0.0 is an unrelated PyPI package; moviepy, apscheduler, edge-tts are listed but never imported and not installed.
The app is structurally broken, not merely incomplete. Root cause: there is no Edit Decision List. Every feature is an independent FFmpeg chain, so nothing composes.

agents/edit_agent.py:61-62 — B-roll clips are generated, then the list is discarded. current_video is never updated. The headline AI feature contributes nothing to the output. No compositor.py was ever written.
edit_agent.py is async def but calls blocking sync functions with no to_thread — freezes the FastAPI event loop for the entire job.
routes/transcription.py:64 reloads the 3 GB Whisper model on every request; the singleton in audio_pipeline/transcriber.py is dead code.
Footage is re-encoded 4× (cut → concat → captions → grade), all libx264. NVENC appears once, at exporter.py:79, with qp = 51 - crf → default CRF 18 becomes -qp 33.
routes/rendering.py:117 uses the last transcript segment's end as video duration → all footage after the final spoken word is silently truncated from every render.
PreviewPlayer.tsx:30 reads project.sourceVideo; backend returns source_video → preview never loads. It also assigns a raw Windows path to video.src, and no video-streaming route exists.
Workspace.tsx:16-20 — infinite refetch loop (useEffect keyed on [project] calls setProject), which also silently reverts every transcript edit.
Transcript edits are zustand-only and never reach the backend. "Edit transcript → edit video" is not implemented.
analyzer.py:16 imports scene_detect (real name: scenedetect) → ImportError always; PySceneDetect never runs.
fumble_detector.py:133-136 fabricates word timings by interpolating character index over segment duration; these become real cut points. Its keyword list cuts ordinary words: so, like, well, right, okay, actually.
silence_detector.py:9 never opens the audio (transcript-gap arithmetic). The real RMS algorithm at :52 is never called.
routes/projects.py:14 — projects live in an in-memory dict; the list is empty after restart and most routes 404 on cold cache.
Zero LLM calls exist anywhere, despite the plan promising an "LLM-powered edit agent."
Tests are named run_test(), so pytest collects zero. test_phase3.py writes to the production data/jobs.json.
No git repository exists. No rollback for any of this.
Deliverable: a new IMPLEMENTATION2_PLAN.md at the project root — self-contained and detailed enough that a different model can pick it up cold and execute it. IMPLEMENTATION_PLAN.md and progress.md are superseded, not deleted.

Decisions locked with the user
Decision	Choice
Scope	Full EDL rewrite — timeline.json as single source of truth
LLM	Local LM Studio (127.0.0.1:1234, OpenAI-compatible, confirmed up)
NLE export	Dropped — no OTIO, no Filmora XML in this plan
B-roll	Stills (Z-Image Turbo / Flux.1 Dev) + FFmpeg zoompan Ken Burns
Verified environment (goes into the new doc as fact)
ComfyUI: B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI, running, PID 2492, 127.0.0.1:8188
LM Studio: up on 127.0.0.1:1234. Models incl. google/gemma-4-12b (6.87 GB Q4_K_M), gemma-4-e4b-uncensored (4.97 GB), qwen/qwen3.6-27b (15.41 GB), devstral-small-2-24b. Ollama is not installed.
VRAM is transient, not permanently saturated: measured at 14161 MiB used mid-audit, 1167 MiB used minutes later (14884 MiB free). ComfyUI releases on idle. This is what makes a lease broker viable rather than a hard CPU fallback.
GPU: RTX 5060 Ti, 16311 MiB, driver 595.97
Embedded Python ...\python_embeded\python.exe = 3.11.8 with torch 2.8.0+cu128. Do not install into it — 318 GB ComfyUI install, no git, no rollback.
python on PATH = 3.14.4, no torch/whisper, no cp314 wheels for ctranslate2/onnxruntime → wrong interpreter.
ffmpeg 8.1 gyan full build: h264_nvenc, libx264, zoompan, xfade, ass/libass, sidechaincompress all present.
Whisper large-v3.pt already cached at C:\Users\singh\.cache\whisper\ (2.9 GB).
Structure of IMPLEMENTATION2_PLAN.md
Status & corrections — what's actually built vs. claimed; the corrected inventory table; explicit list of every false claim in progress.md.
Defect register — every confirmed defect with file:line, severity, and the phase that fixes it. This is the hand-off spine.
Target architecture — EDL as single source of truth; the data flow import → transcribe → VAD → build → ops → compile → render.
timeline.json schema — full literal schema: integer frames on a rational rate, sources[], words[] with state, tracks[] (V1/A1 ripple, V2/A2/CAP anchored), items with enabled (non-destructive removal), locked, origin, anchor_word_id. Plus the op set and the base_revision → 409 concurrency contract.
Subsystem specs — one section each, with file paths and function signatures:
backend/timeline/ — timebase, schema, ops, wordmap, builder
backend/render/ — compiler (pure fn → single filter_complex), encoders (NVENC CQ table, x264 fallback), runner (-progress pipe:1 parsing), captions (ASS, cwd trick for Windows paths)
backend/asr/ — faster_whisper_engine (singleton, int8_float16, real word timestamps), vad (Silero ONNX + silencedetect fallback), disfluency (replaces fumble_detector; deletes the char-interpolation fabrication)
backend/runtime/gpu_broker.py — leases across three tenants: ComfyUI, faster-whisper, LM Studio; nvidia-ml-py for free VRAM; ComfyUI /free escalation gated on /queue being empty; LM Studio TTL auto-unload
backend/llm/ — LM Studio client, response_format: json_schema, frozen prompts, mock.py, fail-safe = keep everything
backend/store/ — project_store (atomic file-backed, kills the in-memory dict), job_store (+ SSE)
backend/routes/media.py — HTTP Range streaming; prerequisite for any working preview 5b. Model selection for LM Studio — google/gemma-4-12b default (6.87 GB, coexists with faster-whisper's 2.6 GB inside 14.9 GB), gemma-4-e4b low-VRAM fallback, explicit "not Devstral" (coding model), explicit "not qwen3.6-27b" (15.4 GB, cannot coexist).
LLM remit — what it decides (retake selection, filler adjudication, chapters, B-roll intent) and the hard boundary: it never emits a timestamp; decisions are keyed by word_id and converted to frames by deterministic code, then snapped to VAD silence.
Render pipeline — per-segment -ss/-t inputs, mandatory branch normalization, one concat, overlays for B-roll, single encode. Chunking rule at N>120 segments with stream-copy join so generation loss stays at 1. Concrete encoder flag blocks.
Phased execution — Phase 0 (git init, venv, pytest, CORS) through Phase 5, each with entry/exit criteria and the defect IDs it closes.
Deletion list — every file/function to remove, with justification (SceneAnalyzer, transitions.py, cutter.py, exporter.py, filmora_exporter.py, most of ffmpeg_utils.py, auto_select_transition, remove_frame_background, the duplicate auto-edit endpoint).
Explicitly not building — xfade-by-default, transition variety, scene detection, NLE export, Wan2.2 video B-roll, the scheduler/power subsystem (keep, invest nothing).
Corrected dependencies — requirements.txt rewritten; faster-whisper removes torch entirely (~600 MB venv vs ~5 GB); the ctranslate2 cuBLAS/cuDNN-on-Windows caveat and its CPU degradation path.
Verification — the three load-bearing tests (compiler golden-string, render duration within 1 frame, EDL op idempotence) plus the wordmap parity test between wordmap.py and src/lib/edl.ts.
Phasing that the document will prescribe
Phase 0 (blocking, ~1h): git init + .gitignore + initial commit; uv venv --python 3.11; rewrite requirements.txt; fix electron/main.cjs:127 interpreter resolution (env var → .venv → fail loudly, never fall through to PATH); pytest scaffolding with a tmp_path fixture — test_phase3.py corrupts production data/jobs.json today, fix before running anything; CORS allow_credentials=False.
Phase 1 (foundations): project_store, job_store + SSE, routes/media.py Range streaming, timeline/ + routes/timeline.py + non-destructive migration of the 4 existing projects, render/ + golden tests. Exit: import → render → export works at one generation of loss, with real progress and a live preview.
Phase 2 (analysis correctness): faster-whisper singleton, Silero VAD, disfluency.py, gpu_broker, timeline/builder.py. Removes torch from the dependency graph (routes/system.py imports it at startup today).
Phase 3 (the missing features): backend/llm/ + retake selection + filler adjudication; B-roll fix (broll_agent returns timeline ops — ~20 lines, no compositor.py needed, the compiler is the compositor); ComfyUI /view output resolution + /object_info model discovery + workflows rebuilt on Z-Image Turbo at native resolution.
Phase 4 (frontend): snake_case store (deletes the casing fault line behind 3 defects), [project?.id] dep fix, src/lib/edl.ts virtual player, real waveform, SSE progress, Timeline renders items with drag/trim → ops.
Phase 5 (deferred): scheduler, thumbnails, Shorts, grading, ducking — all become timeline effects and output profiles rather than new endpoints.
Files
Create: B:\youtubeProjects\Buzzcaf Media\BuzzEdit\IMPLEMENTATION2_PLAN.md
Modify: none — this step is documentation only. No code changes, no git init yet; those are Phase 0 of the new plan, executed on a later approval.
Superseded (left in place): IMPLEMENTATION_PLAN.md, progress.md. The new doc opens by stating both are inaccurate and why.
Filename note: user wrote "implimentation2_plan"; using IMPLEMENTATION2_PLAN.md to match the existing correctly-spelled IMPLEMENTATION_PLAN.md. Trivial to rename.

Verification
Documentation-only, so verification is accuracy of claims:

Every file:line reference in the defect register resolves to the code it describes — spot-check ~10 against the actual files.
Every path in the environment table is confirmed by the probes already run (ComfyUI dir, LM Studio :1234 model list, whisper cache, ffmpeg encoders, VRAM readings, Python interpreter inventory).
No claim in the new doc restates an unverified assertion from the old plan or progress.md.
The document is self-contained: a reader with no access to this conversation can execute Phase 0 without asking a question.