

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
Create: B:\youtubeProjects\Buzzcaf Media\BuzzcafEditor\IMPLEMENTATION2_PLAN.md
Modify: none — this step is documentation only. No code changes, no git init yet; those are Phase 0 of the new plan, executed on a later approval.
Superseded (left in place): IMPLEMENTATION_PLAN.md, progress.md. The new doc opens by stating both are inaccurate and why.
Filename note: user wrote "implimentation2_plan"; using IMPLEMENTATION2_PLAN.md to match the existing correctly-spelled IMPLEMENTATION_PLAN.md. Trivial to rename.

Verification
Documentation-only, so verification is accuracy of claims:

Every file:line reference in the defect register resolves to the code it describes — spot-check ~10 against the actual files.
Every path in the environment table is confirmed by the probes already run (ComfyUI dir, LM Studio :1234 model list, whisper cache, ffmpeg encoders, VRAM readings, Python interpreter inventory).
No claim in the new doc restates an unverified assertion from the old plan or progress.md.
The document is self-contained: a reader with no access to this conversation can execute Phase 0 without asking a question.