# The Presentation Pass ("Director") — Architecture & Implementation Plan

**Status: NOT STARTED.** Written 2026-08-14 against the working tree at that date.

This document is **self-contained**. A model implementing it needs nothing else: every file
path, function signature, JSON schema, LLM prompt and tunable constant it depends on is
reproduced here, along with the traps that have already cost this codebase real debugging
time. Read it start to finish before writing code — §2 (principles) and §11 (traps) decide
most of the design questions that come up later.

Companion document: `AUTO_EDIT_IMPROVEMENT_PLAN.md` covers the *cut* (removing fumbles).
This document covers everything that happens **after** the cut is decided.

---

## 1. What this builds

The user records a talking-head monologue in Hinglish (romanised Hindi mixed with English),
runs the auto-edit, and goes to bed. By morning the software has, unattended:

1. Read the finished transcript and worked out **what the video is about**, topic by topic.
2. Written image and video generation prompts for each topic.
3. Generated those assets through **the user's own ComfyUI workflows** — Z-Image Turbo for
   stills, a user-supplied workflow for video. Both selectable by name; neither hardcoded.
4. Placed them as **full-frame B-roll cutaways** with Ken Burns moves.
5. Added **punch-ins on the speaker's face**, timed to vocal intensity.
6. Popped up **topic text and graphics** at the right moments.
7. Generated captions and a thumbnail.
8. Rendered the result and **saved the whole thing as an editable timeline**.

### The run contract

```
POST /api/scheduler/enqueue
  { "project_id": "...", "job_type": "presentation",
    "start_at": "23:00", "priority": "normal",
    "settings": { ...PresentationSettings... } }
```

The job waits until `start_at`, holds the machine awake, runs, and writes:

```
output/YYYY-MM-DD/<project name>/
    <project>_presented.mp4        the render
    thumbnail.jpg
    presentation_report.json       what it did and why
data/projects/<id>/project.json    timeline with every decision in it, editable
data/projects/<id>/assets/generated/*.png|.mp4 + .json sidecars
```

**The morning test:** the user opens the project and can see, move, retime or delete every
B-roll clip, every zoom and every pop-up in the timeline. Nothing is baked.

---

## 2. Principles — decide arguments with these

**P1. Everything lands in the EDL. Nothing is pre-rendered.**
The existing `broll_agent` bakes a Ken Burns move into an intermediate mp4 with ffmpeg and
then places that mp4. That throws away editability, costs an extra encode, and hardcodes
30fps/1920x1080. The compiler already animates stills natively (`_retime_still`,
`build_overlay_transform`). **Generated assets go onto the timeline as sources with
`Transform` animation.** One render pass, at the end, of everything.

**P2. The LLM plans; deterministic code places.**
The model returns a *validated JSON plan*. It never writes frame numbers that reach the
timeline unchecked. Every beat is clamped to the program, filtered by density budgets, and
de-duplicated before placement. This is the same trust-limit philosophy that makes the
fluency pass safe (`backend/asr/fluency.py:212 judge_window`): a model that hallucinates
can only fail to produce beats, never corrupt the edit.

**P3. Every stage degrades to something shippable.**
- No LM Studio → deterministic keyword fallback for beats; no pop-ups.
- ComfyUI offline → skip all asset generation; still deliver cut + captions + face zooms.
- One asset fails → skip that beat, log the reason, carry on.
- No face detected → fall back to centre-anchored zooms.
A run must never end with nothing. The report says which stages degraded.

**P4. Cache by content hash.** Re-running a project must not regenerate identical assets.
Key = `sha1(workflow_file + positive_prompt + negative_prompt + seed + width + height)`.

**P5. Determinism.** All randomness is seeded from `sha1(project_id)`. Two runs of the same
project with the same settings produce the same plan and the same placements.

**P6. One GPU, one tenant at a time.** Whisper, LM Studio and ComfyUI all want the same
card. They run in **phases**, never concurrently, each acquiring a `gpu_broker` lease.
ComfyUI currently acquires nothing — that is a Phase 0 fix.

**P7. Nothing overwrites the user's hand edits.** All generated timeline items carry an
`origin` (`"broll"`, `"popup"`, `"autozoom"`, `"caption"`). A re-run clears only items with
its own origin, exactly as `authoring.clear_generated` already does for captions.

---

## 3. Current state — what to build on

### 3.1 Reusable, verified working

| What | Where | Signature / note |
|---|---|---|
| Word → program-time projection | `backend/timeline/authoring.py:32,41` | `_program_segments(timeline) -> List[Tuple[int,int,int]]` (source_start, source_end, timeline_start) and `source_to_timeline_frame(segments, source_frame) -> Optional[int]`. **This is how you map a transcript word to a time in the cut video.** Returns `None` for words that were cut out. |
| Origin-scoped regenerate | `backend/timeline/authoring.py:55` | `clear_generated(timeline, origin) -> int` |
| Caption generation | `backend/timeline/authoring.py:65` | `generate_captions(timeline, preset="classic", track="TC", style_overrides=None, min_duration_sec=0.4) -> List[TimelineItem]`. Constants `CAPTION_TRACK="TC"`, `CAPTION_ORIGIN="caption"`. **The template for pop-ups — copy its structure.** |
| Ken Burns on a still | `backend/render/effects.py:449` `build_overlay_transform`; `backend/render/compiler.py:620,632,665` | `_is_still(item)` → `_retime_still(chain, duration_frames)` rewrites `zoompan :d=1:` → `:d=N:`. A PNG on V3 with `Transform(scale=1.0, scale_end=1.12)` already renders as a push-in. **Works today. Do not rebuild it.** |
| Auto-zoom precedent | `backend/style/apply.py:90` | `apply_motion(timeline, profile, options) -> Dict`. Picks N segments, spreads them deterministically, alternates push-in/pull-back via `(index + seed) % 2`, `Transform(scale=1.0, scale_end=1.0+depth)`, depth clamped 0.04–0.6, skips segments < 0.5s. **Copy this shape for the baseline zoom rhythm.** |
| V1 segment list | `backend/style/apply.py:45` | `_program_segments(timeline) -> List[TimelineItem]` — **note: same name, different return type** from the one in `authoring.py`. Filters `track=="V1" and enabled and kind=="media"`, sorted by start. |
| Per-time loudness | `Timeline.energy_envelope` (`backend/timeline/schema.py:362`) | `{"rate": 50.0, "db": [ints]}` — one integer per 20ms, produced by `backend/asr/vad.py` during the auto-edit and persisted with every project. Consumed today **only** by `ops._snap_cut`. This is the intensity curve; it is free. |
| Adjustment layers | `TimelineItem(kind="adjustment")`, `backend/render/compiler.py:538 _apply_adjustment` | The one mechanism that applies a **time-windowed animated zoom** to the program without splitting V1. Splits the composited picture, transforms one copy with `frame_offset` + `clip()` progress, overlays it back with `enable='between(t,a,b)'`. Created by `clip_ops.add_adjustment_item(timeline, timeline_start_frame, duration_frames, track=None, preset=None)` — refuses V1 and non-V tracks. |
| JSON-mode LLM call | `backend/llm/client.py:298` | `async _chat_json(model, system_prompt, user_prompt) -> Optional[str]` — tries `response_format: json_schema`, falls back to plain; parse with `_first_json_object` which tolerates prose and ```json fences. `lm_launcher.ensure_ready(base_url, model_name) -> Optional[str]` returns **the model id to use**. |
| GPU leases | `backend/runtime/gpu_broker.py:62,77` | `async acquire_lease(tenant_name, required_vram_mb=4000.0) -> None` / `async release_lease(tenant_name) -> None`. Singleton `gpu_broker`. |
| Job queue | `backend/agents/scheduler.py` | `JobScheduler`, `JOBS_FILE = DATA_DIR/"jobs.json"`, `enqueue_job(project_id, job_type="full_edit", priority="normal", settings=None)`, `_worker_loop`, `_process_job`. Crash recovery resets `running`→`pending`. `backend/utils/power.py PowerManager.prevent_sleep/restore_sleep`. |
| Media import & placement | `backend/store/media_pool.py`; `backend/timeline/clip_ops.py:149` | `media_kind(path)`, `probe(path, kind)` (images get `duration=5.0`), `add_media_item(timeline, source_id, track, timeline_start_frame, source_start_frame, source_end_frame, origin="manual") -> TimelineItem`. |
| Text | `backend/timeline/clip_ops.py:422,433`; `backend/render/text.py:108` | `build_text_clip(content, preset=None, style=None, current=None) -> TextClip`, `add_text_item(timeline, content, timeline_start_frame, duration_frames, track=None, preset=None, style=None)` — **hard-refuses any track not starting with "T"**. `build_drawtext` animations: `none|fade|pop|slide-up`. Presets in `backend/timeline/presets.py:72` incl. `youtube_pop`, `lower_third`. |
| Frame sampling | `backend/style/frames.py` | `sample(path, width, height, fps, gray=True, max_frames=6000, duration=None) -> (np.ndarray, float)`. Decodes the whole file **once** through an ffmpeg rawvideo pipe. The codebase settled on this after finding OpenCV per-frame seeking "punishingly slow" — use it, do not write a seek loop. |
| Vision deps already present | `backend/requirements.txt` | `opencv-python>=4.8.0`, `onnxruntime>=1.16.0`, `numpy`, `pillow`. **No new dependency is needed for face detection.** |

### 3.2 Broken or stubbed — these block the pass

| # | Where | What is wrong |
|---|---|---|
| B1 | `backend/agents/edit_agent.py` `execute_full_auto_edit` | The timeline is built into a local `tl` and **never saved to the project store**. An overnight run produces an mp4 and leaves the project on disk with no timeline — nothing to review or re-render. |
| B2 | same | `burn_captions` is an accepted parameter that is never used; `self.caption_agent` is constructed and never called. The full auto edit produces **zero captions**. |
| B3 | same | `render_timeline_async` accepts `progress_callback` and is not given one. |
| B4 | `backend/agents/scheduler.py` `_process_job` | Creates a dated output dir `OUTPUT_DIR/YYYY-MM-DD/<name>/`, reports it as `result.output_directory`, and **writes nothing into it** (EditAgent writes to `OUTPUT_DIR/{id}_edited.mp4`). |
| B5 | same | `job["progress"]` never moves between 0.0 and 1.0. |
| B6 | same | `job_type` is stored and **never branched on** — every job runs `execute_full_auto_edit`. |
| B7 | `backend/comfyui_bridge/client.py:66` | `wait_for_prompt` reads only `node_output['images']`. AnimateDiff / SVD / WanVideo / `VHS_VideoCombine` emit `gifs`, `videos` or `animated`. **Every video workflow returns an empty list today** and falls through to the grey fallback card. |
| B8 | `backend/comfyui_bridge/client.py:73` | Output paths are guessed from `COMFYUI_OUTPUT_DIR`. No `/view?filename=` HTTP fallback — if ComfyUI writes elsewhere, the file "doesn't exist". |
| B9 | `backend/comfyui_bridge/client.py:47` | `get_history` swallows every exception into `{}`, so a dead or restarted ComfyUI burns the full timeout instead of failing fast. |
| B10 | `backend/comfyui_bridge/workflow_loader.py` | Parameterises by iterating `class_type` with hardcoded node ids `"6"`/`"7"` for positive/negative prompt. No steps/cfg/sampler/checkpoint control. Breaks on any workflow whose prompt node isn't id 6. |
| B11 | `backend/comfyui_bridge/queue_manager.py` + `runtime/gpu_broker.py` | ComfyUI **never acquires a GPU lease** (Whisper and LM Studio both do). `ComfyUIQueueManager`'s `asyncio.Lock` is per-instance, and `routes/agents.py` and `agents/scheduler.py` each build their own — so submissions are not serialised across the manual route and the nightly worker. |
| B12 | `backend/agents/broll_agent.py:16,71` | Keyword picking is a fixed word-stride + stop-list, no LLM. Ken Burns is baked into an mp4 with hardcoded 30fps/1080p. `max_clips=2`. **Superseded entirely by `presentation/placement.py`.** |
| B13 | `workflows/*.json` | Both bundled workflows are SD1.5 text-to-image at **1280×720** — below the 1920×1080 canvas, so any push-in goes soft. |
| B14 | `backend/routes/scheduler.py:19` | Deprecated `@router.on_event("startup")`. Works under the current FastAPI's merged-lifespan shim but is fragile. |

---

## 4. Architecture

### 4.1 Package layout

```
backend/presentation/
    __init__.py          public: run_presentation_pass, PresentationSettings
    program.py    Stage A  the cut video as a timeline of words + intensity
    shotplan.py   Stage B  LLM → validated ShotPlan
    workflows.py  Stage C1 workflow registry + binding resolution
    assets.py     Stage C2 ComfyUI generation + cache
    placement.py  Stage D  B-roll cutaways, pop-ups, graphics onto the EDL
    facezoom.py   Stage E  face detection + intensity-driven punch-ins
    director.py            the orchestrator
    models.py              dataclasses/pydantic for every contract in §5
backend/data/models/
    face_detection_yunet_2023mar.onnx    (downloaded once, ~340KB)
workflows/
    manifest.json         role → workflow file + bindings
```

### 4.2 Data flow

```
project.json (timeline with words + energy_envelope, post auto-edit)
        │
   [A] program.py ──────────► Program{words[], segments[], duration_s}
        │                       each ProgramWord: text, tl_start_s, tl_end_s,
        │                       db_mean, emphasis_z, rate_wps
        │
   [B] shotplan.py ─── LM ──► ShotPlan{beats[]}  (validated, clamped, budgeted)
        │                                        GPU lease: "lm_studio"
   [C] workflows.py ─────────► ResolvedWorkflow{file, bindings}
       assets.py ─── ComfyUI ► Asset{beat_id, kind, path, w, h, duration_s}
        │                                        GPU lease: "comfyui"
   [D] placement.py ────────► timeline: V3 media items (origin="broll")
        │                               TP text items (origin="popup")
   [E] facezoom.py ─── cv2 ──► timeline: V1 Transform (origin-tagged via anchor)
        │                               V2 adjustment items (origin="autozoom")
        │
   captions (authoring.generate_captions) → TC
   thumbnail (agents/thumbnail_agent + LLM title)
        │
   render_timeline_async ──► mp4     +  project store save  +  report json
```

### 4.3 Track assignment — and why

| Track | Contents | Origin tag |
|---|---|---|
| `V1` | the programme (auto-edit output) | `auto` |
| `V2` | **adjustment layers** carrying windowed face punch-ins | `autozoom` |
| `V3` | **B-roll cutaways** (full-frame images and video) | `broll` |
| `TC` | captions | `caption` |
| `TP` | topic pop-ups | `popup` |

An adjustment layer treats **every track below it and none above**. Punch-ins on V2
therefore scale V1 (the speaker) and leave V3 (the B-roll) alone — which is what you want,
because a cutaway is already a full-frame composed image and zooming it on top of the
speaker's zoom would double up. Text tracks are drawn onto the finished picture regardless
of numbering, so `TP` and `TC` are independent of all of this.

---

## 5. Data contracts

All of these live in `backend/presentation/models.py` as pydantic models (the codebase uses
pydantic v2 throughout; `model_dump()` / `model_validate()`).

### 5.1 `Program` (Stage A output)

```python
class ProgramWord(BaseModel):
    text: str
    tl_start_s: float      # seconds in the CUT video, not the source
    tl_end_s: float
    db_mean: float         # mean dB over the word, from energy_envelope
    emphasis_z: float      # z-score of db_mean across the whole program
    rate_wps: float        # local speaking rate, words/sec over a 3s window

class ProgramSegment(BaseModel):
    item_id: str           # the V1 TimelineItem id
    tl_start_s: float
    tl_end_s: float
    anchor_word_id: Optional[str]

class Program(BaseModel):
    duration_s: float
    words: List[ProgramWord]
    segments: List[ProgramSegment]
    fps: float
    def text_between(self, a: float, b: float) -> str: ...
    def intensity_at(self, t: float) -> float: ...   # smoothed db z-score
```

### 5.2 `ShotPlan` (Stage B output) — the LLM contract

```jsonc
{
  "beats": [
    {
      "id": "b03",                      // assigned by us, not the model
      "start_s": 42.5,
      "end_s": 48.0,
      "topic": "Cornwall University experiment",
      "summary": "A psychology study where students wore an embarrassing t-shirt.",
      "kind": "broll_image",            // broll_image | broll_video | popup | graphic
      "priority": 0.8,                  // 0..1, how much this deserves screen time
      "image_prompt": "A 1970s university psychology laboratory, students seated at desks, warm documentary lighting, 35mm film still, shallow depth of field",
      "video_prompt": null,             // required when kind == broll_video
      "negative_prompt": "text, watermark, logo, deformed hands, blurry",
      "popup_text": "The Spotlight Effect",   // required when kind == popup
      "style_hint": "photoreal"         // photoreal | illustration | diagram | abstract
    }
  ]
}
```

**Validation performed on every beat before it is trusted** (`shotplan.validate_plan`):

1. `start_s`/`end_s` clamped to `[0, program.duration_s]`; beat dropped if `end_s - start_s < MIN_BEAT_SECONDS`.
2. Beat dropped if its span contains no surviving speech.
3. `kind` must be one of the four; unknown → dropped.
4. Required field per kind missing → dropped (`image_prompt` for `broll_image`, `video_prompt` for `broll_video`, `popup_text` for `popup`).
5. `popup_text` longer than `MAX_POPUP_WORDS` → truncated at a word boundary.
6. Prompts longer than 400 chars → truncated at a word boundary.
7. **Density budget**: beats sorted by `priority` descending; accepted greedily while
   `cutaway_count < ceil(duration_min * MAX_BROLL_PER_MINUTE)` and the beat's start is
   `>= MIN_CUTAWAY_GAP_SECONDS` after the last accepted cutaway's end. Pop-ups budgeted
   separately with `MIN_POPUP_GAP_SECONDS`.
8. **Dedup**: two beats whose `topic` normalises to a token-set Jaccard ≥ 0.6 → keep the
   higher priority one.
9. Beats re-sorted by `start_s` and assigned ids `b00, b01, …`.

`validate_plan` returns `(kept: List[Beat], dropped: List[Tuple[Beat, str]])` — the reasons
go into the report. **A model that returns garbage yields an empty plan, never a bad edit.**

### 5.3 `workflows/manifest.json` — the customisation surface

```jsonc
{
  "version": 1,
  "roles": {
    "broll_image": { "file": "zimage_turbo.json" },
    "broll_video": {
      "file": "wan_video_t2v.json",
      "bindings": {
        "positive": ["6", "inputs", "text"],
        "negative": ["7", "inputs", "text"],
        "width":    ["5", "inputs", "width"],
        "height":   ["5", "inputs", "height"],
        "seed":     ["3", "inputs", "seed"],
        "steps":    ["3", "inputs", "steps"],
        "length":   ["5", "inputs", "length"],
        "fps":      ["9", "inputs", "frame_rate"],
        "prefix":   ["9", "inputs", "filename_prefix"]
      }
    },
    "thumbnail": { "file": "thumbnail.json" },
    "graphic":   { "file": "zimage_turbo.json" }
  }
}
```

- A binding is a **path into the workflow JSON**: `[node_id, "inputs", input_name]`.
- **`bindings` is optional.** When absent, `guess_bindings(workflow)` infers them (§6.1).
  This is what lets the user drop in any workflow and have it work.
- The manifest is optional too; `AppSettings` keys override it per role.

**Settings keys** (`backend/store/app_settings.py`, free-form KV, already has a
`GET/PUT /api/settings` route):

| Key | Default | Meaning |
|---|---|---|
| `workflow_broll_image` | `"broll_generation.json"` | file name in `workflows/` |
| `workflow_broll_video` | `null` | when null, `broll_video` beats degrade to `broll_image` |
| `workflow_thumbnail` | `"thumbnail.json"` | |
| `workflow_graphic` | `null` | when null, no graphic beats |
| `gen_image_size` | `[1920, 1080]` | ≥ canvas, for Ken Burns headroom |
| `gen_video_length` | `81` | frames the video workflow is asked for |
| `gen_video_fps` | `16` | |
| `gen_image_timeout_s` | `300` | |
| `gen_video_timeout_s` | `1800` | |

### 5.4 `Asset` (Stage C output)

```python
class Asset(BaseModel):
    beat_id: str
    kind: Literal["image", "video"]
    path: str                # absolute, under project_dir/assets/generated/
    width: int
    height: int
    duration_s: float        # 0.0 for images
    cache_hit: bool
    workflow_file: str
    seed: int
    prompt_sha: str
```

Sidecar `<sha>.json` written next to each asset with the full prompt, workflow file, seed
and timestamp, so the cache is auditable and a stale asset can be traced to its prompt.

### 5.5 `PresentationReport`

```python
class StageTiming(BaseModel):
    stage: str; seconds: float; ok: bool; note: str = ""

class PresentationReport(BaseModel):
    project_id: str
    started_at: str; finished_at: str
    settings: Dict[str, Any]
    program: Dict[str, Any]          # {duration_s, word_count, segment_count}
    beats_planned: int
    beats_dropped: List[Dict[str, str]]      # [{topic, reason}]
    assets_generated: int; assets_cached: int
    assets_failed: List[Dict[str, str]]      # [{beat_id, reason}]
    broll_placed: int; popups_placed: int; graphics_placed: int
    zooms_segment: int; zooms_windowed: int
    faces_detected_pct: float
    captions: int
    thumbnail: Optional[str]
    output_path: Optional[str]
    degraded: List[str]              # e.g. ["llm_unavailable", "comfyui_offline"]
    timings: List[StageTiming]
```

Saved to `data/projects/<id>/presentation_report.json` **and** returned in the job result.

### 5.6 `PresentationSettings` (job settings)

```python
class PresentationSettings(BaseModel):
    broll: bool = True
    broll_video: bool = True          # allow video beats when a workflow is configured
    popups: bool = True
    graphics: bool = False            # off by default — see the alpha trap, §11
    face_zoom: bool = True
    captions: bool = True
    caption_preset: str = "youtube_shorts"
    thumbnail: bool = True
    max_broll_per_minute: float = 3.0
    popup_preset: str = "youtube_pop"
    zoom_depth: float = 0.10
    seed: Optional[int] = None        # None → sha1(project_id)
```

### 5.7 New / changed routes

| Route | Body / returns |
|---|---|
| `GET /api/comfyui/workflows` | `{"workflows": [{"file", "roles_ok": {...}, "bindings": {...}, "detected": bool, "nodes": int}]}` — lists every `.json` in `workflows/`, with auto-detected bindings and which roles it can serve. Drives the settings dropdowns. |
| `POST /api/presentation/{project_id}/run` | body `PresentationSettings`; runs in the foreground for testing. Returns `PresentationReport`. |
| `POST /api/scheduler/enqueue` | **extended**: `EnqueueJobRequest` gains `settings: Dict[str, Any] = {}` and `start_at: Optional[str] = None` (`"HH:MM"` local, or ISO 8601). |

---

## 6. The LLM layer

Two calls, not one. A single giant "segment and write prompts" call is unreliable on a
26B local model; two focused calls each with a worked example are markedly better. Both go
through `lm_studio_client._chat_json` (json_schema with plain-text fallback) and are parsed
with `_first_json_object`.

**Model selection is already solved**: `lm_launcher.ensure_ready(base_url, model_name)`
returns the id of a model that is *actually loaded*, loading the preferred one first. The
current preference is `gemma-4-26b-a4b-it-ultra-uncensored-heretic` (`llm/client.py:59`),
overridable via `AppSettings["llm_model"]`. **Model size decides quality here as much as it
does in the fluency pass** — a 7B will produce generic prompts and mush the topics.

### 6.1 Call 1 — topic segmentation

Input: the surviving transcript, windowed to 1200 words with 150 words of overlap, each
line prefixed with its timestamp in seconds so the model can return times.

```python
TOPIC_SYSTEM = (
    "You are the researcher for a YouTube video. You are given the transcript of a "
    "finished edit, with a timestamp in seconds at the start of each line.\n\n"
    "The speaker talks in Hindi written in Latin letters (Hinglish), mixing in English "
    "words as Hindi speakers do. A machine wrote this transcript down and romanises "
    "Hindi badly — 'jaz' means judge, 'sabsakraaib' means subscribe, 'phinamaanaa' means "
    "phenomenon, 'ooniversitee' means university. Read past the spelling; the speaker "
    "said these words perfectly.\n\n"
    "Split the transcript into the TOPICS it actually covers. A topic is a thing the "
    "speaker is talking about for a stretch of time — a story, a study, an example, a "
    "piece of advice. Not every sentence starts a new topic; a topic usually runs for "
    "15 to 90 seconds.\n\n"
    "For each topic give:\n"
    "  start_s, end_s  - when it runs, in seconds, from the timestamps given\n"
    "  topic           - three to six words naming it, IN ENGLISH\n"
    "  summary         - one sentence in ENGLISH saying what the speaker says about it\n"
    "  visual          - one sentence in ENGLISH describing a single image that would "
    "illustrate it for a viewer. Describe a SCENE, not a concept: 'a student sitting "
    "alone in a lecture hall wearing a bright yellow t-shirt', not 'embarrassment'.\n"
    "  priority        - 0.0 to 1.0, how much this topic would benefit from a picture. "
    "A concrete story or example is high. A general statement or an aside is low.\n\n"
    "RULES:\n"
    "1. Times must come from the timestamps in the transcript and must not overlap.\n"
    "2. Everything you write in topic, summary and visual must be in ENGLISH, however "
    "the transcript is written.\n"
    "3. Never invent a topic the speaker does not discuss.\n"
    "4. Answer with JSON only, no commentary:\n"
    '   {"topics": [{"start_s": 0.0, "end_s": 0.0, "topic": "", "summary": "", '
    '"visual": "", "priority": 0.0}]}'
)
```

User message:

```python
def topic_user_prompt(lines: str) -> str:
    return f"Transcript:\n{lines}\n\nTopics:"
```

where `lines` is `"\n".join(f"[{t:.1f}] {sentence}")`, sentences grouped by the same
2.5s-pause rule `verify.split_sentences` uses.

### 6.2 Call 2 — beat and prompt writing

Input: the accepted topics from call 1, in batches of 8.

```python
BEAT_SYSTEM = (
    "You are the art director for a YouTube video. You are given a list of topics from "
    "the video, each with a summary and a suggested visual. For each topic you decide "
    "what appears on screen and write the prompt that will generate it.\n\n"
    "You are writing prompts for a text-to-image model. It does not know what the video "
    "is about. It only sees your prompt.\n\n"
    "For each topic return one beat:\n"
    "  kind        - 'broll_image' for a still, 'broll_video' for a moving shot, or "
    "'popup' when the point is better made by a few words on screen than by a picture.\n"
    "  image_prompt - for broll_image. Describe the SHOT: subject, setting, lighting, "
    "lens or medium. 20 to 40 words. English only. No text, no words, no logos in the "
    "image — the model renders them as gibberish.\n"
    "  video_prompt - for broll_video. As above, plus ONE simple camera or subject "
    "motion ('slow push in', 'leaves drifting past'). Nothing complex; short clips "
    "cannot carry a complicated action.\n"
    "  popup_text   - for popup. AT MOST SIX WORDS that name the idea. English, or the "
    "speaker's own English term if they used one. No punctuation at the end.\n"
    "  negative_prompt - what must not appear. Always include 'text, watermark, logo'.\n"
    "  style_hint   - 'photoreal', 'illustration', 'diagram' or 'abstract'.\n\n"
    "RULES:\n"
    "1. Prefer 'broll_image'. Use 'broll_video' only when motion is the point of the "
    "shot. Use 'popup' for abstract ideas, numbers, names and definitions, which "
    "generated pictures render badly.\n"
    "2. Never put readable text in an image prompt.\n"
    "3. Never describe a real identifiable person.\n"
    "4. Keep every prompt under 40 words.\n"
    "5. Answer with JSON only, no commentary:\n"
    '   {"beats": [{"topic": "", "kind": "", "image_prompt": null, '
    '"video_prompt": null, "popup_text": null, "negative_prompt": "", '
    '"style_hint": ""}]}\n\n'
    "WORKED EXAMPLE:\n"
    "Topic: Cornwall University experiment | A psychology study where students had to "
    "wear an embarrassing t-shirt and walk into a full room.\n"
    "Beat: {\"topic\": \"Cornwall University experiment\", \"kind\": \"broll_image\", "
    "\"image_prompt\": \"A young student pausing in the doorway of a crowded university "
    "lecture hall, wearing a bright yellow t-shirt, other students seated and looking "
    "down at notes, warm afternoon light through tall windows, 35mm documentary "
    "photograph, shallow depth of field\", \"video_prompt\": null, \"popup_text\": null, "
    "\"negative_prompt\": \"text, watermark, logo, deformed hands, extra limbs\", "
    "\"style_hint\": \"photoreal\"}\n\n"
    "Note what happened: the abstract idea (embarrassment) became a concrete scene a "
    "camera could photograph, and nothing in the image needs to be read."
)
```

### 6.3 Call 3 — thumbnail title (small, optional)

```python
THUMBNAIL_TITLE_SYSTEM = (
    "Write a YouTube thumbnail title for this video: at most five words, in the "
    "language the speaker is using, in capitals, no punctuation. It must promise what "
    "the video actually delivers — never invent a claim the transcript does not make. "
    "Answer with the title only."
)
```

### 6.4 Deterministic fallback (`shotplan.fallback_plan`)

When `ensure_ready` returns `None` or every window fails:

- Split the program into `ceil(duration_min * 2)` even windows.
- In each, take the highest-`emphasis_z` run of 5 consecutive words, drop stop-words and
  tokens ≤ 3 chars (reuse the stop list at `broll_agent.py:20`).
- `image_prompt = f"Cinematic photograph representing {' '.join(keywords)}, "
  f"documentary lighting, 35mm film still, shallow depth of field"`.
- `priority = normalised emphasis_z`, `kind = "broll_image"`, no pop-ups.
- Report records `degraded: ["llm_unavailable"]`.

This is deliberately worse than the LLM path, and deliberately still shippable.

---

## 7. Stage implementations

### 7.1 Stage A — `program.py`

```python
def build_program(timeline: Timeline) -> Program
```

1. `segments = authoring._program_segments(timeline)` — the source→timeline mapping.
2. For each **enabled** `WordItem`, `tl = authoring.source_to_timeline_frame(segments, w.start_frame)`.
   `None` means the word was cut; skip it.
3. Loudness join (**this join does not exist anywhere yet — it is new work**):
   `env = timeline.energy_envelope`; `rate = env["rate"]` (50.0), `db = env["db"]`.
   The envelope is indexed in **source** time, the word's `start_frame` is a **source**
   frame, so: `i0 = int(w.start_frame / fps * rate)`, `i1 = int(w.end_frame / fps * rate)`,
   `db_mean = mean(db[i0:i1])`. Guard empty slices; missing envelope → `db_mean = 0.0` for
   every word and `emphasis_z = 0.0` (face zoom then falls back to even spacing).
4. `emphasis_z = (db_mean - mean) / std` over all words, std guarded against 0.
5. `rate_wps`: words whose `tl_start_s` falls in a centred 3s window, divided by 3.
6. `intensity_at(t)`: `emphasis_z` of the nearest word, smoothed with a 5-word Hann window.

**Test**: a synthetic timeline with a known envelope produces the expected z-scores; a
timeline with no envelope produces a Program with zero emphasis and does not raise.

### 7.2 Stage B — `shotplan.py`

```python
async def plan_shots(program: Program, settings: PresentationSettings,
                     ask_json) -> Tuple[ShotPlan, List[str]]   # (plan, degraded_flags)
def validate_plan(beats: List[Beat], program: Program,
                  settings: PresentationSettings) -> Tuple[List[Beat], List[Tuple[Beat,str]]]
def fallback_plan(program: Program, settings: PresentationSettings) -> List[Beat]
```

`ask_json` is injected (`lambda sys, usr: lm_studio_client._chat_json(model, sys, usr)`) so
the planner is testable with a stand-in, exactly as `fluency.plan_fluent_cuts` takes `ask`.

Windowing, both calls, mirrors `fluency.windows`: 1200-word windows with 150-word overlap,
verdicts written only for the core.

### 7.3 Stage C1 — `workflows.py`

```python
def list_workflows() -> List[Dict[str, Any]]          # for GET /api/comfyui/workflows
def load_manifest() -> Dict[str, Any]
def guess_bindings(workflow: Dict[str, Any]) -> Dict[str, List[str]]
def resolve(role: str) -> Optional[ResolvedWorkflow]  # settings > manifest > default
def apply_bindings(workflow: Dict, bindings: Dict, values: Dict) -> Dict  # deep-copies
```

**`guess_bindings` heuristics** — this is what makes "just drop in your workflow" work:

| Binding | How it is found |
|---|---|
| `positive` / `negative` | All `CLIPTextEncode` nodes (also `CLIPTextEncodeSDXL`, `T5TextEncode`). If exactly two: the one referenced by the sampler's `positive` input is positive, the other negative. If that link cannot be traced, the one with the **longer default text** is positive (negatives are short keyword lists). If exactly one: it is positive. |
| `width`/`height`/`length` | First node whose `class_type` contains `EmptyLatent` (covers `EmptyLatentImage`, `EmptySD3LatentImage`, `EmptyHunyuanLatentVideo`, `WanImageToVideo`) and whose `inputs` carry those keys. `length` present ⇒ the workflow is **video-capable**. |
| `seed` / `steps` / `cfg` | First node with a `seed` or `noise_seed` input (`KSampler`, `KSamplerAdvanced`, `RandomNoise`, `SamplerCustom*`). |
| `prefix` | `SaveImage.filename_prefix`, or `VHS_VideoCombine.filename_prefix`, or `SaveAnimatedWEBP`. |
| `output_kind` | `"video"` if any node is `VHS_VideoCombine` / `SaveAnimatedWEBP` / `SaveAnimatedPNG` or a `length` binding was found; else `"image"`. |

Manifest bindings **override** guessed ones key by key, so a user can correct one binding
without specifying all of them. `apply_bindings` skips any value whose binding path does not
resolve, rather than raising — a workflow with no `steps` input simply ignores `steps`.

**Test**: `guess_bindings` on both bundled workflows returns the ids the current
`workflow_loader` hardcodes (`"6"` positive, `"7"` negative, `"5"` latent, `"3"` seed); on a
synthetic WanVideo-shaped JSON it finds `length`, `fps` and `output_kind == "video"`.

### 7.4 Stage C2 — `assets.py`

```python
async def generate_assets(beats: List[Beat], project_dir: Path,
                          settings: PresentationSettings,
                          progress_cb: Callable[[float, str], None]) -> List[Asset]
```

Per beat: resolve role → cache key → return cached `Asset` if the file exists → otherwise
submit and wait.

**The ComfyUI client fixes (B7–B9) are prerequisites and belong in `comfyui_bridge/client.py`, not here:**

```python
# client.wait_for_prompt — replace the images-only read
OUTPUT_KEYS = ("images", "gifs", "videos", "animated")
for node_output in outputs.values():
    for key in OUTPUT_KEYS:
        for entry in node_output.get(key, []):
            path = self._resolve_output(entry)      # NEW
            if path: results.append(path)

def _resolve_output(self, entry: dict) -> Optional[str]:
    """Filesystem path if ComfyUI wrote where we expect, else fetch over HTTP.

    ComfyUI can be configured with an output directory we do not know, and on a
    remote or containerised instance there is no shared filesystem at all. The
    /view endpoint is the only answer that always works.
    """
    candidate = Path(COMFYUI_OUTPUT_DIR) / entry.get("subfolder", "") / entry["filename"]
    if candidate.exists():
        return str(candidate)
    url = (f"{self.server_url}/view?filename={quote(entry['filename'])}"
           f"&subfolder={quote(entry.get('subfolder',''))}&type={entry.get('type','output')}")
    dest = TEMP_DIR / "comfy" / entry["filename"]
    ... download, return str(dest) or None
```

`get_history` must re-raise connection errors (`URLError`/`ConnectionError`) so
`wait_for_prompt` can abort instead of burning the timeout; a 404 while the prompt is still
queued is normal and stays swallowed.

GPU: wrap the whole batch, not each item —

```python
await gpu_broker.acquire_lease("comfyui", required_vram_mb=8000.0)
try:
    ...generate every beat...
finally:
    await gpu_broker.release_lease("comfyui")
```

A single module-level `queue_manager = ComfyUIQueueManager()` in `comfyui_bridge/__init__.py`,
imported by both the routes and the director, fixes B11's second half.

**Sizing**: ask for `settings.gen_image_size` (default 1920×1080), never the workflow's
own default — a 1280×720 generation upscaled into a 1080p push-in is visibly soft.

### 7.5 Stage D — `placement.py`

```python
def place_broll(timeline: Timeline, beats: List[Beat], assets: List[Asset],
                program: Program, settings: PresentationSettings) -> int
def place_popups(timeline: Timeline, beats: List[Beat],
                 program: Program, settings: PresentationSettings) -> int
def place_graphics(...) -> int        # phase 2, off by default
```

**B-roll cutaway placement**, per asset:

1. `clear_generated(timeline, "broll")` first, so a re-run replaces its own work.
2. Register the source: `SourceFile(id=f"src_broll_{beat.id}", path=asset.path,
   duration_seconds=asset.duration_s or 5.0, kind="image"|"video", width, height, fps_*)`.
3. Duration: `clamp(beat.end_s - beat.start_s, BROLL_MIN_SECONDS, BROLL_MAX_SECONDS)`, and
   for video, additionally clamped to `asset.duration_s`.
4. Start: `beat.start_s`, nudged forward to the next word boundary so the cutaway does not
   land mid-syllable (use `program.words` — the same instinct as `_snap_cut`).
5. Spacing: skip if it would start within `MIN_CUTAWAY_GAP_SECONDS` of the previous
   placed cutaway's end. (The plan validator already budgeted, but placement clamps
   durations, so re-check here.)
6. `clip_ops.add_media_item(timeline, src_id, "V3", tl_start_frame, 0, dur_frames,
   origin="broll")`.
7. **Ken Burns** — images only:
   ```python
   depth = BROLL_ZOOM_MIN + rng.random() * (BROLL_ZOOM_MAX - BROLL_ZOOM_MIN)   # 0.06..0.14
   if index % 2 == 0:   scale, scale_end = 1.0, 1.0 + depth        # push in
   else:                scale, scale_end = 1.0 + depth, 1.0        # pull back
   pan = BROLL_PAN * (1 if index % 4 < 2 else -1)                  # 0.0..0.06
   clip_ops.set_transform(timeline, item.id,
       {"scale": scale, "scale_end": scale_end,
        "pos_x": -pan, "pos_x_end": pan})
   ```
   Video assets get **no transform** — they already move.

**Pop-up placement** — implement as `authoring.generate_popups`, next to
`generate_captions`, because it needs the same projection and clamping machinery:

```python
POPUP_TRACK = "TP"
POPUP_ORIGIN = "popup"

def generate_popups(timeline: Timeline, popups: List[Tuple[float, float, str]],
                    preset: str = "youtube_pop",
                    style_overrides: Optional[Dict[str, Any]] = None) -> List[TimelineItem]
```

- Input is already in **timeline seconds** (Stage A did the projection), so this is simpler
  than `generate_captions` — but reuse its **clamping**: a pop-up must not outlive the next
  one, and must not extend past the program end.
- Duration `POPUP_SECONDS` (3.5s), clamped to the beat.
- Style: `text_preset_style(preset)` then overrides `{"pos_y": POPUP_POS_Y,
  "animation": "pop", "animation_duration": 0.25}`. `POPUP_POS_Y = -0.55` puts it in the
  **upper** third — captions live low (`lower_third` is `pos_y=0.62`), and a pop-up over
  the caption band is unreadable.
- Skip a pop-up whose window overlaps a placed B-roll cutaway: a full-frame cutaway plus
  floating text reads as clutter.
- `clip_ops.add_text_item(timeline, text, start_frame, dur_frames, track="TP",
  preset=preset, style=style)` then set `origin="popup"` on the returned item.

### 7.6 Stage E — `facezoom.py`

```python
def detect_faces(source_path: str, program: Program) -> Dict[str, Tuple[float, float]]
    # V1 item_id -> (pos_x, pos_y) in -1..1 canvas coords, median over the segment
def plan_zooms(program: Program, beats: List[Beat],
               settings: PresentationSettings) -> Tuple[List[SegmentZoom], List[WindowZoom]]
def apply_zooms(timeline: Timeline, segment_zooms, window_zooms,
                anchors: Dict[str, Tuple[float,float]]) -> Tuple[int, int]
```

**Face detection.** OpenCV's YuNet, which ships in `cv2` ≥ 4.8 (already a dependency) and
needs only a 340KB ONNX file:

```python
MODEL_URL = ("https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MODEL_PATH = DATA_DIR / "models" / "face_detection_yunet_2023mar.onnx"

def ensure_model() -> Optional[Path]:
    """Download once. Returns None offline — the caller then centres its zooms."""

detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (w, h),
                                     score_threshold=0.7, nms_threshold=0.3, top_k=5)
detector.setInputSize((w, h))
_, faces = detector.detect(bgr_frame)     # faces: Nx15, cols 0..3 = x,y,w,h
```

Sampling: **`style.frames.sample(source_path, width=640, height=360, fps=2.0, gray=False)`**
— one decode of the whole file, not N seeks. The codebase settled on this deliberately
(`frames.py:5-9`). Convert each sampled frame to BGR for the detector.

Per V1 segment: take frames whose source time falls inside it, keep the largest face per
frame, take the **median** centre (median, not mean — a single false positive on a
background object would drag a mean off the speaker). Fewer than
`MIN_FACE_SAMPLES` (3) detections in a segment → no anchor, zoom centred.

Convert to canvas coords: `pos_x = (cx / frame_w) * 2 - 1`, `pos_y = (cy / frame_h) * 2 - 1`,
then **damp toward centre** by `FACE_ANCHOR_PULL` (0.6) — zooming hard to a face near the
frame edge crops the composition badly. Clamp to ±`MAX_ANCHOR_OFFSET` (0.35).

**Two tiers of zoom:**

*Tier 1 — segment zooms.* Baseline rhythm, exactly `apply_motion`'s algorithm: eligible
segments are those ≥ `MIN_ZOOM_SEGMENT_SECONDS` (1.2s); pick every Nth so the count is
`round(duration_min * ZOOMS_PER_MINUTE)`; alternate push-in/pull-back by
`(index + seed) % 2`; `Transform(scale=1.0, scale_end=1.0+depth, pos_x=anchor_x*0.5,
pos_x_end=anchor_x, ...)` so the move *drifts toward* the face rather than starting on it.
Applied via `clip_ops.set_transform`, which permits `origin="auto"` items
(`_require_effectable`, `clip_ops.py:323`) and survives transcript rebuilds through
`anchor_word_id` carry-over (`ops.rebuild_primary_tracks`).

*Tier 2 — windowed punch-ins.* For emphasis peaks **inside** a segment, where a
whole-segment transform cannot reach. Peak selection: local maxima of
`program.intensity_at` above `PUNCH_Z_THRESHOLD` (1.2), plus the start of any beat with
`priority >= 0.8`. Then filter hard:

- window = `[peak - 0.3s, peak + PUNCH_SECONDS]`, `PUNCH_SECONDS` 0.8–2.5s scaled by z;
- **never crosses a V1 cut** — clip to the containing segment, drop if shorter than 0.8s;
- **at least `PUNCH_EDGE_MARGIN` (0.5s) from either end of its segment** — a punch that
  starts on a cut reads as a mistake;
- **never during a placed B-roll cutaway** (invisible, and jarring on the return);
- **at least `MIN_PUNCH_GAP_SECONDS` (6s) apart**;
- cap the total at `MAX_PUNCHES_PER_MINUTE` (1.5).

Applied as an **adjustment layer on V2**:

```python
item = clip_ops.add_adjustment_item(timeline, start_frame, dur_frames, track="V2")
item.origin = "autozoom"
clip_ops.set_transform(timeline, item.id, {
    "scale": 1.0, "scale_end": 1.0 + punch_depth,      # punch_depth <= 0.25
    "pos_x": 0.0, "pos_x_end": anchor_x,
    "pos_y": 0.0, "pos_y_end": anchor_y,
})
```

The compiler windows this with `frame_offset` + `clip()` progress and composites it back
with `enable='between(t,a,b)'` (`compiler._apply_adjustment`), which is precisely the
"zoom that starts and ends mid-shot" primitive needed. **This is the only way to do it
without splitting V1**, and V1 cannot be split — `clip_ops.split_item` calls
`_require_editable`, which refuses `origin=="auto"` items.

### 7.7 The orchestrator — `director.py`

```python
async def run_presentation_pass(project_id: str,
                                settings: PresentationSettings,
                                progress_cb: Callable[[float, str], None] = None
                                ) -> PresentationReport
```

```
 0.00  load project; if no timeline or no words → run the auto-edit first
                     (asr.auto_edit.plan_auto_edit + build_timeline_from_transcript)
 0.10  [A] build_program
 0.15  [B] plan_shots            lease "lm_studio"  → release
 0.30  [C] generate_assets       lease "comfyui"    → release
 0.65  [D] place_broll, place_popups
 0.72  [E] detect_faces, plan_zooms, apply_zooms
 0.80      authoring.generate_captions(timeline, preset=settings.caption_preset)
 0.84      thumbnail (LLM title, lease "lm_studio"; then lease "comfyui")
 0.88      SAVE the timeline to the project store          ← B1, non-negotiable
 0.90      render_timeline_async(tl, out, progress_callback=...)
 1.00      write presentation_report.json; copy render + thumbnail into the job's dated dir
```

Every stage is wrapped so a failure degrades rather than aborts:

```python
async def _stage(name, coro, report, required=False):
    started = time.time()
    try:
        result = await coro
        report.timings.append(StageTiming(stage=name, seconds=time.time()-started, ok=True))
        return result
    except Exception as e:
        logger.exception("Presentation stage %s failed", name)
        report.timings.append(StageTiming(stage=name, seconds=time.time()-started,
                                          ok=False, note=str(e)))
        report.degraded.append(name)
        if required:
            raise
        return None
```

Only `save` and `render` are `required=True`.

### 7.8 Scheduler changes

```python
# routes/scheduler.py
class EnqueueJobRequest(BaseModel):
    project_id: str
    job_type: str = "full_edit"          # "full_edit" | "presentation"
    priority: str = "normal"
    start_at: Optional[str] = None       # "23:00" or ISO 8601
    settings: Dict[str, Any] = {}
    generate_broll: bool = True          # kept for the old job type
    generate_thumbnail: bool = True
    burn_captions: bool = True
```

`enqueue_job` stores `start_at` resolved to an **absolute ISO timestamp** at enqueue time:
a bare `"23:00"` means the next occurrence of 23:00 local (today if still ahead, else
tomorrow). Storing the resolved instant matters — a job queued at 23:30 for `"23:00"` must
run tomorrow, not immediately, and must not shift if the process restarts.

`_worker_loop` picks the first pending job **whose `start_at` is absent or in the past**,
rather than the first pending job. When the head of the queue is waiting, the loop keeps
polling and does **not** hold `prevent_sleep` — the machine should be allowed to idle until
the work actually starts.

`_process_job` branches on `job_type`:

```python
if job["job_type"] == "presentation":
    settings = PresentationSettings(**job.get("settings", {}))
    report = await run_presentation_pass(job["project_id"], settings,
                                         progress_cb=lambda p, m: self._progress(job, p, m))
    job["result"] = {"output_directory": str(job_output_dir),
                     "report": report.model_dump()}
```

`_progress(job, pct, message)` sets `job["progress"]`, `job["message"]`, and calls
`_save_jobs()` — throttled to at most once every 2 seconds so an overnight run does not
rewrite `jobs.json` thousands of times (B5).

The dated output dir is finally used: the director copies the render and thumbnail into it
(B4).

### 7.9 Frontend

- `JobQueue.tsx`: a "Start at" `<input type="time">` (empty = now), a job-type select
  (`Full edit` / `Presentation pass`), and a small settings disclosure with the
  `PresentationSettings` toggles. Show `job.message` under the progress bar.
- `AgentPanel.tsx`: replace the blocking `triggerFullEdit` fetch with an **enqueue** —
  agent runs are 20+ minutes and must not hold an HTTP request open.
- A **Generation** section (in `SetupWizard.tsx` or a new Settings panel) with three
  dropdowns — image / video / thumbnail workflow — populated from
  `GET /api/comfyui/workflows`, each showing whether bindings were detected and whether the
  role is satisfiable, plus a free-text field for a file name not in the list. Writes
  `workflow_broll_image` etc. via `PUT /api/settings`.

---

## 8. Phase 0 — prerequisite fixes

Do these first, in this order. Each is small, each is independently testable, and the pass
cannot work without them.

| # | Fix | File |
|---|---|---|
| 0.1 | Persist the timeline and status at the end of `execute_full_auto_edit` (`project_store.save_project`) | `backend/agents/edit_agent.py` |
| 0.2 | Call `authoring.generate_captions` when `burn_captions` is true; delete the unused `caption_agent` reference or leave it only for the standalone route | `backend/agents/edit_agent.py` |
| 0.3 | Thread `progress_callback` into `render_timeline_async` | `backend/agents/edit_agent.py` |
| 0.4 | `_progress()` helper + throttled `_save_jobs`; wire it through both job types | `backend/agents/scheduler.py` |
| 0.5 | Branch `_process_job` on `job_type`; write outputs into the dated dir | `backend/agents/scheduler.py` |
| 0.6 | `start_at` (resolved to absolute) in `enqueue_job` + `EnqueueJobRequest`; worker skips future jobs and does not inhibit sleep while waiting | `backend/agents/scheduler.py`, `backend/routes/scheduler.py` |
| 0.7 | `wait_for_prompt` reads `images|gifs|videos|animated`; add `_resolve_output` with `/view` fallback; `get_history` re-raises connection errors | `backend/comfyui_bridge/client.py` |
| 0.8 | Module-level singleton `queue_manager`; import it everywhere instead of constructing | `backend/comfyui_bridge/__init__.py`, `routes/agents.py`, `agents/*.py` |
| 0.9 | ComfyUI acquires `gpu_broker` leases around generation | `backend/presentation/assets.py`, `backend/agents/thumbnail_agent.py` |
| 0.10 | Replace `@router.on_event("startup")` with the app lifespan in `main.py` | `backend/routes/scheduler.py`, `backend/main.py` |

`workflow_loader.prepare_broll_workflow` / `prepare_thumbnail_workflow` stay as they are —
`workflows.py` supersedes them, and the old B-roll route can keep using them until
`broll_agent` is retired in P4.

---

## 9. Implementation order

Each phase: build, test, and **verify against a real render** before moving on. That last
part is not optional — this codebase has repeatedly produced filter graphs that compiled to
plausible-looking strings and rendered wrong (see §11).

| Phase | Deliverable | Acceptance |
|---|---|---|
| **P0** | The fixes in §8 | Existing 389 tests still pass. A queued `full_edit` job now leaves a saved timeline with captions, a moving progress bar, and files in the dated dir. |
| **P1** | `models.py`, `program.py` | Unit tests for the word↔envelope join, emphasis z-scores, cut-word exclusion, and the no-envelope path. `build_program` on the real project `data/projects/7322a87c` returns ~250 words with a plausible z-distribution (mean ≈ 0, std ≈ 1). |
| **P2** | `shotplan.py` + both prompts + fallback | Tests with a stand-in `ask_json`: a good plan validates; overlapping beats are clamped; a beat outside speech is dropped; density budget caps cutaways; near-duplicate topics collapse; a translated/garbage answer yields an **empty** plan, not a bad one; the fallback produces beats with no model at all. |
| **P3** | `workflows.py`, `assets.py`, `GET /api/comfyui/workflows`, settings UI | `guess_bindings` matches the hardcoded ids on both bundled workflows and finds `length`/`fps` on a synthetic video workflow. With a **mock ComfyUI fixture** (serves canned `/prompt`, `/history`, `/view`), `generate_assets` produces `Asset`s, second run is 100% cache hits. **Real check:** point at the live ComfyUI with Z-Image Turbo and generate one 1920×1080 image. |
| **P4** | `placement.py` (B-roll + pop-ups), `authoring.generate_popups` | Invariants: no two cutaways overlap; every cutaway ≥ `MIN_CUTAWAY_GAP_SECONDS` apart; every item is on V3 with `origin="broll"`; pop-ups on TP never overlap a cutaway; a re-run replaces rather than duplicates. **Real render:** a 30s excerpt with two generated stills — confirm the cutaways appear full-frame, the Ken Burns move is visible and smooth, and the audio is untouched. |
| **P5** | `facezoom.py` | Unit: anchor maths (frame px → canvas coords, damping, clamping); median-over-segment ignores a single outlier; window filters (never crosses a cut, never inside a cutaway, spacing, per-minute cap). **Real render:** confirm the punch-in lands on the speaker's face and returns cleanly, and that a punch never straddles a cut. |
| **P6** | Video beats end to end | `broll_video` beats generate through the user's video workflow, are placed with **no** transform, and are trimmed to the beat. **Real render** with one generated clip. |
| **P7** | `director.py`, scheduler branch, `start_at`, UI | A `presentation` job queued with `start_at` two minutes out sits waiting (no sleep inhibition), then runs to completion, moves its progress bar, and writes everything in §1's run contract. |
| **P8** | Report + LLM thumbnail title + docs | `presentation_report.json` accounts for every beat: planned, generated/cached, placed or dropped-with-reason. Update `progress.md`. |

---

## 10. Testing

### 10.1 Unit tests (new files under `backend/tests/`)

- `test_program.py` — the envelope join, emphasis, projection of cut words.
- `test_shotplan.py` — validation clamps, density budget, dedup, fallback, garbage answers.
- `test_workflows.py` — `guess_bindings` on both bundled workflows + a synthetic video one;
  manifest override precedence; `apply_bindings` ignores unresolvable paths.
- `test_placement.py` — spacing/overlap invariants, origin tagging, re-run idempotence,
  Ken Burns alternation determinism under a fixed seed.
- `test_facezoom.py` — anchor maths, median robustness, every window filter.
- `test_presentation_director.py` — the degrade paths: no LLM, ComfyUI offline, one asset
  failing. Each must still save a timeline and render.

Follow the house style visible in `test_retakes.py` and `test_chroma_key.py`: **the test
name states the behaviour**, and roughly half the tests assert the thing that must *not*
happen.

### 10.2 Mock ComfyUI fixture

`backend/tests/fixtures/fake_comfyui.py` — a `http.server` thread serving:
`GET /system_stats` → `{}`; `POST /prompt` → `{"prompt_id": "..."}`;
`GET /history/{id}` → outputs referencing a canned PNG/MP4 in `tests/fixtures/`;
`GET /view` → the file bytes. This is what lets P3–P6 run in CI with no GPU.

### 10.3 The overnight acceptance run

On the real 195-second project `data/projects/7322a87c`, with LM Studio and ComfyUI both
up:

1. Enqueue a `presentation` job with `start_at` two minutes ahead.
2. Confirm the worker waits, then starts, and that the progress bar advances.
3. **Morning state must be:**
   - `output/<date>/<name>/` containing the mp4 and `thumbnail.jpg`;
   - the project's saved timeline containing V3 `origin="broll"` items, V2
     `origin="autozoom"` adjustments, `TP` pop-ups and `TC` captions;
   - `presentation_report.json` with `degraded: []` and every beat accounted for;
   - the render playable, with cutaways full-frame, pop-ups readable and not clashing with
     captions, and punch-ins landing on the face.
4. **Record**: wall-clock per stage, peak VRAM, and that the lease handoffs
   (Whisper → LM → ComfyUI) appear in order in the log with no overlap.

**Regression bar**: the 389 existing tests stay green, and the auto-edit's own acceptance
numbers (`AUTO_EDIT_IMPROVEMENT_PLAN.md` §6) are unchanged — the presentation pass must not
alter the cut.

---

## 11. Traps — every one of these has already cost this codebase debugging time

1. **A filter graph can compile to a plausible string and still render wrong.** Every
   visual phase ends with a real ffmpeg render and a look at the frame. This is how the
   chroma keyer was validated and it caught things unit tests could not.

2. **Transparent PNGs and animated zoom do not mix.** An animated `scale` on an overlay
   routes through `build_canvas_transform`, which `pad`s with **black** — a Ken Burns move
   on a transparent graphic composites a black box over the video. Until
   `pad=...:color=black@0` plus an rgba format is implemented and tested, graphics must use
   **static scale with position-only animation** (the overlay `x`/`y` expressions are
   evaluated per frame and are safe). This is why `settings.graphics` defaults to `False`.

3. **Expressions containing commas must be quoted** in a filtergraph — `x='…'`, `y='…'`,
   `enable='…'`. An unquoted `clip(a,b,c)` reads as a filter separator and the whole graph
   fails to parse.

4. **`zoompan` clamps `z` at 1.0** — it can only zoom *in*. Pull-backs work by pre-scaling
   to `min(scale, scale_end, 1.0)` and zooming up from there. `build_canvas_transform`
   already does this; do not "simplify" it.

5. **Animated stills need both** a retimed `zoompan` (`d=<clip frames>`) *and* a
   `setpts=…+start/TB` shift — without the shift the whole move plays at t=0 and only the
   final frame is ever visible.

6. **One V1 transform forces normalisation of every V1 clip** (`compiler.py:190`) because
   concat requires identical geometry. Adding a single punch-in re-scales the whole
   programme. Correct, but not free — expect the render to slow down once zooms are on.

7. **V1 clips cannot be split.** `clip_ops.split_item` → `_require_editable` refuses
   `origin=="auto"`. Windowed zooms must go through adjustment layers. Do not try to work
   around this by rewriting V1 items; `rebuild_primary_tracks` will wipe them.

8. **`rebuild_primary_tracks` wipes V1/A1 and reassigns ids**, carrying transform and
   colour across by `anchor_word_id` only. Anything else keyed to a V1 item id is lost on
   the next transcript edit. Overlay tracks (V2+) are untouched.

9. **Text goes through `textfile=` sidecars, never inline `text=`.** Hinglish and
   Devanagari with apostrophes and colons need two layers of escaping otherwise.
   `render/text.py:write_text_asset` handles it.

10. **Never animate `fontsize`** — it re-rasterises the glyph cache every frame and is
    ruinously slow. The existing `pop` animation rides the `y` expression instead.

11. **Two different `_program_segments`.** `authoring.py:32` returns
    `List[Tuple[int,int,int]]` for source→timeline mapping; `style/apply.py:45` returns
    `List[TimelineItem]` for V1 segments. Import the right one.

12. **`/v1/models` is not a load check** on LM Studio — it lists everything *downloaded*.
    Use `lm_launcher.ensure_ready`, which reads `/api/v0/models` for real state and returns
    the model id to send requests with.

13. **The GPU lease is advisory.** `acquire_lease` nudges ComfyUI to unload and releases
    the lock immediately; it does not hold exclusive access for the tenant's whole
    lifetime. Sequencing the *phases* is what actually prevents contention — do not run
    stages B and C concurrently even though both are async.

14. **Images are 5 seconds by default** in `media_pool.probe`. A `SourceFile` for a still
    needs an explicit `duration_seconds` or its clip silently truncates.

---

## 12. Constants — all in one place

Put these at the top of the module that owns them, each with a one-line comment saying what
it protects against. Values are starting points measured against a 3-minute talking-head
video; they are meant to be tuned with the report in hand.

| Constant | Value | Owner | Why |
|---|---|---|---|
| `MIN_BEAT_SECONDS` | 2.0 | shotplan | Shorter than this and a cutaway is a flash |
| `MAX_BROLL_PER_MINUTE` | 3.0 | shotplan | Above this the video stops being a talking head |
| `MIN_CUTAWAY_GAP_SECONDS` | 8.0 | shotplan/placement | Back-to-back cutaways lose the speaker |
| `MIN_POPUP_GAP_SECONDS` | 12.0 | shotplan | Pop-ups are punctuation, not decoration |
| `MAX_POPUP_WORDS` | 6 | shotplan | Longer cannot be read before it leaves |
| `TOPIC_DEDUP_JACCARD` | 0.6 | shotplan | Two beats on one topic waste a generation |
| `BROLL_MIN_SECONDS` / `MAX` | 3.0 / 6.0 | placement | Under 3s reads as a glitch; over 6s the viewer wonders where you went |
| `BROLL_ZOOM_MIN` / `MAX` | 0.06 / 0.14 | placement | Enough to see; not enough to notice |
| `BROLL_PAN` | 0.04 | placement | |
| `POPUP_SECONDS` | 3.5 | placement | |
| `POPUP_POS_Y` | −0.55 | placement | Upper third — captions own the lower third |
| `ZOOMS_PER_MINUTE` | 4.0 | facezoom | Baseline rhythm |
| `MIN_ZOOM_SEGMENT_SECONDS` | 1.2 | facezoom | Shorter and a move is a jolt |
| `SEGMENT_ZOOM_DEPTH` | 0.10 | facezoom | |
| `PUNCH_Z_THRESHOLD` | 1.2 | facezoom | How loud counts as emphasis |
| `PUNCH_SECONDS` | 0.8–2.5 | facezoom | Scaled by z |
| `PUNCH_DEPTH_MAX` | 0.25 | facezoom | Beyond this a 1080p source visibly softens |
| `MIN_PUNCH_GAP_SECONDS` | 6.0 | facezoom | |
| `MAX_PUNCHES_PER_MINUTE` | 1.5 | facezoom | |
| `PUNCH_EDGE_MARGIN` | 0.5 | facezoom | A punch starting on a cut reads as an error |
| `FACE_ANCHOR_PULL` | 0.6 | facezoom | Damping toward centre |
| `MAX_ANCHOR_OFFSET` | 0.35 | facezoom | Hard clamp on how far off-centre a zoom goes |
| `MIN_FACE_SAMPLES` | 3 | facezoom | Below this, no anchor — centre instead |
| `FACE_SAMPLE_FPS` | 2.0 | facezoom | Enough for a median; cheap |

---

## 13. Definition of done

- A `presentation` job queued with `start_at` runs unattended and produces the run contract
  in §1 — render, thumbnail, **saved editable timeline**, and a report.
- The user can change which ComfyUI workflow is used for images, video and thumbnails
  **from the UI**, by picking a file or typing a name, with no code change.
- With LM Studio off, the pass still delivers a cut with captions and face zooms.
  With ComfyUI off, it still delivers a cut with captions, zooms and pop-ups.
  Both cases are named in `report.degraded`.
- Every generated element is on its own origin-tagged track and can be moved, retimed or
  deleted in the timeline, and a re-run replaces only its own work.
- New unit tests pass alongside the existing 389, and the auto-edit's acceptance numbers
  are unchanged.
