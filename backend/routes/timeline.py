import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from store.project_store import ProjectStore
from config import PROJECTS_DIR, OUTPUT_DIR
from utils.ffmpeg_utils import extract_audio, get_video_info
from asr import whisper_engine, detect_speech_silence_intervals
from asr.auto_edit import plan_auto_edit, public_report, record_cut_coverage
from timeline import build_timeline_from_transcript, cut_program_range, rebuild_primary_tracks, toggle_word, Timeline
from timeline.schema import AudioMaster, SourceFile, time_to_frame
from render.audio import EQ_PRESETS, VOICE_FX_RECIPES, _REVERB_RECIPES
from timeline import authoring, clip_ops
from store import media_pool
from render import render_timeline_async
from render import preview

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))


class ToggleWordRequest(BaseModel):
    word_id: str
    enabled: bool


class ProgramCutRequest(BaseModel):
    start_frame: int
    end_frame: int


class SplitRequest(BaseModel):
    item_id: str
    at_frame: int
    linked: bool = True                # split the linked audio/video with it


class SplitManyRequest(BaseModel):
    item_ids: List[str]
    at_frame: int
    linked: bool = True


class DeleteClipRequest(BaseModel):
    item_id: str
    ripple: bool = False               # close the hole it leaves
    linked: bool = True


class DeleteManyRequest(BaseModel):
    item_ids: List[str]
    ripple: bool = False
    linked: bool = True


class MoveClipRequest(BaseModel):
    item_id: str
    timeline_start_frame: int
    track: Optional[str] = None
    linked: bool = True


class MoveManyRequest(BaseModel):
    item_ids: List[str]
    delta_frames: int
    track_map: Dict[str, str] = {}     # item_id -> new lane (same kind)
    linked: bool = True


class TrimClipRequest(BaseModel):
    item_id: str
    edge: str  # "start" | "end"
    timeline_frame: int
    linked: bool = True
    ripple: bool = False


class TrimManyRequest(BaseModel):
    item_ids: List[str]
    edge: str
    at_frame: int
    ripple: bool = False


class LinkRequest(BaseModel):
    item_ids: List[str]


class PasteRequest(BaseModel):
    items: List[Dict[str, Any]]
    at_frame: int
    insert: bool = False


class CloseGapRequest(BaseModel):
    track: str
    at_frame: int


class ReplaceTimelineRequest(BaseModel):
    timeline: Dict[str, Any]


class AddMediaRequest(BaseModel):
    path: str
    track: Optional[str] = None            # omit to auto-pick next track of the media's kind
    timeline_start_frame: int = 0
    duration_seconds: Optional[float] = None  # for stills; defaults to 5s


class AddTrackRequest(BaseModel):
    kind: str  # "V" | "A" | "T"


class TrackFlagsRequest(BaseModel):
    track: str
    hidden: Optional[bool] = None
    locked: Optional[bool] = None
    muted: Optional[bool] = None


class DeleteTrackRequest(BaseModel):
    track: str


class DetachAudioRequest(BaseModel):
    item_id: str
    track: Optional[str] = None


class CompoundRequest(BaseModel):
    item_ids: List[str]
    label: Optional[str] = None


class TransformRequest(BaseModel):
    item_id: Optional[str] = None      # omit to target the whole program
    updates: Dict[str, Any] = {}
    reset: bool = False


class ColorRequest(BaseModel):
    item_id: Optional[str] = None      # omit to target the whole program
    preset: Optional[str] = None
    updates: Dict[str, Any] = {}
    reset: bool = False


class ChromaRequest(BaseModel):
    item_id: str                       # keying always targets one clip
    updates: Dict[str, Any] = {}
    reset: bool = False


class AddTextRequest(BaseModel):
    content: str = "Text"
    track: Optional[str] = None
    timeline_start_frame: int = 0
    duration_seconds: float = 3.0
    preset: Optional[str] = None
    style: Dict[str, Any] = {}


class UpdateTextRequest(BaseModel):
    item_id: str
    content: Optional[str] = None
    preset: Optional[str] = None
    style: Dict[str, Any] = {}


class ItemFlagsRequest(BaseModel):
    item_id: str
    enabled: Optional[bool] = None
    locked: Optional[bool] = None
    mute: Optional[bool] = None
    volume: Optional[float] = None
    label: Optional[str] = None
    loop: Optional[bool] = None
    audio_fade_in: Optional[float] = None
    audio_fade_out: Optional[float] = None
    duck: Optional[float] = None


class TransitionRequest(BaseModel):
    item_id: Optional[str] = None      # omit to set the programme default
    preset: Optional[str] = None
    updates: Dict[str, Any] = {}
    reset: bool = False


class EffectRequest(BaseModel):
    preset: Optional[str] = None
    updates: Dict[str, Any] = {}
    item_id: Optional[str] = None      # an adjustment layer, or omit for the programme


class EffectUpdateRequest(BaseModel):
    index: int
    updates: Dict[str, Any] = {}
    item_id: Optional[str] = None


class AddAdjustmentRequest(BaseModel):
    track: Optional[str] = None        # omit to take the next free video track
    timeline_start_frame: int = 0
    duration_seconds: float = 5.0
    preset: Optional[str] = None       # colour preset to start from


class AspectRequest(BaseModel):
    ratio: Optional[float] = None      # null clears the bars


class PreviewProxyRequest(BaseModel):
    # 540 keeps a 1080p programme watchable; 360 is for a long edit you only
    # want to sanity-check.
    height: int = 540
    # A window of programme time (seconds) drafts just that stretch; its proxy
    # sits beside the full one and its status reports `offset_s`.
    start_s: Optional[float] = None
    end_s: Optional[float] = None


class CaptionsRequest(BaseModel):
    preset: str = "classic"
    style: Dict[str, Any] = {}


class IntroRequest(BaseModel):
    preset: str
    title: str = ""
    subtitle: str = ""


class AudioMasterRequest(BaseModel):
    updates: Dict[str, Any] = {}


def _load_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data or "timeline" not in p_data:
        raise HTTPException(status_code=404, detail="Project or timeline not found")
    return p_data, Timeline.model_validate(p_data["timeline"])


def _load_timeline_readonly(project_id: str):
    """`_load_timeline` for readers (filmstrips, waveforms): the project dict
    comes from the store's per-version cache and must not be mutated; the
    Timeline built from it is the caller's own."""
    p_data = project_store.get_project_readonly(project_id)
    if not p_data or "timeline" not in p_data:
        raise HTTPException(status_code=404, detail="Project or timeline not found")
    return p_data, Timeline.model_validate(p_data["timeline"])


def _ensure_timeline(project_id: str):
    """Load the timeline, creating an empty one if the project has none yet.

    A project only grows a timeline when it is transcribed, but dragging clips in
    from the media library has to work before that — otherwise a brand new project
    rejects every drop until you run Transcribe, which is not how a bin is meant
    to behave.
    """
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    if p_data.get("timeline"):
        return p_data, Timeline.model_validate(p_data["timeline"])

    settings = p_data.get("settings") or {}
    tl = Timeline(
        fps_num=int(settings.get("fps") or 30),
        fps_den=1,
        width=int(settings.get("source_width") or 1920),
        height=int(settings.get("source_height") or 1080),
    )
    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)
    return p_data, tl


def _save_timeline(project_id: str, p_data: dict, tl: Timeline):
    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)
    # Composed preview frames are keyed by revision, so a stale one can never be
    # served — but it would sit in memory forever. Dropped on every edit.
    preview.invalidate(project_id)

@router.post("/{project_id}/generate")
async def generate_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    source_video = p_data.get("source_video")
    if not source_video or not Path(source_video).exists():
        raise HTTPException(status_code=400, detail="Source video missing")

    # 1. Extract Audio
    audio_path = extract_audio(source_video)

    # 2. Get Video Info
    v_info = get_video_info(source_video)
    fps_num = v_info.get("fps_num", 30)
    fps_den = v_info.get("fps_den", 1)

    # 3. ASR Transcription. Pin the language across runs — auto-detect flips
    # code-switched speech into an English paraphrase with useless timings.
    settings = p_data.get("settings") or {}
    words, detected_language = await whisper_engine.transcribe_words_async(
        audio_path, language=settings.get("language"))
    settings["language"] = detected_language
    p_data["settings"] = settings

    # 4. Disfluency Analysis (deterministic + LLM adjudication, fail-safe)
    aggressiveness = float(settings.get("fumble_aggressiveness", 0.5))
    plan = await plan_auto_edit(words, audio_path, aggressiveness=aggressiveness,
                                settings=settings)
    edit_report = plan.report

    # 5. Build Timeline
    tl = build_timeline_from_transcript(
        source_path=source_video,
        duration_seconds=v_info.get("duration", 0.0),
        transcript_words=plan.words,
        fps_num=fps_num,
        fps_den=fps_den,
        width=v_info.get("width", 1920),
        height=v_info.get("height", 1080),
        has_audio=v_info.get("audio_codec") not in (None, "none"),
        speech_regions=edit_report.get("speech"),
        max_pause_seconds=settings.get("max_pause_seconds"),
        pause_padding_seconds=settings.get("pause_padding_seconds"),
        # Without this the cuts never move to the quietest nearby instant, so
        # this path produced audibly worse joins than the other three for no
        # reason anybody had chosen.
        energy_envelope=edit_report.get("energy"),
    )
    record_cut_coverage(tl, edit_report)

    # Store timeline in project data
    p_data["timeline"] = tl.model_dump()
    p_data["status"] = "ready"
    project_store.save_project(project_id, p_data)

    # The report was being computed and dropped here, so this path could not tell
    # the user what the edit had done or whether it had used the audio at all.
    return {"status": "success", "timeline": tl, "report": public_report(edit_report)}

@router.get("/{project_id}/waveform/{source_id}")
async def get_waveform(project_id: str, source_id: str):
    """Return normalized audio peaks for a timeline source (for audio-lane rendering)."""
    from utils.waveform import compute_waveform
    p_data, tl = await asyncio.to_thread(_load_timeline_readonly, project_id)
    src = tl.sources.get(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found in timeline")
    data = await asyncio.to_thread(compute_waveform, src.path)
    if not data:
        return {"peaks": [], "points_per_second": 60, "duration": 0.0}
    return data


def _filmstrip_meta(project_id: str, source_id: str):
    """Shared lookup for the two filmstrip endpoints: (source, strip) or a 404."""
    from utils.filmstrip import compute_filmstrip
    _p_data, tl = _load_timeline_readonly(project_id)
    src = tl.sources.get(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found in timeline")
    return src, compute_filmstrip(src.path, getattr(src, "kind", "video") or "video")


@router.get("/{project_id}/filmstrip/{source_id}")
async def get_filmstrip(project_id: str, source_id: str):
    """Layout of a source's filmstrip, so a video lane can show frames not blocks.

    The strip itself is served by the sibling `/image` route; this is only the
    geometry the UI needs to line the right part of it up under each clip.
    `columns: 0` means the source has no picture (an audio file) — the caller
    draws its normal block, and does not ask again.
    """
    _src, strip = await asyncio.to_thread(_filmstrip_meta, project_id, source_id)
    if not strip:
        return {"columns": 0, "tile_width": 0, "tile_height": 0, "duration": 0.0}
    return {k: v for k, v in strip.items() if k != "image_path"}


@router.get("/{project_id}/filmstrip/{source_id}/image")
async def get_filmstrip_image(project_id: str, source_id: str):
    """The strip itself: one row of evenly spaced frames across the whole source."""
    _src, strip = await asyncio.to_thread(_filmstrip_meta, project_id, source_id)
    if not strip:
        raise HTTPException(status_code=404, detail="No filmstrip for this source")
    return FileResponse(strip["image_path"], media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@router.get("/{project_id}")
async def get_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    tl_data = p_data.get("timeline")
    if not tl_data:
        raise HTTPException(status_code=404, detail="Timeline not yet generated for this project")
    return tl_data

@router.post("/{project_id}/toggle_word")
async def toggle_word_endpoint(project_id: str, body: ToggleWordRequest):
    p_data = project_store.get_project(project_id)
    if not p_data or "timeline" not in p_data:
        raise HTTPException(status_code=404, detail="Project or timeline not found")

    tl = Timeline.model_validate(p_data["timeline"])
    primary_src_id = list(tl.sources.keys())[0]

    success = toggle_word(tl, body.word_id, body.enabled, primary_src_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Word {body.word_id} not found")

    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)

    return {"status": "success", "timeline": tl}


class ReviewAnswer(BaseModel):
    answer: str          # "keep" | "cut" | "dismiss"


@router.get("/{project_id}/review")
async def get_review(project_id: str):
    """The AI editor's "decided under doubt" list for this project."""
    _p, tl = _load_timeline(project_id)
    return {"review": tl.review}


@router.post("/{project_id}/review/{take}")
async def answer_review(project_id: str, take: str, body: ReviewAnswer):
    """Answer one review item: the whole take plays (keep), goes (cut), or the item is
    dismissed. A keep/cut answer becomes an example the editor reads on the next edit."""
    from asr import review

    p_data, tl = _load_timeline(project_id)
    settings = p_data.get("settings") or {}
    try:
        result = review.answer(tl, take, body.answer, list(tl.sources.keys())[0],
                               channel=str(settings.get("glossary") or ""), source=project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{take} is not on the review list")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "timeline": tl, **result}


@router.post("/{project_id}/program/cut")
async def cut_program(project_id: str, body: ProgramCutRequest):
    """Delete a span of the finished programme — every track, not just the words."""
    p_data, tl = _load_timeline(project_id)
    primary_src_id = list(tl.sources.keys())[0]
    try:
        cut_program_range(tl, body.start_frame, body.end_frame, primary_src_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "timeline": tl}


@router.post("/{project_id}/program/restore_cuts")
async def restore_program_cuts(project_id: str):
    """Undo every manual program cut, restoring V1/A1 to what the words imply."""
    p_data, tl = _load_timeline(project_id)
    primary_src_id = list(tl.sources.keys())[0]
    tl.manual_cuts = []
    rebuild_primary_tracks(tl, primary_src_id)
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "timeline": tl}


@router.post("/{project_id}/render")
async def render_project_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data or "timeline" not in p_data:
        raise HTTPException(status_code=404, detail="Project or timeline not found")

    tl = Timeline.model_validate(p_data["timeline"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(OUTPUT_DIR / f"{project_id}_rendered.mp4")

    try:
        rendered_path = await render_timeline_async(tl, out_path)
    except (RuntimeError, ValueError) as e:
        # Surface the reason instead of a bare 500 — a render failure is almost
        # always something the user can act on (a missing source file, a clip
        # whose media has no audio), and "Internal Server Error" says none of it.
        raise HTTPException(status_code=400, detail=f"Render failed: {e}")

    p_data["rendered_video"] = rendered_path
    p_data["status"] = "rendered"
    project_store.save_project(project_id, p_data)

    return {"status": "success", "output_path": rendered_path}


# ---------------------------------------------------------------------------
# Multi-track clip editing (split / delete / move / trim / add media)
# ---------------------------------------------------------------------------

def _clip_op(project_id: str, fn):
    """Run a clip op, translating ClipOpError into a 400 and saving on success."""
    p_data, tl = _load_timeline(project_id)
    try:
        fn(tl)
    except clip_ops.ClipOpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "timeline": tl}


@router.post("/{project_id}/clip/split")
async def split_clip(project_id: str, body: SplitRequest):
    return _clip_op(project_id, lambda tl: clip_ops.split_item(
        tl, body.item_id, body.at_frame, linked=body.linked))


@router.post("/{project_id}/clip/split_many")
async def split_clips(project_id: str, body: SplitManyRequest):
    """Split every listed clip the frame falls inside — one undo step, one save."""
    return _clip_op(project_id, lambda tl: clip_ops.split_items(
        tl, body.item_ids, body.at_frame, linked=body.linked))


@router.post("/{project_id}/clip/delete")
async def delete_clip(project_id: str, body: DeleteClipRequest):
    return _clip_op(project_id, lambda tl: clip_ops.delete_item(
        tl, body.item_id, ripple=body.ripple, linked=body.linked))


@router.post("/{project_id}/clip/delete_many")
async def delete_clips(project_id: str, body: DeleteManyRequest):
    """Delete (or ripple-delete) a selection of clips with their linked partners."""
    return _clip_op(project_id, lambda tl: clip_ops.delete_items(
        tl, body.item_ids, ripple=body.ripple, linked=body.linked))


@router.post("/{project_id}/clip/move")
async def move_clip(project_id: str, body: MoveClipRequest):
    return _clip_op(project_id, lambda tl: clip_ops.move_item(
        tl, body.item_id, body.timeline_start_frame, body.track, linked=body.linked))


@router.post("/{project_id}/clip/move_many")
async def move_clips(project_id: str, body: MoveManyRequest):
    """Slide a selection of clips (and their linked partners) together."""
    return _clip_op(project_id, lambda tl: clip_ops.move_items(
        tl, body.item_ids, body.delta_frames, body.track_map, linked=body.linked))


@router.post("/{project_id}/clip/trim")
async def trim_clip(project_id: str, body: TrimClipRequest):
    return _clip_op(project_id, lambda tl: clip_ops.trim_item(
        tl, body.item_id, body.edge, body.timeline_frame,
        linked=body.linked, ripple=body.ripple))


@router.post("/{project_id}/clip/trim_many")
async def trim_clips(project_id: str, body: TrimManyRequest):
    """Trim start/end of the listed clips to a frame (trim to playhead)."""
    return _clip_op(project_id, lambda tl: clip_ops.trim_items(
        tl, body.item_ids, body.edge, body.at_frame, ripple=body.ripple))


@router.post("/{project_id}/clip/link")
async def link_clips(project_id: str, body: LinkRequest):
    return _clip_op(project_id, lambda tl: clip_ops.link_items(tl, body.item_ids))


@router.post("/{project_id}/clip/unlink")
async def unlink_clips(project_id: str, body: LinkRequest):
    return _clip_op(project_id, lambda tl: clip_ops.unlink_items(tl, body.item_ids))


@router.post("/{project_id}/clip/paste")
async def paste_clips(project_id: str, body: PasteRequest):
    """Paste clipboard clips at a frame; `insert` pushes later clips right."""
    p_data, tl = _ensure_timeline(project_id)
    try:
        pasted = clip_ops.paste_items(tl, body.items, body.at_frame, insert=body.insert)
    except clip_ops.ClipOpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "item_ids": [c.id for c in pasted], "timeline": tl}


@router.post("/{project_id}/track/close_gap")
async def close_track_gap(project_id: str, body: CloseGapRequest):
    """Ripple-delete the empty stretch of a track under a frame."""
    return _clip_op(project_id, lambda tl: clip_ops.close_gap(tl, body.track, body.at_frame))


@router.post("/{project_id}/replace")
async def replace_timeline(project_id: str, body: ReplaceTimelineRequest):
    """Put back a whole timeline the editor held earlier — the undo/redo path.

    The revision always moves forward, never back to the snapshot's own number:
    preview frames and proxies are keyed by revision, and reusing an old number
    could serve a picture of a different edit.
    """
    p_data, current = _ensure_timeline(project_id)
    try:
        tl = Timeline.model_validate(body.timeline)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Not a timeline: {e}")
    tl.revision = max(current.revision, tl.revision) + 1
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "timeline": tl}


@router.post("/{project_id}/clip/detach_audio")
async def detach_audio(project_id: str, body: DetachAudioRequest):
    """Split a video clip's audio onto its own track so it can be edited alone."""
    return _clip_op(project_id, lambda tl: clip_ops.detach_audio(
        tl, body.item_id, body.track))


@router.post("/{project_id}/clip/compound")
async def compound_clips(project_id: str, body: CompoundRequest):
    """Group the selected clips into one block that moves and trims as a unit."""
    return _clip_op(project_id, lambda tl: clip_ops.make_compound(
        tl, body.item_ids, body.label))


@router.post("/{project_id}/clip/uncompound")
async def uncompound_clip(project_id: str, body: DeleteClipRequest):
    return _clip_op(project_id, lambda tl: clip_ops.break_compound(tl, body.item_id))


@router.post("/{project_id}/clip/flags")
async def set_clip_flags(project_id: str, body: ItemFlagsRequest):
    """Toggle enabled / locked / mute / loop, set volume, fades, ducking or rename a clip."""
    return _clip_op(project_id, lambda tl: clip_ops.set_item_flags(
        tl, body.item_id, enabled=body.enabled, locked=body.locked,
        mute=body.mute, volume=body.volume, label=body.label,
        loop=body.loop, audio_fade_in=body.audio_fade_in,
        audio_fade_out=body.audio_fade_out, duck=body.duck))


@router.post("/{project_id}/transform")
async def set_transform(project_id: str, body: TransformRequest):
    """Crop / scale / pan / rotate a clip, or the whole program when item_id is omitted."""
    if body.item_id:
        return _clip_op(project_id, lambda tl: clip_ops.set_transform(
            tl, body.item_id, body.updates, reset=body.reset))
    return _clip_op(project_id, lambda tl: clip_ops.set_master_transform(
        tl, body.updates, reset=body.reset))


@router.post("/{project_id}/color")
async def set_color(project_id: str, body: ColorRequest):
    """Grade a clip, or the whole program when item_id is omitted."""
    if body.item_id:
        return _clip_op(project_id, lambda tl: clip_ops.set_color(
            tl, body.item_id, body.updates, preset=body.preset, reset=body.reset))
    return _clip_op(project_id, lambda tl: clip_ops.set_master_color(
        tl, body.updates, preset=body.preset, reset=body.reset))


@router.post("/{project_id}/chroma")
async def set_chroma(project_id: str, body: ChromaRequest):
    """Key a green/blue screen out of an overlay clip."""
    return _clip_op(project_id, lambda tl: clip_ops.set_chroma(
        tl, body.item_id, body.updates, reset=body.reset))


@router.post("/{project_id}/transition")
async def set_transition(project_id: str, body: TransitionRequest):
    """Set the transition into a clip, or the programme default when item_id is omitted."""
    return _clip_op(project_id, lambda tl: clip_ops.set_transition(
        tl, body.item_id, preset=body.preset, updates=body.updates, reset=body.reset))


@router.post("/{project_id}/effects/add")
async def add_effect(project_id: str, body: EffectRequest):
    """Add a generated atmosphere effect (rain, lightning, light leak, …).

    Goes on the whole programme, or on one adjustment layer when item_id is given.
    """
    return _clip_op(project_id, lambda tl: clip_ops.add_effect(
        tl, preset=body.preset, updates=body.updates, item_id=body.item_id))


@router.post("/{project_id}/effects/update")
async def update_effect(project_id: str, body: EffectUpdateRequest):
    return _clip_op(project_id, lambda tl: clip_ops.update_effect(
        tl, body.index, body.updates, item_id=body.item_id))


@router.post("/{project_id}/effects/remove")
async def remove_effect(project_id: str, body: EffectUpdateRequest):
    return _clip_op(project_id, lambda tl: clip_ops.remove_effect(
        tl, body.index, item_id=body.item_id))


@router.post("/{project_id}/adjustment/add")
async def add_adjustment(project_id: str, body: AddAdjustmentRequest):
    """Place an adjustment layer — a clip that treats every layer below it."""
    p_data, tl = _ensure_timeline(project_id)
    frames = max(1, time_to_frame(body.duration_seconds, tl.fps_num, tl.fps_den))
    return _clip_op(project_id, lambda t: clip_ops.add_adjustment_item(
        t, body.timeline_start_frame, frames, track=body.track, preset=body.preset))


@router.post("/{project_id}/aspect")
async def set_aspect(project_id: str, body: AspectRequest):
    """Letterbox the programme to a cinematic aspect ratio (null clears it)."""
    return _clip_op(project_id, lambda tl: clip_ops.set_aspect_bars(tl, body.ratio))


@router.get("/{project_id}/audio_master")
async def get_audio_master(project_id: str):
    """The programme's voice/loudness treatment, plus the catalogues the UI needs."""
    _p_data, tl = _load_timeline(project_id)
    return {
        "audio_master": (tl.audio_master or AudioMaster()).model_dump(),
        "eq_presets": list(EQ_PRESETS),
        "reverbs": list(_REVERB_RECIPES),
        "voice_fx": list(VOICE_FX_RECIPES),
        "enhance_modes": ["auto", "off", "deepfilter", "rnnoise", "ffmpeg"],
    }


@router.post("/{project_id}/audio_master")
async def set_audio_master(project_id: str, body: AudioMasterRequest):
    """Patch the programme's voice/loudness treatment."""
    p_data, tl = _load_timeline(project_id)
    tl.audio_master = clip_ops._merged(AudioMaster, tl.audio_master, body.updates)
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "timeline": tl}


# --- Dressed previews ------------------------------------------------------
# The live preview plays the source and skips the struck words, so it shows the
# cut and nothing else — no grade, no atmosphere, no captions, no B-roll. These
# two routes are where the dressing becomes visible before a full render.

@router.get("/{project_id}/preview/frame")
async def preview_frame(project_id: str, t: float = 0.0, w: int = 640):
    """The fully dressed frame at `t` seconds of programme time, as a JPEG.

    Composed by the real render compiler, so it is exact rather than a
    simulation. 204 means there is nothing on V1 at that moment — a gap, or past
    the end — and the caller should show black rather than the previous frame.
    """
    _p_data, tl = _load_timeline(project_id)
    width = max(160, min(1920, int(w)))
    data = await asyncio.to_thread(preview.compose_frame, project_id, tl, float(t), width)
    if not data:
        return Response(status_code=204)
    return Response(
        content=data,
        media_type="image/jpeg",
        # Immutable per (revision, time, width): the URL carries the revision, so
        # the browser may keep it forever and scrubbing back costs nothing.
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{project_id}/preview/proxy")
async def preview_proxy_status(project_id: str, start_s: Optional[float] = None,
                               end_s: Optional[float] = None):
    """Whether a dressed proxy (the full draft, or the `start_s`-`end_s`
    window's) exists, is building, or is behind the edit."""
    _p_data, tl = _load_timeline(project_id)
    window = (start_s, end_s) if start_s is not None and end_s is not None else None
    return preview.proxy_status(project_id, tl.revision, window=window)


@router.post("/{project_id}/preview/proxy")
async def preview_proxy_build(project_id: str, body: PreviewProxyRequest):
    """Build a draft of the dressed programme (or of one window) in the background.

    A fast 540p draft (render/preview.py), still something the user asks for.
    """
    _p_data, tl = _load_timeline(project_id)
    window = ((body.start_s, body.end_s)
              if body.start_s is not None and body.end_s is not None else None)
    return preview.start_proxy(project_id, tl, height=body.height, window=window)


@router.post("/{project_id}/text/add")
async def add_text(project_id: str, body: AddTextRequest):
    """Place a text element on a text track."""
    p_data, tl = _ensure_timeline(project_id)
    frames = max(1, time_to_frame(body.duration_seconds, tl.fps_num, tl.fps_den))
    return _clip_op(project_id, lambda t: clip_ops.add_text_item(
        t, body.content, body.timeline_start_frame, frames,
        track=body.track, preset=body.preset, style=body.style))


@router.post("/{project_id}/text/update")
async def update_text(project_id: str, body: UpdateTextRequest):
    return _clip_op(project_id, lambda tl: clip_ops.set_text(
        tl, body.item_id, content=body.content, preset=body.preset, style=body.style))


@router.post("/{project_id}/captions/generate")
async def generate_captions(project_id: str, body: CaptionsRequest):
    """(Re)build burned-in captions from the enabled transcript words."""
    p_data, tl = _load_timeline(project_id)
    if not tl.words:
        raise HTTPException(status_code=400,
                            detail="No transcript yet — run Transcribe before generating captions.")
    # A style profile can pin caption geometry to a reference video, and it used
    # to override the picked preset's position outright — so choosing
    # "youtube_shorts" still parked captions at the reference video's band and
    # there was no way to move them back. The preset owns the on-screen area, so
    # drop the profile's measured *position* here; its look (colour, box, font
    # size) still carries, and an explicit style in the request still wins.
    stored = dict((p_data.get("settings") or {}).get("caption_style_overrides") or {})
    for geometry_key in ("pos_x", "pos_y"):
        stored.pop(geometry_key, None)
    overrides = {**stored, **(body.style or {})}
    created = authoring.generate_captions(
        tl, preset=body.preset, style_overrides=overrides,
        script=authoring.caption_script_for(p_data.get("settings")))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "captions_created": len(created), "timeline": tl}


@router.post("/{project_id}/captions/clear")
async def clear_captions(project_id: str):
    p_data, tl = _load_timeline(project_id)
    removed = authoring.remove_captions(tl)
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "captions_removed": removed, "timeline": tl}


@router.post("/{project_id}/intro/apply")
async def apply_intro(project_id: str, body: IntroRequest):
    """Insert a preset intro in front of the program."""
    p_data, tl = _load_timeline(project_id)
    try:
        result = authoring.apply_intro(tl, body.preset, title=body.title, subtitle=body.subtitle)
    except clip_ops.ClipOpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "intro": result["preset"],
            "items_created": len(result["items"]), "timeline": tl}


@router.post("/{project_id}/intro/clear")
async def clear_intro(project_id: str):
    p_data, tl = _load_timeline(project_id)
    authoring.remove_intro(tl)
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "timeline": tl}


@router.post("/{project_id}/add_track")
async def add_track(project_id: str, body: AddTrackRequest):
    """Reserve the next free lane of a kind so it persists before anything lands on it."""
    p_data, tl = _ensure_timeline(project_id)
    name = clip_ops.add_track(tl, body.kind)
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "track": name, "timeline": tl}


@router.post("/{project_id}/track/flags")
async def set_track_flags(project_id: str, body: TrackFlagsRequest):
    """Hide, lock or mute a whole track (layer)."""
    return _clip_op(project_id, lambda tl: clip_ops.set_track_flags(
        tl, body.track, hidden=body.hidden, locked=body.locked, muted=body.muted))


@router.post("/{project_id}/track/delete")
async def delete_track(project_id: str, body: DeleteTrackRequest):
    """Delete a track and every clip on it."""
    return _clip_op(project_id, lambda tl: clip_ops.delete_track(tl, body.track))


def _register_source(tl: Timeline, src_path: Path, kind: str,
                     still_seconds: Optional[float] = None) -> SourceFile:
    """The timeline source for a file, registering it when it is new.

    A path already on the timeline reuses its existing source, so the render
    graph never opens the same file twice. Places nothing on a track."""
    existing = next((s for s in tl.sources.values()
                     if Path(s.path).resolve() == src_path.resolve()), None)
    if existing is not None:
        return existing
    info = media_pool.probe(src_path, kind)
    duration = info["duration"]
    if kind == "image":
        duration = float(still_seconds or 5.0)
    elif duration <= 0:
        raise HTTPException(status_code=400, detail=f"Could not read media: {src_path.name}")

    source_id = f"src_{uuid.uuid4().hex[:8]}"
    source = SourceFile(
        id=source_id,
        path=str(src_path),
        duration_seconds=duration,
        width=int(info["width"] or 1920),
        height=int(info["height"] or 1080),
        fps_num=tl.fps_num,
        fps_den=tl.fps_den,
        has_audio=info["has_audio"],
        kind=kind,
    )
    tl.sources[source_id] = source
    return source


@router.post("/{project_id}/add_media")
async def add_media(project_id: str, body: AddMediaRequest):
    """Register a media file as a timeline source and place it as a clip on a track.

    The file is referenced where it lives rather than copied: material dragged in
    from the media library is usually already a link, and copying it again would
    duplicate gigabytes for no benefit. A path already on the timeline reuses its
    existing source, so the render graph never opens the same file twice.
    """
    p_data, tl = _ensure_timeline(project_id)

    src_path = Path(body.path)
    if not src_path.exists() or not src_path.is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {body.path}")

    kind = media_pool.media_kind(src_path)
    source = _register_source(tl, src_path, kind, body.duration_seconds)
    source_id = source.id
    duration = source.duration_seconds

    # On an empty timeline the first clip becomes the program itself, so a project
    # built purely by dragging media in still renders. Transcribing later rebuilds
    # V1/A1 from the words, which is the documented behaviour of the primary tracks.
    lane_kind = "A" if kind == "audio" else "V"
    primary = "A1" if kind == "audio" else "V1"
    if body.track and clip_ops.track_kind(body.track) != lane_kind:
        raise HTTPException(status_code=400,
                            detail=f"{kind} media can only go on a {lane_kind} track")
    has_primary = any(i.track == primary for i in tl.items)
    track = body.track or (None if has_primary else primary)
    source_end_frame = max(1, time_to_frame(duration, tl.fps_num, tl.fps_den))

    # The first clip of a timeline starts it — there is nothing to play before
    # it, and V1 renders gapless anyway, so a drop at 0:12 would only mislead.
    has_media = any(i.kind == "media" for i in tl.items)
    start = max(0, body.timeline_start_frame) if has_media else 0

    # A transcript-built spine is not a place to drop clips: it is rebuilt from
    # the words. A drop aimed at it goes on the first free overlay lane instead.
    if track in ("V1", "A1") and clip_ops.track_has_auto(tl, track):
        track = None
    if track == clip_ops.MAIN_TRACK:
        # Magnetic main track: land on the nearest edit point and push the rest along.
        start = clip_ops.main_track_insert_frame(tl, start)
    elif track is None or not clip_ops.track_fits(tl, track, start, start + source_end_frame):
        # Never stack a clip on top of another on the same lane: take the first
        # lane (the one aimed at, if it is free) with room for it.
        track = clip_ops.free_track(tl, lane_kind, start, start + source_end_frame,
                                    prefer=[track] if track else [])

    try:
        item = clip_ops.add_media_item(
            tl,
            source_id=source_id,
            track=track,
            timeline_start_frame=start,
            source_start_frame=0,
            source_end_frame=source_end_frame,
            origin="manual",
        )
    except clip_ops.ClipOpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    item.label = src_path.name
    placed = [item]

    # Video tracks carry picture only, so a video's own sound goes on an audio
    # lane beside it — linked, so the two move, trim, split and delete as one
    # clip until the user unlinks them.
    if kind == "video" and source.has_audio:
        if track == clip_ops.MAIN_TRACK and clip_ops.a1_follows_v1(tl)                 and clip_ops.track_fits(tl, "A1", item.timeline_start_frame, item.timeline_end_frame):
            audio_track = "A1"
        else:
            twin = f"A{''.join(c for c in track if c.isdigit()) or '2'}"
            audio_track = clip_ops.free_track(tl, "A", item.timeline_start_frame,
                                              item.timeline_end_frame, prefer=[twin])
        try:
            audio = clip_ops.add_media_item(
                tl,
                source_id=source_id,
                track=audio_track,
                timeline_start_frame=item.timeline_start_frame,
                source_start_frame=item.source_start_frame,
                source_end_frame=item.source_end_frame,
                origin="manual",
            )
        except clip_ops.ClipOpError as e:
            raise HTTPException(status_code=400, detail=str(e))
        audio.label = src_path.name
        item.link_id = audio.link_id = clip_ops.new_link_id()
        placed.append(audio)

    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "item": item, "item_ids": [p.id for p in placed], "timeline": tl}


# ---------------------------------------------------------------------------
# Multicam angles and external audio sync (docs/MULTICAM_SYNC_PLAN.md)
#
# Undo: the angle routes are ordinary timeline edits -- the editor keeps its
# snapshot and puts it back with POST /replace. /sync/audio pushes a version on
# the primary source's audio_history, so POST /api/audio/{id}/undo reverts it.
# ---------------------------------------------------------------------------

class SyncAudioRequest(BaseModel):
    media_id: str
    # Only reported back as a hint: a transcription is never started from here.
    retranscribe: bool = False


class AngleSyncRequest(BaseModel):
    media_id: str


class AngleSetRequest(BaseModel):
    start_frame: int                   # programme frames
    end_frame: int
    source_id: Optional[str] = None    # None = back to the primary


class AngleAutoRequest(BaseModel):
    mode: str                          # "hide_cuts" | "rhythm" | "off"
    min_shot_s: float = 2.5
    cadence_s: float = 8.0


def _primary_source(tl: Timeline) -> SourceFile:
    """The programme's own recording: the first source, as every V1/A1 route uses."""
    if not tl.sources:
        raise HTTPException(status_code=400, detail="The timeline has no source yet")
    return tl.sources[list(tl.sources.keys())[0]]


def _pool_media(p_data: Dict[str, Any], media_id: str) -> Dict[str, Any]:
    entry = next((m for m in p_data.get("media_pool", []) if m.get("id") == media_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No media {media_id!r} in the library")
    path = entry.get("path") or ""
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=400, detail=f"Media file not found: {path}")
    return entry


def _unreliable(result) -> HTTPException:
    return HTTPException(status_code=422, detail={
        "message": (f"The two soundtracks do not line up reliably (confidence "
                    f"{result.confidence:.2f} < 1.5); nothing was changed"),
        **result.to_dict()})


def _measure(ref_path: str, other_path: str):
    from sync.audio_sync import SyncError, find_offset
    try:
        return find_offset(ref_path, other_path)
    except SyncError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/sync/audio")
async def sync_external_audio(project_id: str, body: SyncAudioRequest):
    """Make an external recorder's audio the programme sound of the primary source.

    The recording is aligned to the primary's clock (offset, and drift corrected
    past 20 ppm), written as a 48 kHz WAV exactly as long as the primary into
    the project's audio/versions, and pushed on the primary's audio_history as
    `external_audio_sync` -- so render, preview and live playback use it, and
    POST /api/audio/{id}/undo puts the camera sound back. 422 when the match is
    unreliable (nothing written). Re-transcribing afterwards transcribes it.
    """
    from routes.audio_process import _no_render_running
    from audio import noise, processes
    from sync.audio_sync import write_aligned_wav
    from datetime import datetime

    _no_render_running()
    p_data, tl = _load_timeline(project_id)
    primary = _primary_source(tl)
    entry = _pool_media(p_data, body.media_id)
    if entry.get("kind") not in ("audio", "video") or entry.get("has_audio") is False:
        raise HTTPException(status_code=400, detail="That media has no audio to sync")
    if not primary.has_audio or primary.duration_seconds <= 0:
        raise HTTPException(status_code=400, detail="The primary source has no audio to sync against")
    other_path = str(entry["path"])
    result = await asyncio.to_thread(_measure, primary.path, other_path)
    if not result.reliable:
        raise _unreliable(result)

    def write() -> Dict[str, Any]:
        sid = primary.id
        with processes._source_lock(project_id, sid):
            versions = noise.audio_dir(project_id) / "versions"
            versions.mkdir(parents=True, exist_ok=True)
            final = versions / f"{processes._next_version(versions)}_external_audio_sync.wav"
            write_aligned_wav(other_path, result, primary.duration_seconds, str(final))
            version = {"path": str(final), "effect": "external_audio_sync",
                       "params": {"media_id": body.media_id, "offset_s": result.offset_s,
                                  "confidence": result.confidence,
                                  "drift_ppm": result.drift_ppm},
                       "start_s": None, "end_s": None,
                       "created": datetime.now().isoformat(timespec="seconds")}

            def push(src: Any) -> None:
                src.audio_history.append(version)
                src.audio_override = str(final)

            state = processes._save_source(project_id, sid, push)
        return {**state, "version": version}

    try:
        state = await asyncio.to_thread(write)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not write the synced audio: {e}")
    return {"status": "success", **result.to_dict(), "source_id": primary.id,
            "version_path": state["version"]["path"], "audio_override": state["audio_override"],
            "history_len": state["history_len"], "revision": state["revision"],
            "undo": {"method": "POST", "path": f"/api/audio/{project_id}/undo",
                     "body": {"source_id": primary.id}},
            "retranscribe_hint": ({"method": "POST", "path": "/api/transcription/transcribe",
                                   "body": {"project_id": project_id}}
                                  if body.retranscribe else None)}


def _angles_view(p_data: Dict[str, Any], tl: Timeline) -> Dict[str, Any]:
    from timeline.ops import angle_sources, program_angle_spans
    if not tl.sources:
        return {"primary_source_id": None, "angles": [], "spans": [], "stored_spans": []}
    primary = _primary_source(tl)
    pool = p_data.get("media_pool", []) or []
    angles = []
    for source in angle_sources(tl, primary.id):
        entry = next((m for m in pool if (m.get("angle") or {}).get("source_id") == source.id), None) \
            or next((m for m in pool if m.get("path") and Path(m["path"]).resolve()
                     == Path(source.path).resolve()), None) or {}
        angles.append({"source_id": source.id, "media_id": entry.get("id"),
                       "name": entry.get("name") or Path(source.path).name,
                       "offset_s": source.sync_offset_s, "confidence": source.sync_confidence,
                       "width": source.width, "height": source.height,
                       "fps": entry.get("fps") or None})
    return {"primary_source_id": primary.id, "angles": angles,
            "spans": program_angle_spans(tl), "stored_spans": tl.angle_spans}


@router.post("/{project_id}/angles/sync")
async def sync_angle(project_id: str, body: AngleSyncRequest):
    """Register a library video as a camera angle of the primary recording.

    The offset is measured from the two soundtracks and stored on the new source
    (synced_to / sync_offset_s / sync_confidence) and mirrored on the library
    entry as `angle`. No clip is placed: angles are chosen with /angles/set or
    /angles/auto. 422 when the soundtracks do not line up reliably.
    """
    from timeline.ops import _reapply_angles
    p_data, tl = _load_timeline(project_id)
    primary = _primary_source(tl)
    entry = _pool_media(p_data, body.media_id)
    if entry.get("kind") != "video":
        raise HTTPException(status_code=400, detail="A camera angle has to be a video")
    if Path(entry["path"]).resolve() == Path(primary.path).resolve():
        raise HTTPException(status_code=400, detail="That is the primary recording itself")
    if not primary.has_audio:
        raise HTTPException(status_code=400, detail="The primary source has no audio to sync against")
    result = await asyncio.to_thread(_measure, primary.path, str(entry["path"]))
    if not result.reliable:
        raise _unreliable(result)

    # Read again: the measurement takes a while and the timeline may have moved.
    p_data, tl = _load_timeline(project_id)
    primary = _primary_source(tl)
    entry = _pool_media(p_data, body.media_id)
    source = _register_source(tl, Path(entry["path"]), "video")
    source.synced_to = primary.id
    source.sync_offset_s = float(result.offset_s)
    source.sync_confidence = float(result.confidence)
    entry["angle"] = {"synced_to": primary.id, "offset_s": float(result.offset_s),
                      "confidence": float(result.confidence), "source_id": source.id,
                      "drift_ppm": float(result.drift_ppm)}
    # A source already showing on V1 moves with its new offset.
    _reapply_angles(tl, primary.id)
    _save_timeline(project_id, p_data, tl)
    warning = None
    if abs(result.drift_ppm) > 20:
        warning = (f"The two clocks drift {result.drift_ppm:+.0f} ppm; the angle's picture is "
                   f"placed with one offset, so it slips by up to "
                   f"{abs(result.drift_ppm) * primary.duration_seconds / 1e6:.2f}s over the take")
    return {"status": "success", "source_id": source.id, **result.to_dict(),
            "warning": warning, **_angles_view(p_data, tl), "timeline": tl}


@router.get("/{project_id}/angles")
async def list_angles(project_id: str):
    """The synced camera angles and where each shows on the programme.

    `spans` are programme frames (start/end) with the primary-source frames they
    cover (source_start/source_end); `stored_spans` is Timeline.angle_spans."""
    p_data, tl = _load_timeline_readonly(project_id)
    return _angles_view(p_data, tl)


@router.post("/{project_id}/angles/set")
async def set_angle_range(project_id: str, body: AngleSetRequest):
    """Show an angle (or, with source_id null, the primary) for a programme range."""
    from timeline.ops import set_angle
    p_data, tl = _load_timeline(project_id)
    primary = _primary_source(tl)
    try:
        set_angle(tl, body.start_frame, body.end_frame, body.source_id, primary.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "angle_spans": tl.angle_spans, "timeline": tl}


@router.delete("/{project_id}/angles")
async def clear_angle_spans(project_id: str):
    """Every angle choice removed: V1 is the primary recording again."""
    from timeline.ops import clear_angles
    p_data, tl = _load_timeline(project_id)
    clear_angles(tl, _primary_source(tl).id)
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "angle_spans": [], "timeline": tl}


@router.post("/{project_id}/angles/auto")
async def auto_angle_spans(project_id: str, body: AngleAutoRequest):
    """Let the cut choose the angles: hide_cuts | rhythm | off (see ops.auto_angles)."""
    from timeline.ops import auto_angles
    p_data, tl = _load_timeline(project_id)
    primary = _primary_source(tl)
    try:
        out = auto_angles(tl, primary.id, body.mode, min_shot_s=float(body.min_shot_s),
                          cadence_s=float(body.cadence_s))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "angle_spans": out["angle_spans"], "summary": out["summary"],
            "timeline": tl}
