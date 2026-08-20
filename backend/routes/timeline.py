import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from store.project_store import ProjectStore
from config import PROJECTS_DIR, OUTPUT_DIR
from utils.ffmpeg_utils import extract_audio, get_video_info
from asr import whisper_engine, detect_speech_silence_intervals
from asr.auto_edit import plan_auto_edit, public_report, record_cut_coverage
from timeline import build_timeline_from_transcript, toggle_word, Timeline
from timeline.schema import SourceFile, time_to_frame
from timeline import authoring, clip_ops
from store import media_pool
from render import render_timeline_async

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))


class ToggleWordRequest(BaseModel):
    word_id: str
    enabled: bool


class SplitRequest(BaseModel):
    item_id: str
    at_frame: int


class DeleteClipRequest(BaseModel):
    item_id: str


class MoveClipRequest(BaseModel):
    item_id: str
    timeline_start_frame: int
    track: Optional[str] = None


class TrimClipRequest(BaseModel):
    item_id: str
    edge: str  # "start" | "end"
    timeline_frame: int


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


class CaptionsRequest(BaseModel):
    preset: str = "classic"
    style: Dict[str, Any] = {}


class IntroRequest(BaseModel):
    preset: str
    title: str = ""
    subtitle: str = ""


def _load_timeline(project_id: str):
    p_data = project_store.get_project(project_id)
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
    plan = await plan_auto_edit(words, audio_path, aggressiveness=aggressiveness)
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
    p_data, tl = _load_timeline(project_id)
    src = tl.sources.get(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found in timeline")
    data = await asyncio.to_thread(compute_waveform, src.path)
    if not data:
        return {"peaks": [], "points_per_second": 60, "duration": 0.0}
    return data


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
    return _clip_op(project_id, lambda tl: clip_ops.split_item(tl, body.item_id, body.at_frame))


@router.post("/{project_id}/clip/delete")
async def delete_clip(project_id: str, body: DeleteClipRequest):
    return _clip_op(project_id, lambda tl: clip_ops.delete_item(tl, body.item_id))


@router.post("/{project_id}/clip/move")
async def move_clip(project_id: str, body: MoveClipRequest):
    return _clip_op(project_id, lambda tl: clip_ops.move_item(
        tl, body.item_id, body.timeline_start_frame, body.track))


@router.post("/{project_id}/clip/trim")
async def trim_clip(project_id: str, body: TrimClipRequest):
    return _clip_op(project_id, lambda tl: clip_ops.trim_item(
        tl, body.item_id, body.edge, body.timeline_frame))


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
    """Toggle enabled / locked / mute, set volume or rename a clip."""
    return _clip_op(project_id, lambda tl: clip_ops.set_item_flags(
        tl, body.item_id, enabled=body.enabled, locked=body.locked,
        mute=body.mute, volume=body.volume, label=body.label))


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
    created = authoring.generate_captions(tl, preset=body.preset, style_overrides=overrides)
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

    existing = next((s for s in tl.sources.values()
                     if Path(s.path).resolve() == src_path.resolve()), None)
    if existing is not None:
        source = existing
        source_id = existing.id
        duration = existing.duration_seconds
    else:
        info = media_pool.probe(src_path, kind)
        duration = info["duration"]
        if kind == "image":
            duration = float(body.duration_seconds or 5.0)
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

    # On an empty timeline the first clip becomes the program itself, so a project
    # built purely by dragging media in still renders. Transcribing later rebuilds
    # V1/A1 from the words, which is the documented behaviour of the primary tracks.
    primary = "A1" if kind == "audio" else "V1"
    has_primary = any(i.track == primary for i in tl.items)
    track = body.track or (primary if not has_primary
                           else clip_ops.next_track(tl, "A" if kind == "audio" else "V"))
    source_end_frame = max(1, time_to_frame(duration, tl.fps_num, tl.fps_den))

    try:
        item = clip_ops.add_media_item(
            tl,
            source_id=source_id,
            track=track,
            timeline_start_frame=body.timeline_start_frame,
            source_start_frame=0,
            source_end_frame=source_end_frame,
        )
    except clip_ops.ClipOpError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Video tracks carry picture only, so a clip that becomes the program needs its
    # sound placed on A1 as well — otherwise dropping a video onto an empty
    # timeline would render silent.
    if track == "V1" and kind == "video" and source.has_audio:
        if not any(i.track == "A1" for i in tl.items):
            clip_ops.add_media_item(
                tl,
                source_id=source_id,
                track="A1",
                timeline_start_frame=item.timeline_start_frame,
                source_start_frame=item.source_start_frame,
                source_end_frame=item.source_end_frame,
            )

    _save_timeline(project_id, p_data, tl)
    return {"status": "success", "item": item, "timeline": tl}
