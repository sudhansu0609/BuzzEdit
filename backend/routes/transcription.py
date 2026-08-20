import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException
from models import TranscribeJob, TranscriptionResult, TranscriptSegment
from config import PROJECTS_DIR
from store.project_store import ProjectStore
from utils.ffmpeg_utils import extract_audio, get_video_info
from asr import whisper_engine
from asr.auto_edit import plan_auto_edit, public_report, record_cut_coverage
from timeline import Transition, build_timeline_from_transcript

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
jobs: dict = {}

@router.post("/transcribe")
async def transcribe_video(job: TranscribeJob):
    p_data = project_store.get_project(job.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    source_video = p_data.get("source_video")
    if not source_video or not Path(source_video).exists():
        raise HTTPException(status_code=400, detail="Source video missing")

    job_id = f"trans_{job.project_id}"
    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "transcribe",
        "status": "running",
        "progress": 0.0,
    }

    try:
        # 1. Extract WAV audio. ffmpeg is blocking; off the event loop so the
        #    server stays responsive during a long extraction.
        audio_path = await asyncio.to_thread(extract_audio, source_video)

        # 2. Get Video info
        v_info = await asyncio.to_thread(get_video_info, source_video)

        # 3. Transcribe via faster-whisper (auto-detects language, romanizes to
        #    Hinglish, and produces an English translation for non-English speech)
        settings = p_data.get("settings") or {}
        # Reuse the language settled on last time. Auto-detection is not stable on
        # code-switched speech, and a run that lands on "en" paraphrases the audio
        # into English instead of transcribing it — which produced word timings
        # with no relationship to the recording.
        language = job.language or settings.get("language")
        full = await whisper_engine.transcribe_full_async(audio_path, language=language)
        words = full["words"]
        detected_language = full["language"]
        settings["language"] = detected_language
        p_data["settings"] = settings

        # 4. Auto-edit planning: audio-driven, because Whisper strips the fillers
        #    out of the text before we ever see them.
        aggressiveness = float(settings.get("fumble_aggressiveness", 0.5))
        plan = await plan_auto_edit(words, audio_path, aggressiveness=aggressiveness)
        ann_words = plan.words
        edit_report = plan.report

        # 5. Build EDL Timeline
        tl = build_timeline_from_transcript(
            source_path=source_video,
            duration_seconds=v_info.get("duration", 0.0),
            transcript_words=ann_words,
            fps_num=v_info.get("fps_num", 30),
            fps_den=v_info.get("fps_den", 1),
            width=v_info.get("width", 1920),
            height=v_info.get("height", 1080),
            has_audio=v_info.get("audio_codec") not in (None, "none"),
            speech_regions=edit_report.get("speech"),
            max_pause_seconds=settings.get("max_pause_seconds"),
            pause_padding_seconds=settings.get("pause_padding_seconds"),
            energy_envelope=edit_report.get("energy"),
            default_transition=(None if settings.get("hard_cuts")
                                else Transition(type="fade", duration=float(settings.get("transition_seconds", 0.25)))),
        )
        # The builder rebuilds V1/A1 itself, so this only scores the result: are
        # the words the planner cut actually absent from what will be rendered?
        record_cut_coverage(tl, edit_report)

        # Convert to TranscriptSegment list for legacy UI components
        segments = []
        if ann_words:
            seg = TranscriptSegment(
                id=0,
                start=ann_words[0]["start"],
                end=ann_words[-1]["end"],
                text=full["hinglish_text"] or " ".join([w["word"] for w in ann_words]),
                text_native=full["native_text"],
                text_english=full["english_text"],
                words=ann_words,
                confidence=1.0
            )
            segments.append(seg)

        res = TranscriptionResult(
            segments=segments,
            language=detected_language,
            duration=ann_words[-1]["end"] if ann_words else 0.0
        )

        p_data["transcript"] = res.model_dump()
        p_data["timeline"] = tl.model_dump()
        p_data["status"] = "transcribed"
        project_store.save_project(job.project_id, p_data)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = {"segments_count": len(segments)}

        return {
            "status": "completed",
            "job_id": job_id,
            "segments": len(segments),
            "timeline": tl,
            "report": public_report(edit_report),
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def transcription_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.get("/{project_id}/transcript")
async def get_transcript(project_id: str):
    p_data = project_store.get_project(project_id)
    if not p_data or "transcript" not in p_data:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return p_data["transcript"]
