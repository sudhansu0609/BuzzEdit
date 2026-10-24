import asyncio
import logging
import re
import time
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import TranscribeJob, TranscriptionResult, TranscriptSegment
from config import PROJECTS_DIR
from store.project_store import ProjectStore
from store import glossary_store
from utils.ffmpeg_utils import extract_audio, get_video_info
from asr import transcribe_checkpoint as ckpt
from asr import whisper_engine
from asr.auto_edit import plan_auto_edit, public_report, record_cut_coverage
from asr.transliterate import to_hinglish
from timeline import Transition, build_timeline_from_transcript
from timeline.authoring import CAPTION_TRACK
from timeline.schema import Timeline, frame_to_time

router = APIRouter()
project_store = ProjectStore(base_dir=str(PROJECTS_DIR))
jobs: dict = {}
#: The full answer of each completed transcription, by job id -- kept apart
#: from `jobs` so the status polls do not carry the whole timeline.
_responses: dict = {}


def _stage(job_id: str, text: str) -> None:
    """Record where a running transcription is. The Studio follows this record
    (`GET /status/{job_id}`) once its own wait on the POST runs out, and reads a
    stage that keeps changing as proof the job is alive."""
    job = jobs.get(job_id)
    if job is not None:
        job["stage"] = text
        job["updated_at"] = time.time()


#: How often a second POST for a project already transcribing checks on the
#: first one, and the longest it waits for it.
_JOIN_POLL_SECONDS = 5.0
_JOIN_MAX_SECONDS = 8 * 60 * 60


async def _join_running(job_id: str) -> Optional[dict]:
    """A transcription already running for this project: wait for it and hand
    back its answer, instead of starting a second one on the same recording.

    The Studio retries a transcription when it loses sight of BuzzEdit for a
    while; before this, that retry started Whisper a second time beside the
    first, both on the one GPU, and whichever finished last overwrote the
    project. None when nothing is running."""
    running = jobs.get(job_id)
    if not running or running.get("status") != "running":
        return None
    logging.getLogger("transcription").info(
        "%s is already running (%s); joining it instead of starting another.", job_id, running.get("stage"))
    started = time.monotonic()
    while time.monotonic() - started < _JOIN_MAX_SECONDS:
        await asyncio.sleep(_JOIN_POLL_SECONDS)
        current = jobs.get(job_id)
        if not current or current.get("status") != "running":
            break
    current = jobs.get(job_id) or {}
    if current.get("status") == "completed" and _responses.get(job_id) is not None:
        return _responses[job_id]
    if current.get("status") == "failed":
        raise HTTPException(status_code=500, detail=str(current.get("error") or "transcription failed"))
    raise HTTPException(status_code=409, detail="A transcription is already running for this project")


def _save_project_sync(project_id: str, p_data: dict) -> None:
    project_store.save_project(project_id, p_data)


@router.get("/jobs")
async def transcription_jobs():
    """Every transcription this process knows about, running ones first. The
    Studio asks before unloading LM Studio: a running transcription is using it."""
    items = sorted(jobs.values(), key=lambda j: (j.get("status") != "running", -(j.get("updated_at") or 0)))
    return {"jobs": items, "running": sum(1 for j in items if j.get("status") == "running")}


@router.post("/transcribe")
async def transcribe_video(job: TranscribeJob):
    p_data = project_store.get_project(job.project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")

    source_video = p_data.get("source_video")
    if not source_video or not Path(source_video).exists():
        raise HTTPException(status_code=400, detail="Source video missing")

    job_id = f"trans_{job.project_id}"
    joined = await _join_running(job_id)
    if joined is not None:
        return joined
    jobs[job_id] = {
        "id": job_id,
        "project_id": job.project_id,
        "job_type": "transcribe",
        "status": "running",
        "progress": 0.0,
        "updated_at": time.time(),
    }

    try:
        # 1. Extract WAV audio. ffmpeg is blocking; off the event loop so the
        #    server stays responsive during a long extraction.
        _stage(job_id, "extracting audio")
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
        if job.precut is not None:
            settings["precut"] = bool(job.precut)
        precut = bool(settings.get("precut"))
        # Pacing travels the same way as `precut`: sent once, kept on the
        # project, read by the timeline rebuild below and every later pass.
        for pacing_key in ("max_pause_seconds", "pause_padding_seconds", "fumble_aggressiveness"):
            pacing_value = getattr(job, pacing_key, None)
            if pacing_value is not None:
                settings[pacing_key] = float(pacing_value)
                if pacing_key != "fumble_aggressiveness":
                    # Sent explicitly: a choice, never the genre's to re-decide.
                    settings.pop("pacing_source", None)
        if job.genre:
            settings["genre"] = job.genre
        language = job.language or settings.get("language")
        # Reused exactly like `language`: once settled for a project, later
        # transcriptions (and `respell`) pick the same vocabulary/glossary back
        # up without the caller having to resend them.
        if job.vocabulary is not None:
            settings["vocabulary"] = job.vocabulary
        if job.glossary is not None:
            settings["glossary"] = job.glossary
        if job.planner:
            settings["planner"] = job.planner
        if job.speaker_style is not None:
            settings["speaker_style"] = job.speaker_style
        vocabulary = settings.get("vocabulary")
        glossary_key = settings.get("glossary")
        glossary_map = glossary_store.load_merged(glossary_key)
        # What an interrupted run of this same transcription left behind
        # (asr/transcribe_checkpoint.py): Whisper's words, the planned cut.
        # Each phase is reused when its key matches, so a retry after a crash
        # does not spend the GPU on the same recording again.
        ckpt_key = ckpt.key_for(source_video, language, precut, vocabulary)
        saved = await asyncio.to_thread(ckpt.load, PROJECTS_DIR, job.project_id, ckpt_key)
        full = saved.get(ckpt.PHASE_WHISPER) if isinstance(saved.get(ckpt.PHASE_WHISPER), dict) else None
        if full and full.get("words"):
            _stage(job_id, "reusing the transcription from the interrupted run")
            logging.getLogger("transcription").info(
                "%s: reusing Whisper's words from the checkpoint (%d words); not transcribing again.",
                job_id, len(full["words"]))
        else:
            _stage(job_id, f"transcribing with Whisper ({language or 'auto-detect'})")
            full = await whisper_engine.transcribe_full_async(
                audio_path, language=language, vocabulary=vocabulary, glossary=glossary_map)
            await asyncio.to_thread(ckpt.save, PROJECTS_DIR, job.project_id, ckpt_key, ckpt.PHASE_WHISPER, full)
        words = full["words"]
        detected_language = full["language"]
        settings["language"] = detected_language
        p_data["settings"] = settings

        # 4. Auto-edit planning: audio-driven, because Whisper strips the fillers
        #    out of the text before we ever see them. A precut recording skips it:
        #    the user already made every cut, and the words stay as heard.
        if precut:
            ann_words = words
            edit_report = {"precut": True}
            # Still map the audio's energy: the presentation pass keys its
            # emphasis punch-ins off it, and without it a precut take got one
            # slow push across the whole video and no zooms at all. Only the
            # envelope is kept -- speech regions would let the builder trim.
            try:
                from asr.vad import analyse_speech
                speech_map = await asyncio.to_thread(analyse_speech, audio_path)
                if speech_map is not None:
                    edit_report["energy"] = speech_map.envelope()
            except Exception as exc:  # noqa: BLE001 - zooms degrade, the edit does not
                logging.getLogger("transcription").warning(
                    "Precut: energy analysis failed (%s); no emphasis zooms.", exc)
        else:
            aggressiveness = float(settings.get("fumble_aggressiveness", 0.5))
            # `settings` lets the planner give a project its genre's pacing when
            # nobody chose one: horror lives in its silences.
            saved_plan = saved.get(ckpt.PHASE_PLAN) if isinstance(saved.get(ckpt.PHASE_PLAN), dict) else None
            if saved_plan and saved_plan.get("words"):
                _stage(job_id, "reusing the planned cut from the interrupted run")
                logging.getLogger("transcription").info(
                    "%s: reusing the planned cut from the checkpoint; not planning again.", job_id)
                ann_words = saved_plan["words"]
                edit_report = dict(saved_plan.get("report") or {})
                # Pacing the planner settled on (genre pacing, the source of it)
                # was saved with the plan; the rebuild below reads it from settings.
                for key, value in (saved_plan.get("settings") or {}).items():
                    settings.setdefault(key, value)
            else:
                _stage(job_id, "planning the cut")
                # The written script tags each utterance with the paragraph it follows
                # (a hint for grouping attempts, never a filter). It rides on the same
                # settings dict the planner writes its genre pacing onto, and is
                # popped afterwards so it is never saved with the project.
                _with_script_text(settings, p_data)
                # The planner names its steps (reads, sweep, verification, final
                # read) on this job's stage, which the Studio shows live.
                from asr import editor_planner
                editor_planner.set_progress(lambda text: _stage(job_id, text))
                try:
                    plan = await plan_auto_edit(words, audio_path, aggressiveness=aggressiveness,
                                                settings=settings)
                finally:
                    editor_planner.set_progress(None)
                    settings.pop("script_text", None)
                ann_words = plan.words
                edit_report = plan.report
                await asyncio.to_thread(
                    ckpt.save, PROJECTS_DIR, job.project_id, ckpt_key, ckpt.PHASE_PLAN,
                    {"words": ann_words, "report": edit_report,
                     "settings": {k: settings.get(k) for k in
                                  ("max_pause_seconds", "pause_padding_seconds", "pacing_source", "genre")
                                  if settings.get(k) is not None}})

        # 5. Build EDL Timeline. The rebuild and the coverage check walk every
        #    word of a 40-minute take: off the event loop, so the status polls
        #    (and the health probe the Studio's watchdog relies on) keep answering.
        _stage(job_id, "building the timeline")
        tl = await asyncio.to_thread(
            build_timeline_from_transcript,
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
            keep_full_source=precut,
        )
        # The builder rebuilds V1/A1 itself, so this only scores the result: are
        # the words the planner cut actually absent from what will be rendered?
        if not precut:
            await asyncio.to_thread(record_cut_coverage, tl, edit_report)

        # Report-only cut check: re-transcribe the finished cut and match it
        # against the plan, so the edit ships with a verdict on whether the video
        # actually plays what the transcript says. It NEVER modifies the cut —
        # trimming automatically proved too risky, so this only reports. Costs an
        # extra re-transcription of the edit; set `auto_verify_cut` false to skip.
        # Never fatal — a failure here just leaves the report without a verdict.
        if settings.get("auto_verify_cut", True) and not precut:
            try:
                from asr.cut_verify import verify_cut
                _stage(job_id, "verifying the cut")
                vr = await verify_cut(tl, source_video, language=detected_language)
                edit_report.setdefault("quality", {})["cut_verify"] = {
                    "verdict": vr["verdict"],
                    "coverage": vr["coverage"],
                    "leaked_struck_seconds": vr["leaked_struck_seconds"],
                    "leaked_struck_words": vr["leaked_struck_words"],
                    "dropped_kept_count": vr["dropped_kept_count"],
                    "plan_text": vr["expected_text"],
                    "cut_text": vr["actual_text"],
                }
            except Exception as e:
                logging.getLogger("transcription").warning(
                    "Auto cut-verify skipped (%s); shipping the cut without a verdict.", e)

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
        _stage(job_id, "saving the transcript")
        p_data["status"] = "transcribed"
        await asyncio.to_thread(_save_project_sync, job.project_id, p_data)
        # Saved on the project: the checkpoint has done its job.
        await asyncio.to_thread(ckpt.clear, PROJECTS_DIR, job.project_id)

        response = {
            "status": "completed",
            "job_id": job_id,
            "segments": len(segments),
            "timeline": tl,
            "report": public_report(edit_report),
        }
        jobs[job_id]["result"] = {"segments_count": len(segments)}
        _responses[job_id] = response
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["status"] = "completed"
        _stage(job_id, "completed")
        return response
    except Exception as e:
        logging.getLogger("transcription").exception("%s failed", job_id)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        _stage(job_id, "failed")
        raise HTTPException(status_code=500, detail=str(e))

_ANNOTATION_KEYS = ("enabled", "disfluency", "candidate", "reason", "take", "note", "_edit_report")


def _with_script_text(settings: dict, p_data: dict) -> None:
    """Put the project's written script on `settings["script_text"]` for the
    planner (asr.editor_planner tags utterances with the paragraph they
    follow). Only when the project has one; callers pop the key afterwards."""
    script = p_data.get("script")
    text = script.get("text") if isinstance(script, dict) else None
    if isinstance(text, str) and text.strip():
        settings["script_text"] = text


@router.post("/{project_id}/recut")
async def recut_project(project_id: str):
    """Plan the cut again from the words Whisper already produced, without
    transcribing again: the same audio analysis, planner and timeline rebuild
    as `/transcribe` from step 4 on. The Studio asks for it when the cut rules
    changed after a production (2026-10-08: the editor's majority rule), since
    a restart otherwise reuses the transcription -- and with it the old cut.
    Every presentation item on the timeline goes with the rebuild; the Studio
    re-presents afterwards. Runs under the transcription job id, so a caller
    that follows `trans_<id>` sees it exactly like a transcription."""
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    source_video = p_data.get("source_video")
    if not source_video or not Path(source_video).exists():
        raise HTTPException(status_code=400, detail="Source video missing")
    # Saved as a TranscriptionResult by /transcribe, but a project that went
    # through the Project model (the presentation pass) holds the segment list.
    transcript = p_data.get("transcript") or {}
    if isinstance(transcript, list):
        transcript = {"segments": transcript, "language": (p_data.get("settings") or {}).get("language")}
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    words = [dict(w) for seg in (segments or []) for w in (seg.get("words") or [])
             if not w.get("detected")]
    for w in words:
        for key in _ANNOTATION_KEYS:
            w.pop(key, None)
    if not words:
        raise HTTPException(status_code=400, detail="This project has no transcription to re-cut from")
    settings = p_data.get("settings") or {}
    if settings.get("precut"):
        raise HTTPException(status_code=400, detail="A precut recording has no cut to plan")

    job_id = f"trans_{project_id}"
    running = jobs.get(job_id)
    if running and running.get("status") == "running":
        raise HTTPException(status_code=409, detail="A transcription is already running for this project")
    jobs[job_id] = {"id": job_id, "project_id": project_id, "job_type": "recut",
                    "status": "running", "progress": 0.0}
    try:
        _stage(job_id, "extracting audio")
        audio_path = await asyncio.to_thread(extract_audio, source_video)
        v_info = await asyncio.to_thread(get_video_info, source_video)
        _stage(job_id, "planning the cut")
        _with_script_text(settings, p_data)
        try:
            plan = await plan_auto_edit(words, audio_path,
                                        aggressiveness=float(settings.get("fumble_aggressiveness", 0.5)),
                                        settings=settings)
        finally:
            settings.pop("script_text", None)
        ann_words, edit_report = plan.words, plan.report
        _stage(job_id, "building the timeline")
        tl = await asyncio.to_thread(
            build_timeline_from_transcript,
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
            keep_full_source=False,
        )
        await asyncio.to_thread(record_cut_coverage, tl, edit_report)
        first = segments[0] if segments else {}
        seg = TranscriptSegment(
            id=0, start=ann_words[0]["start"], end=ann_words[-1]["end"],
            text=first.get("text") or " ".join(w.get("word", "") for w in ann_words),
            text_native=first.get("text_native"), text_english=first.get("text_english"),
            words=ann_words, confidence=1.0)
        res = TranscriptionResult(segments=[seg], language=transcript.get("language") or settings.get("language"),
                                  duration=ann_words[-1]["end"])
        p_data["transcript"] = res.model_dump()
        p_data["timeline"] = tl.model_dump()
        p_data["status"] = "transcribed"
        p_data.pop("presented_at", None)
        _stage(job_id, "saving the cut")
        await asyncio.to_thread(_save_project_sync, project_id, p_data)
        response = {"status": "completed", "job_id": job_id, "segments": 1,
                    "report": public_report(edit_report)}
        _responses[job_id] = response
        jobs[job_id].update(status="completed", progress=1.0, result={"segments_count": 1})
        _stage(job_id, "completed")
        return response
    except HTTPException:
        raise
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


class RespellRequest(BaseModel):
    # Both default to the project's own settings (as set by /transcribe) when
    # omitted, so "just apply the glossary fix I saved" needs no body at all.
    vocabulary: Optional[List[str]] = None
    glossary: Optional[str] = None


@router.post("/{project_id}/respell")
async def respell_transcript(project_id: str, body: RespellRequest = RespellRequest()):
    """Re-run the Hinglish romanizer (common-word table, loanword restore,
    colloquial rules) and glossary over an already-transcribed project's
    `word`/`hinglish` fields, from the untouched `word_native` Whisper
    produced — no re-transcription needed. Lets a glossary edit or a
    vocabulary addition reach a project immediately instead of waiting for
    the next full transcribe.
    """
    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    transcript = p_data.get("transcript")
    if not transcript or not transcript.get("segments"):
        raise HTTPException(status_code=404, detail="Transcript not found")

    settings = p_data.get("settings") or {}
    language = transcript.get("language") or settings.get("language")
    vocabulary = body.vocabulary if body.vocabulary is not None else settings.get("vocabulary")
    glossary_key = body.glossary if body.glossary is not None else settings.get("glossary")
    glossary_map = glossary_store.load_merged(glossary_key)

    words_changed = 0
    for seg in transcript["segments"]:
        words = seg.get("words") or []
        tokens = []
        for w in words:
            native = w.get("word_native")
            if not native:
                tokens.append(str(w.get("word") or w.get("hinglish") or ""))
                continue
            new_word = to_hinglish(native, language, vocabulary=vocabulary, glossary=glossary_map)
            if new_word != w.get("word"):
                words_changed += 1
            w["word"] = new_word
            w["hinglish"] = new_word
            tokens.append(new_word)
        joined = " ".join(t for t in tokens if t).strip()
        if joined:
            seg["text"] = joined

    p_data["transcript"] = transcript
    project_store.save_project(project_id, p_data)

    return {
        "status": "ok",
        "project_id": project_id,
        "words_changed": words_changed,
        "language": language,
        "glossary": glossary_key,
    }


class ReplaceWordsRequest(BaseModel):
    find: str
    replace: str
    word_ids: Optional[List[str]] = None
    match_case: bool = False
    whole_word: bool = True


def _word_matches(token: str, find: str, match_case: bool, whole_word: bool) -> bool:
    if whole_word:
        return token == find if match_case else token.lower() == find.lower()
    return (find in token) if match_case else (find.lower() in token.lower())


def _word_replaced(token: str, find: str, replace: str, match_case: bool, whole_word: bool) -> str:
    if whole_word:
        return replace
    flags = 0 if match_case else re.IGNORECASE
    return re.sub(re.escape(find), lambda _m: replace, token, flags=flags)


def _sentence_contains(text: str, find: str, match_case: bool, whole_word: bool) -> bool:
    pattern = (r"\b" + re.escape(find) + r"\b") if whole_word else re.escape(find)
    flags = 0 if match_case else re.IGNORECASE
    return re.search(pattern, text, flags=flags) is not None


def _sentence_replaced(text: str, find: str, replace: str, match_case: bool, whole_word: bool) -> str:
    pattern = (r"\b" + re.escape(find) + r"\b") if whole_word else re.escape(find)
    flags = 0 if match_case else re.IGNORECASE
    return re.sub(pattern, lambda _m: replace, text, flags=flags)


@router.post("/{project_id}/replace_words")
async def replace_words(project_id: str, body: ReplaceWordsRequest):
    """Find/replace across the transcript: the word list, the stored transcript
    segments (matched back to the exact word touched, by id or by timing), and
    any burned-in caption text that shows the same token.
    """
    if not body.find:
        raise HTTPException(status_code=400, detail="find must not be empty")

    p_data = project_store.get_project(project_id)
    if not p_data or "timeline" not in p_data:
        raise HTTPException(status_code=404, detail="Project or timeline not found")

    tl = Timeline.model_validate(p_data["timeline"])
    tol_seconds = 1.0 / (tl.fps_num / max(1, tl.fps_den))
    id_filter = set(body.word_ids) if body.word_ids else None

    changed = []
    for word in tl.words:
        if id_filter is not None and word.id not in id_filter:
            continue
        # The UI searches whichever script it shows, so a hit can be on the
        # romanized `text` or the native `word_native` (Devanagari etc.).
        hit_text = _word_matches(word.text, body.find, body.match_case, body.whole_word)
        hit_native = bool(word.word_native) and _word_matches(
            word.word_native, body.find, body.match_case, body.whole_word)
        if not (hit_text or hit_native):
            continue
        if hit_text:
            word.text = _word_replaced(word.text, body.find, body.replace, body.match_case, body.whole_word)
        if hit_native:
            word.word_native = _word_replaced(
                word.word_native, body.find, body.replace, body.match_case, body.whole_word)
        elif word.word_native and not body.replace.isascii():
            # A native-script correction typed against the romanized form:
            # native captions should show it too. A Latin replacement must not
            # overwrite the Devanagari the native captions render from.
            word.word_native = body.replace
        changed.append(word)

    replaced = len(changed)

    if replaced:
        # Mirror into the stored transcript, matching each segment word back to
        # the timeline word it came from (by id, else by its own timestamp).
        transcript = p_data.get("transcript")
        if transcript and transcript.get("segments"):
            for seg in transcript["segments"]:
                seg_words = seg.get("words") or []
                touched = False
                for tw in seg_words:
                    match = next((w for w in changed if tw.get("id") and tw.get("id") == w.id), None)
                    if match is None:
                        start = tw.get("start")
                        if start is None:
                            continue
                        match = next(
                            (w for w in changed
                             if abs(float(start) - frame_to_time(w.start_frame, tl.fps_num, tl.fps_den))
                             <= tol_seconds),
                            None)
                    if match is None:
                        continue
                    tw["word"] = match.text
                    tw["hinglish"] = match.text
                    if "word_native" in tw and match.word_native:
                        tw["word_native"] = match.word_native
                    touched = True
                if touched:
                    tokens = [str(w.get("word") or w.get("hinglish") or "") for w in seg_words]
                    seg["text"] = " ".join(t for t in tokens if t).strip()
            p_data["transcript"] = transcript

        # Mirror into any burned-in caption clips wherever the old token shows.
        for item in tl.items:
            if item.track not in (CAPTION_TRACK, "CAP") or not item.text:
                continue
            for tw in (item.text.words or []):
                text = tw.get("text")
                if text and _word_matches(text, body.find, body.match_case, body.whole_word):
                    tw["text"] = _word_replaced(
                        text, body.find, body.replace, body.match_case, body.whole_word)
            if item.text.content and _sentence_contains(
                    item.text.content, body.find, body.match_case, body.whole_word):
                item.text.content = _sentence_replaced(
                    item.text.content, body.find, body.replace, body.match_case, body.whole_word)

    p_data["timeline"] = tl.model_dump()
    project_store.save_project(project_id, p_data)
    return {"status": "success", "replaced": replaced, "timeline": tl}


@router.post("/{project_id}/verify-cut")
async def verify_cut_route(project_id: str):
    """Ground-truth check: re-transcribe the actual cut and diff it against the
    intended transcript. `verdict` is 'clean', 'leaks_removed_speech' (the video
    plays fumbles the plan removed — a timing problem) or 'drops_kept_speech'.
    Heavy: it re-transcribes the edit, so call it deliberately, not on every save.
    """
    from timeline.schema import Timeline
    from asr.cut_verify import verify_cut

    p_data = project_store.get_project(project_id)
    if not p_data:
        raise HTTPException(status_code=404, detail="Project not found")
    tl_data = p_data.get("timeline")
    source = p_data.get("source_video")
    if not tl_data or not source:
        raise HTTPException(status_code=400, detail="Project has no timeline or source video")
    timeline = Timeline.model_validate(tl_data)
    language = (p_data.get("settings") or {}).get("language")
    report = await verify_cut(timeline, source, language=language)
    # The bulky side of the report is the two full transcripts; keep them, they
    # are exactly what makes a leak legible when the verdict is not 'clean'.
    return report
