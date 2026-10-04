import asyncio
import logging
import re
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import TranscribeJob, TranscriptionResult, TranscriptSegment
from config import PROJECTS_DIR
from store.project_store import ProjectStore
from store import glossary_store
from utils.ffmpeg_utils import extract_audio, get_video_info
from asr import whisper_engine
from asr.auto_edit import plan_auto_edit, public_report, record_cut_coverage
from asr.transliterate import to_hinglish
from timeline import Transition, build_timeline_from_transcript
from timeline.authoring import CAPTION_TRACK
from timeline.schema import Timeline, frame_to_time

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
        vocabulary = settings.get("vocabulary")
        glossary_key = settings.get("glossary")
        glossary_map = glossary_store.load_merged(glossary_key)
        full = await whisper_engine.transcribe_full_async(
            audio_path, language=language, vocabulary=vocabulary, glossary=glossary_map)
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
            plan = await plan_auto_edit(words, audio_path, aggressiveness=aggressiveness,
                                        settings=settings)
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
            keep_full_source=precut,
        )
        # The builder rebuilds V1/A1 itself, so this only scores the result: are
        # the words the planner cut actually absent from what will be rendered?
        if not precut:
            record_cut_coverage(tl, edit_report)

        # Report-only cut check: re-transcribe the finished cut and match it
        # against the plan, so the edit ships with a verdict on whether the video
        # actually plays what the transcript says. It NEVER modifies the cut —
        # trimming automatically proved too risky, so this only reports. Costs an
        # extra re-transcription of the edit; set `auto_verify_cut` false to skip.
        # Never fatal — a failure here just leaves the report without a verdict.
        if settings.get("auto_verify_cut", True) and not precut:
            try:
                from asr.cut_verify import verify_cut
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
