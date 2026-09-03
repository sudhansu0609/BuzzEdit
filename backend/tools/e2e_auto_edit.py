"""End-to-end validation of the auto-edit on a real recording, with a live model.

Mirrors POST /api/transcription/transcribe exactly — extract audio, transcribe,
plan the auto-edit (all LLM passes live), build the timeline, score the cut —
but from the command line, with every intermediate dumped to a directory so the
run can be inspected afterwards. This is the acceptance run §6 of
AUTO_EDIT_IMPROVEMENT_PLAN.md calls for.

    ./.venv/Scripts/python.exe backend/tools/e2e_auto_edit.py <video> [project_id]

Creates (or reuses) a real project in data/projects so the app can open the
result in the morning. Artifacts land in data/projects/<id>_e2e/.
"""

import asyncio
import json
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECTS_DIR  # noqa: E402


def _setup_logging(log_path: Path) -> None:
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(console)


async def main(video: str, project_id: str = None, language: str = None) -> int:
    from models import Project
    from store.project_store import ProjectStore
    from utils.ffmpeg_utils import extract_audio, get_video_info
    from asr import whisper_engine
    from asr.auto_edit import plan_auto_edit, public_report, record_cut_coverage
    from timeline import Transition, build_timeline_from_transcript

    store = ProjectStore(base_dir=str(PROJECTS_DIR))
    source = Path(video)
    if not source.exists():
        print(f"no such video: {video}")
        return 2

    if project_id and store.get_project(project_id):
        data = store.get_project(project_id)
        source_video = data["source_video"]
    else:
        project_id = project_id or f"e2e_{uuid.uuid4().hex[:6]}"
        dest = PROJECTS_DIR / f"{project_id}_source{source.suffix}"
        if not dest.exists():
            shutil.copy(str(source), str(dest))
        project = Project(id=project_id, name=source.name,
                          source_video=str(dest), status="draft")
        data = project.model_dump()
        store.save_project(project_id, data)
        source_video = str(dest)

    out_dir = PROJECTS_DIR / f"{project_id}_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir / "run.log")
    log = logging.getLogger("e2e")
    timings = {}

    log.info("=== E2E auto-edit run: %s (%s) ===", project_id, source_video)

    t0 = time.time()
    audio_path = await asyncio.to_thread(extract_audio, source_video)
    v_info = await asyncio.to_thread(get_video_info, source_video)
    timings["extract_audio"] = round(time.time() - t0, 1)

    settings = data.get("settings") or {}
    language = language or settings.get("language")

    t0 = time.time()
    full = await whisper_engine.transcribe_full_async(audio_path, language=language)
    timings["transcribe"] = round(time.time() - t0, 1)
    words = full["words"]
    detected_language = full["language"]
    settings["language"] = detected_language
    data["settings"] = settings
    log.info("Transcribed %d words, language=%s", len(words), detected_language)
    (out_dir / "words_raw.json").write_text(
        json.dumps(words, indent=1, ensure_ascii=False), encoding="utf-8")

    t0 = time.time()
    plan = await plan_auto_edit(words, audio_path,
                                aggressiveness=float(settings.get("fumble_aggressiveness", 0.5)))
    timings["plan_auto_edit"] = round(time.time() - t0, 1)
    ann_words = plan.words
    edit_report = plan.report

    t0 = time.time()
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
                            else Transition(type="fade",
                                            duration=float(settings.get("transition_seconds", 0.25)))),
    )
    record_cut_coverage(tl, edit_report)
    timings["build_timeline"] = round(time.time() - t0, 1)

    t0 = time.time()
    cut_verify_report = None
    try:
        from asr.cut_verify import verify_cut
        cut_verify_report = await verify_cut(tl, source_video, language=detected_language)
        edit_report.setdefault("quality", {})["cut_verify"] = {
            k: cut_verify_report[k] for k in
            ("verdict", "coverage", "leaked_struck_seconds", "leaked_struck_words",
             "dropped_kept_count")}
    except Exception as e:
        log.warning("cut_verify failed: %s", e)
    timings["cut_verify"] = round(time.time() - t0, 1)

    data["timeline"] = tl.model_dump()
    data["status"] = "transcribed"
    store.save_project(project_id, data)

    # ---- artifacts ---------------------------------------------------------
    (out_dir / "words_annotated.json").write_text(
        json.dumps(ann_words, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    pub = public_report(edit_report)
    pub["timings_s"] = timings
    (out_dir / "report.json").write_text(
        json.dumps(pub, indent=2, ensure_ascii=False), encoding="utf-8")
    if cut_verify_report:
        (out_dir / "cut_verify.json").write_text(
            json.dumps(cut_verify_report, indent=2, ensure_ascii=False), encoding="utf-8")

    kept = [w for w in ann_words if w.get("enabled")]
    cut = [w for w in ann_words if not w.get("enabled")]
    (out_dir / "kept_hinglish.txt").write_text(
        " ".join(str(w.get("hinglish") or w.get("word") or "") for w in kept),
        encoding="utf-8")
    (out_dir / "kept_native.txt").write_text(
        " ".join(str(w.get("word_native") or w.get("word") or "") for w in kept),
        encoding="utf-8")
    # The full annotated stream: struck words in [brackets] with their reason.
    lines = []
    for w in ann_words:
        token = str(w.get("word_native") or w.get("word") or "").strip()
        if w.get("enabled"):
            lines.append(token)
        else:
            lines.append(f"[{token}|{w.get('reason') or '?'}]")
    (out_dir / "edit_marked.txt").write_text(" ".join(lines), encoding="utf-8")

    v1 = sorted((i for i in tl.items if i.track == "V1"), key=lambda i: i.timeline_start_frame)
    fps = tl.fps_num / max(1, tl.fps_den)
    seg_lines = [
        f"{i.timeline_start_frame / fps:8.2f} - {i.timeline_end_frame / fps:8.2f}  "
        f"src {i.source_start_frame / fps:8.2f} - {i.source_end_frame / fps:8.2f}"
        for i in v1]
    (out_dir / "v1_segments.txt").write_text("\n".join(seg_lines), encoding="utf-8")

    # ---- verdict summary ---------------------------------------------------
    quality = edit_report.get("quality") or {}
    program_s = sum((i.timeline_end_frame - i.timeline_start_frame) for i in v1) / fps
    summary = {
        "project": project_id,
        "language": detected_language,
        "words_total": edit_report.get("words_total"),
        "words_cut": edit_report.get("words_cut"),
        "reasons": edit_report.get("reasons"),
        "seconds_removed": round(v_info.get("duration", 0.0) - program_s, 1),
        "program_seconds": round(program_s, 1),
        "used_fluency": edit_report.get("used_fluency"),
        "fluency_windows": edit_report.get("fluency_windows"),
        "take_swaps": edit_report.get("take_swaps"),
        "final_read_cuts": edit_report.get("final_read_cuts"),
        "verdict": quality.get("verdict"),
        "rounds": quality.get("rounds"),
        "cut_words_still_audible": quality.get("cut_words_still_audible"),
        "kept_words_dropped": quality.get("kept_words_dropped"),
        "cut_verify": quality.get("cut_verify"),
        "warnings": pub.get("warnings"),
        "timings_s": timings,
        "v1_segments": len(v1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/e2e_auto_edit.py <video> [project_id] [language]")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else None)))
