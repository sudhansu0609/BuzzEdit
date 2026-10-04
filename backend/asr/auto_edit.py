"""One way to run the auto-edit, shared by every entry point.

Four routes used to run the planner, each assembling the call themselves, and
they had drifted apart: one forgot to extract the audio at all (so the planner
silently degraded to the text-only behaviour that finds ~1% of the fumbles), one
forgot to hand the energy envelope to the timeline (so cuts stopped landing on
the quiet moment between two sounds), and one computed the edit report and threw
it away. The differences were invisible because every one of them still returned
a plausible-looking timeline.

So the assembly lives here once. A caller supplies words and a source video; it
gets back the annotated words and a report, and the two helpers below put the
audio-derived parts of that report where the timeline rebuild can reach them.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .fumble_engine import last_report, refine_disfluencies

logger = logging.getLogger("auto_edit")

# Payloads that belong to the timeline, not to an HTTP response: the speech map
# is hundreds of intervals and the energy envelope is one integer per 20ms of
# the recording. Sending them to the UI once cost megabytes per transcribe.
_BULK_KEYS = ("speech", "energy")


@dataclass
class AutoEditPlan:
    """The planner's output: annotated words plus everything it measured."""
    words: List[Dict[str, Any]]
    report: Dict[str, Any]
    audio_path: Optional[str] = None

    @property
    def public_report(self) -> Dict[str, Any]:
        """The report as the UI should see it — without the bulk payloads."""
        return {k: v for k, v in self.report.items() if k not in _BULK_KEYS}


async def extract_project_audio(source_video: Optional[str]) -> Optional[str]:
    """The WAV the planner needs, or None if it cannot be made.

    Never raises: a missing audio track degrades the edit, it does not fail the
    request. The degradation is recorded in the report as `used_audio: False` so
    the UI can say so rather than quietly shipping a near-empty edit.
    """
    if not source_video or not Path(source_video).exists():
        return None
    try:
        from utils.ffmpeg_utils import extract_audio
        return await asyncio.to_thread(extract_audio, source_video)
    except Exception as e:
        logger.warning("Auto-edit: could not extract audio from %s (%s); "
                       "this will be a text-only pass, which finds very little.",
                       source_video, e)
        return None


def release_asr_gpu() -> float:
    """Hand the GPU from the ASR to the language model. Returns MB freed.

    Every path runs the ASR first and the model passes second, in one process,
    and the ASR was holding its VRAM through all of them. Measured on the
    reference machine: a 16.3GB card, ~4GB still held by faster-whisper and the
    MMS aligner, and a language model that could not fit in the rest — so
    llama-server answered `{"error":"terminated"}` and every model-driven layer
    (fluency, the verification loop, best-take, the final read) silently did
    nothing. The edit that shipped was the structural fallback.

    Never fatal: if the release fails the edit still runs, it just runs in the
    conditions that caused the failure.
    """
    try:
        from .faster_whisper_engine import whisper_engine
        return whisper_engine.release_gpu()
    except Exception as e:
        logger.warning("Could not release the ASR's GPU memory (%s); the model "
                       "passes will run alongside it.", e)
        return 0.0


async def plan_auto_edit(
    words: List[Dict[str, Any]],
    audio_path: Optional[str],
    aggressiveness: float = 0.5,
    use_llm: bool = True,
    detect_fillers: bool = True,
    settings: Optional[Dict[str, Any]] = None,
) -> AutoEditPlan:
    """Run the planner over `words` and collect its report.

    Pass the project's `settings` and the genre's pacing is filled in when nobody
    chose one (`apply_genre_pacing`); `pacing_kwargs(settings)` then hands the
    pacing to the timeline build.
    """
    if use_llm:
        release_asr_gpu()
    planner = choose_planner(settings) if use_llm else "classic"
    refined, fallback = None, None
    if planner == "editor":
        from .editor_planner import EditorUnavailable, plan_with_editor
        try:
            refined = await plan_with_editor(words, audio_path, settings)
        except EditorUnavailable as e:
            fallback = str(e)
        except Exception as e:                       # the editor must never fail the edit
            logger.exception("Auto-edit: the AI editor failed; using the classic planner.")
            fallback = f"{type(e).__name__}: {e}"
        if fallback:
            logger.warning("Auto-edit: AI editor unavailable (%s); classic planner instead.", fallback)
    if refined is None:
        refined = await refine_disfluencies(
            words,
            aggressiveness=aggressiveness,
            use_llm=use_llm,
            audio_path=audio_path,
            detect_fillers=detect_fillers,
        )
    report = last_report(refined)
    report.setdefault("planner", "classic")
    if fallback:
        report["planner_fallback"] = fallback
    if settings is not None:
        genre = await apply_genre_pacing(settings, words, model_ready=bool(report.get("used_llm"))
                                         and report.get("planner") != "editor")  # LM Studio not loaded
        if str(settings.get("pacing_source") or "").startswith(_GENRE_SOURCE):
            report["pacing"] = {"genre": genre, **pacing_kwargs(settings)}
    return AutoEditPlan(words=refined, report=report, audio_path=audio_path)


PLANNERS = ("editor", "classic")
DEFAULT_PLANNER = "editor"


def choose_planner(settings: Optional[Dict[str, Any]]) -> str:
    """Which planner cuts this project: the project's `planner` setting, else the app setting
    `auto_cut_planner`, else the AI editor (AUTO_CUT_EDITOR_PLAN.md Phase 7). "classic" is the
    rule + local-model planner (fumble_engine), kept as the offline fallback."""
    value = (settings or {}).get("planner")
    if value not in PLANNERS:
        try:
            from store.app_settings import AppSettings
            value = AppSettings().get("auto_cut_planner")
        except Exception:
            value = None
    return value if value in PLANNERS else DEFAULT_PLANNER


def pacing_kwargs(settings: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """The project's pacing as `build_timeline_from_transcript` keywords; None
    leaves the Timeline default."""
    settings = settings or {}
    return {"max_pause_seconds": settings.get("max_pause_seconds"),
            "pause_padding_seconds": settings.get("pause_padding_seconds")}


_GENRE_SOURCE = "genre:"


async def apply_genre_pacing(settings: Dict[str, Any], words: List[Dict[str, Any]],
                             model_ready: bool = False) -> Optional[str]:
    """Give a project its genre's pacing when nobody chose one; returns the genre,
    or None when a chosen pacing stands.

    A pacing the channel sent (the Studio's `auto_edit` block) or the user set
    always wins. Otherwise the genre decides — horror keeps its pauses, see
    `presentation.genre.pacing_for`. The genre is the caller's when it named one
    (`settings["genre"]`), else the model's reading of the opening minutes. Not
    keywords alone: on a Raat3Baje horror story that never says "bhoot" they
    voted "cooking". `model_ready` says the planner's model is already loaded;
    without it this does not start one, and keywords are all there is.

    The pacing is written onto `settings`, so every later rebuild keeps it, with
    `pacing_source` naming the genre — values still equal to what that genre gives
    are the genre's to re-decide on the next run; anything else is a choice.
    """
    from presentation.genre import detect_genre_in_text, normalise, pacing_for

    source = str(settings.get("pacing_source") or "")
    current = (settings.get("max_pause_seconds"), settings.get("pause_padding_seconds"))
    from_genre = (source.startswith(_GENRE_SOURCE)
                  and current == pacing_for(source[len(_GENRE_SOURCE):]))
    if any(value is not None for value in current) and not from_genre:
        return None

    genre = settings.get("genre")
    if not genre:
        ask = None
        if model_ready:
            from llm.client import lm_studio_client

            async def ask(system_prompt: str, user_prompt: str, schema=None):
                return await lm_studio_client.ask_with_schema(system_prompt, user_prompt, schema or {})

        opening = " ".join(str(w.get("word") or "") for w in words
                           if float(w.get("start") or 0.0) < 300.0)
        whole = " ".join(str(w.get("word") or "") for w in words)
        genre = await detect_genre_in_text(opening, whole, ask)
    genre = normalise(genre)

    pacing = pacing_for(genre)
    if pacing:
        settings["max_pause_seconds"], settings["pause_padding_seconds"] = pacing
        settings["pacing_source"] = _GENRE_SOURCE + genre
        logger.info("Auto-edit pacing from the genre (%s): pauses up to %.2fs kept, "
                    "%.2fs either side of a cut", genre, *pacing)
    elif from_genre:
        # The genre that set these no longer applies: back to the defaults.
        for key in ("max_pause_seconds", "pause_padding_seconds", "pacing_source"):
            settings.pop(key, None)
    return genre


def apply_report_to_timeline(timeline, report: Dict[str, Any]) -> None:
    """Copy the audio-derived parts of a report onto an existing timeline.

    Both are load-bearing for the rebuild: `speech_regions` is what lets it cut
    silence out of the *middle* of a word, and `energy_envelope` is what lets each
    cut land on the quietest instant instead of on the ASR's ±50ms guess.
    """
    from timeline.schema import time_to_frame

    if report.get("speech"):
        timeline.speech_regions = [
            [time_to_frame(start, timeline.fps_num, timeline.fps_den),
             time_to_frame(end, timeline.fps_num, timeline.fps_den)]
            for start, end in report["speech"]
        ]
    if report.get("energy"):
        timeline.energy_envelope = report["energy"]


def rebuild_and_check(timeline, primary_source_id: str, report: Dict[str, Any]) -> None:
    """Rebuild V1/A1, then record whether the render matches the plan."""
    from timeline.ops import rebuild_primary_tracks

    rebuild_primary_tracks(timeline, primary_source_id)
    record_cut_coverage(timeline, report)


def record_cut_coverage(timeline, report: Dict[str, Any]) -> None:
    """Record whether the render actually matches the plan.

    The word list is the plan; the render is the truth. A word marked cut is only
    really cut if no V1 segment still covers it, and for a long time short cuts
    were being silently glued back in by the rebuild's gap bridging — the
    transcript panel showed them struck out while the video still played them.
    The check is cheap, so it runs on every path and its answer ships in the
    report instead of being assumed.
    """
    from timeline.ops import audit_cut_coverage

    if report.get("planner") == "editor":
        _finish_editor_cut(timeline, report)
    else:
        timeline.review = []        # a classic plan replaces an editor plan's open questions
    coverage = audit_cut_coverage(timeline)
    quality = report.setdefault("quality", {})
    quality["cut_words_still_audible"] = coverage["still_audible"]
    quality["cut_words_checked"] = coverage["checked"]
    quality["kept_words_dropped"] = coverage.get("kept_but_dropped", 0)
    if coverage["still_audible"]:
        logger.warning(
            "Auto-edit: %d of %d cut words are still covered by the render "
            "(examples: %s). The edit will play them.",
            coverage["still_audible"], coverage["checked"],
            ", ".join(coverage["examples"]) or "-")


def _finish_editor_cut(timeline, report: Dict[str, Any]) -> None:
    """After the AI editor's cut is built: the listening check's small nudges, and the
    "decided under doubt" list onto the timeline, where the Cuts view reads it and it
    survives with the project (AUTO_CUT_EDITOR_PLAN.md §4.6, §4.7)."""
    try:
        from .cut_verify import nudge_cut_edges
        primary = next(iter(timeline.sources.keys()), None) if timeline.sources else None
        if primary and not timeline.keep_full_source:
            listened = nudge_cut_edges(timeline, primary)
            report.setdefault("quality", {})["cut_nudges"] = {
                k: v for k, v in listened.items() if k != "review"}
            report.setdefault("review", []).extend(listened.get("review") or [])
    except Exception as e:
        logger.warning("Listening check skipped (%s).", e)
    timeline.review = list(report.get("review") or [])


def public_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """A report safe to put in an HTTP response.

    Also distils the quality checks into one `warnings` list of plain sentences,
    so no caller has to know which nested counter means trouble — the audits ran
    advisory for a long time and their findings never reached the user.
    """
    out = {k: v for k, v in (report or {}).items() if k not in _BULK_KEYS}
    quality = out.get("quality") or {}
    warnings: List[str] = []
    audible = quality.get("cut_words_still_audible") or 0
    if audible:
        warnings.append(f"{audible} cut words are still audible in the render")
    dropped = quality.get("kept_words_dropped") or 0
    if dropped:
        warnings.append(f"{dropped} kept words were dropped from the render")
    verdict = str(quality.get("verdict") or "")
    if verdict.startswith("still_broken"):
        count = verdict.split(":", 1)[-1]
        warnings.append(f"{count} sentences still read as broken after repair")
    elif verdict == "not_verified":
        warnings.append("the finished edit could not be verified by the model")
    # The loudest thing this report can say. Every model-driven layer — the
    # fluency pass, the verification loop, best-take, the final read — can fail
    # as a group, and when it does the edit that ships is the structural
    # fallback: filler vocabulary and repeated runs only, no judgement about
    # whether what remains is a sentence. That happened for a long time on a
    # machine whose configured model was larger than its GPU, and the only trace
    # was `used_llm: False` in a log nobody reads.
    if out.get("planner_fallback"):
        warnings.append("the AI editor could not run (" + str(out["planner_fallback"])
                        + "), so the classic planner made this cut")
    review = [r for r in out.get("review") or [] if r.get("answer") is None]
    if review:
        warnings.append(f"{len(review)} calls were decided under doubt and kept; "
                        "they are listed in the Cuts view")
    if not out.get("used_fluency"):
        warnings.append("the language model never ran, so this is a structural "
                        "edit only — check that LM Studio is up and its model "
                        "fits in the GPU")
    fluency_windows = out.get("fluency_windows") or {}
    failed = fluency_windows.get("failed") or 0
    if failed:
        warnings.append(f"{failed} fluency windows got no answer from the model; "
                        "those parts fell back to structural cuts")
    # The planner asks the model to name the runs to delete; a model that ignores
    # that and rewrites the transcript instead is handled, but the edit is then
    # being made by the older, guessier contract and the user should know which
    # one produced their cut.
    spans = fluency_windows.get("spans")
    rewrites = fluency_windows.get("rewrites") or 0
    if spans == 0 and rewrites:
        warnings.append("the model would not name the cuts, so all "
                        f"{rewrites} windows were read as rewrites instead")
    declined = fluency_windows.get("none_answers") or 0
    if spans and declined == spans:
        warnings.append(f"the model found nothing to cut in any of its {spans} "
                        "windows; the edit is structural only")
    if warnings:
        out["warnings"] = warnings
    return out
