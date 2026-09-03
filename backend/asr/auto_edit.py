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
) -> AutoEditPlan:
    """Run the planner over `words` and collect its report."""
    if use_llm:
        release_asr_gpu()
    refined = await refine_disfluencies(
        words,
        aggressiveness=aggressiveness,
        use_llm=use_llm,
        audio_path=audio_path,
        detect_fillers=detect_fillers,
    )
    return AutoEditPlan(words=refined, report=last_report(refined), audio_path=audio_path)


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
