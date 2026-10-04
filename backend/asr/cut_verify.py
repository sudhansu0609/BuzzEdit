"""Ground-truth check: does the CUT VIDEO actually say the CUT TRANSCRIPT?

Every other check in the pipeline reasons about the *plan* — the enabled words,
the V1 spans, the coverage audit. None of them listen to what the edit actually
plays. This one does: it reconstructs the cut's audio exactly the way the
compiler builds A1 (the same `source_start_frame..source_end_frame` spans,
concatenated in order), re-transcribes that audio with Whisper, and lines the
result up against the transcript the edit claims to be — the enabled words.

If they match, the video cut is proper. If the re-transcription contains words
that are NOT in the plan, the cut is leaking removed speech (a fumble the render
still plays); if it is MISSING planned words, the cut is dropping kept speech.
Either way the mismatch is named, with the actual words, so the fault is visible
instead of inferred.
"""

import difflib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from timeline.schema import Timeline, frame_to_time
from .fluency import normalise
from .tokens import romanized, spoken

logger = logging.getLogger("cut_verify")


def _cut_spans(timeline: Timeline) -> List[Tuple[float, float]]:
    """The kept source spans, in seconds, exactly as the compiler trims them.

    Reads the A1 audio items (falling back to V1) so the spans are the ones the
    render's audio is actually built from — not the word list, which is the plan
    this check exists to test.
    """
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    items = [i for i in timeline.items if i.track == "A1"]
    if not items:
        items = [i for i in timeline.items if i.track == "V1"]
    items = sorted(items, key=lambda i: i.timeline_start_frame)
    spans: List[Tuple[float, float]] = []
    for i in items:
        start = frame_to_time(i.source_start_frame, fps_num, fps_den)
        end = frame_to_time(i.source_end_frame, fps_num, fps_den)
        if end > start:
            spans.append((start, end))
    return spans


# A short silence dropped between the joined spans. It is NOT in the render — it
# exists only so Whisper does not loop-hallucinate across the abrupt seam where
# two unrelated moments of speech butt together (a hard concat with no gap makes
# the model repeat the last phrase over and over, which reads as a false "leak").
# The speech and its order are unchanged, so the transcript is faithful.
_SEAM_SILENCE = 0.20


def build_cut_audio(timeline: Timeline, source_path: str, out_wav: str,
                    seam_silence: float = _SEAM_SILENCE) -> Optional[str]:
    """Concatenate the kept source spans into one WAV — the edit's real audio.

    Mirrors the compiler's A1 path (atrim per span in `source_start..end` order).
    `seam_silence` drops a short gap between spans so the re-transcription is not
    corrupted by seam hallucination — good for a coverage read, but pass 0 when
    the caller needs cut-time to map linearly back to source frames (the repair
    path). Returns the path, or None if there is nothing to build.
    """
    from utils.ffmpeg_utils import run_ffmpeg

    spans = _cut_spans(timeline)
    if not spans:
        return None
    pad = f",apad=pad_dur={seam_silence}" if seam_silence > 0 else ""
    parts = [f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS{pad}[a{k}]"
             for k, (s, e) in enumerate(spans)]
    concat_inputs = "".join(f"[a{k}]" for k in range(len(spans)))
    filter_complex = ";".join(parts) + \
        f";{concat_inputs}concat=n={len(spans)}:v=0:a=1[out]"
    run_ffmpeg([
        "-i", source_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "16000", "-ac", "1",
        "-y", out_wav,
    ])
    return out_wav


# A struck word with more than this much of its audio inside a kept span is a
# real leak — the cut is replaying removed speech. Below it is just the breath
# padding a cut deliberately keeps either side, and must not be trimmed.
_LEAK_TOLERANCE_SECONDS = 0.2


def struck_audio_leak(timeline: Timeline) -> Dict[str, Any]:
    """Deterministic leak check — no transcription, no hallucination.

    How much of each STRUCK word's audio a kept span actually plays, by real
    frame overlap. Padding-sized overlaps are breath and ignored; an overlap past
    `_LEAK_TOLERANCE_SECONDS` is the cut replaying a removed word, and its covered
    span is returned in `leaked_regions` ready to trim. Enabled words no span
    covers at all are reported as dropped. This is the reliable signal — the one
    the verdict and the repair are built on — because it cannot be fooled by a
    re-transcription hallucination or a romanization spelling difference.
    """
    fps = timeline.fps_num / max(1, timeline.fps_den)
    spans = [(int(round(s * fps)), int(round(e * fps))) for s, e in _cut_spans(timeline)]
    tolerance = _LEAK_TOLERANCE_SECONDS * fps
    leaked_s = 0.0
    leaked_words: List[str] = []
    leaked_regions: List[List[int]] = []
    dropped: List[str] = []
    for w in timeline.words:
        if w.end_frame <= w.start_frame:
            continue
        overlaps = [[max(w.start_frame, a), min(w.end_frame, b)]
                    for a, b in spans if min(w.end_frame, b) > max(w.start_frame, a)]
        covered = sum(b - a for a, b in overlaps)
        if not w.enabled:
            leaked_s += covered / fps
            if covered > tolerance:
                leaked_words.append(w.text)
                leaked_regions.extend(overlaps)
        elif covered == 0:
            dropped.append(w.text)
    return {
        "leaked_struck_seconds": round(leaked_s, 2),
        "leaked_struck_words": leaked_words[:25],
        "leaked_regions": leaked_regions,
        "dropped_kept_words": dropped[:25],
        "dropped_kept_count": len(dropped),
    }


def _planned(word) -> str:
    """A planned word as the SPOKEN script, for comparing against a re-transcription.

    Comparing romanizations made this whole check unreadable: re-transcribing the
    reference project's own unchanged cut audio scored 0.55 coverage purely
    because Whisper romanised the same sounds differently the second time —
    `gayaa` for `gae`, `hai` for `hain`, `par` for `pe`, `doston` for `dosto`.
    Nothing was missing; the spelling had moved. The native script is the model's
    actual output rather than a lossy post-process of it, so it comes back the
    same and the coverage number means what it says.
    """
    return normalise(getattr(word, "word_native", None) or word.text)


def expected_tokens(timeline: Timeline) -> List[str]:
    """Normalised tokens the cut is supposed to say — the enabled words in order."""
    return [_planned(w) for w in timeline.words
            if w.enabled and w.end_frame > w.start_frame and _planned(w)]


def _collapse_repeats(tokens: Sequence[str], run: int = 3) -> List[str]:
    """Fold a run of `run`+ identical consecutive tokens down to one.

    Whisper's seam hallucination is a token repeated many times ("hai hai hai
    …"); a speaker's deliberate emphasis is at most a double ("bahut bahut"). So
    collapsing only runs of three or more strips the hallucination while leaving
    real repetition alone — which keeps the coverage score honest."""
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = i
        while j < n and tokens[j] == tokens[i]:
            j += 1
        length = j - i
        out.extend([tokens[i]] * (1 if length >= run else length))
        i = j
    return out


def diff_transcripts(expected: Sequence[str], actual: Sequence[str]) -> Dict[str, Any]:
    """Score how much of the PLANNED transcript the re-transcribed cut actually says.

    `coverage` is the load-bearing number: the fraction of planned tokens found,
    in order, in the re-transcription. It answers "does the video say what the
    edit claims?" and is robust to Whisper inventing extra words at a seam —
    those simply do not match. `missing` are the planned tokens the video never
    says (kept speech the cut dropped) — the real fault this catches. Extra
    re-transcribed tokens are reported for context but are noisy (hallucination),
    so leaks are judged deterministically by `struck_audio_leak`, not here.
    """
    exp = list(expected)
    act = _collapse_repeats(list(actual))
    matcher = difflib.SequenceMatcher(a=exp, b=act, autojunk=False)
    matched = sum(size for _, _, size in matcher.get_matching_blocks())
    missing: List[str] = []
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            missing.extend(exp[i1:i2])
    return {
        "coverage": round(matched / len(exp), 4) if exp else 1.0,
        "expected_words": len(exp),
        "actual_words": len(act),
        "matched_words": matched,
        "missing_count": len(missing),
        "missing_examples": missing[:25],
    }


def cut_layout(spans: List[tuple]) -> List[tuple]:
    """(cut_start, cut_end, src_start, src_end) seconds for each span, so a moment
    in the concatenated cut can be mapped back to where it came from in source."""
    layout: List[tuple] = []
    cursor = 0.0
    for src_start, src_end in spans:
        length = src_end - src_start
        layout.append((cursor, cursor + length, src_start, src_end))
        cursor += length
    return layout


def cut_time_to_source_frames(layout: List[tuple], fps: float,
                              t0: float, t1: float) -> List[List[int]]:
    """Map a [t0,t1] interval of the cut back to source-frame regions.

    An interval can straddle a join, so it may map to more than one source
    region — one per span it overlaps. Frames, ready for `trim_source_regions`.
    """
    regions: List[List[int]] = []
    for cut_start, cut_end, src_start, _src_end in layout:
        lo = max(t0, cut_start)
        hi = min(t1, cut_end)
        if hi > lo:
            a = int(round((src_start + (lo - cut_start)) * fps))
            b = int(round((src_start + (hi - cut_start)) * fps))
            if b > a:
                regions.append([a, b])
    return regions


# A leaked run shorter than this is not worth a re-cut: it is a word-boundary
# wobble or a transcription artefact, not an audible fumble, and trimming it
# would risk clipping real speech. A genuine leaked retake is far longer.
_MIN_LEAK_SECONDS = 0.5


def find_leaked_regions(timeline: Timeline, actual: List[Dict[str, Any]]) -> List[List[int]]:
    """Source-frame regions the cut plays that the plan does not contain.

    `actual` is the re-transcription of the cut, each word carrying a cut-time
    `start`/`end`. Runs of actual words that do not line up with any planned word
    — and that last longer than `_MIN_LEAK_SECONDS` — are the leaked fumbles;
    each is mapped back to the source frames it came from so it can be trimmed.
    """
    expected = expected_tokens(timeline)
    act_tokens = [normalise(spoken(w)) for w in actual]
    matcher = difflib.SequenceMatcher(a=expected, b=act_tokens, autojunk=False)
    fps = timeline.fps_num / max(1, timeline.fps_den)
    layout = cut_layout(_cut_spans(timeline))
    regions: List[List[int]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("insert", "replace") or j2 <= j1:
            continue
        run = [w for w in actual[j1:j2] if spoken(w)]
        if not run:
            continue
        t0 = float(run[0].get("start", 0.0))
        t1 = float(run[-1].get("end", t0))
        if t1 - t0 < _MIN_LEAK_SECONDS:
            continue
        regions.extend(cut_time_to_source_frames(layout, fps, t0, t1))
    return regions


async def verify_cut(timeline: Timeline, source_path: str,
                     language: Optional[str] = None,
                     keep_audio: bool = False) -> Dict[str, Any]:
    """Re-transcribe the actual cut and diff it against the planned transcript.

    Returns the diff report plus the two transcripts. `language` pins Whisper so
    a Hinglish edit is not re-transcribed as an English paraphrase (which would
    make every cut look broken); pass the project's stored language.
    """
    from config import TEMP_DIR
    from asr.faster_whisper_engine import whisper_engine

    out_wav = str(Path(TEMP_DIR) / f"cutcheck_{Path(source_path).stem}.wav")
    built = build_cut_audio(timeline, source_path, out_wav)
    if not built:
        return {"error": "no cut spans to verify"}

    words, _ = await whisper_engine.transcribe_words_async(built, language=language)
    # Compare on the spoken script, on both sides — see `_planned`. The two
    # transcripts kept in the report stay ROMANIZED, because those are for a
    # person to read and the UI shows romanized text everywhere else.
    actual = [normalise(spoken(w)) for w in words]
    actual = [t for t in actual if t]
    expected = expected_tokens(timeline)

    report = diff_transcripts(expected, actual)
    # The deterministic half: leaks are judged by frame overlap, which no
    # hallucination can fake. Together the two halves fully verify the cut —
    # coverage catches dropped speech, the overlap check catches leaked speech.
    report.update(struck_audio_leak(timeline))
    report["expected_text"] = " ".join(w.text for w in timeline.words if w.enabled)
    report["actual_text"] = " ".join(romanized(w) for w in words)
    # The verdict rides on the DETERMINISTIC frame-overlap signal, never on
    # `coverage`: re-transcription coverage is noisy (Whisper re-segments the cut
    # and romanises words a little differently), so a low number is usually
    # spelling variance, not a real drop. Coverage stays in the report as an
    # advisory cross-check only.
    report["verdict"] = (
        "leaks_removed_speech" if report["leaked_struck_words"]
        else "drops_kept_speech" if report["dropped_kept_count"] > 0
        else "clean")
    if not keep_audio:
        try:
            Path(built).unlink(missing_ok=True)
        except Exception:
            pass
    logger.info("Cut verify: coverage=%.3f leaked=%.2fs dropped=%d verdict=%s",
                report["coverage"], report["leaked_struck_seconds"],
                report["dropped_kept_count"], report["verdict"])
    return report


def repair_leaks(timeline: Timeline, primary_source_id: str,
                 max_passes: int = 3) -> Dict[str, Any]:
    """Trim the deterministic leaks out of the cut — no transcription needed.

    The safety net, driven by frame overlap: any struck word whose audio a kept
    span still plays (past the breath tolerance) is cut out of the primary tracks,
    and we re-measure. Repeats until clean or `max_passes`. Purely subtractive —
    it only ever removes audio a word marked struck — so it cannot introduce a
    fumble or drop kept speech. Cheap and reliable, so it runs on every edit;
    accurate word timings (forced alignment) are what make the leaks it sees real.
    """
    from timeline.ops import trim_source_regions

    repaired = 0
    leak = struck_audio_leak(timeline)
    for _pass in range(max_passes):
        regions = leak.get("leaked_regions") or []
        if not regions:
            break
        applied = trim_source_regions(timeline, regions, primary_source_id)
        if not applied:
            break
        repaired += applied
        leak = struck_audio_leak(timeline)
    if repaired:
        logger.info("Cut repair: trimmed %d leaked region(s) of removed speech.", repaired)
    leak["repaired_regions"] = repaired
    return leak


async def verify_and_repair(timeline: Timeline, source_path: str,
                            primary_source_id: str,
                            language: Optional[str] = None) -> Dict[str, Any]:
    """Repair the cut's deterministic leaks, then deep-verify by re-transcription.

    First trims any removed speech the cut still plays (`repair_leaks`, cheap and
    safe), then re-transcribes the result and diffs it against the plan so the
    report carries the full picture — verdict, coverage and repaired count.
    """
    repair = repair_leaks(timeline, primary_source_id)
    report = await verify_cut(timeline, source_path, language=language)
    report["repaired_regions"] = repair["repaired_regions"]
    return report


# The listening check's nudge (AUTO_CUT_EDITOR_PLAN.md §4.6): a removed word the cut still
# plays by at most this much past the breath tolerance is a cut point a few frames off, and is
# trimmed; anything larger is not "fixed" blindly (automatic trimming of big leaks was switched
# off for good reason) but goes on the review list for a person to hear.
NUDGE_SECONDS = 0.15
_REVIEW_LISTEN_MAX = 12


def nudge_cut_edges(timeline: Timeline, primary_source_id: str) -> Dict[str, Any]:
    """Nudge small leaks out of the cut, list the rest. Returns counts and review items.

    Two faults, both by frame overlap (no transcription):
    - a removed word still audible: trimmed when the leak is within NUDGE_SECONDS of the
      tolerance (then re-measured), else listed as "leak";
    - a kept word whose speech the cut clips: listed as "clipped" (widening a cut is not
      something the timeline does safely on its own).
    """
    from timeline.ops import trim_source_regions

    fps = timeline.fps_num / max(1, timeline.fps_den)
    tolerance = _LEAK_TOLERANCE_SECONDS * fps
    nudge = NUDGE_SECONDS * fps
    regions = sorted((int(a), int(b)) for a, b in (timeline.speech_regions or []) if b > a)

    def spans():
        return [(int(round(s * fps)), int(round(e * fps))) for s, e in _cut_spans(timeline)]

    def covered(a: int, b: int, kept) -> List[List[int]]:
        return [[max(a, s), min(b, e)] for s, e in kept if min(b, e) > max(a, s)]

    def speech_of(a: int, b: int) -> List[Tuple[int, int]]:
        if not regions:
            return [(a, b)]
        return [(max(a, s), min(b, e)) for s, e in regions if min(b, e) > max(a, s)]

    kept = spans()
    small, big = [], []
    for w in timeline.words:
        if w.enabled or w.end_frame <= w.start_frame:
            continue
        over = covered(w.start_frame, w.end_frame, kept)
        amount = sum(b - a for a, b in over)
        if amount <= tolerance:
            continue
        (small if amount <= tolerance + nudge else big).append((w, over))
    nudged = trim_source_regions(timeline, [r for _w, over in small for r in over],
                                 primary_source_id) if small else 0

    review: List[Dict[str, Any]] = []
    for w, _over in big:
        review.append({"take": w.take, "start": round(w.start_frame / fps, 2),
                       "end": round(w.end_frame / fps, 2), "text": w.text, "kind": "leak",
                       "decision": "listen", "why": "a removed word is still audible at this cut",
                       "answer": None})
    kept = spans()
    clipped = 0
    for w in timeline.words:
        if not w.enabled or w.end_frame <= w.start_frame:
            continue
        speech = speech_of(w.start_frame, w.end_frame)
        total = sum(b - a for a, b in speech)
        heard = sum(b - a for s, e in speech for a, b in covered(s, e, kept))
        missing = total - heard
        if total and heard and missing > tolerance / 2:
            clipped += 1
            review.append({"take": w.take, "start": round(w.start_frame / fps, 2),
                           "end": round(w.end_frame / fps, 2), "text": w.text, "kind": "clipped",
                           "decision": "listen", "why": f"{missing / fps:.2f} s of a kept word is cut off",
                           "answer": None})
    leak_left = struck_audio_leak(timeline)
    return {"nudged": nudged, "leaks_listed": len(big), "clipped_listed": clipped,
            "leaked_struck_seconds": leak_left["leaked_struck_seconds"],
            "review": review[:_REVIEW_LISTEN_MAX]}
