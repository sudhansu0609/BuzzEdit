"""Score the auto-edit's cuts against a hand-marked ground truth.

    ./.venv/Scripts/python.exe backend/tools/autocut_eval.py <project-id-or-video> <ground-truth-name> [options]

`<ground-truth-name>` resolves to `data/eval/<name>.json` (or pass a full
path); see `tools/autocut_mark.py` to make one from a project's current,
user-corrected cut list.

Two modes:

- **Live** (default): re-transcribes and re-plans the auto-edit `--runs`
  times (default 3) against the source video, because Whisper transcribes the
  same recording differently on every run (AUTO_EDIT_IMPROVEMENT_PLAN.md
  §2.0, §6) — one run is not a measurement.
- **`--no-rerun`**: scores whatever plan is already stored on the named
  project (its `timeline.words`), once. No transcription, no model.

Either way, scoring never touches the network or a model: it only compares
two lists of `{start, end, kind}` spans in source seconds. That half of this
module (everything above `# --- pipeline ---`) is pure and is what
`tests/test_autocut_eval.py` exercises with synthetic spans.

Per AUTO_EDIT_IMPROVEMENT_PLAN.md §2's `reason` vocabulary, cuts are grouped
into the ground truth's four kinds: "fumble" (fillers, stutters, disfluent or
ungrammatical text), "false_start", "retake", and "silence" (a pause the
timeline rebuild closed up, with no disabled word to blame — a heuristic; see
`cut_spans_from_words`).
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Span = Dict[str, Any]

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_RUNS = 3
# How wide a gap between two consecutive ENABLED words (nothing planned
# between them at all) has to be before it counts as a "silence" cut, rather
# than ordinary inter-word breathing room.
DEFAULT_SILENCE_GAP_FLOOR_S = 0.3

# asr/*.py's `reason` vocabulary (see AUTO_EDIT_IMPROVEMENT_PLAN.md §2),
# folded into the ground truth's four kinds. Anything unrecognised falls back
# to "fumble", the catch-all disfluency bucket.
REASON_TO_KIND = {
    "retake": "retake",
    "false_start": "false_start",
    "filler_sound": "fumble",
    "llm_filler": "fumble",
    "stutter": "fumble",
    "not_fluent": "fumble",
    "not_grammatical": "fumble",
}


# --- scoring math (pure; synthetic-testable) --------------------------------

def iou(a: Span, b: Span) -> float:
    """Temporal intersection-over-union of two spans."""
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter <= 0:
        return 0.0
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def match_spans_iou(
    predicted: List[Span], ground_truth: List[Span], iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """Greedy one-to-one matching, highest IoU first. Returns
    (matches, unmatched predicted indices, unmatched ground-truth indices)."""
    pairs = []
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(ground_truth):
            score = iou(p, g)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_p, used_g = set(), set()
    matches = []
    for score, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append({"predicted_index": pi, "ground_truth_index": gi, "iou": score})
    unmatched_pred = [i for i in range(len(predicted)) if i not in used_p]
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in used_g]
    return matches, unmatched_pred, unmatched_gt


def precision_recall_by_span(
    predicted: List[Span], ground_truth: List[Span], iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, Any]:
    """Cut precision/recall counting whole spans: a predicted cut counts as a
    true positive only if some ground-truth span matches it at IoU >= threshold."""
    matches, unmatched_pred, unmatched_gt = match_spans_iou(predicted, ground_truth, iou_threshold)
    tp, fp, fn = len(matches), len(unmatched_pred), len(unmatched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"matched": tp, "predicted_total": len(predicted), "ground_truth_total": len(ground_truth),
           "precision": precision, "recall": recall, "f1": f1}


def _merge_intervals(spans: List[Span]) -> List[Tuple[float, float]]:
    ivs = sorted((float(s["start"]), float(s["end"])) for s in spans if s["end"] > s["start"])
    merged: List[Tuple[float, float]] = []
    for start, end in ivs:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _total_duration(merged: List[Tuple[float, float]]) -> float:
    return sum(end - start for start, end in merged)


def _overlap_duration(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def seconds_overlap_stats(predicted: List[Span], ground_truth: List[Span]) -> Dict[str, float]:
    """Duration-based precision/recall, plus the two seconds counts the brief
    asks for directly: content wrongly removed (predicted seconds outside any
    ground-truth span) and missed debris (ground-truth seconds no predicted
    cut touches)."""
    merged_pred = _merge_intervals(predicted)
    merged_gt = _merge_intervals(ground_truth)
    pred_total = _total_duration(merged_pred)
    gt_total = _total_duration(merged_gt)
    overlap = _overlap_duration(merged_pred, merged_gt)
    return {
        "predicted_total_s": round(pred_total, 3),
        "ground_truth_total_s": round(gt_total, 3),
        "overlap_s": round(overlap, 3),
        "precision_s": round(overlap / pred_total, 4) if pred_total else 1.0,
        "recall_s": round(overlap / gt_total, 4) if gt_total else 1.0,
        "content_wrongly_removed_s": round(pred_total - overlap, 3),
        "missed_debris_s": round(gt_total - overlap, 3),
    }


def per_kind_recall(
    predicted: List[Span], ground_truth: List[Span], iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, Dict[str, Any]]:
    """Recall of each ground-truth kind ("fumble" | "false_start" | "silence"
    | "retake"), matched against every predicted span regardless of its own
    (best-effort) kind — the ground truth's kind is the one that matters."""
    kinds = sorted({g.get("kind", "unknown") for g in ground_truth})
    result: Dict[str, Dict[str, Any]] = {}
    for kind in kinds:
        subset = [g for g in ground_truth if g.get("kind", "unknown") == kind]
        matches, _unmatched_pred, _unmatched_gt = match_spans_iou(predicted, subset, iou_threshold)
        total = len(subset)
        matched = len(matches)
        result[kind] = {"matched": matched, "total": total, "recall": (matched / total) if total else 1.0}
    return result


def score_run(
    predicted: List[Span], ground_truth: List[Span], iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, Any]:
    """Every metric the brief asks for, for one run."""
    return {
        "span": precision_recall_by_span(predicted, ground_truth, iou_threshold),
        "seconds": seconds_overlap_stats(predicted, ground_truth),
        "per_kind": per_kind_recall(predicted, ground_truth, iou_threshold),
    }


def aggregate_runs(run_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean (+ population stdev) of each metric across N runs — Whisper's
    per-run variance is exactly what `--runs` exists to average out."""
    if not run_scores:
        return {"runs": 0}
    agg: Dict[str, Any] = {"runs": len(run_scores)}
    for section in ("span", "seconds"):
        agg[section] = {}
        for key in run_scores[0][section].keys():
            vals = [float(r[section][key]) for r in run_scores]
            mean = sum(vals) / len(vals)
            stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            agg[section][key] = {"mean": round(mean, 4), "stdev": round(stdev, 4)}
    kinds = sorted({k for r in run_scores for k in r["per_kind"].keys()})
    agg["per_kind"] = {}
    for kind in kinds:
        recalls = [r["per_kind"].get(kind, {}).get("recall", 0.0) for r in run_scores]
        agg["per_kind"][kind] = {"mean_recall": round(sum(recalls) / len(recalls), 4)}
    return agg


# --- deriving predicted spans from a plan's word list -----------------------

def _word_dict(w: Any) -> Dict[str, Any]:
    if isinstance(w, dict):
        return w
    if hasattr(w, "model_dump"):
        return w.model_dump()
    return dict(w)


def cut_spans_from_words(
    words: List[Any], fps_num: int = 30, fps_den: int = 1,
    silence_gap_floor_s: float = DEFAULT_SILENCE_GAP_FLOOR_S,
) -> List[Span]:
    """Predicted cut spans in source seconds, from a timeline's word list
    (`timeline.schema.WordItem`s, or the equivalent dicts a stored project's
    JSON carries at `timeline.words`).

    Two kinds of cut:
    - a contiguous run of disabled words -> one span per run, `kind` mapped
      from the run's most common `reason` via REASON_TO_KIND;
    - a gap between two consecutive ENABLED words (nothing planned between
      them at all) wider than `silence_gap_floor_s` -> a "silence" span for
      the pause the timeline rebuild closed up.

    The silence detection is a heuristic — it does not know the exact
    `pause_padding_seconds` a given rebuild used — so treat "silence" spans as
    approximate. The word-driven spans (fumble/false_start/retake), and every
    seconds/span total that does not key on kind, do not depend on it.
    """
    from timeline.schema import frame_to_time

    rows = sorted((_word_dict(w) for w in words), key=lambda w: w.get("start_frame", 0))
    spans: List[Span] = []
    run: List[Dict[str, Any]] = []

    def _flush_run() -> None:
        if not run:
            return
        start = frame_to_time(run[0]["start_frame"], fps_num, fps_den)
        end = frame_to_time(run[-1]["end_frame"], fps_num, fps_den)
        reasons = [w.get("reason") for w in run if w.get("reason")]
        kind = "fumble"
        if reasons:
            common = max(set(reasons), key=reasons.count)
            kind = REASON_TO_KIND.get(common, "fumble")
        spans.append({"start": start, "end": end, "kind": kind})

    prev_enabled_end: Optional[float] = None
    for w in rows:
        if not w.get("enabled", True):
            run.append(w)
            prev_enabled_end = None  # a disabled word already accounts for this gap
            continue
        _flush_run()
        run = []
        start_s = frame_to_time(w.get("start_frame", 0), fps_num, fps_den)
        if prev_enabled_end is not None and start_s - prev_enabled_end > silence_gap_floor_s:
            spans.append({"start": prev_enabled_end, "end": start_s, "kind": "silence"})
        prev_enabled_end = frame_to_time(w.get("end_frame", 0), fps_num, fps_den)
    _flush_run()

    spans.sort(key=lambda s: s["start"])
    return spans


# --- ground truth -------------------------------------------------------

def _eval_dir() -> Path:
    from config import DATA_DIR
    return DATA_DIR / "eval"


def _results_dir() -> Path:
    return _eval_dir() / "results"


def _resolve_ground_truth_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    if p.suffix != ".json":
        p = Path(f"{name_or_path}.json")
    candidate = _eval_dir() / p.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"no ground truth file: {name_or_path} (looked for {candidate})")


def load_ground_truth_file(path: Path) -> Dict[str, Any]:
    """{"source": ..., "spans": [{"start", "end", "kind"}, ...]} — a bare list
    is accepted too, for a hand-written file with no `source`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"spans": data}
    data["spans"] = [
        {"start": float(s["start"]), "end": float(s["end"]), "kind": s.get("kind", "unknown")}
        for s in data.get("spans", [])
    ]
    return data


# --- pipeline: running the real auto-edit -----------------------------------

def _project_data(project_id: str) -> Dict[str, Any]:
    from config import PROJECTS_DIR
    from store.project_store import ProjectStore
    store = ProjectStore(base_dir=str(PROJECTS_DIR))
    data = store.get_project(project_id)
    if not data:
        raise SystemExit(f"no such project: {project_id}")
    return data


def _resolve_source_video(source: str) -> str:
    if Path(source).exists():
        return str(source)
    video = _project_data(source).get("source_video")
    if not video:
        raise SystemExit(f"project {source} has no source_video")
    return video


def _load_stored_plan(project_id: str) -> Dict[str, Any]:
    """The words and fps of whatever plan is already saved on a project —
    for `--no-rerun`. No transcription, no model."""
    data = _project_data(project_id)
    tl = data.get("timeline") or {}
    return {"words": tl.get("words") or [], "fps_num": tl.get("fps_num", 30),
           "fps_den": tl.get("fps_den", 1), "elapsed_s": 0.0}


async def _run_live_plan(source_video: str, language: Optional[str] = None,
                         fumble_aggressiveness: float = 0.5) -> Dict[str, Any]:
    """One fresh transcribe + auto-edit + timeline build, mirroring
    tools/e2e_auto_edit.py through the timeline rebuild — but in memory only;
    scoring a run must not overwrite anyone's saved project."""
    from asr import whisper_engine
    from asr.auto_edit import plan_auto_edit
    from timeline import build_timeline_from_transcript
    from utils.ffmpeg_utils import extract_audio, get_video_info

    t0 = time.time()
    audio_path = await asyncio.to_thread(extract_audio, source_video)
    v_info = await asyncio.to_thread(get_video_info, source_video)
    full = await whisper_engine.transcribe_full_async(audio_path, language=language)
    words = full["words"]
    plan = await plan_auto_edit(words, audio_path, aggressiveness=fumble_aggressiveness)
    tl = build_timeline_from_transcript(
        source_path=source_video,
        duration_seconds=v_info.get("duration", 0.0),
        transcript_words=plan.words,
        fps_num=v_info.get("fps_num", 30),
        fps_den=v_info.get("fps_den", 1),
        width=v_info.get("width", 1920),
        height=v_info.get("height", 1080),
        has_audio=v_info.get("audio_codec") not in (None, "none"),
        speech_regions=plan.report.get("speech"),
        energy_envelope=plan.report.get("energy"),
    )
    return {"words": [w.model_dump() for w in tl.words], "fps_num": tl.fps_num, "fps_den": tl.fps_den,
           "elapsed_s": round(time.time() - t0, 1)}


async def run_and_score(
    source: str, ground_truth: str, runs: int = DEFAULT_RUNS, no_rerun: bool = False,
    language: Optional[str] = None, iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, Any]:
    gt_path = _resolve_ground_truth_path(ground_truth)
    gt_data = load_ground_truth_file(gt_path)
    ground_truth_spans = gt_data["spans"]

    run_results = []
    plans: List[Dict[str, Any]] = []
    if no_rerun:
        plans.append(_load_stored_plan(source))
    else:
        video_path = _resolve_source_video(source)
        for _ in range(max(1, runs)):
            plans.append(await _run_live_plan(video_path, language=language))

    for plan in plans:
        predicted = cut_spans_from_words(plan["words"], plan["fps_num"], plan["fps_den"])
        run_results.append(score_run(predicted, ground_truth_spans, iou_threshold))

    return {
        "source": source, "ground_truth_file": str(gt_path), "no_rerun": no_rerun,
        "iou_threshold": iou_threshold, "runs": run_results,
        "aggregate": aggregate_runs(run_results),
    }


# --- reporting ---------------------------------------------------------

def format_table(result: Dict[str, Any]) -> str:
    agg = result.get("aggregate", {})
    lines = [
        f"source={result.get('source')}  ground_truth={result.get('ground_truth_file')}  "
        f"runs={agg.get('runs', 0)}  iou>={result.get('iou_threshold')}",
        "",
        f"{'metric':<30}{'mean':>10}{'stdev':>10}",
    ]
    rows = [
        ("span precision", "span", "precision"),
        ("span recall", "span", "recall"),
        ("span f1", "span", "f1"),
        ("seconds precision", "seconds", "precision_s"),
        ("seconds recall", "seconds", "recall_s"),
        ("content wrongly removed (s)", "seconds", "content_wrongly_removed_s"),
        ("missed debris (s)", "seconds", "missed_debris_s"),
    ]
    for label, section, key in rows:
        stat = agg.get(section, {}).get(key, {})
        lines.append(f"{label:<30}{stat.get('mean', 0):>10.3f}{stat.get('stdev', 0):>10.3f}")
    lines.append("")
    lines.append("per-kind recall (mean):")
    for kind, stat in sorted(agg.get("per_kind", {}).items()):
        lines.append(f"  {kind:<24}{stat.get('mean_recall', 0):>8.3f}")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="project id, or a recording path")
    p.add_argument("ground_truth", help="data/eval/<name>.json — a name, or a full path")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                   help=f"re-transcribe and re-plan this many times (default {DEFAULT_RUNS})")
    p.add_argument("--no-rerun", action="store_true",
                   help="score `source`'s already-saved plan once, instead of re-running the auto-edit")
    p.add_argument("--language", default=None)
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    return p


async def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = await run_and_score(
        args.source, args.ground_truth, runs=args.runs, no_rerun=args.no_rerun,
        language=args.language, iou_threshold=args.iou_threshold)
    print(format_table(result))

    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{Path(args.ground_truth).stem}_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
