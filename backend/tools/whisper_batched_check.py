"""Compare batched Whisper decoding with the sequential pass on a real take,
and turn it on when it says the same words.

    .venv/Scripts/python.exe backend/tools/whisper_batched_check.py [--audio <wav>] [--minutes 10] [--adopt]

Runs `FasterWhisperEngine._do_transcribe` twice on the first `--minutes` of
the take (default: the cached raat3baje_ep1 audio), once sequential and once
through `BatchedInferencePipeline` with the engine's own options, and
compares the word sequences (difflib ratio over the romanised words) and the
start times of the matched words. Batched is adopted (`asr_batched` app
setting) when the words agree at >= WORDS_MIN, the p95 start-time difference
is under TIME_P95_MAX and it is at least SPEEDUP_MIN faster. Exit 0 when
adopted, 1 when not, 2 when it could not run.

Why (2026-10-21): Whisper took 24 of the minutes in a production whose whole
budget is 3.5 hours.
"""
import argparse
import difflib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

WORDS_MIN = 0.96
TIME_P95_MAX = 0.5
SPEEDUP_MIN = 1.4
_PUNCT = re.compile(r"[^\w]+", re.UNICODE)


def tokens(words):
    out = []
    for w in words:
        t = _PUNCT.sub("", str(w.get("hinglish") or w.get("word") or "").lower())
        if t:
            out.append(t)
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", default=str(BACKEND.parent / "data" / "eval" / "cache" / "raat3baje_ep1_16k.wav"))
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--language", default="hi")
    p.add_argument("--adopt", action="store_true")
    args = p.parse_args()

    source = Path(args.audio)
    if not source.exists():
        print(f"FAIL no audio at {source}")
        return 2
    work = Path(tempfile.mkdtemp(prefix="whisper_ab_"))
    audio = work / "take.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-t", str(args.minutes * 60),
                    "-ac", "1", "-ar", "16000", str(audio)], check=True, timeout=600,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    from asr import faster_whisper_engine as engine
    eng = engine.FasterWhisperEngine()
    model = eng.load_model(device="cuda", compute_type="float16")

    runs = {}
    for mode in ("sequential", "batched"):
        engine._batched_enabled = (lambda: mode == "batched")  # noqa: E731 - the setting under test
        started = time.time()
        words, detected = eng._do_transcribe(model, str(audio), args.language)
        runs[mode] = {"seconds": round(time.time() - started, 1), "words": words, "language": detected}
        print(f"{mode}: {len(words)} words in {runs[mode]['seconds']} s", flush=True)

    a, b = runs["sequential"]["words"], runs["batched"]["words"]
    ta, tb = tokens(a), tokens(b)
    matcher = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    ratio = matcher.ratio()
    deltas = []
    # Map token index back to word index (tokens drop empties, so rebuild the map).
    idx_a = [i for i, w in enumerate(a) if _PUNCT.sub("", str(w.get("hinglish") or w.get("word") or "").lower())]
    idx_b = [i for i, w in enumerate(b) if _PUNCT.sub("", str(w.get("hinglish") or w.get("word") or "").lower())]
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            wa, wb = a[idx_a[block.a + k]], b[idx_b[block.b + k]]
            deltas.append(abs(float(wa["start"]) - float(wb["start"])))
    p95 = statistics.quantiles(deltas, n=20)[-1] if len(deltas) >= 20 else (max(deltas) if deltas else 9.9)
    median = statistics.median(deltas) if deltas else 9.9
    speedup = runs["sequential"]["seconds"] / max(0.1, runs["batched"]["seconds"])
    ok = ratio >= WORDS_MIN and p95 <= TIME_P95_MAX and speedup >= SPEEDUP_MIN
    summary = {"minutes": args.minutes, "sequential_s": runs["sequential"]["seconds"],
               "batched_s": runs["batched"]["seconds"], "speedup": round(speedup, 2),
               "words_sequential": len(a), "words_batched": len(b), "word_ratio": round(ratio, 4),
               "start_delta_median_s": round(median, 3), "start_delta_p95_s": round(p95, 3), "adopt": ok}
    out_dir = BACKEND.parent / "data" / "bench" / f"{datetime.now():%Y-%m-%d_%H%M}_whisper_batched"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.adopt:
        from store.app_settings import AppSettings
        AppSettings().set("asr_batched", bool(ok))
        print("ADOPTED batched decoding" if ok else "KEPT sequential decoding")
    try:
        eng.release_gpu()
    except Exception:  # noqa: BLE001
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
