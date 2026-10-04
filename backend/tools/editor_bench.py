"""Score the AI editor against the creator's own cuts.

    ./.venv/Scripts/python.exe backend/tools/editor_bench.py raat3baje_ep1 life3baje_ep1 [--runs 2]
        [--model claude-opus-5-5] [--effort medium] [--no-notes] [--no-hints]

Each name is an answer key in `data/eval/<name>.json` (made by `tools/autocut_truth.py`) whose
`source` is the raw recording. For each one this transcribes the raw (cached in
`data/eval/cache/<name>_words.json`; gap recovery included), builds the utterance table
(asr.utterances), runs the editor (asr.editor) and scores its cut against the key:

- **wrongly removed**: speech the creator kept that the editor cut;
- **missed**: speech the creator cut that the editor kept;
- **caught**: share of the creator's removed speech the editor also removed.

Words count for at most 1 s each, so a transcript word stretched over a hole cannot dominate. A
key may list `editorial` spans: whole sections the creator dropped for the story, which no
transcript reveals; they are left out of the score and reported separately.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WORD_CAP_S = 1.0
OPUS_PRICE = (4.0, 20.0)      # $ per million tokens in/out, claude.com/pricing 2026-10-04


def _eval_dir() -> Path:
    from config import DATA_DIR
    return DATA_DIR / "eval"


def _studio_key() -> str:
    """The proxy key BuzzcafStudio uses (its own config), for running this tool by hand."""
    path = Path(__file__).resolve().parents[3] / "BuzzcafStudio" / "backend" / "config" / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("claude_api_key", "")
    except Exception:
        return ""


async def _words(name: str, key: Dict[str, Any], wav: Path) -> List[Dict[str, Any]]:
    cache = _eval_dir() / "cache" / f"{name}_words.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["words"]
    from asr import whisper_engine
    full = await whisper_engine.transcribe_full_async(str(wav), language=key.get("language", "hi"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"words": full["words"], "gap_recovery": full.get("gap_recovery")},
                                ensure_ascii=False), encoding="utf-8")
    return full["words"]


def _wav(name: str, key: Dict[str, Any]) -> Path:
    import soundfile as sf
    from tools.autocut_truth import load_audio, SR
    path = _eval_dir() / "cache" / f"{name}_16k.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), load_audio(key["source"]), SR)
    return path


def score(words: List[Dict[str, Any]], removed_idx: set, removed_spans: List, key: Dict[str, Any]) -> Dict[str, Any]:
    cut = sorted((float(s["start"]), float(s["end"])) for s in key["spans"])
    editorial = [tuple(e) for e in key.get("editorial", [])]

    def inside(t: float, spans) -> bool:
        return any(a <= t < b for a, b in spans)

    items = [(float(w["start"]), float(w["end"]), i in removed_idx) for i, w in enumerate(words)
             if str(w.get("word") or "").strip()]
    for a, b, _reason in removed_spans:                     # untranscribed speech the editor cut
        t = a
        while t < b:
            items.append((t, min(b, t + 0.5), True))
            t += 0.5
    wrong = missed = total = 0.0
    for a, b, removed in items:
        mid, dur = (a + b) / 2, min(WORD_CAP_S, b - a)
        if inside(mid, editorial):
            continue
        truth = inside(mid, cut)
        total += dur if truth else 0.0
        wrong += dur if removed and not truth else 0.0
        missed += dur if truth and not removed else 0.0
    runs, prev = 0, False
    for a, b, removed in sorted(items):
        runs += removed and not prev
        prev = removed
    return {"wrongly_removed_s": round(wrong, 1), "missed_s": round(missed, 1),
            "caught_pct": round(100 * (total - missed) / total, 1) if total else 100.0, "cuts": runs}


async def prepare(name: str, args) -> Dict[str, Any]:
    """Transcript, speech map and utterance table for one answer key."""
    from asr import utterances
    from asr.gap_recovery import speech_regions
    from faster_whisper.audio import decode_audio

    key = json.loads((_eval_dir() / f"{name}.json").read_text(encoding="utf-8"))
    wav = _wav(name, key)
    words = await _words(name, key, wav)
    audio = decode_audio(str(wav), sampling_rate=16000)
    regions = speech_regions(audio)
    script_path = args.script or key.get("script")
    script = Path(script_path).read_text(encoding="utf-8") if script_path and Path(script_path).exists() else None
    utts = utterances.build(words, regions, audio=None if args.no_notes else audio,
                            script_text=utterances.spoken_script(script) if script else None)
    if args.no_hints:
        for u in utts:
            u.similar, u.same_opening, u.script = [], [], None
    return {"name": name, "key": key, "words": words, "utterances": utts}


def examples_for(name: str, prepared: Dict[str, Dict[str, Any]]) -> str:
    """Worked examples from every OTHER key: leave-one-out, so a score measures what carries over."""
    from asr import style
    sources = []
    for other, p in prepared.items():
        if other == name:
            continue
        cut = [(float(s["start"]), float(s["end"])) for s in p["key"]["spans"]]
        decided = style.labels(p["utterances"], cut, [tuple(e) for e in p["key"].get("editorial", [])])
        sources.append((other, p["utterances"], decided))
    from presentation.genre import editing_notes_for
    return style.style_block(sources, editing_notes_for(prepared[name]["key"].get("genre")))


async def bench(p: Dict[str, Any], args, style_text: str = "") -> List[Dict[str, Any]]:
    from asr import double_check, editor

    name, key, words, utts = p["name"], p["key"], p["words"], p["utterances"]
    client = editor.EditorClient(base_url=args.base_url, api_key=args.key or _studio_key(),
                                 model=args.model, effort=args.effort)
    results = []
    for run in range(args.runs):
        started = time.time()
        if args.full:
            decisions, report = await asyncio.to_thread(double_check.check, utts, words, client, style_text,
                                                        args.agree_runs)
            calls = report.calls
            extra = {k: v for k, v in vars(report).items() if k != "calls"}
            answer = None
        else:
            decisions, meta = await asyncio.to_thread(editor.edit, utts, words, client, style_text)
            calls, extra, answer = [meta], {}, meta.get("answer")
        result = score(words, set(decisions.removed_words), decisions.removed_spans, key)
        tin = sum((c.get("usage") or {}).get("prompt_tokens", 0) for c in calls)
        tout = sum((c.get("usage") or {}).get("completion_tokens", 0) for c in calls)
        result.update(recording=name, run=run, mode="full" if args.full else "single",
                      examples=bool(style_text), seconds=round(time.time() - started, 1), calls=len(calls),
                      tokens_in=tin, tokens_out=tout,
                      usd=round((tin * OPUS_PRICE[0] + tout * OPUS_PRICE[1]) / 1e6, 3),
                      groups=decisions.groups, drops=decisions.drops, trims=decisions.trims,
                      refused=len(decisions.refused), utterances=len(utts), **extra)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        out = _eval_dir() / "results" / f"editor_{name}_{time.strftime('%Y%m%dT%H%M%S')}_{run}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"result": result, "answer": answer, "refused": decisions.refused},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keys", nargs="+")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium")
    p.add_argument("--base-url", default="http://127.0.0.1:8787/v1")
    p.add_argument("--key", default="")
    p.add_argument("--script", default="", help="written script to tag utterances with")
    p.add_argument("--no-notes", action="store_true", help="leave the audio notes out (ablation)")
    p.add_argument("--no-hints", action="store_true", help="leave similarity and script hints out (ablation)")
    p.add_argument("--examples", action="store_true",
                   help="show the editor worked examples from the OTHER keys (leave-one-out)")
    p.add_argument("--full", action="store_true", help="run the whole double-check pipeline (asr.double_check)")
    p.add_argument("--agree-runs", type=int, default=3, help="editor passes that must agree, with --full")
    args = p.parse_args(argv)

    async def run_all():
        prepared = {name: await prepare(name, args) for name in args.keys}
        for name, p_ in prepared.items():
            await bench(p_, args, examples_for(name, prepared) if args.examples else "")
    asyncio.run(run_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
