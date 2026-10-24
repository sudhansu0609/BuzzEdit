"""Prove the local editor works on THIS machine before a job uses it.

    .venv/Scripts/python.exe backend/tools/editor_smoke.py [--key raat3baje_ep1] [--lines 60]

Loads the local editor exactly as a job would (`EditorClient.local().prepare()`:
a loaded copy with enough context, else eject everything and load at 32k),
reads the first `--lines` utterances of a cached take through `editor.edit`
and checks that a parseable decisions JSON came back. Exit 0 on success, 1 with
the reason otherwise. Ejects the model afterwards.

Why: on 2026-10-21 the editor's first real run found LM Studio's model loaded
at 16k, three parallel reads sharing it and a halving loop, and lost 36 minutes
of a production. The pieces had passed their unit tests; nothing had run the
whole path against the live server. This is that run, in two minutes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", default="raat3baje_ep1", help="an answer key with cached words in data/eval/cache")
    p.add_argument("--lines", type=int, default=60)
    p.add_argument("--keep-loaded", action="store_true", help="leave the model in memory afterwards")
    args = p.parse_args()

    from config import DATA_DIR
    from asr import editor
    from asr import utterances as utt

    cache = DATA_DIR / "eval" / "cache" / f"{args.key}_words.json"
    if not cache.exists():
        print(f"FAIL no cached words for {args.key} ({cache})")
        return 1
    words = json.loads(cache.read_text(encoding="utf-8"))["words"]
    audio = DATA_DIR / "eval" / "cache" / f"{args.key}_16k.wav"
    regions = None
    if audio.exists():
        from asr.gap_recovery import speech_regions
        from faster_whisper.audio import decode_audio
        regions = speech_regions(decode_audio(str(audio), sampling_rate=16000))
    utterances = utt.build(words, regions or [])[: args.lines]
    if len(utterances) < 10:
        print(f"FAIL only {len(utterances)} utterances built from {args.key}")
        return 1

    started = time.time()
    client = editor.EditorClient.local()
    try:
        client.prepare()
    except Exception as e:  # noqa: BLE001 - that is the finding
        print(f"FAIL the local editor model could not be had: {e}")
        return 1
    print(f"model {client.model} on {client.backend}, context {client.context}, "
          f"input budget {client.input_budget()} tokens ({time.time() - started:.0f}s to prepare)")
    if client.context < editor.LOCAL_CONTEXT:
        print(f"FAIL loaded context {client.context} < {editor.LOCAL_CONTEXT}")
        return 1
    try:
        decisions, meta = editor.edit(utterances, words, client)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL the read failed: {e}")
        return 1
    answer = str(meta.get("answer") or "")
    if editor.first_json(answer) is None:
        print(f"FAIL no JSON in the answer: {answer[:200]!r}")
        return 1
    print(f"OK {len(utterances)} utterances read in {meta.get('seconds')}s: "
          f"{decisions.groups} group(s), {decisions.trims} trim(s), {decisions.drops} drop(s), "
          f"{len(decisions.refused)} refused; served model {meta.get('model')}")
    if not args.keep_loaded:
        import subprocess
        from llm.lm_launcher import find_lms_cli
        lms = find_lms_cli()
        if lms:
            subprocess.run([lms, "unload", "--all"], capture_output=True, text=True, timeout=60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
