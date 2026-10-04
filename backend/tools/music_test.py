"""Try the music generator on its own -- no BuzzEdit app, no Studio, no video.

    python tools/music_test.py --list
    python tools/music_test.py horror_dark
    python tools/music_test.py sad_violin --seconds 45 --variations 3 --tags "sarangi, rain"

Starts ComfyUI (the same Desktop install BuzzEdit's Electron uses) if it is not
already running, generates into the music library (`data/music/<genre>/`, with
a licence/provenance entry per track), offloads every model afterwards per the
GPU policy, stops the ComfyUI it started, and opens the folder.
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from config import COMFYUI_URL  # noqa: E402
from presentation import music_gen  # noqa: E402
from presentation.sound import MUSIC_DIR  # noqa: E402

COMFY_CODE = os.environ.get("COMFY_CODE", r"C:\Users\singh\ComfyUI-Installs\ComfyUI\ComfyUI")
COMFY_DATA = os.environ.get("COMFY_DATA", r"C:\Users\singh\Documents\ComfyUI")
COMFY_PYTHON = os.environ.get("COMFY_PYTHON", str(Path(COMFY_DATA) / ".venv" / "Scripts" / "python.exe"))


def comfy_up() -> bool:
    try:
        return requests.get(f"{COMFYUI_URL}/system_stats", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def start_comfy():
    port = COMFYUI_URL.rsplit(":", 1)[-1].strip("/")
    print(f"Starting ComfyUI on port {port} ...")
    proc = subprocess.Popen(
        [COMFY_PYTHON, "main.py", "--base-directory", COMFY_DATA, "--listen", "127.0.0.1", "--port", port],
        cwd=COMFY_CODE, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(120):
        if comfy_up():
            print("ComfyUI is up.")
            return proc
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited with code {proc.returncode} while starting.")
        time.sleep(1)
    proc.terminate()
    raise SystemExit("ComfyUI did not come up within 2 minutes.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("style", nargs="?", help="style name (see --list)")
    ap.add_argument("--list", action="store_true", help="list the styles and exit")
    ap.add_argument("--seconds", type=float, default=30.0, help="track length (default 30, max 240)")
    ap.add_argument("--variations", type=int, default=1, help="how many variations (default 1, max 6)")
    ap.add_argument("--tags", default="", help="extra ACE-Step tags appended to the style's own")
    ap.add_argument("--seed", type=int, default=None, help="fix the seed for a reproducible run")
    ap.add_argument("--no-open", action="store_true", help="don't open the folder afterwards")
    args = ap.parse_args()

    if args.list or not args.style:
        for name, spec in music_gen.MUSIC_STYLES.items():
            print(f"  {name:20s} {spec['label']:30s} moods: {', '.join(spec['moods'])}")
        return
    if args.style not in music_gen.MUSIC_STYLES:
        raise SystemExit(f"Unknown style {args.style!r}; run with --list.")

    started = None if comfy_up() else start_comfy()
    try:
        t0 = time.time()
        result = asyncio.run(music_gen.generate_variations(
            args.style, count=args.variations, seconds=args.seconds,
            extra_tags=args.tags, seed=args.seed))
        print(f"\nDone in {time.time() - t0:.0f}s (GPU hand-over ready: {result['handover'].get('ready')}).")
        for rel in result["made"]:
            print("  made:", MUSIC_DIR / rel)
        for err in result["errors"]:
            print("  error:", err)
        if result["made"] and not args.no_open:
            folder = (MUSIC_DIR / result["made"][0]).parent
            if sys.platform == "win32":
                os.startfile(folder)  # noqa: S606 - open the folder for a listen
    finally:
        if started is not None:
            print("Stopping the ComfyUI this test started.")
            started.terminate()


if __name__ == "__main__":
    main()
