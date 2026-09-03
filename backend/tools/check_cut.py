"""Validate the cut of a project against its actual audio.

Run this after installing the forced-alignment deps to confirm the fix landed:

    # 1. install a CUDA torch + torchaudio into the backend venv (torchaudio
    #    ships the MMS forced aligner; no extra package to build)
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

    # 2. re-transcribe the project so its timeline gets aligned timestamps
    #    (do this in the app: re-run transcription / auto-edit on the project)

    # 3. check the cut against the real audio
    python tools/check_cut.py <project_id>

`verdict: clean` means the video cut now plays exactly the planned transcript.
`leaks_removed_speech` means fumbles the plan removed are still in the video —
the timestamps are still off (did the project get re-transcribed after install?).
"""

import asyncio
import json
import sys
from pathlib import Path

# The cut transcript can carry a stray native-script char the romanizer left in;
# the Windows console is cp1252 and would crash printing it. Never let display
# encoding sink the check.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr.cut_verify import verify_cut          # noqa: E402
from asr.forced_align import align_available    # noqa: E402
from config import PROJECTS_DIR                 # noqa: E402
from timeline.schema import Timeline            # noqa: E402


async def main(project_id: str) -> int:
    print(f"forced alignment installed: {align_available()}")
    path = Path(PROJECTS_DIR) / f"{project_id}.json"
    if not path.exists():
        path = Path(PROJECTS_DIR) / project_id / "project.json"
    if not path.exists():
        print(f"no project {project_id!r} under {PROJECTS_DIR}")
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    timeline = Timeline.model_validate(data["timeline"])
    source = data.get("source_video")
    language = (data.get("settings") or {}).get("language")

    report = await verify_cut(timeline, source, language=language)
    print(f"\nVERDICT: {report.get('verdict')}")
    print(f"  coverage (planned words the cut says): {report.get('coverage')}")
    print(f"  leaked struck audio: {report.get('leaked_struck_seconds')}s")
    print(f"  dropped kept words:  {report.get('dropped_kept_count')}")
    if report.get("missing_examples"):
        print(f"  missing (dropped) examples: {report['missing_examples'][:15]}")
    print(f"\nPLAN : {report.get('expected_text', '')[:400]}")
    print(f"\nCUT  : {report.get('actual_text', '')[:400]}")
    return 0 if report.get("verdict") == "clean" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tools/check_cut.py <project_id>")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
