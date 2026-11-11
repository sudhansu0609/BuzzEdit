"""Write the ComfyUI graphs that code composes on top of a workflow file to
`workflows/reference/`, so every graph ComfyUI receives exists as a file.

    .venv/Scripts/python.exe backend/tools/export_reference_graphs.py

Today that is one graph: the main-character face swap
(`presentation.character.with_face_swap`) over the still graph `broll_image`
resolves to. It is built by the same function a production calls, with
placeholder values for the per-job inputs, so the copy cannot drift from what is
sent. The files are references for reading and for loading into ComfyUI by hand;
BuzzEdit never loads them (`workflows.py` only lists `workflows/*.json`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from presentation import workflows  # noqa: E402
from presentation.character import with_face_swap  # noqa: E402

OUT_DIR = workflows.WORKFLOWS_DIR / "reference"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    still = workflows.resolve("broll_image")
    if still is None:
        print("broll_image resolves to no workflow; nothing to export")
        return 1
    graph = still.build("a placeholder prompt", width=1024, height=576, seed=1, prefix="reference")
    swapped = with_face_swap(graph, "main_character_reference.png")
    name = f"{Path(still.file).stem}_faceswap.json"
    (OUT_DIR / name).write_text(json.dumps(swapped, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / name} (from {still.file})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
