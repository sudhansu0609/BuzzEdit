"""Smoke-run the presentation pass on a copy of a real project WITHOUT the
language model or ComfyUI, so every non-generated layer (sound, cards from
pattern entities, moods, grade, transitions, karaoke captions, title) goes
through a real render. Usage: python tools/dress_smoke.py <project_id> [genre]
"""
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR, PROJECTS_DIR          # noqa: E402
from presentation import director                     # noqa: E402
from presentation.models import PresentationSettings  # noqa: E402


async def main(project_id: str, genre: str) -> None:
    source = PROJECTS_DIR / f"{project_id}.json"
    copy_id = f"{project_id}_dress"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["id"] = copy_id
    data["name"] = f"{data.get('name', project_id)} (dress smoke)"
    (PROJECTS_DIR / f"{copy_id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    async def no_llm():
        return None
    director._llm_asker = no_llm
    import render.encoder as enc
    enc.get_nvenc_available = lambda: False

    settings = PresentationSettings(broll=False, broll_video=False, thumbnail=False,
                                    graphics=False, genre=genre, maps=False)
    started = time.time()
    report = await director.run_presentation_pass(copy_id, settings,
                                                  progress_cb=lambda f, m="": print(f"{f:5.2f} {m}"))
    print(json.dumps(report.model_dump(exclude={"settings", "beats_dropped"}), indent=1, default=str)[:4000])
    print("seconds:", round(time.time() - started, 1))
    print("output:", report.output_path)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "horror"))
