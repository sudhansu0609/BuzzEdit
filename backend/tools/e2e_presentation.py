"""Run the overnight presentation pass on a project, from the command line.

    ./.venv/Scripts/python.exe backend/tools/e2e_presentation.py <project_id>

Assumes the project already has a timeline (run tools/e2e_auto_edit.py first).
Prints the report; the pass itself writes presentation_report.json and the
rendered file under data/output.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main(project_id: str) -> int:
    from config import PROJECTS_DIR
    from presentation import PresentationSettings, run_presentation_pass

    log_path = PROJECTS_DIR / f"{project_id}_e2e" / "presentation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    console = logging.StreamHandler(sys.stdout)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(console)

    def progress(fraction: float, message: str = "") -> None:
        print(f"[{fraction * 100:5.1f}%] {message}")

    report = await run_presentation_pass(project_id, PresentationSettings(),
                                         progress_cb=progress)
    payload = report.model_dump(exclude={"timings"})
    print("\n=== PRESENTATION REPORT ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tools/e2e_presentation.py <project_id>")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
