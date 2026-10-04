"""Export a project's current cut list as ground truth for tools/autocut_eval.py.

Run this *after* the user has gone through BuzzEdit's Cuts view and fixed
whatever the auto-edit got wrong — ground truth is then defined as the user's
own corrected edit, not a hand-typed guess.

    ./.venv/Scripts/python.exe backend/tools/autocut_mark.py <project_id> <name>

Writes `data/eval/<name>.json`: `{"source": "<project_id>", "spans": [...]}`,
using the same word-run/silence-gap derivation `autocut_eval.py` uses to score
a predicted plan — see `cut_spans_from_words` there.
"""

import argparse
import json
import sys
from pathlib import Path

# Both directories: `tools/` itself (so `import autocut_eval` resolves the
# same way whether this runs as a script or gets imported, e.g. by a test)
# and `backend/` (so autocut_eval's own lazy imports of config/store/timeline
# resolve).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autocut_eval  # noqa: E402


def export_ground_truth(project_id: str, name: str) -> Path:
    data = autocut_eval._project_data(project_id)
    tl = data.get("timeline") or {}
    words = tl.get("words") or []
    spans = autocut_eval.cut_spans_from_words(words, tl.get("fps_num", 30), tl.get("fps_den", 1))
    out = {"source": project_id, "spans": spans}
    out_dir = autocut_eval._eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("project_id")
    p.add_argument("name", help="ground-truth file stem; written to data/eval/<name>.json")
    args = p.parse_args(argv)
    out_path = export_ground_truth(args.project_id, args.name)
    print(f"wrote {out_path} ({len(json.loads(out_path.read_text(encoding='utf-8'))['spans'])} spans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
