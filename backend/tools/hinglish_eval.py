"""Before/after measure for the Hinglish romanizer.

Loads the real sample transcript (`data/projects/7482df63.json`), re-romanizes
every word from its untouched `word_native` through the *current* romanizer,
and prints the first 150 before -> after pairs plus a changed-word count.
Read-only: never writes the sample project back.

    cd backend && python tools/hinglish_eval.py
"""

import json
import sys
from pathlib import Path

# The output is full of Devanagari and the odd stray glyph the old romanizer
# left in; the Windows console is cp1252 and would crash printing either.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr.transliterate import to_hinglish   # noqa: E402
from config import PROJECTS_DIR             # noqa: E402

SAMPLE_PROJECT_ID = "7482df63"
PREVIEW_COUNT = 150


def _words(data: dict) -> list:
    segments = (data.get("transcript") or {}).get("segments") or []
    out = []
    for seg in segments:
        out.extend(seg.get("words") or [])
    return out


def main() -> int:
    path = Path(PROJECTS_DIR) / f"{SAMPLE_PROJECT_ID}.json"
    if not path.exists():
        print(f"no sample project at {path}")
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    language = (data.get("transcript") or {}).get("language") or "hi"
    words = _words(data)

    rows = []
    changed = 0
    for w in words:
        native = w.get("word_native")
        before = str(w.get("word") or w.get("hinglish") or "").strip()
        if not native:
            rows.append((before, before, False))
            continue
        after = to_hinglish(native, language).strip()
        is_changed = after != before
        if is_changed:
            changed += 1
        rows.append((before, after, is_changed))

    print(f"sample: {path}")
    print(f"language: {language}")
    print(f"total words: {len(words)}")
    print(f"changed words: {changed} ({changed / max(1, len(words)):.1%})")
    print()
    print(f"first {min(PREVIEW_COUNT, len(rows))} words (before -> after, '*' = changed):")
    for before, after, is_changed in rows[:PREVIEW_COUNT]:
        marker = "*" if is_changed else " "
        print(f"{marker} {before:<20} -> {after}")

    changed_rows = [(b, a) for b, a, c in rows if c]
    print()
    print(f"10 example changes:")
    for before, after in changed_rows[:10]:
        print(f"  {before} -> {after}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
