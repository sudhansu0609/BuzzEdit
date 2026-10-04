"""Per-channel Hinglish spelling overrides.

Two JSON files merge for a given channel `key`:

    data/glossary/_global.json   -- applies to every project
    data/glossary/<key>.json     -- applies only to that channel

Each is a flat `{"spelling": "correction"}` map; the key can be either the
native (Devanagari) spelling or the romanized one -- `transliterate.py`
checks both. The channel file wins over `_global.json` on a shared key, and
the whole merged dict is what `transliterate.to_hinglish` applies last, after
the common-word/loanword tables and the general rules.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from config import DATA_DIR

GLOSSARY_DIR = DATA_DIR / "glossary"
GLOBAL_KEY = "_global"


def _path_for(key: str) -> Path:
    # Keep this to filename-safe characters; `key` comes from a URL path
    # segment / project settings, never trusted as a literal path.
    safe = "".join(c for c in key if c.isalnum() or c in ("-", "_")) or GLOBAL_KEY
    return GLOSSARY_DIR / f"{safe}.json"


def _read(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(path: Path, data: Dict[str, str]) -> None:
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(GLOSSARY_DIR), prefix="glossary_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_channel(key: str) -> Dict[str, str]:
    """The channel file alone, no global merge -- what `GET /api/glossary/{key}`
    returns."""
    return _read(_path_for(key))


def load_merged(key: Optional[str]) -> Dict[str, str]:
    """`_global.json` plus `<key>.json`, the channel winning on overlap. This
    is what gets passed to `transliterate.to_hinglish` as `glossary`."""
    merged = dict(_read(_path_for(GLOBAL_KEY)))
    if key and key != GLOBAL_KEY:
        merged.update(_read(_path_for(key)))
    return merged


def merge_entries(key: str, entries: Dict[str, str]) -> Dict[str, str]:
    """`PUT /api/glossary/{key}`: merge `entries` into the channel file (new
    keys added, existing keys overwritten) and persist. Returns the resulting
    channel file."""
    path = _path_for(key)
    current = _read(path)
    current.update({str(k): str(v) for k, v in entries.items()})
    _write(path, current)
    return current
