"""Checkpoints for a transcription that is interrupted part-way.

`POST /api/transcription/transcribe` is one request that runs Whisper (14
minutes on a 42-minute Raat3Baje take), the AI cut planner (6-7 minutes of
Claude reads) and the cut verification, and only then saves the project. When
BuzzEdit died in the last phase -- 2026-10-20, three times in a row, stopped by
the Studio's own watchdog while ffmpeg built the verification audio -- every
retry started from zero and the production spent 77 minutes transcribing the
same recording three times without ever finishing.

So each expensive phase now leaves its result here, next to the project
(`<projects>/<id>_transcribe_ckpt.json`), under a key that says what it was
made from (the source file's path, size and mtime, the language, whether the
take is precut, the vocabulary). The next run with the same key picks up
after the last phase that finished; a run with a different key (a new
recording, another language) starts over. The file is removed once the
transcription is saved on the project.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("transcribe_checkpoint")

#: Phases a transcription checkpoints, in order.
PHASE_WHISPER = "whisper"
PHASE_PLAN = "plan"


def _path(projects_dir: Path, project_id: str) -> Path:
    return Path(projects_dir) / f"{project_id}_transcribe_ckpt.json"


def key_for(source_video: str, language: Optional[str], precut: bool,
            vocabulary: Optional[list] = None) -> Dict[str, Any]:
    """What a checkpoint was made from. Two runs with equal keys would produce
    the same Whisper words, so the words may be reused."""
    try:
        stat = os.stat(source_video)
        size, mtime = int(stat.st_size), int(stat.st_mtime)
    except OSError:
        size, mtime = -1, -1
    return {
        "source": str(source_video),
        "size": size,
        "mtime": mtime,
        "language": language or "",
        "precut": bool(precut),
        "vocabulary": sorted(str(v) for v in (vocabulary or [])),
    }


def _json_default(value: Any) -> Any:
    """numpy scalars/arrays and other odd values the reports may carry."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def load(projects_dir: Path, project_id: str, key: Dict[str, Any]) -> Dict[str, Any]:
    """The phases checkpointed for this project under this key: a dict with any
    of `whisper` / `plan`, empty when there is none or it was made from
    something else. Never raises."""
    path = _path(projects_dir, project_id)
    try:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - a damaged checkpoint is no checkpoint
        logger.warning("Transcription checkpoint for %s is unreadable (%s); starting over.", project_id, exc)
        return {}
    if not isinstance(data, dict) or data.get("key") != key:
        if isinstance(data, dict) and data.get("key"):
            logger.info("Transcription checkpoint for %s was made from a different source/settings; ignored.",
                        project_id)
        return {}
    phases = data.get("phases")
    return dict(phases) if isinstance(phases, dict) else {}


def save(projects_dir: Path, project_id: str, key: Dict[str, Any], phase: str, payload: Any) -> None:
    """Record `phase`'s result under `key`, keeping the earlier phases made
    under the same key. Never raises: a checkpoint that cannot be written only
    costs the next retry its time."""
    path = _path(projects_dir, project_id)
    phases = load(projects_dir, project_id, key)
    phases[phase] = payload
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "phases": phases}, fh, ensure_ascii=False, default=_json_default)
        os.replace(tmp, path)
        logger.info("Transcription checkpoint for %s: saved phase '%s'.", project_id, phase)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write the transcription checkpoint for %s (%s).", project_id, exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def clear(projects_dir: Path, project_id: str) -> None:
    try:
        _path(projects_dir, project_id).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not remove the transcription checkpoint for %s (%s).", project_id, exc)
