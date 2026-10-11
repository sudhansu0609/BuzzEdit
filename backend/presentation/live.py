"""Settings a running presentation pass re-reads: the mid-production change.

A pass snapshots its `PresentationSettings` when the job starts, and a
long story video spends hours in the clip phase. Changing the mix there
(fewer clips, more stills, more face) used to mean cancelling and starting
over. Instead the creator's new numbers are written next to the project
(`<project_id>_live_settings.json`) and the pass reads them back at the two
places they matter:

- the clip loop (`assets.generate_assets`), before every clip: the clip
  quota is recomputed from the live `video_broll_share`, so a lowered share
  stops promoting stills to clips once the new share is met (a raised one
  promotes more of the highest-priority stills);
- placement (`director`), before the B-roll is cut in: a lowered
  `target_coverage` drops the lowest-priority cutaways until the new
  coverage is met, and the clip share is re-balanced on what remains.

Only the keys in `LIVE_KEYS` are honoured. The file is written by
`PUT /api/presentation/{project_id}/live_settings` (BuzzcafStudio pushes it
when the creator moves the mix sliders during production) and refreshed
with the settings of every new pass, so a stale change never leaks into a
later render.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Beat, PresentationSettings

logger = logging.getLogger("presentation.live")

#: The settings a running pass re-reads. Anything else in the file is ignored.
LIVE_KEYS: Tuple[str, ...] = (
    "target_coverage", "video_broll_share", "image_broll_share",
    "broll_seconds_min", "broll_seconds_max",
)


def path_for(project_dir: Path) -> Path:
    """`<projects>/<project_id>_live_settings.json`, beside the project file."""
    project_dir = Path(project_dir)
    return project_dir.parent / f"{project_dir.name}_live_settings.json"


def read(project_dir: Path) -> Dict[str, Any]:
    """The live overrides for this project, `{}` when there are none."""
    file = path_for(project_dir)
    if not file.exists():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Live settings %s unreadable (%s); ignoring.", file, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in LIVE_KEYS:
        if data.get(key) is None:
            continue
        try:
            out[key] = float(data[key])
        except (TypeError, ValueError):
            continue
    return out


def write(project_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Store `overrides` (LIVE_KEYS only) for the running pass. Returns what
    was stored. An empty set of overrides removes the file."""
    kept = {k: float(v) for k, v in (overrides or {}).items() if k in LIVE_KEYS and v is not None}
    file = path_for(project_dir)
    if not kept:
        clear(project_dir)
        return {}
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = {**kept, "written_at": time.time()}
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(file)
    logger.info("Live settings for %s: %s", Path(project_dir).name, kept)
    return kept


def clear(project_dir: Path) -> None:
    file = path_for(project_dir)
    try:
        file.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not remove live settings %s (%s).", file, exc)


def apply(settings: PresentationSettings, overrides: Dict[str, Any]) -> PresentationSettings:
    """`settings` with the live overrides on top (validated like any other)."""
    fields = {k: v for k, v in (overrides or {}).items() if k in LIVE_KEYS}
    if not fields:
        return settings
    try:
        return settings.model_copy(update=fields)
    except Exception as exc:  # noqa: BLE001 - a bad override must not stop the pass
        logger.warning("Live settings %s rejected (%s); keeping the pass's own.", fields, exc)
        return settings


def planned_length(beat: Beat) -> float:
    return float(beat.planned_duration_s or beat.duration_s or 0.0)


def clip_quota_seconds(cutaways: List[Beat], share: float) -> float:
    """Seconds of B-roll that should be moving video at `share`: BuzzEdit's
    meaning of `video_broll_share`, a share of the B-roll screen time."""
    total = sum(planned_length(b) for b in cutaways if b.is_cutaway)
    return max(0.0, min(1.0, float(share))) * total


def trim_to_coverage(beats: List[Beat], settings: PresentationSettings,
                     duration_s: float) -> Tuple[List[Beat], List[Dict[str, str]]]:
    """Drop the lowest-priority cutaways until they fit the (live) coverage
    budget. Text beats, pop-ups and music cues cost nothing and stay. Unlike
    the planner's own budget this drops script-requested beats too: the
    creator lowering the coverage mid-run means it, whoever asked for the
    beat."""
    budget = settings.broll_budget_seconds(duration_s)
    cutaways = [b for b in beats if b.is_cutaway or b.kind == "graphic"]
    others = [b for b in beats if not (b.is_cutaway or b.kind == "graphic")]
    spent = sum(planned_length(b) for b in cutaways)
    if spent <= budget + 0.001:
        return beats, []
    dropped: List[Dict[str, str]] = []
    keep: List[Beat] = []
    # Walk from the most important beat down; what does not fit is dropped.
    running = 0.0
    for beat in sorted(cutaways, key=lambda b: (-float(b.priority or 0.0), b.start_s)):
        length = planned_length(beat)
        if running + length > budget + 0.001:
            dropped.append({"beat": beat.id, "topic": beat.topic,
                            "reason": "over the coverage set during the run"})
            continue
        keep.append(beat)
        running += length
    kept = sorted(keep + others, key=lambda b: b.start_s)
    return kept, dropped


def rebalance_clips(beats: List[Beat], settings: PresentationSettings,
                    made_clips: Optional[set] = None) -> List[Beat]:
    """Re-kind cutaways to the (live) video share: beats whose clip exists
    (`made_clips`, by beat id) are kept as video first, then the highest
    priority ones, until the share is met; the rest become stills."""
    share = max(0.0, min(1.0, float(settings.video_broll_share)))
    cutaways = [b for b in beats if b.is_cutaway and b.is_generated]
    quota = clip_quota_seconds(cutaways, share)
    made_clips = made_clips or set()
    out = {b.id: b for b in beats}
    ordered = sorted(cutaways, key=lambda b: (b.id not in made_clips, -float(b.priority or 0.0), b.start_s))
    video_secs = 0.0
    for beat in ordered:
        length = planned_length(beat)
        wants_video = video_secs + length <= quota + 0.001 or (beat.id in made_clips and video_secs < quota)
        if wants_video and beat.image_prompt:
            if beat.kind != "broll_video":
                out[beat.id] = beat.model_copy(update={
                    "kind": "broll_video",
                    "video_prompt": beat.video_prompt or f"{beat.image_prompt}, gentle cinematic camera movement, "
                                                         f"natural motion in the scene"})
            video_secs += length
        elif beat.kind == "broll_video":
            out[beat.id] = beat.model_copy(update={"kind": "broll_image"})
    return [out[b.id] for b in beats]


def describe(overrides: Dict[str, Any]) -> str:
    parts = []
    for key in LIVE_KEYS:
        if key in overrides:
            value = overrides[key]
            parts.append(f"{key}={int(round(value * 100))}%" if "share" in key or "coverage" in key
                         else f"{key}={value:g}s")
    return ", ".join(parts)
