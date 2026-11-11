"""M3 (MOTION_GRAPHICS_PLAN.md): five transparent transition stingers --
``shape_wipe``, ``iris``, ``ink``, ``bars``, ``paper_tear`` -- rendered by the
Remotion engine as ProRes 4444 overlays with alpha and placed on the top
overlay track, centred on the cut a `[fx: <name> dur=0.8]` directive names.

Unlike every other motion kind (kinetic_text, map, chart, stat_callout,
timeline), a stinger is not a cutaway beat competing for a window through
`placement.py` -- it is a short overlay that sits ON TOP of whatever is
already playing at that moment, so this module never builds a real
`presentation.models.Beat` (which would also mean adding a kind to
`BeatKind`/`BEAT_KINDS` and leaking a transition into the planner's own
cutaway vocabulary). `_StingerBeat` is a minimal duck-typed stand-in that
satisfies exactly what `motion.render_motion_assets`/`BUILDERS` need
(`.id`, `.kind`, `.data`, `.planned_duration_s`) -- the caching, batching,
below-normal-priority subprocess and `wait_if_paused` machinery in motion.py
is reused unchanged.
"""

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .models import PresentationSettings
from .motion import MotionContext

logger = logging.getLogger("presentation.motion_transitions")

# directive name -> Remotion composition id (src/registry.ts).
STINGER_COMPOSITIONS: Dict[str, str] = {
    "shape_wipe": "ShapeWipe",
    "iris": "Iris",
    "ink": "Ink",
    "bars": "Bars",
    "paper_tear": "PaperTear",
}
STINGER_NAMES = tuple(STINGER_COMPOSITIONS)

# Composited last (highest track number), so a stinger covers every other
# overlay and adjustment layer -- cards' own V9 shade included.
STINGER_TRACK = "V10"

DEFAULT_STINGER_DUR_S = 0.8


@dataclass
class _StingerBeat:
    """Duck-typed stand-in for a `Beat` -- see module docstring."""
    id: str
    kind: str = "transition_stinger"
    data: Dict[str, Any] = field(default_factory=dict)
    planned_duration_s: Optional[float] = None


def build_transition_stinger(beat: Any, ctx: MotionContext) -> Optional[Dict[str, Any]]:
    """`motion.BUILDERS["transition_stinger"]` -- one Remotion job per named
    stinger. `beat` is a `_StingerBeat`, not a real `Beat` -- see the module
    docstring."""
    name = (beat.data or {}).get("name") if getattr(beat, "data", None) else None
    composition = STINGER_COMPOSITIONS.get(name)
    if not composition:
        return None
    duration_s = max(0.2, float(beat.planned_duration_s or DEFAULT_STINGER_DUR_S))
    frames = max(2, round(duration_s * ctx.fps))
    return {
        "composition": composition,
        "props": {"theme": ctx.theme},
        "width": ctx.width,
        "height": ctx.height,
        "fps": ctx.fps,
        "frames": frames,
        "transparent": True,
    }


def place_stinger(timeline, name: str, at_s: float, dur_s: float, program,
                  project_dir: Path, width: int, height: int, genre: str,
                  settings: PresentationSettings, origin: str) -> bool:
    """Render (or reuse the cache for) one stinger and place it on
    `STINGER_TRACK`, centred on `at_s`. Never raises -- a render/placement
    failure returns False so the caller falls back (whiteflash, or nothing);
    a motion failure must never fail a render."""
    if name not in STINGER_COMPOSITIONS:
        return False
    try:
        from . import motion as motion_stage
        from timeline import clip_ops
        from timeline.schema import SourceFile, time_to_frame

        beat_id = f"stinger_{uuid.uuid4().hex[:8]}"
        beat = _StingerBeat(id=beat_id, data={"name": name}, planned_duration_s=dur_s)
        assets, _failures, handled = motion_stage.render_motion_assets(
            [beat], project_dir, width, height, genre, settings)
        if beat_id not in handled or not assets:
            return False
        asset = assets[0]
        if not Path(asset.path).exists():
            return False

        half = (asset.duration_s or dur_s) / 2.0
        program_len = float(getattr(program, "duration_s", at_s + half) or (at_s + half))
        start_s = max(0.0, at_s - half)
        end_s = min(program_len, start_s + (asset.duration_s or dur_s))
        if end_s <= start_s:
            return False

        source_id = f"src_stinger_{beat_id}"
        timeline.sources[source_id] = SourceFile(
            id=source_id, path=asset.path,
            duration_seconds=asset.duration_s or dur_s,
            width=asset.width or width, height=asset.height or height,
            fps_num=timeline.fps_num, fps_den=timeline.fps_den,
            has_audio=False, kind="video")
        start_frame = time_to_frame(start_s, timeline.fps_num, timeline.fps_den)
        frames = max(1, time_to_frame(end_s - start_s, timeline.fps_num, timeline.fps_den))
        item = clip_ops.add_media_item(timeline, source_id, STINGER_TRACK, start_frame, 0,
                                       frames, origin=origin)
        item.label = f"fx: {name}"
        return True
    except Exception as e:  # noqa: BLE001 - never fail a render over a stinger
        logger.warning("Transition stinger %r could not be placed: %s", name, e)
        return False
