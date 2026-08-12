import logging
from pathlib import Path
from typing import List, Optional
from utils.ffmpeg_utils import concat_with_transitions, FFmpegError

logger = logging.getLogger(__name__)


def apply_transitions(
    clip_files: List[str],
    output_path: str,
    transition_type: str = "xfade",
    transition_duration: float = 0.5,
    width: int = 1920,
    height: int = 1080,
) -> str:
    if len(clip_files) <= 1:
        if clip_files:
            return clip_files[0]
        raise ValueError("No clips to process")

    logger.info(
        f"Applying {transition_type} transitions "
        f"({transition_duration}s) to {len(clip_files)} clips"
    )

    try:
        concat_with_transitions(
            clip_files,
            output_path,
            transition=transition_type,
            transition_duration=transition_duration,
        )
        logger.info(f"Transition composition complete: {output_path}")
        return output_path

    except FFmpegError as e:
        logger.error(f"Transition composition failed: {e}")
        raise


def auto_select_transition(
    scene_type: str,
    pace: str = "medium",
) -> tuple[str, float]:
    pace_durations = {
        "slow": 1.0,
        "medium": 0.5,
        "fast": 0.25,
    }

    duration = pace_durations.get(pace, 0.5)

    transition_map = {
        "interview": ("xfade", duration),
        "tutorial": ("xfade", duration * 0.75),
        "vlog": ("zoomin", duration),
        "gaming": ("xfade", duration * 0.5),
        "cinematic": ("xfade", duration * 1.5),
    }

    return transition_map.get(scene_type, ("xfade", duration))
