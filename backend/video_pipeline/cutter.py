import logging
from pathlib import Path
from typing import List
from utils.ffmpeg_utils import cut_clip, FFmpegError

logger = logging.getLogger(__name__)


def cut_video_segments(
    source_video: str,
    good_segments: List[tuple],
    project_dir: Path,
) -> List[str]:
    clip_dir = project_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_files = []

    for i, (start, end) in enumerate(good_segments):
        duration = end - start
        if duration < 0.1:
            continue

        clip_path = clip_dir / f"clip_{i:04d}.mp4"
        try:
            cut_clip(source_video, str(clip_path), start, end)
            clip_files.append(str(clip_path))
            logger.info(f"Cut clip {i}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
        except FFmpegError as e:
            logger.error(f"Failed to cut clip {i}: {e}")
            continue

    logger.info(f"Cut {len(clip_files)} clips from {len(good_segments)} segments")
    return clip_files


def cut_single_clip(
    source_video: str,
    start: float,
    end: float,
    output_path: str,
    reencode: bool = True,
) -> str:
    try:
        cut_clip(source_video, output_path, start, end)
        return output_path
    except FFmpegError as e:
        logger.error(f"Failed to cut clip: {e}")
        raise
