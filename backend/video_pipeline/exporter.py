import logging
from pathlib import Path
from typing import List
from utils.ffmpeg_utils import (
    concat_with_transitions,
    resize_video,
    FFmpegError,
    run_ffmpeg,
)

logger = logging.getLogger(__name__)


def export_final_video(
    clip_files: List[str],
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    transition_type: str = "xfade",
    transition_duration: float = 0.5,
    crf: int = 18,
    use_nvenc: bool = True,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(clip_files) == 1:
        source = clip_files[0]
        _encode_single(source, output_path, width, height, fps, crf, use_nvenc)
        return output_path

    temp_output = str(Path(output_path).with_stem("temp_transitions"))

    concat_with_transitions(
        clip_files,
        temp_output,
        transition=transition_type,
        transition_duration=transition_duration,
    )

    _encode_single(temp_output, output_path, width, height, fps, crf, use_nvenc)

    temp_path = Path(temp_output)
    if temp_path.exists():
        temp_path.unlink()

    logger.info(f"Export complete: {output_path}")
    return output_path


def _encode_single(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: int,
    crf: int,
    use_nvenc: bool,
) -> None:
    if use_nvenc:
        _encode_nvenc(input_path, output_path, width, height, fps, crf)
    else:
        _encode_x264(input_path, output_path, width, height, fps, crf)


def _encode_nvenc(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: int,
    crf: int,
) -> None:
    qp = max(1, min(51, 51 - crf))
    run_ffmpeg([
        "-i", input_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps}",
        "-c:v", "h264_nvenc",
        "-qp", str(qp),
        "-preset", "p4",
        "-tune", "hq",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-y",
        output_path,
    ])


def _encode_x264(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: int,
    crf: int,
) -> None:
    run_ffmpeg([
        "-i", input_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-y",
        output_path,
    ])
