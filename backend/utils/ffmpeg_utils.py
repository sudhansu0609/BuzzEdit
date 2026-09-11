import subprocess
import json
import logging
from pathlib import Path
from typing import Optional
from config import FFMPEG_BIN, FFPROBE_BIN, TEMP_DIR
from utils.proc import NO_WINDOW

logger = logging.getLogger(__name__)


class FFmpegError(Exception):
    pass


def run_ffmpeg(args: list[str], timeout: int = 3600) -> str:
    cmd = [FFMPEG_BIN] + args
    logger.info(f"FFmpeg: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=NO_WINDOW,
    )
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise FFmpegError(f"FFmpeg failed: {result.stderr}")
    return result.stdout


def run_ffprobe(path: str, stream: Optional[str] = None) -> dict:
    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
    ]
    if stream:
        cmd.extend(["-select_streams", stream])
    cmd.append(path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                            creationflags=NO_WINDOW)
    if result.returncode != 0:
        raise FFmpegError(f"FFprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def get_video_duration(path: str) -> float:
    info = run_ffprobe(path)
    if "format" in info and "duration" in info["format"]:
        return float(info["format"]["duration"])
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and "duration" in stream:
            return float(stream["duration"])
    raise FFmpegError("Could not determine duration")


def parse_frame_rate(rate: str, default_num: int = 30, default_den: int = 1) -> tuple[int, int]:
    """Parse an ffprobe frame-rate string (e.g. "30000/1001" or "25/1") into an
    exact (numerator, denominator) integer pair. Falls back to the default on
    malformed or zero rates (ffprobe reports "0/0" for streams with no timing)."""
    try:
        num_str, _, den_str = str(rate).partition("/")
        num = int(num_str)
        den = int(den_str) if den_str else 1
        if num <= 0 or den <= 0:
            return default_num, default_den
        return num, den
    except (ValueError, TypeError):
        return default_num, default_den


def get_video_info(path: str) -> dict:
    info = run_ffprobe(path)
    video_stream = None
    audio_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
        elif s.get("codec_type") == "audio":
            audio_stream = s

    raw_fps = video_stream.get("r_frame_rate", "30/1") if video_stream else "30/1"
    fps_num, fps_den = parse_frame_rate(raw_fps)

    return {
        "duration": float(info["format"]["duration"]) if "duration" in info["format"] else 0,
        "width": int(video_stream.get("width", 0)) if video_stream else 0,
        "height": int(video_stream.get("height", 0)) if video_stream else 0,
        "fps": fps_num / fps_den,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "video_codec": video_stream.get("codec_name", "unknown") if video_stream else "none",
        "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else "none",
        "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
        "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        "bitrate": int(info["format"].get("bit_rate", 0)) if "bit_rate" in info["format"] else 0,
    }


def extract_audio(video_path: str, output_path: Optional[str] = None, sample_rate: int = 48000) -> str:
    if output_path is None:
        output_path = str(TEMP_DIR / f"{Path(video_path).stem}_audio.wav")
    run_ffmpeg([
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-y",
        output_path,
    ])
    return output_path


def extract_frames(video_path: str, fps: float = 0.5, output_dir: Optional[str] = None) -> str:
    if output_dir is None:
        output_dir = str(TEMP_DIR / f"{Path(video_path).stem}_frames")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-y",
        f"{output_dir}/frame_%05d.jpg",
    ])
    return output_dir


def concat_clips(input_files: list[str], output_path: str, transition: str = "none", transition_duration: float = 0.5) -> str:
    if len(input_files) == 1:
        run_ffmpeg([
            "-i", input_files[0],
            "-c", "copy",
            "-y",
            output_path,
        ])
        return output_path

    if transition == "none":
        filter_complex = ""
        for i, f in enumerate(input_files):
            filter_complex += f"[{i}:v][{i}:a]"
        filter_complex += f"concat=n={len(input_files)}:v=1:a=1[outv][outa]"

        run_ffmpeg([
            *[arg for f in input_files for arg in ["-i", f]],
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            output_path,
        ])
    else:
        concat_with_transitions(input_files, output_path, transition, transition_duration)

    return output_path


def concat_with_transitions(input_files: list[str], output_path: str, transition: str = "xfade", transition_duration: float = 0.5) -> str:
    n = len(input_files)
    if n < 2:
        concat_clips(input_files, output_path, "none")
        return output_path

    video_filters = []
    audio_filters = []
    last_video = "[0:v]"
    last_audio = "[0:a]"

    accumulated_duration = get_video_duration(input_files[0])

    for i in range(1, n):
        offset = max(0.0, accumulated_duration - transition_duration)
        clip_dur = get_video_duration(input_files[i])

        if transition == "xfade":
            video_filters.append(
                f"{last_video}[{i}:v]xfade=transition=fade:duration={transition_duration}:offset={offset:.3f}[v{i}];"
            )
        elif transition == "zoomin":
            video_filters.append(
                f"{last_video}[{i}:v]xfade=transition=zoomin:duration={transition_duration}:offset={offset:.3f}[v{i}];"
            )
        elif transition == "dissolve":
            video_filters.append(
                f"{last_video}[{i}:v]xfade=transition=dissolve:duration={transition_duration}:offset={offset:.3f}[v{i}];"
            )
        else:
            video_filters.append(
                f"{last_video}[{i}:v]xfade=transition=fade:duration={transition_duration}:offset={offset:.3f}[v{i}];"
            )

        audio_filters.append(
            f"{last_audio}[{i}:a]acrossfade=d={transition_duration}:c1=tri:c2=tri[a{i}];"
        )

        accumulated_duration = offset + clip_dur
        last_video = f"[v{i}]"
        last_audio = f"[a{i}]"

    filter_str = "".join(video_filters + audio_filters).rstrip(";")

    run_ffmpeg([
        *[arg for f in input_files for arg in ["-i", f]],
        "-filter_complex", filter_str,
        "-map", last_video,
        "-map", last_audio,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path,
    ])
    return output_path



def cut_clip(input_path: str, output_path: str, start: float, end: float) -> str:
    duration = end - start
    run_ffmpeg([
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path,
    ])


def add_fade_in(audio_path: str, output_path: str, duration: float = 0.5) -> str:
    run_ffmpeg([
        "-i", audio_path,
        "-af", f"afade=t=in:st=0:d={duration}",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path,
    ])


def add_fade_out(audio_path: str, output_path: str, duration: float = 0.5, total_duration: Optional[float] = None) -> str:
    if total_duration is None:
        total_duration = get_video_duration(audio_path)
    start = total_duration - duration
    run_ffmpeg([
        "-i", audio_path,
        "-af", f"afade=t=out:st={start}:d={duration}",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path,
    ])


def resize_video(input_path: str, output_path: str, width: int, height: int) -> str:
    run_ffmpeg([
        "-i", input_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        "-c:v", "libx264",
        "-crf", "18",
        "-c:a", "copy",
        "-y",
        output_path,
    ])


def add_background_music(video_path: str, music_path: str, output_path: str, volume: float = 0.1) -> str:
    run_ffmpeg([
        "-i", video_path,
        "-i", music_path,
        "-filter_complex", f"[1:a]aloop=loop=-1:size=2e6,volume={volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-y",
        output_path,
    ])
