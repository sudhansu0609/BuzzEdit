import logging
from pathlib import Path
from utils.ffmpeg_utils import run_ffmpeg

logger = logging.getLogger(__name__)

COLOR_PRESETS = {
    "cinematic": "eq=contrast=1.15:brightness=-0.02:saturation=1.2:gamma=0.95",
    "vibrant": "eq=contrast=1.25:saturation=1.5:brightness=0.02",
    "warm": "eq=contrast=1.1:saturation=1.25:gamma_r=1.1:gamma_b=0.9",
    "cool": "eq=contrast=1.1:saturation=1.1:gamma_r=0.9:gamma_b=1.15",
    "dark": "eq=contrast=1.3:brightness=-0.08:saturation=1.3",
}


def apply_color_grading(input_video: str, output_video: str, preset: str = "cinematic") -> str:
    eq_filter = COLOR_PRESETS.get(preset.lower(), COLOR_PRESETS["cinematic"])
    run_ffmpeg([
        "-i", input_video,
        "-vf", eq_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-c:a", "copy",
        "-y",
        output_video,
    ])
    return output_video


def convert_to_vertical_shorts(input_video: str, output_video: str, target_w: int = 1080, target_h: int = 1920) -> str:
    crop_filter = f"scale=-1:{target_h},crop={target_w}:{target_h}"
    run_ffmpeg([
        "-i", input_video,
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-c:a", "copy",
        "-y",
        output_video,
    ])
    return output_video


def mix_audio_with_ducking(vocal_video: str, music_file: str, output_video: str, music_volume: float = 0.25) -> str:
    filter_complex = (
        f"[1:a]volume={music_volume}[music];"
        f"[music][0:a]sidechaincompress=threshold=0.08:ratio=4:attack=20:release=300[ducked];"
        f"[0:a][ducked]amix=inputs=2:duration=first[aout]"
    )

    run_ffmpeg([
        "-i", vocal_video,
        "-i", music_file,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-y",
        output_video,
    ])
    return output_video


def remove_frame_background(input_image_path: str, output_image_path: str) -> str:
    import rembg
    from PIL import Image

    inp = Image.open(input_image_path)
    out = rembg.remove(inp)
    out.save(output_image_path)
    return output_image_path
