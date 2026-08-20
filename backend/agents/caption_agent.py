import logging
from pathlib import Path
from typing import List, Optional
from models import Project, TranscriptSegment
from utils.ffmpeg_utils import run_ffmpeg

logger = logging.getLogger(__name__)


class CaptionAgent:
    def create_srt_file(self, segments: List[TranscriptSegment], output_path: str) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            start_str = self._format_timestamp(seg.start)
            end_str = self._format_timestamp(seg.end)
            text = seg.text.strip()
            lines.append(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")

        Path(output_path).write_text("".join(lines), encoding="utf-8")
        return output_path

    def _format_timestamp(self, seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    def burn_captions(
        self,
        input_video: str,
        output_video: str,
        segments: List[TranscriptSegment],
        font_size: int = 24,
        font_color: str = "white",
        box_color: str = "black@0.6",
    ) -> str:
        if not segments:
            import shutil
            shutil.copy(input_video, output_video)
            return output_video

        srt_path = str(Path(output_video).with_suffix(".srt"))
        self.create_srt_file(segments, srt_path)

        # Convert backslashes for FFmpeg filter path escaping on Windows
        escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")

        try:
            run_ffmpeg([
                "-i", input_video,
                "-vf", f"subtitles='{escaped_srt}':force_style='FontSize={font_size},PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=4'",
                "-c:v", "libx264",
                "-crf", "18",
                "-c:a", "copy",
                "-y",
                output_video,
            ])
            return output_video
        except Exception as e:
            logger.warning(f"Subtitles filter failed ({e}), rendering fallback drawtext subtitles.")
            return self._burn_fallback_drawtext(input_video, output_video, segments, font_size)

    def _burn_fallback_drawtext(
        self,
        input_video: str,
        output_video: str,
        segments: List[TranscriptSegment],
        font_size: int,
    ) -> str:
        # drawtext with no fontfile falls back to fontconfig's default, which on a
        # Windows ffmpeg build with no fontconfig config aborts the whole render
        # (exit 139). Resolve a real font once and pin it on every line.
        from utils.fonts import resolve_font_file
        from render.effects import escape_filter_path
        font_file = resolve_font_file(None)
        font_arg = f"fontfile='{escape_filter_path(font_file)}':" if font_file else ""

        drawtext_filters = []
        for seg in segments[:20]: # Escape text for FFmpeg drawtext
            safe_text = seg.text.replace("'", "").replace(":", "").replace("\\", "")
            drawtext_filters.append(
                f"drawtext={font_arg}text='{safe_text}':fontcolor=white:fontsize={font_size}:"
                f"x=(w-text_w)/2:y=h-80:enable='between(t,{seg.start},{seg.end})':"
                f"box=1:boxcolor=black@0.6:boxborderw=6"
            )

        filter_str = ",".join(drawtext_filters) if drawtext_filters else "null"

        run_ffmpeg([
            "-i", input_video,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-crf", "18",
            "-c:a", "copy",
            "-y",
            output_video,
        ])
        return output_video
