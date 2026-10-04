import logging
from pathlib import Path
from typing import Optional
from models import Project
from comfyui_bridge.workflow_loader import prepare_thumbnail_workflow
from comfyui_bridge.queue_manager import ComfyUIQueueManager
from utils.ffmpeg_utils import run_ffmpeg

logger = logging.getLogger(__name__)


class ThumbnailAgent:
    def __init__(self, queue_manager: Optional[ComfyUIQueueManager] = None):
        self.queue_manager = queue_manager or ComfyUIQueueManager()

    async def generate_thumbnail(self, project: Project, project_dir: Path,
                                 title: Optional[str] = None,
                                 genre: Optional[str] = None,
                                 scene: Optional[str] = None) -> str:
        thumbnail_path = project_dir / "thumbnail.jpg"
        keyframe_path = project_dir / "keyframe.jpg"

        # 1. Extract best frame around 1/3 of the video duration
        timestamp = 2.0
        if project.settings.get("source_duration"):
            timestamp = max(1.0, float(project.settings["source_duration"]) * 0.3)

        run_ffmpeg([
            "-ss", str(timestamp),
            "-i", project.source_video,
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(keyframe_path),
        ])

        display_title = title or project.name.split('.')[0].replace('_', ' ').replace('-', ' ').title()
        # Style the thumbnail to the video's genre — a horror video gets a
        # horror thumbnail, not the generic "dramatic lighting" one.
        from presentation.genre import style_for
        style = style_for(genre).thumbnail
        # The picture is the video's own subject (the scene the model read out
        # of the transcript); without one, the title is the best description we
        # have. Text is never asked of the image model -- it garbles letters --
        # and is drawn on top afterwards.
        subject = scene or f"a scene that shows '{display_title}' literally"
        prompt = (f"{subject}. YouTube thumbnail composition, one clear subject, close "
                  f"framing, bold contrast, {style}, 4k, no text, no letters, no watermark")

        try:
            workflow = prepare_thumbnail_workflow(prompt=prompt, output_prefix=f"thumb_{project.id}")
            output_files = await self.queue_manager.submit_and_wait(workflow, timeout=120)

            if output_files and Path(output_files[0]).exists():
                self._overlay_title(Path(output_files[0]), thumbnail_path, display_title, boost=False)
                logger.info(f"Generated AI thumbnail via ComfyUI: {thumbnail_path}")
                return str(thumbnail_path)
        except Exception as e:
            logger.warning(f"ComfyUI thumbnail generation offline ({e}). Generating fallback thumbnail.")

        # Fallback thumbnail: the title over a keyframe of the video itself.
        self._overlay_title(keyframe_path, thumbnail_path, display_title, boost=True)
        return str(thumbnail_path)

    @staticmethod
    def _overlay_title(source: Path, destination: Path, title: str, boost: bool) -> None:
        """Draw the title large across the lower third. drawtext needs an
        explicit fontfile -- the fontconfig default crashes on Windows ffmpeg
        builds with no config. Scaled to the frame so it reads as a title."""
        from utils.fonts import resolve_font_file
        from render.effects import escape_filter_path
        font_file = resolve_font_file(None)
        font_arg = f"fontfile='{escape_filter_path(font_file)}':" if font_file else ""
        safe_title = title.replace("'", "").replace('"', "").replace(":", " ").replace("%", " percent")
        grade = "eq=contrast=1.2:saturation=1.3," if boost else ""
        run_ffmpeg([
            "-i", str(source),
            "-vf", (f"{grade}"
                    f"drawtext={font_arg}text='{safe_title}':fontcolor=yellow:"
                    f"fontsize=h/9:x=(w-text_w)/2:y=h-text_h-(h/10):"
                    f"borderw=6:bordercolor=black:"
                    f"box=1:boxcolor=black@0.45:boxborderw=14"),
            "-q:v", "2",
            "-y",
            str(destination),
        ])
