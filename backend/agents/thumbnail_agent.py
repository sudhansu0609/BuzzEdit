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
                                 genre: Optional[str] = None) -> str:
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
        prompt = f"Eye-catching YouTube thumbnail, title '{display_title}', {style}, 4k"

        try:
            workflow = prepare_thumbnail_workflow(prompt=prompt, output_prefix=f"thumb_{project.id}")
            output_files = await self.queue_manager.submit_and_wait(workflow, timeout=120)

            if output_files and Path(output_files[0]).exists():
                import shutil
                shutil.copy(output_files[0], thumbnail_path)
                logger.info(f"Generated AI thumbnail via ComfyUI: {thumbnail_path}")
                return str(thumbnail_path)
        except Exception as e:
            logger.warning(f"ComfyUI thumbnail generation offline ({e}). Generating fallback thumbnail.")

        # Fallback thumbnail with text overlay on keyframe. drawtext needs an
        # explicit fontfile — the fontconfig default crashes on Windows ffmpeg
        # builds with no config. Scale the title to the frame so it reads as a
        # title rather than a tiny fixed 48px line pinned near the bottom.
        from utils.fonts import resolve_font_file
        from render.effects import escape_filter_path
        font_file = resolve_font_file(None)
        font_arg = f"fontfile='{escape_filter_path(font_file)}':" if font_file else ""
        safe_title = display_title.replace("'", "").replace('"', "")
        run_ffmpeg([
            "-i", str(keyframe_path),
            "-vf", (f"eq=contrast=1.2:saturation=1.3,"
                    f"drawtext={font_arg}text='{safe_title}':fontcolor=yellow:"
                    f"fontsize=h/12:x=(w-text_w)/2:y=h-(h/8):"
                    f"box=1:boxcolor=black@0.6:boxborderw=10"),
            "-q:v", "2",
            "-y",
            str(thumbnail_path),
        ])

        return str(thumbnail_path)
