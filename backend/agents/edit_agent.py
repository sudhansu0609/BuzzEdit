import logging
from pathlib import Path
from typing import Dict, Any
from models import Project
from config import PROJECTS_DIR, OUTPUT_DIR
from agents.broll_agent import BRollAgent
from agents.thumbnail_agent import ThumbnailAgent
from agents.caption_agent import CaptionAgent
from render.runner import render_timeline_async
from timeline import build_timeline_from_transcript, Timeline
from asr import whisper_engine, analyze_disfluencies
from utils.ffmpeg_utils import get_video_info

logger = logging.getLogger(__name__)

class EditAgent:
    def __init__(self):
        self.broll_agent = BRollAgent()
        self.thumbnail_agent = ThumbnailAgent()
        self.caption_agent = CaptionAgent()

    async def execute_full_auto_edit(
        self,
        project: Project,
        generate_broll: bool = True,
        generate_thumbnail: bool = True,
        burn_captions: bool = True,
    ) -> Dict[str, Any]:
        project_dir = PROJECTS_DIR / project.id
        results = {}

        logger.info(f"EditAgent: Starting full automated edit for project {project.id}")

        # 1. Transcribe & Build EDL Timeline
        v_info = get_video_info(project.source_video)
        words = await whisper_engine.transcribe_audio_async(project.source_video)
        ann_words = analyze_disfluencies(words)

        tl = build_timeline_from_transcript(
            source_path=project.source_video,
            duration_seconds=v_info.get("duration", 0.0),
            transcript_words=ann_words,
            fps_num=int(round(v_info.get("fps", 30))),
            fps_den=1,
            width=v_info.get("width", 1920),
            height=v_info.get("height", 1080)
        )

        # 2. Generate B-Roll into Timeline
        if generate_broll:
            logger.info("EditAgent: Generating B-Roll overlays into EDL timeline")
            added_broll = await self.broll_agent.generate_and_apply_broll(tl, project_dir)
            results["broll_clips"] = added_broll

        # 3. Single-pass Render
        output_path = str(OUTPUT_DIR / f"{project.id}_edited.mp4")
        logger.info(f"EditAgent: Single-pass render to {output_path}")
        rendered_path = await render_timeline_async(tl, output_path)
        results["final_video"] = rendered_path

        # 4. Generate Thumbnail
        if generate_thumbnail:
            logger.info("EditAgent: Generating Thumbnail")
            thumb_path = await self.thumbnail_agent.generate_thumbnail(project, project_dir)
            results["thumbnail"] = thumb_path

        project.output_path = rendered_path
        project.status = "rendered"

        logger.info(f"EditAgent: Complete edit executed successfully for {project.id}")
        return results
