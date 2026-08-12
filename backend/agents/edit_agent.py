import logging
from pathlib import Path
from typing import Dict, Any, Optional
from models import Project
from config import PROJECTS_DIR, OUTPUT_DIR
from agents.broll_agent import BRollAgent
from agents.thumbnail_agent import ThumbnailAgent
from agents.caption_agent import CaptionAgent
from routes.transcription import run_transcription
from routes.analysis import run_analysis
from routes.rendering import run_render

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

        # 1. Transcribe
        if not project.transcript:
            logger.info("EditAgent: Step 1 - Transcribing video")
            project.transcript = run_transcription(project.source_video)
            project.status = "transcribed"

        # 2. Analyze & Cut fumbles/silence
        if not project.detected_segments:
            logger.info("EditAgent: Step 2 - Analyzing fumbles & silence gaps")
            project.detected_segments = run_analysis(
                project,
                remove_silence=project.settings.get("remove_silence", True),
                remove_fumbles=project.settings.get("remove_fumbles", True),
            )
            project.status = "analyzed"

        # 3. Base Render Cut
        output_cut_path = str(project_dir / "clean_cut.mp4")
        logger.info("EditAgent: Step 3 - Rendering clean cut video")
        run_render(project, output_cut_path, project.settings)
        results["clean_cut"] = output_cut_path

        current_video = output_cut_path

        # 4. Generate B-Roll
        if generate_broll and project.transcript:
            logger.info("EditAgent: Step 4 - Generating B-Roll overlays")
            broll_clips = await self.broll_agent.generate_broll_for_project(project, project_dir)
            results["broll_clips"] = len(broll_clips)

        # 5. Burn-in Styled Captions
        if burn_captions and project.transcript and project.transcript.segments:
            logger.info("EditAgent: Step 5 - Burning in captions")
            captioned_video = str(OUTPUT_DIR / f"{project.id}_edited.mp4")
            self.caption_agent.burn_captions(
                input_video=current_video,
                output_video=captioned_video,
                segments=project.transcript.segments,
            )
            current_video = captioned_video
            results["final_video"] = captioned_video

        # 6. Generate Thumbnail
        if generate_thumbnail:
            logger.info("EditAgent: Step 6 - Generating Thumbnail")
            thumb_path = await self.thumbnail_agent.generate_thumbnail(project, project_dir)
            results["thumbnail"] = thumb_path

        project.output_path = current_video
        project.status = "rendered"

        # Save updated project
        (project_dir / "project.json").write_text(project.model_dump_json(indent=2))

        logger.info(f"EditAgent: Complete edit executed successfully for {project.id}")
        return results
