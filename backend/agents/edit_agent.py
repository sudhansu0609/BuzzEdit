import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from models import Project
from config import PROJECTS_DIR, OUTPUT_DIR
from agents.thumbnail_agent import ThumbnailAgent
from agents.caption_agent import CaptionAgent
from render.runner import render_timeline_async
from timeline import build_timeline_from_transcript, Timeline, generate_captions
from timeline.ops import apply_auto_zoom
from timeline.schema import Transition
from asr import whisper_engine
from asr.auto_edit import extract_project_audio, pacing_kwargs, plan_auto_edit, record_cut_coverage
from store.project_store import ProjectStore
from utils.ffmpeg_utils import get_video_info

logger = logging.getLogger(__name__)


class EditAgent:
    def __init__(self):
        self.thumbnail_agent = ThumbnailAgent()
        self.caption_agent = CaptionAgent()
        self.store = ProjectStore(base_dir=str(PROJECTS_DIR))

    async def execute_full_auto_edit(
        self,
        project: Project,
        generate_thumbnail: bool = True,
        burn_captions: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        project_dir = PROJECTS_DIR / project.id
        results: Dict[str, Any] = {}
        report = progress_cb or (lambda fraction, message="": None)

        logger.info(f"EditAgent: Starting full automated edit for project {project.id}")

        # 1. Transcribe & Build EDL Timeline
        #
        # The audio has to be extracted here, not assumed. This call used to pass
        # an `audio_path` that was never assigned, so the route raised NameError
        # every time it was invoked; before that it planned the edit from the
        # transcript alone, which finds about 1% of the fumbles.
        report(0.02, "Extracting audio")
        v_info = get_video_info(project.source_video)
        audio_path = await extract_project_audio(project.source_video)

        report(0.05, "Transcribing")
        # Pin the language across runs — auto-detect flips code-switched speech
        # into an English paraphrase with useless timings.
        agent_settings = project.settings or {}
        words, detected_language = await whisper_engine.transcribe_words_async(
            audio_path or project.source_video, language=agent_settings.get("language"))
        agent_settings["language"] = detected_language
        project.settings = agent_settings

        report(0.35, "Planning the edit")
        aggressiveness = float(agent_settings.get("fumble_aggressiveness", 0.5))
        plan = await plan_auto_edit(words, audio_path, aggressiveness=aggressiveness,
                                    settings=agent_settings)
        edit_report = plan.report

        tl = build_timeline_from_transcript(
            source_path=project.source_video,
            duration_seconds=v_info.get("duration", 0.0),
            transcript_words=plan.words,
            fps_num=v_info.get("fps_num", 30),
            fps_den=v_info.get("fps_den", 1),
            width=v_info.get("width", 1920),
            height=v_info.get("height", 1080),
            has_audio=v_info.get("audio_codec") not in (None, "none"),
            speech_regions=edit_report.get("speech"),
            energy_envelope=edit_report.get("energy"),
            **pacing_kwargs(agent_settings),
        )
        record_cut_coverage(tl, edit_report)
        results["edit_report"] = plan.public_report

        # Match the interactive auto-edit: soften every jump cut with a short
        # dissolve, and give the still talking-head footage a gentle Ken Burns
        # move on each segment. Both are on by default and tunable per project.
        if tl.default_transition is None and not agent_settings.get("hard_cuts"):
            tl.default_transition = Transition(
                type="fade", duration=float(agent_settings.get("transition_seconds", 0.25)))
        if agent_settings.get("auto_zoom", True):
            apply_auto_zoom(tl, depth=float(agent_settings.get("zoom_depth", 0.10)))

        # 2. Captions, in the EDL rather than burned into a finished file. The
        #    standalone CaptionAgent re-encodes an already-rendered mp4; doing it
        #    here means one render and captions the user can restyle afterwards.
        if burn_captions:
            report(0.45, "Generating captions")
            settings = project.settings or {}
            items = generate_captions(
                tl,
                preset=settings.get("caption_preset", "classic"),
                style_overrides=settings.get("caption_style_overrides"),
            )
            results["captions"] = len(items)

        # 3. Save the timeline BEFORE rendering.
        #
        # This is what makes an overnight run reviewable: the render is a file,
        # but the timeline is the edit. Without this the project on disk still
        # had no timeline the next morning and there was nothing to adjust.
        self._save(project, tl)

        # 4. Single-pass Render
        output_path = str(OUTPUT_DIR / f"{project.id}_edited.mp4")
        logger.info(f"EditAgent: Single-pass render to {output_path}")
        rendered_path = await render_timeline_async(
            tl, output_path,
            progress_callback=lambda pct: report(0.6 + 0.35 * (pct / 100.0), "Rendering"),
            output_resolution=(project.settings or {}).get("resolution"),
        )
        results["final_video"] = rendered_path

        # 5. Generate Thumbnail
        if generate_thumbnail:
            report(0.96, "Generating thumbnail")
            try:
                thumb_path = await self.thumbnail_agent.generate_thumbnail(project, project_dir)
                results["thumbnail"] = thumb_path
            except Exception as e:
                logger.warning("EditAgent: thumbnail generation failed (%s); continuing", e)

        project.output_path = rendered_path
        project.status = "rendered"
        self._save(project, tl, rendered_path)

        if output_dir:
            results["copied"] = _collect_outputs(
                output_dir, rendered_path, results.get("thumbnail"))

        report(1.0, "Done")
        logger.info(f"EditAgent: Complete edit executed successfully for {project.id}")
        return results

    def _save(self, project: Project, timeline: Timeline,
              output_path: Optional[str] = None) -> None:
        """Write the timeline back to the project on disk."""
        data = self.store.get_project(project.id) or {}
        data["timeline"] = timeline.model_dump()
        if output_path:
            data["output_path"] = output_path
            data["status"] = "rendered"
        self.store.save_project(project.id, data)


def _collect_outputs(output_dir: Path, *paths: Optional[str]) -> list:
    """Copy the run's deliverables into the job's dated folder.

    The scheduler has always created `output/YYYY-MM-DD/<project>/` and reported
    it as the job's output directory, while the agent wrote to a flat file
    elsewhere — so the advertised folder was always empty.
    """
    copied = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path:
            continue
        source = Path(path)
        if not source.exists():
            continue
        try:
            destination = output_dir / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination))
        except Exception as e:
            logger.warning("Could not copy %s into %s: %s", source, output_dir, e)
    return copied
