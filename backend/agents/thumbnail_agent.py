"""The thumbnail: the story's scene, the creator's own picture, a big title.

Since 2026-10-21 it is composed the way the creator's reference thumbnails
are (`presentation/thumbnail_compose.py`): an AI picture of the story's key
moment fills the frame, his picture takes the right third behind a torn white
edge, and two to four Devanagari words in blood red sit over the scene with
a small channel label top-left. Before that it was one SDXL picture with a
yellow caption across the bottom.

Pieces, each with a fallback:
- the scene: the project's own still workflow (`broll_image`, Z-Image -- the
  same world as the video's pictures) from the concept's scene line; else the
  old thumbnail.json graph; else no scene (the creator's frame, darkened);
- the creator's picture: `thumbnail_base.*` beside the recording or in its
  channel folder (`thumbnail_compose.find_base_picture`), else a frame of the
  recording itself;
- the words: the concept's `thumb` line (Devanagari, 2-4 words), else the
  title, else the project name.
"""
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from models import Project
from comfyui_bridge.workflow_loader import prepare_thumbnail_workflow
from comfyui_bridge.queue_manager import ComfyUIQueueManager
from utils.ffmpeg_utils import run_ffmpeg

logger = logging.getLogger(__name__)

SCENE_TIMEOUT_S = 300
DEFAULT_LABELS = {"horror": "HORROR STORY", "true_crime": "TRUE CRIME"}
_CAMERA = re.compile(r"\bcameras?\b", re.IGNORECASE)   # the word draws a camera into the picture


class ThumbnailAgent:
    def __init__(self, queue_manager: Optional[ComfyUIQueueManager] = None):
        self.queue_manager = queue_manager or ComfyUIQueueManager()

    async def generate_thumbnail(self, project: Project, project_dir: Path,
                                 title: Optional[str] = None,
                                 genre: Optional[str] = None,
                                 scene: Optional[str] = None,
                                 words: Optional[str] = None,
                                 label: Optional[str] = None,
                                 art_style: str = "real") -> str:
        from presentation import thumbnail_compose as compose

        thumbnail_path = project_dir / "thumbnail.jpg"
        keyframe_path = project_dir / "keyframe.jpg"
        self._extract_keyframe(project, keyframe_path)

        display_title = title or project.name.split('.')[0].replace('_', ' ').replace('-', ' ').title()
        text = (words or "").strip() or display_title
        if label is None:
            label = DEFAULT_LABELS.get((genre or "").lower(), "")

        base = compose.find_base_picture(project.source_video, project.settings) \
            or (keyframe_path if keyframe_path.exists() else None)
        scene_file = await self._scene_picture(project, project_dir, scene or display_title, genre, art_style)

        try:
            compose.compose(scene_file, base, text, thumbnail_path, label=label, genre=genre,
                            seed=sum(ord(c) for c in project.id))
            logger.info("Thumbnail composed: %s (scene=%s, creator=%s, words=%r)", thumbnail_path,
                        scene_file.name if scene_file else None, base.name if base else None, text)
            return str(thumbnail_path)
        except Exception as e:  # noqa: BLE001 - the old caption-over-a-frame still beats nothing
            logger.warning("Thumbnail composition failed (%s); falling back to a captioned frame", e)
        source = scene_file if scene_file and scene_file.exists() else keyframe_path
        self._overlay_title(source, thumbnail_path, display_title, boost=source is keyframe_path)
        return str(thumbnail_path)

    # ------------------------------------------------------------------ parts

    @staticmethod
    def _extract_keyframe(project: Project, keyframe_path: Path) -> None:
        """A frame around a third of the way in, the creator's own picture of
        last resort."""
        timestamp = 2.0
        if project.settings.get("source_duration"):
            timestamp = max(1.0, float(project.settings["source_duration"]) * 0.3)
        try:
            run_ffmpeg(["-ss", str(timestamp), "-i", project.source_video, "-vframes", "1",
                        "-q:v", "2", "-y", str(keyframe_path)])
        except Exception as e:  # noqa: BLE001
            logger.debug("Keyframe extraction failed: %s", e)

    async def _scene_picture(self, project: Project, project_dir: Path, subject: str,
                             genre: Optional[str], art_style: str = "real") -> Optional[Path]:
        """The story's scene as a picture, from the still workflow the video's
        own pictures come from; the old thumbnail graph when that is not set.
        A drawn art style gets its medium instead of the film-still words."""
        from presentation import art_style as art_mod, workflows
        from presentation.genre import style_for

        if art_mod.is_real(art_style):
            style = style_for(genre).thumbnail
            prompt = _CAMERA.sub("", f"{subject}. Cinematic film still, one clear subject, close framing, "
                                     f"strong contrast, {style}, no text, no letters, no logos, no watermark")
        else:
            mood = art_mod.genre_mood(genre)
            prompt = _CAMERA.sub("", f"{subject}. {art_mod.thumbnail_look(art_style)}, "
                                     f"{art_mod.look_for(art_style)}, {mood + ', ' if mood else ''}"
                                     f"no text, no letters, no logos, no watermark")
        destination = project_dir / "thumbnail_scene.png"
        graph = None
        resolved = workflows.resolve("broll_image")
        if resolved is not None and resolved.output_kind == "image":
            try:
                graph = resolved.build(positive=prompt, width=1280, height=720,
                                       seed=sum(ord(c) for c in project.id) % 1_000_000,
                                       prefix=f"thumb_scene_{project.id}")
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not build the still graph for the thumbnail scene: %s", e)
        if graph is None:
            graph = prepare_thumbnail_workflow(prompt=prompt, output_prefix=f"thumb_{project.id}")
        try:
            outputs = await self.queue_manager.submit_and_wait(graph, timeout=SCENE_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            logger.warning("Thumbnail scene generation failed (%s); composing without a scene", e)
            return None
        chosen = next((o for o in outputs or [] if Path(o).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")),
                      None)
        if not chosen or not Path(chosen).exists():
            return None
        try:
            shutil.copy2(chosen, destination)
            return destination
        except Exception:  # noqa: BLE001
            return Path(chosen)

    @staticmethod
    def _overlay_title(source: Path, destination: Path, title: str, boost: bool) -> None:
        """The old look, kept as the last fallback: the title large across the
        lower third. drawtext needs an explicit fontfile -- the fontconfig
        default crashes on Windows ffmpeg builds with no config."""
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
