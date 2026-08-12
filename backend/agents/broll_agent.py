import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from models import Project, Clip, ClipType, TranscriptSegment
from comfyui_bridge.workflow_loader import prepare_broll_workflow
from comfyui_bridge.queue_manager import ComfyUIQueueManager
from utils.ffmpeg_utils import run_ffmpeg

logger = logging.getLogger(__name__)


class BRollAgent:
    def __init__(self, queue_manager: Optional[ComfyUIQueueManager] = None):
        self.queue_manager = queue_manager or ComfyUIQueueManager()

    def extract_broll_prompts(self, segments: List[TranscriptSegment]) -> List[Dict[str, Any]]:
        broll_requests = []
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        for seg in segments:
            duration = seg.end - seg.start
            if duration < 2.0:
                continue

            text = seg.text.strip()
            words = [w.strip(".,!?") for w in text.split() if w.lower() not in stop_words and len(w) > 3]

            if words:
                keywords = " ".join(words[:4])
                prompt = f"Cinematic b-roll footage representing: {keywords}, 4k ultra detailed, photorealistic"
                broll_requests.append({
                    "start": seg.start,
                    "end": seg.end,
                    "duration": duration,
                    "prompt": prompt,
                    "keywords": keywords,
                })

        return broll_requests

    async def generate_broll_for_project(self, project: Project, project_dir: Path) -> List[Dict[str, Any]]:
        if not project.transcript or not project.transcript.segments:
            return []

        broll_dir = project_dir / "broll"
        broll_dir.mkdir(parents=True, exist_ok=True)

        requests = self.extract_broll_prompts(project.transcript.segments)
        generated_clips = []

        for req in requests[:3]: # Limit to top 3 B-roll clips per project for performance
            prompt = req["prompt"]
            out_filename = f"broll_{int(req['start'])}.mp4"
            output_path = broll_dir / out_filename

            try:
                # Try ComfyUI generation
                workflow = prepare_broll_workflow(prompt=prompt, output_prefix=f"broll_{project.id}_{int(req['start'])}")
                output_files = await self.queue_manager.submit_and_wait(workflow, timeout=120)

                if output_files and Path(output_files[0]).exists():
                    img_path = output_files[0]
                    # Convert generated image to video clip of requested duration
                    run_ffmpeg([
                        "-loop", "1",
                        "-i", img_path,
                        "-c:v", "libx264",
                        "-t", str(req["duration"]),
                        "-pix_fmt", "yuv420p",
                        "-vf", "scale=1280:720",
                        "-y",
                        str(output_path),
                    ])
                else:
                    self._generate_fallback_broll(req["keywords"], str(output_path), req["duration"])
            except Exception as e:
                logger.warning(f"ComfyUI B-Roll generation offline/failed ({e}). Using fallback generator.")
                self._generate_fallback_broll(req["keywords"], str(output_path), req["duration"])

            if output_path.exists():
                generated_clips.append({
                    "start": req["start"],
                    "end": req["end"],
                    "duration": req["duration"],
                    "file_path": str(output_path),
                    "prompt": prompt,
                })

        return generated_clips

    def _generate_fallback_broll(self, label: str, output_path: str, duration: float):
        # Generate stylized color title card fallback clip
        safe_label = label.replace("'", "").replace('"', "")
        run_ffmpeg([
            "-f", "lavfi",
            "-i", f"color=c=0x1e1e2e:s=1280x720:d={duration}",
            "-vf", f"drawtext=text='[B-ROLL] {safe_label}':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            output_path
        ])
