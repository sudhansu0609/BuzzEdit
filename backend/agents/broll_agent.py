import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..timeline import Timeline, SourceFile, add_broll_item
from ..comfyui_bridge.workflow_loader import prepare_broll_workflow
from ..comfyui_bridge.queue_manager import ComfyUIQueueManager
from ..utils.ffmpeg_utils import run_ffmpeg

logger = logging.getLogger("broll_agent")

class BRollAgent:
    def __init__(self, queue_manager: Optional[ComfyUIQueueManager] = None):
        self.queue_manager = queue_manager or ComfyUIQueueManager()

    def generate_broll_prompts(self, timeline: Timeline) -> List[Dict[str, Any]]:
        """
        Extract prompt requests from timeline words.
        """
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are"}
        requests = []
        
        # Look for noun phrases/keywords across enabled words
        enabled_words = [w for w in timeline.words if w.enabled]
        for i in range(0, len(enabled_words) - 5, 10):
            chunk = enabled_words[i : i + 5]
            keywords = [w.text.strip(".,!?") for w in chunk if w.text.lower() not in stop_words and len(w.text) > 3]
            if keywords:
                kw_str = " ".join(keywords[:3])
                prompt = f"Cinematic b-roll photo representing {kw_str}, 4k photorealistic, detailed"
                requests.append({
                    "start_frame": chunk[0].start_frame,
                    "end_frame": chunk[-1].end_frame,
                    "duration_frames": chunk[-1].end_frame - chunk[0].start_frame,
                    "prompt": prompt,
                    "keywords": kw_str,
                    "anchor_word_id": chunk[0].id
                })

        return requests

    async def generate_and_apply_broll(
        self,
        timeline: Timeline,
        project_dir: Path,
        max_clips: int = 2
    ) -> int:
        """
        Generate B-roll overlay stills via ComfyUI and insert as V2 items into Timeline.
        """
        broll_dir = project_dir / "broll"
        broll_dir.mkdir(parents=True, exist_ok=True)

        requests = self.generate_broll_prompts(timeline)
        added_count = 0

        for req in requests[:max_clips]:
            prompt = req["prompt"]
            src_id = f"broll_src_{uuid.uuid4().hex[:6]}"
            out_img_path = broll_dir / f"{src_id}.jpg"
            out_video_path = broll_dir / f"{src_id}.mp4"

            try:
                workflow = prepare_broll_workflow(prompt=prompt, output_prefix=f"broll_{src_id}")
                output_files = await self.queue_manager.submit_and_wait(workflow, timeout=60)

                if output_files and Path(output_files[0]).exists():
                    img_path = output_files[0]
                    duration_sec = req["duration_frames"] / (timeline.fps_num / timeline.fps_den)
                    # Convert image to MP4 clip with zoompan Ken Burns effect
                    run_ffmpeg([
                        "-loop", "1",
                        "-i", img_path,
                        "-vf", f"zoompan=z='min(zoom+0.0015,1.15)':d={int(duration_sec * 30)}:s=1920x1080",
                        "-c:v", "libx264",
                        "-t", f"{duration_sec:.3f}",
                        "-pix_fmt", "yuv420p",
                        "-y",
                        str(out_video_path)
                    ])
                else:
                    self._generate_fallback_broll(req["keywords"], str(out_video_path), req["duration_frames"] / 30.0)
            except Exception as e:
                logger.warning(f"ComfyUI offline/failed ({e}). Generating fallback B-roll card.")
                self._generate_fallback_broll(req["keywords"], str(out_video_path), req["duration_frames"] / 30.0)

            if out_video_path.exists():
                duration_sec = req["duration_frames"] / (timeline.fps_num / timeline.fps_den)
                timeline.sources[src_id] = SourceFile(
                    id=src_id,
                    path=str(out_video_path),
                    duration_seconds=duration_sec,
                    width=1920,
                    height=1080,
                    fps_num=timeline.fps_num,
                    fps_den=timeline.fps_den,
                    has_audio=False
                )

                add_broll_item(
                    timeline,
                    broll_source_id=src_id,
                    timeline_start_frame=req["start_frame"],
                    duration_frames=req["duration_frames"],
                    anchor_word_id=req["anchor_word_id"]
                )
                added_count += 1

        return added_count

    def _generate_fallback_broll(self, label: str, output_path: str, duration_sec: float):
        safe_label = label.replace("'", "").replace('"', "")
        run_ffmpeg([
            "-f", "lavfi",
            "-i", f"color=c=0x181825:s=1920x1080:d={duration_sec:.3f}",
            "-vf", f"drawtext=text='[B-ROLL] {safe_label}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            output_path
        ])
