import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils.ffmpeg_utils import run_ffprobe, extract_frames, FFmpegError

logger = logging.getLogger(__name__)


class SceneAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.scenes: List[Dict[str, Any]] = []

    def detect_scenes(self, threshold: float = 30.0) -> List[Dict[str, Any]]:
        try:
            from scene_detect import detect_scenes_opencv
            self.scenes = detect_scenes_opencv(self.video_path, threshold)
        except ImportError:
            logger.warning("PySceneDetect not available, using frame diff method")
            self.scenes = self._detect_scenes_frame_diff()

        logger.info(f"Detected {len(self.scenes)} scenes in {self.video_path}")
        return self.scenes

    def _detect_scenes_frame_diff(self, sample_fps: float = 0.5) -> List[Dict[str, Any]]:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        step = max(1, int(fps / sample_fps))
        scenes = []
        prev_frame = None
        scene_start = 0.0
        threshold = 25.0

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))

            if prev_frame is not None:
                diff = np.mean(np.abs(small.astype(float) - prev_frame.astype(float)))
                timestamp = i / fps

                if diff > threshold:
                    scenes.append({
                        "start": round(scene_start, 2),
                        "end": round(timestamp, 2),
                        "duration": round(timestamp - scene_start, 2),
                        "change_score": round(float(diff), 2),
                    })
                    scene_start = timestamp

            prev_frame = small

        if scene_start < duration:
            scenes.append({
                "start": round(scene_start, 2),
                "end": round(duration, 2),
                "duration": round(duration - scene_start, 2),
                "change_score": 0.0,
            })

        cap.release()
        return scenes

    def get_keyframes(self, scenes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        if scenes is None:
            scenes = self.scenes

        keyframes = []
        for scene in scenes:
            mid_point = (scene["start"] + scene["end"]) / 2
            keyframes.append({
                "scene_index": len(keyframes),
                "timestamp": round(mid_point, 2),
                "scene_start": scene["start"],
                "scene_end": scene["end"],
            })

        return keyframes
