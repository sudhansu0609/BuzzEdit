from typing import List, Dict, Optional
from pydantic import BaseModel, Field

def time_to_frame(seconds: float, fps_num: int = 30, fps_den: int = 1) -> int:
    fps = fps_num / fps_den
    return max(0, int(round(seconds * fps)))

def frame_to_time(frame: int, fps_num: int = 30, fps_den: int = 1) -> float:
    fps = fps_num / fps_den
    return frame / fps

class SourceFile(BaseModel):
    id: str
    path: str
    duration_seconds: float
    width: int = 1920
    height: int = 1080
    fps_num: int = 30
    fps_den: int = 1
    has_audio: bool = True

class WordItem(BaseModel):
    id: str
    text: str
    start_frame: int
    end_frame: int
    enabled: bool = True
    disfluency: bool = False
    speaker: Optional[str] = None

class TimelineEffect(BaseModel):
    type: str  # "zoompan", "fade", "lut"
    params: Dict[str, float | str] = Field(default_factory=dict)

class TimelineItem(BaseModel):
    id: str
    track: str  # "V1", "A1", "V2", "CAP"
    source_id: str
    source_start_frame: int
    source_end_frame: int
    timeline_start_frame: int
    timeline_end_frame: int
    enabled: bool = True
    effects: List[TimelineEffect] = Field(default_factory=list)
    anchor_word_id: Optional[str] = None

class Timeline(BaseModel):
    version: str = "1.0"
    fps_num: int = 30
    fps_den: int = 1
    duration_frames: int = 0
    sources: Dict[str, SourceFile] = Field(default_factory=dict)
    words: List[WordItem] = Field(default_factory=list)
    items: List[TimelineItem] = Field(default_factory=list)
    revision: int = 1

    def recalculate_duration(self) -> None:
        max_frame = 0
        for item in self.items:
            if item.enabled:
                max_frame = max(max_frame, item.timeline_end_frame)
        self.duration_frames = max_frame
