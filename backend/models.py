from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class ClipType(str, Enum):
    SPEECH = "speech"
    SILENCE = "silence"
    FUMBLE = "fumble"
    BROLL = "broll"
    INTRO = "intro"
    OUTRO = "outro"
    TRANSITION = "transition"


class TransitionType(str, Enum):
    CROSSFADE = "crossfade"
    ZOOM = "zoom"
    DISSOLVE = "dissolve"
    NONE = "none"


class Clip(BaseModel):
    id: str
    start_time: float
    end_time: float
    clip_type: ClipType = ClipType.SPEECH
    file_path: Optional[str] = None
    label: Optional[str] = None
    transcript: Optional[str] = None
    confidence: Optional[float] = None
    transition: Optional[TransitionType] = None
    transition_duration: float = 0.5
    metadata: Dict[str, Any] = {}


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0


class TranscriptionResult(BaseModel):
    segments: List[TranscriptSegment]
    language: str = "en"
    duration: float = 0.0


class DetectedSegment(BaseModel):
    start: float
    end: float
    segment_type: ClipType
    confidence: float = 0.0
    label: Optional[str] = None


class Project(BaseModel):
    id: str
    name: str
    source_video: str
    clips: List[Clip] = []
    transcript: Optional[TranscriptionResult] = None
    detected_segments: List[DetectedSegment] = []
    output_path: Optional[str] = None
    status: Literal["draft", "transcribed", "analyzed", "rendered", "error"] = "draft"
    settings: Dict[str, Any] = {
        "resolution": "1920x1080",
        "fps": 30,
        "audio_sample_rate": 48000,
        "remove_silence": True,
        "remove_fumbles": True,
        "auto_transitions": True,
        "transition_type": "crossfade",
        "transition_duration": 0.5,
    }


class Job(BaseModel):
    id: str
    project_id: str
    job_type: Literal["transcribe", "analyze", "render", "broll", "tts", "comfyui"]
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class RenderJob(BaseModel):
    project_id: str
    output_path: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class TranscribeJob(BaseModel):
    project_id: str
    model: str = "large-v3"
    language: str = "en"


class AnalyzeJob(BaseModel):
    project_id: str
    remove_silence: bool = True
    remove_fumbles: bool = True


class HealthResponse(BaseModel):
    status: str = "ok"
    whisper_available: bool = False
    comfyui_connected: bool = False
    ffmpeg_available: bool = False
