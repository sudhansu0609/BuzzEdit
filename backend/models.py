from pydantic import BaseModel, Field, field_validator
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
    text: str                              # primary display text (Hinglish for Indic speech, else native)
    text_native: Optional[str] = None      # original script (e.g. Devanagari)
    text_english: Optional[str] = None     # English translation
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
    # Modification time of the rendered `output_path`, filled in by the read
    # route. The Rendered preview appends it to the stream URL as a cache-buster:
    # every re-render overwrites the same filename, so without a token that
    # changes the browser keeps serving the previous render and a fresh zoom /
    # B-roll pass looks like it did nothing.
    output_version: Optional[float] = None
    media_pool: List[Dict[str, Any]] = []
    # "presented" is what the overnight presentation pass writes; leaving it out
    # of the Literal made Project.model_validate throw on every presented
    # project, which silently killed the thumbnail stage each night.
    status: Literal["draft", "transcribed", "analyzed", "presented", "rendered",
                    "error"] = "draft"

    @field_validator("transcript", mode="before")
    @classmethod
    def _coerce_transcript(cls, v):
        # The frontend persists `transcript` as a bare list of segments, while
        # the backend writes a full TranscriptionResult ({segments, language,
        # duration}). Accept either so loading a project never 500s.
        if isinstance(v, list):
            return {"segments": v}
        return v

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
    language: Optional[str] = None  # None = auto-detect the spoken language


class AnalyzeJob(BaseModel):
    project_id: str
    remove_silence: bool = True
    remove_fumbles: bool = True


class HealthResponse(BaseModel):
    # The identity block. GUARDIAN_PLAN.md section 11 rule 4: every health route
    # answers who it is, on which port, in which process — so a scan that finds
    # *a* listener can tell whether it found *this* app, and a shell can refuse
    # to publish a port that some other program answered on.
    app: str = "buzzedit"
    status: str = "ok"
    port: int = 0
    pid: int = 0
    version: str = ""

    whisper_available: bool = False
    comfyui_connected: bool = False
    ffmpeg_available: bool = False
