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
    # When the presentation pass saved the timeline (just before rendering).
    # BuzzcafStudio compares it with a failed job's start to re-render the
    # saved edit instead of running the whole pass again.
    presented_at: Optional[str] = None

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
    # The recording is already cut: transcribe it, but keep every frame (see
    # Timeline.keep_full_source). Stored on the project's settings so later
    # passes (the presentation pass's own auto-edit) honour it too.
    precut: Optional[bool] = None
    # Auto-edit pacing, per channel (the Studio sends its channel profile's
    # `auto_edit` block). Measured from the creator's own reference cut of
    # Life3Baje ep1: pauses up to ~1.1s are left in, ~0.35s of room stays
    # either side of a cut. Stored on project settings like `precut`, so the
    # timeline rebuild and later passes keep the same pacing. None = keep
    # whatever the project already has (or the Timeline defaults).
    max_pause_seconds: Optional[float] = None
    pause_padding_seconds: Optional[float] = None
    fumble_aggressiveness: Optional[float] = None
    # What the recording is (the Studio sends its channel profile's genre). When
    # no pacing was chosen, the genre's own applies -- horror keeps its pauses
    # (presentation.genre.pacing_for). None = the model reads the transcript and
    # decides. Stored on project settings like `precut`.
    genre: Optional[str] = None
    # Latin words from the user's own script (proper nouns, brand names,
    # loanwords) -- fed to the Hinglish romanizer as a first-preference
    # source for restoring English words Whisper wrote out in Devanagari,
    # and to Whisper itself as `initial_prompt`. Stored on project settings
    # like `language`, so a later respell reuses it without the caller
    # having to resend it.
    vocabulary: Optional[List[str]] = None
    # Channel key selecting a glossary file (data/glossary/<key>.json) whose
    # entries override both the common-word table and the loanword restore.
    # Stored on project settings like `language`.
    glossary: Optional[str] = None
    # Which auto-cut planner: "editor" (the AI editor, asr.editor_planner) or
    # "classic" (fumble_engine). None = the project's or app's choice, else the
    # editor (asr.auto_edit.choose_planner). Stored on project settings.
    planner: Optional[Literal["editor", "classic"]] = None


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
