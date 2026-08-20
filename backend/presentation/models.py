"""Data contracts for the presentation pass.

Every boundary in this pipeline is a validated model rather than a loose dict,
for one reason: the middle of it is a language model. A beat that reaches
placement carries times that have already been clamped to the programme, a kind
that is known, and the fields that kind requires — so the placement code can be
written as if the plan were trustworthy, because by then it is.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

BeatKind = Literal["broll_image", "broll_video", "popup", "graphic"]
StyleHint = Literal["photoreal", "illustration", "diagram", "abstract"]


# --- Stage A: the programme ------------------------------------------------

class ProgramWord(BaseModel):
    """One surviving word, positioned in the CUT video rather than the source."""
    text: str
    tl_start_s: float
    tl_end_s: float
    source_start_frame: int
    db_mean: float = 0.0        # mean loudness over the word
    emphasis_z: float = 0.0     # how loud, relative to the rest of the programme
    rate_wps: float = 0.0       # local speaking rate


class ProgramSegment(BaseModel):
    """One V1 clip of the finished programme."""
    item_id: str
    tl_start_s: float
    tl_end_s: float
    source_start_frame: int
    source_end_frame: int
    anchor_word_id: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.tl_end_s - self.tl_start_s)


class Program(BaseModel):
    """The finished cut, as something the director can reason about."""
    duration_s: float
    fps: float
    words: List[ProgramWord] = Field(default_factory=list)
    segments: List[ProgramSegment] = Field(default_factory=list)
    has_energy: bool = False

    def text_between(self, start_s: float, end_s: float) -> str:
        return " ".join(w.text for w in self.words
                        if w.tl_start_s < end_s and w.tl_end_s > start_s)

    def words_between(self, start_s: float, end_s: float) -> List[ProgramWord]:
        return [w for w in self.words
                if w.tl_start_s < end_s and w.tl_end_s > start_s]

    def has_speech_between(self, start_s: float, end_s: float) -> bool:
        return any(True for _ in self.words_between(start_s, end_s))

    def segment_at(self, time_s: float) -> Optional[ProgramSegment]:
        for segment in self.segments:
            if segment.tl_start_s <= time_s < segment.tl_end_s:
                return segment
        return None


# --- Stage B: the plan -----------------------------------------------------

class Topic(BaseModel):
    """One subject the speaker covers, as the model heard it."""
    start_s: float
    end_s: float
    topic: str
    summary: str = ""
    visual: str = ""
    priority: float = 0.5


class Beat(BaseModel):
    """One thing that will happen on screen."""
    id: str = ""
    start_s: float
    end_s: float
    topic: str = ""
    summary: str = ""
    kind: BeatKind = "broll_image"
    priority: float = 0.5
    image_prompt: Optional[str] = None
    video_prompt: Optional[str] = None
    popup_text: Optional[str] = None
    negative_prompt: str = "text, watermark, logo, deformed hands, blurry"
    style_hint: StyleHint = "photoreal"
    # How long the cutaway for this beat should actually run on screen, decided
    # by the seconds budget at validation time. None on older stored plans —
    # placement then falls back to the settings' clamp.
    planned_duration_s: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def is_cutaway(self) -> bool:
        return self.kind in ("broll_image", "broll_video")

    def prompt(self) -> Optional[str]:
        return self.video_prompt if self.kind == "broll_video" else self.image_prompt


class ShotPlan(BaseModel):
    beats: List[Beat] = Field(default_factory=list)
    dropped: List[Dict[str, str]] = Field(default_factory=list)
    source: str = "llm"          # "llm" | "fallback"


# --- Stage C: generated assets ---------------------------------------------

class Asset(BaseModel):
    beat_id: str
    kind: Literal["image", "video"]
    path: str
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    cache_hit: bool = False
    workflow_file: str = ""
    seed: int = 0
    prompt_sha: str = ""


# --- Stage E: zooms --------------------------------------------------------

class SegmentZoom(BaseModel):
    """A move that runs the whole length of one programme clip."""
    item_id: str
    push_in: bool
    depth: float
    anchor_x: float = 0.0
    anchor_y: float = 0.0


class WindowZoom(BaseModel):
    """A punch-in that starts and ends inside a clip, via an adjustment layer."""
    start_s: float
    end_s: float
    depth: float
    anchor_x: float = 0.0
    anchor_y: float = 0.0
    reason: str = "emphasis"


# --- Settings and reporting ------------------------------------------------

class PresentationSettings(BaseModel):
    broll: bool = True
    broll_video: bool = True
    popups: bool = True
    # Off by default: an animated zoom on a transparent PNG composites a black
    # box, because the canvas transform pads with black. See placement.py.
    graphics: bool = False
    face_zoom: bool = True
    captions: bool = True
    caption_preset: str = "youtube_shorts"
    thumbnail: bool = True
    render: bool = True
    # The one knob that drives B-roll density: what fraction of the programme is
    # covered by cutaways. Gap and budget are derived from it — the old fixed
    # per-minute count and 8s gaps capped coverage near 20% however this was set.
    target_coverage: float = 0.75
    # The speaker must reappear at least this long between cutaways, whatever
    # the coverage asks for — an unbroken picture wall stops being their video.
    # Kept below the gap the coverage target derives (~1s at 75%) so the target,
    # not this floor, governs: at 2.5s it silently capped a 75%-coverage plan
    # near 55% and spaced the cutaways so far apart that only a handful of long
    # stills were ever generated.
    min_oncamera_gap_s: float = 0.8
    # Short cutaways, many of them: a ~2.5s average fills the coverage budget
    # with several distinct images instead of one long still, which is what "70%
    # of a 10s clip should be B-roll, and generate many more images" asks for.
    # Each still gets a Ken Burns move so a 2-3s hold does not read as a freeze.
    broll_seconds_min: float = 2.0
    broll_seconds_max: float = 3.0
    # Optional cap on cutaway count. None means "derived from coverage alone";
    # kept because the API accepted it since the first release.
    max_broll_per_minute: Optional[float] = None
    popup_preset: str = "youtube_pop"
    zoom_depth: float = 0.10
    seed: Optional[int] = None

    @field_validator("target_coverage")
    @classmethod
    def _clamp_coverage(cls, value: float) -> float:
        # Hard ceiling: above 0.8 the result is a slideshow with narration, and
        # no slider or API caller may push it there.
        return max(0.0, min(0.80, float(value)))

    @property
    def avg_broll_seconds(self) -> float:
        return max(1.0, (self.broll_seconds_min + self.broll_seconds_max) / 2.0)

    @property
    def cutaway_gap_s(self) -> float:
        """On-camera gap between cutaways, derived from the coverage target.

        At coverage c with average shot length d, the on-camera share per shot
        is d*(1-c)/c; the floor keeps the speaker a presence at any coverage.
        """
        if self.target_coverage <= 0.0:
            return self.min_oncamera_gap_s
        derived = self.avg_broll_seconds * (1.0 - self.target_coverage) / self.target_coverage
        return max(self.min_oncamera_gap_s, derived)

    def broll_budget_seconds(self, duration_s: float) -> float:
        """Total screen seconds cutaways may occupy on a programme this long."""
        budget = self.target_coverage * max(0.0, duration_s)
        if self.max_broll_per_minute:
            minutes = max(1.0 / 60.0, duration_s / 60.0)
            budget = min(budget,
                         self.max_broll_per_minute * minutes * self.broll_seconds_max)
        return budget


class StageTiming(BaseModel):
    stage: str
    seconds: float
    ok: bool = True
    note: str = ""


class PresentationReport(BaseModel):
    project_id: str
    started_at: str = ""
    finished_at: str = ""
    settings: Dict[str, Any] = Field(default_factory=dict)
    program: Dict[str, Any] = Field(default_factory=dict)
    beats_planned: int = 0
    beats_dropped: List[Dict[str, str]] = Field(default_factory=list)
    plan_source: str = ""
    assets_generated: int = 0
    assets_cached: int = 0
    assets_failed: List[Dict[str, str]] = Field(default_factory=list)
    broll_placed: int = 0
    popups_placed: int = 0
    graphics_placed: int = 0
    # Beats the planner budgeted but placement could not fit — silent drops here
    # are exactly how "70% coverage" quietly became 20%.
    beats_placement_rejected: List[Dict[str, str]] = Field(default_factory=list)
    coverage_target: float = 0.0
    coverage_achieved: float = 0.0
    comfyui_online: Optional[bool] = None
    zooms_segment: int = 0
    zooms_windowed: int = 0
    zooms_suppressed_by_broll: int = 0
    faces_detected_pct: float = 0.0
    captions: int = 0
    thumbnail: Optional[str] = None
    output_path: Optional[str] = None
    degraded: List[str] = Field(default_factory=list)
    timings: List[StageTiming] = Field(default_factory=list)
