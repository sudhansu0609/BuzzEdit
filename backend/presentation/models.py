"""Data contracts for the presentation pass.

Every boundary in this pipeline is a validated model rather than a loose dict,
for one reason: the middle of it is a language model. A beat that reaches
placement carries times that have already been clamped to the programme, a kind
that is known, and the fields that kind requires — so the placement code can be
written as if the plan were trustworthy, because by then it is.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

BeatKind = Literal[
    # generated pictures (full-frame cutaways)
    "broll_image", "broll_video",
    # locally rendered pictures (full-frame cutaways, no ComfyUI)
    "map", "chart",
    # two generated stills side by side ("A vs B"); expanded into a pair of
    # broll_image beats before generation
    "split",
    # text over the picture
    "popup", "graphic", "location_card", "character_card", "source_card",
    "chapter_title", "stat_callout", "definition_card", "quote_card", "end_screen",
]
BEAT_KINDS: Tuple[str, ...] = (
    "broll_image", "broll_video", "map", "chart", "split", "popup", "graphic",
    "location_card", "character_card", "source_card", "chapter_title",
    "stat_callout", "definition_card", "quote_card", "end_screen",
)
CUTAWAY_KINDS: Tuple[str, ...] = ("broll_image", "broll_video", "map", "chart", "split")
TEXT_KINDS: Tuple[str, ...] = (
    "location_card", "character_card", "source_card", "chapter_title",
    "stat_callout", "definition_card", "quote_card", "end_screen",
)
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
    # The paragraph's heading when the topic came from the user's script.
    heading: Optional[str] = None
    # Filled by later planners: the emotional register ("tense", "reveal",
    # …) and the act of the story ("setup", "climax", …).
    mood: Optional[str] = None
    act: Optional[str] = None
    # The strongest instant inside the topic (the reveal, the scare) — where
    # the mood recipe puts its hit. None means "the loudest word".
    key_moment_s: Optional[float] = None
    origin: str = "llm"


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
    # Text kinds: the words on the card, and a smaller second line (a
    # character's role, a source's year, a stat's label).
    text: Optional[str] = None
    subtext: Optional[str] = None
    # Map beats: the place to draw. Chart beats: {"labels": [...], "values":
    # [...], "unit": "%"}. Stat call-outs: {"value": 25, "suffix": "%"}.
    place: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    # An effect to fire when the beat lands (a stinger on a reveal).
    sfx: Optional[str] = None
    # Where the beat came from: "llm", "fallback", "script" (a stage direction
    # the user wrote), "entity" (extracted from the transcript), "structure".
    origin: str = "llm"

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def is_cutaway(self) -> bool:
        return self.kind in CUTAWAY_KINDS

    @property
    def is_generated(self) -> bool:
        """Needs ComfyUI (as opposed to a map or chart drawn locally)."""
        return self.kind in ("broll_image", "broll_video")

    @property
    def is_text(self) -> bool:
        return self.kind in TEXT_KINDS

    def prompt(self) -> Optional[str]:
        return self.video_prompt if self.kind == "broll_video" else self.image_prompt


class ShotPlan(BaseModel):
    beats: List[Beat] = Field(default_factory=list)
    dropped: List[Dict[str, str]] = Field(default_factory=list)
    source: str = "llm"          # "llm" | "fallback"
    # The kind of video the prompts were styled for ("horror", "comedy", ...);
    # "general" means no styling was applied.
    genre: str = "general"
    # The topics the beats came from. Later stages (transitions at the chapter
    # boundaries, mood recipes, chapter titles) work per topic, not per beat.
    topics: List[Topic] = Field(default_factory=list)


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


# --- Stage F: sound --------------------------------------------------------

class MusicCue(BaseModel):
    path: str
    start_s: float
    end_s: float
    gain: float = 0.16
    duck: float = 0.85
    fade_in_s: float = 1.5
    fade_out_s: float = 3.0
    synthesised: bool = False
    act: str = ""


class SfxCue(BaseModel):
    tag: str
    at_s: float
    gain: float = 0.5


class AmbienceCue(BaseModel):
    kind: str
    path: str
    gain: float = 0.05
    synthesised: bool = False


class LoopCue(BaseModel):
    """A looped effect under a stretch (a heartbeat through the climax)."""
    tag: str
    start_s: float
    end_s: float
    gain: float = 0.35


class SoundPlan(BaseModel):
    music: List[MusicCue] = Field(default_factory=list)
    sfx: List[SfxCue] = Field(default_factory=list)
    loops: List[LoopCue] = Field(default_factory=list)
    ambience: List[AmbienceCue] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


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
    caption_preset: str = "karaoke_pop"
    # An animated title over the opening footage (the thumbnail title doubles as
    # the video title). Overlay presets draw over the footage; card presets
    # (title_card, bold_slam, …) push the programme back behind a card.
    title: bool = True
    title_preset: str = "overlay_hook"
    # A subtle full-length atmosphere layer picked by genre (film grain for a
    # documentary look, fog for horror, …). "auto" maps from the genre; a preset
    # name from EFFECT_PRESETS forces one; "" / "off" disables it.
    atmosphere: str = "auto"
    thumbnail: bool = True
    render: bool = True
    # Check the dressed timeline (text off the face, cards not stacked, bed
    # ducked) and the render (loudness, A/V length) and report failures.
    verify: bool = True
    # --- distribution ---
    # The best topics cut as 9:16 Shorts (0 = none), reframed around the
    # speaker's face when the source is landscape; and, if asked, the whole
    # programme as a vertical file.
    shorts_clips: int = 3
    shorts_full: bool = False
    # --- text cards from what the speaker names ---
    # One extraction pass per topic feeds all of these. Each switch is a kind
    # of card; `cards` turns the whole family off.
    cards: bool = True
    location_cards: bool = True      # "Rajasthan, 1987" top-left when a place is named
    character_cards: bool = True     # name + role lower third on a first mention
    stat_callouts: bool = True       # a number counting up when a statistic is spoken
    source_cards: bool = True        # "Source: …" when a study or book is cited
    definition_cards: bool = True    # term + gloss when the speaker defines something
    quote_cards: bool = True         # attributed quotes
    chapter_titles: bool = True      # each topic's name slams in at its start
    end_screen: bool = True          # subscribe call-out over the last seconds
    end_screen_text: str = "SUBSCRIBE"
    # A map cutaway for the first place named in a topic (drawn offline).
    maps: bool = True
    # "A versus B" as two stills side by side when the speaker compares things.
    split_screens: bool = True
    map_style: str = "auto"          # "auto" (by genre) | "clean" | "noir"
    # --- structure ---
    # Tag the topics with story acts (the density of pictures follows them),
    # open on the most gripping sentence, and write the YouTube listing.
    structure: bool = True
    cold_open: bool = True
    metadata: bool = True
    # --- moods ---
    # Each topic is tagged with its register (calm, build, tense, reveal,
    # climax, aftermath, comedic, hopeful) and gets a recipe: a grade shift, a
    # slow push, an atmosphere window, and at the key moment a hit — flash,
    # shake, thunder — at full strength in horror and true crime, subtle
    # elsewhere. `mood_strength` scales all of it; 0 turns the moods off.
    moods: bool = True
    mood_strength: float = 1.0
    # --- look ---
    # Programme-wide grade: "auto" picks one for the genre (moody for horror,
    # punchy for an explainer), a COLOR_PRESETS name forces one, "off" skips it.
    grade: str = "auto"
    # A transition into the first clip of every topic, where the boundary lands
    # on a cut. "auto" picks the genre's (dip to black for horror, a whip for
    # explainers); an xfade name forces one.
    topic_transitions: bool = True
    topic_transition: str = "auto"
    # --- sound ---
    # A music bed under the whole programme, ducked under speech. Comes from
    # data/music/<genre>/ when the user has put music there; horror and true
    # crime fall back to a synthesised drone when the library is empty.
    music: bool = True
    music_volume: float = 0.16          # linear gain relative to the voice (~-16 dB)
    music_duck: float = 0.85            # 0..1, how hard speech pushes the bed down
    music_synth_fallback: bool = True
    # A different cue per story act (a calm bed for the setup, a tense one for
    # the build, the drone rising at the climax), crossfaded at the boundary.
    music_by_act: bool = True
    # --- captions ---
    # Numbers and stressed words drawn larger in the accent colour (karaoke
    # captions only), and an English line under Indic captions when a model
    # can translate ("auto"), always ("on"), or never ("off").
    caption_emphasis: bool = True
    captions_bilingual: str = "auto"
    # --- composites ---
    # The speaker in a corner over cutaways longer than `pip_min_seconds`:
    # "auto" (explainer genres only), "on", "off".
    pip: str = "auto"
    pip_min_seconds: float = 4.0
    # Freeze the speaker for a moment under each stat call-out.
    freeze_on_stats: bool = True
    # "Edit like this channel": the id of a reference-video style profile
    # (data/styles/, made by /api/style/analyze) applied after the genre look.
    style_profile: Optional[str] = None
    # Whooshes on cutaways, pops on text, stingers/thunder from the mood recipes.
    sfx: bool = True
    sfx_volume: float = 0.45
    # A full-length ambience loop (wind for horror, room tone for true crime).
    # "auto" maps from the genre; a kind name forces one.
    ambience: bool = True
    ambience_kind: str = "auto"
    ambience_volume: float = 0.05
    # Voice clean-up and the loudness target. "off" leaves the audio untouched.
    voice_preset: str = "clean"
    loudness_lufs: Optional[float] = -14.0
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
    # What fraction of B-roll screen time is generated VIDEO rather than stills.
    # Video costs minutes per clip where a still costs seconds, so it goes to
    # the highest-priority beats — the parts of the video that matter most.
    # 0.18 targets the asked-for 15-20%; 0 turns video promotion off.
    video_broll_share: float = 0.18
    # Optional cap on cutaway count. None means "derived from coverage alone";
    # kept because the API accepted it since the first release.
    max_broll_per_minute: Optional[float] = None
    popup_preset: str = "youtube_pop"
    zoom_depth: float = 0.10
    seed: Optional[int] = None
    # The video's category ("horror", "comedy", "finance", ...). Styles every
    # generated prompt — B-roll and thumbnail — to match the kind of video.
    # None means detect it from the transcript; unknown names style nothing.
    genre: Optional[str] = None

    @field_validator("target_coverage")
    @classmethod
    def _clamp_coverage(cls, value: float) -> float:
        # Hard ceiling: above 0.8 the result is a slideshow with narration, and
        # no slider or API caller may push it there.
        return max(0.0, min(0.80, float(value)))

    @field_validator("video_broll_share")
    @classmethod
    def _clamp_video_share(cls, value: float) -> float:
        # Above half the B-roll being generated video, an overnight pass turns
        # into a multi-night one; the cap keeps a typo from costing a day.
        return max(0.0, min(0.5, float(value)))

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
    # What the prompts were styled as — set from the transcript when the user
    # did not choose, so a wrong detection is visible in the morning report.
    genre: str = "general"
    assets_generated: int = 0
    assets_cached: int = 0
    assets_failed: List[Dict[str, str]] = Field(default_factory=list)
    broll_placed: int = 0
    popups_placed: int = 0
    graphics_placed: int = 0
    # Text cards placed, by kind ({"stat_callout": 3, "chapter_title": 4, …}),
    # and how many entities the extraction found.
    cards_placed: Dict[str, int] = Field(default_factory=dict)
    entities_found: int = 0
    maps_placed: int = 0
    splits_placed: int = 0
    pip_placed: int = 0
    freezes_placed: int = 0
    caption_emphasis: int = 0
    caption_translations: int = 0
    style_profile_applied: str = ""
    # Beats the planner budgeted but placement could not fit — silent drops here
    # are exactly how "70% coverage" quietly became 20%.
    beats_placement_rejected: List[Dict[str, str]] = Field(default_factory=list)
    coverage_target: float = 0.0
    coverage_achieved: float = 0.0
    # The edit's jump cuts (V1 joins) and how many a cutaway hides. A join under
    # B-roll is invisible; these two numbers are how the morning report says how
    # much of the edit's choppiness the pictures absorbed.
    jump_cuts_total: int = 0
    jump_cuts_covered: int = 0
    comfyui_online: Optional[bool] = None
    zooms_segment: int = 0
    zooms_windowed: int = 0
    zooms_suppressed_by_broll: int = 0
    faces_detected_pct: float = 0.0
    captions: int = 0
    # The script the captions were written in ("romanized" / "native").
    caption_script: str = ""
    # The opening title that was drawn, if any, and the atmosphere effect applied.
    title_text: str = ""
    atmosphere_applied: str = ""
    grade_applied: str = ""
    topic_transitions: int = 0
    # Structure: the acts, the cold open that was cut (or why not), the listing.
    acts: List[str] = Field(default_factory=list)
    cold_open: Optional[Dict[str, Any]] = None
    metadata_written: bool = False
    # Moods: what each topic was tagged, how many recipe layers and hits landed.
    moods: List[str] = Field(default_factory=list)
    mood_layers: int = 0
    mood_hits: int = 0
    # The user's script, when one was aligned: how much of it matched the
    # transcript, how many caption spellings it corrected, and how many stage
    # directions it carried into the plan.
    script_aligned_words: int = 0
    script_spelling_fixed: int = 0
    script_directives: int = 0
    # Sound: the bed that was used (file name, or "synth:drone"), how many
    # effects and ambience loops landed, and the voice preset applied.
    music_used: str = ""
    sfx_placed: int = 0
    ambience_used: str = ""
    voice_preset: str = ""
    thumbnail: Optional[str] = None
    output_path: Optional[str] = None
    # The checks run on the dressed timeline and the rendered file
    # ({"name", "ok", "detail", "value"}); failures are also listed in `degraded`.
    verification: List[Dict[str, Any]] = Field(default_factory=list)
    # Vertical outputs: [{"kind": "short"|"vertical", "path", "topic", "seconds"}].
    shorts: List[Dict[str, Any]] = Field(default_factory=list)
    degraded: List[str] = Field(default_factory=list)
    timings: List[StageTiming] = Field(default_factory=list)
