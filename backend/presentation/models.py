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
    "map", "chart", "newspaper", "case_file",
    # two generated stills side by side ("A vs B"); expanded into a pair of
    # broll_image beats before generation
    "split",
    # text over the picture
    "popup", "graphic", "location_card", "character_card", "source_card",
    "chapter_title", "stat_callout", "definition_card", "quote_card", "end_screen",
    # designed graphics (presentation/graphics.py): full-frame layouts...
    "statement_card", "canvas_card", "source_quote", "document", "timeline",
    # ...and overlays on the speaker
    "pill_labels", "numbered_point", "name_title",
]
BEAT_KINDS: Tuple[str, ...] = (
    "broll_image", "broll_video", "map", "chart", "newspaper", "case_file", "split",
    "popup", "graphic", "location_card", "character_card", "source_card", "chapter_title",
    "stat_callout", "definition_card", "quote_card", "end_screen",
    "statement_card", "canvas_card", "source_quote", "document", "timeline",
    "pill_labels", "numbered_point", "name_title",
)
CUTAWAY_KINDS: Tuple[str, ...] = (
    "broll_image", "broll_video", "map", "chart", "newspaper", "case_file", "split",
    "statement_card", "canvas_card", "source_quote", "document", "timeline",
)
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
    # The story's main character (ShotPlan.character) is the visible subject:
    # the still gets that character's face swapped in, and a clip made from it
    # keeps the face. False for anyone else, crowds, places and objects.
    shows_character: bool = False
    # "in" / "out" when a gradual genre took a zoom out of this clip's prompt
    # (presentation.camera): the clip is generated locked off and placement
    # puts the move back as a slow timeline zoom. None = the clip moves itself.
    camera_move: Optional[str] = None

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


class Character(BaseModel):
    """The one person a story follows, so every shot of them has one face."""
    name: str
    # What they look like, written for an image model: age, build, face, hair,
    # clothing. Added to every prompt of a beat that shows them.
    description: str
    # "male" | "female" | "" -- when known, the swap only replaces a face of
    # that gender, so the person beside the lead never gets the lead's face.
    gender: str = ""

    def portrait_prompt(self) -> str:
        """The reference face: frontal, even light, nothing in the way."""
        return (f"Close-up portrait photograph of {self.description}, facing the camera, "
                "neutral expression, soft even daylight, plain grey background, sharp "
                "focus on the face, realistic skin texture")


class ShotPlan(BaseModel):
    beats: List[Beat] = Field(default_factory=list)
    dropped: List[Dict[str, str]] = Field(default_factory=list)
    source: str = "llm"          # "llm" | "fallback"
    # The kind of video the prompts were styled for ("horror", "comedy", ...);
    # "general" means no styling was applied.
    genre: str = "general"
    # The blended-in second genre, when settings.genre_secondary named one
    # distinct from the primary; "" otherwise.
    genre_secondary: str = ""
    # The topics the beats came from. Later stages (transitions at the chapter
    # boundaries, mood recipes, chapter titles) work per topic, not per beat.
    topics: List[Topic] = Field(default_factory=list)
    # The story's main character, when it has one; beats with shows_character
    # get this face. None for explainers, lists and anything without one lead.
    character: Optional[Character] = None


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
    # Where this came from: "generated" (the default, via ComfyUI) or the free
    # stock source that filled a beat ComfyUI could not — see
    # presentation.stock. The four fields below are only set for stock assets,
    # so a report can show provenance without touching placement.
    source: str = "generated"
    stock_url: Optional[str] = None
    stock_author: Optional[str] = None
    stock_licence: Optional[str] = None


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
    # True: a punch-in (wide -> tight on the anchor). False: a pull-back (tight
    # on the anchor -> wide) -- see facezoom._plan_punches, which alternates
    # the two so consecutive punches do not all read the same way.
    push_in: bool = True


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
    # Where this cue's file came from ("library" / "generated" / "synth"), for
    # the report's `music_source_used`.
    source: str = "library"


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


class VoiceFxCue(BaseModel):
    """One line-level voice effect: "phone", "megaphone", "echo_tail",
    "reverb_room" or "reverb_hall", applied only inside [start_s, end_s]."""
    start_s: float
    end_s: float
    effect: str


class SoundPlan(BaseModel):
    music: List[MusicCue] = Field(default_factory=list)
    sfx: List[SfxCue] = Field(default_factory=list)
    loops: List[LoopCue] = Field(default_factory=list)
    ambience: List[AmbienceCue] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# --- Settings and reporting ------------------------------------------------

# One Studio-facing "density" knob standing in for several numeric settings —
# see PresentationSettings.resolve_density(). Values match the CONTRACT table
# BuzzcafStudio and BuzzEdit agreed on.
# Bounds on one B-roll cutaway's screen time (and so on a generated clip).
BROLL_SECONDS_FLOOR, BROLL_SECONDS_CAP = 1.0, 8.0

DENSITY_PRESETS: Dict[str, Dict[str, float]] = {
    "calm":     {"target_coverage": 0.30, "video_broll_share": 0.10,
                "broll_seconds_min": 3.0, "broll_seconds_max": 5.0,
                "min_oncamera_gap_s": 1.5, "zoom_depth": 0.08,
                "punch_rate_per_minute": 1.0, "text_fx_per_minute": 1.0},
    "balanced": {"target_coverage": 0.50, "video_broll_share": 0.20,
                "broll_seconds_min": 3.0, "broll_seconds_max": 5.0,
                "min_oncamera_gap_s": 1.0, "zoom_depth": 0.10,
                "punch_rate_per_minute": 1.5, "text_fx_per_minute": 2.0},
    "busy":     {"target_coverage": 0.65, "video_broll_share": 0.30,
                "broll_seconds_min": 3.0, "broll_seconds_max": 5.0,
                "min_oncamera_gap_s": 0.8, "zoom_depth": 0.12,
                "punch_rate_per_minute": 2.5, "text_fx_per_minute": 3.0},
    "max":      {"target_coverage": 0.80, "video_broll_share": 0.45,
                "broll_seconds_min": 3.0, "broll_seconds_max": 5.0,
                "min_oncamera_gap_s": 0.6, "zoom_depth": 0.14,
                "punch_rate_per_minute": 3.5, "text_fx_per_minute": 4.5},
}

# The Studio-contract fields whose provenance the report exposes as
# `PresentationReport.settings_sources` — see presentation.models.settings_sources_for.
CONTRACT_FIELDS: Tuple[str, ...] = (
    "genre", "genre_secondary", "density", "target_coverage", "video_broll_share",
    "broll_seconds_min", "broll_seconds_max", "min_oncamera_gap_s", "zoom_depth",
    "punch_rate_per_minute", "text_fx_per_minute", "voice_preset", "voice_enhance",
    "sfx_source", "music_source", "allow_free_stock",
)


class PresentationSettings(BaseModel):
    broll: bool = True
    broll_video: bool = True
    # One face for the story's main character across every still and clip
    # (ReActor face swap on the beats that show them; see presentation/character.py).
    character_consistency: bool = True
    # The lead, decided by the caller (BuzzcafStudio's cast sheet):
    # {"name", "description", "gender"} is used as-is instead of asking the
    # model; {} means the story has no single on-screen lead (no swaps, no
    # model call); None (absent) means find it from the transcript.
    main_character: Optional[Dict[str, Any]] = None
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
    # A locally rendered newspaper clipping cutaway.
    newspapers: bool = True
    # A locally rendered case-file dossier cutaway.
    case_files: bool = True
    # Recreation-style cutaways.
    recreations: bool = True
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
    # The key-moment hits alone (flash/shutter/shake/glitch and their stinger,
    # thunder and riser). Off keeps the grade shift and slow push of each mood
    # -- a calm, scary telling without the jump-cut punctuation.
    mood_hits: bool = True
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
    # Linear gain on the music bed (0.08 ~ -22 dB). At 0.16 a library track
    # mastered at -15 LUFS sat only ~14 dB under the voice, which read as
    # "the music is too loud" on a narrated horror story.
    music_volume: float = 0.08
    music_duck: float = 0.85            # 0..1, how hard speech pushes the bed down
    music_synth_fallback: bool = True
    # A different cue per story act (a calm bed for the setup, a tense one for
    # the build, the drone rising at the climax), crossfaded at the boundary.
    music_by_act: bool = True
    # Music the video asks for by name. `music_brief` is one cue for the whole
    # video (the Studio's per-project music section); `[music: ...]` directives
    # in the script start a new cue at their point when `music_cues` is on.
    # Both use music_gen.parse_music_cue's grammar -- "sad solo violin, 70 bpm"
    # or "style=...; instruments=...; mood=...; bpm=...; engine=..." -- and
    # each cue's track is generated once (with `music_engine` unless the cue
    # names one) and reused after. "off" in a cue is silence from there.
    music_brief: str = ""
    music_cues: bool = True
    music_engine: str = "ace_step"
    # Tracks the creator picked from the library, each at a fixed level:
    # [{"file": "uploads/rain.flac", "volume_db": -20, "duck": false}], `file`
    # relative to data/music. When any is set they replace the automatic
    # choice (brief, cues, act beds and generation). "together" layers them
    # all under the whole video; "sequence" plays them one after another,
    # looping the list until the video ends.
    music_tracks: List[Dict[str, Any]] = Field(default_factory=list)
    music_tracks_mode: str = "together"
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
    # "studio_mic" (EQ + de-ess + comp + limiter, see render/audio.EQ_PRESETS)
    # is the default: a plain afftdn with no EQ after it is what made the voice
    # sound boxy, so a caller that never sets this now gets the de-boxed chain.
    voice_preset: str = "studio_mic"
    loudness_lufs: Optional[float] = -14.0
    # Which denoiser build_voice_chain reaches for: "auto" (best available),
    # "deepfilter" (needs the optional `df` package), "rnnoise" (needs a model
    # under data/models/rnnoise/*.rnnn), "ffmpeg" (a gentle afftdn), "off".
    voice_enhance: str = "auto"
    # Line-level effects (a phone-call cutaway, a hard echo on one word). Not
    # planned by any LLM yet — a beat/directive can populate this list and it
    # will be applied; see presentation.sound.apply_voice_master.
    voice_fx: List["VoiceFxCue"] = Field(default_factory=list)
    # Where sound effects and the music bed come from, in order of preference:
    # "auto" = the user's own library (data/sfx, data/music) → a cached
    # ComfyUI generation → nothing. NEVER the synthesised FFmpeg noise-based
    # fallback by default — that one-shot white/pink noise is what produced
    # the periodic hiss. "synth" opts back into it explicitly; "library" /
    # "generated" / "off" each use exactly one source and nothing else.
    # Music keeps its own genre-specific synthesised bed (a horror drone, wind)
    # as explicit, deliberate genre behaviour regardless of this setting —
    # that is a continuous mood cue, not the one-shot hiss this setting fixes.
    sfx_source: str = "auto"
    music_source: str = "auto"
    # Fades for the music/ambience beds and the programme's own end, so a bed
    # never starts or stops on a click. Short per-SFX fades (fixed, not a
    # setting) kill clicks the same way on one-shot effects.
    fade_in_s: float = 0.5
    fade_out_s: float = 1.5
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
    # 3-5 s by default (the creator's call, 2026-09-27); never past
    # BROLL_SECONDS_CAP, which is also the longest clip Wan is asked for.
    broll_seconds_min: float = 3.0
    broll_seconds_max: float = 5.0
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

    # --- density & zoom (Studio contract) ---
    # A second genre blended into the first: the primary alone still drives the
    # grade, atmosphere and topic transitions, but the FX palette and B-roll
    # style tags become the union of both, the primary weighted ~65/35 over the
    # secondary. See presentation.genre.blended_fx_palette / blended_style_tags.
    genre_secondary: Optional[str] = None
    # One Studio-facing knob standing in for the handful of numeric ones below:
    # "calm" | "balanced" | "busy" | "max". Whichever of those the caller did
    # NOT set explicitly are filled from this density's row in DENSITY_PRESETS
    # by `resolve_density()` — the caller's own explicit values always win.
    density: Optional[str] = None
    # Emphasis punch-ins (facezoom._plan_punches) per minute. None resolves to
    # the density's own rate, else the historical 1.5 — see
    # `effective_punch_rate_per_minute`.
    punch_rate_per_minute: Optional[float] = None
    # Text-effect beats (pop-ups, cards, …) per minute. No pass reads this yet;
    # it is stored and resolved so a later text-effect pass can via
    # `effective_text_fx_per_minute`.
    text_fx_per_minute: Optional[float] = None
    # Studio's own asset sourcing is licensed/generated only; this opts a
    # project into free stock sources as well.
    allow_free_stock: bool = False

    # --- which language model plans the pass ---
    # "auto" (default): the local LM Studio model BuzzEdit has always used.
    # "openai_compat": an OpenAI-compatible chat endpoint reached with an API
    # key -- BuzzcafStudio sends its claude-local-api proxy here when the user
    # turns on "Use Claude for video production", so the moods, topics,
    # thumbnail title and picture prompts come from Claude in seconds instead
    # of waiting on a local model to load and answer. Pictures themselves are
    # still ComfyUI's. Falls back to LM Studio when the endpoint is down.
    llm_provider: str = "auto"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # --- text effects ---
    # Text behind the speaker's head, knockout, focus, newspaper highlighter
    # sweeps, kinetic keyword pop-ins, hand-drawn annotations, and
    # typewriter/glitch reveals — see presentation.text_fx. Scaled by
    # `text_fx_per_minute`/`effective_text_fx_per_minute` above; off turns the
    # whole pass off regardless of density.
    text_fx: bool = True
    # None picks the per-genre palette (presentation.text_fx.palette_for);
    # a list restricts the pass to exactly those style names.
    text_fx_styles: Optional[List[str]] = None
    # "Generative AI" in the corner of every generated picture on screen
    # (presentation/graphics.py::place_ai_labels).
    ai_label: bool = True
    # "auto" uses the best person-segmentation backend already importable in
    # this venv (nothing is installed or downloaded to get one — see
    # presentation.matte.detect_backend); "off" skips behind_head entirely and
    # runs focus as a plain face spotlight, the same degrade as no backend
    # being available at all.
    person_matte: str = "auto"

    def resolve_density(self) -> "PresentationSettings":
        """Fill any density-covered field the caller did not explicitly set
        from this settings' `density` preset. Fields the caller DID pass —
        explicit numbers, or values already applied by a BuzzEdit override —
        are left exactly as given; density only supplies a default."""
        preset = DENSITY_PRESETS.get((self.density or "").strip().lower())
        if not preset:
            return self
        set_fields = self.model_fields_set
        updates = {field: value for field, value in preset.items()
                  if field not in set_fields}
        return self.model_copy(update=updates) if updates else self

    @property
    def effective_punch_rate_per_minute(self) -> float:
        if self.punch_rate_per_minute is not None:
            return self.punch_rate_per_minute
        preset = DENSITY_PRESETS.get((self.density or "").strip().lower())
        return preset["punch_rate_per_minute"] if preset else 1.5

    @property
    def effective_text_fx_per_minute(self) -> float:
        if self.text_fx_per_minute is not None:
            return self.text_fx_per_minute
        preset = DENSITY_PRESETS.get((self.density or "").strip().lower())
        return preset["text_fx_per_minute"] if preset else 2.0

    @property
    def crowd_gap_s(self) -> float:
        """The minimum on-camera gap enforced when deciding whether one
        cutaway crowds another. Busy/max density allows cutaways back-to-back
        (0s); calmer densities floor it at `min_oncamera_gap_s` rather than the
        larger, coverage-derived `cutaway_gap_s` this check used before."""
        if (self.density or "").strip().lower() in ("busy", "max"):
            return 0.0
        return self.min_oncamera_gap_s

    @field_validator("broll_seconds_min", "broll_seconds_max")
    @classmethod
    def _clamp_broll_seconds(cls, v: float) -> float:
        return max(BROLL_SECONDS_FLOOR, min(BROLL_SECONDS_CAP, float(v)))

    @field_validator("target_coverage")
    @classmethod
    def _clamp_coverage(cls, value: float) -> float:
        # Up to the whole video (the creator's call, 2026-10-03: video, images
        # and his face each settable to 100%). Above ~0.8 the speaker is barely
        # seen -- a choice now, not a bug.
        return max(0.0, min(1.0, float(value)))

    @field_validator("video_broll_share")
    @classmethod
    def _clamp_video_share(cls, value: float) -> float:
        # Up to every cutaway being a clip (the creator's call, 2026-10-02, for
        # story channels). Each clip costs ~2.5 min of Wan where a still costs
        # ~20 s, so a high share is a long overnight pass -- slower, not broken.
        return max(0.0, min(1.0, float(value)))

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


def settings_sources_for(incoming: Dict[str, Any], override_fields: Dict[str, Any],
                         resolved: "PresentationSettings") -> Dict[str, str]:
    """Where each Studio-contract field's final value came from, for the
    morning report's `settings_sources`.

    `incoming` is the raw settings dict Studio sent (before overrides);
    `override_fields` is whatever the BuzzEdit-side override applied on top
    (see store.app_settings.apply_presentation_overrides); `resolved` is the
    PresentationSettings actually used, after `resolve_density()`.
    """
    density_fields = DENSITY_PRESETS.get((resolved.density or "").strip().lower(), {})
    sources: Dict[str, str] = {}
    for field in CONTRACT_FIELDS:
        if field in override_fields:
            sources[field] = "buzzedit_override"
        elif field in incoming:
            sources[field] = "studio"
        elif field in density_fields:
            sources[field] = "density"
        else:
            sources[field] = "default"
    return sources


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
    # Where each Studio-contract field's final value came from — see
    # presentation.models.settings_sources_for.
    settings_sources: Dict[str, str] = Field(default_factory=dict)
    program: Dict[str, Any] = Field(default_factory=dict)
    beats_planned: int = 0
    beats_dropped: List[Dict[str, str]] = Field(default_factory=list)
    plan_source: str = ""
    # What the prompts were styled as — set from the transcript when the user
    # did not choose, so a wrong detection is visible in the morning report.
    genre: str = "general"
    genre_secondary: str = ""
    assets_generated: int = 0
    assets_cached: int = 0
    assets_failed: List[Dict[str, str]] = Field(default_factory=list)
    # Beats filled from free stock (Pexels/Pixabay) because ComfyUI could not
    # make them and the user allowed it — see presentation.stock. Each entry:
    # {beat_id, source, url, author}.
    stock_used: List[Dict[str, str]] = Field(default_factory=list)
    broll_placed: int = 0
    popups_placed: int = 0
    graphics_placed: int = 0
    # Text cards placed, by kind ({"stat_callout": 3, "chapter_title": 4, …}),
    # and how many entities the extraction found.
    cards_placed: Dict[str, int] = Field(default_factory=dict)
    entities_found: int = 0
    maps_placed: int = 0
    newspapers_placed: int = 0
    case_files_placed: int = 0
    recreations_placed: int = 0
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
    # The windowed punches split by direction (zooms_windowed = the sum of
    # these two) — how many were a punch-in versus a pull-back.
    zooms_punch_in: int = 0
    zooms_punch_out: int = 0
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
    # Which denoiser engine actually ran ("deepfilter"/"rnnoise"/"ffmpeg"/"off"/
    # ""), and where each placed sound effect and the music bed came from
    # ({"library": n, "generated": n, "synth": n, "off": n}) — what the hiss
    # diagnosis needed and the morning report did not have.
    voice_enhance_used: str = ""
    sfx_source_used: Dict[str, int] = Field(default_factory=dict)
    music_source_used: str = ""
    thumbnail: Optional[str] = None
    output_path: Optional[str] = None
    # The checks run on the dressed timeline and the rendered file
    # ({"name", "ok", "detail", "value"}); failures are also listed in `degraded`.
    verification: List[Dict[str, Any]] = Field(default_factory=list)
    # Vertical outputs: [{"kind": "short"|"vertical", "path", "topic", "seconds"}].
    shorts: List[Dict[str, Any]] = Field(default_factory=list)
    # Text effects (presentation.text_fx): how many of each style landed
    # ({"kinetic_words": 4, "behind_head": 2, ...}), which person-segmentation
    # backend actually ran ("grabcut" / "off" / ...), and every instance that
    # could not be placed ([{"style", "reason"}] — a behind_head with no matte
    # for its window, mainly).
    text_fx_placed: Dict[str, int] = Field(default_factory=dict)
    person_matte_used: str = ""
    text_fx_skipped: List[Dict[str, str]] = Field(default_factory=list)
    # Designed graphics (presentation/graphics.py) placed, by kind -- the
    # full-frame layouts, the live speaker slots and the overlays.
    designed_graphics: Dict[str, int] = Field(default_factory=dict)
    ai_labels: int = 0
    degraded: List[str] = Field(default_factory=list)
    # How B-roll generation went, phase by phase (presentation/assets.py):
    # the memory hand-over verdicts and, for video, clips made / kept as
    # stills and why any were skipped.
    generation: Dict[str, Any] = Field(default_factory=dict)
    timings: List[StageTiming] = Field(default_factory=list)
