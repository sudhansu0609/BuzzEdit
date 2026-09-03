from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field

# The shortest span the pipeline will both *detect* as a filler and *remove*
# from the cut. One constant on purpose: when the detector's admission floor
# (asr/fumble_engine) sat below the rebuild's removal floor here, a 0.20s "uh"
# was marked cut in the UI and then silently bridged back into the render.
FILLER_MIN_SECONDS = 0.18

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
    kind: str = "video"  # "video" | "audio" | "image"

class WordItem(BaseModel):
    id: str
    text: str
    start_frame: int
    end_frame: int
    enabled: bool = True
    disfluency: bool = False
    speaker: Optional[str] = None
    # Why the planner cut this word ("retake", "filler_sound", "not_fluent",
    # "not_grammatical", …). The decision was being thrown away at every
    # dict-to-WordItem conversion, so the transcript panel could show that a word
    # was cut but never why — which is the one thing a user needs in order to
    # judge whether to put it back.
    reason: Optional[str] = None
    # The planner was unsure and something else decided. Kept so the UI can mark
    # a cut as a judgement call rather than a certainty.
    candidate: bool = False
    # The word in the script the speaker actually spoke (Devanagari for Hindi),
    # while `text` holds the romanized form the captions and the UI use. Every
    # grammar/fluency pass judges the native script — to a model, romanized Hindi
    # reads as broken English, so it scores good Hindi as wrong. This was being
    # dropped at every dict-to-WordItem conversion, which meant the re-cut path
    # ("Auto Edit" on an existing timeline) fed romanized text to prompts that
    # state the input is Devanagari, and the Devanagari negations in
    # verify.PROTECTED_WORDS could never match. None for English speech and for
    # words added by hand.
    word_native: Optional[str] = None

class TimelineEffect(BaseModel):
    type: str  # "zoompan", "fade", "lut"
    params: Dict[str, float | str] = Field(default_factory=dict)


class Transform(BaseModel):
    """Geometry for a clip: crop, then zoom/pan within the project canvas.

    Crop values are fractions (0..1) eaten off each edge of the *source* frame.
    `scale` is relative to the canvas: 1.0 fills it, >1 zooms in, <1 shrinks.
    `pos_x`/`pos_y` are -1..1 offsets from centre (±1 = flush against an edge).

    Setting `scale_end`/`pos_x_end`/`pos_y_end` animates from the start value to
    the end value across the clip (Ken Burns). Animated zoom is driven by FFmpeg's
    `zoompan`, which cannot zoom below its input, so the chain pre-scales to the
    smaller of the two scales and zooms up from there — see render/effects.py.
    """
    crop_left: float = 0.0
    crop_top: float = 0.0
    crop_right: float = 0.0
    crop_bottom: float = 0.0
    scale: float = 1.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    rotation: float = 0.0          # degrees, clockwise
    flip_h: bool = False           # mirror left-to-right
    flip_v: bool = False           # mirror top-to-bottom
    opacity: float = 1.0           # overlay tracks only; V1 is always opaque
    scale_end: Optional[float] = None
    pos_x_end: Optional[float] = None
    pos_y_end: Optional[float] = None

    def is_identity(self) -> bool:
        return (
            self.crop_left == 0 and self.crop_top == 0
            and self.crop_right == 0 and self.crop_bottom == 0
            and self.scale == 1.0 and self.pos_x == 0.0 and self.pos_y == 0.0
            and self.rotation == 0.0 and self.opacity == 1.0
            and not self.flip_h and not self.flip_v
            and not self.is_animated()
        )

    def is_animated(self) -> bool:
        return (
            (self.scale_end is not None and self.scale_end != self.scale)
            or (self.pos_x_end is not None and self.pos_x_end != self.pos_x)
            or (self.pos_y_end is not None and self.pos_y_end != self.pos_y)
        )


class ColorWheel(BaseModel):
    """One colour-balance wheel: a red/green/blue push for a tonal range.

    These are the primaries of a real grade — lift (shadows), gamma (midtones)
    and gain (highlights). Each value is -1..1 and maps straight onto FFmpeg's
    `colorbalance`, which is the only filter that adjusts colour *per tonal
    range*; `eq` can only move the whole picture at once.
    """
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0

    def is_identity(self) -> bool:
        return self.r == 0.0 and self.g == 0.0 and self.b == 0.0

    def clamped(self) -> "ColorWheel":
        limit = lambda v: max(-1.0, min(1.0, float(v)))
        return ColorWheel(r=limit(self.r), g=limit(self.g), b=limit(self.b))


class ColorGrade(BaseModel):
    """Colour correction applied to a clip (or to the whole program via master).

    Values are the neutral identity by default so an untouched grade compiles to
    no filter at all. `preset` is only a label for the UI — the numbers are the
    truth, so a preset that gets tweaked stays tweaked.

    The fields fall into three groups, and they are applied in that order so the
    result matches how a colourist works:

      1. *Correction* — exposure, white balance, and shadow/highlight recovery:
         getting the picture right before anything creative happens.
      2. *Grade* — the three colour wheels, saturation/vibrance, hue.
      3. *Look* — LUT, sharpen, vignette, fades.
    """
    preset: Optional[str] = None
    # --- correction ---
    exposure: float = 0.0          # stops, -3..3
    brightness: float = 0.0        # -1..1  (eq)
    contrast: float = 1.0          # 0..3   (eq)
    gamma: float = 1.0             # 0.1..10 (eq)
    temperature: float = 0.0       # -1 (cool) .. 1 (warm)
    tint: float = 0.0              # -1 (green) .. 1 (magenta)
    shadows: float = 0.0           # -1..1, lift or crush the low end
    highlights: float = 0.0        # -1..1, recover or push the high end
    # --- grade ---
    lift: ColorWheel = Field(default_factory=ColorWheel)     # shadows
    midtones: ColorWheel = Field(default_factory=ColorWheel)  # gamma
    gain: ColorWheel = Field(default_factory=ColorWheel)     # highlights
    saturation: float = 1.0        # 0..3   (eq)
    vibrance: float = 0.0          # -2..2, saturates the *unsaturated* colours
    hue: float = 0.0               # degrees
    # --- look ---
    lut_file: Optional[str] = None  # path to a .cube 3D LUT
    lut_strength: float = 1.0       # 0..1, how much of the LUT to keep
    sharpen: float = 0.0           # 0..2, unsharp luma amount
    denoise: float = 0.0           # 0..1, temporal+spatial noise reduction
    vignette: float = 0.0          # 0..1, corner darkening
    fade_in: float = 0.0           # seconds of fade from black at clip start
    fade_out: float = 0.0          # seconds of fade to black at clip end

    def is_identity(self) -> bool:
        return (
            self.exposure == 0.0
            and self.brightness == 0.0 and self.contrast == 1.0
            and self.saturation == 1.0 and self.gamma == 1.0
            and self.temperature == 0.0 and self.tint == 0.0
            and self.shadows == 0.0 and self.highlights == 0.0
            and self.lift.is_identity() and self.midtones.is_identity()
            and self.gain.is_identity()
            and self.vibrance == 0.0 and self.hue == 0.0
            and not self.lut_file
            and self.sharpen == 0.0 and self.denoise == 0.0
            and self.vignette == 0.0
            and self.fade_in == 0.0 and self.fade_out == 0.0
        )


class ChromaKey(BaseModel):
    """Green/blue-screen keying for one clip.

    A key is only as good as its edges, so this is deliberately more than a
    single "remove green" switch. The four controls that actually decide quality:

      * `similarity` — how far from the key colour still counts as background.
        Too low leaves green patches, too high eats the subject.
      * `blend` — the soft edge of the key itself, in colour space.
      * `choke` — shrinks the matte by whole pixels, which is what removes the
        bright green fringe that survives any colour-space tweak.
      * `feather` — blurs the matte so the composite edge is not a hard staircase.

    `spill` is separate from all of them: it fixes green *reflected onto the
    subject*, which is not an edge problem and cannot be keyed away without
    cutting into them.
    """
    enabled: bool = True
    # "chroma" keys in YUV (right for green/blue screens — brightness variation
    # across a lit screen does not move the hue). "color" keys in RGB, which is
    # better for a flat graphic colour that is not a screen at all.
    key_type: str = "chroma"       # "chroma" | "color"
    color: str = "0x00FF00"        # the key colour, 0xRRGGBB
    similarity: float = 0.18       # 0.01..1
    blend: float = 0.08            # 0..1
    choke: float = 0.0             # 0..1, shrinks the matte (erosion passes)
    feather: float = 0.0           # 0..1, gaussian blur on the matte
    spill: float = 0.0             # 0..1, despill mix
    spill_expand: float = 0.0      # 0..1, how far the despill reaches
    # Show the matte instead of the picture: white = keep, black = drop. The one
    # reliable way to judge a key, because a fringe that is invisible against a
    # dark background is glaring against a light one.
    show_matte: bool = False

    def is_identity(self) -> bool:
        return not self.enabled


class TextStyle(BaseModel):
    """Everything drawtext needs to paint one text element.

    `font_file` wins over `font_family`: the family is a human label kept for the
    UI, while the file is what FFmpeg actually loads. Positions are the same
    -1..1 canvas coordinates used by Transform so the two panels feel alike.
    """
    font_family: str = "Arial"
    font_file: Optional[str] = None
    font_size: int = 64
    color: str = "white"
    opacity: float = 1.0
    bold: bool = False
    italic: bool = False
    line_spacing: int = 8
    align: str = "center"          # "left" | "center" | "right"
    pos_x: float = 0.0             # -1..1, 0 = centre
    pos_y: float = 0.0             # -1..1, 0 = centre (positive = down)
    # Outline
    stroke_width: int = 0
    stroke_color: str = "black"
    # Drop shadow
    shadow_x: int = 0
    shadow_y: int = 0
    shadow_color: str = "black@0.6"
    # Background box
    box: bool = False
    box_color: str = "black@0.6"
    box_padding: int = 12
    # Animation. drawtext draws none/fade/pop/slide-up; the rest go through the
    # libass engine (render/ass.py): typewriter, karaoke, scale_in, blur_in,
    # glitch, shake, flicker.
    animation: str = "none"
    animation_duration: float = 0.3
    # Karaoke: the colour of the word being spoken.
    highlight_color: str = "#FFE23A"


class TrackState(BaseModel):
    """Per-track (layer) switches, kept on the Timeline rather than on the clips.

    Tracks are implicit — they exist because clips reference them — so their state
    has to live somewhere that survives an empty lane, and applying "hidden" to
    every clip individually would be undone the moment a new clip lands there.
    """
    hidden: bool = False       # not rendered at all
    locked: bool = False       # clips on it cannot be edited or moved
    muted: bool = False        # audio not mixed into the programme


class Transition(BaseModel):
    """A transition *into* a clip, from whatever precedes it.

    Only meaningful at a junction between two programme segments. `duration` is
    clamped at render time to what the two clips can actually cover — `xfade`
    needs both sides to span the overlap.
    """
    type: str = "fade"          # any name from render/transitions.CATALOGUE
    duration: float = 0.5
    # Who chose it. The presentation pass tags the transitions it picks per
    # topic boundary ("presentation") so a re-run replaces only those and a
    # transition the user set by hand is never overwritten.
    origin: Optional[str] = None


class AudioMaster(BaseModel):
    """Programme-wide sound treatment: clean the voice, then hit a loudness target.

    The voice chain (denoise, de-ess, compression) is applied to the A1 programme
    audio *before* music and effects are mixed in, so it never pumps the music;
    `loudness_lufs` is applied to the finished mix, which is what YouTube
    measures (it normalises to -14 LUFS and turns down anything louder).
    All values are neutral by default so an untouched master compiles to nothing.
    """
    voice_denoise: float = 0.0     # 0..1, FFT noise reduction on the voice
    voice_deess: float = 0.0       # 0..1, de-esser strength
    voice_compress: float = 0.0    # 0..1, gentle broadcast compression
    # Integrated loudness target for the final mix, or None to leave levels alone.
    loudness_lufs: Optional[float] = None
    true_peak_db: float = -1.5
    origin: Optional[str] = None

    def is_identity(self) -> bool:
        return (self.voice_denoise <= 0.0 and self.voice_deess <= 0.0
                and self.voice_compress <= 0.0 and self.loudness_lufs is None)


class AtmosphereEffect(BaseModel):
    """A generated overlay: rain, snow, lightning, sunlight, light leak, fog, wind, grain."""
    type: str
    intensity: float = 0.5      # 0..1
    speed: float = 1.0          # 0.1..4, relative
    color: Optional[str] = None  # 8-digit hex for the coloured effects
    enabled: bool = True
    # Who put it here. The presentation pass tags its own layer ("presentation")
    # so a re-run replaces exactly that one and leaves the user's effects alone.
    origin: Optional[str] = None


class TextClip(BaseModel):
    """Content + styling for a text item. Lives on TimelineItem.text."""
    content: str = "Text"
    style: TextStyle = Field(default_factory=TextStyle)
    preset: Optional[str] = None
    # An animated number: {"to": 25, "from": 0, "prefix": "", "suffix": "%",
    # "decimals": 0, "seconds": 1.2}. The content is then drawn as the count
    # rising from `from` to `to` over `seconds` from the clip's start, and
    # `content` is only the label kept for the UI.
    counter: Optional[Dict[str, Any]] = None
    # Word timings for karaoke-style captions: [{"text", "start_s", "end_s",
    # "emphasis"?}] relative to the clip start. Filled by the caption
    # generator; used by the animated text engine to highlight the word being
    # spoken, and to draw an emphasised word (a number, a shouted word) larger.
    words: List[Dict[str, Any]] = Field(default_factory=list)
    # A smaller second line under the text (the English translation of a
    # Hindi caption). Drawn by the animated engine only.
    second_line: Optional[str] = None


class TimelineItem(BaseModel):
    id: str
    track: str  # "V1", "A1", "V2", "A2", "T1", "CAP", ...
    source_id: Optional[str] = None   # None for text items
    source_start_frame: int = 0
    source_end_frame: int = 0
    timeline_start_frame: int = 0
    timeline_end_frame: int = 0
    enabled: bool = True
    effects: List[TimelineEffect] = Field(default_factory=list)
    anchor_word_id: Optional[str] = None
    # "auto" = AI/word-managed (V1/A1, rebuilt from transcript, not hand-editable);
    # "manual" = user-placed clip on an overlay/mix track, freely editable.
    origin: str = "auto"
    locked: bool = False
    # "media" = trims a source file; "text" = drawtext element;
    # "compound" = a group of child items moved and trimmed as one block;
    # "adjustment" = holds no picture of its own — its transform, grade and
    # atmosphere are applied to every layer *below* it for its span.
    kind: str = "media"
    transform: Optional[Transform] = None
    color: Optional[ColorGrade] = None
    # Green/blue-screen key. Applied before the grade, because grading first
    # moves the very colour the key is looking for.
    chroma: Optional[ChromaKey] = None
    text: Optional[TextClip] = None
    # Generated atmosphere carried by an adjustment clip. The programme-wide list
    # lives on the Timeline; this one is windowed to the clip.
    atmosphere: List[AtmosphereEffect] = Field(default_factory=list)
    # Transition INTO this clip from the previous programme segment.
    transition: Optional[Transition] = None
    # Compound children. Their timeline frames are RELATIVE to the parent's start
    # so the whole group slides together when the parent moves.
    children: List["TimelineItem"] = Field(default_factory=list)
    mute: bool = False
    volume: float = 1.0
    # Audio-only switches for mix-track items (A2, A3, …). `loop` repeats a
    # short source (a music bed, an ambience loop) for the item's whole span
    # instead of going silent when the file runs out; the fades are seconds at
    # either end; `duck` (0..1) is how hard the A1 voice pushes this item down
    # while somebody is speaking — 0 leaves it alone, 1 is full ducking. Ducking
    # is done with a sidechain compressor keyed off the programme voice, so it
    # follows the speech exactly rather than a guessed envelope.
    loop: bool = False
    audio_fade_in: float = 0.0
    audio_fade_out: float = 0.0
    duck: float = 0.0
    label: Optional[str] = None

    @property
    def duration_frames(self) -> int:
        return self.timeline_end_frame - self.timeline_start_frame


TimelineItem.model_rebuild()


class Timeline(BaseModel):
    version: str = "1.0"
    fps_num: int = 30
    fps_den: int = 1
    width: int = 1920
    height: int = 1080
    duration_frames: int = 0
    sources: Dict[str, SourceFile] = Field(default_factory=dict)
    words: List[WordItem] = Field(default_factory=list)
    items: List[TimelineItem] = Field(default_factory=list)
    # Program-wide grade/geometry, applied after V1 concat. Unlike per-clip
    # effects on V1 these survive a transcript rebuild, so they are the right
    # home for "correct the whole video" adjustments.
    master_color: Optional[ColorGrade] = None
    master_transform: Optional[Transform] = None
    # Leading blank card the program is pushed behind, so an intro's text has
    # somewhere to live without covering the first seconds of footage. The word
    # rebuild starts V1/A1 at this offset, and the compiler concats a matching
    # colour+silence segment in front, so the pad survives transcript edits.
    program_offset_frames: int = 0
    program_offset_color: str = "black"
    # How much of that offset is the cold open (a copied hook sentence played
    # before the title). Kept apart so an intro card can be added or removed
    # without shifting the cold open along with it.
    cold_open_frames: int = 0
    # Auto-edit pacing. A pause longer than `max_pause_seconds` is trimmed down to
    # `pause_padding_seconds` either side of the speech rather than deleted, so a
    # tightened edit still breathes; shorter pauses are left exactly as recorded.
    max_pause_seconds: float = 0.40
    pause_padding_seconds: float = 0.12
    # A removal shorter than this is not worth the jump cut it would cost, so the
    # material is kept and the segments stay joined. Must never sit above the
    # filler admission floor, or a detected filler plays despite being marked cut.
    min_removal_seconds: float = FILLER_MIN_SECONDS
    # A kept fragment shorter than this, with removals either side, is a flash on
    # screen rather than content; dropping it merges two cuts into one.
    min_segment_seconds: float = 0.35
    # Coarse loudness envelope of the source audio: {"rate": frames/sec, "db":
    # [ints]}. Stored so the rebuild can land every cut on the quietest nearby
    # moment — word toggles happen long after the audio was analysed, so the
    # envelope has to live with the project rather than be recomputed.
    energy_envelope: Optional[Dict[str, object]] = None
    # Where speech actually is in the source, in source frames. Clamping a word's
    # edges is not enough on its own: a word that claims eight seconds and is 74%
    # speech still hides two seconds of silence *inside* it, which no gap-based
    # rule can see. The rebuild intersects kept words with these regions.
    speech_regions: List[List[int]] = Field(default_factory=list)
    # Per-track switches, keyed by track name ("V2", "A1", "T1"). Absent means
    # every switch is off.
    tracks: Dict[str, TrackState] = Field(default_factory=dict)
    # Empty lanes the user has reserved but not yet dropped anything on. Tracks are
    # otherwise implicit, so without this an added-but-empty track vanishes on reload.
    extra_tracks: List[str] = Field(default_factory=list)
    # Applied at every programme cut that has no transition of its own.
    default_transition: Optional[Transition] = None
    # Generated atmosphere composited over the finished picture.
    effects: List[AtmosphereEffect] = Field(default_factory=list)
    # Voice clean-up and the loudness target for the finished mix.
    audio_master: Optional[AudioMaster] = None
    # Letterbox the programme to this aspect ratio (2.39 = cinematic scope).
    # The output resolution is unchanged; black bars are matted on.
    aspect_bars: Optional[float] = None
    revision: int = 1

    def recalculate_duration(self) -> None:
        max_frame = 0
        for item in self.items:
            if item.enabled:
                max_frame = max(max_frame, item.timeline_end_frame)
        self.duration_frames = max_frame
