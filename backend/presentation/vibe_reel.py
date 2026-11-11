"""The vibe reel (REELS_DESK_PLAN.md §6, VR1): a soothing Instagram short made
of generated portrait scenes, 1-4 lines faded on screen, and (when asked for)
a music bed under ambience -- nothing else. A lines reel's `music` is often
null (2026-10-24, VR8: a Life3Baje mood reel gets no AI music at all -- the
creator adds his own song in the Instagram app, over the ambience); its
`<name>.mp4` then carries ambience only, and no `<name>_nomusic.mp4` is
written (there would be nothing left to drop). No zooms on a generated clip,
no transitions besides a 1 s dissolve, no captions, no popups, no logo, no
voice.

This module is self-contained on purpose: it does not go through the talking-
video presentation pipeline (planner, placement, compiler) at all -- the
pipeline's own card/popup/zoom machinery is exactly what produced the first,
wrong mood reel (REELS_DESK_PLAN.md §6). It still reuses the pipeline's proven
pieces directly: `presentation.assets.generate_assets` for the Z-Image stills
and Wan clips (GPU hand-over, caching, retries included), `render.ass` for
Devanagari-correct text (libass, never drawtext), `render.transitions` for the
dissolve math, and `presentation.music_gen` / `presentation.sfx_library` for
the bed and the ambience.

A shot is EITHER a still (a Z-Image picture only, held with a very slow
"breathing" push -- the only motion a still gets) or a Wan clip (its own
natural motion -- rain, mist, clouds -- and no zoom at all, so the clip's own
motion is never fought by a second one). A shot that cannot be made usable --
ComfyUI empty-handed, a clip too short to hold its slot, or a near-black
placeholder frame -- is retried twice more and then fails the whole job: a
vibe reel with a missing scene is not a reel, never a gap filled with a card.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from config import FFMPEG_BIN
from presentation import art_style as art_mod
from presentation import assets as assets_mod
from presentation import caption_align
from presentation import character as character_mod
from presentation import music_gen, sfx_library
from presentation import workflows as workflows_mod
from presentation.models import Beat, Character, PresentationSettings
from render import ass as ass_mod
from render.encoder import get_encoder_flags
from render.transitions import build_video as xfade_video, clamp_duration as xfade_clamp
from runtime import gpu_handover
from timeline.schema import TextClip, TextStyle
from utils.ffmpeg_utils import FFmpegError, get_video_duration
from utils.proc import NO_WINDOW

logger = logging.getLogger("presentation.vibe_reel")


class VibeReelError(RuntimeError):
    """A vibe reel could not be made; the message is the job's failure reason."""


# --- constants --------------------------------------------------------------

WIDTH, HEIGHT = 1080, 1920
FPS = 30
DISSOLVE_S = 1.0
FADE_IN_S = 0.8          # from black, at the very start
FADE_OUT_S = 1.5         # to black, at the very end
AUDIO_FADE_IN_S = 1.5
AUDIO_FADE_OUT_S = 2.5
LOUDNESS_LUFS = -14.0
TRUE_PEAK_DBTP = -1.5
CARD_FADE_S = 0.8
# ~64 px cards / ~72 px captions at 1080 wide (2026-10-23 night rules, VR7):
# his first look (2026-10-23 day) had bumped cards from 62 to 72 for
# readability -- the new Poppins Bold face (below) reads bolder at a smaller
# size, so cards settle at 64 and captions (meant to sit a touch larger,
# always-on during narration) take the 72 cards used to have.
CARD_FONT_PX = 66      # big and bold: a long line wraps onto a second row instead of shrinking
CREDIT_FONT_PX = 44
CARD_POS_Y_MINIMAL = 0.28   # TextStyle's -1..1 scale -> ~64% down the frame (text_style="minimal")
CARD_POS_Y_BOLD = -0.34     # -> ~33% down the frame, the upper third (text_style="bold")
CARD_POS_Y = CARD_POS_Y_MINIMAL
CREDIT_POS_Y = 0.34      # just under the card
MAX_CARD_WIDTH_FRAC = 0.85
MIN_CARD_FONT_PX = 24
MAX_CARD_ROWS = 4       # a couplet's two lines, each wrapped once at most
CARD_LINE_SPACING = 1.3
STILL_PUSH = 0.04        # "about 4% over the shot"

# 1-3 shots per mood reel (2026-10-23 night rules, VR7: "fewer cuts... one
# long shot is ideal" -- supersedes the earlier 3-7 floor); a narrated story
# reel still plans well inside this (VR1b: ~6 shots for a minute).
SHOTS_MIN, SHOTS_MAX = 1, 7
NARRATED_SHOTS_MAX = 80   # a 3-minute narrated reel at ~4.5 s a shot is ~40; room for repairs
# A shot's seconds, as validated on the request: up to the length of a whole
# mood reel (one continuous shot, VR7 rule B) -- a single generated clip's
# own practical ceiling is `CHAIN_SEGMENT_SECONDS_MAX` below; anything longer
# chains into several image-to-video segments.
SHOT_SECONDS_MIN, SHOT_SECONDS_MAX = 4.0, 30.0
LINES_MIN, LINES_MAX = 1, 4
AMBIENCE_MAX = 2

MAX_SHOT_ATTEMPTS = 3    # the first try plus 2 retries
SHOT_PAD_TOLERANCE_S = 0.5   # a clip short by up to this is held, not retried
BLACK_LUMA_MAX = 12.0    # 0..255 mean brightness; at or under this, a retry

# --- continuous shots (2026-10-23 night rules, VR7) --------------------------
# "A shot longer than one Wan clip (~5s) is a CONTINUOUS shot made by chaining
# image-to-video: the next segment starts from the previous segment's last
# frame, same prompt, joined with no dissolve." One generated clip already
# covers up to today's own trim/pad ceiling (8s) convincingly; past that, the
# shot is split into segments of at most that length, each an image-to-video
# call seeded from the previous segment's own last frame, then joined with a
# `CHAIN_SEAM_S` (about 2 frames) crossfade at each seam -- motion continuing,
# not a cut.
CHAIN_SEGMENT_SECONDS_MAX = 8.0
CHAIN_SEGMENTS_MAX = 4
CHAIN_SEAM_S = 2.0 / FPS

# --- quality (2026-10-23): hyper-realistic AI clips, not stock footage --------
# "standard" is today's behaviour unchanged. "reel_hq" swaps the still and
# clip workflows for `zimage_wan22_i2v_reel_portrait.json` (native portrait
# Z-Image -> Wan 2.2 Q5_K_M + lightx2v -> FlashVSR -> RIFE x2), falling back to
# the plain HQ16 graph and then today's own `broll_video` resolution (never a
# production's choice, never the global `clip_generation_mode`) -- recorded
# per shot via the asset's own `workflow_file`. Stills use the HQ still graph
# at a similar native-portrait size. Every prompt gets a photographic-realism
# suffix/negative so the clips read as filmed, not generated.
QUALITY_STANDARD = "standard"
QUALITY_REEL_HQ = "reel_hq"
QUALITIES = (QUALITY_STANDARD, QUALITY_REEL_HQ)
REEL_HQ_STILL_FILE = "zimage1_hq.json"
REEL_HQ_STILL_SIZE = (720, 1280)   # base latent; the graph's own 1.5x hires fix lands near 1080x1920

# Extended 2026-10-23 night (VR7): the creator's reference reels read as
# ordinary phone footage of a real place, not a posed/cinematic photograph --
# never the words camera/lens/phone/DSLR/"shot on"/"photo of" (those draw a
# literal camera even here; `strip_camera_words` runs on the scene first).
REALISM_SUFFIX = ("candid real photograph, natural available light, true-to-life colours, "
                  "fine film grain, real textures and small imperfections, shallow depth of "
                  "field, documentary realism, ordinary everyday real place in India, "
                  "natural imperfect light, muted true colours, handheld amateur footage feel, "
                  "slightly uneven framing")
REALISM_NEGATIVE = ("CGI, 3D render, illustration, painting, cartoon, oversaturated, HDR, "
                    "plastic skin, airbrushed, glossy, unreal engine, smooth AI look, text, watermark, "
                    # the VR7 test's tell: "steam" came out as a big swirling smoke plume
                    "swirling smoke, thick steam plume, ornate fantasy lamp, riderless moving vehicle, "
                    "vehicle driving by itself, objects moving on their own, morphing")
# Only weather and light move: in the VR7 test a parked, riderless scooter drove itself
# down the street (2026-10-24), which gives an AI clip away at once.
REALISM_MOTION_SUFFIX = ("slow, natural, steady; parked vehicles, buildings, furniture and objects "
                         "stay completely still, nothing moves on its own; only rain, wind, water, "
                         "mist, leaves, flames and light move")

# ffmpeg "look pass" (reel_hq only), applied before the text/caption overlay
# so the burned-in words stay clean: a faint temporal grain, a touch less
# saturation, a gentle filmic contrast curve, a faint vignette -- tuned to
# read as filmed, not as a filter.
LOOK_PASS_FILTER = ("eq=saturation=0.92,curves=preset=medium_contrast,"
                    "vignette=angle=PI/5,noise=alls=6:allf=t+u")
# A very subtle, slow handheld drift (2026-10-23 night rules, VR7): the whole
# frame scaled up a touch and panned by a small sine drift (~0.4-0.5% of the
# frame, two different periods so it never repeats obviously) -- applied in
# the same global pass as the grain/desaturation above, before the text.
HANDHELD_DRIFT_FILTER = (
    f"scale=w=iw*1.012:h=ih*1.012,crop=w={WIDTH}:h={HEIGHT}:"
    f"x='(iw-{WIDTH})/2+iw*0.005*sin(2*PI*t/7.3)':"
    f"y='(ih-{HEIGHT})/2+ih*0.004*sin(2*PI*t/5.1+1.0)'")

# The "illustrated" style's own look pass (2026-10-24, VR8): NO handheld
# drift and none of the real style's filmic contrast curve/vignette above --
# those are photographic tells, wrong on a drawing. A hand-animated feel
# instead: a touch of colour flattening, plus a STATIC paper-texture grain
# (ffmpeg's `noise` filter with no 't' flag holds the same random pattern on
# every frame -- a texture, not moving film grain). The hand-animated motion
# itself (an illustrated video shot stepped to `ANIMATION_HOLD_FPS`, held
# frames, "on twos") is applied per-shot in `video_shot_filter`, not here.
ILLUSTRATED_LOOK_PASS_FILTER = "eq=saturation=0.90:contrast=0.96,noise=alls=9:allf=u"
# Animation "on twos": a video shot's motion resampled to this, then back up
# to FPS -- each drawing held for the extra frames instead of smooth video
# motion (`video_shot_filter`'s `stepped=True`, illustrated art style only).
ANIMATION_HOLD_FPS = 12
# `grade` request field (2026-10-23 night rules, VR7): "natural" (default,
# nothing extra) | "bw" (full desaturation, on top of the look pass).
GRADE_NATURAL, GRADE_BW = "natural", "bw"
GRADES = (GRADE_NATURAL, GRADE_BW)
BW_FILTER = "hue=s=0"

# `text_style` request field (2026-10-23 night rules, VR7): "minimal" (1-2
# plain lines, lower middle) | "bold" (a short line, upper third, one keyword
# in yellow, named by `highlight`).
TEXT_STYLE_MINIMAL, TEXT_STYLE_BOLD = "minimal", "bold"
TEXT_STYLES = (TEXT_STYLE_MINIMAL, TEXT_STYLE_BOLD)
HIGHLIGHT_COLOR = "#FFD54A"

# A soft, feathered dark gradient band behind the text block (never a hard
# box): a horizontal Gaussian darkening of luma only, centred on the text's
# own Y, strength (`alpha`) scaled by how bright that part of the frame
# measures (brighter background -> a stronger scrim) -- see `scrim_filter`
# and `_scrim_alpha_for`.
SCRIM_ALPHA_MIN, SCRIM_ALPHA_MAX = 0.35, 0.50
SCRIM_BAND_FRAC = 0.16   # the Gaussian's sigma, as a fraction of frame height

# The lead character's reference portrait (VR4): one `seed_base` shared by
# every shot that shows them, so `assets.generate_assets`' face-key is the
# same call to call and the one portrait is reused -- a shot's own picture
# still varies because its seed also carries its beat id. Other shots keep
# their own varied seed_base for variety.
CHARACTER_SEED_BASE = 500_000

# --- narration (VR1b) --------------------------------------------------------

NARRATION_LEAD_S = 0.5      # silence before the first word
NARRATION_TAIL_S = 1.5      # silence after the last word
NARRATION_DISSOLVE_S = 0.4  # the default dissolve for a narrated reel
DISSOLVE_S_MIN, DISSOLVE_S_MAX = 0.2, 1.5
PUSH_MIN, PUSH_MAX = 0.04, 0.10
# A still's weighted share of the time left after the video shots, held this
# long: long enough to read, short enough that nothing overstays its scene.
STILL_SECONDS_MIN_NARR, STILL_SECONDS_MAX_NARR = 2.5, 7.0
# Music ~20dB under a full-level narration voice; ambience's own gain_db (the
# caller's own "~26dB under" in practice) is unchanged. Neither moves once set
# -- there is no further, speech-aware ducking.
NARRATION_MUSIC_GAIN_DB = -20.0

CAPTION_FONT_PX = 96   # 72 read small on a phone in the first Raat3Baje reel (creator, 2026-10-24)
CAPTION_MIN_FONT_PX = 28
CAPTION_POS_Y = 0.4          # TextStyle's -1..1 scale -> ~70% down the frame
CAPTION_WIDTH_FRAC = 0.88
CAPTION_FADE_S = 0.08        # effectively a hard cut, timed to the voice

# ffmpeg blackdetect, tuned for dark horror imagery (REELS_DESK_PLAN.md VR1b):
# a hit inside the opening fade-in or the closing fade-out is the fade itself,
# not a gap in the coverage.
BLACKDETECT_D = 0.25
BLACKDETECT_PIX_TH = 0.02
BLACKDETECT_PIC_TH = 0.98
_BLACKDETECT_RE = re.compile(
    r"black_start:([0-9.]+)\s+black_end:([0-9.]+)\s+black_duration:([0-9.]+)")

# A camera word draws a literal camera into the frame even as a negative
# (Z-Image ignores negatives for this) -- stripped from every prompt instead.
_CAMERA_WORDS_RE = re.compile(
    r"\bcamera\b|\bdslr\b|\blens(?:es)?\b|\btripod\b|shot on [a-z0-9 .\-]+|photo(?:graph)?s? of",
    re.IGNORECASE)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Kokila is Windows' own literary Devanagari face -- a system font (DirectWrite
# resolves it for libass without any bundled file).
DEVANAGARI_FONT = "Kokila"

# The Hinglish (Roman-letter) face (2026-10-23 night rules, VR7): "bold clean
# Latin sans -- download Poppins (Bold + SemiBold, OFL)... into BuzzEdit
# assets\fonts\ and reference it by file; fallback Montserrat/Segoe UI Bold
# if download fails." `render.ass.fonts_dir()` already points libass's own
# `fontsdir` at this app's `assets/fonts/` (its bundled-font search path), so
# dropping the two .ttf files there is the whole integration -- no code
# change needed for libass to find them. Georgia (the old, literary-serif
# default) is kept as the still-older fallback behind Montserrat/Segoe UI, in
# case none of the new files exist either.
_BUNDLED_FONTS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
_POPPINS_BOLD_FILE = _BUNDLED_FONTS_DIR / "Poppins-Bold.ttf"
_POPPINS_SEMIBOLD_FILE = _BUNDLED_FONTS_DIR / "Poppins-SemiBold.ttf"
_MONTSERRAT_BOLD_CANDIDATES = (r"C:\Windows\Fonts\Montserrat-Bold.ttf",
                               r"C:\Windows\Fonts\Montserrat Bold.ttf")
_SEGOE_UI_BOLD_FILE = r"C:\Windows\Fonts\segoeuib.ttf"


def _resolve_roman_font() -> str:
    """"Poppins" when the downloaded files are there, else "Montserrat" when
    that is installed on this machine, else Windows' own "Segoe UI" (its
    Bold weight) -- in the order the creator named."""
    if _POPPINS_BOLD_FILE.is_file():
        return "Poppins"
    if any(Path(c).is_file() for c in _MONTSERRAT_BOLD_CANDIDATES):
        return "Montserrat"
    return "Segoe UI"


ROMAN_FONT = _resolve_roman_font()
_FONT_FILES = {
    "Kokila": r"C:\Windows\Fonts\kokila.ttf",
    "Georgia": r"C:\Windows\Fonts\georgia.ttf",
    "Poppins": str(_POPPINS_BOLD_FILE) if _POPPINS_BOLD_FILE.is_file() else _SEGOE_UI_BOLD_FILE,
    "Montserrat": next((c for c in _MONTSERRAT_BOLD_CANDIDATES if Path(c).is_file()),
                       _MONTSERRAT_BOLD_CANDIDATES[0]),
    "Segoe UI": _SEGOE_UI_BOLD_FILE,
}


# --- request models ----------------------------------------------------------

class Shot(BaseModel):
    prompt: str = ""
    motion: str = ""
    seconds: float = 6.0
    # "still": a Z-Image picture only, held with a slow breathing push (the
    # only motion a still gets). "video": a Wan clip, its own motion, no
    # zoom. "stick" (STICK_WORLD_PLAN.md SW2): a StickScene, over its own
    # `bg` picture (made like a still shot) or today's plain ground -- the
    # fields below are `[stick: ...]`'s own (presentation/script.py::
    # directive_beats), `prompt` unused.
    kind: str = "still"
    # VR4: this shot is the reel's lead character -- the request's own
    # `character` must be set, and `generate_assets` swaps that one face in.
    shows_character: bool = False
    # kind="stick" only -- see presentation/script.py::directive_beats'
    # `stick` branch for what each of these means; `action` defaults to
    # "idle" exactly as the directive does, everything else is optional.
    action: str = "idle"
    scene: str = ""
    era: Optional[str] = None
    outfit: Optional[str] = None
    bg: Optional[str] = None
    caption: Optional[str] = None

    @field_validator("seconds")
    @classmethod
    def _seconds_range(cls, v: float) -> float:
        if not (SHOT_SECONDS_MIN <= v <= SHOT_SECONDS_MAX):
            raise ValueError(f"a shot's seconds must be {SHOT_SECONDS_MIN}-{SHOT_SECONDS_MAX}, got {v}")
        return v

    @field_validator("kind")
    @classmethod
    def _kind_known(cls, v: str) -> str:
        v = (v or "still").strip().lower()
        if v not in ("still", "video", "stick"):
            raise ValueError("a shot's kind must be 'still', 'video' or 'stick'")
        return v

    @model_validator(mode="after")
    def _prompt_required_unless_stick(self) -> "Shot":
        if self.kind != "stick" and not (self.prompt or "").strip():
            raise ValueError("a shot's prompt must not be empty")
        return self


class Line(BaseModel):
    text: str
    start_s: float
    end_s: float
    # "devanagari" | "roman" | None (detect: any Devanagari codepoint in the
    # text -> devanagari, else roman -- never Urdu/Nastaliq).
    script: Optional[str] = None

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("a line's text must not be empty")
        return v

    @field_validator("script")
    @classmethod
    def _script_known(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in ("devanagari", "roman"):
            raise ValueError("a line's script must be 'devanagari' or 'roman'")
        return v

    @model_validator(mode="after")
    def _times_ok(self) -> "Line":
        if self.end_s <= self.start_s:
            raise ValueError("a line's end_s must be after its start_s")
        return self


class Ambience(BaseModel):
    prompt: str
    gain_db: float = -20.0

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("an ambience's prompt must not be empty")
        return v


class Music(BaseModel):
    brief: Optional[str] = None
    bpm: Optional[int] = None
    path: Optional[str] = None

    @model_validator(mode="after")
    def _one_source(self) -> "Music":
        if not ((self.brief or "").strip() or (self.path or "").strip()):
            raise ValueError("music needs a brief to generate from, or a path to an existing file")
        return self


class Narration(BaseModel):
    """A voice track to cover end to end, plus the words as spoken (for
    captions -- see `Captions`): the reel's length becomes this file's
    duration + `NARRATION_LEAD_S` + `NARRATION_TAIL_S`, and every shot's
    `seconds` is read as a weight rather than an exact duration."""
    path: str
    text: str
    language: str = "hi"

    @field_validator("path")
    @classmethod
    def _path_exists(cls, v: str) -> str:
        if not v or not Path(v).is_file():
            raise ValueError(f"narration path does not exist: {v}")
        return v

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("narration needs the words as spoken, for captions")
        return v


class Captions(BaseModel):
    style: str = "phrases"   # "phrases" | "off"

    @field_validator("style")
    @classmethod
    def _style_known(cls, v: str) -> str:
        v = (v or "phrases").strip().lower()
        if v not in ("phrases", "off"):
            raise ValueError("captions.style must be 'phrases' or 'off'")
        return v


class VibeReelRequest(BaseModel):
    out_dir: str
    name: str = "reel"
    art_style: str = "real"
    # "standard" (today's workflows) | "reel_hq" (hyper-realistic AI clips --
    # see the module-level comment above `QUALITY_REEL_HQ`).
    quality: str = QUALITY_STANDARD
    # "natural" (default) | "bw" (2026-10-23 night rules, VR7) -- see `GRADES`.
    grade: str = GRADE_NATURAL
    # "minimal" (default, 1-2 plain lines low in the frame) | "bold" (a short
    # line, upper third, one keyword in `highlight` set in yellow) -- see
    # `TEXT_STYLES`. Only meaningful for the line cards (`lines`), not captions.
    text_style: str = TEXT_STYLE_MINIMAL
    # The one word/phrase to pick out in yellow when text_style="bold" --
    # matched case-insensitively against each card's own text.
    highlight: Optional[str] = None
    world: str = ""
    shots: List[Shot]
    lines: List[Line] = Field(default_factory=list)
    credit: Optional[str] = None
    music: Optional[Music] = None
    ambience: List[Ambience] = Field(default_factory=list)
    font: Optional[str] = None
    # Narration (VR1b): a voice track the reel is built around, with captions
    # (its own words, heard-timed) in place of the line cards above.
    narration: Optional[Narration] = None
    captions: Optional[Captions] = None
    dissolve_s: Optional[float] = None
    push: float = STILL_PUSH
    # VR4: the reel's lead, when a shot carries `shows_character` -- one face
    # swapped into every such shot, the same way the production pipeline
    # keeps a story's main character one person (`presentation.character`).
    character: Optional[Character] = None

    @field_validator("out_dir")
    @classmethod
    def _out_dir_abs(cls, v: str) -> str:
        if not v or not Path(v).is_absolute():
            raise ValueError("out_dir must be an absolute path")
        return v

    @field_validator("art_style")
    @classmethod
    def _style_known(cls, v: str) -> str:
        v = (v or "real").strip().lower()
        if v not in ("real", "painterly", "illustrated", "stick_world"):
            raise ValueError("art_style must be 'real', 'painterly', 'illustrated' or 'stick_world'")
        return v

    @field_validator("quality")
    @classmethod
    def _quality_known(cls, v: str) -> str:
        v = (v or QUALITY_STANDARD).strip().lower()
        if v not in QUALITIES:
            raise ValueError(f"quality must be one of {QUALITIES}, got {v!r}")
        return v

    @field_validator("grade")
    @classmethod
    def _grade_known(cls, v: str) -> str:
        v = (v or GRADE_NATURAL).strip().lower()
        if v not in GRADES:
            raise ValueError(f"grade must be one of {GRADES}, got {v!r}")
        return v

    @field_validator("text_style")
    @classmethod
    def _text_style_known(cls, v: str) -> str:
        v = (v or TEXT_STYLE_MINIMAL).strip().lower()
        if v not in TEXT_STYLES:
            raise ValueError(f"text_style must be one of {TEXT_STYLES}, got {v!r}")
        return v

    @field_validator("shots")
    @classmethod
    def _shots_count(cls, v: List[Shot]) -> List[Shot]:
        # The lines-reel cap (SHOTS_MAX) is checked in _lines_captions_music, where the
        # request's narration is known: a narrated story reel covers its whole voice track
        # (19 shots for 85 s, 2026-10-24) and was refused by the flat 1-7.
        if not (SHOTS_MIN <= len(v) <= NARRATED_SHOTS_MAX):
            raise ValueError(f"need {SHOTS_MIN}-{NARRATED_SHOTS_MAX} shots, got {len(v)}")
        return v

    @field_validator("lines")
    @classmethod
    def _lines_count(cls, v: List[Line]) -> List[Line]:
        if len(v) > LINES_MAX:
            raise ValueError(f"at most {LINES_MAX} line cards, got {len(v)}")
        return v

    @field_validator("ambience")
    @classmethod
    def _ambience_count(cls, v: List[Ambience]) -> List[Ambience]:
        if len(v) > AMBIENCE_MAX:
            raise ValueError(f"at most {AMBIENCE_MAX} ambience layers, got {len(v)}")
        return v

    @field_validator("dissolve_s")
    @classmethod
    def _dissolve_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not (DISSOLVE_S_MIN <= v <= DISSOLVE_S_MAX):
            raise ValueError(f"dissolve_s must be {DISSOLVE_S_MIN}-{DISSOLVE_S_MAX}, got {v}")
        return v

    @field_validator("push")
    @classmethod
    def _push_range(cls, v: float) -> float:
        if not (PUSH_MIN <= v <= PUSH_MAX):
            raise ValueError(f"push must be {PUSH_MIN}-{PUSH_MAX}, got {v}")
        return v

    @model_validator(mode="after")
    def _lines_captions_music(self) -> "VibeReelRequest":
        captions_on = self.narration is not None and (
            self.captions is None or self.captions.style != "off")
        if captions_on and self.lines:
            raise ValueError("lines/cards and captions may not both be used in one request")
        if self.narration is None and not self.lines:
            raise ValueError(f"need {LINES_MIN}-{LINES_MAX} line cards, got 0")
        if self.narration is None and len(self.shots) > SHOTS_MAX:
            raise ValueError(f"need {SHOTS_MIN}-{SHOTS_MAX} shots, got {len(self.shots)}")
        # Music is optional for a lines reel (2026-10-24, VR8): the creator
        # adds his own song in Instagram over the ambience, so `music: null`
        # is accepted -- the bed is then ambience only. A narrated reel could
        # already go without one (`_render_main_narrated`'s own
        # `with_music=music_idx is not None`); this just brings a lines reel
        # in line with it.
        return self

    @model_validator(mode="after")
    def _character_needed_for_shows_character(self) -> "VibeReelRequest":
        if self.character is None and any(s.shows_character for s in self.shots):
            raise ValueError("a shot has shows_character=True but the request has no character")
        return self


# --- prompts ------------------------------------------------------------------

def strip_camera_words(text: str) -> str:
    """Remove camera/lens/tripod/"shot on .."/"photo of" phrasing: a camera
    word in the prompt draws a literal camera into the frame even as a
    negative (Z-Image does not honour negatives for this)."""
    cleaned = _CAMERA_WORDS_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.-")
    return cleaned


def detect_script(text: str) -> str:
    """"devanagari" if any Devanagari codepoint is present, else "roman"."""
    return "devanagari" if _DEVANAGARI_RE.search(text or "") else "roman"


def font_for_script(script: str) -> str:
    return DEVANAGARI_FONT if script == "devanagari" else ROMAN_FONT


def _stick_shot_beat(shot: Shot, index: int, start_s: float, seconds: float, world: str) -> Beat:
    """A `kind: "stick"` shot (STICK_WORLD_PLAN.md SW2) -> a `stick_scene`
    beat, the same `data` shape `[stick: ...]` builds (presentation/
    script.py::directive_beats) -- `action`/`scene`/`era`/`outfits`/`bg`/
    `caption`, `figures` defaulting to the number of outfits named (else
    1). `bg`, when named, gets the reel's own `world` appended and its
    camera words stripped, the same way every other shot's prompt does."""
    data: Dict[str, Any] = {"action": (shot.action or "idle").strip().lower() or "idle",
                            "scene": shot.scene or ""}
    era = (shot.era or "").strip().lower() or None
    if era:
        data["era"] = era
    outfits = [o.strip().lower() for o in (shot.outfit or "").split(";") if o.strip()][:3]
    if outfits:
        data["outfits"] = outfits
    data["figures"] = len(outfits) if outfits else 1
    bg = strip_camera_words(shot.bg or "").strip()
    if bg:
        if world:
            bg = f"{bg}, {world}"
        data["bg"] = bg
    caption = (shot.caption or "").strip()[:60] or None
    if caption:
        data["caption"] = caption
    return Beat(id=f"shot{index}", start_s=start_s, end_s=start_s + seconds,
               topic="vibe reel scene", kind="stick_scene", data=data,
               text=caption or "", planned_duration_s=seconds, origin="vibe_reel")


def build_beats(req: VibeReelRequest, durations: Optional[List[float]] = None) -> List[Beat]:
    """One `broll_image` (still) or `broll_video` (clip) beat per shot, in
    order, each carrying the shared `world` so every scene reads as one
    place, and the art style's own look/motion/negative words.

    `durations` overrides each shot's `seconds` as the planned length to
    generate -- used for a narrated reel, where `seconds` is a weight and the
    real per-shot length comes from `compute_shot_durations` instead."""
    style = art_mod.get(req.art_style)
    is_real = art_mod.is_real(req.art_style)
    # Photographic realism (2026-10-23): only for a real-style reel asking for
    # it -- `REALISM_NEGATIVE` bans illustration/CGI words a painterly reel
    # wants drawn, so it is never added there.
    reel_hq = req.quality == QUALITY_REEL_HQ and is_real
    plan = durations if durations is not None else [s.seconds for s in req.shots]
    beats: List[Beat] = []
    t = 0.0
    for i, (shot, seconds) in enumerate(zip(req.shots, plan)):
        if shot.kind == "stick":
            beats.append(_stick_shot_beat(shot, i, t, seconds, req.world))
            t += seconds
            continue
        scene = strip_camera_words(shot.prompt)
        if req.world:
            scene = f"{scene}, {req.world}"
        image_prompt = f"{scene}, {style.look}" if style.look else scene
        if reel_hq:
            image_prompt = f"{image_prompt}, {REALISM_SUFFIX}"
        video_prompt = None
        kind = "broll_video" if shot.kind == "video" else "broll_image"
        if kind == "broll_video":
            motion = strip_camera_words(shot.motion or "")
            motion_bits = ", ".join(b for b in (motion, style.motion) if b)
            if reel_hq:
                motion_bits = ", ".join(b for b in (motion_bits, REALISM_MOTION_SUFFIX) if b)
            video_prompt = f"{scene}, {motion_bits}" if motion_bits else scene
            if style.look:
                video_prompt = f"{video_prompt}, {style.look}"
            if reel_hq:
                video_prompt = f"{video_prompt}, {REALISM_SUFFIX}"
        negative = ", ".join(
            b for b in ("text, watermark, logo, camera, dslr, lens, tripod", style.negative) if b)
        if reel_hq:
            negative = ", ".join(b for b in (negative, REALISM_NEGATIVE) if b)
        beats.append(Beat(
            id=f"shot{i}", start_s=t, end_s=t + seconds, topic="vibe reel scene",
            kind=kind, image_prompt=image_prompt, video_prompt=video_prompt,
            negative_prompt=negative,
            style_hint="photoreal" if is_real else "illustration",
            planned_duration_s=seconds, origin="vibe_reel",
            shows_character=shot.shows_character,
        ))
        t += seconds
    # VR4: the same step the production pipeline uses (`character.apply_to_beats`)
    # -- keeps the lead's own description in every tagged beat's prompt, and
    # clears the tag wherever there is no character to put there.
    return character_mod.apply_to_beats(beats, req.character)


def compute_shot_durations(shots: List[Shot], total_duration: float, dissolve_s: float
                           ) -> List[float]:
    """Per-shot durations when `shots[i].seconds` are WEIGHTS covering
    `total_duration` seconds end to end (a narrated reel): a video shot's
    weight is simply its planned length (kept as generated -- trimmed or held,
    never slowed, same as a lines reel); the stills share what is left in
    proportion to their own weight, each held `STILL_SECONDS_MIN_NARR`-
    `STILL_SECONDS_MAX_NARR` seconds.

    Raises ValueError with a clear "need N-M shots for Xs" message when no
    split of the stills' share can land every still in that range -- this
    runs before any generation, so a reel that cannot be covered never starts
    one."""
    n = len(shots)
    if n == 0:
        raise ValueError("no shots to cover the narration")
    target_sum = total_duration + max(0, n - 1) * dissolve_s
    video_total = sum(s.seconds for s in shots if s.kind == "video")
    stills = [s for s in shots if s.kind != "video"]
    remaining = target_sum - video_total
    if not stills:
        return [s.seconds for s in shots]
    weight_total = sum(s.seconds for s in stills)
    if weight_total <= 0:
        raise ValueError("the still shots need a positive weight to share the remaining time")
    lo = max(1, math.ceil(remaining / STILL_SECONDS_MAX_NARR)) if remaining > 0 else 1
    hi = max(lo, math.floor(remaining / STILL_SECONDS_MIN_NARR)) if remaining > 0 else lo
    durations = [s.seconds if s.kind == "video" else remaining * (s.seconds / weight_total)
                for s in shots]
    still_durations = [d for s, d in zip(shots, durations) if s.kind != "video"]
    feasible = (remaining > 0 and lo <= len(stills) <= hi
               and all(STILL_SECONDS_MIN_NARR - 1e-6 <= d <= STILL_SECONDS_MAX_NARR + 1e-6
                      for d in still_durations))
    if not feasible:
        raise ValueError(f"need {lo}-{hi} shots for {max(0.0, remaining):.1f}s")
    return durations


# --- text / ASS ----------------------------------------------------------------

def _measure_width(text: str, family: str, size_px: int) -> float:
    """Pixel width of `text` set in `family` at `size_px`, via the Windows
    font file PIL can actually load; a generous per-glyph estimate when the
    file is not there (e.g. a non-Windows test machine) so a line is shrunk
    rather than left to overflow."""
    path = _FONT_FILES.get(family)
    if path and Path(path).is_file():
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(path, size_px)
            return float(font.getlength(text))
        except Exception as e:
            logger.debug("Font measurement unavailable for %s (%s); estimating", family, e)
    return len(text) * size_px * 0.62


def _fit_font_size(text: str, family: str, base_px: int, max_width: float) -> int:
    size = base_px
    while size > MIN_CARD_FONT_PX and _measure_width(text, family, size) > max_width:
        size -= 2
    return size


def _card_font_size(content: str, family: str) -> int:
    max_width = WIDTH * MAX_CARD_WIDTH_FRAC
    longest = max(content.split("\n"), key=len, default="")
    return _fit_font_size(longest, family, CARD_FONT_PX, max_width)


def _wrap_rows(line: str, family: str, size_px: int, max_width: float) -> List[str]:
    """One line as one row when it fits, else two rows split where they come out most
    even (a sher's half-line reads as a unit; greedy wrapping left one word dangling)."""
    if _measure_width(line, family, size_px) <= max_width:
        return [line]
    words = line.split()
    if len(words) < 2:
        return [line]
    best = min(range(1, len(words)), key=lambda i: max(
        _measure_width(" ".join(words[:i]), family, size_px),
        _measure_width(" ".join(words[i:]), family, size_px)))
    left, right = " ".join(words[:best]), " ".join(words[best:])
    if max(_measure_width(left, family, size_px), _measure_width(right, family, size_px)) > max_width:
        return [left, right, ""]          # still too wide: one row too many, so the caller shrinks
    return [left, right]


def _wrap_card(content: str, family: str) -> Tuple[str, int, int]:
    """(wrapped text, font px, rows): the card's lines wrapped to the width at the full
    card size, shrinking only when the wrap would need more than MAX_CARD_ROWS rows.
    2026-10-23: a 51-character Hinglish line shrank to ~30 px on one row -- he wants
    the text bold and big, so it wraps instead."""
    max_width = WIDTH * MAX_CARD_WIDTH_FRAC
    size = CARD_FONT_PX
    while True:
        rows = [r for ln in content.split("\n") for r in _wrap_rows(ln, family, size, max_width)]
        if len(rows) <= MAX_CARD_ROWS or size <= MIN_CARD_FONT_PX:
            return "\n".join(rows), size, len(rows)
        size -= 2


def _text_style(family: str, size_px: int, pos_y: float, bold: bool = True,
                color: str = "white") -> TextStyle:
    # "bold clean Latin sans... pure white, 3-4px dark outline + soft shadow"
    # (2026-10-23 night rules, VR7) -- bold/outlined by default; the credit
    # line asks for plain (bold=False) the way it always has.
    return TextStyle(font_family=family, font_size=size_px, color=color, align="center",
                     pos_x=0.0, pos_y=pos_y, bold=bold, stroke_width=3, stroke_color="black",
                     shadow_x=2, shadow_y=2, shadow_color="black@0.55",
                     box=False, animation="fade", animation_duration=CARD_FADE_S)


def _card_pos_y(text_style: str) -> float:
    """lower middle (~64% height, "minimal") or the upper third (~33%,
    "bold") -- TextStyle's -1..1 scale."""
    return CARD_POS_Y_BOLD if text_style == TEXT_STYLE_BOLD else CARD_POS_Y_MINIMAL


def _find_highlight_span(text: str, highlight: Optional[str]) -> Optional[Tuple[int, int]]:
    """The (start, end) character span of `highlight` in `text`, matched as a
    whole word, case-insensitively -- None when there is nothing to pick out."""
    if not highlight or not highlight.strip():
        return None
    pattern = re.compile(r"(?<!\w)" + re.escape(highlight.strip()) + r"(?!\w)", re.IGNORECASE)
    match = pattern.search(text)
    return (match.start(), match.end()) if match else None


def _highlight_clips(text: str, highlight: Optional[str], family: str, size_px: int,
                     pos_y: float) -> List[TextClip]:
    """One centred TextClip (white) when there is nothing to highlight in
    `text`, else three adjacent left-aligned TextClips (before/word/after,
    the word in `HIGHLIGHT_COLOR`) laid out by measured pixel width so they
    read as one line, same font and position. `render.ass`/`timeline.schema`
    have no partial-text colour of their own outside the karaoke animation
    (a different thing, built for word-by-word timing) -- this stays inside
    vibe_reel.py rather than touching that shared renderer."""
    span = _find_highlight_span(text, highlight)
    if span is None:
        return [TextClip(content=text, style=_text_style(family, size_px, pos_y))]
    before, word, after = text[:span[0]], text[span[0]:span[1]], text[span[1]:]
    total_w = _measure_width(text, family, size_px)
    x = WIDTH / 2.0 - total_w / 2.0
    clips: List[TextClip] = []
    for part, color in ((before, "white"), (word, HIGHLIGHT_COLOR), (after, "white")):
        if not part:
            continue
        w = _measure_width(part, family, size_px)
        pos_x = max(-1.0, min(1.0, (x / WIDTH) * 2.0 - 1.0))
        style = _text_style(family, size_px, pos_y, color=color)
        style.align = "left"
        style.pos_x = pos_x
        clips.append(TextClip(content=part, style=style))
        x += w
    return clips


def build_text_clips(lines: List[Line], credit: Optional[str],
                     font_override: Optional[str] = None,
                     text_style: str = TEXT_STYLE_MINIMAL,
                     highlight: Optional[str] = None
                     ) -> List[Tuple[TextClip, float, float]]:
    """(TextClip, start_s, end_s) for every line card, plus the credit (if
    any) under the last card for that card's span. `text_style` (2026-10-23
    night rules, VR7): "minimal" (default) keeps the plain lower-middle
    card; "bold" moves it to the upper third and picks `highlight` out in
    yellow on every card that contains it."""
    pos_y = _card_pos_y(text_style)
    highlight_word = highlight if text_style == TEXT_STYLE_BOLD else None
    entries: List[Tuple[TextClip, float, float]] = []
    for ln in lines:
        script = ln.script or detect_script(ln.text)
        family = font_override or font_for_script(script)
        if highlight_word:
            text, size, rows = ln.text, _card_font_size(ln.text, family), ln.text.count("\n") + 1
        else:
            text, size, rows = _wrap_card(ln.text, family)
        for clip in _highlight_clips(text, highlight_word, family, size, pos_y):
            entries.append((clip, ln.start_s, ln.end_s))
        last_block = (pos_y, size, rows)
    if credit and lines:
        last = lines[-1]
        script = last.script or detect_script(credit)
        family = font_override or font_for_script(script)
        size = _fit_font_size(credit, family, CREDIT_FONT_PX, WIDTH * MAX_CARD_WIDTH_FRAC)
        # just under the last card's block, however many rows it wrapped to
        block_y, block_px, block_rows = last_block
        credit_y = block_y + (block_rows * block_px * CARD_LINE_SPACING / 2 + size) / (HEIGHT / 2)
        clip = TextClip(content=credit, style=_text_style(family, size, credit_y, bold=False))
        entries.append((clip, last.start_s, last.end_s))
    return entries


def build_ass_script(lines: List[Line], credit: Optional[str],
                     font_override: Optional[str] = None,
                     text_style: str = TEXT_STYLE_MINIMAL,
                     highlight: Optional[str] = None) -> str:
    entries = build_text_clips(lines, credit, font_override, text_style, highlight)
    return ass_mod.build_ass(entries, WIDTH, HEIGHT)


# --- the scrim: a soft dark band behind the text, scaled by brightness ----------

def scrim_filter(pos_y: float, alpha: float, band_frac: float = SCRIM_BAND_FRAC) -> str:
    """A horizontal Gaussian darkening of luma only (never a hard box),
    centred on the text's own Y (`pos_y`, TextStyle's -1..1 scale), peak
    strength `alpha` (2026-10-23 night rules, VR7)."""
    y_frac = max(0.0, min(1.0, (pos_y + 1.0) / 2.0))
    return (f"geq=lum='lum(X,Y)*(1-{alpha:.3f}*exp(-((Y-({y_frac:.4f}*H))^2)/"
           f"(2*({band_frac:.3f}*H)^2)))':cb='cb(X,Y)':cr='cr(X,Y)'")


def _mean_luma_band(path: Path, at_s: float, pos_y: float, band_frac: float = SCRIM_BAND_FRAC
                    ) -> Optional[float]:
    """0..255 mean brightness of the horizontal band the text sits over, in
    one frame of `path` at `at_s` -- same extraction as `mean_luma`, cropped
    to the band before averaging."""
    import tempfile
    try:
        duration = get_video_duration(str(path))
    except Exception:
        duration = 0.0
    seek = max(0.0, min(at_s, duration)) if duration else max(0.0, at_s)
    with tempfile.TemporaryDirectory(prefix="vibe_scrim_") as tmp:
        frame = Path(tmp) / "frame.png"
        cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
              "-ss", f"{seek:.3f}", "-i", str(path), "-frames:v", "1", str(frame)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, creationflags=NO_WINDOW)
        except Exception as e:
            logger.warning("Could not extract a frame from %s for the scrim (%s)", path, e)
            return None
        if not frame.is_file():
            return None
        try:
            from PIL import Image, ImageStat
            with Image.open(frame) as img:
                y_frac = max(0.0, min(1.0, (pos_y + 1.0) / 2.0))
                cy = y_frac * img.height
                half = max(1.0, band_frac * img.height)
                top, bottom = max(0, int(cy - half)), min(img.height, int(cy + half))
                if bottom <= top:
                    top, bottom = 0, img.height
                band = img.crop((0, top, img.width, bottom)).convert("L")
                return float(ImageStat.Stat(band).mean[0])
        except Exception as e:
            logger.warning("Could not read the extracted frame for the scrim (%s)", e)
            return None


def _scrim_alpha_for(luma: Optional[float]) -> float:
    """Brighter behind the text -> a stronger scrim, between `SCRIM_ALPHA_MIN`
    (a dark background) and `SCRIM_ALPHA_MAX` (a bright one); the mid-point
    when brightness could not be measured at all."""
    if luma is None:
        return (SCRIM_ALPHA_MIN + SCRIM_ALPHA_MAX) / 2.0
    frac = max(0.0, min(1.0, luma / 255.0))
    return SCRIM_ALPHA_MIN + frac * (SCRIM_ALPHA_MAX - SCRIM_ALPHA_MIN)


# --- captions (VR1b) ------------------------------------------------------------

def _caption_text_style(family: str, size_px: int) -> TextStyle:
    return TextStyle(font_family=family, font_size=size_px, color="white", align="center",
                     pos_x=0.0, pos_y=CAPTION_POS_Y, bold=True, stroke_width=4,
                     stroke_color="black", shadow_x=2, shadow_y=2, shadow_color="black@0.5",
                     box=False, animation="fade", animation_duration=CAPTION_FADE_S)


def _caption_font_size(text: str, family: str) -> int:
    max_width = WIDTH * CAPTION_WIDTH_FRAC
    size = CAPTION_FONT_PX
    while size > CAPTION_MIN_FONT_PX and _measure_width(text, family, size) > max_width:
        size -= 2
    return size


def build_caption_clips(phrases: List[Dict[str, Any]], font_override: Optional[str] = None
                        ) -> List[Tuple[TextClip, float, float]]:
    """(TextClip, start_s, end_s) for each phrase from
    `caption_align.build_caption_phrases`, bold and centred at ~70% down the
    frame, in the script's own Devanagari/Roman font."""
    entries: List[Tuple[TextClip, float, float]] = []
    for phrase in phrases:
        text = phrase["text"]
        script = detect_script(text)
        family = font_override or font_for_script(script)
        size = _caption_font_size(text, family)
        clip = TextClip(content=text, style=_caption_text_style(family, size))
        entries.append((clip, phrase["start"], phrase["end"]))
    return entries


def build_caption_ass(phrases: List[Dict[str, Any]], font_override: Optional[str] = None) -> str:
    entries = build_caption_clips(phrases, font_override)
    return ass_mod.build_ass(entries, WIDTH, HEIGHT)


# --- Hinglish captions (2026-10-23 night rules, VR7) -----------------------------

TRANSLITERATE_SCHEMA = {
    "name": "transliteration",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
        "required": ["lines"],
        "additionalProperties": False,
    },
}


def _parse_transliteration(content: Optional[str]) -> Optional[List[str]]:
    if not content:
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None
    lines = data.get("lines") if isinstance(data, dict) else None
    return lines if isinstance(lines, list) else None


async def transliterate_to_hinglish(lines: List[str]) -> Tuple[List[str], bool]:
    """Devanagari caption phrases -> natural, casual Hinglish (Roman letters),
    one local-LLM call (same count in, same count out, every line non-empty,
    no Devanagari left); one retry; on failure the given lines come back
    unchanged and `False` records the fallback (2026-10-23 night rules,
    VR7). A line that is already Roman is left exactly as it is -- nothing
    to call the model for."""
    if not lines or not any(_DEVANAGARI_RE.search(t) for t in lines):
        return lines, True
    from llm.client import lm_studio_client
    system_prompt = (
        "Transliterate each given Hindi/Urdu line (Devanagari) into natural, "
        "casual Hinglish -- the same words, in Roman letters, the way a young "
        "Indian would text them. Do not translate the meaning, do not add or "
        "drop lines, keep the same order. Reply with ONLY one JSON object: "
        "{\"lines\": [<one Hinglish string per input line>]}.")
    user_prompt = json.dumps({"lines": lines}, ensure_ascii=False)
    for _attempt in range(2):
        try:
            content = await lm_studio_client.ask_with_schema(
                system_prompt, user_prompt, TRANSLITERATE_SCHEMA)
        except Exception as e:
            logger.warning("Transliteration call failed (%s)", e)
            content = None
        out = _parse_transliteration(content)
        if (isinstance(out, list) and len(out) == len(lines)
                and all(isinstance(t, str) and t.strip() for t in out)
                and not any(_DEVANAGARI_RE.search(t) for t in out)):
            return [t.strip() for t in out], True
    logger.warning("Transliteration unusable after retry; keeping Devanagari for %d line(s)",
                   len(lines))
    return lines, False


# --- video filter graph ---------------------------------------------------------

def _scale_crop_body() -> str:
    return (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},setsar=1")


def video_shot_filter(index: int, seconds: float, pad_s: float = 0.0, stepped: bool = False) -> str:
    """A generated clip: scaled/cropped to the canvas, a held last frame if it
    fell short by up to `SHOT_PAD_TOLERANCE_S`, trimmed to exactly `seconds`.
    No zoom, no pan -- the art style's own `motion` is the only move.

    `stepped` (illustrated art style, 2026-10-24 VR8): the clip's motion is
    resampled down to `ANIMATION_HOLD_FPS` and back up to `FPS` -- each
    drawing held for several output frames ("on twos") instead of smooth
    video motion."""
    fps_chain = f"fps={ANIMATION_HOLD_FPS},fps={FPS}" if stepped else f"fps={FPS}"
    chain = f"{fps_chain},{_scale_crop_body()}"
    if pad_s > 0:
        chain += f",tpad=stop_mode=clone:stop_duration={pad_s:.3f}"
    chain += f",trim=duration={seconds:.3f},setpts=PTS-STARTPTS"
    return f"[{index}:v]{chain}[v{index}]"


def still_shot_filter(index: int, seconds: float, push: float = STILL_PUSH) -> str:
    """A still: scaled/cropped to the canvas, then a very slow, eased,
    centred push of about `push` (4%) over the whole shot -- the only motion
    a still gets."""
    frames = max(1, round(seconds * FPS))
    ease = f"(0.5-0.5*cos(PI*min(on/{max(1, frames - 1)}\\,1)))"
    zoom = f"'1+{push}*{ease}'"
    chain = (f"{_scale_crop_body()},"
            f"zoompan=z={zoom}:x='iw/2-(iw/zoom)/2':y='ih/2-(ih/zoom)/2':"
            f"d=1:s={WIDTH}x{HEIGHT}:fps={FPS},"
            f"trim=duration={seconds:.3f},setpts=PTS-STARTPTS")
    return f"[{index}:v]{chain}[v{index}]"


def build_video_graph(shot_durations: List[float], shot_kinds: List[str],
                      shot_pads: List[float], ass_path: Path,
                      dissolve_s: float = DISSOLVE_S, push: float = STILL_PUSH,
                      look_pass: bool = False, grade: str = GRADE_NATURAL,
                      scrim: Optional[Tuple[float, float]] = None,
                      art_style: str = "real"
                      ) -> Tuple[str, float]:
    """The filter_complex for the whole video: per-shot scale/trim(/push), a
    dissolve between each pair (`render.transitions`), a fade from/to black,
    then the reel_hq look pass when `look_pass` -- BEFORE the text, so the
    burned-in words stay clean -- then the scrim (`scrim_filter`,
    `(pos_y, alpha)`) when there is text to sit behind, then the lines (or
    captions) burned in through libass. Returns (filter_complex, total_duration_s).

    The look pass itself is style-aware (2026-10-24, VR8): `art_style ==
    "illustrated"` gets `ILLUSTRATED_LOOK_PASS_FILTER` (a static paper grain
    and a touch of colour flattening, no handheld drift) and every video
    shot is stepped to `ANIMATION_HOLD_FPS` ("on twos") in `video_shot_filter`;
    any other style keeps the real-style pass (`LOOK_PASS_FILTER` +
    `HANDHELD_DRIFT_FILTER`), as before. `grade="bw"` adds a full
    desaturation on top either way."""
    if not shot_durations:
        raise VibeReelError("no shots to render")
    # stick_world's pictures are the illustrated look too (STICK_WORLD_PLAN.md).
    stepped = look_pass and art_style in ("illustrated", "stick_world")
    parts: List[str] = []
    for i, (seconds, kind) in enumerate(zip(shot_durations, shot_kinds)):
        pad = shot_pads[i] if i < len(shot_pads) else 0.0
        parts.append(still_shot_filter(i, seconds, push=push) if kind == "still"
                    else video_shot_filter(i, seconds, pad, stepped=stepped))
    cur_label = "v0"
    running = shot_durations[0]
    for i in range(1, len(shot_durations)):
        d = shot_durations[i]
        out_label = f"x{i}"
        dur = xfade_clamp(dissolve_s, running, d)
        parts.append(xfade_video(f"[{cur_label}]", f"[v{i}]", f"[{out_label}]", "fade", dur, running))
        running = running + d - dur
        cur_label = out_label
    total = running
    fade_out_start = max(0.0, total - FADE_OUT_S)
    parts.append(
        f"[{cur_label}]fade=t=in:st=0:d={FADE_IN_S:.3f},"
        f"fade=t=out:st={fade_out_start:.3f}:d={FADE_OUT_S:.3f}[faded]")
    graded_label = "faded"
    if look_pass:
        chain = (ILLUSTRATED_LOOK_PASS_FILTER if art_style in ("illustrated", "stick_world")
                 else f"{HANDHELD_DRIFT_FILTER},{LOOK_PASS_FILTER}")
        if grade == GRADE_BW:
            chain += f",{BW_FILTER}"
        parts.append(f"[faded]{chain}[graded]")
        graded_label = "graded"
    if scrim is not None:
        pos_y, alpha = scrim
        parts.append(f"[{graded_label}]{scrim_filter(pos_y, alpha)}[scrimmed]")
        graded_label = "scrimmed"
    ass_filter = ass_mod.build_ass_filter(ass_path)
    parts.append(f"[{graded_label}]{ass_filter}[vout]")
    return ";".join(parts), total


# --- audio filter graph ---------------------------------------------------------

def build_audio_filter(music_idx: Optional[int], ambience: List[Tuple[int, float]],
                       total_duration: float, with_music: bool) -> str:
    """The bed: the music track (when `with_music`) plus 0-2 ambience layers
    at their own gain, mixed, faded in/out, loudness-normalised. `ambience`
    is [(ffmpeg input index, gain_db), ...]."""
    parts: List[str] = []
    labels: List[str] = []
    if with_music and music_idx is not None:
        parts.append(f"[{music_idx}:a]atrim=duration={total_duration:.3f},"
                    f"asetpts=PTS-STARTPTS[mus]")
        labels.append("[mus]")
    for n, (idx, gain_db) in enumerate(ambience):
        label = f"amb{n}"
        parts.append(f"[{idx}:a]atrim=duration={total_duration:.3f},asetpts=PTS-STARTPTS,"
                    f"volume={gain_db:.1f}dB[{label}]")
        labels.append(f"[{label}]")
    if len(labels) > 1:
        mix_label = "mix"
        parts.append("".join(labels)
                    + f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0[{mix_label}]")
    elif labels:
        mix_label = labels[0].strip("[]")
    else:
        parts.append(f"anullsrc=r=48000:cl=stereo:d={total_duration:.3f}[mix]")
        mix_label = "mix"
    fade_out_start = max(0.0, total_duration - AUDIO_FADE_OUT_S)
    parts.append(
        f"[{mix_label}]afade=t=in:st=0:d={AUDIO_FADE_IN_S:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={AUDIO_FADE_OUT_S:.3f},"
        f"loudnorm=I={LOUDNESS_LUFS:.1f}:TP={TRUE_PEAK_DBTP:.1f}:LRA=11,aresample=48000[aout]")
    return ";".join(parts)


def build_narration_audio_filter(narration_idx: int, music_idx: Optional[int],
                                 ambience: List[Tuple[int, float]], total_duration: float,
                                 lead_s: float, with_music: bool) -> str:
    """The bed for a narrated reel: the narration at full level (delayed by
    `lead_s`, held to `total_duration`), a music track (when `with_music`)
    fixed `NARRATION_MUSIC_GAIN_DB` under it, and 0-2 ambience layers at their
    own gain -- none of them ducked any further once set. `ambience` is
    [(ffmpeg input index, gain_db), ...]."""
    parts: List[str] = []
    bed_labels: List[str] = []
    if with_music and music_idx is not None:
        parts.append(f"[{music_idx}:a]atrim=duration={total_duration:.3f},"
                    f"asetpts=PTS-STARTPTS,volume={NARRATION_MUSIC_GAIN_DB:.1f}dB[mus]")
        bed_labels.append("[mus]")
    for n, (idx, gain_db) in enumerate(ambience):
        label = f"amb{n}"
        parts.append(f"[{idx}:a]atrim=duration={total_duration:.3f},asetpts=PTS-STARTPTS,"
                    f"volume={gain_db:.1f}dB[{label}]")
        bed_labels.append(f"[{label}]")
    bed_label: Optional[str] = None
    if len(bed_labels) > 1:
        parts.append("".join(bed_labels)
                    + f"amix=inputs={len(bed_labels)}:duration=longest:dropout_transition=0[bed]")
        bed_label = "bed"
    elif bed_labels:
        bed_label = bed_labels[0].strip("[]")

    delay_ms = int(round(max(0.0, lead_s) * 1000))
    parts.append(f"[{narration_idx}:a]adelay=delays={delay_ms}:all=1,"
                f"apad=whole_dur={total_duration:.3f},atrim=duration={total_duration:.3f},"
                f"asetpts=PTS-STARTPTS[narr]")

    if bed_label:
        fade_out_start = max(0.0, total_duration - AUDIO_FADE_OUT_S)
        parts.append(f"[{bed_label}]afade=t=in:st=0:d={AUDIO_FADE_IN_S:.3f},"
                    f"afade=t=out:st={fade_out_start:.3f}:d={AUDIO_FADE_OUT_S:.3f}[bedf]")
        parts.append("[narr][bedf]amix=inputs=2:duration=longest:dropout_transition=0[mixed]")
        mix_label = "mixed"
    else:
        mix_label = "narr"
    parts.append(f"[{mix_label}]loudnorm=I={LOUDNESS_LUFS:.1f}:TP={TRUE_PEAK_DBTP:.1f}:LRA=11,"
                f"aresample=48000[aout]")
    return ";".join(parts)


# --- shot generation -------------------------------------------------------------

def shot_duration_action(actual_s: float, wanted_s: float) -> str:
    """'trim' (the clip covers its slot), 'pad' (short by no more than
    `SHOT_PAD_TOLERANCE_S` -- hold the last frame), or 'retry' (too short)."""
    if actual_s + 0.02 >= wanted_s:
        return "trim"
    if wanted_s - actual_s <= SHOT_PAD_TOLERANCE_S:
        return "pad"
    return "retry"


def mean_luma(path: Path, at_s: Optional[float] = None) -> Optional[float]:
    """0..255 mean brightness of one frame (the midpoint by default) -- a
    clip stuck on a dark/placeholder frame reads near 0. None if a frame
    could not be read at all (never treated as black)."""
    import tempfile
    try:
        duration = get_video_duration(str(path))
    except (FFmpegError, Exception):
        duration = 0.0
    seek = at_s if at_s is not None else max(0.0, duration / 2.0)
    with tempfile.TemporaryDirectory(prefix="vibe_luma_") as tmp:
        frame = Path(tmp) / "frame.png"
        cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
              "-ss", f"{seek:.3f}", "-i", str(path), "-frames:v", "1", str(frame)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, creationflags=NO_WINDOW)
        except Exception as e:
            logger.warning("Could not extract a frame from %s for brightness (%s)", path, e)
            return None
        if not frame.is_file():
            return None
        try:
            from PIL import Image, ImageStat
            with Image.open(frame) as img:
                return float(ImageStat.Stat(img.convert("L")).mean[0])
        except Exception as e:
            logger.warning("Could not read the extracted frame for %s (%s)", path, e)
            return None


async def _generate_shot(beat: Beat, project_dir: Path, settings: PresentationSettings,
                         seed_base: int, character: Optional[Character] = None,
                         character_cache_dir: Optional[Path] = None,
                         workflow_files: Optional[Dict[str, List[str]]] = None,
                         gen_size_override: Optional[Dict[str, Tuple[int, int]]] = None,
                         start_image_override: Optional[str] = None
                         ) -> Tuple[Optional[Path], float, float, str, str]:
    """(path, actual_duration_s, pad_s, reason, workflow_file) for one shot,
    retried up to `MAX_SHOT_ATTEMPTS` times against ComfyUI failure, a
    too-short clip, or a near-black frame. path is None once every attempt
    has failed; workflow_file is "" in that case too.

    `character_cache_dir`, when the beat `shows_character`, is the one
    directory every such shot points `generate_assets` at for its reference
    portrait -- this call's own `project_dir` stays the per-shot cache
    directory for the picture itself (VR1's near-match fix).

    `workflow_files`/`gen_size_override` (quality="reel_hq") name the graph(s)
    and generation size to use for this shot only -- see `generate_assets`'
    own parameters of the same name.

    `start_image_override` (2026-10-23 night rules, VR7 chaining): the exact
    image-to-video start frame to use instead of a freshly generated still --
    one chain segment continuing from the previous segment's own last frame.
    Left out of the call entirely when None (the default, every ordinary
    shot), so an older caller/mock of `generate_assets` is untouched."""
    wanted = beat.planned_duration_s or beat.duration_s
    last_reason = "generation failed"
    extra: Dict[str, Any] = ({"start_image_override": start_image_override}
                             if start_image_override is not None else {})
    for attempt in range(MAX_SHOT_ATTEMPTS):
        try:
            made, failures = await assets_mod.generate_assets(
                [beat], project_dir, settings, seed_base=seed_base + attempt * 97,
                canvas_size=(WIDTH, HEIGHT), character=character,
                character_cache_dir=character_cache_dir,
                workflow_files=workflow_files, gen_size_override=gen_size_override,
                **extra)
        except Exception as e:
            last_reason = str(e)[:200]
            continue
        if failures or not made:
            last_reason = failures[0]["reason"] if failures else "no asset produced"
            continue
        path = Path(made[0].path)
        workflow_file = made[0].workflow_file or ""
        if not path.is_file():
            last_reason = "generated file is missing on disk"
            continue
        if beat.kind == "broll_video":
            try:
                actual = get_video_duration(str(path))
            except Exception as e:
                last_reason = f"could not read the clip's length ({e})"
                continue
            action = shot_duration_action(actual, wanted)
            if action == "retry":
                last_reason = f"clip too short ({actual:.2f}s of {wanted:.2f}s)"
                continue
            pad = max(0.0, wanted - actual) if action == "pad" else 0.0
        else:
            actual, pad = wanted, 0.0
        luma = mean_luma(path)
        if luma is not None and luma <= BLACK_LUMA_MAX:
            last_reason = f"near-black frame (mean luma {luma:.1f})"
            continue
        return path, actual, pad, "", workflow_file
    return None, 0.0, 0.0, last_reason, ""


async def _generate_stick_shot(beat: Beat, project_dir: Path, settings: PresentationSettings,
                               seed_base: int) -> Tuple[Optional[Path], float, float, str, str]:
    """A `kind: "stick"` shot (STICK_WORLD_PLAN.md SW2): its own background
    picture, when `bg` named one, is made exactly like an ordinary still
    shot (`_generate_shot` on a one-off `broll_image` beat, same retries);
    the figures are then drawn over it -- or over today's plain ground, with
    no picture at all -- by `presentation.motion`'s own StickScene builder,
    at this reel's own size. Returns (path, duration_s, pad_s, reason,
    "still"|"video") -- "video" only when the motion engine actually
    rendered; the picture itself (or `motion_stick.render_stick_still` when
    there is none) is the fallback, placed as a still shot like any other
    scene -- never failing the whole reel."""
    from . import motion as motion_stage
    from . import motion_stick as motion_stick_stage

    wanted_s = beat.planned_duration_s or beat.duration_s
    bg_prompt = motion_stick_stage.compose_bg_prompt(beat.data or {})
    if bg_prompt:
        style = art_mod.get(settings.art_style)
        full_prompt = f"{bg_prompt}, {style.look}" if style.look else bg_prompt
        bg_beat = Beat(id=f"{beat.id}_bg", start_s=beat.start_s, end_s=beat.start_s,
                       kind="broll_image", image_prompt=full_prompt)
        bg_path, _actual, _pad, bg_reason, _wf = await _generate_shot(
            bg_beat, project_dir, settings, seed_base=seed_base)
        if bg_path is not None:
            beat.data["bg_image"] = str(bg_path)
            beat.data["bg_luma"] = motion_stick_stage.measure_bg_luma(str(bg_path))
        else:
            logger.info("Stick shot %s: background picture failed (%s); drawing the plain ground",
                       beat.id, bg_reason)

    assets, failures, handled = await asyncio.to_thread(
        motion_stage.render_motion_assets, [beat], project_dir, WIDTH, HEIGHT, "general", settings)
    if beat.id in handled and assets:
        asset = assets[0]
        return Path(asset.path), asset.duration_s, 0.0, "", "video"
    fallback = motion_stage.render_fallback(beat, project_dir, WIDTH, HEIGHT)
    if fallback is not None and Path(fallback.path).is_file():
        return Path(fallback.path), wanted_s, 0.0, "", "still"
    reason = failures[0]["reason"] if failures else "stick scene could not be made"
    return None, 0.0, 0.0, reason, "still"


# --- continuous shots: chaining several image-to-video segments (VR7) -----------

def _plan_chain_segments(wanted_s: float) -> List[float]:
    """Split a shot's seconds into continuous image-to-video segments, each
    at most `CHAIN_SEGMENT_SECONDS_MAX` (one generated clip's own practical
    ceiling) -- a single-segment list (no chaining) when the shot already
    fits one clip."""
    if wanted_s <= CHAIN_SEGMENT_SECONDS_MAX + SHOT_PAD_TOLERANCE_S:
        return [wanted_s]
    n = min(CHAIN_SEGMENTS_MAX, max(2, math.ceil(wanted_s / CHAIN_SEGMENT_SECONDS_MAX)))
    return [wanted_s / n] * n


def _extract_last_frame(video_path: Path, out_path: Path) -> bool:
    """The last frame of a generated clip, as a PNG -- the next chain
    segment's image-to-video start, so its motion continues from here.

    `-sseof` (seek relative to end-of-file) rather than a forward `-ss` close
    to the read duration: a clip's real frame interval (its own fps, whatever
    that turns out to be) can easily be wider than a small margin before the
    duration, landing the seek past the last decodable frame and producing
    nothing at all (rc=0, no file). Seeking a few seconds before the end and
    decoding everything after it onto the same file (`-update 1`, no
    `-frames:v`) always leaves the actual last frame as the final write."""
    try:
        duration = get_video_duration(str(video_path))
    except Exception as e:
        logger.warning("Could not read %s's length to pull its last frame (%s)", video_path, e)
        return False
    back = min(3.0, max(0.2, duration))
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
          "-sseof", f"-{back:.3f}", "-i", str(video_path), "-update", "1", str(out_path)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, creationflags=NO_WINDOW)
    except Exception as e:
        logger.warning("Could not extract the last frame of %s (%s)", video_path, e)
        return False
    return out_path.is_file()


async def _join_chain_segments(paths: List[Path], durations: List[float], out_path: Path) -> float:
    """Join chained segments end to end with a `CHAIN_SEAM_S` (about 2
    frames) crossfade at each seam -- "joined with no dissolve": one shot's
    motion continuing, not a transition between scenes. Returns the joined
    clip's total duration."""
    if len(paths) == 1:
        return durations[0]
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", str(p)]
    parts: List[str] = []
    cur_label = "0:v"
    running = durations[0]
    for i in range(1, len(paths)):
        d = durations[i]
        out_label = f"c{i}"
        seam = max(0.02, min(CHAIN_SEAM_S, running - 0.05, d - 0.05))
        parts.append(xfade_video(f"[{cur_label}]", f"[{i}:v]", f"[{out_label}]", "fade", seam, running))
        running = running + d - seam
        cur_label = out_label
    parts.append(f"[{cur_label}]null[vout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]"]
    cmd += get_encoder_flags(prefer_nvenc=False)
    cmd += [str(out_path)]
    await _run_ffmpeg(cmd, timeout=600)
    return running


async def _generate_chained_shot(beat: Beat, project_dir: Path, settings: PresentationSettings,
                                 seed_base: int, character: Optional[Character] = None,
                                 character_cache_dir: Optional[Path] = None,
                                 workflow_files: Optional[Dict[str, List[str]]] = None,
                                 gen_size_override: Optional[Dict[str, Tuple[int, int]]] = None
                                 ) -> Tuple[Optional[Path], float, float, str, str, int]:
    """A shot wanting more seconds than one generated clip practically covers
    (`CHAIN_SEGMENT_SECONDS_MAX`) is made of several image-to-video segments
    chained end to end: every segment after the first starts from the one
    before it's own last frame (same prompt, via `_generate_shot`'s
    `start_image_override`), then they are joined with a tiny crossfade --
    one continuous shot, not a cut (2026-10-23 night rules, VR7 rule B).

    Falls back to the segments made so far when a LATER segment cannot be
    made (held/joined as one shorter shot); the whole shot fails only when
    the first segment does -- the same "never deliver a reel with a missing
    scene" rule `render_vibe_reel` already enforces. Returns (path,
    actual_duration_s, pad_s, reason, workflow_file, segments_made)."""
    wanted = beat.planned_duration_s or beat.duration_s
    plan = _plan_chain_segments(wanted)
    if len(plan) == 1:
        path, actual, pad, reason, workflow_file = await _generate_shot(
            beat, project_dir, settings, seed_base=seed_base, character=character,
            character_cache_dir=character_cache_dir, workflow_files=workflow_files,
            gen_size_override=gen_size_override)
        return path, actual, pad, reason, workflow_file, 1

    chain_dir = project_dir / "chain"
    chain_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: List[Path] = []
    seg_durations: List[float] = []
    workflow_file = ""
    start_image: Optional[str] = None
    for i, seg_s in enumerate(plan):
        seg_beat = beat.model_copy(update={"planned_duration_s": seg_s,
                                           "end_s": beat.start_s + seg_s})
        seg_dir = chain_dir / f"seg{i}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        path, actual, pad, reason, wf = await _generate_shot(
            seg_beat, seg_dir, settings, seed_base=seed_base + i * 211,
            character=character, character_cache_dir=character_cache_dir,
            workflow_files=workflow_files, gen_size_override=gen_size_override,
            start_image_override=start_image)
        if path is None:
            if i == 0:
                return None, 0.0, 0.0, f"first chain segment failed: {reason}", "", 0
            logger.warning("Chain segment %d of %s failed (%s); keeping the %d segment(s) made",
                           i, beat.id, reason, len(seg_paths))
            break
        seg_paths.append(path)
        seg_durations.append(actual + pad)
        workflow_file = wf or workflow_file
        if i < len(plan) - 1:
            frame_path = chain_dir / f"seg{i}_last.png"
            if not _extract_last_frame(path, frame_path):
                logger.warning("Could not pull a last frame from chain segment %d of %s; "
                               "stopping the chain at %d segment(s)", i, beat.id, len(seg_paths))
                break
            start_image = str(frame_path)

    if not seg_paths:
        return None, 0.0, 0.0, "no chain segment could be made", "", 0
    if len(seg_paths) == 1:
        return seg_paths[0], seg_durations[0], 0.0, "", workflow_file, 1

    joined_path = chain_dir / "joined.mp4"
    total = await _join_chain_segments(seg_paths, seg_durations, joined_path)
    shortfall = wanted - total
    pad = max(0.0, shortfall) if 0.0 < shortfall <= SHOT_PAD_TOLERANCE_S * len(seg_paths) else 0.0
    return joined_path, total, pad, "", workflow_file, len(seg_paths)


# --- music / ambience ------------------------------------------------------------

async def _resolve_music(music: Music, total_seconds: float) -> Tuple[Path, str]:
    if music.path:
        path = Path(music.path)
        if not path.is_file():
            raise VibeReelError(f"music path does not exist: {music.path}")
        return path, str(path)
    result = await music_gen.generate_variations(
        style=None, count=1, seconds=max(8.0, total_seconds), extra_tags=music.brief or "",
        bpm=music.bpm, engine=music_gen.DEFAULT_ENGINE)
    made = result.get("made") or []
    if not made:
        raise VibeReelError("music generation made nothing usable: "
                            + "; ".join(result.get("errors") or ["unknown error"]))
    return music_gen.MUSIC_DIR / made[0], made[0]


async def _resolve_ambience(items: List[Ambience], total_seconds: float
                            ) -> List[Tuple[Path, float]]:
    out: List[Tuple[Path, float]] = []
    for i, item in enumerate(items):
        path, _score, _source = sfx_library.resolve_effect(
            item.prompt, random.Random(1700 + i), kind="bed")
        if path is None:
            try:
                path = await sfx_library.generated_effect(
                    item.prompt, min(47.0, max(4.0, total_seconds)), seed=1700 + i)
            except Exception as e:
                logger.warning("Ambience %r could not be generated (%s)", item.prompt, e)
                path = None
        if path is None:
            logger.warning("No ambience found or generated for %r; leaving it out", item.prompt)
            continue
        out.append((Path(path), item.gain_db))
    return out


# --- rendering ---------------------------------------------------------------------

async def _run_ffmpeg(cmd: List[str], timeout: int = 1800) -> None:
    def _run():
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=NO_WINDOW)
    result = await asyncio.to_thread(_run)
    if result.returncode != 0:
        raise VibeReelError(f"ffmpeg failed ({result.returncode}): {result.stderr[-1500:]}")


async def _render_main(shot_paths: List[Path], shot_durations: List[float],
                       shot_kinds: List[str], shot_pads: List[float], ass_path: Path,
                       music_path: Optional[Path], ambience: List[Tuple[Path, float]],
                       out_path: Path, dissolve_s: float = DISSOLVE_S,
                       push: float = STILL_PUSH, look_pass: bool = False,
                       grade: str = GRADE_NATURAL,
                       scrim: Optional[Tuple[float, float]] = None,
                       art_style: str = "real") -> float:
    """`music_path` is None for a lines reel with no `music` in the request
    (2026-10-24, VR8: the creator adds his own Instagram song over the
    ambience) -- the bed is then ambience only, same loudness target, and no
    music input is added to the ffmpeg command at all."""
    video_graph, total = build_video_graph(shot_durations, shot_kinds, shot_pads, ass_path,
                                           dissolve_s=dissolve_s, push=push, look_pass=look_pass,
                                           grade=grade, scrim=scrim, art_style=art_style)
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    for path, seconds, kind in zip(shot_paths, shot_durations, shot_kinds):
        if kind == "still":
            cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds + 0.5:.3f}", "-i", str(path)]
        else:
            cmd += ["-i", str(path)]
    music_idx: Optional[int] = None
    next_idx = len(shot_paths)
    if music_path is not None:
        music_idx = next_idx
        cmd += ["-i", str(music_path)]
        next_idx += 1
    ambience_base = next_idx
    for path, _gain in ambience:
        cmd += ["-i", str(path)]
    ambience_refs = [(ambience_base + i, gain) for i, (_p, gain) in enumerate(ambience)]
    audio_graph = build_audio_filter(music_idx, ambience_refs, total, with_music=music_idx is not None)
    graph = video_graph + ";" + audio_graph
    script_path = ass_path.parent / f"{out_path.stem}_graph.txt"
    script_path.write_text(graph, encoding="utf-8")
    cmd += ["-filter_complex_script", str(script_path), "-map", "[vout]", "-map", "[aout]",
           "-t", f"{total:.3f}"]
    cmd += get_encoder_flags(prefer_nvenc=False)
    cmd += [str(out_path)]
    await _run_ffmpeg(cmd)
    return total


async def _render_nomusic(main_path: Path, ambience: List[Tuple[Path, float]],
                          total: float, out_path: Path) -> None:
    """The same video, re-muxed with ambience only (no music bed) -- the
    video stream is copied verbatim from `main_path`, never re-encoded, so
    the two files' pictures are guaranteed identical."""
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(main_path)]
    for path, _gain in ambience:
        cmd += ["-i", str(path)]
    ambience_refs = [(i + 1, gain) for i, (_p, gain) in enumerate(ambience)]
    audio_graph = build_audio_filter(None, ambience_refs, total, with_music=False)
    cmd += ["-filter_complex", audio_graph, "-map", "0:v", "-map", "[aout]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-t", f"{total:.3f}", "-movflags", "+faststart", str(out_path)]
    await _run_ffmpeg(cmd)


async def _render_main_narrated(shot_paths: List[Path], shot_durations: List[float],
                                shot_kinds: List[str], shot_pads: List[float], ass_path: Path,
                                narration_path: Path, lead_s: float,
                                music_path: Optional[Path],
                                ambience: List[Tuple[Path, float]], out_path: Path,
                                dissolve_s: float, push: float, look_pass: bool = False,
                                grade: str = GRADE_NATURAL,
                                scrim: Optional[Tuple[float, float]] = None,
                                art_style: str = "real") -> float:
    """Like `_render_main`, but the bed is the narration (full level) plus an
    optional music track `NARRATION_MUSIC_GAIN_DB` under it and ambience,
    instead of a music track at the lines reel's own levels."""
    video_graph, total = build_video_graph(shot_durations, shot_kinds, shot_pads, ass_path,
                                           dissolve_s=dissolve_s, push=push, look_pass=look_pass,
                                           grade=grade, scrim=scrim, art_style=art_style)
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error"]
    for path, seconds, kind in zip(shot_paths, shot_durations, shot_kinds):
        if kind == "still":
            cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds + 0.5:.3f}", "-i", str(path)]
        else:
            cmd += ["-i", str(path)]
    narration_idx = len(shot_paths)
    cmd += ["-i", str(narration_path)]
    music_idx: Optional[int] = None
    if music_path is not None:
        music_idx = narration_idx + 1
        cmd += ["-i", str(music_path)]
    ambience_base = (music_idx if music_idx is not None else narration_idx) + 1
    for path, _gain in ambience:
        cmd += ["-i", str(path)]
    ambience_refs = [(ambience_base + i, gain) for i, (_p, gain) in enumerate(ambience)]
    audio_graph = build_narration_audio_filter(narration_idx, music_idx, ambience_refs, total,
                                               lead_s, with_music=music_idx is not None)
    graph = video_graph + ";" + audio_graph
    script_path = ass_path.parent / f"{out_path.stem}_graph.txt"
    script_path.write_text(graph, encoding="utf-8")
    cmd += ["-filter_complex_script", str(script_path), "-map", "[vout]", "-map", "[aout]",
           "-t", f"{total:.3f}"]
    cmd += get_encoder_flags(prefer_nvenc=False)
    cmd += [str(out_path)]
    await _run_ffmpeg(cmd)
    return total


async def _render_nomusic_narrated(main_path: Path, narration_path: Path, lead_s: float,
                                   ambience: List[Tuple[Path, float]], total: float,
                                   out_path: Path) -> None:
    """`<name>_nomusic.mp4` for a narrated reel: narration + ambience, no
    music -- the video stream copied verbatim from `main_path`, as for a
    lines reel's own `_render_nomusic`."""
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(main_path),
          "-i", str(narration_path)]
    for path, _gain in ambience:
        cmd += ["-i", str(path)]
    ambience_refs = [(i + 2, gain) for i, (_p, gain) in enumerate(ambience)]
    audio_graph = build_narration_audio_filter(1, None, ambience_refs, total, lead_s,
                                               with_music=False)
    cmd += ["-filter_complex", audio_graph, "-map", "0:v", "-map", "[aout]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-t", f"{total:.3f}", "-movflags", "+faststart", str(out_path)]
    await _run_ffmpeg(cmd)


def _extract_cover(video_path: Path, lines: List[Line], cover_path: Path,
                   fallback_at: Optional[float] = None) -> None:
    """A frame inside the first card's time (scene + text), or `fallback_at`
    (a narrated reel's first caption, or just past its lead-in) when there
    are no line cards."""
    if lines:
        at = (lines[0].start_s + lines[0].end_s) / 2.0
    elif fallback_at is not None:
        at = fallback_at
    else:
        at = 1.0
    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
          "-ss", f"{max(0.0, at):.3f}", "-i", str(video_path), "-frames:v", "1", str(cover_path)]
    subprocess.run(cmd, capture_output=True, timeout=60, creationflags=NO_WINDOW)


def parse_blackdetect(stderr_text: str, total_duration: float) -> List[Dict[str, float]]:
    """Every `blackdetect` hit in ffmpeg's stderr that is not entirely inside
    the opening fade-in (0..`FADE_IN_S`) or the closing fade-out
    (`total_duration - FADE_OUT_S`..`total_duration`) -- those are the fade
    itself, not a gap in the coverage."""
    hits: List[Dict[str, float]] = []
    for match in _BLACKDETECT_RE.finditer(stderr_text or ""):
        start, end, duration = (float(match.group(1)), float(match.group(2)),
                                float(match.group(3)))
        if end <= FADE_IN_S + 1e-6:
            continue
        if start >= total_duration - FADE_OUT_S - 1e-6:
            continue
        hits.append({"start": start, "end": end, "duration": duration})
    return hits


def check_no_black_frames(video_path: Path, total_duration: float) -> None:
    """Every frame of the output must be a picture. Raises VibeReelError with
    the timings of every hit outside the opening/closing fade -- a vibe reel
    with a black gap in it fails the job, never delivered quietly."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-i", str(video_path),
          "-vf", f"blackdetect=d={BLACKDETECT_D}:pix_th={BLACKDETECT_PIX_TH}:"
                f"pic_th={BLACKDETECT_PIC_TH}",
          "-an", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                            creationflags=NO_WINDOW)
    hits = parse_blackdetect(result.stderr, total_duration)
    if hits:
        timings = "; ".join(f"{h['start']:.2f}-{h['end']:.2f}s" for h in hits)
        raise VibeReelError(f"black frame(s) detected: {timings}")


_LOUDNORM_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)


def _measure_lufs(video_path: Path) -> Optional[float]:
    cmd = [FFMPEG_BIN, "-hide_banner", "-i", str(video_path),
          "-af", f"loudnorm=I={LOUDNESS_LUFS:.1f}:TP={TRUE_PEAK_DBTP:.1f}:LRA=11:print_format=json",
          "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                creationflags=NO_WINDOW)
        match = _LOUDNORM_JSON_RE.search(result.stderr)
        if match:
            return float(json.loads(match.group(0))["input_i"])
    except Exception as e:
        logger.warning("Loudness measurement failed for %s (%s)", video_path, e)
    return None


# --- the whole job -----------------------------------------------------------------

async def render_vibe_reel(req: VibeReelRequest,
                           progress_cb: Optional[Callable[[float, str], None]] = None
                           ) -> Dict[str, Any]:
    """Build every scene, write the lines, lay the bed, render `<name>.mp4`
    and `<name>_nomusic.mp4`, and pull a cover frame. Raises VibeReelError on
    anything that would otherwise come back incomplete (a missing shot, a
    music/ambience source that could not be resolved, a failed ffmpeg pass)
    -- a vibe reel either comes back whole or does not come back at all."""
    report = progress_cb or (lambda frac, msg="": None)
    out_dir = Path(req.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    dissolve_s = req.dissolve_s if req.dissolve_s is not None else (
        NARRATION_DISSOLVE_S if req.narration else DISSOLVE_S)
    captions_on = req.narration is not None and (
        req.captions is None or req.captions.style != "off")

    narration_duration: Optional[float] = None
    shot_durations_plan: Optional[List[float]] = None
    if req.narration is not None:
        # Read the narration's length and check every shot *before* anything
        # is generated -- a reel that cannot be covered never starts one.
        try:
            narration_duration = get_video_duration(req.narration.path)
        except Exception as e:
            raise VibeReelError(f"could not read the narration's length ({e})")
        total_duration_target = narration_duration + NARRATION_LEAD_S + NARRATION_TAIL_S
        try:
            shot_durations_plan = compute_shot_durations(req.shots, total_duration_target,
                                                         dissolve_s)
        except ValueError as e:
            raise VibeReelError(str(e))

    beats = build_beats(req, shot_durations_plan)
    settings = PresentationSettings(art_style=req.art_style,
                                    character_consistency=req.character is not None)

    # reel_hq (2026-10-23): the still and clip graphs for every shot in this
    # reel, a fallback chain ending at today's own `broll_video` resolution --
    # never the global `clip_generation_mode` a production reads.
    workflow_files: Optional[Dict[str, List[str]]] = None
    gen_size_override: Optional[Dict[str, Tuple[int, int]]] = None
    if req.quality == QUALITY_REEL_HQ:
        reel_mode = workflows_mod.CLIP_GENERATION_MODES.get(QUALITY_REEL_HQ, {})
        video_file = reel_mode.get("file")
        video_fallback = reel_mode.get("fallback")
        video_files = [f for f in (video_file, video_fallback) if f]
        workflow_files = {"broll_image": [REEL_HQ_STILL_FILE], "broll_video": video_files}
        video_size = tuple(reel_mode.get("size") or ()) or None
        gen_size_override = {"broll_image": REEL_HQ_STILL_SIZE}
        if video_size:
            gen_size_override["broll_video"] = video_size

    try:
        # The words on screen are built BEFORE any picture is generated
        # (2026-10-23 night rules, VR7): a narrated reel's captions need the
        # narration transcribed and transliterated to Hinglish first, and
        # that local-LLM call should have the card to itself -- ejected
        # (`gpu_handover.eject_lm_studio`) before ComfyUI's own phase starts.
        # A lines reel's cards need nothing generated first either; building
        # them here (instead of after the shots, as before) changes nothing
        # about what they say, only when.
        report(0.03, "Writing the words on screen")
        caption_phrases: List[Dict[str, Any]] = []
        transliterated = True
        if captions_on:
            report(0.04, "Transcribing the narration")
            from asr.faster_whisper_engine import whisper_engine
            heard_words, _detected = await whisper_engine.transcribe_words_async(
                req.narration.path, language=req.narration.language)
            raw_phrases = caption_align.build_caption_phrases(
                req.narration.text, heard_words, narration_duration or 0.0)
            caption_phrases = [{"text": p["text"], "start": p["start"] + NARRATION_LEAD_S,
                               "end": p["end"] + NARRATION_LEAD_S} for p in raw_phrases]
            report(0.045, "Hinglish captions")
            hinglish, transliterated = await transliterate_to_hinglish(
                [p["text"] for p in caption_phrases])
            for phrase, text in zip(caption_phrases, hinglish):
                phrase["text"] = text
            # The transliteration model's own card, freed before the image/
            # video phase -- `generate_assets` hands over again regardless,
            # but a local LLM is ejected here explicitly, as asked.
            await gpu_handover.eject_lm_studio()
            gpu_handover.release_in_process_models()
            ass_text = build_caption_ass(caption_phrases, req.font)
        else:
            ass_text = build_ass_script(req.lines, req.credit, req.font,
                                        text_style=req.text_style, highlight=req.highlight)
        ass_path = ass_mod.write_ass_asset(ass_text, assets_dir)

        report(0.05, "Generating scenes")
        shot_paths: List[Path] = []
        shot_durations: List[float] = []
        shot_pads: List[float] = []
        shot_workflows: List[str] = []
        shot_chains: List[int] = []
        shot_kinds_resolved: List[str] = []
        failures: List[Dict[str, str]] = []
        for i, beat in enumerate(beats):
            # Master pause (BuzzcafStudio docs/STUDIO_DIRECTOR_PLAN.md Part
            # 1): between shots, before the next one starts -- every shot
            # already made is cached on disk (below), so holding here never
            # loses one, and the job just carries on once resumed.
            from runtime import master_control
            await master_control.wait_if_paused_async(f"vibe_reel: shot {i + 1}/{len(beats)}", report)
            # Its own cache directory, not the shared one: `generate_assets`'
            # near-match reuse (meant for a full video's many beats) needs at
            # least 4 prompts before it treats the shared world/style words as
            # boilerplate -- with one beat per call that never happens, so two
            # different shots sharing the same world line could otherwise be
            # "matched" to each other's picture.
            shot_dir = out_dir / "shots" / beat.id
            shot_dir.mkdir(parents=True, exist_ok=True)
            # A character shot's seed_base is fixed (not `1000 + i*7`) so the
            # face reference's cache key is the same call to call -- made once
            # in `out_dir` (shared across every shot's own directory) and
            # reused, not regenerated per shot; the picture itself still
            # varies, since its own seed also carries the beat id.
            seed_base = CHARACTER_SEED_BASE if beat.shows_character else 1000 + i * 7
            wanted_s = beat.planned_duration_s or beat.duration_s
            if beat.kind == "stick_scene":
                for _retry in range(2):
                    path, actual, pad, reason, resolved_kind = await _generate_stick_shot(
                        beat, shot_dir, settings, seed_base=seed_base)
                    if path is not None or not master_control.is_paused():
                        break
                    await master_control.wait_if_paused_async(
                        f"vibe_reel: shot {i + 1}/{len(beats)} (retry after the pause)", report)
                if path is None:
                    failures.append({"beat_id": beat.id, "reason": reason})
                    continue
                shot_paths.append(path)
                shot_durations.append(actual)
                shot_pads.append(pad)
                shot_workflows.append("")
                shot_chains.append(1)
                shot_kinds_resolved.append(resolved_kind)
                report(0.05 + 0.55 * (i + 1) / len(beats), f"Scene {i + 1}/{len(beats)} ready")
                continue
            chained = beat.kind == "broll_video" and wanted_s > CHAIN_SEGMENT_SECONDS_MAX + SHOT_PAD_TOLERANCE_S
            # "now" mode interrupts ComfyUI mid-shot (master_control._interrupt_comfyui_now):
            # that reads here as an ordinary generation failure. Hold instead of
            # recording it and retry the same shot once resumed, so a "now" pause
            # never fails the job either -- only a shot that was never cached
            # regenerates; the rest of the loop is unaffected.
            for _retry in range(2):
                if chained:
                    path, actual, pad, reason, workflow_file, segments = await _generate_chained_shot(
                        beat, shot_dir, settings, seed_base=seed_base,
                        character=req.character, character_cache_dir=out_dir,
                        workflow_files=workflow_files, gen_size_override=gen_size_override)
                else:
                    path, actual, pad, reason, workflow_file = await _generate_shot(
                        beat, shot_dir, settings, seed_base=seed_base,
                        character=req.character, character_cache_dir=out_dir,
                        workflow_files=workflow_files, gen_size_override=gen_size_override)
                    segments = 1
                if path is not None or not master_control.is_paused():
                    break
                await master_control.wait_if_paused_async(
                    f"vibe_reel: shot {i + 1}/{len(beats)} (retry after the pause)", report)
            if path is None:
                failures.append({"beat_id": beat.id, "reason": reason})
                continue
            shot_paths.append(path)
            shot_durations.append(beat.planned_duration_s if beat.kind == "broll_image" else actual)
            shot_pads.append(pad)
            shot_workflows.append(workflow_file)
            shot_chains.append(segments)
            shot_kinds_resolved.append(req.shots[i].kind)
            report(0.05 + 0.55 * (i + 1) / len(beats), f"Scene {i + 1}/{len(beats)} ready")

        if failures:
            raise VibeReelError(
                f"{len(failures)} shot(s) could not be made: "
                + "; ".join(f"{f['beat_id']} ({f['reason']})" for f in failures))

        # A "stick" shot resolves to "video" (the motion engine actually
        # rendered it) or "still" (its picture, or motion_stick.
        # render_stick_still, as a fallback) -- never literally "stick",
        # which build_video_graph does not know (STICK_WORLD_PLAN.md SW2).
        shot_kinds = shot_kinds_resolved
        total_estimate = sum(shot_durations) - max(0, len(shot_durations) - 1) * dissolve_s

        music_path: Optional[Path] = None
        music_used: Optional[str] = None
        if req.music is not None:
            report(0.68, "Composing the music bed")
            music_path, music_used = await _resolve_music(req.music, total_estimate)

        report(0.76, "Laying the ambience")
        ambience = await _resolve_ambience(req.ambience, total_estimate)

        report(0.84, "Rendering")
        video_path = out_dir / f"{req.name}.mp4"
        nomusic_path = out_dir / f"{req.name}_nomusic.mp4"
        look_pass = req.quality == QUALITY_REEL_HQ
        # A scrim (soft dark band) behind the text only when there is text to
        # sit behind, scaled by how bright that part of the first shot reads.
        scrim: Optional[Tuple[float, float]] = None
        if req.lines or captions_on:
            scrim_pos_y = CAPTION_POS_Y if captions_on else _card_pos_y(req.text_style)
            sample_at = (caption_phrases[0]["start"] if caption_phrases
                        else (req.lines[0].start_s if req.lines else 0.0))
            luma = _mean_luma_band(shot_paths[0], sample_at, scrim_pos_y) if shot_paths else None
            scrim = (scrim_pos_y, _scrim_alpha_for(luma))
        # Only a lines reel's `<name>_nomusic.mp4` can be skipped (2026-10-24,
        # VR8): a narrated reel always writes one (unchanged); a lines reel
        # with no music has nothing left to drop from it, so none is written
        # and `result["video_nomusic"]` comes back None.
        nomusic_written = True
        if req.narration is not None:
            narration_path = Path(req.narration.path)
            total = await _render_main_narrated(
                shot_paths, shot_durations, shot_kinds, shot_pads, ass_path,
                narration_path, NARRATION_LEAD_S, music_path, ambience, video_path,
                dissolve_s, req.push, look_pass=look_pass, grade=req.grade, scrim=scrim,
                art_style=req.art_style)
            check_no_black_frames(video_path, total)
            await _render_nomusic_narrated(video_path, narration_path, NARRATION_LEAD_S,
                                           ambience, total, nomusic_path)
        else:
            total = await _render_main(shot_paths, shot_durations, shot_kinds, shot_pads, ass_path,
                                       music_path, ambience, video_path,
                                       dissolve_s=dissolve_s, push=req.push, look_pass=look_pass,
                                       grade=req.grade, scrim=scrim, art_style=req.art_style)
            check_no_black_frames(video_path, total)
            if music_path is not None:
                await _render_nomusic(video_path, ambience, total, nomusic_path)
            else:
                nomusic_written = False

        report(0.95, "Cover frame")
        cover_path = out_dir / "cover.jpg"
        fallback_at = (caption_phrases[0]["start"] if caption_phrases
                      else (NARRATION_LEAD_S if req.narration is not None else None))
        _extract_cover(video_path, req.lines, cover_path, fallback_at=fallback_at)

        lufs = _measure_lufs(video_path)
        result = {"video": str(video_path),
                 "video_nomusic": str(nomusic_path) if nomusic_written else None,
                 "cover": str(cover_path), "duration_s": round(total, 3),
                 "shots": [str(p) for p in shot_paths], "failures": [], "lufs": lufs,
                 "music_used": music_used,
                 "ambience_used": [str(p) for p, _g in ambience],
                 "caption_phrases": len(caption_phrases),
                 "character": req.character.model_dump() if req.character else None,
                 "character_shots": [b.id for b in beats if b.shows_character],
                 "quality": req.quality, "shot_workflows": shot_workflows,
                 "grade": req.grade, "text_style": req.text_style,
                 "shot_chain_segments": shot_chains, "chained": any(s > 1 for s in shot_chains),
                 "transliterated": transliterated}
        (out_dir / "vibe_reel.json").write_text(
            json.dumps({"request": req.model_dump(), "result": result}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        report(1.0, "Done")
        return result
    finally:
        try:
            await gpu_handover.offload_all("vibe reel finished")
        except Exception as e:
            logger.debug("Vibe reel offload skipped: %s", e)
