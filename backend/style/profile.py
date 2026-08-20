"""The StyleProfile: a measured description of how a reference video was edited.

A profile is not the reference's effect stack — that information does not survive
rendering. It is a set of measurements: how fast it cuts, whether it cuts to the
beat, how often and how far shots push in, what the picture's exposure and colour
cast are, and where any burned-in captions sit. Every field carries the accuracy
it was measured with, because these are not all equally trustworthy and the UI
should say so rather than present a guess as a fact.

Profiles live in `data/styles/` and are app-wide, so a reference analysed once can
be applied to any project.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import DATA_DIR
from timeline.schema import ColorGrade
from style import color as color_mod
from style import frames as frames_mod
from style import motion as motion_mod
from style import rhythm as rhythm_mod
from style import shots as shots_mod
from style import text_regions

logger = logging.getLogger(__name__)

STYLES_DIR = Path(DATA_DIR) / "styles"

# Sampling sizes. Cut detection needs temporal resolution more than spatial;
# flow needs enough pixels to fit a gradient; colour needs real RGB but very
# few frames, since a look is a whole-video statistic.
_CUT_SIZE, _CUT_FPS = (64, 36), 12.0   # RGB: colour catches equal-brightness cuts
_FLOW_SIZE, _FLOW_FPS = (160, 90), 6.0
_COLOR_SIZE, _COLOR_FPS = (128, 72), 0.5
_TEXT_SIZE, _TEXT_FPS = (192, 108), 1.0
_AUDIO_RATE = 22050


class Pacing(BaseModel):
    shots: int = 0
    cuts_per_minute: float = 0.0
    median_shot_seconds: float = 0.0
    p25_shot_seconds: float = 0.0
    p75_shot_seconds: float = 0.0


class Rhythm(BaseModel):
    bpm: float = 0.0
    beat_alignment: float = 0.0
    confidence: float = 0.0
    cuts_to_music: bool = False


class Motion(BaseModel):
    zoom_share: float = 0.0
    pan_share: float = 0.0
    mean_zoom_ratio: float = 0.0
    mean_pan_fraction: float = 0.0
    mean_move_seconds: float = 0.0


class Look(BaseModel):
    description: str = ""
    stats: Dict[str, Any] = Field(default_factory=dict)


class Captions(BaseModel):
    present: bool = False
    pos_y: float = 0.75
    size_fraction: float = 0.0
    coverage: float = 0.0
    boxed: bool = False
    confidence: float = 0.0


class StyleProfile(BaseModel):
    id: str
    name: str
    source_path: str
    duration: float = 0.0
    created_at: str = ""
    pacing: Pacing = Field(default_factory=Pacing)
    rhythm: Rhythm = Field(default_factory=Rhythm)
    motion: Motion = Field(default_factory=Motion)
    look: Look = Field(default_factory=Look)
    captions: Captions = Field(default_factory=Captions)
    transitions: Dict[str, Any] = Field(default_factory=dict)
    # How much to trust each part, so the UI never presents a weak signal as fact.
    confidence: Dict[str, str] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "style"


def analyze_video(path: str, name: Optional[str] = None) -> StyleProfile:
    """Measure a reference video and return its style profile."""
    source = Path(path)
    if not source.is_file():
        raise frames_mod.FrameSampleError(f"File not found: {path}")

    duration = frames_mod.probe_duration(str(source))
    notes: List[str] = []

    # 1. Cuts and transitions
    cut_frames, cut_fps = frames_mod.sample(
        str(source), *_CUT_SIZE, fps=_CUT_FPS, gray=False, duration=duration)
    shot_analysis = shots_mod.detect_boundaries(cut_frames, cut_fps, duration)
    cut_times = [b.time for b in shot_analysis.boundaries]

    lengths = shot_analysis.shot_lengths or [duration]
    import numpy as np
    pacing = Pacing(
        shots=len(lengths),
        cuts_per_minute=round(shot_analysis.cuts_per_minute, 2),
        median_shot_seconds=round(float(np.median(lengths)), 2),
        p25_shot_seconds=round(float(np.percentile(lengths, 25)), 2),
        p75_shot_seconds=round(float(np.percentile(lengths, 75)), 2),
    )

    # 2. Rhythm
    audio = frames_mod.sample_audio(str(source), _AUDIO_RATE)
    rhythm_result = rhythm_mod.analyse_rhythm(audio, _AUDIO_RATE, cut_times)
    # Cutting to music needs both a believable tempo and cuts that actually land
    # on it; either alone is not evidence.
    cuts_to_music = (rhythm_result.alignment >= 0.55
                     and rhythm_result.confidence >= 0.15
                     and len(cut_times) >= 6)
    rhythm = Rhythm(
        bpm=rhythm_result.bpm,
        beat_alignment=rhythm_result.alignment,
        confidence=rhythm_result.confidence,
        cuts_to_music=cuts_to_music,
    )
    if not rhythm_result.has_audio:
        notes.append("Reference has no audio track; rhythm was not measured.")

    # 3. Motion, measured inside shots so a cut is never read as a camera move
    bounds: List[tuple] = []
    previous = 0.0
    for boundary in shot_analysis.boundaries:
        bounds.append((previous, boundary.time))
        previous = boundary.time
    bounds.append((previous, duration))

    try:
        flow_frames, flow_fps = frames_mod.sample(
            str(source), *_FLOW_SIZE, fps=_FLOW_FPS, gray=True, duration=duration)
        motion_result = motion_mod.analyse_motion(flow_frames, flow_fps, bounds)
    except frames_mod.FrameSampleError as exc:
        logger.warning("Motion analysis skipped: %s", exc)
        motion_result = motion_mod.MotionAnalysis()
        notes.append("Motion could not be measured on this reference.")

    motion = Motion(
        zoom_share=motion_result.zoom_share,
        pan_share=motion_result.pan_share,
        mean_zoom_ratio=motion_result.mean_zoom_ratio,
        mean_pan_fraction=motion_result.mean_pan_fraction,
        mean_move_seconds=motion_result.mean_move_seconds,
    )

    # 4. Look
    color_frames, _ = frames_mod.sample(
        str(source), *_COLOR_SIZE, fps=_COLOR_FPS, gray=False,
        max_frames=400, duration=duration)
    stats = color_mod.measure(color_frames)
    look = Look(description=color_mod.describe(stats), stats=stats.to_dict())

    # 5. Captions. Raise the rate on short references so there are always enough
    # frames to judge persistence by — one per second gives a 6-second clip too
    # few samples to say anything.
    text_fps = max(_TEXT_FPS, 30.0 / max(1.0, duration))
    text_frames, _ = frames_mod.sample(
        str(source), *_TEXT_SIZE, fps=text_fps, gray=True,
        max_frames=900, duration=duration)
    caption_result = text_regions.analyse_captions(text_frames)
    captions = Captions(
        present=caption_result.present,
        pos_y=caption_result.pos_y if caption_result.present else 0.75,
        size_fraction=caption_result.size_fraction,
        coverage=caption_result.coverage,
        boxed=caption_result.boxed,
        confidence=caption_result.confidence,
    )

    if len(cut_times) < 3:
        notes.append("Very few cuts detected — pacing figures are weak on this reference.")

    profile = StyleProfile(
        id=f"{_slug(name or source.stem)}-{uuid.uuid4().hex[:6]}",
        name=name or source.stem,
        source_path=str(source),
        duration=round(duration, 2),
        created_at=datetime.now().isoformat(timespec="seconds"),
        pacing=pacing,
        rhythm=rhythm,
        motion=motion,
        look=look,
        captions=captions,
        transitions=shots_mod.transition_mix(shot_analysis),
        confidence={
            "pacing": "high" if len(cut_times) >= 6 else "low",
            "rhythm": "high" if rhythm_result.confidence >= 0.3 else "moderate"
                      if rhythm_result.has_audio else "none",
            "motion": "moderate",     # cannot separate camera move from post zoom
            "look": "high",
            "captions": "moderate" if caption_result.present else "none",
            "transitions": "moderate",
        },
        notes=notes,
    )
    return profile


# ---------------------------------------------------------------------------
# Storage — profiles are app-wide, not per project
# ---------------------------------------------------------------------------

def _path_for(profile_id: str) -> Path:
    return STYLES_DIR / f"{profile_id}.json"


def save(profile: StyleProfile) -> StyleProfile:
    STYLES_DIR.mkdir(parents=True, exist_ok=True)
    _path_for(profile.id).write_text(
        json.dumps(profile.model_dump(), indent=2), encoding="utf-8")
    return profile


def load(profile_id: str) -> Optional[StyleProfile]:
    path = _path_for(profile_id)
    if not path.exists():
        return None
    try:
        return StyleProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Could not read style profile %s: %s", profile_id, exc)
        return None


def list_profiles() -> List[StyleProfile]:
    if not STYLES_DIR.exists():
        return []
    found = []
    for path in sorted(STYLES_DIR.glob("*.json")):
        profile = load(path.stem)
        if profile:
            found.append(profile)
    return sorted(found, key=lambda p: p.created_at, reverse=True)


def delete(profile_id: str) -> bool:
    path = _path_for(profile_id)
    if path.exists():
        path.unlink()
        return True
    return False
