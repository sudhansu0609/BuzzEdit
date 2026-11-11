"""The presentation pass: turning a finished cut into a presentable video.

Run overnight, unattended. Reads the transcript, works out what the video is
about, generates images and video through the user's own ComfyUI workflows,
cuts them in as B-roll, adds face-aware punch-ins timed to how the speaker is
talking, pops up topic text, captions it, renders it, and saves the whole thing
as a timeline that can be edited in the morning.

Stages are documented in their own modules; `director.run_presentation_pass` is
the only entry point anything outside this package should need.
"""

# motion first: its builder modules (motion_collage, motion_stick, ...) import
# from it, and it imports them back at its bottom -- loading one of them first
# hit a circular ImportError (2026-10-29; only BuzzEdit's startup order hid it).
from . import motion  # noqa: F401
from .director import run_presentation_pass
from .models import (
    Asset,
    Beat,
    PresentationReport,
    PresentationSettings,
    Program,
    ShotPlan,
)

__all__ = [
    "run_presentation_pass",
    "PresentationSettings",
    "PresentationReport",
    "Program",
    "ShotPlan",
    "Beat",
    "Asset",
]
