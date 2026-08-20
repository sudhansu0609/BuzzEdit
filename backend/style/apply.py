"""Apply a measured style profile to a project timeline.

Each part of a profile maps onto something the EDL already renders, which is why
this needs no new render path:

* **look**   -> `Timeline.master_color`, fitted from the project's own footage to
                the reference's statistics
* **motion + pacing** -> per-segment `Transform` moves on V1
* **captions** -> geometry overrides handed to the existing caption generator
* **transitions** -> programme-level fades

The pacing/motion pairing deserves a word. In a multi-camera reference, pace comes
from cutting between angles. A single-camera talking head has no angles to cut to,
so the same felt rhythm is produced the way editors actually produce it: punch-ins.
Matching the reference's cuts-per-minute therefore means placing a visual change —
a push in or a pull back — at roughly that rate, on segment boundaries the
transcript already gives us.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from style import color as color_mod
from style import frames as frames_mod
from style.profile import StyleProfile
from timeline.schema import ColorGrade, Timeline, TimelineItem, Transform, frame_to_time

logger = logging.getLogger(__name__)

_COLOR_SIZE, _COLOR_FPS = (128, 72), 0.5


@dataclass
class ApplyOptions:
    look: bool = True
    motion: bool = True
    captions: bool = True
    transitions: bool = True
    # Deterministic placement so re-applying a profile gives the same edit rather
    # than reshuffling every move.
    seed: int = 7


def _program_segments(timeline: Timeline) -> List[TimelineItem]:
    return sorted(
        [i for i in timeline.items
         if i.track == "V1" and i.enabled and i.kind == "media"],
        key=lambda i: i.timeline_start_frame,
    )


def apply_look(timeline: Timeline, profile: StyleProfile,
               source_path: Optional[str]) -> Dict[str, Any]:
    """Match the programme's grade to the reference's measured look.

    The fit needs both sides: the same target statistics call for opposite
    corrections depending on whether your own footage started flat or punchy.
    """
    reference_stats = profile.look.stats
    if not reference_stats:
        return {"applied": False, "reason": "profile carries no colour statistics"}
    if not source_path:
        return {"applied": False, "reason": "project has no source video to measure"}

    try:
        own_frames, _ = frames_mod.sample(
            source_path, *_COLOR_SIZE, fps=_COLOR_FPS, gray=False, max_frames=300)
    except frames_mod.FrameSampleError as exc:
        return {"applied": False, "reason": f"could not read project footage: {exc}"}

    own = color_mod.measure(own_frames)
    target = color_mod.ColorStats(
        mean=float(reference_stats["mean"]),
        std=float(reference_stats["std"]),
        saturation=float(reference_stats["saturation"]),
        channel_mean={k: float(v) for k, v in reference_stats["channel_mean"].items()},
    )
    grade = color_mod.fit_grade(own, target)
    grade.preset = None
    timeline.master_color = grade
    return {
        "applied": True,
        "grade": grade.model_dump(),
        "from": color_mod.describe(own),
        "to": profile.look.description,
    }


def apply_motion(timeline: Timeline, profile: StyleProfile,
                 options: ApplyOptions) -> Dict[str, Any]:
    """Place push-ins and pull-backs to match the reference's pace and zoom depth."""
    segments = _program_segments(timeline)
    if not segments:
        return {"applied": False, "reason": "no programme segments to move"}

    fps = timeline.fps_num / max(1, timeline.fps_den)
    duration_sec = frame_to_time(timeline.duration_frames, timeline.fps_num, timeline.fps_den)
    if duration_sec <= 0:
        return {"applied": False, "reason": "timeline has no duration"}

    target_cuts = max(0.0, profile.pacing.cuts_per_minute) * (duration_sec / 60.0)
    # Zoom depth: fall back to a gentle default if the reference used none, so
    # "match the pacing" still produces visible change.
    depth = profile.motion.mean_zoom_ratio or 0.12
    depth = max(0.04, min(0.6, depth))

    # Some segments are too short to read as a move rather than a jolt. This is a
    # property of perception, not of the reference — gating on the reference's own
    # move length would disable motion entirely for any reference that happens not
    # to zoom, which is exactly when borrowing its *pace* matters most.
    min_frames = int(round(0.5 * fps))
    eligible = [s for s in segments if s.duration_frames >= min_frames]
    if not eligible:
        return {"applied": False, "reason": "programme segments are too short for moves"}

    wanted = int(round(min(len(eligible), max(0.0, target_cuts))))
    if wanted <= 0:
        return {"applied": False, "reason": "reference pace implies no moves"}

    # Spread the moves evenly over the programme instead of clustering them, and
    # keep it deterministic so re-applying does not reshuffle the edit.
    step = len(eligible) / wanted
    chosen = [eligible[min(len(eligible) - 1, int(i * step))] for i in range(wanted)]

    applied = 0
    for index, segment in enumerate(chosen):
        # Alternate direction: a run of identical push-ins reads as a mistake.
        push_in = ((index + options.seed) % 2 == 0)
        start_scale = 1.0 if push_in else 1.0 + depth
        end_scale = 1.0 + depth if push_in else 1.0
        segment.transform = Transform(
            scale=round(start_scale, 3),
            scale_end=round(end_scale, 3),
        )
        applied += 1

    return {
        "applied": True,
        "moves": applied,
        "eligible_segments": len(eligible),
        "target_cuts_per_minute": profile.pacing.cuts_per_minute,
        "zoom_depth": round(depth, 3),
    }


def caption_overrides(profile: StyleProfile, timeline: Timeline) -> Dict[str, Any]:
    """TextStyle overrides that put captions where the reference put them."""
    if not profile.captions.present:
        return {}

    canvas_height = timeline.height or 1080
    # The measured band includes the glyphs' full extent; cap size so a wide
    # detection cannot produce absurd type.
    size_fraction = min(0.22, max(0.02, profile.captions.size_fraction))
    font_size = int(round(canvas_height * size_fraction * 0.85))

    overrides: Dict[str, Any] = {
        "pos_y": round(profile.captions.pos_y, 3),
        "font_size": max(20, min(160, font_size)),
    }
    if profile.captions.boxed:
        overrides["box"] = True
        overrides["box_color"] = "black@0.6"
    return overrides


def apply_transitions(timeline: Timeline, profile: StyleProfile) -> Dict[str, Any]:
    """Copy programme-level fades when the reference opens or closes on one.

    Only the head and tail are touched. A dissolve at every cut needs `xfade`,
    which the concat-based programme path cannot express, and faking it with a
    fade-out/fade-in per cut would look like a fault rather than a style.
    """
    fade = profile.transitions.get("fade") or {}
    dissolve = profile.transitions.get("dissolve") or {}
    duration = fade.get("mean_duration") or dissolve.get("mean_duration") or 0.0
    if duration <= 0:
        return {"applied": False, "reason": "reference uses hard cuts throughout"}

    duration = float(min(2.0, max(0.2, duration)))
    grade = timeline.master_color or ColorGrade()
    grade.fade_in = duration
    grade.fade_out = duration
    timeline.master_color = grade
    return {"applied": True, "fade_seconds": round(duration, 2)}


def apply_profile(
    timeline: Timeline,
    profile: StyleProfile,
    source_path: Optional[str] = None,
    options: Optional[ApplyOptions] = None,
) -> Dict[str, Any]:
    """Apply the selected parts of a profile and report exactly what changed."""
    options = options or ApplyOptions()
    report: Dict[str, Any] = {"profile": profile.name}

    if options.look:
        report["look"] = apply_look(timeline, profile, source_path)
    if options.motion:
        report["motion"] = apply_motion(timeline, profile, options)
    if options.transitions:
        report["transitions"] = apply_transitions(timeline, profile)
    if options.captions:
        overrides = caption_overrides(profile, timeline)
        report["captions"] = ({"applied": True, "overrides": overrides} if overrides
                              else {"applied": False, "reason": "no captions detected in the reference"})

    timeline.recalculate_duration()
    timeline.revision += 1
    return report
