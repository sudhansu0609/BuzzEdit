"""Reference-video style matching.

Measures how a reference video was edited — pace, rhythm, camera moves, look,
caption placement — and reproduces that grammar on a project's own footage.

The effects themselves cannot be recovered from a rendered file; what is stored
is a set of measurements, each carrying the confidence it was measured with.
"""

from .apply import ApplyOptions, apply_profile, caption_overrides
from .profile import (
    StyleProfile,
    analyze_video,
    delete,
    list_profiles,
    load,
    save,
)

__all__ = [
    "StyleProfile",
    "analyze_video",
    "apply_profile",
    "caption_overrides",
    "ApplyOptions",
    "save",
    "load",
    "list_profiles",
    "delete",
]
