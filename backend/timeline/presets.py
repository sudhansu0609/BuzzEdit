"""Named looks for colour, text, captions and intros.

Every preset is a plain dict of field overrides for `ColorGrade` / `TextStyle`,
never a bespoke object. That means applying a preset is just a model update, the
user can then tweak any single value, and the tweak survives — the preset name is
carried along as a label only. Adding a look here needs no other code change.
"""

from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Colour correction
# --------------------------------------------------------------------------

COLOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "label": "None",
        "values": {},
    },
    "cinematic": {
        "label": "Cinematic",
        "values": {"contrast": 1.15, "brightness": -0.02, "saturation": 1.12,
                   "gamma": 0.95, "temperature": -0.12, "vignette": 0.25},
    },
    "vibrant": {
        "label": "Vibrant",
        "values": {"contrast": 1.2, "saturation": 1.45, "brightness": 0.02, "sharpen": 0.4},
    },
    "warm": {
        "label": "Warm",
        "values": {"contrast": 1.08, "saturation": 1.2, "temperature": 0.35},
    },
    "cool": {
        "label": "Cool",
        "values": {"contrast": 1.08, "saturation": 1.08, "temperature": -0.35},
    },
    "moody": {
        "label": "Moody Dark",
        "values": {"contrast": 1.3, "brightness": -0.09, "saturation": 0.9,
                   "gamma": 0.9, "temperature": -0.2, "vignette": 0.45},
    },
    "teal_orange": {
        "label": "Teal & Orange",
        "values": {"contrast": 1.18, "saturation": 1.3, "temperature": 0.25,
                   "hue": -6, "vignette": 0.2},
    },
    "clean": {
        "label": "Clean Bright",
        "values": {"brightness": 0.05, "contrast": 1.05, "saturation": 1.05, "sharpen": 0.3},
    },
    "vintage": {
        "label": "Vintage Film",
        "values": {"contrast": 0.92, "saturation": 0.75, "gamma": 1.1,
                   "temperature": 0.3, "vignette": 0.35},
    },
    "bw": {
        "label": "Black & White",
        "values": {"saturation": 0.0, "contrast": 1.25, "sharpen": 0.3},
    },
    "punchy_talking_head": {
        "label": "Punchy Talking Head",
        "values": {"contrast": 1.12, "saturation": 1.18, "brightness": 0.03,
                   "sharpen": 0.6, "temperature": 0.1},
    },
}


# --------------------------------------------------------------------------
# Text looks
# --------------------------------------------------------------------------

TEXT_PRESETS: Dict[str, Dict[str, Any]] = {
    "title_bold": {
        "label": "Bold Title",
        "style": {"font_family": "Montserrat", "font_size": 96, "bold": True,
                  "color": "white", "stroke_width": 4, "stroke_color": "black",
                  "shadow_x": 3, "shadow_y": 3, "pos_y": 0.0, "animation": "pop"},
    },
    "youtube_pop": {
        "label": "YouTube Pop",
        "style": {"font_family": "Impact", "font_size": 110, "bold": True,
                  "color": "#FFE23A", "stroke_width": 8, "stroke_color": "black",
                  "shadow_x": 4, "shadow_y": 4, "animation": "pop"},
    },
    "minimal": {
        "label": "Minimal",
        "style": {"font_family": "Inter", "font_size": 56, "color": "white",
                  "stroke_width": 0, "shadow_x": 2, "shadow_y": 2,
                  "shadow_color": "black@0.5", "animation": "fade"},
    },
    "lower_third": {
        "label": "Lower Third",
        "style": {"font_family": "Poppins", "font_size": 48, "bold": True,
                  "color": "white", "box": True, "box_color": "#111111@0.75",
                  "box_padding": 18, "align": "left", "pos_x": -0.55, "pos_y": 0.62,
                  "animation": "slide-up"},
    },
    "neon": {
        "label": "Neon Glow",
        "style": {"font_family": "Bebas Neue", "font_size": 88, "color": "#39FF14",
                  "stroke_width": 3, "stroke_color": "#003300",
                  "shadow_x": 0, "shadow_y": 0, "shadow_color": "#39FF14@0.7",
                  "animation": "fade"},
    },
    "elegant_serif": {
        "label": "Elegant Serif",
        "style": {"font_family": "Georgia", "font_size": 64, "italic": True,
                  "color": "#F5F0E6", "stroke_width": 0,
                  "shadow_x": 2, "shadow_y": 2, "animation": "fade",
                  "animation_duration": 0.6},
    },
    "quote_card": {
        "label": "Quote Card",
        "style": {"font_family": "Georgia", "font_size": 58, "italic": True,
                  "color": "white", "box": True, "box_color": "black@0.55",
                  "box_padding": 28, "line_spacing": 16, "animation": "fade"},
    },
    "code_mono": {
        "label": "Mono Tag",
        "style": {"font_family": "Consolas", "font_size": 40, "color": "#7FFFD4",
                  "box": True, "box_color": "#0A0A0A@0.85", "box_padding": 14,
                  "align": "left", "pos_x": -0.6, "pos_y": -0.7, "animation": "fade"},
    },
    "handwritten_note": {
        "label": "Sticky Note",
        "style": {"font_family": "Segoe Script", "font_size": 52, "color": "#1A1A1A",
                  "box": True, "box_color": "#FFE97F@0.95", "box_padding": 22,
                  "animation": "pop"},
    },
}


# --------------------------------------------------------------------------
# Caption looks (transcript -> burned-in text)
# --------------------------------------------------------------------------
# `words_per_caption` and `max_gap_seconds` control how the word stream is cut
# into caption cards; the style block is a TextStyle override.

CAPTION_PRESETS: Dict[str, Dict[str, Any]] = {
    "classic": {
        "label": "Classic Subtitles",
        "words_per_caption": 8,
        "max_gap_seconds": 0.8,
        "uppercase": False,
        "style": {"font_family": "Arial", "font_size": 46, "color": "white",
                  "box": True, "box_color": "black@0.6", "box_padding": 12,
                  "pos_y": 0.75, "animation": "none"},
    },
    "youtube_shorts": {
        "label": "Shorts Word Pop",
        "words_per_caption": 3,
        "max_gap_seconds": 0.5,
        "uppercase": True,
        "style": {"font_family": "Montserrat", "font_size": 84, "bold": True,
                  "color": "white", "stroke_width": 7, "stroke_color": "black",
                  "pos_y": 0.35, "animation": "pop", "animation_duration": 0.15},
    },
    "podcast": {
        "label": "Podcast Clean",
        "words_per_caption": 6,
        "max_gap_seconds": 0.8,
        "uppercase": False,
        "style": {"font_family": "Inter", "font_size": 52, "color": "white",
                  "stroke_width": 3, "stroke_color": "black@0.8",
                  "pos_y": 0.72, "animation": "fade", "animation_duration": 0.12},
    },
    "highlight_yellow": {
        "label": "Yellow Highlight",
        "words_per_caption": 4,
        "max_gap_seconds": 0.6,
        "uppercase": True,
        "style": {"font_family": "Impact", "font_size": 76, "color": "#FFD400",
                  "stroke_width": 6, "stroke_color": "black",
                  "shadow_x": 3, "shadow_y": 3, "pos_y": 0.6, "animation": "pop"},
    },
    "cinematic_lower": {
        "label": "Cinematic Lower",
        "words_per_caption": 10,
        "max_gap_seconds": 1.0,
        "uppercase": False,
        "style": {"font_family": "Poppins", "font_size": 44, "color": "#F2F2F2",
                  "shadow_x": 2, "shadow_y": 2, "shadow_color": "black@0.8",
                  "pos_y": 0.8, "animation": "fade", "animation_duration": 0.25},
    },
}


# --------------------------------------------------------------------------
# Intros
# --------------------------------------------------------------------------
# An intro is a duration plus a list of text beats. Each beat is a text preset
# name with an override block and a start/end offset in seconds relative to the
# intro's start. `push_program` shifts existing clips to make room.

INTRO_PRESETS: Dict[str, Dict[str, Any]] = {
    "title_card": {
        "label": "Title Card",
        "duration": 3.0,
        "background": "black",
        "beats": [
            {"preset": "title_bold", "text": "{title}", "start": 0.2, "end": 3.0,
             "style": {"pos_y": -0.08, "font_size": 104}},
            {"preset": "minimal", "text": "{subtitle}", "start": 0.7, "end": 3.0,
             "style": {"pos_y": 0.16, "font_size": 44, "color": "#BBBBBB"}},
        ],
    },
    "bold_slam": {
        "label": "Bold Slam",
        "duration": 2.2,
        "background": "#101010",
        "beats": [
            {"preset": "youtube_pop", "text": "{title}", "start": 0.0, "end": 2.2,
             "style": {"animation": "pop", "animation_duration": 0.18}},
        ],
    },
    "minimal_fade": {
        "label": "Minimal Fade",
        "duration": 2.5,
        "background": "black",
        "beats": [
            {"preset": "minimal", "text": "{title}", "start": 0.3, "end": 2.5,
             "style": {"font_size": 72, "animation": "fade", "animation_duration": 0.7}},
        ],
    },
    "channel_bumper": {
        "label": "Channel Bumper",
        "duration": 3.5,
        "background": "#0B1D3A",
        "beats": [
            {"preset": "title_bold", "text": "{title}", "start": 0.2, "end": 3.5,
             "style": {"pos_y": -0.12, "color": "#FFFFFF", "font_size": 92}},
            {"preset": "code_mono", "text": "{subtitle}", "start": 0.9, "end": 3.5,
             "style": {"pos_x": 0.0, "pos_y": 0.18, "align": "center", "font_size": 38}},
        ],
    },
    "quote_open": {
        "label": "Quote Open",
        "duration": 4.0,
        "background": "#141414",
        "beats": [
            {"preset": "quote_card", "text": "{title}", "start": 0.4, "end": 4.0,
             "style": {"font_size": 60}},
        ],
    },
    "overlay_hook": {
        "label": "Overlay Hook (no black card)",
        "duration": 2.5,
        "background": None,   # draws over the existing footage instead of a card
        "beats": [
            {"preset": "youtube_pop", "text": "{title}", "start": 0.0, "end": 2.5,
             "style": {"pos_y": -0.45, "font_size": 84}},
        ],
    },
}


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------

TRANSITION_PRESETS: Dict[str, Dict[str, Any]] = {
    "cut": {"label": "Hard Cut", "type": "fade", "duration": 0.0},
    "dissolve": {"label": "Dissolve", "type": "dissolve", "duration": 0.5},
    "crossfade": {"label": "Crossfade", "type": "fade", "duration": 0.5},
    "dip_to_black": {"label": "Dip to Black", "type": "fadeblack", "duration": 0.7},
    "dip_to_white": {"label": "Dip to White", "type": "fadewhite", "duration": 0.5},
    "slide_left": {"label": "Slide Left", "type": "slideleft", "duration": 0.45},
    "wipe_right": {"label": "Wipe Right", "type": "wiperight", "duration": 0.45},
    "circle_open": {"label": "Circle Open", "type": "circleopen", "duration": 0.6},
    "zoom_in": {"label": "Zoom In", "type": "zoomin", "duration": 0.5},
    "whip_left": {"label": "Whip Left", "type": "hlwind", "duration": 0.35},
    "blur": {"label": "Blur Through", "type": "hblur", "duration": 0.5},
    "pixelize": {"label": "Pixelise", "type": "pixelize", "duration": 0.5},
    "squeeze": {"label": "Squeeze", "type": "squeezeh", "duration": 0.45},
    "radial": {"label": "Radial", "type": "radial", "duration": 0.6},
}


# --------------------------------------------------------------------------
# Atmosphere effects
# --------------------------------------------------------------------------
# Each entry seeds an AtmosphereEffect; the values stay editable afterwards.

EFFECT_PRESETS: Dict[str, Dict[str, Any]] = {
    "light_rain": {"label": "Light Rain",
                   "values": {"type": "rain", "intensity": 0.35, "speed": 0.85}},
    "heavy_rain": {"label": "Heavy Rain",
                   "values": {"type": "rain", "intensity": 0.9, "speed": 1.3}},
    "snow": {"label": "Snowfall",
             "values": {"type": "snow", "intensity": 0.55, "speed": 1.0}},
    "storm": {"label": "Lightning Storm",
              "values": {"type": "lightning", "intensity": 0.7, "speed": 1.0}},
    "golden_hour": {"label": "Golden Hour",
                    "values": {"type": "sunlight", "intensity": 0.55, "speed": 1.0,
                               "color": "0xFFD9A0FF"}},
    "sun_flare": {"label": "Sun Flare",
                  "values": {"type": "sunlight", "intensity": 0.85, "speed": 1.4,
                             "color": "0xFFF2CCFF"}},
    "warm_leak": {"label": "Warm Light Leak",
                  "values": {"type": "light_leak", "intensity": 0.5, "speed": 1.0,
                             "color": "0xFF7A2AFF"}},
    "cool_leak": {"label": "Cool Light Leak",
                  "values": {"type": "light_leak", "intensity": 0.5, "speed": 1.0,
                             "color": "0x4FA8FFFF"}},
    "fog": {"label": "Fog", "values": {"type": "fog", "intensity": 0.45, "speed": 1.0}},
    "wind": {"label": "Wind", "values": {"type": "wind", "intensity": 0.5, "speed": 1.0}},
    "film_grain": {"label": "Film Grain",
                   "values": {"type": "grain", "intensity": 0.35, "speed": 1.0}},
}


# Common cinematic shapes, offered as one-click aspect ratios.
ASPECT_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {"label": "Full frame", "ratio": None},
    "cinemascope": {"label": "Cinematic 2.39:1", "ratio": 2.39},
    "ultrawide": {"label": "Ultrawide 21:9", "ratio": 2.333},
    "widescreen": {"label": "Widescreen 1.85:1", "ratio": 1.85},
    "classic": {"label": "Classic 4:3", "ratio": 1.333},
    "square": {"label": "Square 1:1", "ratio": 1.0},
}


# --------------------------------------------------------------------------
# Lookup helpers
# --------------------------------------------------------------------------

def color_preset_values(name: Optional[str]) -> Dict[str, Any]:
    entry = COLOR_PRESETS.get((name or "").lower())
    return dict(entry["values"]) if entry else {}


def text_preset_style(name: Optional[str]) -> Dict[str, Any]:
    entry = TEXT_PRESETS.get((name or "").lower())
    return dict(entry["style"]) if entry else {}


def caption_preset(name: Optional[str]) -> Dict[str, Any]:
    return CAPTION_PRESETS.get((name or "").lower()) or CAPTION_PRESETS["classic"]


def intro_preset(name: Optional[str]) -> Optional[Dict[str, Any]]:
    return INTRO_PRESETS.get((name or "").lower())


def _listing(source: Dict[str, Dict[str, Any]], *extra_keys: str) -> List[Dict[str, Any]]:
    out = []
    for key, entry in source.items():
        row = {"id": key, "label": entry.get("label", key)}
        for extra in extra_keys:
            if extra in entry:
                row[extra] = entry[extra]
        out.append(row)
    return out


def transition_preset(name: Optional[str]) -> Dict[str, Any]:
    entry = TRANSITION_PRESETS.get((name or "").lower())
    return {k: v for k, v in (entry or {}).items() if k != "label"}


def effect_preset(name: Optional[str]) -> Dict[str, Any]:
    entry = EFFECT_PRESETS.get((name or "").lower())
    return dict(entry["values"]) if entry else {}


def list_all() -> Dict[str, List[Dict[str, Any]]]:
    """Everything the preset pickers need, in one round trip."""
    from render import transitions as transition_lib

    return {
        "color": _listing(COLOR_PRESETS, "values"),
        "text": _listing(TEXT_PRESETS, "style"),
        "caption": _listing(CAPTION_PRESETS, "style", "words_per_caption", "uppercase"),
        "intro": _listing(INTRO_PRESETS, "duration", "background"),
        "transition": _listing(TRANSITION_PRESETS, "type", "duration"),
        "effect": _listing(EFFECT_PRESETS, "values"),
        "aspect": _listing(ASPECT_PRESETS, "ratio"),
        # The full xfade set, grouped, for users who want more than the presets.
        "transition_catalogue": transition_lib.catalogue(),
    }
