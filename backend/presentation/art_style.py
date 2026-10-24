"""The art style: the medium every generated picture, clip, thumbnail scene
and text card is made in -- real (photoreal, the default), explainer (flat
vector), stick_figure (whiteboard marker) or cartoon (2D cel).

Set by the caller (`PresentationSettings.art_style`; BuzzcafStudio resolves it
per channel and project and keeps the same table in
`integrations/art_style.py`). It is applied the way the genre is
(`genre.py`): the prompt-writing model is TOLD the medium, and the medium's
look is APPENDED to every finished prompt anyway, because a local model told
"stick figures" still writes a photograph one time in five. For a non-real
style the genre's photographic look words (film grain, low-key lighting)
are left out and only its mood is kept, the main character's real face is
never swapped in (a real face on a drawing is a mistake, not continuity),
and the text cards take the medium's own fonts and colours.
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple


class ArtStyle(NamedTuple):
    label: str
    look: str          # appended to every positive prompt
    motion: str        # how a clip in this medium moves
    negative: str      # appended to every negative prompt
    thumbnail: str     # medium phrase for the thumbnail scene
    face_swap: bool    # may the main character's real face be swapped in
    card_style: Dict[str, object]   # text-card restyling (timeline text style keys)
    prefix: str = ""   # put BEFORE the scene when the medium must lead (stick figures: the
                       # scene words otherwise win and the model photographs a whiteboard)


STYLES: Dict[str, ArtStyle] = {
    "real": ArtStyle(
        label="Real (cinematic, photoreal)",
        look="",
        motion="slow steady cinematic motion, subtle natural movement, no fast motion, no shaking",
        negative="",
        thumbnail="cinematic film still, one clear subject, close framing, strong contrast",
        face_swap=True,
        card_style={},
    ),
    "explainer": ArtStyle(
        label="Explainer (flat vector illustration)",
        look=("flat vector explainer illustration, clean simple geometric shapes, bold flat colours, "
              "soft gradients, clear silhouettes, minimal detail, generous empty space, "
              "modern infographic style"),
        motion=("smooth simple 2D animation, one element moves at a time, gentle easing, "
                "flat shapes sliding and fading, no camera shake"),
        negative="photograph, photorealistic, realistic skin, 3d render, film grain, clutter",
        thumbnail="flat vector explainer illustration, bold flat colours, one clear subject, clean background",
        face_swap=False,
        card_style={"font_family": "Poppins", "bold": True, "color": "#FFFFFF",
                    "stroke_color": "#1E2A44", "stroke_width": 4, "shadow_x": 0, "shadow_y": 0,
                    "animation": "fade", "animation_duration": 0.3},
    ),
    "stick_figure": ArtStyle(
        label="Stick figure (whiteboard drawing)",
        look=("crude stick figure drawing: people are a circle for the head and five straight black lines "
              "for the body, no face, no clothes, no hands; thin black marker on plain white paper, "
              "childlike whiteboard doodle, objects drawn with three or four lines, no colour fill, "
              "no shading, a single red accent line"),
        motion=("hand-drawn whiteboard animation, the lines appear as if being drawn, slight line "
                "wobble, simple movement of one figure, white background stays still"),
        negative="photograph, photorealistic, colour fill, 3d render, shading, gradients, detailed faces",
        thumbnail="stick figure drawing in black marker on white paper, one red accent, hand-drawn",
        face_swap=False,
        card_style={"font_family": "Segoe Print", "bold": True, "color": "#1A1A1A",
                    "stroke_color": "#FFFFFF", "stroke_width": 6, "shadow_x": 0, "shadow_y": 0,
                    "animation": "fade", "animation_duration": 0.25},
        prefix=("A crude whiteboard doodle in black marker on plain white paper, the whole picture is "
                "the drawing, nothing photographed, nothing in the frame but the one scene described. Any "
                "person is drawn as a stick figure (a circle head, straight lines for the body, no face, "
                "no clothes, no skin colour), and no other figures are added. The doodle shows only:"),
    ),
    "cartoon": ArtStyle(
        label="Cartoon (2D animated)",
        look=("bold 2D cartoon illustration, thick clean outlines, flat cel shading, bright saturated "
              "colours, expressive simplified characters, animated series style"),
        motion=("smooth 2D cartoon animation, expressive movement of one character or object, "
                "flat cel-shaded colours, steady framing, no camera shake"),
        negative="photograph, photorealistic, realistic skin, 3d render, film grain",
        thumbnail="bold 2D cartoon illustration, thick outlines, bright colours, one expressive subject",
        face_swap=False,
        card_style={"font_family": "Impact", "bold": True, "color": "#FFE23A",
                    "stroke_color": "#1A1A1A", "stroke_width": 8,
                    "animation": "pop", "animation_duration": 0.3},
    ),
    "painterly": ArtStyle(
        label="Painterly (soft oil / gouache)",
        look=("soft painterly illustration, loose oil and gouache brushwork, muted earthy palette with "
              "one warm light source, visible brush texture, no outlines, soft edges, atmospheric depth, "
              "quiet and melancholic"),
        motion=("very slow drift, mist and light shifting gently, brush texture steady, no fast motion, "
                "no shaking"),
        negative="photograph, photorealistic, 3d render, hard outlines, cartoon, neon, text, watermark",
        thumbnail="soft painterly illustration, muted palette, one warm light, one clear subject",
        face_swap=False,
        card_style={"font_family": "Georgia", "color": "#F3EAD3",
                    "stroke_color": "#2B2520", "stroke_width": 4,
                    "animation": "fade", "animation_duration": 0.5},
    ),
    # 2026-10-24 (VR8): a soothing hand-drawn look for Life3Baje mood reels,
    # next to "real" -- frame-by-frame Procreate-style animation, never a
    # photograph. `vibe_reel.py`'s own look pass steps a video shot's motion
    # to ~12fps held frames and adds a static paper-grain texture for this
    # style instead of the "real" style's handheld drift/film grain.
    "illustrated": ArtStyle(
        label="Illustrated animation (hand-drawn)",
        look=("soothing hand-drawn digital illustration, frame-by-frame animation style, soft "
              "gouache and pencil brush texture, visible hand-drawn strokes, gentle pastel and "
              "muted colour palette, simple clean shapes, cozy storybook calm, a slight paper "
              "grain, no hard or glossy outlines"),
        motion=("gentle hand-animated movement, drawings held and stepped, soft easing, no fast "
                "motion, no shaking"),
        negative=("photograph, photorealistic, 3d render, CGI, unreal engine, anime, glossy, "
                  "neon, text, watermark"),
        thumbnail="soothing hand-drawn illustration, soft pastel palette, one clear subject, storybook calm",
        face_swap=False,
        card_style={"font_family": "Georgia", "color": "#4A3B2E",
                    "stroke_color": "#FBF3E6", "stroke_width": 4,
                    "animation": "fade", "animation_duration": 0.5},
    ),
}

ART_STYLES: Tuple[str, ...] = tuple(STYLES.keys())
DEFAULT = "real"

_ALIASES = {
    "photoreal": "real", "photo": "real", "realistic": "real", "cinematic": "real", "film": "real",
    "stick": "stick_figure", "stickfigure": "stick_figure", "stick_figures": "stick_figure",
    "whiteboard": "stick_figure", "doodle": "stick_figure",
    "vector": "explainer", "flat": "explainer", "infographic": "explainer",
    "animated": "cartoon", "animation": "cartoon", "toon": "cartoon", "anime": "cartoon",
    "oil": "painterly", "gouache": "painterly", "watercolor": "painterly", "watercolour": "painterly",
    "painting": "painterly",
}

#: Genre mood words that survive into a drawn style (the genre's own `look`
#: is photographic -- fog, film grain, low-key lighting -- and would fight
#: "white paper"). Sparse on purpose; "general" keeps nothing.
_GENRE_MOOD: Dict[str, str] = {
    "horror": "eerie unsettling mood, dark palette",
    "true_crime": "grim tense mood, muted palette",
    "mystery": "mysterious tense mood",
    "documentary": "serious factual tone",
    "motivational": "warm hopeful mood",
    "comedy": "playful bright mood",
    "devotional": "serene warm mood",
    "geopolitics": "serious tone, muted palette",
    "finance": "clean confident tone",
}


def normalise(style: Any) -> str:
    if not style:
        return DEFAULT
    slug = str(style).strip().lower().replace("-", "_").replace(" ", "_")
    if slug in STYLES:
        return slug
    return _ALIASES.get(slug, DEFAULT)


def get(style: Any) -> ArtStyle:
    return STYLES[normalise(style)]


def is_real(style: Any) -> bool:
    return normalise(style) == DEFAULT


def uses_face_swap(style: Any) -> bool:
    return get(style).face_swap


def look_for(style: Any) -> str:
    return get(style).look


def negative_for(style: Any) -> str:
    return get(style).negative


def prefix_for(style: Any) -> str:
    return get(style).prefix


def thumbnail_look(style: Any) -> str:
    return get(style).thumbnail


def genre_mood(genre: Optional[str]) -> str:
    return _GENRE_MOOD.get(str(genre or "").lower(), "")


def card_style(style: Any) -> Dict[str, object]:
    return dict(get(style).card_style)


def block(style: Any) -> str:
    """The paragraph appended to prompt-writing system prompts for a non-real
    style ("" for real), next to `genre.genre_block`."""
    name = normalise(style)
    if name == DEFAULT:
        return ""
    s = STYLES[name]
    return (
        f"\n\nART STYLE: every picture in this video is made as {s.label.lower()}: {s.look}. "
        f"Describe every scene in that medium -- never as a photograph or a film still, with no "
        f"lens, lighting or texture words. Clip motion: {s.motion}. style_hint is always "
        f"'illustration'. End every image_prompt and video_prompt with: \"{s.look}\"."
    )
