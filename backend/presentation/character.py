"""One face for the story's main character.

A story video follows one person, but every beat's picture is generated from
its own prompt, so the inspector in shot 3 is a different man from the
inspector in shot 7. This module keeps him one man:

1. `find_main_character` asks the model who the story follows and what they
   look like (or that it follows no one -- an explainer, a list, real people).
2. The beat writer tags each beat `shows_character` (shotplan.write_beats),
   and `apply_to_beats` keeps the description in every tagged prompt.
3. The asset stage makes one reference portrait per project and swaps that
   face into each tagged still (`with_face_swap`, ReActor in ComfyUI). A clip
   starts from its still, so it keeps the face too.

Beats that show anyone else -- another character, a crowd, an empty room --
are never swapped.
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional

from .models import Beat, Character, Program, Topic
from .program import transcript_lines

logger = logging.getLogger("presentation.character")

CHARACTER_SYSTEM = (
    "You read a YouTube video's transcript and decide whether it tells a story that "
    "follows ONE main character on screen -- the protagonist the pictures would keep "
    "showing (in a first-person story, the narrator as they appear in the story).\n\n"
    "Answer has_character false when there is no such person: an explainer, a list, "
    "news, a review, a story about places or events with no lead, or when the lead is "
    "a real, identifiable public person.\n\n"
    "When true:\n"
    "  name        - what the story calls them (a role like 'the night guard' is fine)\n"
    "  description - how they look, for an image generator, in ENGLISH, 20 to 35 "
    "words: approximate age, gender, ethnicity matching the story's setting, build, "
    "face, hair, facial hair, and the clothing they wear through the story. Invent "
    "plausible details the story does not give, and keep them ordinary and specific. "
    "No name in the description.\n\n"
    'Answer with JSON only: {"has_character": false, "name": "", "description": ""}'
)

CHARACTER_SCHEMA = {
    "name": "main_character",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "has_character": {"type": "boolean"},
            "name": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["has_character", "name", "description"],
        "additionalProperties": False,
    },
}

# Enough of the story to know its lead without paying for the whole transcript.
EXCERPT_LINES = 120

# The beat writer's extra instructions when the story has a lead.
def beat_block(character: Character) -> str:
    return (
        "\n\nMAIN CHARACTER: this story follows " + character.name + ", who looks like: "
        + character.description + ".\n"
        "For every beat also return shows_main_character:\n"
        "  true  - the main character is the visible subject of the shot, face in view.\n"
        "  false - anyone else (another character, a crowd, a stranger), a place, an "
        "object, hands only, or a back view.\n"
        "When true, describe the main character in the prompt with the appearance above, "
        "never by name."
    )


async def find_main_character(program: Program, ask, topics: Optional[List[Topic]] = None
                              ) -> Optional[Character]:
    """The story's lead, or None when it follows no one (or the model is unsure)."""
    from .shotplan import _ask, first_json_object
    lines = transcript_lines(program).splitlines()
    if not lines and not topics:
        return None
    excerpt = "\n".join(lines[:EXCERPT_LINES])
    if topics:
        excerpt += "\n\nTopics:\n" + "\n".join(f"- {t.topic}: {t.summary}" for t in topics[:40])
    try:
        answer = await _ask(ask, CHARACTER_SYSTEM, f"Transcript:\n{excerpt}\n\nAnswer:", CHARACTER_SCHEMA)
    except Exception as e:
        logger.warning("Main character: the model call failed (%s); no character", e)
        return None
    parsed = first_json_object(answer or "") or {}
    name = str(parsed.get("name") or "").strip()
    description = _clean(str(parsed.get("description") or ""))
    if not parsed.get("has_character") or not name or len(description.split()) < 6:
        logger.info("Main character: none (%s)", "model said none" if not parsed.get("has_character")
                    else "answer too thin")
        return None
    logger.info("Main character: %s -- %s", name, description)
    return Character(name=name[:60], description=description)


def from_settings(raw: Optional[Dict[str, Any]]) -> Optional[Character]:
    """PresentationSettings.main_character -> Character; {} or a thin entry
    means no lead."""
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    description = _clean(str(raw.get("description") or ""))
    if not name or len(description.split()) < 6:
        return None
    gender = str(raw.get("gender") or "").strip().lower()
    return Character(name=name[:60], description=description,
                     gender=gender if gender in ("male", "female") else "")


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return " ".join(text.split()[:45])


def apply_to_beats(beats: List[Beat], character: Optional[Character]) -> List[Beat]:
    """Every tagged beat's prompts carry the description (the shot-variation
    pass rewrites prompts and drops it); untagged beats are left alone. With no
    character, every tag is cleared."""
    out = []
    for beat in beats:
        if not beat.shows_character:
            out.append(beat)
            continue
        if character is None or not beat.is_generated:
            out.append(beat.model_copy(update={"shows_character": False}))
            continue
        out.append(beat.model_copy(update={
            "image_prompt": _with_description(beat.image_prompt, character),
            "video_prompt": _with_description(beat.video_prompt, character),
        }))
    return out


def _with_description(prompt: Optional[str], character: Character) -> Optional[str]:
    if not prompt:
        return prompt
    if character.description.lower()[:40] in prompt.lower():
        return prompt
    return f"{character.description}, {prompt}"


# --- the swap, inside a still's ComfyUI graph ---------------------------------

FACE_SWAP_CLASS = "ReActorFaceSwap"
_SAVE_IMAGE = ("SaveImage", "Image Save")
# Measured 2026-10-04 (data/bench/2026-10-04_character): swapped stills match the
# reference at 0.80-0.86 cosine (insightface), unswapped ones 0.41-0.50.
FACE_SWAP_INPUTS = {
    "enabled": True,
    "swap_model": "inswapper_128.onnx",
    "facedetection": "retinaface_resnet50",
    "face_restore_model": "codeformer-v0.1.0.pth",
    "face_restore_visibility": 1.0,
    "codeformer_weight": 0.7,
    "detect_gender_input": "no",
    "detect_gender_source": "no",
    "input_faces_index": "0",
    "source_faces_index": "0",
    "console_log_level": 1,
}


def with_face_swap(graph: Dict[str, Any], reference_image: str, gender: str = "") -> Dict[str, Any]:
    """A copy of a still graph whose saved image has the reference face swapped
    in. `reference_image` is a file name in ComfyUI's input folder. Only the
    largest face is swapped (ReActor's default order), and with `gender` known
    only a face of that gender. A picture with no face passes through
    unchanged; a graph with no image save node comes back unchanged."""
    save_id = next((nid for nid, node in graph.items() if isinstance(node, dict)
                    and node.get("class_type") in _SAVE_IMAGE
                    and isinstance((node.get("inputs") or {}).get("images"), list)), None)
    if save_id is None:
        return graph
    out = copy.deepcopy(graph)
    out["bz_face_ref"] = {"class_type": "LoadImage", "inputs": {"image": reference_image}}
    filter_gender = gender if gender in ("male", "female") else "no"
    out["bz_face_swap"] = {"class_type": FACE_SWAP_CLASS, "inputs": {
        **FACE_SWAP_INPUTS,
        "detect_gender_input": filter_gender,
        "detect_gender_source": filter_gender,
        "input_image": out[save_id]["inputs"]["images"],
        "source_image": ["bz_face_ref", 0],
    }}
    out[save_id]["inputs"]["images"] = ["bz_face_swap", 0]
    return out
