"""Fonts and preset catalogues for the inspector panels.

These are process-global (they describe the machine and the built-in looks, not a
project), so they live outside the project-scoped timeline router.
"""

from fastapi import APIRouter, Query

from timeline import presets as preset_lib
from utils.fonts import resolve_font_file, scan_fonts

router = APIRouter()


@router.get("/fonts")
async def list_fonts(refresh: bool = Query(False, description="Re-scan the font directories")):
    """Installed font families, each with its regular/bold/italic files."""
    families = scan_fonts(force=refresh)
    return {"count": len(families), "families": families}


@router.get("/fonts/resolve")
async def resolve_font(family: str, bold: bool = False, italic: bool = False):
    """The exact font file drawtext would load for a family + weight."""
    return {"family": family, "bold": bold, "italic": italic,
            "font_file": resolve_font_file(family, bold=bold, italic=italic)}


@router.get("/")
async def list_presets():
    """Every preset catalogue — colour, text, caption and intro — in one call."""
    return preset_lib.list_all()


@router.get("/color")
async def list_color_presets():
    return {"presets": preset_lib.list_all()["color"]}


@router.get("/text")
async def list_text_presets():
    return {"presets": preset_lib.list_all()["text"]}


@router.get("/caption")
async def list_caption_presets():
    return {"presets": preset_lib.list_all()["caption"]}


@router.get("/intro")
async def list_intro_presets():
    return {"presets": preset_lib.list_all()["intro"]}
