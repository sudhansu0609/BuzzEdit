"""System font discovery for the text tool.

FFmpeg's drawtext wants a concrete `fontfile=` path, but users pick a *family*
("Montserrat") and a weight ("Bold"). This module walks the platform font
directories, reads each file's `name` table to get the real family/subfamily,
and groups the results so a family + bold/italic pair resolves to one file.

The name-table parse is done by hand rather than pulling in fontTools: it is a
well-specified ~50 lines and this backend already carries enough heavyweight
dependencies. Anything that fails to parse falls back to its filename, so a
malformed font degrades to a usable entry instead of vanishing.
"""

import logging
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_FONT_EXTS = {".ttf", ".otf", ".ttc", ".otc"}

# Families worth surfacing first — they exist on stock Windows installs and read
# well at title sizes, so the picker opens on something usable.
_PREFERRED = [
    "Montserrat", "Poppins", "Bebas Neue", "Oswald", "Roboto", "Open Sans",
    "Inter", "Impact", "Anton", "Bahnschrift", "Segoe UI", "Arial",
    "Arial Black", "Verdana", "Tahoma", "Georgia", "Times New Roman",
    "Trebuchet MS", "Calibri", "Cambria", "Consolas", "Courier New", "Franklin Gothic",
]

_cache: Optional[List[Dict]] = None


def _font_dirs() -> List[Path]:
    dirs: List[Path] = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(windir) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"),
                 Path.home() / "Library" / "Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                 Path.home() / ".fonts", Path.home() / ".local/share/fonts"]
    # Fonts bundled with the app take precedence over anything installed.
    bundled = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
    if bundled.exists():
        dirs.insert(0, bundled)
    return [d for d in dirs if d.exists()]


def _decode_name(raw: bytes, platform_id: int, encoding_id: int) -> str:
    try:
        if platform_id == 3 and encoding_id in (0, 1, 10):
            return raw.decode("utf-16-be", errors="ignore")
        if platform_id == 0:
            return raw.decode("utf-16-be", errors="ignore")
        return raw.decode("latin-1", errors="ignore")
    except Exception:
        return ""


def _read_name_table(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return (family, subfamily) from a font file's name table, or (None, None)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
            if head == b"ttcf":
                # Font collection: read the offset of the first contained font.
                fh.seek(12)
                first_offset = struct.unpack(">I", fh.read(4))[0]
                fh.seek(first_offset)
                fh.read(4)  # sfntVersion of the contained font
            elif head not in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"):
                return None, None

            num_tables = struct.unpack(">H", fh.read(2))[0]
            fh.read(6)  # searchRange, entrySelector, rangeShift

            name_offset = None
            for _ in range(num_tables):
                record = fh.read(16)
                if len(record) < 16:
                    return None, None
                tag, _checksum, offset, _length = struct.unpack(">4sIII", record)
                if tag == b"name":
                    name_offset = offset
                    break
            if name_offset is None:
                return None, None

            fh.seek(name_offset)
            _fmt, count, string_offset = struct.unpack(">HHH", fh.read(6))
            records = []
            for _ in range(count):
                chunk = fh.read(12)
                if len(chunk) < 12:
                    break
                records.append(struct.unpack(">HHHHHH", chunk))

            family = subfamily = None
            for platform_id, encoding_id, _lang, name_id, length, offset in records:
                if name_id not in (1, 2, 16, 17):
                    continue
                fh.seek(name_offset + string_offset + offset)
                value = _decode_name(fh.read(length), platform_id, encoding_id).strip()
                if not value:
                    continue
                # 16/17 are the "typographic" names and are more accurate for
                # families with many weights, so let them override 1/2.
                if name_id == 16 or (name_id == 1 and family is None):
                    family = value
                elif name_id == 17 or (name_id == 2 and subfamily is None):
                    subfamily = value
            return family, subfamily
    except Exception:
        return None, None


def _style_flags(subfamily: Optional[str], filename: str) -> Tuple[bool, bool]:
    probe = f"{subfamily or ''} {filename}".lower()
    bold = "bold" in probe or probe.endswith("bd") or "-bd" in probe or "black" in probe
    italic = "italic" in probe or "oblique" in probe
    return bold, italic


def scan_fonts(force: bool = False) -> List[Dict]:
    """Enumerate installed fonts grouped by family.

    Each entry: {family, regular, bold, italic, bold_italic, styles: [...]}.
    Cached after the first scan — walking a Windows font directory costs a
    couple hundred milliseconds and the set does not change mid-session.
    """
    global _cache
    if _cache is not None and not force:
        return _cache

    families: Dict[str, Dict] = {}
    for directory in _font_dirs():
        try:
            entries = sorted(directory.rglob("*"))
        except Exception as exc:
            logger.warning("Could not read font directory %s: %s", directory, exc)
            continue
        for path in entries:
            if not path.is_file() or path.suffix.lower() not in _FONT_EXTS:
                continue
            family, subfamily = _read_name_table(path)
            if not family:
                family = path.stem.replace("_", " ").replace("-", " ").title()
            bold, italic = _style_flags(subfamily, path.name)

            slot = families.setdefault(family, {
                "family": family, "regular": None, "bold": None,
                "italic": None, "bold_italic": None, "styles": [],
            })
            key = ("bold_italic" if bold and italic else
                   "bold" if bold else "italic" if italic else "regular")
            if slot[key] is None:
                slot[key] = str(path)
            style_name = subfamily or key.replace("_", " ").title()
            if style_name not in slot["styles"]:
                slot["styles"].append(style_name)

    # A family with only italic/bold files still needs a default to draw with.
    for slot in families.values():
        if slot["regular"] is None:
            slot["regular"] = slot["bold"] or slot["italic"] or slot["bold_italic"]

    order = {name.lower(): i for i, name in enumerate(_PREFERRED)}
    result = sorted(
        (f for f in families.values() if f["regular"]),
        key=lambda f: (order.get(f["family"].lower(), len(order)), f["family"].lower()),
    )
    _cache = result
    logger.info("Discovered %d font families", len(result))
    return result


def resolve_font_file(family: Optional[str], bold: bool = False, italic: bool = False) -> Optional[str]:
    """Best font file for a family + weight, falling back through the preferred list."""
    fonts = scan_fonts()
    if not fonts:
        return None
    index = {f["family"].lower(): f for f in fonts}
    slot = index.get((family or "").lower().strip())
    if slot is None:
        # Substring match handles "Segoe UI" vs "Segoe UI Variable" style drift.
        needle = (family or "").lower().strip()
        if needle:
            slot = next((f for f in fonts if needle in f["family"].lower()), None)
    if slot is None:
        slot = fonts[0]

    if bold and italic:
        return slot["bold_italic"] or slot["bold"] or slot["italic"] or slot["regular"]
    if bold:
        return slot["bold"] or slot["regular"]
    if italic:
        return slot["italic"] or slot["regular"]
    return slot["regular"]
