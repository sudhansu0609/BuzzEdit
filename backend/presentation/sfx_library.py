"""The creator's own sound-effects library, indexed by what each file is.

`sound.py` knows effects by a fixed tag (`data/sfx/<tag>/`: whoosh, pop,
stinger...). A scene sound is free text -- "footsteps running on wet leaves",
"a jeep engine idling" -- so this module indexes every audio file under the
library roots (`data/sfx` plus the `sfx_library_dirs` app setting, by default
the creator's Effects folder: 1813 files whose folders are the tags --
footSteps, rain, "door knock", heartBeat...), maps the words of an effect onto
canonical tags (`CANON`), and picks the file whose folder and name share the
most of them (`resolve_effect`).

Durations are probed once (ffprobe) and cached in `data/sfx_index.json` by
path + mtime, so a rerun probes only new files. A file is used only with a
licence on record: "own" for the creator's folder and `data/sfx` (declared
usable by the creator 2026-10-09), else its folder manifest's.

An effect the library has nothing for can be made by Stable Audio, but only
ahead of placement: `warm_effects` (async, card handed over once) fills the
generation cache and `resolve_effect` -- called from the synchronous
`sound.apply_sound` -- only reads it, noise-guarded, the way `sound._resolve_sfx`
does for the tags. Every placed file is loudness-normalised once into
`data/sfx_cache/` (`prepare_effect`): -18 LUFS for an accent, -24 for a bed.
"""

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config import DATA_DIR, FFMPEG_BIN
from utils.proc import NO_WINDOW

logger = logging.getLogger("presentation.sfx_library")

LIBRARY_DIR = Path(DATA_DIR) / "sfx"
# The creator's own folder; used when the `sfx_library_dirs` app setting is unset.
CREATOR_LIBRARY = Path(r"B:\youtubeProjects\Khayal3Baje\Effects\Audios")
INDEX_PATH = Path(DATA_DIR) / "sfx_index.json"
CACHE_DIR = Path(DATA_DIR) / "sfx_cache"

# A folder named like these holds music, not effects ("chill audios": 778 tracks).
_MUSIC_FOLDERS = {"chillaudio", "autotempo"}
# A loose root file longer than this whose name reads "Title - Artist" is a
# YouTube Audio Library track.
MUSIC_MIN_S = 60.0
_AMBIENCE_TAGS = {"rain", "wind", "crickets", "frogs", "ambient", "room", "forest", "water"}
AMBIENCE_MIN_S = 20.0
# An accent prefers an effect no longer than this when one matches.
ACCENT_PREFER_MAX_S = 20.0
# Lengths a generated effect is made at (and cached by).
GEN_ACCENT_S = 4.0
GEN_BED_S = 20.0
# Loudness targets for the normalised copies, and how much of a file is kept:
# an accent plays at most a few seconds and a bed loops, so a long file is not
# normalised whole (areverse buffers the entire stream).
LOUDNESS_I = {"accent": -18.0, "bed": -24.0}
LOUDNESS_TP = -1.5
EDGE_FADE_S = 0.015
PREPARE_MAX_S = {"accent": 30.0, "bed": 120.0}
# Probing 1813 files once: in parallel. The index is re-walked after this long.
PROBE_WORKERS = 8
INDEX_MEMO_S = 120.0

# Effect word -> canonical tag. Two-word keys are phrases; "fire" is a gunshot
# only next to a gun (see `canonical_tags`).
_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "footsteps": ("walk", "walking", "steps", "step", "foot", "boots", "gravel", "footstep"),
    "running": ("run", "sprint", "chase", "chasing"),
    "shout": ("yell", "yelling", "shouting", "scream", "screams", "screaming", "cry out"),
    "thunder": ("storm", "lightning"),
    "rain": ("drizzle", "downpour", "raining"),
    "wind": ("howl", "howling", "breeze", "gust"),
    "door": ("creak", "knock", "slam", "bang", "hinge"),
    "phone": ("ring", "ringing", "mobile", "cell"),
    "car": ("jeep", "engine", "vehicle", "drive", "driving", "truck"),
    "bike": ("motorcycle", "scooter"),
    "train": (),
    "bell": ("temple", "bells"),
    "dog": ("bark", "barking"),
    "birds": ("bird", "chirp", "crow"),
    "crickets": ("insects", "cicada", "night insects"),
    "owl": (),
    "bats": ("bat",),
    "heartbeat": ("heart", "pulse"),
    "knife": ("blade", "stab"),
    "gunshot": ("gun", "gunshots", "shot"),
    "fall": ("thud", "drop", "falling", "collapse"),
    "glass": ("shatter", "break"),
    "whisper": ("whispers", "whispering"),
    "laugh": ("laughs", "laughter", "giggle"),
    "weep": ("cry", "crying", "sob", "sobbing"),
    "water": ("drip", "dripping", "river", "lapping", "splash"),
    "fire": ("crackle", "flames", "burning"),
    "crowd": ("chatter", "gossip", "kids", "people", "market"),
}
CANON: Dict[str, str] = {}
for _canonical, _words in _SYNONYMS.items():
    CANON[_canonical] = _canonical
    for _word in _words:
        CANON[_word] = _canonical
_PHRASES = {key: value for key, value in CANON.items() if " " in key}

_STOP = {"the", "and", "with", "from", "into", "onto", "over", "under", "for", "its",
         "his", "her", "their", "some", "very", "sound", "sounds", "effect", "effects",
         "sfx", "audio", "are", "was", "this", "that"}
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_RE = re.compile(r"[\W_]+")


def words(text: str) -> List[str]:
    """Lower-case word tokens of a folder name, a file name or an effect:
    split on camelCase, spaces, underscores and hyphens; numbers, stop words
    and one- or two-letter tokens dropped."""
    out: List[str] = []
    for piece in _SPLIT_RE.split(text or ""):
        for part in _CAMEL_RE.split(piece):
            token = part.lower()
            if len(token) < 3 or token.isdigit() or token in _STOP:
                continue
            out.append(token)
    return out


def canonical_tags(tokens: Sequence[str]) -> Set[str]:
    """The canonical tags a run of tokens names: walking -> footsteps,
    "cry out" -> shout (not weep), "night insects" -> crickets, and "fire" a
    gunshot only when a gun is named too."""
    tokens = list(tokens)
    out: Set[str] = set()
    used: Set[int] = set()
    for i in range(len(tokens) - 1):
        phrase = f"{tokens[i]} {tokens[i + 1]}"
        if phrase in _PHRASES:
            out.add(_PHRASES[phrase])
            used.update((i, i + 1))
    gun = any(CANON.get(t) == "gunshot" for t in tokens)
    for i, token in enumerate(tokens):
        if i in used:
            continue
        if token == "fire" and gun:
            out.add("gunshot")
        elif token in CANON:
            out.add(CANON[token])
    return out


def _name_part(stem: str) -> str:
    """A freesound "191102__bia87__foot-steps-on-gravel" keeps only its name."""
    parts = stem.split("__")
    if len(parts) >= 3 and parts[0].strip().isdigit():
        return "__".join(parts[2:])
    if len(parts) == 2 and parts[0].strip().isdigit():
        return parts[1]
    return stem


# --- the index -----------------------------------------------------------------

@dataclass
class Entry:
    path: str
    duration_s: float
    tags: List[str]
    category: str            # "effect" | "music" | "ambience"
    license: str             # "" = no licence on record: never used
    # Where each tag came from, for the score (+2 a folder tag, +1 a name tag).
    folder_tags: List[str] = field(default_factory=list)
    name_tags: List[str] = field(default_factory=list)


def library_roots(extra: Optional[Sequence[str]] = None) -> List[Path]:
    """`data/sfx` plus `extra` (a job's `sfx_library_dirs`) when given, else
    the `sfx_library_dirs` app setting, else the creator's folder."""
    dirs = [str(d) for d in (extra or []) if str(d).strip()]
    if not dirs:
        configured: Any = None
        try:
            from store.app_settings import AppSettings
            configured = AppSettings().get("sfx_library_dirs")
        except Exception as e:
            logger.warning("Could not read the sfx_library_dirs app setting (%s)", e)
        if isinstance(configured, str):
            configured = [configured]
        dirs = [str(d) for d in (configured or []) if str(d).strip()]
    if not dirs:
        dirs = [str(CREATOR_LIBRARY)]
    roots: List[Path] = []
    for root in [LIBRARY_DIR] + [Path(d) for d in dirs]:
        if all(_norm(root) != _norm(r) for r in roots):
            roots.append(root)
    return roots


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _walk(root: Path) -> List[Path]:
    from .sound import AUDIO_EXTS
    found: List[Path] = []
    if not root.is_dir():
        return found
    for folder, subfolders, names in os.walk(root):
        subfolders.sort()
        for name in sorted(names):
            if Path(name).suffix.lower() in AUDIO_EXTS:
                found.append(Path(folder) / name)
    return found


def _probe(path: str) -> float:
    try:
        from utils.ffmpeg_utils import get_video_duration
        return max(0.0, float(get_video_duration(path) or 0.0))
    except Exception:
        return 0.0


def _load_durations() -> Dict[str, float]:
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("Unreadable SFX index %s (%s); probing again", INDEX_PATH, e)
        return {}
    durations = data.get("durations") if isinstance(data, dict) else None
    out: Dict[str, float] = {}
    for key, value in (durations or {}).items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _save_durations(durations: Dict[str, float]) -> None:
    try:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        part = INDEX_PATH.with_name(INDEX_PATH.name + ".part")
        part.write_text(json.dumps({"version": 1, "durations": durations}, ensure_ascii=False),
                        encoding="utf-8")
        os.replace(part, INDEX_PATH)
    except Exception as e:
        logger.warning("Could not save the SFX index %s (%s)", INDEX_PATH, e)


def _folder_key(name: str) -> str:
    return _SPLIT_RE.sub("", name).lower().rstrip("s")


def _category(folders: Sequence[str], loose: bool, stem: str, duration_s: float,
              tags: Set[str]) -> str:
    if any(_folder_key(f) in _MUSIC_FOLDERS for f in folders):
        return "music"
    if loose and duration_s > MUSIC_MIN_S and " - " in stem:
        return "music"
    if tags & _AMBIENCE_TAGS and duration_s >= AMBIENCE_MIN_S:
        return "ambience"
    return "effect"


def _license(path: Path, own: Sequence[str], manifests: Dict[str, Dict]) -> str:
    norm = _norm(path)
    if any(norm == o or norm.startswith(o.rstrip(os.sep) + os.sep) for o in own):
        return "own"
    folder = str(path.parent)
    if folder not in manifests:
        from .sound import _manifest
        manifests[folder] = _manifest(path.parent)
    entry = manifests[folder].get(path.name) or {}
    return str(entry.get("license") or "").strip().lower()


def _entry(path: Path, root: Path, duration_s: float, own: Sequence[str],
           manifests: Dict[str, Dict]) -> Entry:
    try:
        folders = list(path.relative_to(root).parts[:-1])
    except ValueError:
        folders = []
    folder_words = [w for f in folders for w in words(f)]
    name_words = words(_name_part(path.stem))
    folder_tags = set(folder_words) | canonical_tags(folder_words)
    name_tags = set(name_words) | canonical_tags(name_words)
    tags = folder_tags | name_tags
    return Entry(path=str(path), duration_s=round(duration_s, 3), tags=sorted(tags),
                 category=_category(folders, not folders, path.stem, duration_s, tags),
                 license=_license(path, own, manifests),
                 folder_tags=sorted(folder_tags), name_tags=sorted(name_tags))


_memo: Dict[Tuple[str, ...], Tuple[float, List[Entry]]] = {}


def index(roots: Optional[Sequence[Path]] = None, refresh: bool = False) -> List[Entry]:
    """Every audio file under the roots (recursive), with its duration, tags,
    category and licence. Durations come from `data/sfx_index.json` (by path
    + mtime) and only new or changed files are probed; the walk itself is
    reused for INDEX_MEMO_S. `refresh` walks and probes everything again."""
    roots = [Path(r) for r in (roots if roots is not None else library_roots())]
    key = tuple(_norm(r) for r in roots)
    now = time.monotonic()
    if not refresh and key in _memo and now - _memo[key][0] < INDEX_MEMO_S:
        return _memo[key][1]
    stored = _load_durations()
    known = {} if refresh else stored
    files: List[Tuple[Path, Path, str]] = []
    seen: Set[str] = set()
    for root in roots:
        for path in _walk(root):
            norm = _norm(path)
            if norm in seen:
                continue
            seen.add(norm)
            try:
                stamp = f"{path}|{path.stat().st_mtime_ns}"
            except OSError:
                continue
            files.append((path, root, stamp))
    missing = [(path, stamp) for path, _root, stamp in files if stamp not in known]
    if missing:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
            probed = list(pool.map(_probe, [str(path) for path, _stamp in missing]))
        current = {stamp.rsplit("|", 1)[0]: stamp for _path, _root, stamp in files}
        for old in list(stored):
            path_part = old.rsplit("|", 1)[0]
            if path_part in current and current[path_part] != old:
                del stored[old]
        for (_path, stamp), seconds in zip(missing, probed):
            stored[stamp] = round(seconds, 3)
        _save_durations(stored)
    # Every root being indexed counts as the creator's own folder: a folder the
    # creator configures (`sfx_library_dirs`) is declared usable, the same as
    # `data/sfx` and the creator's default Effects folder.
    own = [_norm(r) for r in roots] + [_norm(LIBRARY_DIR), _norm(CREATOR_LIBRARY)]
    manifests: Dict[str, Dict] = {}
    entries = [_entry(path, root, stored.get(stamp, 0.0), own, manifests)
               for path, root, stamp in files]
    _memo[key] = (now, entries)
    logger.info("SFX library: %d files under %d roots (%d probed)",
                len(entries), len(roots), len(missing))
    return entries


def summary(roots: Optional[Sequence[Path]] = None, refresh: bool = False) -> dict:
    """The library at a glance, for the Settings page: the configured folders,
    the default, the roots actually indexed, file counts by category, the top
    folder tags, any configured folder that is no longer there, and when this
    was indexed."""
    actual_roots = [Path(r) for r in roots] if roots is not None else library_roots()
    entries = index(actual_roots, refresh=refresh)
    configured: Any = None
    try:
        from store.app_settings import AppSettings
        configured = AppSettings().get("sfx_library_dirs")
    except Exception as e:
        logger.warning("Could not read the sfx_library_dirs app setting (%s)", e)
    if isinstance(configured, str):
        configured = [configured]
    dirs = [str(d) for d in (configured or []) if str(d).strip()]
    missing = [d for d in dirs if not Path(d).is_dir()]
    tag_counts: Counter = Counter()
    category_counts: Counter = Counter()
    for entry in entries:
        tag_counts.update(entry.folder_tags)
        category_counts[entry.category] += 1
    top_tags = [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(40)]
    return {
        "dirs": dirs,
        "default": [str(CREATOR_LIBRARY)],
        "roots": [str(r) for r in actual_roots],
        "count": len(entries),
        "effects": category_counts.get("effect", 0),
        "beds": category_counts.get("ambience", 0),
        "music": category_counts.get("music", 0),
        "tags": top_tags,
        "missing": missing,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


# --- resolving an effect -------------------------------------------------------

def _score(wanted: Set[str], entry: Entry) -> int:
    folder, name = set(entry.folder_tags), set(entry.name_tags)
    return sum((2 if tag in folder else 0) + (1 if tag in name else 0) for tag in wanted)


def _gen_prompt(text: str) -> str:
    return " ".join(text.split()) + " sound effect"


def _gen_seconds(kind: str) -> float:
    return GEN_BED_S if kind == "bed" else GEN_ACCENT_S


def cached_effect(text: str, duration: float) -> Optional[Path]:
    """The cached Stable Audio take of this effect, if `warm_effects` made one
    that is not noise. Read-only: safe from the synchronous placement."""
    from .sound import _cached_generated, sfx_is_noise
    path = _cached_generated(_gen_prompt(text), duration)
    if path is None or sfx_is_noise(path):
        return None
    return path


def resolve_effect(text: str, rng: random.Random, kind: str = "accent",
                   min_s: float = 0.3, max_s: Optional[float] = None,
                   roots: Optional[Sequence[Path]] = None) -> Tuple[Optional[Path], int, str]:
    """(file, score, source) for an effect in words.

    The library entry sharing the most canonical tags with the words (+2 per
    folder tag, +1 per file-name tag; licensed files only, never music; a bed
    may use an ambience entry, an accent prefers an effect of at most
    ACCENT_PREFER_MAX_S), ties broken by `rng` -> "library". Else a cached
    generation -> "generated". Else (None, 0, "none")."""
    kind = "bed" if kind == "bed" else "accent"
    tokens = words(text)
    wanted = set(tokens) | canonical_tags(tokens)
    if wanted:
        try:
            entries = index(roots)
        except Exception as e:
            logger.warning("SFX library could not be indexed (%s)", e)
            entries = []
        from .music_gen import ALLOWED_LICENSES
        allowed = ("effect", "ambience") if kind == "bed" else ("effect",)
        scored: List[Tuple[int, Entry]] = []
        for entry in entries:
            if entry.category not in allowed or entry.license not in ALLOWED_LICENSES:
                continue
            if entry.duration_s < min_s or (max_s is not None and entry.duration_s > max_s):
                continue
            score = _score(wanted, entry)
            if score > 0:
                scored.append((score, entry))
        if kind == "accent":
            short = [(s, e) for s, e in scored if e.duration_s <= ACCENT_PREFER_MAX_S]
            scored = short or scored
        if scored:
            best = max(s for s, _e in scored)
            ties = sorted((e for s, e in scored if s == best), key=lambda e: e.path)
            return Path(rng.choice(ties).path), best, "library"
    try:
        path = cached_effect(text, _gen_seconds(kind))
    except Exception as e:
        logger.warning("Generated-effect lookup failed for %r (%s)", text, e)
        path = None
    if path is not None:
        return path, 0, "generated"
    return None, 0, "none"


async def generated_effect(text: str, duration: float, seed: int = 0) -> Optional[Path]:
    """Make this effect with Stable Audio (cached by prompt + duration); a
    take that is noise is deleted and nothing is returned."""
    from .sound import AUDIO_GEN_DIR, _SFX_GEN_NEGATIVE, _generate_audio, sfx_is_noise
    made = await _generate_audio(_gen_prompt(text), duration, seed, negative=_SFX_GEN_NEGATIVE)
    if made is None:
        return None
    if sfx_is_noise(made):
        if made.parent == AUDIO_GEN_DIR:
            try:
                made.unlink()
            except OSError:
                pass
        return None
    return made


async def warm_effects(cues: Sequence[Any], seed: int, comfyui_online: bool = True,
                       library_dirs: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Before placement: generate the scene sounds nothing in the library or
    the cache matches. Like `sound.warm_generated_cache`: nothing when ComfyUI
    is offline or no "audio" workflow exists, the card handed over once, and
    every failure only means one cue fewer."""
    import asyncio
    counts = {"generated": 0, "failed": 0}
    roots = library_roots(library_dirs)
    try:
        # Index off the event loop: the first run probes every file.
        await asyncio.to_thread(index, roots)
    except Exception as e:
        logger.warning("SFX library could not be indexed (%s)", e)
    if not comfyui_online or not cues:
        return counts
    from . import workflows
    if workflows.resolve("audio") is None:
        return counts
    rng = random.Random(seed)
    wanted: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for cue in cues:
        text = " ".join(str(getattr(cue, "effect", "") or getattr(cue, "text", "") or "").split())
        kind = "bed" if getattr(cue, "kind", "accent") == "bed" else "accent"
        if not text or (text.lower(), kind) in seen:
            continue
        seen.add((text.lower(), kind))
        try:
            path, _found_score, _source = resolve_effect(text, rng, kind=kind, roots=roots)
        except Exception as e:
            logger.warning("Scene sound %r not resolved (%s)", text, e)
            continue
        if path is None:
            wanted.append((text, kind))
    if not wanted:
        return counts
    from runtime import gpu_handover
    # Stable Audio Open is a different model: swap it in cleanly, once.
    await gpu_handover.prepare_for_phase("sfx", need_vram_mb=4000, need_ram_mb=4000)
    for i, (text, kind) in enumerate(wanted):
        try:
            made = await generated_effect(text, _gen_seconds(kind), seed + i * 7919)
        except Exception as e:
            logger.warning("Scene sound %r not generated (%s)", text, e)
            made = None
        counts["generated" if made is not None else "failed"] += 1
    return counts


# --- the normalised copy -------------------------------------------------------

def prepare_effect(path: Path, kind: str = "accent") -> Path:
    """A loudness-normalised copy of an effect file, made once per file and
    kind under `data/sfx_cache/` (-18 LUFS accent, -24 bed, TP -1.5, silence
    trimmed, 15 ms edge fades). The original comes back when ffmpeg fails."""
    kind = "bed" if kind == "bed" else "accent"
    source = Path(path)
    try:
        stamp = f"{source}|{source.stat().st_mtime_ns}"
    except OSError:
        return source
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:16]
    destination = CACHE_DIR / f"{digest}_{kind}.wav"
    if destination.exists() and destination.stat().st_size > 1000:
        return destination
    trim = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1:"
            "detection=peak,areverse,"
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1:"
            "detection=peak,areverse")
    fade = f"afade=t=in:st=0:d={EDGE_FADE_S},areverse,afade=t=in:st=0:d={EDGE_FADE_S},areverse"
    loud = f"loudnorm=I={LOUDNESS_I[kind]:.1f}:TP={LOUDNESS_TP:.1f}:LRA=11"
    partial = CACHE_DIR / f"{digest}_{kind}.part.wav"
    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
               "-t", f"{PREPARE_MAX_S[kind]:.0f}", "-i", str(source), "-vn",
               "-af", f"{trim},{fade},{loud}",
               "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(partial)]
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120, creationflags=NO_WINDOW)
        if not partial.exists() or partial.stat().st_size <= 1000:
            raise RuntimeError("no audio written")
        os.replace(partial, destination)
    except Exception as e:
        logger.warning("Could not normalise effect %s (%s); using it as it is", source, e)
        try:
            partial.unlink()
        except OSError:
            pass
        return source
    return destination
