"""Sound design for the presentation pass: a music bed, effects on the beats,
an ambience loop, and a mastered voice.

The programme used to ship with the raw voice track and nothing else, which
for a horror story is the difference between a story and a man reading. This
module adds three mix lanes — A2 music, A3 effects, A4 ambience — as ordinary
timeline items, so the morning edit can mute, move or swap any of them.

**Where the sounds come from, in order of preference:**

1. The user's own library: `data/music/<genre>/`, `data/sfx/<tag>/`,
   `data/ambience/<kind>/`. Drop files in and they are used; an optional
   `manifest.json` next to them adds tags.
2. Synthesised fallbacks, built once with FFmpeg's lavfi sources and cached
   under `data/audio_synth/`. A low drone for horror, wind and rain loops,
   a whoosh, a pop, a boom, a riser, a stinger, a heartbeat. Deliberately
   simple and deliberately quiet — they are there so a night with an empty
   library still produces a video with a soundtrack, not so they can replace
   real music.

The voice is mastered on the timeline's `audio_master`: light denoise,
de-ess, gentle compression, and a -14 LUFS target on the final mix. That
target is what YouTube normalises to, so a video mixed to it plays at the
level the user heard.

Every level here is relative to the voice, not absolute: the music sits
~16 dB under, ducks a further ~12 dB while the speaker talks, and comes
back up in pauses; effects sit ~8 dB under; ambience ~26 dB under.
"""

import hashlib
import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import DATA_DIR, FFMPEG_BIN
from utils.proc import NO_WINDOW
from timeline import clip_ops
from timeline.authoring import clear_generated
from timeline.schema import AudioMaster, SourceFile, Timeline, time_to_frame

from . import workflows
from .models import (AmbienceCue, LoopCue, MusicCue, PresentationSettings, Program,
                     SfxCue, SoundPlan)

logger = logging.getLogger("presentation.sound")

MUSIC_TRACK = "A2"
SFX_TRACK = "A3"
AMBIENCE_TRACK = "A4"
MUSIC_ORIGIN = "music"
SFX_ORIGIN = "sfx"
AMBIENCE_ORIGIN = "ambience"
MASTER_ORIGIN = "presentation"

MUSIC_DIR = Path(DATA_DIR) / "music"
SFX_DIR = Path(DATA_DIR) / "sfx"
AMBIENCE_DIR = Path(DATA_DIR) / "ambience"
SYNTH_DIR = Path(DATA_DIR) / "audio_synth"
# ComfyUI text-to-audio generations, cached by prompt+duration so each sound
# (and each music mood/genre) is made once and then reused like a library file.
AUDIO_GEN_DIR = Path(DATA_DIR) / "audio_gen"

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"}

# The effect vocabulary the planner may ask for. Anything else is ignored
# rather than guessed at.
SFX_TAGS = ("whoosh", "whoosh_soft", "pop", "boom", "riser", "stinger",
            "heartbeat", "thunder", "flash", "click")
SYNTH_KINDS = ("drone", "drone_low", "drone_high", "wind", "rain", "room_tone") + SFX_TAGS

# Two whooshes closer than this read as a stutter.
MIN_SFX_GAP_S = 1.2
SYNTH_LOOP_SECONDS = 48.0
# A one-shot SFX gets this short a fade on each edge — inaudible as a fade,
# enough to kill the click a hard start/stop (or a programme-boundary trim)
# would otherwise leave.
SFX_CLICK_FADE_S = 0.015

# Ambience per genre, paired with the visual atmosphere in genre.py.
_AMBIENCE_FOR_GENRE: Dict[str, Optional[str]] = {
    "horror": "wind",
    "true_crime": "room_tone",
}

# Genres whose synthesised bed is acceptable when no library music exists.
_SYNTH_BED_FOR_GENRE: Dict[str, str] = {
    "horror": "drone",
    "true_crime": "drone_low",
}

# Voice-master presets: (denoise, deess, compress) strengths for the
# `voice_denoise`/`voice_deess`/`voice_compress` 0..1 knobs. The preset NAME is
# also written straight to `AudioMaster.voice_eq_preset` — render/audio.py's
# EQ_PRESETS holds the actual EQ/character curve for each of these same names,
# so this one lookup carries both the strength and the tone.
VOICE_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "studio_mic": (0.4, 0.35, 0.55),
    "broadcast": (0.35, 0.3, 0.7),
    "warm_radio": (0.35, 0.2, 0.5),
    "rap_vocal": (0.3, 0.35, 0.75),
    "horror_intimate": (0.5, 0.25, 0.7),
    "clean": (0.35, 0.3, 0.45),
    "podcast": (0.3, 0.35, 0.65),
    "light": (0.2, 0.15, 0.25),
}

# Prompts and target durations for ComfyUI text-to-audio generation, one entry
# per SFX tag. A tag with no prompt here is simply never generated (it falls
# through to "off" in "auto"/"generated" mode).
_SFX_GEN_PROMPTS: Dict[str, str] = {
    "whoosh": "fast cinematic air whoosh transition, swipe sound effect",
    "whoosh_soft": "soft gentle air whoosh sound effect, subtle swipe",
    "pop": "short clean UI pop click sound effect",
    "boom": "deep cinematic boom impact sound effect, trailer hit",
    "riser": "rising tension sound effect, cinematic riser building to a hit",
    "stinger": "dramatic dissonant orchestral stinger hit, cinematic accent",
    "heartbeat": "tense human heartbeat loop, thumping pulse",
    "thunder": "thunder clap and distant rumble sound effect",
    "flash": "bright magical flash whoosh hit sound effect",
    "click": "short mechanical click sound effect",
}
_SFX_GEN_DURATION: Dict[str, float] = {
    "whoosh": 0.8, "whoosh_soft": 0.7, "pop": 0.3, "boom": 2.5, "riser": 3.5,
    "stinger": 2.0, "heartbeat": 20.0, "thunder": 4.5, "flash": 0.8, "click": 0.2,
}
# Music bed prompts by genre, for the same generator. Genres missing here get
# a generic "<genre> background music" prompt rather than nothing.
_MUSIC_GEN_PROMPTS: Dict[str, str] = {
    "horror": "dark ominous horror drone, unsettling cinematic tension, no melody",
    "true_crime": "moody tense true crime documentary background music, minimal",
    "comedy": "upbeat playful comedic background music, light and bouncy",
    "finance": "corporate motivational background music, confident and modern",
    "tech": "modern electronic background music, clean and minimal",
}
# Kept off a music bed: it is background under a voice, not a song, so vocals
# and obvious artifacts are exactly what the negative prompt should steer away
# from. SFX one-shots have no equivalent — there is nothing a two-second
# whoosh needs to be told to avoid.
_MUSIC_GEN_NEGATIVE = "low quality, noise, hiss, distortion, vocals"
# SFX one-shots DO need one after all: Stable Audio answered "whoosh" and
# "pop" with bursts of broadband static (94% of the energy above 5 kHz), and
# those two clips, reused 137 times, were the hiss that ran through a render.
_SFX_GEN_NEGATIVE = "hiss, static, white noise, pink noise, harsh high frequencies, distortion, crackle"

# A generated SFX whose spectrum looks like noise rather than a sound is
# rejected (`sfx_is_noise`) and the cue plays nothing instead.
SFX_NOISE_HIGH_BAND_HZ = 5000.0
SFX_NOISE_MAX_HIGH_SHARE = 0.6
SFX_NOISE_MAX_FLATNESS = 0.3
SFX_NOISE_MIN_CENTROID_HZ = 4500.0
# Hard ceiling on one-shots per minute of programme, whatever the planners
# asked for: at ~17/min a whoosh on every cut stops being punctuation.
MAX_SFX_PER_MINUTE = 6.0
# B-roll whooshes: at most one per this many seconds, at this share of sfx_volume.
WHOOSH_MIN_GAP_S = 30.0
WHOOSH_GAIN = 0.5
MIN_SFX_CEILING = 8


# --- library -------------------------------------------------------------------

def _audio_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _manifest(folder: Path) -> Dict[str, Dict]:
    path = folder / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Bad audio manifest %s: %s", path, e)
        return {}


def _as_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


_unlicensed_logged: set = set()


def _licensed(files: List[Path]) -> List[Path]:
    """Library files whose manifest names an allowed licence (see
    music_gen.license_ok). Copyright policy: a file with no licence on record
    is never used; it is logged once so the user knows to label it."""
    from .music_gen import license_ok
    kept = []
    for path in files:
        if license_ok(path):
            kept.append(path)
        elif str(path) not in _unlicensed_logged:
            _unlicensed_logged.add(str(path))
            logger.warning("Skipping %s: no allowed licence in its manifest.json "
                           "(add \"license\": \"own\" / \"cc0\" / \"royalty-free\" ...)", path)
    return kept


def _entry(path: Path) -> Dict[str, Any]:
    return _manifest(path.parent).get(path.name) or {}


_NO_BORROWED_MUSIC = {"horror", "true_crime", "mystery"}


def _music_candidates(genre: str, library: Path = MUSIC_DIR) -> List[Path]:
    """Licensed tracks for the genre: `<genre>/`, plus `general/` tracks whose
    manifest lists the genre (a Music-page track made "for" several genres).
    Falls back to all of `general/`, then loose files, when that is empty."""
    candidates: List[Path] = _licensed(_audio_files(library / genre))
    general = _licensed(_audio_files(library / "general")) if genre != "general" else []
    candidates += [p for p in general
                   if genre in _as_list(_entry(p).get("genre") or _entry(p).get("genres"))]
    # Any general track is a fair stand-in for a vlog, not for horror: an
    # empty horror/ folder put "sad violin" under a Raat3Baje story. Those
    # genres get only tracks made for them (or the dark drone fallback).
    if not candidates and genre in _NO_BORROWED_MUSIC:
        return []
    if not candidates:
        candidates = general
    if not candidates:
        manifest = _manifest(library)
        candidates = []
        for path in _audio_files(library):
            entry = manifest.get(path.name) or {}
            genres = _as_list(entry.get("genre") or entry.get("genres"))
            if not genres or genre in genres:
                candidates.append(path)
        candidates = _licensed(candidates)
    return candidates


def _with_mood(paths: List[Path], mood: str) -> List[Path]:
    return [p for p in paths
            if mood in _as_list(_entry(p).get("mood") or _entry(p).get("moods"))]


def find_music(genre: str, rng: random.Random,
               mood: Optional[str] = None,
               library: Path = MUSIC_DIR) -> Optional[Path]:
    """A music file for the genre from the user's library, or None.

    Looks in `<library>/<genre>/` and the `general/` tracks made for the genre,
    then all of `general/`, then loose files in `<library>/` whose manifest
    entry lists the genre (or has no genre at all). A manifest `mood` list
    narrows the choice when the caller has one.
    """
    candidates = _music_candidates(genre, library)
    if mood and candidates:
        moody = _with_mood(candidates, mood)
        if moody:
            candidates = moody
    if not candidates:
        return None
    return rng.choice(candidates)


def has_music_for(genre: str, mood: Optional[str], library: Path = MUSIC_DIR) -> bool:
    """Whether the library has a licensed track for this genre in this act
    mood (any track, when no mood is given)."""
    if not mood:
        return find_music(genre, random.Random(0)) is not None if library == MUSIC_DIR             else bool(_music_candidates(genre, library))
    return bool(_with_mood(_music_candidates(genre, library), mood))


def music_cue_sections(settings: PresentationSettings,
                       music_cues: Optional[Sequence[Tuple[float, str]]],
                       duration: float) -> List[Tuple[float, float, Dict[str, Any], str]]:
    """(start, end, recipe, cue text) for the music the video asks for by name.

    The script's `[music: ...]` cues each start a section that runs to the next
    one; the project's `music_brief` covers the opening before the first cue
    (or the whole video when the script has none). Empty when neither is set,
    and the act-by-act library beds apply instead.
    """
    from .music_gen import parse_music_cue
    raw: List[Tuple[float, str]] = []
    if settings.music_cues and music_cues:
        raw = sorted((max(0.0, float(at)), str(text)) for at, text in music_cues if str(text).strip())
    brief = (settings.music_brief or "").strip()
    if brief and (not raw or raw[0][0] > MIN_SECTION_S):
        raw.insert(0, (0.0, brief))
    if not raw or duration <= 0:
        return []
    raw[0] = (0.0, raw[0][1])
    # Cues closer together than a section merge: the later one wins.
    merged: List[Tuple[float, str]] = []
    for at, text in raw:
        if merged and at - merged[-1][0] < MIN_SECTION_S:
            merged[-1] = (merged[-1][0], text)
        else:
            merged.append((at, text))
    out = []
    for i, (at, text) in enumerate(merged):
        end = merged[i + 1][0] if i + 1 < len(merged) else duration
        if end > at:
            out.append((at, end, parse_music_cue(text), text))
    return out


def find_sfx(tag: str, rng: random.Random, library: Path = SFX_DIR) -> Optional[Path]:
    files = _licensed(_audio_files(library / tag))
    return rng.choice(files) if files else None


def find_ambience(kind: str, rng: random.Random,
                  library: Path = AMBIENCE_DIR) -> Optional[Path]:
    files = _licensed(_audio_files(library / kind))
    return rng.choice(files) if files else None


# --- synthesis -----------------------------------------------------------------
#
# Each recipe is a lavfi graph ending in one audio stream, rendered once to a
# 48 kHz WAV and cached by recipe hash.

def _recipe(kind: str) -> Optional[Tuple[str, float]]:
    """(lavfi filter graph, seconds) for a synthesised sound, or None."""
    loop = SYNTH_LOOP_SECONDS
    recipes: Dict[str, Tuple[str, float]] = {
        # A horror drone: two detuned low sines beating against each other,
        # a sub under them, and filtered brown noise for air. Slow tremolo so
        # it breathes rather than hums.
        "drone": (
            f"sine=f=55:d={loop},volume=0.35[s1];"
            f"sine=f=55.7:d={loop},volume=0.3[s2];"
            f"sine=f=27.5:d={loop},volume=0.25[s3];"
            f"anoisesrc=color=brown:d={loop}:seed=7,lowpass=f=180,volume=0.5[n];"
            f"[s1][s2][s3][n]amix=inputs=4:normalize=0,"
            f"volume=0.75+0.25*sin(t*0.45):eval=frame,lowpass=f=400,"
            f"afade=t=in:st=0:d=2,afade=t=out:st={loop - 2}:d=2", loop),
        "drone_high": (
            f"sine=f=82.4:d={loop},volume=0.3[s1];"
            f"sine=f=83.6:d={loop},volume=0.28[s2];"
            f"sine=f=41.2:d={loop},volume=0.25[s3];"
            f"sine=f=164.8:d={loop},volume=0.08[s4];"
            f"anoisesrc=color=brown:d={loop}:seed=13,lowpass=f=260,volume=0.5[n];"
            f"[s1][s2][s3][s4][n]amix=inputs=5:normalize=0,"
            f"volume=0.7+0.3*sin(t*0.9):eval=frame,lowpass=f=600,"
            f"afade=t=in:st=0:d=2,afade=t=out:st={loop - 2}:d=2", loop),
        "drone_low": (
            f"sine=f=41.2:d={loop},volume=0.35[s1];"
            f"sine=f=41.6:d={loop},volume=0.3[s2];"
            f"anoisesrc=color=brown:d={loop}:seed=11,lowpass=f=140,volume=0.45[n];"
            f"[s1][s2][n]amix=inputs=3:normalize=0,volume=0.75+0.25*sin(t*0.33):eval=frame,"
            f"lowpass=f=300,afade=t=in:st=0:d=2,afade=t=out:st={loop - 2}:d=2", loop),
        # Wind: brown noise through a low-pass, swelling on two slow cycles.
        "wind": (
            f"anoisesrc=color=brown:d={loop}:seed=3,lowpass=f=600,"
            f"volume=0.45+0.3*sin(t*0.7)+0.25*sin(t*0.27):eval=frame,"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={loop - 1.5}:d=1.5", loop),
        # Rain: pink noise with the lows taken out and a fast shimmer.
        "rain": (
            f"anoisesrc=color=pink:d={loop}:seed=5,highpass=f=900,lowpass=f=9000,"
            f"tremolo=f=13:d=0.25,volume=0.6,"
            f"afade=t=in:st=0:d=1.5,afade=t=out:st={loop - 1.5}:d=1.5", loop),
        # Room tone: nearly nothing — a faint hiss that stops silence sounding
        # like a dropout.
        "room_tone": (
            f"anoisesrc=color=pink:d={loop}:seed=9,lowpass=f=3000,volume=0.25,"
            f"afade=t=in:st=0:d=1,afade=t=out:st={loop - 1}:d=1", loop),
        # A whoosh: white noise through a band-pass inside a short envelope.
        "whoosh": (
            "anoisesrc=color=white:d=0.7:seed=21,"
            "bandpass=f=1200:width_type=o:w=2,"
            "afade=t=in:st=0:d=0.25:curve=qsin,afade=t=out:st=0.3:d=0.4:curve=qsin,"
            "volume=0.9", 0.7),
        "whoosh_soft": (
            "anoisesrc=color=pink:d=0.6:seed=22,"
            "bandpass=f=800:width_type=o:w=2,"
            "afade=t=in:st=0:d=0.2:curve=qsin,afade=t=out:st=0.25:d=0.35:curve=qsin,"
            "volume=0.6", 0.6),
        # A pop for text: a short decaying sine with a click of noise.
        "pop": (
            "aevalsrc=0.8*exp(-t*38)*sin(2*PI*820*t)+0.25*exp(-t*90)*random(0):"
            "d=0.18:s=48000", 0.18),
        "click": (
            "aevalsrc=0.5*exp(-t*120)*sin(2*PI*2400*t):d=0.08:s=48000", 0.08),
        # A boom: sub sine with a fast pitch drop and slow decay.
        "boom": (
            "aevalsrc=exp(-t*2.2)*sin(2*PI*(38+50*exp(-t*9))*t):d=2.2:s=48000,"
            "lowpass=f=160,volume=1.0", 2.2),
        # A flash hit: a short boom with a noise burst on top.
        "flash": (
            "aevalsrc=exp(-t*9)*sin(2*PI*(60+120*exp(-t*20))*t)+0.35*exp(-t*40)*random(0):"
            "d=0.6:s=48000,volume=1.0", 0.6),
        # A riser: noise with the level rising to the end, where the reveal lands.
        "riser": (
            "anoisesrc=color=white:d=3.2:seed=31,"
            "highpass=f=200,lowpass=f=6000,"
            "volume=0.05+0.9*pow(t/3.2\\,2):eval=frame,"
            "afade=t=out:st=3.0:d=0.2", 3.2),
        # A stinger: a dissonant cluster that hits and decays over 1.6 s.
        "stinger": (
            "aevalsrc=exp(-t*2.5)*(sin(2*PI*196*t)+sin(2*PI*207*t)+0.6*sin(2*PI*392*t)"
            "+0.4*sin(2*PI*622*t))*0.35+0.5*exp(-t*30)*random(0):d=1.6:s=48000", 1.6),
        # Thunder: a boom followed by a long rumble of filtered brown noise.
        "thunder": (
            "aevalsrc=exp(-t*1.4)*sin(2*PI*(30+40*exp(-t*6))*t):d=4:s=48000[b];"
            "anoisesrc=color=brown:d=4:seed=41,lowpass=f=220,"
            "volume=0.9*exp(-t*0.9):eval=frame[r];"
            "[b][r]amix=inputs=2:normalize=0,volume=1.0", 4.0),
        # A heartbeat: lub-dub at 62 bpm, looped.
        "heartbeat": (
            "aevalsrc=0.9*exp(-mod(t\\,0.968)*22)*sin(2*PI*52*mod(t\\,0.968))"
            "+0.6*exp(-max(mod(t\\,0.968)-0.19\\,0)*24)*sin(2*PI*46*max(mod(t\\,0.968)-0.19\\,0))"
            "*gt(mod(t\\,0.968)\\,0.19):d=19.36:s=48000,lowpass=f=150", 19.36),
    }
    return recipes.get(kind)


def synth_path(kind: str) -> Optional[Path]:
    """Render (once) a synthesised sound and return its WAV path."""
    recipe = _recipe(kind)
    if recipe is None:
        return None
    graph, seconds = recipe
    digest = hashlib.sha1(f"{kind}:{graph}".encode("utf-8")).hexdigest()[:10]
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    path = SYNTH_DIR / f"{kind}_{digest}.wav"
    if path.exists() and path.stat().st_size > 1000:
        return path
    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", graph,
               "-t", f"{seconds:.3f}", "-ar", "48000", "-ac", "2",
               "-c:a", "pcm_s16le", str(path)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120,
                       creationflags=NO_WINDOW)
    except subprocess.CalledProcessError as e:
        logger.warning("Could not synthesise %r: %s", kind,
                       (e.stderr or b"").decode("utf-8", "replace")[-300:])
        return None
    except Exception as e:
        logger.warning("Could not synthesise %r: %s", kind, e)
        return None
    return path


def synth_duration(kind: str) -> float:
    recipe = _recipe(kind)
    return recipe[1] if recipe else 0.0


# --- ComfyUI generation ----------------------------------------------------------
#
# "generated" sourcing (see _resolve_sfx / _bed_for below): a text-to-audio
# ComfyUI workflow (role "audio" in presentation.workflows — Stable Audio
# Open, ACE-Step, ...) makes the sound once, and it is cached by prompt+
# duration so every later render just reads the file. Nothing here ever
# raises past `warm_generated_cache`: a missing workflow, an offline ComfyUI,
# or a failed job all just mean fewer sounds get generated, same as an empty
# library folder does today.

def _audio_gen_key(prompt: str, duration_s: float) -> str:
    return hashlib.sha1(f"{prompt}|{duration_s:.2f}".encode("utf-8")).hexdigest()[:16]


MAX_GAP_TRACKS = 3   # act moods filled per video, at most


async def _warm_music(settings: PresentationSettings, genre: str, seed: int,
                      acts: Optional[Sequence[str]],
                      music_cues: Optional[Sequence[Tuple[float, str]]],
                      duration: float, counts: Dict[str, int]) -> bool:
    """Generate the music this video needs; True when the genre still has no
    bed at all (no music model, or it made nothing), so the caller can fall
    back to a Stable Audio Open loop."""
    from . import music_gen
    source = (settings.music_source or "auto").lower()
    if not settings.music or source == "off" or settings.music_tracks:
        return False
    genre_empty = False
    jobs: List[Dict[str, Any]] = []
    cues = music_cue_sections(settings, music_cues, duration)
    if cues:
        engine_default = settings.music_engine if settings.music_engine in music_gen.ENGINES \
            else music_gen.DEFAULT_ENGINE
        seen = set()
        for start, end, recipe, text in cues:
            engine = recipe.get("engine") or engine_default
            if recipe.get("off") or not music_gen.cue_has_content(recipe):
                continue
            key = music_gen.cue_key(recipe, engine)
            if key in seen or music_gen.find_cue_track(recipe, engine) is not None:
                continue
            seen.add(key)
            jobs.append({"cue": (recipe, engine, text), "seconds": end - start})
    elif source in ("auto", "generated"):
        genre_empty = not has_music_for(genre, None)
        moods: List[str] = []
        for act in (acts or [""]):
            mood = ACT_MUSIC.get((act or "").lower(), ("", "", 1.0))[0]
            if mood not in moods:
                moods.append(mood)
        for mood in moods:
            if not has_music_for(genre, mood or None):
                jobs.append({"style": music_gen.style_for(genre, mood or None), "mood": mood})
        jobs = jobs[:MAX_GAP_TRACKS]
    if not jobs:
        return genre_empty
    roles = {music_gen.ENGINES[j["cue"][1]]["role"] if "cue" in j else "music" for j in jobs}
    if any(workflows.resolve(role) is None for role in roles):
        logger.info("Music generation skipped: no workflow for %s", sorted(roles))
        return genre_empty
    made_any = False
    from runtime import gpu_handover
    await gpu_handover.prepare_for_phase(
        "music", need_vram_mb=music_gen.MUSIC_PHASE_MIN_VRAM_MB,
        need_ram_mb=music_gen.MUSIC_PHASE_MIN_RAM_MB)
    rng = random.Random(seed)
    for job in jobs:
        try:
            if "cue" in job:
                recipe, engine, text = job["cue"]
                max_s = music_gen.ENGINES[engine]["max_seconds"]
                seconds = max(30.0, min(max_s, job["seconds"] + 4.0))
                made = await music_gen.ensure_cue_track(recipe, engine, seconds, genre=genre,
                                                        seed=rng.randint(1, 2**31 - 1),
                                                        cue_text=text)
            else:
                made = await music_gen.generate_track(job["style"], music_gen.DEFAULT_SECONDS,
                                                      seed=rng.randint(1, 2**31 - 1), genre=genre)
            if made is not None:
                counts["music"] += 1
                made_any = True
        except Exception as e:
            logger.warning("Music generation for %s failed: %s", job, e)
    return genre_empty and not made_any


def _cached_generated(prompt: str, duration_s: float) -> Optional[Path]:
    """Read-only lookup: the cached WAV for this prompt+duration, if one exists.

    Safe to call from the synchronous planning/placement code — it never
    triggers a generation, only reads what `warm_generated_cache` already made.
    """
    path = AUDIO_GEN_DIR / f"{_audio_gen_key(prompt, duration_s)}.wav"
    return path if path.exists() and path.stat().st_size > 1000 else None


_noise_verdicts: Dict[str, bool] = {}


def sfx_is_noise(path: Path) -> bool:
    """True when a clip is mostly broadband high-frequency noise -- hiss, not a
    whoosh or a pop. Decided once per file (by path+mtime) from its spectrum:
    too much energy above SFX_NOISE_HIGH_BAND_HZ, or a flat (noise-like)
    spectrum centred high. An unreadable file counts as noise."""
    try:
        stamp = f"{path}|{path.stat().st_mtime_ns}"
    except OSError:
        return True
    if stamp in _noise_verdicts:
        return _noise_verdicts[stamp]
    verdict = True
    try:
        import numpy as np
        rate = 44100
        raw = subprocess.run(
            [FFMPEG_BIN, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate),
             "-t", "4", "-f", "f32le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            creationflags=NO_WINDOW).stdout
        samples = np.frombuffer(raw, dtype=np.float32)
        if samples.size >= 1024:
            spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) + 1e-9
            freqs = np.fft.rfftfreq(samples.size, 1.0 / rate)
            total = float(spectrum.sum())
            high_share = float(spectrum[freqs > SFX_NOISE_HIGH_BAND_HZ].sum()) / total
            centroid = float((spectrum * freqs).sum()) / total
            flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
            verdict = (high_share > SFX_NOISE_MAX_HIGH_SHARE
                       or (flatness > SFX_NOISE_MAX_FLATNESS and centroid > SFX_NOISE_MIN_CENTROID_HZ))
            if verdict:
                logger.warning("Generated SFX %s rejected as noise (%.0f%% above %.0f Hz, "
                               "centroid %.0f Hz, flatness %.2f)", path.name, high_share * 100,
                               SFX_NOISE_HIGH_BAND_HZ, centroid, flatness)
    except Exception as e:
        logger.warning("Could not analyse generated SFX %s (%s); not using it", path, e)
    _noise_verdicts[stamp] = verdict
    return verdict


def _generated_sfx_path(tag: str) -> Optional[Path]:
    prompt = _SFX_GEN_PROMPTS.get(tag)
    if not prompt:
        return None
    path = _cached_generated(prompt, _SFX_GEN_DURATION.get(tag, 1.0))
    if path is None or sfx_is_noise(path):
        return None
    return path


def _generated_music_path(genre: str) -> Optional[Path]:
    prompt = _MUSIC_GEN_PROMPTS.get(genre, f"{genre} background music, cinematic instrumental")
    return _cached_generated(prompt, SYNTH_LOOP_SECONDS)


# The edge fade applied to every generated clip before it is cached — short
# enough to be inaudible as a fade, long enough to kill the click a hard
# ComfyUI-decoded start/stop (or the silence trim right before it) would
# otherwise leave. Same duration as the one the placed-SFX path already uses.
_GEN_FADE_S = SFX_CLICK_FADE_S
# A generated clip is background material (a bed, a one-shot effect), not a
# mastered voice track, so it is normalised to a gentler target than the
# programme's own -14 LUFS — loud enough to sit correctly under the mix's own
# gain stages, quiet enough that a single hot generation cannot clip them.
_GEN_LOUDNESS_I = -16.0
_GEN_LOUDNESS_TP = -1.5


def _master_generated(source: Path, destination: Path, lowpass_hz: Optional[int] = None) -> bool:
    """Trim leading/trailing silence, fade both edges, and loudness-normalise
    a ComfyUI generation, once, before it enters the cache.

    Best-effort like every other step in this pipeline: `destination` is left
    unwritten on any ffmpeg failure, and the caller falls back to caching the
    raw generation exactly as if this pass had never run.
    """
    trim = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1:"
            "detection=peak,areverse,"
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1:"
            "detection=peak,areverse")
    # A fade-in at the start, mirrored (via areverse) onto the end, so both
    # edges get the same short fade without needing the clip's own duration.
    fade = f"afade=t=in:st=0:d={_GEN_FADE_S},areverse,afade=t=in:st=0:d={_GEN_FADE_S},areverse"
    loud = f"loudnorm=I={_GEN_LOUDNESS_I:.1f}:TP={_GEN_LOUDNESS_TP:.1f}:LRA=11"
    # SFX only: roll off the top octave, where a one-shot's residual fizz lives.
    shape = f"lowpass=f={int(lowpass_hz)}," if lowpass_hz else ""
    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(source), "-af", f"{shape}{trim},{fade},{loud}",
               "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(destination)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60, creationflags=NO_WINDOW)
    except Exception as e:
        logger.warning("Could not master generated audio %s: %s", source, e)
        return False
    return destination.exists() and destination.stat().st_size > 1000


async def _generate_audio(prompt: str, duration_s: float, seed: int,
                          negative: str = "", lowpass_hz: Optional[int] = None) -> Optional[Path]:
    """Submit one text-to-audio job and cache the result by prompt+duration."""
    cached = _cached_generated(prompt, duration_s)
    if cached is not None:
        return cached
    workflow = workflows.resolve("audio")
    if workflow is None:
        return None
    try:
        graph = workflow.build(positive=prompt, negative=negative, seed=seed,
                               length=max(1, int(round(duration_s))))
        from comfyui_bridge import queue_manager
        outputs = await queue_manager.submit_and_wait(graph, timeout=120)
    except Exception as e:
        logger.warning("Audio generation failed for %r: %s", prompt[:60], e)
        return None
    if not outputs:
        return None
    chosen = next((o for o in outputs if Path(o).suffix.lower() in AUDIO_EXTS), outputs[0])
    AUDIO_GEN_DIR.mkdir(parents=True, exist_ok=True)
    destination = AUDIO_GEN_DIR / f"{_audio_gen_key(prompt, duration_s)}.wav"
    if _master_generated(Path(chosen), destination, lowpass_hz=lowpass_hz):
        return destination
    try:
        import shutil
        shutil.copy2(chosen, destination)
    except Exception as e:
        logger.warning("Could not cache generated audio %s: %s", chosen, e)
        return Path(chosen)
    return destination


async def warm_generated_cache(settings: PresentationSettings, genre: str, seed: int,
                               comfyui_online: bool = True,
                               acts: Optional[Sequence[str]] = None,
                               music_cues: Optional[Sequence[Tuple[float, str]]] = None,
                               duration: float = 0.0) -> Dict[str, int]:
    """Best-effort pre-generation of the SFX/music the library does not cover.

    Run BEFORE the (synchronous) planning/placement pass so `_resolve_sfx` and
    `_bed_for` find an already-cached file rather than needing to be async
    themselves. Only fills gaps: a tag or genre the library already has is
    never regenerated, and this is a no-op entirely when ComfyUI is offline or
    no "audio" workflow is configured.
    """
    counts = {"sfx": 0, "music": 0}
    if not comfyui_online:
        return counts
    from runtime import gpu_handover
    rng = random.Random(seed)

    # Music first, with the card handed over to the music model once for the
    # whole batch. Named music (the script's [music: ...] cues, the project's
    # music brief) gets exactly its track, made once and reused after; with no
    # names, every act mood the video needs that the library cannot cover gets
    # a track in the genre's style for that mood. Everything lands in the
    # library, so later videos reuse it.
    music_wanted = await _warm_music(settings, genre, seed, acts, music_cues, duration, counts)

    if workflows.resolve("audio") is None:
        return counts
    handed_over = False
    if settings.sfx_source in ("auto", "generated") and (settings.sfx or settings.ambience):
        for tag in SFX_TAGS:
            prompt = _SFX_GEN_PROMPTS.get(tag)
            if not prompt or find_sfx(tag, rng) is not None:
                continue
            if not handed_over:
                # Stable Audio Open is a different model: swap it in cleanly.
                await gpu_handover.prepare_for_phase("sfx", need_vram_mb=4000, need_ram_mb=4000)
                handed_over = True
            duration = _SFX_GEN_DURATION.get(tag, 1.0)
            made = None
            # A cached clip that is really static gets thrown away and made
            # again, steered off noise, with a couple of fresh seeds.
            for attempt in range(3):
                cached = _cached_generated(prompt, duration)
                if cached is not None and sfx_is_noise(cached):
                    try:
                        cached.unlink()
                    except OSError:
                        break
                made = await _generate_audio(prompt, duration, seed + attempt * 7919,
                                             negative=_SFX_GEN_NEGATIVE, lowpass_hz=9000)
                if made is None or not sfx_is_noise(made):
                    break
                made = None
            if made is not None:
                counts["sfx"] += 1
    if music_wanted:
        # No ACE-Step: fall back to a Stable Audio Open loop.
        if not handed_over:
            await gpu_handover.prepare_for_phase("sfx", need_vram_mb=4000, need_ram_mb=4000)
        prompt = _MUSIC_GEN_PROMPTS.get(genre, f"{genre} background music, cinematic instrumental")
        made = await _generate_audio(prompt, SYNTH_LOOP_SECONDS, seed,
                                     negative=_MUSIC_GEN_NEGATIVE)
        if made is not None:
            counts["music"] += 1
    return counts


# --- planning ------------------------------------------------------------------

def plan_sound(timeline: Timeline, program: Program, settings: PresentationSettings,
               genre: str, seed: int,
               broll_windows: Optional[Sequence[Tuple[float, float]]] = None,
               popup_times: Optional[Sequence[float]] = None,
               extra_sfx: Optional[Sequence[Tuple[float, str]]] = None,
               loops: Optional[Sequence[Tuple[float, float, str, float]]] = None,
               sections: Optional[Sequence[Tuple[float, float, str]]] = None,
               music_cues: Optional[Sequence[Tuple[float, str]]] = None) -> SoundPlan:
    """Decide the music, effects and ambience for the programme.

    `extra_sfx` are (time, tag) pairs from other planners (the mood recipes
    add stingers and thunder); they are merged and spaced with the rest.
    `loops` are (start, end, tag, gain) stretches of a looped effect (a
    heartbeat through a climax). `sections` are (start, end, act) from the
    story structure: the bed changes cue at act boundaries with a crossfade.
    """
    rng = random.Random(seed)
    plan = SoundPlan()
    duration = max(0.0, program.duration_s)
    if duration <= 0.0:
        return plan

    if settings.music:
        cue_sections = music_cue_sections(settings, music_cues, duration)
        if settings.music_tracks:
            # The creator's own picks, at their own levels, replace the
            # automatic choice; picks that cannot be used leave silence (and
            # a note) rather than a track nobody chose.
            chosen = chosen_tracks(settings)
            plan.music = _plan_chosen_music(settings, duration, chosen) if chosen else []
        elif cue_sections:
            plan.music = _plan_cue_music(genre, settings, rng, duration, cue_sections)
        else:
            plan.music = _plan_music(genre, settings, rng, duration,
                                     sections if settings.music_by_act else None)
        if not plan.music:
            plan.notes.append("music_missing")

    if settings.ambience:
        kind = (settings.ambience_kind if settings.ambience_kind not in ("", "auto")
                else _AMBIENCE_FOR_GENRE.get(genre))
        if kind:
            path = find_ambience(kind, rng) or synth_path(kind)
            if path is not None:
                plan.ambience.append(AmbienceCue(
                    kind=kind, path=str(path), gain=settings.ambience_volume,
                    synthesised=path.parent == SYNTH_DIR))

    if settings.sfx:
        events: List[Tuple[float, str, float]] = []
        # A whoosh marks a change of scene, not every cutaway: one on each of
        # 63 B-roll cuts (plus a soft one on each return) was the "periodic
        # hiss" through a 14-minute Raat3Baje render -- the high-frequency
        # bursts in the mix lined up with them. At most one per
        # WHOOSH_MIN_GAP_S, quieter, and none on the way back.
        last_whoosh = -WHOOSH_MIN_GAP_S
        for start, _end in sorted(broll_windows or []):
            if start - last_whoosh >= WHOOSH_MIN_GAP_S:
                events.append((start, "whoosh", WHOOSH_GAIN))
                last_whoosh = start
        for at in (popup_times or []):
            events.append((at, "pop", 0.8))
        for at, tag in (extra_sfx or []):
            if tag in SFX_TAGS:
                events.append((at, tag, 1.0))
        plan.sfx = _space_sfx(events, duration, settings.sfx_volume)
        for start, end, tag, gain in (loops or []):
            if tag in SFX_TAGS and end - start >= 2.0:
                plan.loops.append(LoopCue(tag=tag, start_s=max(0.0, start),
                                          end_s=min(duration, end),
                                          gain=gain * settings.sfx_volume))

    return plan


# The music mood each act asks for: a library manifest tag, and for the
# synthesised beds a variant + gain so the drone rises toward the climax.
ACT_MUSIC: Dict[str, Tuple[str, str, float]] = {
    # act: (library mood tag, synth variant suffix, gain multiplier)
    "hook": ("intense", "high", 1.15),
    "setup": ("calm", "", 0.85),
    "context": ("calm", "", 0.85),
    "build": ("tense", "", 1.0),
    "point": ("tense", "", 1.0),
    "reveal": ("intense", "high", 1.1),
    "climax": ("intense", "high", 1.25),
    "aftermath": ("soft", "low", 0.8),
    "takeaway": ("soft", "", 0.9),
    "cta": ("upbeat", "", 0.9),
}
# Sections shorter than this fold into the previous one: a cue that swaps
# every fifteen seconds reads as a broken playlist.
MIN_SECTION_S = 24.0
MUSIC_CROSSFADE_S = 2.0


def music_sections(sections: Optional[Sequence[Tuple[float, float, str]]],
                   duration: float) -> List[Tuple[float, float, str]]:
    """Merge consecutive same-act topics and swallow short ones."""
    if not sections:
        return [(0.0, duration, "")]
    merged: List[List] = []
    for start, end, act in sorted(sections, key=lambda s: s[0]):
        act = (act or "").lower()
        if merged and (merged[-1][2] == act or end - start < MIN_SECTION_S):
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end, act])
    if merged and merged[0][0] > 0.0:
        merged[0][0] = 0.0
    # A short opening section (the hook) folds forward into the next.
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_SECTION_S:
        merged[1][0] = merged[0][0]
        del merged[0]
    if merged:
        merged[-1][1] = max(merged[-1][1], duration)
    return [(float(s), float(e), str(a)) for s, e, a in merged]


def _bed_for(genre: str, act: str, settings: PresentationSettings,
             rng: random.Random, previous: Optional[str]) -> Tuple[Optional[Path], bool, str]:
    """(file, synthesised, source label) for one section, avoiding the
    previous section's file.

    Governed by `music_source`: "auto" (default) is the user's library, then a
    cached ComfyUI generation, then the genre's own synthesised bed (a horror
    drone, true-crime room drone) — that last fallback is kept for "auto" on
    purpose, as explicit genre behaviour, not the one-shot SFX noise that
    caused the periodic hiss `sfx_source` fixes below. "library"/"generated"/
    "synth"/"off" each use exactly the one named source.
    """
    source = (settings.music_source or "auto").lower()
    if source == "off":
        return None, False, "off"
    mood, variant, _ = ACT_MUSIC.get(act, ("", "", 1.0))
    path: Optional[Path] = None
    if source in ("auto", "library"):
        path = find_music(genre, rng, mood=mood or None)
        if path is not None and previous and str(path) == previous:
            alternative = find_music(genre, random.Random(rng.random()), mood=mood or None)
            if alternative is not None and str(alternative) != previous:
                path = alternative
    if path is not None:
        return path, False, "library"
    if source == "library":
        return None, False, "off"
    if source in ("auto", "generated"):
        generated = _generated_music_path(genre)
        if generated is not None:
            return generated, False, "generated"
    if source == "generated":
        return None, False, "off"
    kind = _SYNTH_BED_FOR_GENRE.get(genre)
    if kind and settings.music_synth_fallback:
        base = kind.replace("_low", "")
        candidate = f"{base}_{variant}" if variant else kind
        synth = synth_path(candidate) or synth_path(kind)
        return synth, synth is not None, ("synth" if synth is not None else "off")
    return None, False, "off"


def _plan_music(genre: str, settings: PresentationSettings, rng: random.Random,
                duration: float,
                sections: Optional[Sequence[Tuple[float, float, str]]] = None) -> List[MusicCue]:
    """One cue per act section, overlapping by the crossfade at each boundary."""
    cues: List[MusicCue] = []
    previous: Optional[str] = None
    parts = music_sections(sections, duration)
    for index, (start, end, act) in enumerate(parts):
        path, synthesised, source_label = _bed_for(genre, act, settings, rng, previous)
        if path is None:
            continue
        _, _, gain_mult = ACT_MUSIC.get(act, ("", "", 1.0))
        first, last = index == 0, index == len(parts) - 1
        cue_start = 0.0 if first else max(0.0, start - MUSIC_CROSSFADE_S / 2)
        cue_end = duration if last else min(duration, end + MUSIC_CROSSFADE_S / 2)
        cues.append(MusicCue(
            path=str(path), start_s=cue_start, end_s=cue_end,
            gain=settings.music_volume * gain_mult, duck=settings.music_duck,
            fade_in_s=settings.fade_in_s if first else MUSIC_CROSSFADE_S,
            fade_out_s=min(settings.fade_out_s, duration / 3.0) if last else MUSIC_CROSSFADE_S,
            synthesised=synthesised, act=act, source=source_label))
        previous = str(path)
    return cues


def _plan_cue_music(genre: str, settings: PresentationSettings, rng: random.Random,
                    duration: float,
                    cue_sections: Sequence[Tuple[float, float, Dict[str, Any], str]]) -> List[MusicCue]:
    """One cue per named-music section: its own generated track, else the
    library's best match for the cue's mood; "off" sections stay silent."""
    from . import music_gen
    engine_default = settings.music_engine if settings.music_engine in music_gen.ENGINES \
        else music_gen.DEFAULT_ENGINE
    cues: List[MusicCue] = []
    for index, (start, end, recipe, _text) in enumerate(cue_sections):
        if recipe.get("off"):
            continue
        engine = recipe.get("engine") or engine_default
        path = music_gen.find_cue_track(recipe, engine)
        source_label = "cue"
        if path is None:
            act_moods = [music_gen.MOODS[m] for m in recipe.get("moods") or [] if m in music_gen.MOODS]
            path = find_music(genre, rng, mood=act_moods[0] if act_moods else None)
            source_label = "library"
        if path is None:
            continue
        first, last = index == 0, index == len(cue_sections) - 1
        cue_start = 0.0 if first else max(0.0, start - MUSIC_CROSSFADE_S / 2)
        cue_end = duration if last else min(duration, end + MUSIC_CROSSFADE_S / 2)
        cues.append(MusicCue(
            path=str(path), start_s=cue_start, end_s=cue_end,
            gain=settings.music_volume, duck=settings.music_duck,
            fade_in_s=settings.fade_in_s if first else MUSIC_CROSSFADE_S,
            fade_out_s=min(settings.fade_out_s, duration / 3.0) if last else MUSIC_CROSSFADE_S,
            synthesised=False, act="cue", source=source_label))
    return cues


# --- tracks the creator picked ---------------------------------------------------

# A picked track's level, in dB of linear gain on the file (library tracks are
# mastered to about -16 LUFS, near the voice, so -20 dB sits well under it).
# Outside this range is clamped: below is silence, above risks clipping.
CHOSEN_MIN_DB, CHOSEN_MAX_DB = -60.0, 6.0
CHOSEN_DEFAULT_DB = -20.0


def chosen_tracks(settings: PresentationSettings,
                  library: Optional[Path] = None) -> List[Tuple[Path, float, bool]]:
    """(file, linear gain, duck) for each of `settings.music_tracks` that is
    in the library and has a cleared licence. Anything else is skipped and
    logged: a pick is never swapped for a guess."""
    from . import music_gen
    library = library or MUSIC_DIR
    root = library.resolve()
    out: List[Tuple[Path, float, bool]] = []
    for raw in settings.music_tracks or []:
        if not isinstance(raw, dict):
            continue
        relative = str(raw.get("file") or "").strip().replace("\\", "/")
        if not relative:
            continue
        path = (library / relative).resolve()
        if root not in path.parents or not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            logger.warning("Chosen music %r is not in the library; skipped", relative)
            continue
        if not music_gen.license_ok(path):
            logger.warning("Chosen music %r has no cleared licence; skipped", relative)
            continue
        try:
            db = float(raw.get("volume_db", CHOSEN_DEFAULT_DB))
        except (TypeError, ValueError):
            db = CHOSEN_DEFAULT_DB
        db = max(CHOSEN_MIN_DB, min(CHOSEN_MAX_DB, db))
        out.append((path, 10.0 ** (db / 20.0), bool(raw.get("duck"))))
    return out


def _plan_chosen_music(settings: PresentationSettings, duration: float,
                       tracks: Sequence[Tuple[Path, float, bool]]) -> List[MusicCue]:
    """The picked tracks at exactly their own levels -- no act multipliers.
    "together": every track under the whole video, looped. "sequence": each
    track plays through once, crossfading into the next, the list looping
    until the video ends."""
    end_fade = min(settings.fade_out_s, duration / 3.0)

    def cue(path: Path, gain: float, duck: bool, start: float, end: float,
            fade_in: float, fade_out: float) -> MusicCue:
        return MusicCue(path=str(path), start_s=start, end_s=end, gain=gain,
                        duck=settings.music_duck if duck else 0.0,
                        fade_in_s=fade_in, fade_out_s=fade_out,
                        synthesised=False, act="", source="chosen")

    if (settings.music_tracks_mode or "together").lower() != "sequence":
        return [cue(path, gain, duck, 0.0, duration, settings.fade_in_s, end_fade)
                for path, gain, duck in tracks]

    playable = [(path, gain, duck, file_duration(str(path))) for path, gain, duck in tracks]
    playable = [t for t in playable if t[3] > MUSIC_CROSSFADE_S * 2]
    cues: List[MusicCue] = []
    at, index = 0.0, 0
    while playable and at < duration - 0.5:
        path, gain, duck, length = playable[index % len(playable)]
        index += 1
        end = min(duration, at + length)
        last = end >= duration - 0.5
        cues.append(cue(path, gain, duck, at, duration if last else end,
                        settings.fade_in_s if not cues else MUSIC_CROSSFADE_S,
                        end_fade if last else MUSIC_CROSSFADE_S))
        if last:
            break
        at = end - MUSIC_CROSSFADE_S
    return cues


_SFX_PRIORITY = {"stinger": 5, "thunder": 5, "boom": 4, "flash": 4, "riser": 3,
                 "heartbeat": 3, "whoosh": 2, "pop": 1, "click": 1, "whoosh_soft": 1}


def _space_sfx(events: List[Tuple[float, str, float]], duration: float,
               master_gain: float) -> List[SfxCue]:
    """Keep effects apart: when two land within the gap, the weightier tag
    (a stinger over a whoosh) wins and the other is dropped."""
    ordered = sorted(events, key=lambda e: (e[0], -_SFX_PRIORITY.get(e[1], 0)))
    kept: List[SfxCue] = []
    for at, tag, gain in ordered:
        if at < 0.0 or at > duration:
            continue
        clash = next((k for k in kept if abs(k.at_s - at) < MIN_SFX_GAP_S), None)
        if clash is not None:
            if _SFX_PRIORITY.get(tag, 0) > _SFX_PRIORITY.get(clash.tag, 0):
                kept.remove(clash)
            else:
                continue
        kept.append(SfxCue(tag=tag, at_s=round(at, 3), gain=gain * master_gain))
    # A floor keeps a Short or a teaser from being cut to a single hit.
    ceiling = max(MIN_SFX_CEILING, int(MAX_SFX_PER_MINUTE * max(duration, 1.0) / 60.0))
    if len(kept) > ceiling:
        # The weightiest survive; ties go to whichever sits farthest from its
        # neighbours, so what is left stays spread through the programme.
        kept.sort(key=lambda c: c.at_s)

        def isolation(i: int) -> float:
            before = kept[i].at_s - kept[i - 1].at_s if i > 0 else duration
            after = kept[i + 1].at_s - kept[i].at_s if i + 1 < len(kept) else duration
            return min(before, after)

        ranked = sorted(range(len(kept)),
                        key=lambda i: (-_SFX_PRIORITY.get(kept[i].tag, 0), -isolation(i)))
        kept = [kept[i] for i in sorted(ranked[:ceiling])]
    kept.sort(key=lambda c: c.at_s)
    return kept


# --- placement -----------------------------------------------------------------

def _resolve_sfx(tag: str, rng: random.Random, source: str) -> Tuple[Optional[Path], str]:
    """(file, source label) for one sound effect, per `sfx_source`.

    "library" / "generated" / "synth" each use exactly the one named source.
    "off" plays nothing. "auto" (the default) is the user's own data/sfx/<tag>/
    first, then a cached ComfyUI generation, then nothing at all — it NEVER
    reaches for the synthesised FFmpeg noise fallback, which is what produced
    the periodic hiss (18 white/pink-noise whooshes in the diagnosed render):
    that fallback now only runs when a caller asks for "synth" by name.
    """
    source = (source or "auto").lower()
    if source == "off":
        return None, "off"
    if source == "library":
        path = find_sfx(tag, rng)
        return path, ("library" if path is not None else "off")
    if source == "generated":
        path = _generated_sfx_path(tag)
        return path, ("generated" if path is not None else "off")
    if source == "synth":
        path = synth_path(tag)
        return path, ("synth" if path is not None else "off")
    path = find_sfx(tag, rng)
    if path is not None:
        return path, "library"
    path = _generated_sfx_path(tag)
    if path is not None:
        return path, "generated"
    return None, "off"


def _register(timeline: Timeline, path: str, duration_s: float) -> str:
    """A SourceFile for an audio file, reused when the same file is already in."""
    for source in timeline.sources.values():
        if source.path == path:
            return source.id
    source_id = f"aud_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:8]}"
    timeline.sources[source_id] = SourceFile(
        id=source_id, path=path, duration_seconds=max(0.01, duration_s),
        width=0, height=0, has_audio=True, kind="audio")
    return source_id


def _probe_duration(path: str) -> float:
    try:
        from utils.ffmpeg_utils import get_video_duration
        return float(get_video_duration(path) or 0.0)
    except Exception:
        return 0.0


def file_duration(path: str) -> float:
    """Seconds of audio in a file: known for synthesised sounds, probed otherwise."""
    file = Path(path)
    if file.parent == SYNTH_DIR:
        for kind in sorted(SYNTH_KINDS, key=len, reverse=True):
            if file.name.startswith(kind + "_"):
                return synth_duration(kind)
    return _probe_duration(path)


def apply_sound(timeline: Timeline, plan: SoundPlan, settings: PresentationSettings,
                seed: int = 0, source_counts: Optional[Dict[str, int]] = None
                ) -> Dict[str, int]:
    """Put the plan on the timeline. Replaces the pass's own lanes only.

    `source_counts`, when given, is incremented in place with how many placed
    SFX came from each source ("library"/"generated"/"synth"/"off") — an
    opt-in the report reads from; every existing caller that omits it keeps
    today's return value unchanged.
    """
    for origin in (MUSIC_ORIGIN, SFX_ORIGIN, AMBIENCE_ORIGIN):
        clear_generated(timeline, origin)
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    counts = {"music": 0, "sfx": 0, "ambience": 0}
    rng = random.Random(seed)
    programme_frames = timeline.duration_frames

    def note_source(label: str) -> None:
        if source_counts is not None:
            source_counts[label] = source_counts.get(label, 0) + 1

    for cue in plan.music:
        duration = file_duration(cue.path)
        if duration <= 0.0:
            logger.warning("Music %s has no readable duration; skipped", cue.path)
            continue
        source_id = _register(timeline, cue.path, duration)
        span = time_to_frame(cue.end_s - cue.start_s, fps_num, fps_den)
        if span <= 0:
            continue
        item = clip_ops.add_media_item(
            timeline, source_id, MUSIC_TRACK,
            time_to_frame(cue.start_s, fps_num, fps_den), 0, span, origin=MUSIC_ORIGIN)
        item.loop = True
        item.volume = cue.gain
        item.duck = cue.duck
        item.audio_fade_in = cue.fade_in_s
        item.audio_fade_out = cue.fade_out_s
        item.label = f"Music: {Path(cue.path).stem}" + (f" [{cue.act}]" if cue.act else "")
        counts["music"] += 1

    for cue in plan.ambience:
        duration = file_duration(cue.path)
        if duration <= 0.0 or programme_frames <= 0:
            continue
        source_id = _register(timeline, cue.path, duration)
        item = clip_ops.add_media_item(
            timeline, source_id, AMBIENCE_TRACK, 0, 0, programme_frames,
            origin=AMBIENCE_ORIGIN)
        item.loop = True
        item.volume = cue.gain
        item.audio_fade_in = settings.fade_in_s
        item.audio_fade_out = settings.fade_out_s
        item.label = f"Ambience: {cue.kind}"
        counts["ambience"] += 1

    for cue in plan.sfx:
        path, src = _resolve_sfx(cue.tag, rng, settings.sfx_source)
        if path is None:
            note_source(src)
            continue
        duration = file_duration(str(path))
        if duration <= 0.0:
            continue
        source_id = _register(timeline, str(path), duration)
        start_frame = time_to_frame(cue.at_s, fps_num, fps_den)
        end_frame = min(start_frame + time_to_frame(duration, fps_num, fps_den),
                        programme_frames)
        if end_frame - start_frame <= 0:
            continue
        item = clip_ops.add_media_item(
            timeline, source_id, SFX_TRACK, start_frame, 0, end_frame - start_frame,
            origin=SFX_ORIGIN)
        item.volume = cue.gain
        # A short edge fade so a one-shot effect never clicks, whether the
        # source file itself starts/stops hard or the programme boundary
        # trimmed its tail.
        item.audio_fade_in = SFX_CLICK_FADE_S
        item.audio_fade_out = SFX_CLICK_FADE_S
        item.label = f"SFX: {cue.tag}"
        counts["sfx"] += 1
        note_source(src)

    for cue in plan.loops:
        path, src = _resolve_sfx(cue.tag, rng, settings.sfx_source)
        if path is None:
            note_source(src)
            continue
        duration = file_duration(str(path))
        if duration <= 0.0:
            continue
        source_id = _register(timeline, str(path), duration)
        start_frame = time_to_frame(cue.start_s, fps_num, fps_den)
        end_frame = min(time_to_frame(cue.end_s, fps_num, fps_den), programme_frames)
        if end_frame - start_frame <= 0:
            continue
        item = clip_ops.add_media_item(
            timeline, source_id, SFX_TRACK, start_frame, 0, end_frame - start_frame,
            origin=SFX_ORIGIN)
        item.loop = True
        item.volume = cue.gain
        item.audio_fade_in = 1.5
        item.audio_fade_out = 1.5
        item.label = f"SFX loop: {cue.tag}"
        counts["sfx"] += 1
        note_source(src)

    _drop_orphan_audio_sources(timeline)
    timeline.recalculate_duration()
    timeline.revision += 1
    return counts


def add_sfx_events(timeline: Timeline, events: Sequence[Tuple[float, str]],
                   settings: PresentationSettings, seed: int = 0) -> int:
    """Append effects from another planner (moods) without touching the rest."""
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    rng = random.Random(seed)
    existing = sfx_times(timeline)
    count = 0
    for at, tag in sorted(events):
        if tag not in SFX_TAGS:
            continue
        if any(abs(at - other) < MIN_SFX_GAP_S for other in existing):
            continue
        path, _src = _resolve_sfx(tag, rng, settings.sfx_source)
        if path is None:
            continue
        duration = file_duration(str(path))
        if duration <= 0.0:
            continue
        source_id = _register(timeline, str(path), duration)
        start_frame = time_to_frame(at, fps_num, fps_den)
        end_frame = min(start_frame + time_to_frame(duration, fps_num, fps_den),
                        timeline.duration_frames)
        if end_frame - start_frame <= 0:
            continue
        item = clip_ops.add_media_item(
            timeline, source_id, SFX_TRACK, start_frame, 0, end_frame - start_frame,
            origin=SFX_ORIGIN)
        item.volume = settings.sfx_volume
        item.audio_fade_in = SFX_CLICK_FADE_S
        item.audio_fade_out = SFX_CLICK_FADE_S
        item.label = f"SFX: {tag}"
        existing.append(at)
        count += 1
    return count


def master_for_preset(preset: str, loudness_lufs: Optional[float] = -14.0,
                      voice_enhance: str = "auto",
                      voice_fx: Optional[Sequence[Any]] = None) -> Optional[AudioMaster]:
    """The AudioMaster one named voice preset resolves to, with no timeline
    needed — used by `apply_voice_master` and the audio-preview endpoint alike.

    Returns None for "off"/"none"/"" (no treatment at all).
    """
    preset = (preset or "").lower()
    if preset in ("off", "none", ""):
        return None
    denoise, deess, compress = VOICE_PRESETS.get(preset, VOICE_PRESETS["studio_mic"])
    eq_preset = preset if preset in VOICE_PRESETS else "studio_mic"
    fx_dicts = [fx.model_dump() if hasattr(fx, "model_dump") else dict(fx)
                for fx in (voice_fx or [])]
    return AudioMaster(
        voice_denoise=denoise, voice_deess=deess, voice_compress=compress,
        voice_enhance=voice_enhance, voice_eq_preset=eq_preset, voice_fx=fx_dicts,
        loudness_lufs=loudness_lufs, origin=MASTER_ORIGIN)


def apply_voice_master(timeline: Timeline, settings: PresentationSettings) -> Optional[str]:
    """Set the programme's voice treatment and loudness target.

    A master the user set by hand (one with no origin that is not neutral, or
    a foreign origin) is left alone.
    """
    current = timeline.audio_master
    if current is not None and current.origin != MASTER_ORIGIN and not current.is_identity():
        return None
    preset = (settings.voice_preset or "").lower()
    master = master_for_preset(preset, settings.loudness_lufs, settings.voice_enhance,
                               settings.voice_fx)
    if master is None:
        if current is not None and current.origin == MASTER_ORIGIN:
            timeline.audio_master = None
        return None
    timeline.audio_master = master
    timeline.revision += 1
    return preset


def _drop_orphan_audio_sources(timeline: Timeline) -> None:
    used = {i.source_id for i in timeline.items if i.source_id}
    for source_id in list(timeline.sources):
        source = timeline.sources[source_id]
        if source.kind == "audio" and source_id not in used:
            del timeline.sources[source_id]


def sfx_times(timeline: Timeline) -> List[float]:
    from timeline.schema import frame_to_time
    return [frame_to_time(i.timeline_start_frame, timeline.fps_num, timeline.fps_den)
            for i in timeline.items if i.origin == SFX_ORIGIN]
