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
from typing import Dict, List, Optional, Sequence, Tuple

from config import DATA_DIR, FFMPEG_BIN
from utils.proc import NO_WINDOW
from timeline import clip_ops
from timeline.authoring import clear_generated
from timeline.schema import AudioMaster, SourceFile, Timeline, time_to_frame

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

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"}

# The effect vocabulary the planner may ask for. Anything else is ignored
# rather than guessed at.
SFX_TAGS = ("whoosh", "whoosh_soft", "pop", "boom", "riser", "stinger",
            "heartbeat", "thunder", "flash", "click")
SYNTH_KINDS = ("drone", "drone_low", "drone_high", "wind", "rain", "room_tone") + SFX_TAGS

# Two whooshes closer than this read as a stutter.
MIN_SFX_GAP_S = 1.2
# Music fades: long enough to breathe, short enough to be under the title.
MUSIC_FADE_IN_S = 1.5
MUSIC_FADE_OUT_S = 3.0
SYNTH_LOOP_SECONDS = 48.0

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

# Voice-master presets: (denoise, deess, compress).
VOICE_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "clean": (0.35, 0.3, 0.45),
    "podcast": (0.3, 0.35, 0.65),
    "horror_intimate": (0.5, 0.25, 0.7),
    "light": (0.2, 0.15, 0.25),
}


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


def find_music(genre: str, rng: random.Random,
               mood: Optional[str] = None,
               library: Path = MUSIC_DIR) -> Optional[Path]:
    """A music file for the genre from the user's library, or None.

    Looks in `<library>/<genre>/`, then `<library>/general/`, then loose files
    in `<library>/` whose manifest entry lists the genre (or has no genre at
    all). A manifest `mood` list narrows the choice when the caller has one.
    """
    candidates: List[Path] = _audio_files(library / genre)
    if not candidates:
        candidates = _audio_files(library / "general")
    if not candidates:
        manifest = _manifest(library)
        candidates = []
        for path in _audio_files(library):
            entry = manifest.get(path.name) or {}
            genres = _as_list(entry.get("genre") or entry.get("genres"))
            if not genres or genre in genres:
                candidates.append(path)
    if mood and candidates:
        manifest = _manifest(candidates[0].parent)
        moody = [p for p in candidates
                 if mood in _as_list((manifest.get(p.name) or {}).get("mood")
                                     or (manifest.get(p.name) or {}).get("moods"))]
        if moody:
            candidates = moody
    if not candidates:
        return None
    return rng.choice(candidates)


def find_sfx(tag: str, rng: random.Random, library: Path = SFX_DIR) -> Optional[Path]:
    files = _audio_files(library / tag)
    return rng.choice(files) if files else None


def find_ambience(kind: str, rng: random.Random,
                  library: Path = AMBIENCE_DIR) -> Optional[Path]:
    files = _audio_files(library / kind)
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


# --- planning ------------------------------------------------------------------

def plan_sound(timeline: Timeline, program: Program, settings: PresentationSettings,
               genre: str, seed: int,
               broll_windows: Optional[Sequence[Tuple[float, float]]] = None,
               popup_times: Optional[Sequence[float]] = None,
               extra_sfx: Optional[Sequence[Tuple[float, str]]] = None,
               loops: Optional[Sequence[Tuple[float, float, str, float]]] = None,
               sections: Optional[Sequence[Tuple[float, float, str]]] = None) -> SoundPlan:
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
        for start, end in (broll_windows or []):
            events.append((start, "whoosh", 1.0))
            if end - start >= 1.5 and end < duration - 0.5:
                events.append((end, "whoosh_soft", 0.7))
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
             rng: random.Random, previous: Optional[str]) -> Tuple[Optional[Path], bool]:
    """(file, synthesised) for one section, avoiding the previous section's file."""
    mood, variant, _ = ACT_MUSIC.get(act, ("", "", 1.0))
    path = find_music(genre, rng, mood=mood or None)
    if path is not None and previous and str(path) == previous:
        alternative = find_music(genre, random.Random(rng.random()), mood=mood or None)
        if alternative is not None and str(alternative) != previous:
            path = alternative
    if path is not None:
        return path, False
    kind = _SYNTH_BED_FOR_GENRE.get(genre)
    if kind and settings.music_synth_fallback:
        base = kind.replace("_low", "")
        candidate = f"{base}_{variant}" if variant else kind
        synth = synth_path(candidate) or synth_path(kind)
        return synth, synth is not None
    return None, False


def _plan_music(genre: str, settings: PresentationSettings, rng: random.Random,
                duration: float,
                sections: Optional[Sequence[Tuple[float, float, str]]] = None) -> List[MusicCue]:
    """One cue per act section, overlapping by the crossfade at each boundary."""
    cues: List[MusicCue] = []
    previous: Optional[str] = None
    parts = music_sections(sections, duration)
    for index, (start, end, act) in enumerate(parts):
        path, synthesised = _bed_for(genre, act, settings, rng, previous)
        if path is None:
            continue
        _, _, gain_mult = ACT_MUSIC.get(act, ("", "", 1.0))
        first, last = index == 0, index == len(parts) - 1
        cue_start = 0.0 if first else max(0.0, start - MUSIC_CROSSFADE_S / 2)
        cue_end = duration if last else min(duration, end + MUSIC_CROSSFADE_S / 2)
        cues.append(MusicCue(
            path=str(path), start_s=cue_start, end_s=cue_end,
            gain=settings.music_volume * gain_mult, duck=settings.music_duck,
            fade_in_s=MUSIC_FADE_IN_S if first else MUSIC_CROSSFADE_S,
            fade_out_s=min(MUSIC_FADE_OUT_S, duration / 3.0) if last else MUSIC_CROSSFADE_S,
            synthesised=synthesised, act=act))
        previous = str(path)
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
    kept.sort(key=lambda c: c.at_s)
    return kept


# --- placement -----------------------------------------------------------------

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
                seed: int = 0) -> Dict[str, int]:
    """Put the plan on the timeline. Replaces the pass's own lanes only."""
    for origin in (MUSIC_ORIGIN, SFX_ORIGIN, AMBIENCE_ORIGIN):
        clear_generated(timeline, origin)
    fps_num, fps_den = timeline.fps_num, timeline.fps_den
    counts = {"music": 0, "sfx": 0, "ambience": 0}
    rng = random.Random(seed)
    programme_frames = timeline.duration_frames

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
        item.audio_fade_in = 2.0
        item.audio_fade_out = 2.0
        item.label = f"Ambience: {cue.kind}"
        counts["ambience"] += 1

    for cue in plan.sfx:
        path = find_sfx(cue.tag, rng) or synth_path(cue.tag)
        if path is None:
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
        item.label = f"SFX: {cue.tag}"
        counts["sfx"] += 1

    for cue in plan.loops:
        path = find_sfx(cue.tag, rng) or synth_path(cue.tag)
        if path is None:
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
        path = find_sfx(tag, rng) or synth_path(tag)
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
        item.label = f"SFX: {tag}"
        existing.append(at)
        count += 1
    return count


def apply_voice_master(timeline: Timeline, settings: PresentationSettings) -> Optional[str]:
    """Set the programme's voice treatment and loudness target.

    A master the user set by hand (one with no origin that is not neutral, or
    a foreign origin) is left alone.
    """
    current = timeline.audio_master
    if current is not None and current.origin != MASTER_ORIGIN and not current.is_identity():
        return None
    preset = (settings.voice_preset or "clean").lower()
    if preset in ("off", "none", ""):
        if current is not None and current.origin == MASTER_ORIGIN:
            timeline.audio_master = None
        return None
    denoise, deess, compress = VOICE_PRESETS.get(preset, VOICE_PRESETS["clean"])
    timeline.audio_master = AudioMaster(
        voice_denoise=denoise, voice_deess=deess, voice_compress=compress,
        loudness_lufs=settings.loudness_lufs, origin=MASTER_ORIGIN)
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
