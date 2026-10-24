"""Generated background music (ACE-Step), and the licence guard on every
music/audio file the edit may use.

**Copyright policy.** A video only ever gets audio whose right to use is on
record:
- tracks generated here with ACE-Step 1.0 (Apache-2.0 model; the output is
  ours), each with a provenance entry — model, licence, tags, seed, date —
  in its folder's `manifest.json`;
- SFX generated with Stable Audio Open (Stability AI Community Licence:
  commercial use allowed under its revenue cap) and FFmpeg-synthesised beds;
- a file the user drops into `data/music|sfx|ambience/` only when its manifest
  entry names a licence from `ALLOWED_LICENSES` (see `license_ok`). A file
  with no licence on record is skipped and reported, never used on a guess.

**Styles.** `MUSIC_STYLES` maps a style name to ACE-Step tags, the genres it
serves and the act moods it fits (`sound.ACT_MUSIC`: intense/calm/tense/
soft/upbeat). Each generation is one variation (a seed); several variations
per style give different videos different tracks. Tracks land in
`data/music/<genre>/` so `sound.find_music` — and the editor's A2 lane —
treat them exactly like library files.

**Engines and builder.** `ENGINES` offers ACE-Step (full-length beds) and
Stable Audio Open (the SFX workflow, <= 47 s loops/stingers). A generation can
start from a style preset, from the builder's genre/instrument/mood/tempo
picks, or both -- `compose_tags` merges them, and a picked mood is filed
under its act mood (`MOODS`) so the edit still finds the track.

**GPU.** A batch starts with `gpu_handover.prepare_for_phase("music")` and
ends with `offload_all`, per the policy in runtime/gpu_handover.py.
"""

import datetime as _dt
import json
import logging
import os
import random
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np

from config import COMFYUI_BASE_DIR, FFMPEG_BIN
from utils.proc import NO_WINDOW

from . import workflows
from .sound import AUDIO_EXTS, MUSIC_DIR, _manifest

logger = logging.getLogger("presentation.music_gen")

MODEL_LABEL = "ACE-Step v1 3.5B"
MODEL_LICENSE = "Apache-2.0"
GENERATED_LICENSE = "generated"

# Licences under which a library file may go into a monetised video.
ALLOWED_LICENSES = frozenset({
    "generated", "own", "owned", "original", "cc0", "public-domain", "public domain",
    "cc-by", "cc-by-4.0", "cc-by-3.0", "royalty-free", "licensed", "youtube-audio-library",
    "stability-community", "apache-2.0",
})

MUSIC_PHASE_MIN_VRAM_MB = 9000.0   # 3.5B fp16 checkpoint + VAE + activations
MUSIC_PHASE_MIN_RAM_MB = 8000.0
DEFAULT_SECONDS = 120.0
MAX_SECONDS = 240.0
MAX_VARIATIONS = 6

# style: tags (ACE-Step's comma list), genres it serves, act moods it fits.
MUSIC_STYLES: Dict[str, Dict[str, Any]] = {
    "horror_dark": {
        "label": "Dark horror",
        "tags": "dark ambient, horror, cinematic, eerie drones, low strings, dissonant piano, "
                "slow, 60 bpm, ominous, suspense, instrumental",
        "genres": ["horror", "mystery"], "moods": ["tense", "intense", "calm"]},
    "horror_climax": {
        "label": "Horror climax",
        "tags": "horror, orchestral, aggressive strings, taiko drums, brass stabs, rising tension, "
                "90 bpm, terrifying, instrumental",
        "genres": ["horror"], "moods": ["intense"]},
    "suspense": {
        "label": "Suspense / thriller",
        "tags": "suspense, thriller, pulsing synth bass, ticking percussion, muted strings, "
                "100 bpm, tense, cinematic, instrumental",
        "genres": ["mystery", "true_crime", "documentary", "news", "geopolitics"],
        "moods": ["tense", "intense"]},
    "true_crime": {
        "label": "True crime noir",
        "tags": "noir, true crime documentary, dark piano, soft pads, subtle percussion, "
                "80 bpm, moody, minimal, instrumental",
        "genres": ["true_crime"], "moods": ["calm", "tense", "soft"]},
    "sad_violin": {
        "label": "Sad violin",
        "tags": "sad, emotional, solo violin, cello, soft piano, strings, 70 bpm, melancholic, "
                "slow, cinematic, instrumental",
        "genres": ["general", "vlog", "documentary", "motivational", "devotional"],
        "moods": ["soft", "calm"]},
    "morning_raga_flute": {
        "label": "Morning raga (bansuri flute)",
        "tags": "indian classical, hindustani, morning raga, alap, solo bansuri flute, "
                "tanpura drone, slow, calm, meditative, long sustained notes, no drums, "
                "no tabla, 60 bpm, serene, sunrise, instrumental",
        "genres": ["vlog", "devotional", "travel", "health_fitness", "general"],
        "moods": ["calm", "soft"]},
    "morning_raga_violin": {
        "label": "Morning raga (Indian classical violin)",
        "tags": "indian classical, hindustani, morning raga, alap, solo violin, tanpura drone, "
                "slow, calm, meditative, long sustained notes, gentle meend glides, no drums, "
                "no tabla, 60 bpm, serene, sunrise, instrumental",
        "genres": ["vlog", "devotional", "travel", "health_fitness", "general"],
        "moods": ["calm", "soft"]},
    "morning_raga_sitar": {
        "label": "Morning raga (sitar)",
        "tags": "indian classical, hindustani, morning raga, alap, solo sitar, tanpura drone, "
                "slow, calm, meditative, long sustained notes, no drums, no tabla, "
                "60 bpm, serene, sunrise, instrumental",
        "genres": ["vlog", "devotional", "travel", "health_fitness", "general"],
        "moods": ["calm", "soft"]},
    "evening_raga_flute": {
        "label": "Evening raga (bansuri flute)",
        "tags": "indian classical, hindustani, evening raga, alap, solo bansuri flute, "
                "tanpura drone, slow, calm, meditative, long sustained notes, no drums, "
                "no tabla, 58 bpm, serene, dusk, instrumental",
        "genres": ["vlog", "devotional", "travel", "health_fitness", "general"],
        "moods": ["calm", "soft"]},
    "evening_raga_violin": {
        "label": "Evening raga (Indian classical violin)",
        "tags": "indian classical, hindustani, evening raga, alap, solo violin, tanpura drone, "
                "slow, calm, meditative, long sustained notes, gentle meend glides, no drums, "
                "no tabla, 58 bpm, serene, dusk, instrumental",
        "genres": ["vlog", "devotional", "travel", "health_fitness", "general"],
        "moods": ["calm", "soft"]},
    "evening_raga_sitar": {
        "label": "Evening raga (sitar)",
        "tags": "indian classical, hindustani, evening raga, alap, solo sitar, tanpura drone, "
                "slow, calm, meditative, long sustained notes, no drums, no tabla, "
                "58 bpm, serene, dusk, instrumental",
        "genres": ["vlog", "devotional", "travel", "health_fitness", "general"],
        "moods": ["calm", "soft"]},
    "happy_flute": {
        "label": "Happy flute",
        "tags": "happy, uplifting, flute melody, acoustic guitar, light percussion, claps, "
                "110 bpm, cheerful, bright, instrumental",
        "genres": ["vlog", "cooking", "travel", "comedy", "general"],
        "moods": ["upbeat", "calm"]},
    "cinematic_epic": {
        "label": "Cinematic epic",
        "tags": "epic, cinematic, orchestral, choir pads, big drums, brass, soaring strings, "
                "120 bpm, heroic, trailer, instrumental",
        "genres": ["documentary", "geopolitics", "motivational", "gaming", "general"],
        "moods": ["intense", "upbeat"]},
    "dramatic": {
        "label": "Dramatic",
        "tags": "dramatic, cinematic, strings ostinato, piano, timpani, emotional build, "
                "95 bpm, intense, instrumental",
        "genres": ["documentary", "news", "geopolitics", "mystery", "general"],
        "moods": ["intense", "tense"]},
    "motivational": {
        "label": "Motivational",
        "tags": "inspirational, motivational, piano, strings, uplifting drums, building, "
                "115 bpm, hopeful, cinematic, instrumental",
        "genres": ["motivational", "vlog", "finance", "health_fitness", "general"],
        "moods": ["upbeat", "intense"]},
    "lofi_calm": {
        "label": "Calm lo-fi",
        "tags": "lo-fi, chill, mellow electric piano, vinyl crackle, soft drums, warm bass, "
                "80 bpm, relaxed, cozy, instrumental",
        "genres": ["vlog", "tech", "science_education", "cooking", "general"],
        "moods": ["calm", "soft"]},
    "tech_minimal": {
        "label": "Tech minimal",
        "tags": "electronic, minimal, clean synth arpeggio, soft kick, modern, "
                "110 bpm, focused, corporate, instrumental",
        "genres": ["tech", "finance", "science_education", "news"],
        "moods": ["calm", "upbeat"]},
    "comedy_playful": {
        "label": "Playful comedy",
        "tags": "playful, comedic, pizzicato strings, bassoon, ukulele, bouncy, "
                "120 bpm, quirky, fun, instrumental",
        "genres": ["comedy", "gaming"], "moods": ["upbeat"]},
    "devotional_calm": {
        "label": "Devotional calm",
        "tags": "devotional, spiritual, sitar, harmonium, tanpura, soft bells, "
                "70 bpm, peaceful, meditative, instrumental",
        "genres": ["devotional"], "moods": ["calm", "soft"]},
}

# Slow, calm Hindustani ragas -- morning and evening -- for ACE-Step 1.5's real
# bpm/key conditioning. `bpm` is a slow default (50-66); `keyscale` is a plain
# approximation (ACE-Step 1.5 only takes a Western root + major/minor), picked
# to sit close to the raga's characteristic mood, not a claim of equivalence.
RAGAS: Dict[str, Dict[str, Any]] = {
    "bhairav": {
        "label": "Bhairav", "time": "morning",
        "description": "komal re and komal dha, devotional dawn gravity",
        "keyscale": "C major", "bpm": 56},
    "ahir_bhairav": {
        "label": "Ahir Bhairav", "time": "morning",
        "description": "komal re with shuddh dha, gentler and warmer than Bhairav",
        "keyscale": "C major", "bpm": 58},
    "lalit": {
        "label": "Lalit", "time": "morning",
        "description": "both madhyams and no pancham, wistful pre-dawn stillness",
        "keyscale": "F major", "bpm": 54},
    "bilaskhani_todi": {
        "label": "Bilaskhani Todi", "time": "morning",
        "description": "komal re, ga and dha, tender plaintive dawn sorrow",
        "keyscale": "C minor", "bpm": 52},
    "bhatiyar": {
        "label": "Bhatiyar", "time": "morning",
        "description": "shuddh and komal dha both used, open pastoral sunrise feel",
        "keyscale": "D major", "bpm": 60},
    "yaman": {
        "label": "Yaman", "time": "evening",
        "description": "tivra madhyam and no shuddh madhyam, warm devotional evening calm",
        "keyscale": "C major", "bpm": 62},
    "puriya_dhanashri": {
        "label": "Puriya Dhanashri", "time": "evening",
        "description": "komal re and dha with tivra ma, tender bittersweet dusk",
        "keyscale": "C minor", "bpm": 58},
    "marwa": {
        "label": "Marwa", "time": "evening",
        "description": "komal re, shuddh dha and no pancham, glowing restless sunset tension",
        "keyscale": "Db major", "bpm": 60},
    "bageshree": {
        "label": "Bageshree", "time": "evening",
        "description": "komal gandhar and nishad, tender longing, late-evening calm",
        "keyscale": "C minor", "bpm": 56},
    "shree": {
        "label": "Shree", "time": "evening",
        "description": "komal re and tivra ma, solemn grand dusk devotion",
        "keyscale": "C major", "bpm": 58},
}

# The style a genre's bed is made from when nothing else is asked for.
DEFAULT_STYLE_FOR_GENRE: Dict[str, str] = {
    "horror": "horror_dark", "mystery": "suspense", "true_crime": "true_crime",
    "comedy": "comedy_playful", "gaming": "cinematic_epic", "tech": "tech_minimal",
    "science_education": "lofi_calm", "finance": "tech_minimal",
    "motivational": "motivational", "health_fitness": "motivational",
    "cooking": "happy_flute", "travel": "morning_raga_flute", "devotional": "devotional_calm",
    "news": "suspense", "vlog": "lofi_calm", "documentary": "cinematic_epic",
    "geopolitics": "dramatic", "general": "cinematic_epic",
}


# ACE-Step 1.5's own model files (the turbo UNET, its two Qwen text encoders
# and its VAE) -- checked on disk before the engine is reported "ready", since
# the workflow graph alone always loads whether or not the weights are there.
ACE_STEP15_FILES: Dict[str, Any] = {
    "diffusion_models": "acestep_v1.5_turbo.safetensors",
    "text_encoders": ["qwen_0.6b_ace15.safetensors", "qwen_1.7b_ace15.safetensors"],
    "vae": "ace_1.5_vae.safetensors",
}
# fp16 turbo UNET (~4.7 GB) + two Qwen text encoders (~1.2 GB, ~3.7 GB) + VAE,
# loaded together for one generation.
ACE_STEP15_MIN_VRAM_MB = 11000.0

# Every ComfyUI install's `models/` this machine might have the 1.5 files
# under -- the configured install (settings/env, same as COMFYUI_BASE_DIR) and
# the B: portable tree, since either one may hold what the Desktop downloaded.
_COMFYUI_MODEL_ROOTS: List[Path] = [
    Path(COMFYUI_BASE_DIR) / "models",
    Path(os.environ.get(
        "COMFYUI_PORTABLE_BASE_DIR",
        r"B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI",
    )) / "models",
]


def _model_file_exists(subdir: str, filename: str) -> bool:
    return any((root / subdir / filename).is_file() for root in _COMFYUI_MODEL_ROOTS)


def ace_step15_ready() -> bool:
    """Whether every ACE-Step 1.5 model file is on disk in a known ComfyUI
    install (its workflow graph alone always resolves, files or not)."""
    if not _model_file_exists("diffusion_models", ACE_STEP15_FILES["diffusion_models"]):
        return False
    if not all(_model_file_exists("text_encoders", f) for f in ACE_STEP15_FILES["text_encoders"]):
        return False
    return _model_file_exists("vae", ACE_STEP15_FILES["vae"])


# Engines a bed can be made with. Each runs a ComfyUI workflow role: ACE-Step
# 1.5 ("music15") and ACE-Step 1.0 ("music") make full-length beds -- 1.5
# actually conditions on bpm/key instead of only tagging them; Stable Audio
# Open ("audio", the SFX workflow) caps at 47 s, so it serves loops, intros
# and stingers.
ENGINES: Dict[str, Dict[str, Any]] = {
    "ace_step15": {
        "label": "ACE-Step 1.5", "model": "ACE-Step 1.5 Turbo",
        # Not confirmed offline against the model's own card; treated like a
        # generated-output licence (GENERATED_LICENSE) the same as v1 until checked.
        "model_license": "Unverified (treated as generated, like v1)",
        "role": "music15", "max_seconds": 360.0, "license": GENERATED_LICENSE,
        "vram_mb": ACE_STEP15_MIN_VRAM_MB, "negative": "",
        "note": "Full-length beds with real bpm/key conditioning -- best for tempo-accurate "
                "music and the slow ragas."},
    "ace_step": {
        "label": "ACE-Step", "model": MODEL_LABEL, "model_license": MODEL_LICENSE,
        "role": "music", "max_seconds": MAX_SECONDS, "license": GENERATED_LICENSE,
        "vram_mb": MUSIC_PHASE_MIN_VRAM_MB, "negative": "",
        "note": "Full-length beds up to 4 min; tempo is a tag, not conditioned."},
    "stable_audio": {
        "label": "Stable Audio Open", "model": "Stable Audio Open 1.0",
        "model_license": "Stability AI Community Licence",
        "role": "audio", "max_seconds": 47.0, "license": "stability-community",
        "vram_mb": 6000.0,
        "negative": "vocals, singing, speech, low quality, distorted, noise, clipping",
        "note": "Short loops and stingers up to 47 s, for intros, transitions and Shorts."},
}
DEFAULT_ENGINE = "ace_step15"


def engine_ready(engine: str) -> bool:
    """Whether an engine can actually generate right now -- its workflow
    resolves AND, for ACE-Step 1.5, its model files are on disk."""
    eng = ENGINES.get(engine)
    if eng is None:
        return False
    if engine == "ace_step15" and not ace_step15_ready():
        return False
    return workflows.resolve(eng["role"]) is not None


def resolve_engine(engine: str, needs_tempo: bool = False) -> Tuple[str, Optional[str]]:
    """The engine to actually generate with, and a note when it was swapped.

    The default engine (ACE-Step 1.5) falls back to v1 while its model files
    are not installed. The other way round, a request that depends on tempo or
    calm (a bpm, a raga, a slow/calm mood -- `needs_tempo`) is moved off v1
    onto 1.5 whenever 1.5 is installed: v1 has no tempo conditioning, and on
    2026-10-20 two Yaman takes on v1 measured 95-148 bpm for a slow request in
    all six attempts."""
    if needs_tempo and engine == "ace_step" and "ace_step15" in ENGINES and ace_step15_ready():
        return "ace_step15", (f"{ENGINES['ace_step']['label']} cannot hold a tempo; "
                              f"used {ENGINES['ace_step15']['label']} instead.")
    if engine == DEFAULT_ENGINE and engine in ENGINES and not ace_step15_ready():
        note = (f"{ENGINES[engine]['label']} model files are not installed yet; "
                f"used {ENGINES['ace_step']['label']} instead.")
        return "ace_step", note
    return engine, None

# The builder's vocabulary. Anything outside it still works through extra_tags.
MUSIC_GENRES: List[str] = [
    "cinematic", "orchestral", "ambient", "lo-fi", "hip hop", "trap", "electronic", "edm",
    "synthwave", "pop", "rock", "acoustic", "jazz", "classical", "piano solo",
    "indian classical", "bollywood", "sufi", "indian folk", "devotional",
]
INSTRUMENTS: Dict[str, List[str]] = {
    "Keys": ["piano", "electric piano", "synth pads", "organ", "harmonium"],
    "Strings": ["violin", "cello", "string section", "acoustic guitar", "electric guitar",
                "bass guitar", "sitar", "sarangi", "santoor", "ukulele"],
    "Wind & brass": ["bansuri flute", "flute", "shehnai", "saxophone", "trumpet", "brass section"],
    "Percussion": ["tabla", "dholak", "dhol", "drums", "taiko drums", "808 bass", "claps"],
    "Texture": ["choir", "tanpura drone", "bells", "synth arpeggio", "vinyl crackle"],
}
# Descriptive mood -> the act mood the edit looks for (sound.ACT_MUSIC).
MOODS: Dict[str, str] = {
    "calm": "calm", "peaceful": "calm", "dreamy": "calm",
    "sad": "soft", "romantic": "soft", "nostalgic": "soft",
    "happy": "upbeat", "hopeful": "upbeat", "playful": "upbeat", "energetic": "upbeat",
    "dark": "tense", "eerie": "tense", "mysterious": "tense", "suspenseful": "tense",
    "epic": "intense", "dramatic": "intense", "aggressive": "intense", "triumphant": "intense",
}
MIN_BPM, MAX_BPM = 50, 180
_BPM = re.compile(r"\b\d{2,3}\s*bpm\b", re.IGNORECASE)


# Words that make a preset tag an instrument (or a mood), so the builder's
# picks can override the preset's: "soft piano" in Sad violin must not survive
# a pick of violin alone.
_INSTRUMENT_WORDS = {w for items in INSTRUMENTS.values() for i in items for w in i.split()
                     if w not in ("section", "solo", "acoustic", "electric", "bass")} | {
    "strings", "guitar", "drum", "drums", "percussion", "synth", "bass", "kick", "pads",
    "brass", "bassoon", "pizzicato", "timpani", "stabs", "sax", "piano", "claps"}
# Every mood word the presets use, by act mood; a preset mood is dropped only
# when it points at a different act mood than the ones picked.
_MOOD_ACT = {**MOODS,
             "emotional": "soft", "melancholic": "soft", "moody": "soft",
             "uplifting": "upbeat", "cheerful": "upbeat", "bright": "upbeat", "fun": "upbeat",
             "quirky": "upbeat", "comedic": "upbeat", "inspirational": "upbeat",
             "serene": "calm", "relaxed": "calm", "cozy": "calm", "meditative": "calm",
             "spiritual": "calm", "focused": "calm",
             "ominous": "tense", "tense": "tense",
             "terrifying": "intense", "heroic": "intense", "intense": "intense"}


def _words(tag: str) -> set:
    return set(re.findall(r"[a-z0-9]+", tag.lower()))


# Tempo in words, for ACE-Step 1.5's caption (its `tags` input reads best as a
# natural-language description, not a keyword list). Ceiling bpm -> word.
_TEMPO_WORDS: List[Tuple[int, str]] = [
    (66, "very slow"), (85, "slow"), (105, "moderate"), (130, "upbeat"), (10**6, "fast"),
]


def _tempo_word(bpm: Optional[int]) -> str:
    if not bpm:
        return ""
    for ceiling, word in _TEMPO_WORDS:
        if bpm <= ceiling:
            return word
    return "fast"


def compose_caption(style: Optional[str] = None, music_genres: Optional[List[str]] = None,
                    instruments: Optional[List[str]] = None, moods: Optional[List[str]] = None,
                    bpm: Optional[int] = None, raga: Optional[str] = None,
                    extra_tags: str = "") -> str:
    """The natural-language caption ACE-Step 1.5's `tags` input reads best as --
    built from the raga (if one is picked), the instrument, the style/moods and
    tempo words. Only used for the "ace_step15" engine; v1 and Stable Audio Open
    keep the plain tag list from `compose_prompt`."""
    raga_spec = RAGAS.get(raga) if raga else None
    instrument = instruments[0] if instruments else None
    tempo_word = _tempo_word(bpm or (raga_spec["bpm"] if raga_spec else None))
    pieces: List[str] = []
    if raga_spec:
        time_word = raga_spec["time"]
        lead = f", played on solo {instrument}" if instrument else ""
        pieces.append(f"A {tempo_word}, calm Hindustani classical alap in the {time_word} raga "
                      f"{raga_spec['label']}{lead} over a soft tanpura drone.")
        pieces.append(f"{raga_spec['description'].capitalize()}.")
        pieces.append("Long sustained notes with gentle meend glides, spacious and unhurried, "
                      "no drums, no percussion.")
    elif style and style in MUSIC_STYLES:
        label = MUSIC_STYLES[style]["label"]
        lead_word = f"A {tempo_word} {label.lower()} piece" if tempo_word else f"A {label.lower()} piece"
        pieces.append(lead_word + ".")
        if instrument:
            pieces.append(f"Led by solo {instrument}.")
        if moods:
            pieces.append(f"Mood: {', '.join(moods)}.")
    else:
        descriptor = ", ".join(p for p in (tempo_word, ", ".join(moods or [])) if p)
        lead = instrument or (music_genres[0] if music_genres else "instrumental")
        pieces.append((f"A {descriptor} {lead} piece." if descriptor else f"A {lead} piece."))
    if (extra_tags or "").strip():
        pieces.append(extra_tags.strip().rstrip(".") + ".")
    pieces.append("Instrumental.")
    return " ".join(p for p in pieces if p).strip()


def compose_prompt(style: Optional[str] = None, music_genres: Optional[List[str]] = None,
                   instruments: Optional[List[str]] = None, moods: Optional[List[str]] = None,
                   bpm: Optional[int] = None, extra_tags: str = "", raga: Optional[str] = None,
                   engine: Optional[str] = None) -> Dict[str, Any]:
    """The prompt the model gets: {"tags", "dropped"} (plus "caption" for
    "ace_step15", whose tags are led by a natural-language caption).

    Picks lead (the models weight early tags most): genres, then instruments
    (one pick becomes "solo <instrument>"), then moods, then the preset's tags.
    A preset tag naming an instrument the user did not pick is dropped once any
    instrument is picked, and likewise a preset mood word once a mood is
    picked -- so "Sad violin" + violin is violin, not violin over soft piano.
    Tempo replaces the preset's bpm. Always ends "instrumental".

    `raga`/`engine` only change anything for "ace_step15" (a composed caption
    is prepended to the tags); v1 and Stable Audio Open behave exactly as
    before.
    """
    music_genres, instruments, moods = list(music_genres or []), list(instruments or []), list(moods or [])
    picked_instr_words = set().union(*(_words(i) for i in instruments)) if instruments else set()
    picked_acts = {MOODS.get(m) for m in moods} - {None}
    base = [t.strip() for t in (MUSIC_STYLES[style]["tags"] if style else "").split(",") if t.strip()]
    kept, dropped = [], []
    for tag in base:
        words = _words(tag)
        if _BPM.search(tag) or tag.lower() == "instrumental":
            continue
        if instruments and words & _INSTRUMENT_WORDS and not words & picked_instr_words:
            dropped.append(tag)
        elif (picked_acts and tag.lower() in _MOOD_ACT
              and _MOOD_ACT[tag.lower()] not in picked_acts):
            dropped.append(tag)
        else:
            kept.append(tag)
    preset_bpm = next((t for t in base if _BPM.search(t)), None)
    if bpm:
        tempo = f"{max(MIN_BPM, min(MAX_BPM, int(bpm)))} bpm"
    else:
        tempo = preset_bpm
    lead_instruments = [f"solo {instruments[0]}"] if len(instruments) == 1 else instruments
    ordered = [*music_genres, *lead_instruments, *moods, *kept,
               *(extra_tags or "").split(","), tempo or "", "instrumental"]
    seen, out = set(), []
    for tag in ordered:
        tag = tag.strip()
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            out.append(tag)
    tags = ", ".join(out)
    result: Dict[str, Any] = {"tags": tags, "dropped": dropped}
    if engine == "ace_step15":
        caption = compose_caption(style, music_genres, instruments, moods, bpm, raga, extra_tags)
        result["caption"] = caption
        result["tags"] = f"{caption} {tags}" if tags else caption
    return result


def compose_tags(*args, **kwargs) -> str:
    return compose_prompt(*args, **kwargs)["tags"]


def style_for(genre: str, mood: Optional[str] = None) -> str:
    """The best style for a genre (and an act mood, when one is given)."""
    default = DEFAULT_STYLE_FOR_GENRE.get(genre, "cinematic_epic")
    if not mood:
        return default
    if mood in MUSIC_STYLES[default]["moods"]:
        return default
    for name, spec in MUSIC_STYLES.items():
        if genre in spec["genres"] and mood in spec["moods"]:
            return name
    return default


def license_ok(path: Path) -> bool:
    """Whether this library file's manifest entry names an allowed licence."""
    entry = _manifest(path.parent).get(path.name) or {}
    license_name = str(entry.get("license") or "").strip().lower()
    return license_name in ALLOWED_LICENSES


_MASTER_FADES = "afade=t=in:st=0:d=1.5,areverse,afade=t=in:st=0:d=2.5,areverse"
# LRA 20 keeps loudnorm's second pass linear for wide-range music (a sparse alap
# measures LRA ~12-18); with LRA 11 it fell back to dynamic gain riding.
_MASTER_TARGET = "I=-16:TP=-1.5:LRA=20"


def _master(source: Path, destination: Path) -> bool:
    """Loudness-normalise a generated bed and fade its edges, once.

    Two passes: measure, then apply ONE steady gain (loudnorm linear=true).
    The old single pass was loudnorm's dynamic mode, which rides the gain up
    in every quiet gap -- audible breathing on a slow raga, and false "notes"
    that made a 52 bpm take read 68 (2026-10-20). Falls back to the single
    pass if the measurement cannot be read."""
    measure_cmd = [FFMPEG_BIN, "-hide_banner", "-nostats", "-i", str(source),
                   "-af", f"{_MASTER_FADES},loudnorm={_MASTER_TARGET}:print_format=json",
                   "-f", "null", "-"]
    af = f"{_MASTER_FADES},loudnorm={_MASTER_TARGET}"
    try:
        probe = subprocess.run(measure_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=180, creationflags=NO_WINDOW)
        text = probe.stderr.decode("utf-8", "ignore")
        blob = text[text.rfind("{"): text.rfind("}") + 1]
        m = json.loads(blob)
        af = (f"{_MASTER_FADES},loudnorm={_MASTER_TARGET}:measured_I={m['input_i']}:"
              f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
              f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    except Exception as e:
        logger.info("Loudness measurement of %s failed (%s); single-pass master", source, e)
    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
               "-af", af, "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(destination)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=180, creationflags=NO_WINDOW)
    except Exception as e:
        logger.warning("Could not master generated music %s: %s", source, e)
        return False
    return destination.exists() and destination.stat().st_size > 1000


def _record(folder: Path, filename: str, entry: Dict[str, Any]) -> None:
    manifest = _manifest(folder)
    manifest[filename] = entry
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                          encoding="utf-8")


def library_tracks(library: Path = MUSIC_DIR) -> List[Dict[str, Any]]:
    """Every music file in the library with its folder, manifest entry and
    whether its licence clears it for use."""
    out: List[Dict[str, Any]] = []
    if not library.exists():
        return out
    folders = [library] + sorted(p for p in library.iterdir() if p.is_dir())
    for folder in folders:
        manifest = _manifest(folder)
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
                continue
            entry = manifest.get(path.name) or {}
            out.append({
                "file": str(path.relative_to(library)).replace("\\", "/"),
                "genre": folder.name if folder != library else "",
                "style": entry.get("style", ""),
                "label": entry.get("label", path.stem),
                "moods": entry.get("mood") or [],
                "license": entry.get("license", ""),
                "license_ok": license_ok(path),
                "source": entry.get("source", ""),
                "seed": entry.get("seed"),
                "seconds": entry.get("seconds"),
                "engine": entry.get("engine", ""),
                "timings": entry.get("timings") or {},
                "bpm": entry.get("bpm"),
                "instruments": entry.get("instruments") or [],
                "created": entry.get("created", ""),
                "raga": entry.get("raga"),
                "measured_bpm": entry.get("measured_bpm"),
                "onset_rate": entry.get("onset_rate"),
                "attempts": entry.get("attempts"),
                "passed": entry.get("passed"),
            })
    return out


# Stage reporting: `on_stage(key, fraction)` marks `key` as the running stage
# (every earlier one is then finished) and, for the sampler, how far it is.
# Keys: "gpu", "v<n>:load", "v<n>:sample", "v<n>:decode", "v<n>:master", "offload".
StageFn = Callable[[str, Optional[float]], None]
VARIATION_STAGES = [("load", "Load model"), ("sample", "Sampling"),
                    ("decode", "Decode & save"), ("master", "Master (loudness, fades)")]
_SAMPLERS = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")


class _StageClock:
    """Forwards stage changes and keeps each stage's duration for the manifest."""

    def __init__(self, prefix: str, on_stage: Optional[StageFn]):
        self.prefix, self.on_stage = prefix, on_stage
        self.current: Optional[str] = None
        self.since = self.start = time.monotonic()
        self.timings: Dict[str, float] = {}

    def __call__(self, stage: str, fraction: Optional[float] = None) -> None:
        if stage != self.current:
            now = time.monotonic()
            if self.current:
                self.timings[self.current] = round(now - self.since, 1)
            self.current, self.since = stage, now
        if self.on_stage:
            try:
                self.on_stage(f"{self.prefix}{stage}", fraction)
            except Exception as e:
                logger.debug("Stage report failed: %s", e)

    def finish(self) -> Dict[str, float]:
        now = time.monotonic()
        if self.current:
            self.timings[self.current] = round(now - self.since, 1)
        self.timings["total"] = round(now - self.start, 1)
        return self.timings


def _comfy_listener(graph: Dict[str, Any], clock: _StageClock):
    """Map ComfyUI's executing/progress events onto load/sample/decode."""
    classes = {nid: str(node.get("class_type", "")) for nid, node in graph.items()
               if isinstance(node, dict)}

    def on_event(kind: str, data: Dict[str, Any]) -> None:
        if kind == "executing" and data.get("node") is not None:
            cls = classes.get(str(data["node"]), "")
            if cls in _SAMPLERS:
                clock("sample", 0.0)
            elif cls.startswith(("VAEDecode", "Save")):
                clock("decode")
        elif kind == "progress" and data.get("max"):
            clock("sample", min(1.0, float(data.get("value", 0)) / float(data["max"])))
    return on_event


def _generation_meta(style: Optional[str], music_genres: List[str], instruments: List[str],
                     moods: List[str], genre: Optional[str]) -> Dict[str, Any]:
    """Label, act moods and video genres for a preset or a custom build."""
    spec = MUSIC_STYLES.get(style) if style else None
    act_moods = sorted({MOODS[m] for m in moods if m in MOODS})
    if spec:
        return {"label": spec["label"], "moods": act_moods or list(spec["moods"]),
                "genres": list(spec["genres"]), "folder": genre or spec["genres"][0]}
    picked = [*music_genres, *instruments][:3]
    return {"label": "Custom: " + ", ".join(picked) if picked else "Custom",
            "moods": act_moods, "genres": [genre or "general"], "folder": genre or "general"}


# --- felt speed -------------------------------------------------------------
# `librosa.feature.tempo` carries a prior centred on ~120 bpm, so a slow raga
# alap -- its melody moving at a couple of notes a second, with nothing
# periodic for a beat tracker to lock onto -- gets read back at roughly double
# its real pace (measured 2026-10-20: a 50 bpm Yaman take read 99-103 across
# its tempogram peaks). `measure_speed` separates "is there actually a pulse"
# from "how fast does this feel", and only trusts a tempogram reading once the
# first question is answered yes.
TEMPO_PEAK_MIN_BPM = 40.0
TEMPO_PEAK_MAX_BPM = 220.0
TEMPO_PEAK_COUNT = 4
# A tempogram peak at least this many times the median bin (in the same
# 40-220 bpm range) is a clear, periodic groove -- a kick/bass pulse the
# autocorrelation keeps finding every cycle. Below this ratio the "peak" is
# only ever the loudest bin of an otherwise flat read-out: a raga alap, a
# tanpura drone, an ambient pad. Chosen empirically against the controlled
# 2026-10-20 takes (the lo-fi drums+piano reads clear well above 2x; the
# Yaman flute's peaks sit within ~1.1x of each other and of the median).
PULSE_CLARITY_THRESHOLD = 1.8
EVENTS_PER_BEAT = 1.0   # melodic/onset events per "felt beat" for unpulsed audio
TEMPO_TOLERANCE = 0.06  # lock triggers once felt tempo is off by more than this
# One rubberband pass, clamped. Beyond ~0.7x a bowed or blown line smears into
# audible warble (a Yaman violin pushed to 0.55 read FASTER after, 2026-10-20):
# a take that far off is a retry's job, not the stretch's.
TEMPO_STRETCH_MIN, TEMPO_STRETCH_MAX = 0.7, 1.45
FINAL_TEMPO_TOLERANCE = 0.10   # the manifest's "passed" is within 10% of asked

# --- verify and retry -------------------------------------------------------
# A generated track is measured against what was actually asked for and
# regenerated with a new seed (up to VERIFY_MAX_ATTEMPTS times) when it does
# not match -- the best-scoring attempt (closest felt tempo, then fewest
# onsets) is what reaches the library, even on a run where none of them pass.
# The winning attempt is then tempo-locked (see `_lock_tempo`) before it is
# mastered, so a retry's job is to find the take that needs the least
# correction, not to land on the target by chance.
VERIFY_TEMPO_TOLERANCE = 0.15      # within 15% of the requested bpm, or half/double it
VERIFY_MAX_ONSET_RATE = 2.5        # onsets/s ceiling for a slow/calm request
# ...and a floor: a take with almost no notes is a near-silent drone, not a slow
# raga (2026-10-20: a Bhairav attempt read 0.09 notes/s). Never the kept take.
VERIFY_MIN_ONSET_RATE = 0.25
TEMPO_LOCK_ROUNDS = 3              # stretch, trim, measure the delivered audio, adjust
VERIFY_ONSET_WINDOW_S = 60.0       # kept for callers; measure_speed now reads the whole track
# Note counting (measure_speed): SuperFlux onset strength (lag 2, max filter 3)
# so a held note's vibrato is not counted as new notes, at least this many
# seconds between notes, and this peak threshold. Calibrated 2026-10-20 on
# synthetic vibrato notes at 0.4-2.5 notes/s (read within ~5%); librosa's
# default detector read 1.5 notes/s as 2.9 and made slow ragas look fast.
NOTE_MIN_GAP_S = 0.25
NOTE_DELTA = 0.2
VERIFY_MAX_ATTEMPTS = 3


def _tempo_peaks(y: np.ndarray, sr: int) -> Tuple[List[Tuple[float, float]], float]:
    """The tempogram's local-maximum bpm readings between `TEMPO_PEAK_MIN_BPM`
    and `TEMPO_PEAK_MAX_BPM`, strongest first (up to `TEMPO_PEAK_COUNT`), and
    the pulse-clarity score described above `PULSE_CLARITY_THRESHOLD`."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tg_mean = np.mean(tempogram, axis=1)
    freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
    valid = np.isfinite(freqs) & (freqs >= TEMPO_PEAK_MIN_BPM) & (freqs <= TEMPO_PEAK_MAX_BPM)
    if not np.any(valid):
        return [], 0.0
    order = np.argsort(freqs[valid])
    bpms, vals = freqs[valid][order], tg_mean[valid][order]
    is_peak = np.ones(len(vals), dtype=bool)
    if len(vals) > 2:
        is_peak[1:-1] = (vals[1:-1] >= vals[:-2]) & (vals[1:-1] >= vals[2:])
    peak_bpms, peak_vals = bpms[is_peak], vals[is_peak]
    top = np.argsort(peak_vals)[::-1][:TEMPO_PEAK_COUNT]
    peaks = [(float(peak_bpms[i]), float(peak_vals[i])) for i in top]
    median = float(np.median(vals)) if len(vals) else 0.0
    clarity = (peaks[0][1] / median) if peaks and median > 0 else 0.0
    return peaks, clarity


def measure_speed(path: Path, requested_bpm: Optional[float] = None,
                  slow_calm: bool = False) -> Dict[str, Optional[float]]:
    """How fast `path` actually feels.

    `pulse_bpm` is the tempogram's single strongest reading in
    `TEMPO_PEAK_MIN_BPM`-`TEMPO_PEAK_MAX_BPM`, for reference. `onset_rate` is
    notes/s over the whole track (vibrato-resistant, see `NOTE_MIN_GAP_S`), so a
    reading before and after a stretch covers the same music. `pulsed` is
    whether that reading is a clear, periodic groove rather than the loudest
    bin of an otherwise flat tempogram (`PULSE_CLARITY_THRESHOLD`).

    `felt_bpm` is what matters: for pulsed audio that is not a free-rhythm
    line, the one of the top `TEMPO_PEAK_COUNT` tempogram peaks nearest
    `requested_bpm` in log ratio (a groove's bpm conditioning legitimately
    lands on either octave -- a listener hears the same beat). Otherwise --
    unpulsed audio, or `slow_calm` set for a free-rhythm line (a raga alap)
    -- `onset_rate * 60 / EVENTS_PER_BEAT`, since an alap's or ambient
    line's felt pace is how often a new note lands, not any tempogram bin.
    A slow GROOVE (lo-fi at 70) is pulsed and must be read by its beat: its
    hi-hats double the onset rate, and reading it by onsets halved a correct
    70 bpm take (2026-10-20). With no `requested_bpm`, a pulsed reading
    reports its single strongest peak: there is nothing to correct towards.

    None values on a file librosa cannot read, rather than raising -- a
    measurement failure should not be mistaken for a track that failed
    verification."""
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as e:
        logger.warning("Could not analyse %s: %s", path, e)
        return {"pulse_bpm": None, "onset_rate": None, "pulsed": False, "felt_bpm": None}
    if y.size == 0:
        return {"pulse_bpm": None, "onset_rate": None, "pulsed": False, "felt_bpm": None}

    try:
        peaks, clarity = _tempo_peaks(y, sr)
    except Exception as e:
        logger.warning("Tempogram failed for %s: %s", path, e)
        peaks, clarity = [], 0.0
    pulse_bpm = round(peaks[0][0], 1) if peaks else None
    pulsed = bool(peaks) and clarity >= PULSE_CLARITY_THRESHOLD

    window = y
    window_s = max(len(window) / sr, 1e-6)
    try:
        envelope = librosa.onset.onset_strength(y=window, sr=sr, lag=2, max_size=3)
        onsets = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, units="time",
                                            wait=max(1, int(NOTE_MIN_GAP_S * sr / 512)),
                                            delta=NOTE_DELTA)
        onset_rate = round(len(onsets) / window_s, 2)
    except Exception as e:
        logger.warning("Onset detection failed for %s: %s", path, e)
        onset_rate = None

    if pulsed and not slow_calm:
        if requested_bpm:
            felt_bpm = min((bpm for bpm, _ in peaks),
                           key=lambda bpm: abs(np.log(bpm / requested_bpm)))
        else:
            felt_bpm = pulse_bpm
    else:
        felt_bpm = round(onset_rate * 60.0 / EVENTS_PER_BEAT, 1) if onset_rate is not None else None

    return {"pulse_bpm": pulse_bpm, "onset_rate": onset_rate, "pulsed": pulsed,
            "felt_bpm": felt_bpm}


def _is_slow_calm_request(bpm: Optional[int], moods: Optional[List[str]],
                          raga: Optional[str]) -> bool:
    """Whether a request should sound slow and calm -- a raga, or a mood whose
    act mood is "calm"/"soft" -- and so is held to the onset-rate ceiling too,
    and scored/passed against `felt_bpm` with no octave leniency. A low bpm
    alone is not "calm": a 70 bpm lo-fi groove has busy hats and was retried
    three times against the ceiling (2026-10-20)."""
    if raga:
        return True
    acts = {MOODS.get(m) for m in (moods or [])}
    return bool(acts & {"calm", "soft"})


def _needs_tempo(style: Optional[str], bpm: Optional[int], moods: Optional[List[str]],
                 raga: Optional[str]) -> bool:
    """A request whose point is its tempo or calm: an explicit bpm, a raga or
    raga preset, or a slow/calm mood."""
    return bool(bpm) or bool(style and "raga" in style) or _is_slow_calm_request(bpm, moods, raga)


def _tempo_log_distance(requested_bpm: Optional[int], felt_bpm: Optional[float],
                        calm_required: bool) -> float:
    """|log(requested/felt)|, allowing an octave reading (half/double) for a
    pulsed, non-calm request -- a groove's bpm conditioning can legitimately
    land on either octave. A slow/calm request has no such ambiguity (there
    is no groove to misread an octave of), so it is scored against `felt_bpm`
    itself, with the half/double leniency dropped."""
    if not requested_bpm or not felt_bpm:
        return 0.0
    candidates = (felt_bpm,) if calm_required else (felt_bpm * 0.5, felt_bpm, felt_bpm * 2.0)
    return min(abs(np.log(requested_bpm / c)) for c in candidates if c > 0)


def _tempo_ok(requested_bpm: Optional[int], felt_bpm: Optional[float], calm_required: bool,
             tolerance: float = VERIFY_TEMPO_TOLERANCE) -> bool:
    if not requested_bpm:
        return True
    if felt_bpm is None:
        return False
    if calm_required:
        return abs(felt_bpm - requested_bpm) <= requested_bpm * tolerance
    return any(abs(felt_bpm - requested_bpm * f) <= requested_bpm * f * tolerance
              for f in (0.5, 1.0, 2.0))


def _onset_ok(onset_rate: Optional[float], calm_required: bool,
             max_rate: float = VERIFY_MAX_ONSET_RATE) -> bool:
    if not calm_required:
        return True
    return onset_rate is not None and VERIFY_MIN_ONSET_RATE <= onset_rate <= max_rate


def _verify_passes(requested_bpm: Optional[int], calm_required: bool,
                   measurement: Dict[str, Optional[float]]) -> bool:
    """A track passes when (a) a requested bpm is within 15% of the felt
    tempo -- or its half/double for a non-calm request -- AND (b) a slow/calm
    request's onset rate is at most `VERIFY_MAX_ONSET_RATE`."""
    return (_tempo_ok(requested_bpm, measurement.get("felt_bpm"), calm_required)
            and _onset_ok(measurement.get("onset_rate"), calm_required))


async def _generate_with_verification(
        attempt: Callable[[int], Awaitable[Optional[Path]]],
        measure: Callable[[Path], Dict[str, Optional[float]]],
        requested_bpm: Optional[int], calm_required: bool,
        max_attempts: int = VERIFY_MAX_ATTEMPTS,
) -> Tuple[Optional[Path], Dict[str, Optional[float]], int, bool]:
    """Runs `attempt(attempt_number)` up to `max_attempts` times, measuring
    each result with `measure`, and keeps the best-scoring one: a passing
    attempt always beats a failing one, and ties are broken by the closest
    tempo, then the fewest onsets. Stops as soon as one passes. Every attempt
    but the kept one has its file removed."""
    best: Optional[Tuple[Tuple[int, float, float], Path, Dict[str, Optional[float]], bool]] = None
    attempts_used = 0
    for attempt_number in range(1, max_attempts + 1):
        attempts_used = attempt_number
        path = await attempt(attempt_number)
        if path is None:
            continue
        measurement = measure(path)
        passed = _verify_passes(requested_bpm, calm_required, measurement)
        distance = _tempo_log_distance(requested_bpm, measurement.get("felt_bpm"), calm_required)
        onset = measurement.get("onset_rate")
        near_silent = onset is not None and onset < VERIFY_MIN_ONSET_RATE
        key = (0 if passed else (2 if near_silent else 1), distance,
               onset if onset is not None else 999.0)
        logger.info("Music verify attempt %d/%d: felt_bpm=%s onset_rate=%s passed=%s",
                   attempt_number, max_attempts, measurement.get("felt_bpm"), onset, passed)
        if best is None or key < best[0]:
            if best is not None:
                best[1].unlink(missing_ok=True)
            best = (key, path, measurement, passed)
        else:
            path.unlink(missing_ok=True)
        if passed:
            break
    if best is None:
        return None, {}, attempts_used, False
    _, path, measurement, passed = best
    return path, measurement, attempts_used, passed


# --- tempo lock --------------------------------------------------------------
# The verify loop above picks the take needing the least correction; this
# finishes the job by actually time-stretching it (pitch preserved) onto the
# requested bpm, since the model's own bpm conditioning on a slow raga barely
# moves the melody's real pace (see `measure_speed`).

def _clamp_ratio(ratio: float) -> Tuple[float, bool]:
    """`ratio` kept to one ffmpeg pass's safe range; whether it had to be."""
    clamped = min(max(ratio, TEMPO_STRETCH_MIN), TEMPO_STRETCH_MAX)
    return clamped, abs(clamped - ratio) > 1e-9


def _tempo_lock_ratio(requested_bpm: Optional[float],
                      felt_bpm: Optional[float]) -> Optional[float]:
    """The stretch ratio (>1 speeds up) that would move `felt_bpm` onto
    `requested_bpm`, or None when there is nothing to target, nothing
    measured, or the two already agree within `TEMPO_TOLERANCE`."""
    if not requested_bpm or not felt_bpm:
        return None
    ratio = requested_bpm / felt_bpm
    if abs(ratio - 1.0) <= TEMPO_TOLERANCE:
        return None
    return ratio


def _stretch_tempo(source: Path, destination: Path, ratio: float) -> bool:
    """Time-stretches `source` into `destination` by `ratio`, pitch
    preserved, with ffmpeg's rubberband filter; falls back to `atempo` (a
    lower-quality stretch, but always available) when rubberband errors --
    e.g. an ffmpeg build without librubberband."""
    rubberband = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                 "-af", f"rubberband=tempo={ratio:.6f}:pitch=1:transients=mixed:"
                        f"detector=soft:phase=laminar:window=long",
                 "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(destination)]
    try:
        subprocess.run(rubberband, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=180, creationflags=NO_WINDOW)
        if destination.exists() and destination.stat().st_size > 1000:
            return True
    except Exception as e:
        logger.warning("rubberband tempo lock failed for %s, trying atempo: %s", source, e)
    atempo = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
             "-af", f"atempo={ratio:.6f}", "-ar", "48000", "-ac", "2",
             "-c:a", "pcm_s16le", str(destination)]
    try:
        subprocess.run(atempo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=180, creationflags=NO_WINDOW)
    except Exception as e:
        logger.warning("atempo tempo lock failed for %s: %s", source, e)
        return False
    return destination.exists() and destination.stat().st_size > 1000


def _lock_tempo(path: Path, requested_bpm: Optional[float], free_rhythm: bool,
                measurement: Dict[str, Optional[float]]) -> Tuple[Path, Dict[str, Any]]:
    """Stretches `path` onto `requested_bpm` when `measurement`'s felt tempo
    (already computed by `measure_speed`) is off by more than
    `TEMPO_TOLERANCE`, then re-measures. Returns the file to actually use --
    `path` itself when no stretch was needed or the stretch failed -- and the
    manifest fields the lock produced (`felt_bpm_before`, `tempo_ratio`,
    `tempo_clamped`, and the post-lock `measured_bpm`/`onset_rate`)."""
    felt_before = measurement.get("felt_bpm")
    info: Dict[str, Any] = {
        "felt_bpm_before": felt_before, "tempo_ratio": 1.0, "tempo_clamped": False,
        "measured_bpm": felt_before, "onset_rate": measurement.get("onset_rate"),
    }
    ratio = _tempo_lock_ratio(requested_bpm, felt_before)
    if ratio is None:
        return path, info
    clamped_ratio, clamped = _clamp_ratio(ratio)
    stretched = path.with_name(path.stem + "_stretch" + path.suffix)
    if not _stretch_tempo(path, stretched, clamped_ratio):
        return path, info
    after = measure_speed(stretched, requested_bpm, free_rhythm)
    info.update({"tempo_ratio": clamped_ratio, "tempo_clamped": clamped,
                "measured_bpm": after.get("felt_bpm"), "onset_rate": after.get("onset_rate")})
    return stretched, info


def _lock_and_trim(path: Path, requested_bpm: Optional[float], free_rhythm: bool,
                   measurement: Dict[str, Optional[float]], seconds: float,
                   generated_seconds: float) -> Tuple[Path, Dict[str, Any]]:
    """Stretch `path` (the kept take), trim it to `seconds` and master it,
    measuring the FINISHED audio and correcting the stretch until it matches
    `requested_bpm`, at most `TEMPO_LOCK_ROUNDS` times.

    Why the finished audio: a raga's density changes as the alap develops
    (a Marwa take averaging 109 delivered an opening at 106 after a "90"
    stretch), and mastering changes what the note counter hears. Each round
    re-stretches the ORIGINAL take by the combined ratio (never a stretch of a
    stretch). The round closest to the target is kept -- not the last -- and
    the loop stops when a correction moves the reading the wrong way (a
    heavily stretched line can read faster, not slower).

    Returns the mastered file and the manifest fields (`felt_bpm_before`,
    `tempo_ratio`, `tempo_clamped`, post-lock `measured_bpm`/`onset_rate`,
    `lock_rounds`, `mastered`). Every other intermediate file is removed."""
    felt_before = measurement.get("felt_bpm")
    info: Dict[str, Any] = {
        "felt_bpm_before": felt_before, "tempo_ratio": 1.0, "tempo_clamped": False,
        "measured_bpm": felt_before, "onset_rate": measurement.get("onset_rate"),
        "lock_rounds": 0, "mastered": False,
    }
    needs_trim = generated_seconds > seconds
    initial = _tempo_lock_ratio(requested_bpm, felt_before)
    ratio, clamped = _clamp_ratio(initial) if initial is not None else (1.0, False)
    temps: List[Path] = []
    best: Optional[Tuple[float, Path, Dict[str, Any]]] = None
    previous_distance: Optional[float] = None
    for round_number in range(1, TEMPO_LOCK_ROUNDS + 1):
        stretched = path
        if abs(ratio - 1.0) > 1e-3:
            candidate = path.with_name(f"{path.stem}_stretch{round_number}{path.suffix}")
            if _stretch_tempo(path, candidate, ratio):
                stretched = candidate
                temps.append(candidate)
            else:
                ratio = 1.0
        cut = stretched
        if needs_trim:
            trimmed = path.with_name(f"{path.stem}_trim{round_number}{path.suffix}")
            if _trim(stretched, trimmed, seconds):
                cut = trimmed
                temps.append(trimmed)
        finished = path.with_name(f"{path.stem}_master{round_number}{path.suffix}")
        mastered = _master(cut, finished)
        if mastered:
            temps.append(finished)
        else:
            finished = cut
        after = measure_speed(finished, requested_bpm, free_rhythm)
        felt = after.get("felt_bpm")
        distance = abs(math.log(felt / requested_bpm)) if (requested_bpm and felt) else 0.0
        logger.info("Tempo lock round %d: ratio %.3f -> finished felt_bpm %s (asked %s)",
                    round_number, ratio, felt, requested_bpm)
        round_info = {"tempo_ratio": ratio, "tempo_clamped": clamped, "lock_rounds": round_number,
                      "measured_bpm": felt, "onset_rate": after.get("onset_rate"),
                      "mastered": mastered}
        if best is None or distance < best[0]:
            best = (distance, finished, round_info)
        if not requested_bpm or not felt:
            break
        if previous_distance is not None and distance >= previous_distance:
            break   # the last correction did not help: keep the best round
        previous_distance = distance
        correction = _tempo_lock_ratio(requested_bpm, felt)
        if correction is None:
            break
        new_ratio, clamped = _clamp_ratio(ratio * correction)
        if abs(new_ratio - ratio) < 0.01:   # pinned at the clamp
            break
        ratio = new_ratio
    assert best is not None
    _, final, round_info = best
    info.update(round_info)
    for temp in temps:
        if temp != final:
            temp.unlink(missing_ok=True)
    return final, info


def _trim(source: Path, destination: Path, seconds: float) -> bool:
    """The first `seconds` of `source` -- a bpm-locked take was generated
    longer than asked (see `generate_track`) so the stretch has runway; this
    cuts it back to the length the user actually requested."""
    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
              "-t", f"{max(seconds, 0.1):.3f}", "-ar", "48000", "-ac", "2",
              "-c:a", "pcm_s16le", str(destination)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=120, creationflags=NO_WINDOW)
    except Exception as e:
        logger.warning("Could not trim %s to %.1fs: %s", source, seconds, e)
        return False
    return destination.exists() and destination.stat().st_size > 1000


async def generate_track(style: Optional[str] = None, seconds: float = DEFAULT_SECONDS,
                         seed: Optional[int] = None, extra_tags: str = "",
                         genre: Optional[str] = None, engine: str = DEFAULT_ENGINE,
                         music_genres: Optional[List[str]] = None,
                         instruments: Optional[List[str]] = None,
                         moods: Optional[List[str]] = None,
                         bpm: Optional[int] = None, raga: Optional[str] = None,
                         on_stage: Optional[StageFn] = None,
                         stage_prefix: str = "",
                         extra_meta: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """One variation into the library, from a style preset, a raga, the
    builder's picks, or any mix. No GPU hand-over here -- `generate_variations`
    (or the caller) owns the phase.

    The result is verified against what was asked (tempo, and onset rate for
    a slow/calm request) and regenerated with a new seed up to
    `VERIFY_MAX_ATTEMPTS` times, keeping the best-scoring attempt. The default
    engine (ACE-Step 1.5) falls back to v1 automatically when its model files
    are not installed yet, recorded on the manifest entry as `engine_fallback`."""
    if style and style not in MUSIC_STYLES:
        raise ValueError(f"Unknown music style {style!r}")
    if raga and raga not in RAGAS:
        raise ValueError(f"Unknown raga {raga!r}")
    engine, fallback_note = resolve_engine(engine, _needs_tempo(style, bpm, moods, raga))
    eng = ENGINES.get(engine)
    if eng is None:
        raise ValueError(f"Unknown music engine {engine!r}")
    if fallback_note:
        logger.info(fallback_note)
    music_genres, instruments, moods = list(music_genres or []), list(instruments or []), list(moods or [])
    if not (style or music_genres or instruments or moods or raga or (extra_tags or "").strip()):
        raise ValueError("Pick a style, a raga, or at least one genre, instrument or mood.")
    workflow = workflows.resolve(eng["role"])
    if workflow is None:
        logger.info("No %s workflow configured; skipping", eng["label"])
        return None
    seconds = max(10.0, min(eng["max_seconds"], float(seconds)))
    base_seed = int(seed if seed is not None else random.randint(1, 2**31 - 1))
    raga_spec = RAGAS.get(raga) if raga else None
    effective_bpm = bpm or (raga_spec["bpm"] if raga_spec else None)
    # A bpm request gets extra runway: the tempo lock stretches to the asked
    # tempo, and a slowed take needs the generated audio to run longer so the
    # stretched-then-trimmed result still covers `seconds`.
    generated_seconds = min(eng["max_seconds"], seconds * 1.5) if effective_bpm else seconds
    prompt = compose_prompt(style, music_genres, instruments, moods, effective_bpm, extra_tags,
                            raga=raga, engine=engine)
    tags = prompt["tags"]
    negative = eng["negative"]
    if negative and prompt["dropped"]:
        negative = ", ".join([*prompt["dropped"], negative])
    name = style or raga or "custom"
    meta = _generation_meta(style, music_genres, instruments, moods, genre)
    folder = MUSIC_DIR / meta["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    calm_required = _is_slow_calm_request(effective_bpm, moods, raga)
    # How the speed is read (measure_speed): a raga is a free-rhythm line, read
    # by how often notes land; anything else is read by its beat when it has one.
    free_rhythm = bool(raga) or bool(style and "raga" in style)
    rng = random.Random(base_seed)
    clock = _StageClock(stage_prefix, on_stage)
    clock("load")

    async def _attempt(attempt_number: int) -> Optional[Path]:
        attempt_seed = base_seed if attempt_number == 1 else rng.randint(1, 2**31 - 1)
        extra: Dict[str, Any] = {}
        if effective_bpm:
            extra["bpm"] = int(max(10, min(300, effective_bpm)))
        if raga_spec:
            extra["keyscale"] = raga_spec["keyscale"]
        graph = workflow.build(positive=tags, negative=negative, seed=attempt_seed,
                               length=int(round(generated_seconds)),
                               prefix=f"buzzedit_audio/music_{name}", extra=extra or None)
        from comfyui_bridge import queue_manager
        # Generous: ~2 min of music is a few minutes of sampling on a 16 GB card.
        outputs = await queue_manager.submit_and_wait(
            graph, timeout=int(300 + generated_seconds * 6), on_event=_comfy_listener(graph, clock))
        audio = next((Path(o) for o in outputs or [] if Path(o).suffix.lower() in AUDIO_EXTS), None)
        if audio is None or not audio.exists():
            logger.warning("Music generation attempt %d for %s returned no audio", attempt_number, name)
            return None
        clock("master")
        temp_dest = folder / f"_verify_{name}_{engine}_{attempt_seed}.wav"
        try:
            shutil.copy2(audio, temp_dest)
        except Exception as e:
            logger.warning("Could not stage attempt %d for %s: %s", attempt_number, name, e)
            return None
        return temp_dest

    def _measure(path: Path) -> Dict[str, Optional[float]]:
        return measure_speed(path, effective_bpm, free_rhythm)

    winning_path, measurement, attempts_used, _loop_passed = await _generate_with_verification(
        _attempt, _measure, effective_bpm, calm_required)
    if winning_path is None:
        return None

    # Tempo lock: stretch the winning take onto `effective_bpm` (pitch
    # preserved) before mastering, so loudnorm/fades run on the final audio.
    # Stretch + trim, measured on exactly the audio that is delivered; the
    # existing fade-out then runs on the final-length audio.
    source_for_master, lock_info = _lock_and_trim(winning_path, effective_bpm, free_rhythm,
                                                  measurement, seconds, generated_seconds)
    locked_path, trimmed_path = source_for_master, None

    destination = folder / f"{name}_{engine}_{base_seed}.wav"
    if destination.exists():
        destination.unlink()
    if lock_info.get("mastered") and source_for_master != winning_path:
        shutil.copy2(source_for_master, destination)
    elif not _master(source_for_master, destination):
        shutil.copy2(source_for_master, destination)
    for temp in (winning_path, locked_path, trimmed_path):
        if temp is not None and temp != destination:
            temp.unlink(missing_ok=True)

    final_measured = lock_info.get("measured_bpm")
    final_passed = (final_measured is not None
                    and abs(final_measured - effective_bpm) <= effective_bpm * FINAL_TEMPO_TOLERANCE
                    ) if effective_bpm else True
    _record(folder, destination.name, {
        "style": style or "", "label": meta["label"], "mood": meta["moods"],
        "mood_tags": moods, "instruments": instruments, "music_genres": music_genres,
        "bpm": effective_bpm, "genre": meta["genres"], "tags": tags, "seed": base_seed,
        "seconds": seconds, "engine": engine, "license": eng["license"], "timings": clock.finish(),
        "source": f"{eng['model']} ({eng['model_license']}), generated locally",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "raga": raga, "requested_bpm": effective_bpm,
        "felt_bpm_before": lock_info.get("felt_bpm_before"),
        "tempo_ratio": lock_info.get("tempo_ratio"), "tempo_clamped": lock_info.get("tempo_clamped"),
        "measured_bpm": final_measured, "onset_rate": lock_info.get("onset_rate"),
        "attempts": attempts_used, "passed": final_passed, "generated_seconds": generated_seconds,
        **({"engine_fallback": fallback_note} if fallback_note else {}),
        **(extra_meta or {}),
    })
    logger.info("Generated music %s (%s via %s, seed %s, %.0fs, attempts=%d, passed=%s)",
                destination.name, name, engine, base_seed, seconds, attempts_used, final_passed)
    return destination


async def generate_variations(style: Optional[str] = None, count: int = 2,
                              seconds: float = DEFAULT_SECONDS,
                              extra_tags: str = "", genre: Optional[str] = None,
                              seed: Optional[int] = None, offload_after: bool = True,
                              engine: str = DEFAULT_ENGINE,
                              music_genres: Optional[List[str]] = None,
                              instruments: Optional[List[str]] = None,
                              moods: Optional[List[str]] = None,
                              bpm: Optional[int] = None, raga: Optional[str] = None,
                              on_stage: Optional[StageFn] = None) -> Dict[str, Any]:
    """`count` variations of one style/raga (or builder recipe), on a card
    handed over to the chosen engine, with every model offloaded afterwards."""
    from runtime import gpu_handover
    engine, fallback_note = resolve_engine(engine, _needs_tempo(style, bpm, moods, raga))
    eng = ENGINES.get(engine)
    if eng is None:
        raise ValueError(f"Unknown music engine {engine!r}")
    count = max(1, min(MAX_VARIATIONS, int(count)))
    name = style or raga or "custom"
    made: List[str] = []
    errors: List[str] = []
    report: StageFn = on_stage or (lambda key, fraction=None: None)
    report("gpu", None)
    ready = await gpu_handover.prepare_for_phase(
        "music", need_vram_mb=eng["vram_mb"], need_ram_mb=MUSIC_PHASE_MIN_RAM_MB)
    try:
        if not ready.get("ready"):
            # Still try: the models degrade to slow, not broken, when short.
            logger.warning("Music phase starting without a full hand-over: %s", ready.get("reason"))
        rng = random.Random(seed)
        for index in range(count):
            try:
                path = await generate_track(style, seconds, seed=rng.randint(1, 2**31 - 1),
                                            extra_tags=extra_tags, genre=genre, engine=engine,
                                            music_genres=music_genres, instruments=instruments,
                                            moods=moods, bpm=bpm, raga=raga, on_stage=on_stage,
                                            stage_prefix=f"v{index + 1}:")
                if path is not None:
                    made.append(str(path.relative_to(MUSIC_DIR)).replace("\\", "/"))
            except Exception as e:
                errors.append(str(e)[:200])
                logger.warning("Music variation for %s failed: %s", name, e)
    finally:
        if offload_after:
            report("offload", None)
            await gpu_handover.offload_all(f"music generation ({name})")
    result: Dict[str, Any] = {"style": name, "engine": engine, "made": made, "errors": errors,
                              "handover": ready}
    if fallback_note:
        result["notes"] = [fallback_note]
    return result


def delete_track(relative: str, library: Path = MUSIC_DIR) -> bool:
    """Remove one library file and its manifest entry. Only inside the library."""
    target = (library / relative).resolve()
    root = library.resolve()
    if root not in target.parents or not target.is_file() or target.suffix.lower() not in AUDIO_EXTS:
        return False
    manifest = _manifest(target.parent)
    target.unlink()
    if manifest.pop(target.name, None) is not None:
        (target.parent / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Deleted music track %s", relative)
    return True


UPLOAD_FOLDER = "uploads"


def import_track(source: Path, original_name: str, license_name: str, label: str = "",
                 library: Path = MUSIC_DIR) -> Dict[str, Any]:
    """Add a creator's own track to `<library>/uploads/`, loudness-normalised
    to -16 LUFS like a generated bed (so a level picked for one track means
    the same for another) and recorded with its licence. Raises ValueError on
    a licence the guard would not clear or a file ffmpeg cannot read."""
    license_name = (license_name or "").strip().lower()
    if license_name not in ALLOWED_LICENSES:
        raise ValueError(f"Licence {license_name!r} is not one the music guard accepts "
                         f"({', '.join(sorted(ALLOWED_LICENSES))}).")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_name).stem).strip("_")[:60] or "track"
    folder = library / UPLOAD_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{stem}.flac"
    n = 2
    while destination.exists():
        destination = folder / f"{stem}_{n}.flac"
        n += 1
    command = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
               "-vn", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2",
               "-c:a", "flac", str(destination)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=600, creationflags=NO_WINDOW)
    except subprocess.CalledProcessError as e:
        destination.unlink(missing_ok=True)
        detail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise ValueError(f"Could not read {original_name}: {detail[-1] if detail else e}")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    seconds = None
    try:
        from utils.ffmpeg_utils import get_video_duration
        seconds = round(float(get_video_duration(str(destination)) or 0.0), 1) or None
    except Exception:
        pass
    _record(folder, destination.name, {
        "label": (label or "").strip()[:80] or Path(original_name).stem,
        "license": license_name, "source": "upload", "original_name": original_name,
        "seconds": seconds, "created": _dt.datetime.now().isoformat(timespec="seconds"),
    })
    logger.info("Imported music track %s (%s)", destination.name, license_name)
    return {"file": f"{UPLOAD_FOLDER}/{destination.name}", "seconds": seconds}



# --- music cues ---------------------------------------------------------------
# One grammar for every way a video asks for music: a `[music: ...]` directive
# in the script (the Studio's agents write these), the project's music brief,
# and the Music page. Either free text -- "sad solo violin, 70 bpm" -- matched
# against the builder's vocabulary, or explicit fields --
# "style=sad_violin; instruments=violin; mood=sad; bpm=70; engine=ace_step".
# "off" (or none/stop/silence) means no music from that point.

_CUE_OFF = {"off", "none", "stop", "silence", "silent", "no music", "mute", "nothing"}
_CUE_FILLER = {"music", "bed", "background", "track", "score", "with", "and", "a", "an", "the",
               "some", "of", "in", "on", "for", "style", "vibe", "feel", "type", "bgm", "tune"}
_CUE_KEYS = {
    "style": "style", "preset": "style",
    "genre": "music_genres", "genres": "music_genres",
    "instrument": "instruments", "instruments": "instruments",
    "mood": "moods", "moods": "moods",
    "bpm": "bpm", "tempo": "bpm",
    "tags": "extra_tags", "extra": "extra_tags",
    "engine": "engine", "raga": "raga",
}


def _empty_recipe() -> Dict[str, Any]:
    return {"off": False, "style": None, "music_genres": [], "instruments": [], "moods": [],
            "bpm": None, "extra_tags": "", "engine": None, "raga": None}


def _engine_from(text: str) -> Optional[str]:
    t = text.lower()
    if "stable" in t:
        return "stable_audio"
    if "ace" in t:
        return "ace_step"
    return None


def _style_from(text: str) -> Optional[str]:
    t = text.strip().lower().replace("-", " ")
    for name, spec in MUSIC_STYLES.items():
        label = re.sub(r"\(.*?\)", "", spec["label"].lower()).strip()
        if t in (name, name.replace("_", " "), label, spec["label"].lower()):
            return name
    return None


def _raga_from(text: str) -> Optional[str]:
    """A raga name (key or label) mentioned in `text`, or None."""
    for key, spec in RAGAS.items():
        for phrase in (key.replace("_", " "), spec["label"].lower()):
            if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text):
                return key
    return None


def parse_music_cue(text: str) -> Dict[str, Any]:
    """A cue's text -> {off, style, music_genres, instruments, moods, bpm,
    extra_tags, engine, raga}. Never raises: unknown words become extra tags.
    A raga name, or a bare "morning raga"/"evening raga", sets `raga` (when
    named) and picks the matching raga style preset."""
    recipe = _empty_recipe()
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    low = raw.lower().strip(" .")
    if not low or low in _CUE_OFF:
        recipe["off"] = True
        return recipe

    if "=" in low:
        for part in re.split(r"[;|]", raw):
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            field = _CUE_KEYS.get(key.strip().lower())
            value = value.strip()
            if not field or not value:
                continue
            if field == "style":
                recipe["style"] = _style_from(value) or (value if value in MUSIC_STYLES else None)
            elif field == "bpm":
                digits = re.findall(r"\d+", value)
                recipe["bpm"] = max(MIN_BPM, min(MAX_BPM, int(digits[0]))) if digits else None
            elif field == "engine":
                recipe["engine"] = value if value in ENGINES else _engine_from(value)
            elif field == "extra_tags":
                recipe["extra_tags"] = value
            elif field == "raga":
                recipe["raga"] = value.strip().lower() if value.strip().lower() in RAGAS else None
            else:
                recipe[field] = [v.strip().lower() for v in value.split(",") if v.strip()]
        return recipe

    work = f" {low} "
    engine = _engine_from(low) if re.search(r"stable audio|ace.?step", low) else None
    if engine:
        recipe["engine"] = engine
        work = re.sub(r"stable audio( open)?|ace.?step", " ", work)
    bpm = re.search(r"(\d{2,3})\s*bpm", work)
    if bpm:
        recipe["bpm"] = max(MIN_BPM, min(MAX_BPM, int(bpm.group(1))))
        work = work.replace(bpm.group(0), " ")
    # A named raga, or a bare "morning raga"/"evening raga", picks the raga
    # preset for that time of day (and whichever of flute/violin/sitar is
    # named nearby; flute when none is).
    raga = _raga_from(work)
    time_word = None
    if raga:
        recipe["raga"] = raga
        time_word = RAGAS[raga]["time"]
        for phrase in (RAGAS[raga]["label"].lower(), raga.replace("_", " ")):
            work = re.sub(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", " ", work)
    else:
        generic = re.search(r"\b(morning|evening)\s+raga\b", work)
        if generic:
            time_word = generic.group(1)
            work = work.replace(generic.group(0), " ")
    if time_word:
        if re.search(r"\bsitar\b", work):
            instrument_key = "sitar"
        elif re.search(r"\bviolin\b", work):
            instrument_key = "violin"
        else:
            instrument_key = "flute"
        candidate = f"{time_word}_raga_{instrument_key}"
        if candidate in MUSIC_STYLES:
            recipe["style"] = candidate
    plain = " " + re.sub(r"[^a-z0-9]+", " ", work).strip() + " "
    if not recipe["style"]:
        for name, spec in MUSIC_STYLES.items():
            label = re.sub(r"[^a-z0-9]+", " ", re.sub(r"\(.*?\)", "", spec["label"].lower())).strip()
            if f" {label} " in plain or f" {name.replace('_', ' ')} " in plain:
                recipe["style"] = name
                break
    vocab = ([(g, "music_genres") for g in MUSIC_GENRES]
             + [(i, "instruments") for items in INSTRUMENTS.values() for i in items]
             + [(m, "moods") for m in MOODS])
    for phrase, field in sorted(vocab, key=lambda v: -len(v[0])):
        pattern = rf"(?<![a-z]){re.escape(phrase)}(?![a-z])"
        if re.search(pattern, work):
            if phrase not in recipe[field]:
                recipe[field].append(phrase)
            work = re.sub(pattern, " ", work)
    if recipe["style"]:
        # The preset's own name is not an extra tag.
        spec = MUSIC_STYLES[recipe["style"]]
        for word in _words(spec["label"]) | set(recipe["style"].split("_")):
            work = re.sub(rf"(?<![a-z]){re.escape(word)}(?![a-z])", " ", work)
    extras = []
    for chunk in re.split(r"[,/;]|\band\b", work):
        words = [w for w in re.findall(r"[a-z][a-z'-]*", chunk) if w not in _CUE_FILLER and w != "solo"]
        if words:
            extras.append(" ".join(words))
    recipe["extra_tags"] = ", ".join(extras)
    return recipe


def cue_key(recipe: Dict[str, Any], engine: str) -> str:
    """A stable identity for a cue, so the same request reuses its track."""
    parts = [engine, recipe.get("style") or "", recipe.get("raga") or "",
             ",".join(sorted(recipe.get("music_genres") or [])),
             ",".join(sorted(recipe.get("instruments") or [])),
             ",".join(sorted(recipe.get("moods") or [])),
             str(recipe.get("bpm") or ""), (recipe.get("extra_tags") or "").strip().lower()]
    return "|".join(parts)


def cue_has_content(recipe: Dict[str, Any]) -> bool:
    return bool(recipe.get("style") or recipe.get("raga") or recipe.get("music_genres")
                or recipe.get("instruments") or recipe.get("moods")
                or (recipe.get("extra_tags") or "").strip())


def find_cue_track(recipe: Dict[str, Any], engine: str,
                   library: Path = MUSIC_DIR) -> Optional[Path]:
    """A licensed library track already made for exactly this cue, or None."""
    key = cue_key(recipe, engine)
    if not library.exists():
        return None
    for folder in [library] + sorted(p for p in library.iterdir() if p.is_dir()):
        manifest = _manifest(folder)
        for name, entry in manifest.items():
            if isinstance(entry, dict) and entry.get("cue_key") == key:
                path = folder / name
                if path.is_file() and license_ok(path):
                    return path
    return None


async def ensure_cue_track(recipe: Dict[str, Any], engine: str, seconds: float,
                           genre: Optional[str] = None, seed: Optional[int] = None,
                           cue_text: str = "") -> Optional[Path]:
    """The track for a cue: reused when one was made before, else generated
    now (the caller owns the GPU phase). None for an "off" or empty cue."""
    if recipe.get("off") or not cue_has_content(recipe):
        return None
    engine = engine if engine in ENGINES else DEFAULT_ENGINE
    engine, _ = resolve_engine(engine, _needs_tempo(recipe.get("style"), recipe.get("bpm"),
                                                    recipe.get("moods"), recipe.get("raga")))
    existing = find_cue_track(recipe, engine)
    if existing is not None:
        return existing
    return await generate_track(
        recipe.get("style"), seconds, seed=seed, extra_tags=recipe.get("extra_tags") or "",
        genre=genre, engine=engine, music_genres=recipe.get("music_genres"),
        instruments=recipe.get("instruments"), moods=recipe.get("moods"), bpm=recipe.get("bpm"),
        raga=recipe.get("raga"),
        extra_meta={"cue_key": cue_key(recipe, engine), "cue": cue_text[:200]})
