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
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import FFMPEG_BIN
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
        "tags": "indian classical, morning raga, bansuri flute, tanpura drone, soft tabla, "
                "peaceful, 75 bpm, serene, sunrise, instrumental",
        "genres": ["vlog", "devotional", "travel", "health_fitness", "general"],
        "moods": ["calm", "soft", "upbeat"]},
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


# Engines a bed can be made with. Each runs a ComfyUI workflow role:
# ACE-Step ("music") makes full-length beds; Stable Audio Open ("audio", the
# SFX workflow) caps at 47 s, so it serves loops, intros and stingers.
ENGINES: Dict[str, Dict[str, Any]] = {
    "ace_step": {
        "label": "ACE-Step", "model": MODEL_LABEL, "model_license": MODEL_LICENSE,
        "role": "music", "max_seconds": MAX_SECONDS, "license": GENERATED_LICENSE,
        "vram_mb": MUSIC_PHASE_MIN_VRAM_MB, "negative": "",
        "note": "Full-length beds up to 4 min, best for whole-video music."},
    "stable_audio": {
        "label": "Stable Audio Open", "model": "Stable Audio Open 1.0",
        "model_license": "Stability AI Community Licence",
        "role": "audio", "max_seconds": 47.0, "license": "stability-community",
        "vram_mb": 6000.0,
        "negative": "vocals, singing, speech, low quality, distorted, noise, clipping",
        "note": "Short loops and stingers up to 47 s, for intros, transitions and Shorts."},
}
DEFAULT_ENGINE = "ace_step"

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


def compose_prompt(style: Optional[str] = None, music_genres: Optional[List[str]] = None,
                   instruments: Optional[List[str]] = None, moods: Optional[List[str]] = None,
                   bpm: Optional[int] = None, extra_tags: str = "") -> Dict[str, Any]:
    """The prompt the model gets: {"tags", "dropped"}.

    Picks lead (the models weight early tags most): genres, then instruments
    (one pick becomes "solo <instrument>"), then moods, then the preset's tags.
    A preset tag naming an instrument the user did not pick is dropped once any
    instrument is picked, and likewise a preset mood word once a mood is
    picked -- so "Sad violin" + violin is violin, not violin over soft piano.
    Tempo replaces the preset's bpm. Always ends "instrumental".
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
    return {"tags": ", ".join(out), "dropped": dropped}


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


def _master(source: Path, destination: Path) -> bool:
    """Loudness-normalise a generated bed and fade its edges, once."""
    af = ("afade=t=in:st=0:d=1.5,areverse,afade=t=in:st=0:d=2.5,areverse,"
          "loudnorm=I=-16:TP=-1.5:LRA=11")
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


async def generate_track(style: Optional[str] = None, seconds: float = DEFAULT_SECONDS,
                         seed: Optional[int] = None, extra_tags: str = "",
                         genre: Optional[str] = None, engine: str = DEFAULT_ENGINE,
                         music_genres: Optional[List[str]] = None,
                         instruments: Optional[List[str]] = None,
                         moods: Optional[List[str]] = None,
                         bpm: Optional[int] = None, on_stage: Optional[StageFn] = None,
                         stage_prefix: str = "",
                         extra_meta: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """One variation into the library, from a style preset, the builder's
    picks, or both. No GPU hand-over here -- `generate_variations` (or the
    caller) owns the phase."""
    if style and style not in MUSIC_STYLES:
        raise ValueError(f"Unknown music style {style!r}")
    eng = ENGINES.get(engine)
    if eng is None:
        raise ValueError(f"Unknown music engine {engine!r}")
    music_genres, instruments, moods = list(music_genres or []), list(instruments or []), list(moods or [])
    if not (style or music_genres or instruments or moods or (extra_tags or "").strip()):
        raise ValueError("Pick a style, or at least one genre, instrument or mood.")
    workflow = workflows.resolve(eng["role"])
    if workflow is None:
        logger.info("No %s workflow configured; skipping", eng["label"])
        return None
    seconds = max(10.0, min(eng["max_seconds"], float(seconds)))
    seed = int(seed if seed is not None else random.randint(1, 2**31 - 1))
    prompt = compose_prompt(style, music_genres, instruments, moods, bpm, extra_tags)
    tags = prompt["tags"]
    negative = eng["negative"]
    if negative and prompt["dropped"]:
        negative = ", ".join([*prompt["dropped"], negative])
    name = style or "custom"
    graph = workflow.build(positive=tags, negative=negative, seed=seed,
                           length=int(round(seconds)), prefix=f"buzzedit_audio/music_{name}")
    clock = _StageClock(stage_prefix, on_stage)
    clock("load")
    from comfyui_bridge import queue_manager
    # Generous: ~2 min of music is a few minutes of sampling on a 16 GB card.
    outputs = await queue_manager.submit_and_wait(graph, timeout=int(300 + seconds * 6),
                                                  on_event=_comfy_listener(graph, clock))
    audio = next((Path(o) for o in outputs or [] if Path(o).suffix.lower() in AUDIO_EXTS), None)
    if audio is None or not audio.exists():
        logger.warning("Music generation for %s returned no audio", name)
        return None
    meta = _generation_meta(style, music_genres, instruments, moods, genre)
    folder = MUSIC_DIR / meta["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{name}_{engine}_{seed}.wav"
    clock("master")
    if not _master(audio, destination):
        import shutil
        shutil.copy2(audio, destination)
    _record(folder, destination.name, {
        "style": style or "", "label": meta["label"], "mood": meta["moods"],
        "mood_tags": moods, "instruments": instruments, "music_genres": music_genres,
        "bpm": bpm, "genre": meta["genres"], "tags": tags, "seed": seed, "seconds": seconds,
        "engine": engine, "license": eng["license"], "timings": clock.finish(),
        "source": f"{eng['model']} ({eng['model_license']}), generated locally",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        **(extra_meta or {}),
    })
    logger.info("Generated music %s (%s via %s, seed %s, %.0fs)",
                destination.name, name, engine, seed, seconds)
    return destination


async def generate_variations(style: Optional[str] = None, count: int = 2,
                              seconds: float = DEFAULT_SECONDS,
                              extra_tags: str = "", genre: Optional[str] = None,
                              seed: Optional[int] = None, offload_after: bool = True,
                              engine: str = DEFAULT_ENGINE,
                              music_genres: Optional[List[str]] = None,
                              instruments: Optional[List[str]] = None,
                              moods: Optional[List[str]] = None,
                              bpm: Optional[int] = None,
                              on_stage: Optional[StageFn] = None) -> Dict[str, Any]:
    """`count` variations of one style (or builder recipe), on a card handed
    over to the chosen engine, with every model offloaded afterwards."""
    from runtime import gpu_handover
    eng = ENGINES.get(engine)
    if eng is None:
        raise ValueError(f"Unknown music engine {engine!r}")
    count = max(1, min(MAX_VARIATIONS, int(count)))
    name = style or "custom"
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
                                            moods=moods, bpm=bpm, on_stage=on_stage,
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
    return {"style": name, "engine": engine, "made": made, "errors": errors, "handover": ready}


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
    "engine": "engine",
}


def _empty_recipe() -> Dict[str, Any]:
    return {"off": False, "style": None, "music_genres": [], "instruments": [], "moods": [],
            "bpm": None, "extra_tags": "", "engine": None}


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


def parse_music_cue(text: str) -> Dict[str, Any]:
    """A cue's text -> {off, style, music_genres, instruments, moods, bpm,
    extra_tags, engine}. Never raises: unknown words become extra tags."""
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
    plain = " " + re.sub(r"[^a-z0-9]+", " ", work).strip() + " "
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
    parts = [engine, recipe.get("style") or "",
             ",".join(sorted(recipe.get("music_genres") or [])),
             ",".join(sorted(recipe.get("instruments") or [])),
             ",".join(sorted(recipe.get("moods") or [])),
             str(recipe.get("bpm") or ""), (recipe.get("extra_tags") or "").strip().lower()]
    return "|".join(parts)


def cue_has_content(recipe: Dict[str, Any]) -> bool:
    return bool(recipe.get("style") or recipe.get("music_genres") or recipe.get("instruments")
                or recipe.get("moods") or (recipe.get("extra_tags") or "").strip())


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
    existing = find_cue_track(recipe, engine)
    if existing is not None:
        return existing
    return await generate_track(
        recipe.get("style"), seconds, seed=seed, extra_tags=recipe.get("extra_tags") or "",
        genre=genre, engine=engine, music_genres=recipe.get("music_genres"),
        instruments=recipe.get("instruments"), moods=recipe.get("moods"), bpm=recipe.get("bpm"),
        extra_meta={"cue_key": cue_key(recipe, engine), "cue": cue_text[:200]})
