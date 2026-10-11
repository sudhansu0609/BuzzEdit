"""Voice conversion in BuzzEdit (VOICE_CHANGER_PLAN.md, briefs S1 and S2a).

Five parts, all on top of the seed-vc worker (`audio/seedvc_env.py`, `WORKER`) and the process
lane (`audio/processes.py`):

* **Voices library** -- `data/voices/<name>.wav` (48 kHz mono, 3-60 s) + `data/voices/voices.json`
  `{name: {kind, consent, created, seconds, source}}`: `add_voice`, `list_voices`, `rename_voice`,
  `delete_voice`, `voice_path`.
* **Presets** -- `engines/seedvc/presets.json` (SkinWalker's five): each = a voice stem, engine
  overrides and an FX rack of AudioEffect dicts built from the audio catalogue's pitch_shifter /
  distortion / reverb / sub_octave (the ones that exist when asked; the formant intent has no
  catalogue id and is skipped).
* **The `voice_conversion` process effect** -- `process_def()` is its EffectDef, `process_wav()`
  its processor: the lane's 48 kHz mono WAV goes through the worker and comes back resampled to
  48 kHz at exactly the same length (`length_adjust` is always 1.0, so the cut holds).
  `convert_range()` = `processes.run_process(..., "voice_conversion", ...)`.
* **The voice map** -- the project setting `voice_map` `{speaker: {voice, preset?, match_pitch,
  pitch_shift, steps}}`; `apply_voice_map()` converts every span of a mapped speaker
  (`script_align.speaker_spans` on the project's script and kept words; the narrator and unlabelled
  paragraphs never), splices each with the lane's crossfades into ONE new version, sets
  `audio_override`, and reports `voice_conversions: [{speaker, voice, spans, seconds}]` (also kept
  in the project's `metadata`). Undo pops it like any process version. Applying is idempotent and
  replaces instead of stacking: the setting `voice_map_applied` records the last apply, so the
  same map and spans on its version skip, a changed map takes its place, and an emptied map takes
  it back.
* **The Voice panel's backend** (brief S2a, §2.7) -- the project setting `voice_regions`
  (programme seconds; `set_regions` checks them and trims overlaps like SkinWalker's
  `parse_manifest`); `apply_regions()` converts the pending (or listed) regions' source pieces
  in ONE version through the voice map's machinery with 15 ms seams, marks them applied/failed
  and puts each region's `fx` on the A1 rack windowed to it, tagged `voice_region:<id>`;
  `undo_regions()` takes all of that back. `convert_file()` converts a whole media-pool item
  (or the project source) into a new media-pool item under `audio/converted/`; `batch()` does
  it for several items, reporting each.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from asr import script_align
from audio import noise, seedvc_env
from audio.analysis import SAMPLE_RATE, AudioToolError, decode_wav, run_ffmpeg
from audio.noise import Conflict, NotFound
from timeline.program_time import a1_segments, program_to_source
from timeline.schema import AudioEffect, frame_to_time

logger = logging.getLogger("audio.voice_conversion")

Progress = Optional[Callable[[float, str], None]]

EFFECT = "voice_conversion"
CUSTOM = "custom"
VOICE_RATE = 48000
MIN_VOICE_S = 3.0
MAX_VOICE_S = 60.0
PRESETS_FILE = seedvc_env.ENGINE_DIR / "presets.json"
FALLBACK_ENGINE_DEFAULTS: Dict[str, Any] = {
    "diffusion_steps": 25, "length_adjust": 1.0, "inference_cfg_rate": 0.7,
    "f0_condition": False, "auto_f0_adjust": True, "pitch_shift": 0,
}
STEPS_RANGE = (10.0, 50.0)
PITCH_RANGE = (-12.0, 12.0)
FX_ORDER = ("pitch_shifter", "distortion", "reverb", "sub_octave")   # SkinWalker's order
VOICE_MAP_KEYS = ("voice", "preset", "match_pitch", "pitch_shift", "steps", "cfg_rate")
APPLIED_KEY = "voice_map_applied"      # project setting: what the last voice-map apply wrote
ON_TOP_WARNING = "applied on top of later processing"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,62}$")
_library_lock = threading.RLock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ voices

def voices_dir() -> Path:
    return seedvc_env.voices_dir()


def _index_path() -> Path:
    return voices_dir() / "voices.json"


def _load_index() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(_index_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("voices.json unreadable, treated as empty: %s", e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}


def _save_index(index: Dict[str, Dict[str, Any]]) -> None:
    path = _index_path()
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(part, path)


def _check_name(name: Any) -> str:
    clean = str(name or "").strip()
    if not _NAME_RE.match(clean) or clean.endswith(".") or clean.lower().endswith(".wav"):
        raise ValueError("A voice name is 1-63 letters, digits, spaces, '_', '-' or '.', starting "
                         f"with a letter or digit: {clean!r}")
    return clean


def _wav_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate) if info.samplerate else 0.0


def list_voices() -> List[Dict[str, Any]]:
    """Every reference clip: the recorded ones with their notes, and any `<name>.wav` put into
    the folder by hand (its notes empty)."""
    with _library_lock:
        folder = voices_dir()
        index = _load_index()
        voices: List[Dict[str, Any]] = []
        seen = set()
        for name, record in index.items():
            path = folder / f"{name}.wav"
            if not path.is_file():
                continue
            seen.add(name.casefold())
            voices.append({"name": name, "kind": record.get("kind"), "consent": record.get("consent"),
                           "created": record.get("created"), "seconds": record.get("seconds"),
                           "source": record.get("source"), "path": str(path)})
        for path in folder.glob("*.wav"):
            if path.stem.casefold() in seen:
                continue
            try:
                seconds: Optional[float] = round(_wav_seconds(path), 3)
            except Exception:
                seconds = None
            voices.append({"name": path.stem, "kind": None, "consent": None, "created": None,
                           "seconds": seconds, "source": None, "path": str(path)})
        return sorted(voices, key=lambda v: v["name"].casefold())


def voice_names() -> List[str]:
    return [v["name"] for v in list_voices()]


def _find_voice(name: Any) -> Optional[Dict[str, Any]]:
    wanted = str(name or "").strip().casefold()
    return next((v for v in list_voices() if v["name"].casefold() == wanted), None) if wanted else None


def voice_path(name: Any) -> Path:
    """The reference clip of a library voice (names match case-insensitively)."""
    found = _find_voice(name)
    if found is None:
        raise NotFound(f"Voice {str(name or '').strip()!r} is not in the voices library "
                       "(add or record it first)")
    return Path(found["path"])


def add_voice(name: Any, src_path: Any, kind: Any = "custom", consent: Any = "",
              source: Optional[str] = None) -> Dict[str, Any]:
    """Add a reference clip: any audio ffmpeg reads becomes `<name>.wav`, 48 kHz mono. Refused
    when shorter than 3 s or longer than 60 s, or when the name is taken."""
    clean = _check_name(name)
    src = Path(str(src_path))
    if not src.is_file():
        raise NotFound(f"No file {src}")
    with _library_lock:
        if _find_voice(clean) is not None:
            raise Conflict(f"A voice named {clean!r} exists; rename or delete it first")
        folder = voices_dir()
        staged = folder / f".{uuid.uuid4().hex}.wav.part"
        try:
            try:
                run_ffmpeg(["-y", "-loglevel", "error", "-i", str(src), "-map", "0:a:0", "-vn",
                            "-ac", "1", "-ar", str(VOICE_RATE), "-c:a", "pcm_s16le", "-f", "wav", str(staged)])
                seconds = _wav_seconds(staged)
            except (AudioToolError, RuntimeError) as exc:
                raise ValueError(f"{src.name} has no audio ffmpeg can read: {exc}") from exc
            if seconds < MIN_VOICE_S:
                raise ValueError(f"The clip is {seconds:.1f} s; a voice needs at least {MIN_VOICE_S:g} s "
                                 "(5-20 s of dry speech works best)")
            if seconds > MAX_VOICE_S:
                raise ValueError(f"The clip is {seconds:.1f} s; a voice takes at most {MAX_VOICE_S:g} s "
                                 "(5-20 s of dry speech works best)")
            final = folder / f"{clean}.wav"
            os.replace(staged, final)
        finally:
            if staged.exists():
                staged.unlink()
        record = {"kind": str(kind or "custom").strip() or "custom", "consent": str(consent or "").strip(),
                  "created": _now(), "seconds": round(seconds, 3), "source": source or src.name}
        index = _load_index()
        index[clean] = record
        _save_index(index)
    return {"name": clean, **record, "path": str(final)}


def rename_voice(name: Any, new_name: Any) -> Dict[str, Any]:
    target = _check_name(new_name)
    with _library_lock:
        current = _find_voice(name)
        if current is None:
            raise NotFound(f"Voice {str(name or '').strip()!r} is not in the voices library")
        clash = _find_voice(target)
        if clash is not None and clash["name"] != current["name"]:
            raise Conflict(f"A voice named {clash['name']!r} exists")
        src = Path(current["path"])
        dst = src.with_name(f"{target}.wav")
        if src != dst:
            os.replace(src, dst)
        index = _load_index()
        record = index.pop(current["name"], None) or {
            "kind": current.get("kind"), "consent": current.get("consent"), "created": current.get("created"),
            "seconds": current.get("seconds"), "source": current.get("source")}
        index[target] = record
        _save_index(index)
    return {"name": target, **record, "path": str(dst)}


def delete_voice(name: Any) -> Dict[str, Any]:
    with _library_lock:
        current = _find_voice(name)
        if current is None:
            raise NotFound(f"Voice {str(name or '').strip()!r} is not in the voices library")
        Path(current["path"]).unlink(missing_ok=True)
        index = _load_index()
        index.pop(current["name"], None)
        _save_index(index)
    return {"deleted": current["name"]}


# ----------------------------------------------------------------- presets

def _presets_data() -> Dict[str, Any]:
    try:
        data = json.loads(Path(PRESETS_FILE).read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("seed-vc presets unreadable (%s): %s", PRESETS_FILE, e)
        return {}
    return data if isinstance(data, dict) else {}


def _preset_table(data: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    table = (data if data is not None else _presets_data()).get("presets") or {}
    return {str(k): v for k, v in table.items() if isinstance(v, dict)}


def engine_defaults(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = (data if data is not None else _presets_data()).get("engine_defaults")
    return {**FALLBACK_ENGINE_DEFAULTS, **(defaults if isinstance(defaults, dict) else {})}


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def fx_rack(preset_name: str, fx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A preset's creature-FX intent as AudioEffect dicts, only for intents that are non-zero and
    catalogue ids that exist now: pitch_semitones -> Pitch Shifter, drive -> Distortion
    (drive x 30 dB), reverb -> Studio Reverb (room = reverb, wet = reverb / 2), sub_octave ->
    Sub-Octave (mix). Values are clamped to the catalogue's ranges."""
    try:
        from audio.catalogue import EFFECTS
    except Exception as e:
        logger.debug("audio catalogue unavailable, no FX rack: %s", e)
        return []
    pitch, drive = _number(fx.get("pitch_semitones")), _number(fx.get("drive"))
    reverb, sub = _number(fx.get("reverb")), _number(fx.get("sub_octave"))
    wanted = {
        "pitch_shifter": (pitch, {"semitones": pitch}),
        "distortion": (drive, {"drive": drive * 30.0}),
        "reverb": (reverb, {"room_size": reverb * 100.0, "wet": reverb * 50.0}),
        "sub_octave": (sub, {"mix": sub * 100.0}),
    }
    slug = re.sub(r"[^a-z0-9]+", "_", str(preset_name).lower()).strip("_") or "preset"
    rack: List[Dict[str, Any]] = []
    for effect_id in FX_ORDER:
        intent, values = wanted[effect_id]
        definition = EFFECTS.get(effect_id)
        if not intent or definition is None:
            continue
        specs = {spec["key"]: spec for spec in definition.get("params", [])}
        params: Dict[str, Any] = {}
        for key, value in values.items():
            spec = specs.get(key)
            if spec is None:
                continue
            if spec.get("min") is not None:
                value = max(float(spec["min"]), value)
            if spec.get("max") is not None:
                value = min(float(spec["max"]), value)
            params[key] = round(value, 3)
        rack.append({"id": f"vc_{slug}_{effect_id}", "effect": effect_id, "params": params,
                     "enabled": True, "start_s": None, "end_s": None, "preset": None})
    return rack


def _preset_params(name: str, preset: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """A preset as values of the voice_conversion parameters (its voice comes in at run time)."""
    engine = {**defaults, **(preset.get("engine") if isinstance(preset.get("engine"), dict) else {})}
    return {"preset": name,
            "match_pitch": bool(engine.get("f0_condition")),
            "pitch_shift": _clamp(round(_number(engine.get("pitch_shift"))), PITCH_RANGE),
            "steps": _clamp(round(_number(engine.get("diffusion_steps")) or 25.0), STEPS_RANGE),
            "cfg_rate": _clamp(_number(engine.get("inference_cfg_rate")), (0.0, 1.0))}


def presets() -> Dict[str, Any]:
    """`{engine_defaults, presets: {name: {name, voice, voice_in_library, engine, fx, fx_rack,
    params}}}`: the voice stem, the engine settings it converts with, the FX intent and its rack,
    and the values it gives the voice_conversion parameters."""
    data = _presets_data()
    defaults = engine_defaults(data)
    library = {name.casefold() for name in voice_names()}
    out: Dict[str, Dict[str, Any]] = {}
    for name, preset in _preset_table(data).items():
        engine = {**defaults, **(preset.get("engine") if isinstance(preset.get("engine"), dict) else {})}
        engine["length_adjust"] = 1.0
        fx = dict(preset.get("fx") or {}) if isinstance(preset.get("fx"), dict) else {}
        voice = str(preset.get("voice") or "")
        out[name] = {"name": name, "voice": voice, "voice_in_library": voice.casefold() in library,
                     "engine": engine, "fx": fx, "fx_rack": fx_rack(name, fx),
                     "params": _preset_params(name, preset, defaults)}
    return {"engine_defaults": defaults, "presets": out}


# ------------------------------------------------------------- the process

def _param(key: str, label: str, type_: str, default: Any, min_: Optional[float] = None,
           max_: Optional[float] = None, step: Optional[float] = None, unit: str = "",
           choices: Optional[List[Any]] = None) -> Dict[str, Any]:
    """processes._param's shape (that module imports this one, so it is not imported here)."""
    param: Dict[str, Any] = {"key": key, "label": label, "type": type_, "min": min_, "max": max_,
                             "step": step, "default": default, "unit": unit, "live": False}
    if choices is not None:
        param["choices"] = choices
    return param


def process_def() -> Dict[str, Any]:
    """The EffectDef of `voice_conversion` (AUDIO_SUITE_PLAN.md §2.6, engine "process").

    `voice` "" = the preset's own voice; its choices follow the library (`refresh_choices`)."""
    data = _presets_data()
    defaults = engine_defaults(data)
    table = _preset_table(data)
    return {
        "id": EFFECT,
        "name": "Voice Conversion",
        "category": "Special",
        "engine": "process",
        "description": "Re-voices the speech in a reference voice from the voices library (seed-vc, "
                       "on the GPU), keeping the timing, intonation and pauses. Preset = a voice + "
                       "engine settings (its creature FX are rack effects); Match my pitch contour "
                       "follows the speaker's own melody; Pitch Shift in semitones; Quality Steps = "
                       "diffusion steps (more = slower, cleaner).",
        "params": [
            _param("voice", "Voice", "choice", "", choices=[""] + voice_names()),
            _param("preset", "Preset", "choice", CUSTOM, choices=list(table) + [CUSTOM]),
            _param("match_pitch", "Match my pitch contour", "bool", bool(defaults.get("f0_condition"))),
            _param("pitch_shift", "Pitch Shift", "number", 0.0, PITCH_RANGE[0], PITCH_RANGE[1], 1.0, "st"),
            _param("steps", "Quality Steps", "number",
                   _clamp(round(_number(defaults.get("diffusion_steps")) or 25.0), STEPS_RANGE),
                   STEPS_RANGE[0], STEPS_RANGE[1], 1.0, ""),
            _param("cfg_rate", "CFG Rate", "number",
                   _clamp(_number(defaults.get("inference_cfg_rate")), (0.0, 1.0)), 0.0, 1.0, 0.05, ""),
        ],
        "presets": {name: _preset_params(name, preset, defaults) for name, preset in table.items()},
        "windowable": True,
    }


def refresh_choices(definition: Dict[str, Any]) -> None:
    """Point the `voice` choices of a voice_conversion EffectDef at the library as it is now."""
    for spec in definition.get("params", []):
        if spec.get("key") == "voice":
            spec["choices"] = [""] + voice_names()


def engine_params(values: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """(voice name, seed-vc parameters) for resolved voice_conversion values. An empty voice is
    the preset's; length_adjust is always 1.0."""
    data = _presets_data()
    defaults = engine_defaults(data)
    preset_name = str(values.get("preset") or CUSTOM)
    preset = _preset_table(data).get(preset_name) if preset_name != CUSTOM else None
    voice = str(values.get("voice") or "").strip() or (str(preset.get("voice") or "").strip() if preset else "")
    if not voice:
        raise ValueError("Pick a voice from the library (or a preset that names one)")
    engine = preset.get("engine") if preset and isinstance(preset.get("engine"), dict) else {}
    params = {
        "diffusion_steps": int(round(_number(values.get("steps")) or defaults["diffusion_steps"])),
        "length_adjust": 1.0,
        "inference_cfg_rate": float(_number(values.get("cfg_rate"))),
        "f0_condition": bool(values.get("match_pitch")),
        "auto_f0_adjust": bool(engine.get("auto_f0_adjust", defaults.get("auto_f0_adjust", True))),
        "pitch_shift": int(round(_number(values.get("pitch_shift")))),
    }
    return voice, params


def _to_lane_rate(src: Path, dst: Path, frames: int) -> None:
    """The engine's WAV (22.05 or 44.1 kHz) as a 48 kHz mono float WAV of exactly `frames`."""
    data, rate = sf.read(str(src), dtype="float32", always_2d=True)
    mono = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
    if int(rate) != SAMPLE_RATE:
        from scipy.signal import resample_poly
        g = math.gcd(int(rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, int(rate) // g)
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size >= frames:
        mono = mono[:frames]
    else:
        mono = np.pad(mono, (0, frames - mono.size))
    sf.write(str(dst), mono, SAMPLE_RATE, subtype="FLOAT")


def process_wav(in_wav: Path, out_wav: Path, values: Dict[str, Any], progress: Progress = None,
                worker: Optional[seedvc_env.Worker] = None) -> Dict[str, Any]:
    """The lane's processor: `in_wav` (48 kHz mono) in the chosen voice into `out_wav` (48 kHz
    mono float, the same number of samples)."""
    voice, params = engine_params(values)
    ref = voice_path(voice)
    in_wav, out_wav = Path(in_wav), Path(out_wav)
    frames = sf.info(str(in_wav)).frames
    engine_out = out_wav.with_name(out_wav.stem + ".seedvc.wav")

    def report(fraction: float, stage: str) -> None:
        if progress is not None:
            progress(fraction, f"voice conversion: {stage}")

    try:
        answer = (worker or seedvc_env.WORKER).convert(in_wav, ref, engine_out, params, progress=report)
        _to_lane_rate(Path(answer.get("out") or engine_out), out_wav, frames)
    finally:
        if engine_out.exists():
            engine_out.unlink()
    return {"voice": voice, "params": params, "device": answer.get("device"),
            "seconds": answer.get("seconds")}


def convert_range(project_id: str, source_id: Optional[str], start_s: Optional[float],
                  end_s: Optional[float], params: Optional[Dict[str, Any]],
                  progress: Progress = None) -> Dict[str, Any]:
    """One range (or the whole source) in a voice, as a process-lane version."""
    from audio import processes
    return processes.run_process(project_id, source_id, EFFECT, params, start_s, end_s, progress)


# ----------------------------------------------------------- the voice map

def script_text(p_data: Dict[str, Any]) -> str:
    """The project's written script (where the auto-edit's planner reads it)."""
    script = p_data.get("script")
    text = script.get("text") if isinstance(script, dict) else None
    return text if isinstance(text, str) else ""


def kept_words(tl: Any) -> List[Dict[str, Any]]:
    """The words that survived the cut, in source seconds, in time order."""
    words = sorted((w for w in tl.words if w.enabled), key=lambda w: w.start_frame)
    return [{"text": w.text, "word_native": w.word_native,
             "start": frame_to_time(w.start_frame, tl.fps_num, tl.fps_den),
             "end": frame_to_time(w.end_frame, tl.fps_num, tl.fps_den)} for w in words]


def project_speaker_spans(project_id: str) -> Dict[str, Any]:
    """`{has_script, spans: [{speaker, start_s, end_s, text}], speakers}` in source seconds."""
    _pid, p_data, tl = noise.load_project(project_id)
    text = script_text(p_data)
    if not text.strip():
        return {"has_script": False, "spans": [], "speakers": []}
    spans = script_align.speaker_spans(script_align.parse_script(text), kept_words(tl))
    speakers: List[str] = []
    for span in spans:
        if span["speaker"] and span["speaker"] not in speakers:
            speakers.append(span["speaker"])
    return {"has_script": True, "spans": spans, "speakers": speakers}


def entry_params(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A voice-map entry as voice_conversion values: the preset's values, then the entry's own."""
    preset_name = str(entry.get("preset") or CUSTOM)
    data = _presets_data()
    preset = _preset_table(data).get(preset_name)
    values = _preset_params(preset_name, preset, engine_defaults(data)) if preset else {}
    values.update({key: entry[key] for key in VOICE_MAP_KEYS if key in entry and entry[key] is not None})
    values["preset"] = preset_name
    return values


def _resolved_entry(speaker: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """voice_conversion values for one voice-map entry; ValueError when it cannot run."""
    from audio import processes
    try:
        if str(entry.get("voice") or "").strip():
            voice_path(entry["voice"])                     # a clear message before the choice check
        values = processes.resolve_params(EFFECT, entry_params(entry))
        voice, _params = engine_params(values)
        voice_path(voice)
    except NotFound as exc:
        raise ValueError(f"voice_map[{speaker!r}]: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"voice_map[{speaker!r}]: {exc}") from exc
    return values


def get_voice_map(project_id: str) -> Dict[str, Any]:
    _pid, p_data, _tl = noise.load_project(project_id)
    voice_map = (p_data.get("settings") or {}).get("voice_map")
    return {"voice_map": dict(voice_map) if isinstance(voice_map, dict) else {}}


def set_voice_map(project_id: str, voice_map: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the project's voice map. Every entry must name a library voice (itself or through
    its preset); the narrator cannot be mapped."""
    if not isinstance(voice_map, dict):
        raise ValueError("voice_map must be an object {speaker: {voice, preset?, match_pitch, pitch_shift, steps}}")
    clean: Dict[str, Dict[str, Any]] = {}
    for speaker, entry in voice_map.items():
        name = str(speaker or "").strip()
        if not name:
            raise ValueError("A voice_map speaker needs a name")
        if name.casefold() in script_align.NARRATOR_LABELS:
            raise ValueError(f"{name!r} is the narrator, who is never converted")
        if not isinstance(entry, dict):
            raise ValueError(f"voice_map[{name!r}] must be an object")
        kept = {key: entry[key] for key in VOICE_MAP_KEYS if key in entry and entry[key] is not None}
        _resolved_entry(name, kept)
        clean[name] = kept
    _pid, p_data, _tl = noise.load_project(project_id)
    settings = dict(p_data.get("settings") or {})
    settings["voice_map"] = clean
    p_data["settings"] = settings
    noise.project_store.save_project(project_id, p_data)
    return {"voice_map": clean}


def preflight_voice_map(project_id: str) -> None:
    """What `apply_voice_map` needs before it is queued: a voice map whose voices exist (else
    ValueError) and a script (else Conflict). An empty map passes while an apply is recorded:
    applying it takes that apply back."""
    _pid, p_data, _tl = noise.load_project(project_id)
    voice_map = (p_data.get("settings") or {}).get("voice_map")
    if not isinstance(voice_map, dict) or not voice_map:
        if _applied_record(p_data) is not None:
            return
        raise Conflict("The project has no voice map (PUT /api/audio/{project}/voice_map)")
    if not script_text(p_data).strip():
        raise Conflict("The project has no script: speaker spans come from its speaker labels")
    for speaker, entry in voice_map.items():
        _resolved_entry(str(speaker), entry if isinstance(entry, dict) else {})


def _splice_into(out_wav: Path, segment_wav: Path, segment_start: int, a: int, b: int,
                 fade_s: float) -> None:
    """Samples [a, b) of `out_wav` taken from the converted segment (which starts at
    `segment_start`), with the process lane's linear crossfades of `fade_s` centred on a and b
    (none at the file's own ends). Only that stretch is read and written back."""
    with sf.SoundFile(str(out_wav), "r+") as out, sf.SoundFile(str(segment_wav)) as segment:
        rate, total = out.samplerate, out.frames
        half = max(1, int(round(fade_s * rate / 2.0)))
        low = a if a <= 0 else max(segment_start, a - half)
        high = b if b >= total else min(segment_start + segment.frames, b + half)
        if high <= low:
            return
        out.seek(low)
        dry = out.read(high - low, dtype="float32")
        segment.seek(low - segment_start)
        wet = segment.read(high - low, dtype="float32")
        if wet.size < dry.size:
            wet = np.pad(wet, (0, dry.size - wet.size))
        index = np.arange(low, low + dry.size, dtype=np.float64)
        gain = np.ones(dry.size, dtype=np.float64)
        if a > 0:
            gain = np.minimum(gain, np.clip((index - (a - half)) / (2.0 * half), 0.0, 1.0))
        if b < total:
            gain = np.minimum(gain, np.clip(((b + half) - index) / (2.0 * half), 0.0, 1.0))
        out.seek(low)
        out.write((dry * (1.0 - gain) + wet[:dry.size] * gain).astype(np.float32))


# ------------------------------------- the applied record (an idempotent apply)

def spans_signature(jobs: List[Dict[str, Any]]) -> List[List[Any]]:
    """The converted spans as [speaker, start_s, end_s]: speakers case-folded, times rounded to
    10 ms, in time order."""
    return [[str(job["speaker"]).strip().casefold(), round(float(job["start_s"]), 2),
             round(float(job["end_s"]), 2)]
            for job in sorted(jobs, key=lambda j: (j["start_s"], j["end_s"]))]


def voice_map_hash(voice_map: Dict[str, Any], jobs: List[Dict[str, Any]]) -> str:
    """A stable hash of the normalised voice map (speakers case-folded, only the entry keys an
    apply reads, none empty) plus the spans it converts (`spans_signature`)."""
    normalised: Dict[str, Dict[str, Any]] = {}
    for speaker, entry in voice_map.items():
        entry = entry if isinstance(entry, dict) else {}
        normalised[str(speaker).strip().casefold()] = {
            key: entry[key] for key in VOICE_MAP_KEYS if key in entry and entry[key] is not None}
    blob = json.dumps({"voice_map": normalised, "spans": spans_signature(jobs)}, sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _applied_record(p_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    record = (p_data.get("settings") or {}).get(APPLIED_KEY)
    return record if isinstance(record, dict) and record.get("version_path") else None


def applied_view(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The record without its span lists (`applied` on GET voice_map)."""
    if not record:
        return None
    return {key: value for key, value in record.items() if key not in ("spans_signature", "voice_conversions")}


def get_applied(project_id: str) -> Optional[Dict[str, Any]]:
    _pid, p_data, _tl = noise.load_project(project_id)
    return applied_view(_applied_record(p_data))


def _same(a: Any, b: Any) -> bool:
    return bool(a) and bool(b) and noise._same_path(str(a), str(b))


def _applied_state(record: Optional[Dict[str, Any]], source: Any) -> Optional[str]:
    """Where the recorded version sits on the source: "top" (it is the `audio_override`),
    "under" (later versions were made on top of it) or None (nothing recorded, or that version
    was undone since)."""
    if record is None:
        return None
    if _same(source.audio_override, record["version_path"]):
        return "top"
    if any(isinstance(entry, dict) and _same(entry.get("path"), record["version_path"])
           for entry in source.audio_history):
        return "under"
    return None


def _below(source: Any, path: str) -> List[Dict[str, Any]]:
    """The source's history once the version at `path` is undone (it leaves only when newest)."""
    history = list(source.audio_history)
    if history and _same(history[-1].get("path"), path):
        history.pop()
    return history


def _undo_top(path: str, popped: Dict[str, Any]) -> Callable[[Any], None]:
    """`processes.undo`'s pop for the recorded version: it leaves the history when it is the
    newest entry and the override goes back to the version before it (or the original); the file
    stays on disk."""
    def pop(src: Any) -> None:
        if src.audio_history and _same(src.audio_history[-1].get("path"), path):
            popped.update(src.audio_history.pop())
        src.audio_override = src.audio_history[-1]["path"] if src.audio_history else None
    return pop


def _source_state(sid: str, source: Any, tl: Any) -> Dict[str, Any]:
    """`processes._save_source`'s answer, for a run that saves no version."""
    return {"source_id": sid, "audio_override": source.audio_override,
            "history_len": len(source.audio_history), "revision": tl.revision}


def _history_entry(source: Any, path: str) -> Optional[Dict[str, Any]]:
    return next((dict(entry) for entry in reversed(source.audio_history)
                 if isinstance(entry, dict) and _same(entry.get("path"), path)), None)


def _store_applied(project_id: str, record: Optional[Dict[str, Any]]) -> None:
    """Set the project setting `voice_map_applied` (None clears it)."""
    p_data = noise.project_store.get_project(project_id) or {}
    settings = dict(p_data.get("settings") or {})
    if record is None:
        settings.pop(APPLIED_KEY, None)
    else:
        settings[APPLIED_KEY] = record
    p_data["settings"] = settings
    noise.project_store.save_project(project_id, p_data)


def _take_back(project_id: str, sid: str, source: Any, tl: Any, record: Optional[Dict[str, Any]],
               where: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Inside the source lock: the recorded version undone when it is on top (`_undo_top`), then
    the record cleared. Returns the source state and the undone version's path."""
    from audio import processes
    popped: Dict[str, Any] = {}
    if where == "top" and record is not None:
        state = processes._save_source(project_id, sid, _undo_top(record["version_path"], popped))
    else:
        state = _source_state(sid, source, tl)
    _store_applied(project_id, None)
    return state, popped.get("path")


def _remove_applied(project_id: str, report: Callable[[float, str], None]) -> Dict[str, Any]:
    """An emptied voice map with a recorded apply: its version is undone when it is what the
    source sounds like now (forgotten when it was undone already) and the record is cleared.
    Conflict, nothing changed, when later processing sits on top of it (undo that first)."""
    from audio import processes
    _pid, p_data, tl = noise.load_project(project_id)
    sid, _source = noise.resolve_source(tl, p_data, None)
    with processes._source_lock(project_id, sid):
        _pid, p_data, tl = noise.load_project(project_id)
        sid, source = noise.resolve_source(tl, p_data, sid)
        record = _applied_record(p_data)
        where = _applied_state(record, source)
        if where == "under":
            raise Conflict("Nothing removed: later processing sits on top of the voice map's version "
                           "(undo that first)")
        state, undone = _take_back(project_id, sid, source, tl, record, where)
    report(1.0, "removed")
    return {**state, "voice_conversions": [], "failed_spans": [], "version": None,
            "removed": True, "undone": undone}


def apply_voice_map(project_id: str, progress: Progress = None,
                    worker: Optional[seedvc_env.Worker] = None) -> Dict[str, Any]:
    """Convert every span of every mapped speaker into ONE new version of the programme source.

    Spans come from the project's script and kept words; each is converted with 0.5 s of
    context either side and spliced back with the lane's crossfades; audio outside the spans is
    left sample for sample. A span that fails keeps the original audio and is listed in
    `failed_spans`; when none converts nothing is written.

    Idempotent, replace-not-stack: a successful apply records the project setting
    `voice_map_applied` {map_hash, version_path, spans_signature, applied_at, voice_conversions}.
    While that version is what the source sounds like (`audio_override`), the same map and spans
    skip (`skipped: true, reason: "already applied"`), and a changed map or changed spans convert
    the audio beneath it and take its place (`replaced: true`; `processes.undo` semantics, its file
    stays). When later processing sits on top of it nothing is undone and the new version goes on
    top (`warning`). An emptied map takes the recorded version back (`removed: true`)."""
    from audio import processes

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, fraction)), message)
            except Exception:
                logger.debug("progress callback failed", exc_info=True)

    _pid, p_data, tl = noise.load_project(project_id)
    voice_map = (p_data.get("settings") or {}).get("voice_map")
    if not isinstance(voice_map, dict) or not voice_map:
        if _applied_record(p_data) is not None:
            return _remove_applied(project_id, report)
        raise Conflict("The project has no voice map (PUT /api/audio/{project}/voice_map)")
    text = script_text(p_data)
    if not text.strip():
        raise Conflict("The project has no script: speaker spans come from its speaker labels")
    spans = script_align.speaker_spans(script_align.parse_script(text), kept_words(tl))
    lookup = {str(key).strip().casefold(): str(key).strip() for key in voice_map}
    resolved: Dict[str, Dict[str, Any]] = {}
    jobs: List[Dict[str, Any]] = []
    for span in spans:
        speaker = span.get("speaker")
        if not speaker:
            continue                                  # the narrator is never converted
        key = lookup.get(str(speaker).casefold())
        if key is None:
            continue
        if key not in resolved:
            values = _resolved_entry(key, voice_map[key] if isinstance(voice_map[key], dict) else {})
            resolved[key] = {"values": values, "voice": engine_params(values)[0]}
        if span["end_s"] - span["start_s"] <= 0.0:
            continue
        jobs.append({"speaker": speaker, "key": key, "voice": resolved[key]["voice"],
                     "values": resolved[key]["values"], "start_s": float(span["start_s"]),
                     "end_s": float(span["end_s"])})
    no_spans = {"voice_conversions": [], "failed_spans": [], "version": None,
                "message": "No span of a mapped speaker was found in the script's kept words"}
    if not jobs and _applied_record(p_data) is None:
        return no_spans
    jobs.sort(key=lambda j: (j["voice"].casefold(), j["start_s"]))     # grouped per voice
    new_hash = voice_map_hash(voice_map, jobs)

    sid, _source = noise.resolve_source(tl, p_data, None)
    with processes._source_lock(project_id, sid):
        _pid, p_data, tl = noise.load_project(project_id)
        sid, source = noise.resolve_source(tl, p_data, sid)
        record = _applied_record(p_data)
        where = _applied_state(record, source)
        if where == "top" and record.get("map_hash") == new_hash:
            report(1.0, "already applied")
            return {**_source_state(sid, source, tl),
                    "voice_conversions": list(record.get("voice_conversions") or []), "failed_spans": [],
                    "version": _history_entry(source, record["version_path"]),
                    "skipped": True, "reason": "already applied"}
        if not jobs:
            if where != "top":
                return no_spans
            state, undone = _take_back(project_id, sid, source, tl, record, where)
            report(1.0, "removed")
            return {**state, **no_spans, "removed": True, "undone": undone}
        base = source
        if where == "top":            # convert the audio beneath the recorded version, as Undo leaves it
            below = _below(source, record["version_path"])
            base = source.model_copy(update={"audio_override": below[-1]["path"] if below else None})
        audio_path = noise.source_audio(base)
        folder = noise.audio_dir(project_id)
        versions = folder / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="voice_map_", dir=str(folder)))
        done: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        try:
            report(0.02, "decoding the source")
            base = decode_wav(audio_path, work / "base.wav", sample_rate=SAMPLE_RATE, mono=True)
            out_wav = work / "converted.wav"
            # written by libsndfile, not copied: ffmpeg's INFO tag would keep it from opening r+
            with sf.SoundFile(str(base)) as src, sf.SoundFile(str(out_wav), "w", samplerate=src.samplerate,
                                                              channels=1, subtype="FLOAT") as dst:
                processes._copy_frames(src, dst, 0, src.frames)
            total = sf.info(str(base)).frames
            pad = int(round(processes.PAD_S * SAMPLE_RATE))
            total_s = sum(j["end_s"] - j["start_s"] for j in jobs) or 1.0
            done_s = 0.0
            for k, job in enumerate(jobs):
                length = job["end_s"] - job["start_s"]
                low, high = 0.05 + 0.88 * done_s / total_s, 0.05 + 0.88 * (done_s + length) / total_s
                label = f"{job['speaker']} -> {job['voice']} ({k + 1}/{len(jobs)})"
                report(low, label)
                try:
                    a, b = processes._range_samples(job["start_s"], job["end_s"], total, SAMPLE_RATE)
                    pa, pb = max(0, a - pad), min(total, b + pad)
                    segment_in, segment_out = work / f"span_{k:03d}.wav", work / f"span_{k:03d}_out.wav"
                    with sf.SoundFile(str(base)) as src:
                        src.seek(pa)
                        sf.write(str(segment_in), src.read(pb - pa, dtype="float32"), SAMPLE_RATE,
                                 subtype="FLOAT")
                    process_wav(segment_in, segment_out, job["values"],
                                lambda f, m, lo=low, hi=high: report(lo + (hi - lo) * f, f"{label}: {m}"),
                                worker)
                    processes._fit_length(segment_out, pb - pa)
                    _splice_into(out_wav, segment_out, pa, a, b, processes.FADE_S)
                    done.append(job)
                except (seedvc_env.SeedVCUnavailable, seedvc_env.SeedVCBusy):
                    raise                                   # no span can run: nothing is written
                except Exception as exc:
                    logger.warning("Voice map: %s keeps the original audio: %s", label, exc)
                    failed.append({"speaker": job["speaker"], "voice": job["voice"],
                                   "span": [round(job["start_s"], 3), round(job["end_s"], 3)],
                                   "error": str(exc) or exc.__class__.__name__})
                done_s += length
            if not done:
                raise RuntimeError("No span converted: " + "; ".join(f["error"] for f in failed[:3]))
            report(0.95, "saving the version")
            number = processes._next_version(versions)
            final = versions / f"{number}_{EFFECT}.wav"
            os.replace(out_wav, final)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        conversions: List[Dict[str, Any]] = []
        for job in sorted(done, key=lambda j: j["start_s"]):
            entry = next((c for c in conversions if c["speaker"] == job["speaker"] and c["voice"] == job["voice"]),
                         None)
            if entry is None:
                entry = {"speaker": job["speaker"], "voice": job["voice"], "spans": [], "seconds": 0.0}
                conversions.append(entry)
            entry["spans"].append([round(job["start_s"], 3), round(job["end_s"], 3)])
            entry["seconds"] = round(entry["seconds"] + job["end_s"] - job["start_s"], 3)
        version = {"path": str(final), "effect": EFFECT, "params": {"voice_map": dict(voice_map)},
                   "start_s": None, "end_s": None, "created": _now(), "voice_conversions": conversions}
        popped: Dict[str, Any] = {}

        def push(src: Any) -> None:
            if where == "top":
                _undo_top(record["version_path"], popped)(src)       # replaced, not stacked
            src.audio_history.append(version)
            src.audio_override = str(final)

        state = processes._save_source(project_id, sid, push)
        p_data = noise.project_store.get_project(project_id) or {}
        metadata = dict(p_data.get("metadata") or {})
        metadata["voice_conversions"] = conversions
        metadata["voice_conversions_at"] = version["created"]
        p_data["metadata"] = metadata
        settings = dict(p_data.get("settings") or {})
        settings[APPLIED_KEY] = {"map_hash": voice_map_hash(voice_map, done), "version_path": str(final),
                                 "spans_signature": spans_signature(done), "applied_at": version["created"],
                                 "voice_conversions": conversions}
        p_data["settings"] = settings
        noise.project_store.save_project(project_id, p_data)
    report(1.0, "done")
    result = {**state, "voice_conversions": conversions, "failed_spans": failed, "version": version}
    if where == "top":
        result.update(replaced=True, undone=popped.get("path"))
    elif where == "under":
        result["warning"] = ON_TOP_WARNING
    return result


# -------------------------------------------------------------------- jobs

def start_apply_job(project_id: str) -> str:
    """`apply_voice_map` as a job of the process lane (`processes.JOBS`): poll it with
    `GET /api/audio/{project}/process/{job_id}`; progress 0..1 across the spans."""
    from audio import processes
    job_id = f"audio_process_{uuid.uuid4().hex[:12]}"
    with processes._jobs_guard:
        processes.JOBS[job_id] = {"id": job_id, "job_type": "audio_process", "project_id": project_id,
                                  "source_id": None, "effect": EFFECT, "op": "apply_voice_map",
                                  "status": "pending", "progress": 0.0, "message": "queued",
                                  "result": None, "error": None, "created": _now()}
    threading.Thread(target=_run_apply_job, args=(job_id, project_id), name=job_id, daemon=True).start()
    return job_id


def _run_apply_job(job_id: str, project_id: str) -> None:
    from audio import processes

    def progress(fraction: float, message: str) -> None:
        with processes._jobs_guard:
            job = processes.JOBS.get(job_id)
            if job is not None:
                job["progress"] = round(max(job["progress"], fraction), 4)
                job["message"] = message

    processes._update_job(job_id, status="running", message="starting")
    try:
        result = apply_voice_map(project_id, progress)
    except Exception as exc:
        logger.warning("Voice map %s failed: %s", job_id, exc, exc_info=True)
        processes._update_job(job_id, status="failed", error=str(exc) or exc.__class__.__name__,
                              message="failed", finished=_now())
        return
    processes._update_job(job_id, status="completed", progress=1.0, message="done", result=result,
                          finished=_now())


# ------------------------------------------------- the Voice panel (brief S2a)

REGION_STATUSES = ("pending", "applied", "failed")
REGION_PARAM_KEYS = ("match_pitch", "pitch_shift", "steps", "cfg_rate")
REGION_TAG = "voice_region:"     # the `preset` of a region's FX on the A1 rack: "voice_region:<id>"
REGION_FADE_S = 0.015            # SkinWalker's seams (VOICE_CHANGER_PLAN.md §2.7); the lane's own are 10 ms
_REGION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FILE_STEM_RE = re.compile(r"[^A-Za-z0-9 _.-]+")
_regions_lock = threading.RLock()     # read-modify-write of `voice_regions` (taken after a source lock)
_pool_lock = threading.Lock()         # a converted file's name and its media-pool entry


def request_entry(voice: Any = None, preset: Any = None,
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A request's `{voice, preset?, params}` as one voice-map-shaped entry: the known keys of
    `params`, then the voice and the preset named beside them (when given)."""
    entry = {key: value for key, value in dict(params or {}).items()
             if key in VOICE_MAP_KEYS and value is not None}
    if str(voice or "").strip():
        entry["voice"] = str(voice).strip()
    if str(preset or "").strip():
        entry["preset"] = str(preset).strip()
    return entry


def resolved_values(entry: Dict[str, Any]) -> Dict[str, Any]:
    """voice_conversion values for a voice-map-shaped entry (the preset's values, then the
    entry's own); ValueError when it cannot run (no voice, a voice not in the library, a value
    outside its choices)."""
    from audio import processes
    try:
        if str(entry.get("voice") or "").strip():
            voice_path(entry["voice"])                     # a clear message before the choice check
        values = processes.resolve_params(EFFECT, entry_params(entry))
        voice, _params = engine_params(values)
        voice_path(voice)
    except NotFound as exc:
        raise ValueError(str(exc)) from exc
    return values


def request_params(params: Optional[Dict[str, Any]], voice: Optional[str] = None,
                   preset: Optional[str] = None) -> Dict[str, Any]:
    """The voice_conversion params of a `/voice/convert` request: `params` as given (S1), or --
    when a voice or a preset is named beside them -- the preset's values, then `params`, then
    that voice and preset."""
    if voice is None and preset is None:
        return dict(params or {})
    return entry_params(request_entry(voice, preset, params))


def programme_seconds(tl: Any) -> float:
    """Where the programme ends: the last A1 trim's end in programme seconds (0 without A1)."""
    segments = a1_segments(tl)
    return float(segments[-1].prog_end_s) if segments else 0.0


def program_pieces(tl: Any, start_s: float, end_s: float) -> List[Tuple[str, str, float, float]]:
    """Programme [start_s, end_s) as source pieces `(source_id, path, start_s, end_s)`, split
    at every cut (`timeline/program_time.program_to_source`)."""
    if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s <= start_s:
        raise ValueError(f"Empty range: {start_s}..{end_s} s")
    if not a1_segments(tl):
        raise NotFound("The timeline has no A1 clips, so there is no programme to map")
    pieces = program_to_source(tl, start_s, end_s)
    if not pieces:
        raise ValueError(f"Programme {start_s}..{end_s} s holds no A1 audio")
    return pieces


def one_source(pieces: List[Tuple[str, str, float, float]]) -> str:
    """The one source the pieces are on (Conflict when they are on several)."""
    sources = {sid for sid, _path, _a, _b in pieces}
    if len(sources) > 1:
        raise Conflict("The range spans clips of more than one source -- one source at a time")
    return pieces[0][0]


def _finite_number(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, not {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, not {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite")
    return number


def _region_fx(raw: Any, where: str) -> List[Dict[str, Any]]:
    """A region's FX as AudioEffect dicts, unwindowed and untagged (the window, the tag and a
    fresh id are set when the region is applied)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{where}.fx must be a list of effects")
    effects: List[Dict[str, Any]] = []
    for k, item in enumerate(raw):
        if not isinstance(item, dict) or not str(item.get("effect") or "").strip():
            raise ValueError(f"{where}.fx[{k}] must be an effect {{effect, params?, enabled?}}")
        try:
            effect = AudioEffect.model_validate({
                "id": str(item.get("id") or "").strip() or f"fx_{uuid.uuid4().hex[:8]}",
                "effect": str(item["effect"]).strip(),
                "params": item.get("params") if item.get("params") is not None else {},
                "enabled": item.get("enabled", True)})
        except ValueError as exc:                      # pydantic's ValidationError is a ValueError
            raise ValueError(f"{where}.fx[{k}] is not an audio effect: {exc}") from exc
        effects.append(effect.model_dump())
    return effects


def normalize_regions(raw: Any, programme_s: float) -> List[Dict[str, Any]]:
    """Voice regions as they are stored. Each is checked (ValueError for a malformed one: times,
    voice or preset, params, fx, status, a repeated id; a missing id is made), then -- like
    SkinWalker's `parse_manifest` -- clamped to the programme [0, programme_s], dropped when
    nothing is left, sorted by start, and a region overlapping the one before it starts where
    that one ends (dropped when that leaves nothing)."""
    if not isinstance(raw, list):
        raise ValueError("regions must be a list")
    seen: set = set()
    cleaned: List[Dict[str, Any]] = []
    for k, item in enumerate(raw):
        where = f"regions[{k}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        rid = str(item.get("id") or "").strip() or f"vr_{uuid.uuid4().hex[:8]}"
        if not _REGION_ID_RE.match(rid):
            raise ValueError(f"{where}.id {rid!r} must be 1-64 letters, digits, '_' or '-'")
        if rid in seen:
            raise ValueError(f"Region id {rid!r} appears twice")
        seen.add(rid)
        start = _finite_number(item.get("start_s"), f"{where}.start_s")
        end = _finite_number(item.get("end_s"), f"{where}.end_s")
        voice, preset = item.get("voice"), item.get("preset")
        if voice is not None and not isinstance(voice, str):
            raise ValueError(f"{where}.voice must be a voice name")
        if preset is not None and not isinstance(preset, str):
            raise ValueError(f"{where}.preset must be a preset name")
        voice, preset = (voice or "").strip(), (preset or "").strip() or None
        if not voice and not preset:
            raise ValueError(f"{where} needs a voice (or a preset that names one)")
        params = item.get("params") if item.get("params") is not None else {}
        if not isinstance(params, dict):
            raise ValueError(f"{where}.params must be an object {{match_pitch, pitch_shift, steps, cfg_rate}}")
        clean_params: Dict[str, Any] = {}
        for key in REGION_PARAM_KEYS:
            value = params.get(key)
            if value is None:
                continue
            if key == "match_pitch":
                clean_params[key] = (value.strip().lower() in ("1", "true", "yes", "on")
                                     if isinstance(value, str) else bool(value))
            else:
                clean_params[key] = _finite_number(value, f"{where}.params.{key}")
        status = item.get("status") or "pending"
        if status not in REGION_STATUSES:
            raise ValueError(f"{where}.status must be one of {list(REGION_STATUSES)}, not {status!r}")
        region: Dict[str, Any] = {
            "id": rid,
            "start_s": round(min(max(start, 0.0), programme_s), 6),
            "end_s": round(min(max(end, 0.0), programme_s), 6),
            "voice": voice, "preset": preset, "params": clean_params,
            "fx": _region_fx(item.get("fx"), where), "status": status}
        if status == "failed" and item.get("error"):
            region["error"] = str(item["error"])
        if status == "applied" and item.get("applied_version"):
            region["applied_version"] = str(item["applied_version"])
        if region["end_s"] <= region["start_s"]:
            continue                                   # outside the programme
        cleaned.append(region)
    cleaned.sort(key=lambda r: r["start_s"])
    resolved: List[Dict[str, Any]] = []
    for region in cleaned:
        if resolved and region["start_s"] < resolved[-1]["end_s"]:
            region["start_s"] = resolved[-1]["end_s"]
        if region["end_s"] <= region["start_s"]:
            continue                                   # covered by the region before it
        resolved.append(region)
    return resolved


def _stored_regions(p_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    regions = (p_data.get("settings") or {}).get("voice_regions")
    if not isinstance(regions, list):
        return []
    return [dict(region) for region in regions if isinstance(region, dict) and region.get("id")]


def _save_regions(project_id: str, p_data: Dict[str, Any], tl: Any,
                  regions: List[Dict[str, Any]]) -> None:
    """Store `voice_regions` (and the timeline as `tl` has it), revision + 1."""
    settings = dict(p_data.get("settings") or {})
    settings["voice_regions"] = regions
    p_data["settings"] = settings
    tl.revision += 1
    p_data["timeline"] = tl.model_dump()
    noise.project_store.save_project(project_id, p_data)


def get_regions(project_id: str) -> Dict[str, Any]:
    _pid, p_data, _tl = noise.load_project(project_id)
    return {"regions": _stored_regions(p_data)}


def set_regions(project_id: str, regions: Any) -> Dict[str, Any]:
    """Replace the project's voice regions (programme seconds), checked and trimmed by
    `normalize_regions`; the timeline's revision goes up by one."""
    with _regions_lock:
        _pid, p_data, tl = noise.load_project(project_id)
        programme_s = programme_seconds(tl)
        if isinstance(regions, list) and regions and programme_s <= 0.0:
            raise Conflict("The timeline has no A1 clips: voice regions are programme time")
        clean = normalize_regions(regions, programme_s)
        _save_regions(project_id, p_data, tl, clean)
    return {"regions": clean}


def _selected(regions: List[Dict[str, Any]], ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    """The pending regions, or the listed ones (NotFound for an id that is not a region)."""
    if ids is None:
        return [region for region in regions if region.get("status") == "pending"]
    wanted = [str(i) for i in ids]
    known = {region["id"] for region in regions}
    missing = [i for i in wanted if i not in known]
    if missing:
        raise NotFound(f"No voice region {missing[0]!r} in the project")
    return [region for region in regions if region["id"] in set(wanted)]


def preflight_regions(project_id: str, ids: Optional[List[str]] = None) -> None:
    """What `apply_regions` needs before it is queued: a region to apply (Conflict when none is
    pending or listed; NotFound for an unknown id)."""
    _pid, p_data, _tl = noise.load_project(project_id)
    if not _selected(_stored_regions(p_data), ids):
        raise Conflict("No pending voice region to apply" if ids is None else "No voice region listed")


def _taken_effect_ids(tl: Any) -> set:
    """Every rack effect id in the project, track racks and clip racks (ids name ffmpeg instances)."""
    taken = {effect.id for rack in tl.audio_racks.values() for effect in rack}

    def walk(items: List[Any]) -> None:
        for item in items:
            taken.update(effect.id for effect in item.audio_effects)
            walk(item.children)

    walk(tl.items)
    return taken


def _fresh_effect_id(taken: set) -> str:
    """`fx_<8 hex>`, as the Audio panel makes them, not in `taken` (and added to it)."""
    while True:
        candidate = f"fx_{uuid.uuid4().hex[:8]}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate


def apply_regions(project_id: str, ids: Optional[List[str]] = None, progress: Progress = None,
                  worker: Optional[seedvc_env.Worker] = None) -> Dict[str, Any]:
    """Convert every pending voice region (or the listed ones) into ONE new version of the
    programme source.

    Each region's programme range becomes source pieces (`program_to_source`, split at every
    cut); each piece is converted with 0.5 s of context and spliced back with REGION_FADE_S
    crossfades, as the voice map's spans are, so audio outside the pieces is left sample for
    sample. A region whose voice cannot run, whose range holds no audio of the programme
    source, or whose conversion fails keeps the original audio and is marked failed with its
    error; the others are marked applied, and each one's `fx` goes on the A1 rack windowed to
    the region (programme seconds) and tagged `preset: "voice_region:<id>"` (a re-applied
    region's earlier effects are replaced). An unavailable or busy engine aborts with nothing
    written. Returns `{applied: [ids], failed: [{id, error}], version}` (version None when no
    region converted)."""
    from audio import processes

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, fraction)), message)
            except Exception:
                logger.debug("progress callback failed", exc_info=True)

    _pid, p_data, tl = noise.load_project(project_id)
    sid, _source = noise.resolve_source(tl, p_data, None)
    failed: List[Dict[str, Any]] = []
    done: List[Dict[str, Any]] = []
    version: Optional[Dict[str, Any]] = None
    with processes._source_lock(project_id, sid):
        _pid, p_data, tl = noise.load_project(project_id)
        sid, source = noise.resolve_source(tl, p_data, sid)
        selected = sorted(_selected(_stored_regions(p_data), ids), key=lambda r: r["start_s"])
        order = {region["id"]: k for k, region in enumerate(selected)}
        jobs: List[Dict[str, Any]] = []
        for region in selected:
            try:
                values = resolved_values(request_entry(region.get("voice"), region.get("preset"),
                                                       region.get("params")))
                pieces = program_pieces(tl, float(region["start_s"]), float(region["end_s"]))
                others = sorted({piece[0] for piece in pieces if piece[0] != sid})
                if others:
                    raise ValueError(f"The region plays clips of another source ({', '.join(others)}); "
                                     f"voice regions convert the programme source {sid!r}")
            except (ValueError, NotFound, Conflict) as exc:
                failed.append({"id": region["id"], "error": str(exc) or exc.__class__.__name__})
                continue
            jobs.append({"region": region, "values": values, "voice": engine_params(values)[0],
                         "pieces": [(a, b) for _sid, _path, a, b in pieces]})
        if jobs:
            audio_path = noise.source_audio(source)
            folder = noise.audio_dir(project_id)
            versions = folder / "versions"
            versions.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix="voice_regions_", dir=str(folder)))
            try:
                report(0.02, "decoding the source")
                base = decode_wav(audio_path, work / "base.wav", sample_rate=SAMPLE_RATE, mono=True)
                out_wav = work / "converted.wav"
                # written by libsndfile, not copied: ffmpeg's INFO tag would keep it from opening r+
                with sf.SoundFile(str(base)) as src, sf.SoundFile(str(out_wav), "w", samplerate=src.samplerate,
                                                                  channels=1, subtype="FLOAT") as dst:
                    processes._copy_frames(src, dst, 0, src.frames)
                total = sf.info(str(base)).frames
                pad = int(round(processes.PAD_S * SAMPLE_RATE))
                total_s = sum(b - a for job in jobs for a, b in job["pieces"]) or 1.0
                done_s = 0.0
                for k, job in enumerate(jobs):
                    region = job["region"]
                    label = f"region {region['id']} -> {job['voice']} ({k + 1}/{len(jobs)})"
                    report(0.05 + 0.88 * done_s / total_s, label)
                    try:
                        converted: List[Tuple[Path, int, int, int]] = []
                        at_s = done_s
                        for n, (start_s, end_s) in enumerate(job["pieces"]):
                            a, b = processes._range_samples(start_s, end_s, total, SAMPLE_RATE)
                            pa, pb = max(0, a - pad), min(total, b + pad)
                            segment_in = work / f"region_{k:03d}_{n:02d}.wav"
                            segment_out = work / f"region_{k:03d}_{n:02d}_out.wav"
                            with sf.SoundFile(str(base)) as src:
                                src.seek(pa)
                                sf.write(str(segment_in), src.read(pb - pa, dtype="float32"), SAMPLE_RATE,
                                         subtype="FLOAT")
                            low = 0.05 + 0.88 * at_s / total_s
                            high = 0.05 + 0.88 * (at_s + end_s - start_s) / total_s
                            process_wav(segment_in, segment_out, job["values"],
                                        lambda f, m, lo=low, hi=high: report(lo + (hi - lo) * f, f"{label}: {m}"),
                                        worker)
                            processes._fit_length(segment_out, pb - pa)
                            converted.append((segment_out, pa, a, b))
                            at_s += end_s - start_s
                        for segment_out, pa, a, b in converted:      # all of a region's pieces, or none
                            _splice_into(out_wav, segment_out, pa, a, b, REGION_FADE_S)
                        done.append(job)
                    except (seedvc_env.SeedVCUnavailable, seedvc_env.SeedVCBusy):
                        raise                                   # no region can run: nothing is written
                    except Exception as exc:
                        logger.warning("Voice regions: %s keeps the original audio: %s", label, exc)
                        failed.append({"id": region["id"], "error": str(exc) or exc.__class__.__name__})
                    done_s += sum(b - a for a, b in job["pieces"])
                if done:
                    report(0.95, "saving the version")
                    number = processes._next_version(versions)
                    final = versions / f"{number}_{EFFECT}.wav"
                    os.replace(out_wav, final)
                    version = {
                        "path": str(final), "effect": EFFECT,
                        "params": {"voice_regions": [
                            {"id": job["region"]["id"], "voice": job["voice"],
                             "start_s": job["region"]["start_s"], "end_s": job["region"]["end_s"]}
                            for job in done]},
                        "start_s": None, "end_s": None, "created": _now(),
                        "voice_regions": [job["region"]["id"] for job in done]}
            finally:
                shutil.rmtree(work, ignore_errors=True)

        applied = [job["region"]["id"] for job in done]
        failed.sort(key=lambda entry: order.get(entry["id"], 0))
        if applied or failed:
            errors = {entry["id"]: entry["error"] for entry in failed}
            with _regions_lock:
                _pid, p_data, tl = noise.load_project(project_id)
                if version is not None:
                    if sid not in tl.sources:
                        raise NotFound(f"Source {sid!r} left the timeline while its regions were converted")
                    tl.sources[sid].audio_history.append(version)
                    tl.sources[sid].audio_override = version["path"]
                    tags = {REGION_TAG + rid for rid in applied}
                    rack = [effect for effect in tl.audio_racks.get("A1", []) if effect.preset not in tags]
                    taken = _taken_effect_ids(tl)
                    for job in done:
                        region = job["region"]
                        for fx in region.get("fx") or []:
                            rack.append(AudioEffect.model_validate({
                                **fx, "id": _fresh_effect_id(taken), "start_s": region["start_s"],
                                "end_s": region["end_s"], "preset": REGION_TAG + region["id"]}))
                    tl.audio_racks["A1"] = rack
                regions = _stored_regions(p_data)
                for region in regions:
                    if region["id"] in applied:
                        region["status"] = "applied"
                        region["applied_version"] = version["path"] if version else None
                        region.pop("error", None)
                    elif region["id"] in errors:
                        region["status"] = "failed"
                        region["error"] = errors[region["id"]]
                        region.pop("applied_version", None)
                _save_regions(project_id, p_data, tl, regions)
            if version is not None:
                processes._timeline_changed(project_id)
    report(1.0, "done")
    return {"applied": applied, "failed": failed, "version": version}


def undo_regions(project_id: str) -> Dict[str, Any]:
    """Undo the last voice-region apply: `processes.undo` pops its version (the override goes
    back to the version before it, or to the original), the A1 rack loses the effects tagged
    with its regions, and those regions are pending again. Conflict when the newest version of
    the programme source is not a voice-region apply (undo that one first)."""
    from audio import processes
    _pid, p_data, tl = noise.load_project(project_id)
    sid, source = noise.resolve_source(tl, p_data, None)
    last = source.audio_history[-1] if source.audio_history else None
    if not isinstance(last, dict) or not isinstance(last.get("voice_regions"), list):
        reason = ("the newest version of the programme source is not a voice-region apply (undo it first)"
                  if last else "no voice region has been applied")
        raise Conflict(f"Nothing to undo: {reason}")
    undone = processes.undo(project_id, sid).get("undone") or {}
    ids = [str(rid) for rid in undone.get("voice_regions") or []]
    tags = {REGION_TAG + rid for rid in ids}
    with _regions_lock:
        _pid, p_data, tl = noise.load_project(project_id)
        if "A1" in tl.audio_racks:
            tl.audio_racks["A1"] = [effect for effect in tl.audio_racks["A1"] if effect.preset not in tags]
        regions = _stored_regions(p_data)
        for region in regions:
            if region["id"] in ids:
                region["status"] = "pending"
                region.pop("error", None)
                region.pop("applied_version", None)
        _save_regions(project_id, p_data, tl, regions)
    processes._timeline_changed(project_id)
    return {"regions": regions}


def convert_input(project_id: str, media_id: Optional[str] = None,
                  path: Optional[str] = None) -> Tuple[Path, str]:
    """The file a whole-file conversion reads and its name: a media-pool item, a file path, or
    (neither given) the project's programme source as it sounds now."""
    if media_id and path:
        raise ValueError("Give media_id or path, not both")
    _pid, p_data, tl = noise.load_project(project_id)
    if media_id:
        entry = next((m for m in p_data.get("media_pool") or []
                      if isinstance(m, dict) and m.get("id") == media_id), None)
        if entry is None:
            raise NotFound(f"No media {media_id!r} in the project's library")
        if entry.get("kind") == "image" or entry.get("has_audio") is False:
            raise ValueError(f"{entry.get('name') or media_id} has no audio to convert")
        file = Path(str(entry.get("path") or ""))
        if not entry.get("path") or not file.is_file():
            raise NotFound(f"The file of media {media_id!r} is missing: {entry.get('path')}")
        return file, str(entry.get("name") or file.name)
    if path:
        file = Path(str(path))
        if not file.is_file():
            raise NotFound(f"No file {file}")
        return file, file.name
    _sid, source = noise.resolve_source(tl, p_data, None)
    return Path(noise.source_audio(source)), Path(source.path).name


def _converted_path(folder: Path, stem: str) -> Path:
    """`<folder>/<stem>.wav`, numbered `_2`, `_3`... when taken; the stem made filename-safe."""
    clean = _FILE_STEM_RE.sub("_", stem).strip(" ._")[:80].strip(" ._") or "converted"
    candidate, number = folder / f"{clean}.wav", 2
    while candidate.exists():
        candidate, number = folder / f"{clean}_{number}.wav", number + 1
    return candidate


def convert_file(project_id: str, media_id: Optional[str] = None, path: Optional[str] = None,
                 voice: Any = None, preset: Any = None, params: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None, progress: Progress = None,
                 worker: Optional[seedvc_env.Worker] = None) -> Dict[str, Any]:
    """A whole file in a voice, as a new media-pool item: a media-pool item (`media_id`), a file
    (`path`) or the project's source as it sounds now (neither) goes through the worker and comes
    back a 48 kHz mono WAV of the same length under the project's `audio/converted/`, registered
    the way `routes/projects.py` adds media (`media_pool.import_paths`; owned by the project, so
    not `linked`). `name` names the file and the item (default `<input>_<voice>`). Returns
    `{media_id, path, seconds}`."""
    from store import media_pool

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, fraction)), message)
            except Exception:
                logger.debug("progress callback failed", exc_info=True)

    values = resolved_values(request_entry(voice, preset, params))
    voice_name = engine_params(values)[0]
    src, label = convert_input(project_id, media_id, path)
    folder = noise.audio_dir(project_id)
    converted = folder / "converted"
    converted.mkdir(parents=True, exist_ok=True)
    title = str(name).strip() if name is not None and str(name).strip() else None
    work = Path(tempfile.mkdtemp(prefix="voice_file_", dir=str(folder)))
    try:
        report(0.02, f"decoding {label}")
        source_wav = decode_wav(str(src), work / "source.wav", sample_rate=SAMPLE_RATE, mono=True)
        out_wav = work / "converted.wav"
        process_wav(source_wav, out_wav, values, lambda f, m: report(0.05 + 0.85 * f, m), worker)
        frames = sf.info(str(out_wav)).frames
        report(0.93, "adding it to the media pool")
        with _pool_lock:
            final = _converted_path(converted, title or f"{Path(label).stem}_{voice_name}")
            os.replace(out_wav, final)
            try:
                _pid, p_data, _tl = noise.load_project(project_id)
                pool = p_data.get("media_pool")
                if not isinstance(pool, list):
                    pool = []
                added, skipped = media_pool.import_paths(project_id, pool, [str(final)])
                if not added:
                    raise RuntimeError("The converted file could not join the media pool: "
                                       + ("; ".join(skipped) or "unreadable"))
                added[0]["name"] = title or final.name
                added[0]["linked"] = False              # the project owns it, like its own source
                p_data["media_pool"] = pool
                noise.project_store.save_project(project_id, p_data)
            except Exception:
                final.unlink(missing_ok=True)
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report(1.0, "done")
    return {"media_id": added[0]["id"], "path": str(final), "seconds": round(frames / SAMPLE_RATE, 3)}


def check_media_ids(media_ids: Any) -> List[str]:
    """A batch's media ids: a list of at least one non-empty id (ValueError otherwise)."""
    if not isinstance(media_ids, (list, tuple)) or not media_ids:
        raise ValueError("media_ids must list at least one media-pool item")
    ids = [str(media_id or "").strip() for media_id in media_ids]
    if not all(ids):
        raise ValueError("media_ids holds an empty id")
    return ids


def batch(project_id: str, media_ids: Any, voice: Any = None, preset: Any = None,
          params: Optional[Dict[str, Any]] = None, progress: Progress = None,
          worker: Optional[seedvc_env.Worker] = None) -> Dict[str, Any]:
    """Several media-pool items in one voice, each into a new media-pool item (`convert_file`),
    one after another with progress per item. Reported per item: `{items: [{media_id,
    out_media_id, path} | {media_id, error}]}`. An item that fails does not stop the rest; an
    unavailable or busy engine does, and the items it did not reach carry its error."""

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, fraction)), message)
            except Exception:
                logger.debug("progress callback failed", exc_info=True)

    ids = check_media_ids(media_ids)
    resolved_values(request_entry(voice, preset, params))        # a bad voice is refused before any item
    items: List[Dict[str, Any]] = []
    for k, media_id in enumerate(ids):
        low, high = k / len(ids), (k + 1) / len(ids)
        tag = f"{k + 1}/{len(ids)} {media_id}"
        report(low, tag)
        try:
            got = convert_file(project_id, media_id=media_id, voice=voice, preset=preset, params=params,
                               progress=lambda f, m, lo=low, hi=high, tag=tag: report(lo + (hi - lo) * f,
                                                                                    f"{tag}: {m}"),
                               worker=worker)
        except (seedvc_env.SeedVCUnavailable, seedvc_env.SeedVCBusy) as exc:
            items.extend({"media_id": rest, "error": str(exc)} for rest in ids[k:])
            break
        except Exception as exc:
            logger.warning("Voice batch: %s failed: %s", media_id, exc)
            items.append({"media_id": media_id, "error": str(exc) or exc.__class__.__name__})
            continue
        items.append({"media_id": media_id, "out_media_id": got["media_id"], "path": got["path"]})
    report(1.0, "done")
    return {"items": items}


# -------------------------------------------------------------- S2a jobs

def _start_job(project_id: str, op: str, work: Callable[[Progress], Dict[str, Any]]) -> str:
    """`work(progress)` as a job of the process lane (`processes.JOBS`), polled like the others
    on `GET /api/audio/{project}/process/{job_id}`; progress 0..1."""
    from audio import processes
    job_id = f"audio_process_{uuid.uuid4().hex[:12]}"
    with processes._jobs_guard:
        processes.JOBS[job_id] = {"id": job_id, "job_type": "audio_process", "project_id": project_id,
                                  "source_id": None, "effect": EFFECT, "op": op, "status": "pending",
                                  "progress": 0.0, "message": "queued", "result": None, "error": None,
                                  "created": _now()}
    threading.Thread(target=_run_job, args=(job_id, op, work), name=job_id, daemon=True).start()
    return job_id


def _run_job(job_id: str, op: str, work: Callable[[Progress], Dict[str, Any]]) -> None:
    from audio import processes

    def progress(fraction: float, message: str) -> None:
        with processes._jobs_guard:
            job = processes.JOBS.get(job_id)
            if job is not None:
                job["progress"] = round(max(job["progress"], fraction), 4)
                job["message"] = message

    processes._update_job(job_id, status="running", message="starting")
    try:
        result = work(progress)
    except Exception as exc:
        logger.warning("Voice %s %s failed: %s", op, job_id, exc, exc_info=True)
        processes._update_job(job_id, status="failed", error=str(exc) or exc.__class__.__name__,
                              message="failed", finished=_now())
        return
    processes._update_job(job_id, status="completed", progress=1.0, message="done", result=result,
                          finished=_now())


def start_regions_job(project_id: str, ids: Optional[List[str]] = None) -> str:
    """`apply_regions` as a job; its result is `{applied, failed, version}`."""
    return _start_job(project_id, "apply_regions",
                      lambda progress: apply_regions(project_id, ids, progress))


def start_convert_file_job(project_id: str, media_id: Optional[str] = None, path: Optional[str] = None,
                           voice: Any = None, preset: Any = None, params: Optional[Dict[str, Any]] = None,
                           name: Optional[str] = None) -> str:
    """`convert_file` as a job; its result is `{media_id, path, seconds}`."""
    return _start_job(project_id, "convert_file",
                      lambda progress: convert_file(project_id, media_id, path, voice, preset, params, name,
                                                    progress))


def start_batch_job(project_id: str, media_ids: Any, voice: Any = None, preset: Any = None,
                    params: Optional[Dict[str, Any]] = None) -> str:
    """`batch` as a job; its result is `{items: [...]}`."""
    return _start_job(project_id, "batch",
                      lambda progress: batch(project_id, media_ids, voice, preset, params, progress))
