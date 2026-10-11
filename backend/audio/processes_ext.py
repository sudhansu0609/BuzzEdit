"""Process-lane effects added by brief A3 (AUDIO_SUITE_PLAN.md §2.6, §3).

    PROCESSES_EXT = {"voice_changer": EffectDef, "enhance_speech_ai": EffectDef}
    run_process_ext(effect, in_wav, out_wav, params, progress) -> Path

Each entry has the §2.6 EffectDef shape of `audio/catalogue.py` (same
ParamDef helpers) with engine "process", windowable False, plus
`available`: False hides the entry from the catalogue (pyworld missing for
the Voice Changer; no engine in data/models/enhance/status.json for Enhance
Speech (AI)). The process registry merges this dict in a follow-up;
`run_process_ext` is its runner for these two ids. Both write a new WAV and
never touch the input; `progress(fraction, message)` as in the other
processes.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from audio import enhance_ai
from audio.catalogue import _choice, _num
from audio.compile import _coerce, compile_rack
from audio.voice_changer import VOICE_PRESETS, change_voice
from config import FFMPEG_BIN

Progress = Optional[Callable[[float, str], None]]

VOICE_CUSTOM = "Custom"


def _pyworld_available() -> bool:
    try:
        import pyworld  # noqa: F401
    except Exception:
        return False
    return True


def _process_def(eid: str, name: str, category: str, description: str,
                 params: List[Dict[str, Any]], presets: Dict[str, Dict[str, Any]],
                 available: bool) -> Dict[str, Any]:
    keys = {p["key"] for p in params}
    for pname, values in presets.items():
        assert set(values) <= keys, (eid, pname)
    return {"id": eid, "name": name, "category": category, "engine": "process",
            "description": description, "params": params, "presets": presets,
            "windowable": False, "available": bool(available)}


PROCESSES_EXT: Dict[str, Dict[str, Any]] = {
    "voice_changer": _process_def(
        "voice_changer", "Voice Changer", "Time and Pitch",
        "Changes the voice: pitch and formants independently (WORLD re-synthesis), speed, "
        "and character presets with their own effects (Robot, Ghost, Monster, ...).",
        [_num("pitch", "Pitch", 0.0, -24.0, 24.0, 0.1, "st"),
         _num("formant", "Formant", 1.0, 0.5, 2.0, 0.01, "x"),
         _num("speed", "Speed", 1.0, 0.5, 2.0, 0.01, "x"),
         _choice("preset", "Preset", VOICE_CUSTOM, [VOICE_CUSTOM] + list(VOICE_PRESETS))],
        {name: {"pitch": v["pitch_semitones"], "formant": v["formant_ratio"],
                "speed": v["speed"], "preset": name}
         for name, v in VOICE_PRESETS.items()},
        _pyworld_available()),
    "enhance_speech_ai": _process_def(
        "enhance_speech_ai", "Enhance Speech (AI)", "Special",
        "Speech restoration with a neural model (noise, reverb, band-limited or "
        "clipped recordings); runs on the GPU when one is free.",
        [], {}, enhance_ai.available()),
}


def _effect_id(effect: Any) -> str:
    if isinstance(effect, str):
        return effect
    if isinstance(effect, Mapping):
        return str(effect.get("effect", ""))
    return str(getattr(effect, "effect", ""))


def _resolve(edef: Dict[str, Any], params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    given = dict(params or {})
    return {d["key"]: _coerce(d, given.get(d["key"], d["default"])) for d in edef["params"]}


def _apply_rack(src: Path, dst: Path, rack: List[Dict[str, Any]]) -> None:
    """Run an AudioEffect rack (ffmpeg catalogue) over a file, keeping its
    rate, channel count and length."""
    import soundfile as sf

    chain = compile_rack(rack, scope="process").chain
    if not chain:
        shutil.copyfile(src, dst)
        return
    info = sf.info(str(src))
    r = subprocess.run([FFMPEG_BIN, "-hide_banner", "-nostats", "-v", "error", "-y",
                        "-i", str(src), "-filter_complex", f"[0:a]{chain}[out]",
                        "-map", "[out]", "-ar", str(info.samplerate), "-ac", str(info.channels),
                        "-c:a", "pcm_f32le", str(dst)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"effect rack failed: {r.stderr[-800:]}")


def run_process_ext(effect: Any, in_wav: Path, out_wav: Path,
                    params: Optional[Mapping[str, Any]] = None,
                    progress: Progress = None) -> Path:
    """Run one of PROCESSES_EXT on `in_wav` into `out_wav`."""
    eid = _effect_id(effect)
    edef = PROCESSES_EXT.get(eid)
    if edef is None:
        raise ValueError(f"unknown process effect {eid!r}")
    if not edef["available"]:
        raise RuntimeError(f"{edef['name']} is not available on this machine")
    p = _resolve(edef, params)
    in_wav, out_wav = Path(in_wav), Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if eid == "enhance_speech_ai":
        return enhance_ai.enhance(in_wav, out_wav, progress)

    rack = VOICE_PRESETS.get(p["preset"], {}).get("fx_rack", [])
    if not rack:
        return change_voice(in_wav, out_wav, p["pitch"], p["formant"], p["speed"], progress)
    with tempfile.TemporaryDirectory(prefix="voice_changer_") as tmp:
        mid = Path(tmp) / "voice.wav"

        def inner(frac: float, msg: str) -> None:
            if progress:
                progress(0.85 * frac, msg)

        change_voice(in_wav, mid, p["pitch"], p["formant"], p["speed"], inner)
        if progress:
            progress(0.9, f"Voice Changer: {p['preset']} effects")
        _apply_rack(mid, out_wav, rack)
    if progress:
        progress(1.0, "Voice Changer: done")
    return out_wav
