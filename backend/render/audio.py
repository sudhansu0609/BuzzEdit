"""Audio treatment filters: voice clean-up, ducking, and the loudness target.

Three separate builders because they sit at three different points in the
graph. The voice chain runs on the programme audio alone, before the mix, so
nothing it does reaches the music. Ducking is a sidechain compressor on ONE
mix-track item, keyed off the voice, so a music bed drops under speech and
comes back in the pauses without any guessed envelope. The loudness chain runs
last, on the finished mix, because -14 LUFS is a property of the whole
soundtrack and not of any one stem.

Every value maps from a 0..1 strength rather than exposing the filter's raw
options: a caller (the presentation pass, a UI slider) says "a little" or "a
lot", and the numbers that actually sound right live here in one place.

**Why the voice used to sound boxy and hissy** (see the presentation report's
`voice_enhance_used` / `voice_preset` fields for what a given render actually
did): `afftdn` scaled all the way to 24dB of reduction strips the high-frequency
"air" off a voice and leaves it sounding like it was recorded in a box, and
nothing after it ever put an EQ curve back in. `build_voice_chain` now treats
denoise as a *choice of engine* (`resolve_voice_enhance`) capped gently when it
falls back to FFmpeg's own filter, and adds a proper EQ/compression/limiter
chain per named preset (`EQ_PRESETS`) so a voice preset shapes the tone, not
just the noise floor.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR, FFMPEG_BIN
from utils.proc import NO_WINDOW
from timeline.schema import AudioMaster

logger = logging.getLogger("render.audio")

RNNOISE_MODEL_DIR = Path(DATA_DIR) / "models" / "rnnoise"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


# --- denoise engine selection -----------------------------------------------
#
# "auto" tries the best speech-trained option first and only reaches for
# FFmpeg's generic spectral denoiser (afftdn) when nothing better is on the
# machine. Nothing here ever raises: an engine that is not available is simply
# not chosen, which is what keeps a fresh install working without any of these
# extras installed.

def deepfilter_available() -> bool:
    """Whether the optional `df` (DeepFilterNet) package can be imported.

    Not installed by this app, and — on this machine — not installable: the
    `df` package pins a `torchaudio` older than the one the rest of the stack
    needs, so `import df` will keep failing and `find_spec` will keep
    returning None. That makes this check, and the "deepfilter" branch of
    `resolve_voice_enhance` it feeds, effectively dead in practice; both stay
    in place (harmlessly reporting "unavailable") rather than being ripped
    out, so a machine where the conflict is someday resolved picks it back up
    for free. In the meantime `resolve_voice_enhance("auto")`'s real first
    pick is RNNoise, not this.
    """
    try:
        import importlib.util
        return importlib.util.find_spec("df") is not None
    except Exception:
        return False


def rnnoise_model_path() -> Optional[Path]:
    """The first `.rnnn` RNNoise model under data/models/rnnoise/, if any."""
    if not RNNOISE_MODEL_DIR.exists():
        return None
    try:
        models = sorted(RNNOISE_MODEL_DIR.glob("*.rnnn"))
    except Exception:
        return None
    return models[0] if models else None


def _ffmpeg_filter_path(path: Path) -> str:
    """A filesystem path safe to embed as an ffmpeg filter option value.

    ffmpeg's filtergraph parser splits an option string on `:` to separate
    key=value pairs, and it does so BEFORE the surrounding quotes are
    resolved — so `arnndn=m='B:/data/model.rnnn'` fails to parse with "No
    option name near '/data/model.rnnn'": the colon after the Windows drive
    letter is read as the next option's separator, quotes notwithstanding.
    Escaping that colon with a backslash inside the quotes (`'B\\:/data/...'`)
    is what ffmpeg actually accepts. Verified against a real ffmpeg process
    on a `B:\\...` path, not just read off the docs.
    """
    return path.as_posix().replace(":", "\\:")


def resolve_voice_enhance(mode: Optional[str]) -> str:
    """The denoise engine that will actually run for a requested `voice_enhance`.

    Returns one of "deepfilter", "rnnoise", "ffmpeg", "off" — always something
    usable, because "ffmpeg" (a gentle afftdn) needs nothing beyond the FFmpeg
    binary the whole render already depends on.
    """
    mode = (mode or "auto").lower()
    if mode == "off":
        return "off"
    if mode == "deepfilter":
        return "deepfilter" if deepfilter_available() else "ffmpeg"
    if mode == "rnnoise":
        return "rnnoise" if rnnoise_model_path() is not None else "ffmpeg"
    if mode == "ffmpeg":
        return "ffmpeg"
    # "auto", or anything unrecognised: the best thing actually on this machine.
    # The check order is deepfilter, then rnnoise, then ffmpeg, but see
    # `deepfilter_available`'s docstring — the first slot is a no-op on a
    # machine where `df` cannot be installed, so in practice "auto" here
    # means RNNoise when its model is present, ffmpeg's afftdn otherwise.
    if deepfilter_available():
        return "deepfilter"
    if rnnoise_model_path() is not None:
        return "rnnoise"
    return "ffmpeg"


def enhance_with_deepfilter(input_wav: str, output_wav: str) -> bool:
    """Run DeepFilterNet over a WAV file as a standalone pre-pass.

    Best-effort: any failure (package missing, model download failed, a
    corrupt WAV) logs and returns False rather than raising, so a caller can
    fall back to the in-graph denoiser exactly as if "deepfilter" had never
    been available. Not wired into the render graph itself — FFmpeg's
    filtergraph has no way to call an external neural model mid-stream, so
    this is meant to be run over an already-extracted voice track and the
    result fed back in as a new input.
    """
    if not deepfilter_available():
        return False
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio  # type: ignore
        model, df_state, _ = init_df()
        audio, _ = load_audio(input_wav, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(output_wav, enhanced, df_state.sr())
        return Path(output_wav).exists()
    except Exception as e:
        logger.warning("DeepFilterNet pre-pass failed (%s); falling back", e)
        return False


def _denoise_stage(master: AudioMaster) -> List[str]:
    """The one filter (or nothing) that removes noise from the voice.

    `voice_denoise` still sets the strength, but the ceiling is now the
    engine's, not a straight line to 24dB: afftdn beyond ~6dB of reduction is
    what stripped the air off the voice and made it sound "in a box", so the
    FFmpeg fallback is capped there. rnnoise and deepfilter are trained on
    speech and do not have that trade-off, so they are preferred whenever
    available.
    """
    denoise = _clamp(master.voice_denoise)
    if denoise <= 0.0:
        return []
    engine = resolve_voice_enhance(master.voice_enhance)
    if engine == "off":
        return []
    if engine == "deepfilter":
        # The heavy lifting already happened in the pre-pass that produced
        # this audio (see enhance_with_deepfilter); nothing left to do here.
        return []
    if engine == "rnnoise":
        model = rnnoise_model_path()
        if model is not None:
            # mix is 0..1, how much of the denoised signal to use; strength
            # scales it rather than an on/off switch.
            mix = 0.5 + 0.5 * denoise
            return [f"arnndn=m='{_ffmpeg_filter_path(model)}':mix={mix:.2f}"]
        engine = "ffmpeg"  # model vanished between resolve() and here
    # "ffmpeg": nr is the reduction in dB, capped well short of where afftdn
    # starts eating the voice's own high end. nf is the assumed noise floor;
    # tracking (tn) lets it follow a floor that drifts across a long take.
    nr = 1.5 + 4.5 * denoise
    return [f"afftdn=nr={min(6.0, nr):.1f}:nf=-45:tn=1"]


# --- EQ / character presets -------------------------------------------------
#
# Named curves, not raw filter strings, so `build_voice_chain` stays the only
# place that turns a preset into FFmpeg syntax. `bands` are (freq_hz, gain_db,
# q) triples for the parametric `equalizer` filter; `shelf` is (freq_hz,
# gain_db) for a high-shelf brightening the air band. `comp_ratio` overrides
# the generic 0..1 strength curve with the preset's own broadcast-style ratio,
# and `limiter_tp` is the true-peak ceiling `alimiter` is not allowed to cross.
EQ_PRESETS: Dict[str, Dict[str, Any]] = {
    "studio_mic": {
        "hpf": 80, "bands": [(300, -3.0, 1.0), (4000, 3.0, 1.0)],
        "shelf": (11000, 2.0), "comp_ratio": 3.0, "comp_threshold_db": -20.0,
        "limiter_tp": -1.0,
    },
    "broadcast": {
        "hpf": 90, "bands": [(3000, 2.0, 1.0)], "shelf": None,
        "comp_ratio": 4.0, "comp_threshold_db": -22.0, "limiter_tp": -1.0,
    },
    "warm_radio": {
        "hpf": 100, "bands": [(200, 2.0, 1.0)], "shelf": None, "lpf": 9000,
        "comp_ratio": 3.0, "comp_threshold_db": -20.0, "limiter_tp": -1.0,
    },
    "rap_vocal": {
        "hpf": 100, "bands": [(350, -3.0, 1.0), (5000, 4.0, 1.0)],
        "shelf": (12000, 3.0), "comp_ratio": 5.0, "comp_threshold_db": -24.0,
        "parallel_compress": True, "saturation": 0.35, "limiter_tp": -1.0,
    },
    "horror_intimate": {
        "hpf": 60, "bands": [(250, 1.0, 1.0)], "shelf": None,
        "comp_ratio": 3.0, "comp_threshold_db": -20.0, "reverb": "dark_room",
        "limiter_tp": -1.0,
    },
    # The plain presets keep their old names and intent (light clean-up) but
    # now carry the studio_mic curve at a reduced strength rather than no EQ
    # at all, and lean on the capped afftdn from `_denoise_stage` instead of
    # the old uncapped one.
    "clean": {
        "hpf": 70, "bands": [(300, -1.5, 1.0), (4000, 1.5, 1.0)],
        "shelf": (11000, 1.0), "comp_ratio": 2.5, "comp_threshold_db": -18.0,
        "limiter_tp": -1.0,
    },
    "podcast": {
        "hpf": 70, "bands": [(300, -1.5, 1.0), (4000, 1.5, 1.0)],
        "shelf": (11000, 1.0), "comp_ratio": 3.0, "comp_threshold_db": -18.0,
        "limiter_tp": -1.0,
    },
    "light": {
        "hpf": 60, "bands": [(300, -1.0, 1.0), (4000, 1.0, 1.0)],
        "shelf": (11000, 0.5), "comp_ratio": 2.0, "comp_threshold_db": -16.0,
        "limiter_tp": -1.0,
    },
}

# Windowed reverb "rooms", shared by the whole-voice `voice_reverb` (a preset
# character, e.g. horror_intimate) and the per-line `voice_fx` "reverb_room"/
# "reverb_hall" effects. `aecho=in_gain:out_gain:delays:decays`.
_REVERB_RECIPES: Dict[str, str] = {
    "dark_room": "aecho=0.85:0.55:35|55:0.22|0.12",
    "hall": "aecho=0.85:0.7:400|600|800:0.35|0.25|0.15",
}


def _eq_stage(preset: Dict[str, Any]) -> List[str]:
    chain: List[str] = []
    hpf = preset.get("hpf")
    if hpf:
        chain.append(f"highpass=f={int(hpf)}")
    for freq, gain_db, q in preset.get("bands", []):
        if abs(gain_db) < 0.05:
            continue
        chain.append(f"equalizer=f={int(freq)}:width_type=q:width={q:.2f}:g={gain_db:.1f}")
    shelf = preset.get("shelf")
    if shelf:
        freq, gain_db = shelf
        chain.append(f"treble=g={gain_db:.1f}:f={int(freq)}:width_type=o:width=0.7")
    lpf = preset.get("lpf")
    if lpf:
        chain.append(f"lowpass=f={int(lpf)}")
    return chain


def _compress_stage(preset: Optional[Dict[str, Any]], strength: float) -> List[str]:
    """Broadcast-style compression: a named preset sets its own ratio/threshold;
    with no preset the old 0..1 strength curve still applies unchanged."""
    strength = _clamp(strength)
    if strength <= 0.0 and not (preset or {}).get("parallel_compress"):
        return []
    if preset and "comp_ratio" in preset:
        threshold_db = float(preset.get("comp_threshold_db", -20.0))
        ratio = float(preset["comp_ratio"])
        makeup = 1.0 + 0.8 * strength
    else:
        threshold_db = -14.0 - 8.0 * strength
        ratio = 2.0 + 2.5 * strength
        makeup = 1.0 + 1.2 * strength
    main = (f"acompressor=threshold={threshold_db:.1f}dB:ratio={ratio:.2f}:"
            f"attack=8:release=140:knee=4:makeup={makeup:.2f}")
    if not (preset or {}).get("parallel_compress"):
        return [main]
    # Parallel ("New York") compression: a much harder, faster compressed copy
    # blended back under the lightly-compressed main signal via `sidechaincompress`
    # would need a second stream, which this single-chain builder cannot split
    # and rejoin; `acompressor` itself supports a wet/dry `mix`, which gets the
    # same effect (density without losing all the original dynamics) in one
    # filter instead.
    return [main, f"acompressor=threshold=-32dB:ratio=8:attack=2:release=100:"
                  f"knee=1:makeup=1.5:mix=0.35"]


def _saturation_stage(amount: float) -> List[str]:
    amount = _clamp(amount)
    if amount <= 0.0:
        return []
    # aexciter adds harmonic content above a blend frequency; amount maps to
    # how much of the excited signal is blended back in.
    return [f"aexciter=level_in=1:level_out=1:amount={1.0 + 3.0 * amount:.2f}:"
            f"drive={2.0 + 6.0 * amount:.2f}:blend=0:freq=7500:ceil=9500"]


def _limiter_stage(preset: Optional[Dict[str, Any]]) -> List[str]:
    tp = float((preset or {}).get("limiter_tp", -1.0))
    return [f"alimiter=limit={tp:.1f}dB:attack=5:release=50"]


# --- line effects (voice_fx) -------------------------------------------------
#
# Each one is a short chain of filters, windowed with `enable='between(t,a,b)'`
# so it only touches the seconds it was asked for and the chain is silently a
# no-op everywhere else — no need to split and re-concat the audio.
VOICE_FX_RECIPES: Dict[str, List[str]] = {
    "phone": ["highpass=f=300", "lowpass=f=3400", "acrusher=bits=8:mode=log:aa=1:samples=2"],
    "megaphone": ["highpass=f=500", "lowpass=f=2500",
                  "acrusher=bits=6:mode=log:aa=1:samples=3", "volume=1.4"],
    "echo_tail": ["aecho=0.8:0.7:180:0.35"],
    "reverb_room": [_REVERB_RECIPES["dark_room"]],
    "reverb_hall": [_REVERB_RECIPES["hall"]],
}


def _with_enable(filters: List[str], start_s: float, end_s: float) -> List[str]:
    window = f"enable='between(t,{max(0.0, start_s):.3f},{max(0.0, end_s):.3f})'"
    return [f"{f}:{window}" for f in filters]


def build_voice_fx_chain(fx_cues: Optional[List[Dict[str, Any]]]) -> List[str]:
    """One windowed effect per cue, in the order they were given."""
    chain: List[str] = []
    for cue in (fx_cues or []):
        effect = str(cue.get("effect") or "")
        recipe = VOICE_FX_RECIPES.get(effect)
        if not recipe:
            continue
        start_s = float(cue.get("start_s", 0.0))
        end_s = float(cue.get("end_s", start_s))
        if end_s <= start_s:
            continue
        chain.extend(_with_enable(recipe, start_s, end_s))
    return chain


def build_voice_chain(master: Optional[AudioMaster]) -> List[str]:
    """Denoise → EQ → saturation → de-ess → compress → reverb → limiter → line fx.

    With no `voice_eq_preset` this is exactly the old chain (denoise/de-ess/
    compress from the 0..1 strengths), so an old, EQ-less master still compiles
    to what it always did.
    """
    if master is None:
        return []
    chain: List[str] = list(_denoise_stage(master))
    if master.voice_gain_db != 0.0:
        # Prepended: a flat gain trim on the raw voice, before anything that
        # reacts to level (denoise floor, compressor, de-esser) sees it.
        chain.insert(0, f"volume={master.voice_gain_db:.2f}dB")

    preset = EQ_PRESETS.get((master.voice_eq_preset or "").lower())
    if preset:
        chain.extend(_eq_stage(preset))
        saturation = max(master.voice_saturation, float(preset.get("saturation", 0.0)))
    else:
        saturation = master.voice_saturation
    chain.extend(_saturation_stage(saturation))

    deess = _clamp(master.voice_deess)
    if deess > 0.0:
        chain.append(f"deesser=i={0.15 + 0.6 * deess:.2f}:m=0.5:f=0.5")

    chain.extend(_compress_stage(preset, master.voice_compress))

    reverb_kind = master.voice_reverb or (preset or {}).get("reverb", "")
    if reverb_kind:
        recipe = _REVERB_RECIPES.get(reverb_kind)
        if recipe:
            chain.append(recipe)

    if preset and (master.voice_compress > 0.0 or preset.get("parallel_compress")):
        chain.extend(_limiter_stage(preset))

    chain.extend(build_voice_fx_chain(master.voice_fx))
    return chain


def build_ducking(amount: float) -> str:
    """The sidechain compressor that pushes one mix item under the voice.

    Filter takes `[item][voice]` and yields the ducked item. Threshold sits far
    below normal speech level so any speaking at all triggers it; the ratio is
    what sets the depth — at full strength a voice at -20 dBFS pulls the bed
    down by roughly 12 dB. Attack is fast enough that the first syllable is
    not stepped on; release long enough that the bed does not flutter between
    words.
    """
    strength = _clamp(amount)
    ratio = 1.0 + 11.0 * strength
    return (f"sidechaincompress=threshold=0.02:ratio={ratio:.2f}:attack=25:"
            f"release=450:knee=6:level_sc=1:makeup=1")


def build_loudness_chain(master: Optional[AudioMaster],
                         measured: Optional[Dict[str, float]] = None) -> List[str]:
    """EBU R128 normalisation to the target, then back to the graph's sample rate.

    loudnorm resamples internally and emits 192 kHz; without the aresample the
    encoder would be handed a rate the rest of the graph never agreed to.

    `measured` is the first pass's stats (see `measure_loudness_stats`): with
    it, loudnorm runs in linear mode against the ACTUAL measured loudness
    instead of guessing frame-by-frame, which is what stops a talky programme
    from audibly pumping. Without it (no stats, or the measuring pass failed)
    this is exactly the single-pass filter the module always used.
    """
    if master is None or master.loudness_lufs is None:
        return []
    target = max(-40.0, min(-5.0, float(master.loudness_lufs)))
    peak = max(-9.0, min(0.0, float(master.true_peak_db)))
    if measured:
        try:
            return [
                f"loudnorm=I={target:.1f}:TP={peak:.1f}:LRA=11:"
                f"measured_I={float(measured['input_i']):.2f}:"
                f"measured_TP={float(measured['input_tp']):.2f}:"
                f"measured_LRA={float(measured['input_lra']):.2f}:"
                f"measured_thresh={float(measured['input_thresh']):.2f}:"
                f"offset={float(measured.get('target_offset', 0.0)):.2f}:linear=true",
                "aresample=48000",
            ]
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Bad loudness measurement %r (%s); using single-pass", measured, e)
    return [f"loudnorm=I={target:.1f}:TP={peak:.1f}:LRA=11", "aresample=48000"]


_TRAILING_LABELS = re.compile(r"((?:\[[^\]]+\])+)\s*$")
_LABEL = re.compile(r"\[([^\]]+)\]")


def audio_subgraph(filter_complex: str, label: str) -> str:
    """The statements `label` depends on, in their original order.

    Walks back from `label` through each statement's input labels to the
    inputs (`[n:a]`) or sources (`anullsrc`, `amovie`) it starts from, so the
    loudness pass runs the audio mix alone instead of the whole video graph.
    Statements that also produce video (a `split` shared with a picture chain,
    say) keep their extra outputs, which are sunk so FFmpeg accepts the graph.
    """
    statements = [s.strip() for s in filter_complex.split(";") if s.strip()]
    producer: Dict[str, int] = {}
    for index, statement in enumerate(statements):
        tail = _TRAILING_LABELS.search(statement)
        if tail:
            for out in _LABEL.findall(tail.group(1)):
                producer[out] = index
    wanted, keep = [label.strip("[]")], set()
    while wanted:
        index = producer.get(wanted.pop())
        if index is None or index in keep:
            continue
        keep.add(index)
        statement = statements[index]
        tail = _TRAILING_LABELS.search(statement)
        body = statement[:tail.start()] if tail else statement
        wanted.extend(x for x in _LABEL.findall(body) if ":" not in x)
    kept = [statements[i] for i in sorted(keep)]
    consumed, produced = set(), set()
    for statement in kept:
        tail = _TRAILING_LABELS.search(statement)
        body = statement[:tail.start()] if tail else statement
        consumed.update(x for x in _LABEL.findall(body) if ":" not in x)
        if tail:
            produced.update(_LABEL.findall(tail.group(1)))
    for spare in sorted(produced - consumed - {label.strip("[]")}):
        kept.append(f"[{spare}]{'anullsink' if spare.startswith('a') else 'nullsink'}")
    return ";".join(kept)


def measure_loudness_stats(ffmpeg_bin: str, inputs: List[str], filter_complex: str,
                           audio_label: str, target_i: float, target_tp: float,
                           timeout: int = 300) -> Optional[Dict[str, float]]:
    """First loudnorm pass: measure `audio_label` and return its stats, or None.

    Runs the SAME inputs/graph the real render already built, plus one more
    filter mapped alone to a null muxer — nothing is written to disk, and
    nothing about the real render's inputs or graph is touched. `None` on any
    failure (ffmpeg missing, a bad graph, a timeout) is the signal the caller
    falls back to single-pass loudnorm on, exactly as if two-pass had never
    been attempted.
    """
    measure_label = "[loud_measure]"
    extra = (f"{audio_label}loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA=11:"
             f"print_format=json{measure_label}")
    # Only the statements the audio depends on: with the whole graph every
    # video branch ran too, so on a real edit this pass never finished inside
    # its timeout and two-pass silently fell back to single-pass every time.
    audio_graph = audio_subgraph(filter_complex, audio_label) if filter_complex else ""
    graph = f"{audio_graph};{extra}" if audio_graph else extra
    from render.cmdline import prune_inputs
    inputs, graph, _ = prune_inputs(inputs, graph)
    script_path: Optional[str] = None
    try:
        fd, script_path = tempfile.mkstemp(suffix=".txt", prefix="loudmeasure_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(graph)
        from render.cmdline import fit_command
        command = [ffmpeg_bin, "-y", "-hide_banner", *inputs,
                   "-filter_complex_script", script_path,
                   "-map", measure_label, "-f", "null", "-"]
        with fit_command(command) as (command, cwd):
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=timeout, cwd=cwd, creationflags=NO_WINDOW)
        stderr = (result.stderr or b"").decode("utf-8", "replace")
        match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr)
        if not match:
            logger.warning("Loudness measurement pass produced no stats; using single-pass")
            return None
        stats = json.loads(match.group(0))
        return {
            "input_i": float(stats["input_i"]),
            "input_tp": float(stats["input_tp"]),
            "input_lra": float(stats["input_lra"]),
            "input_thresh": float(stats["input_thresh"]),
            "target_offset": float(stats.get("target_offset", 0.0)),
        }
    except Exception as e:
        logger.warning("Loudness measurement pass failed (%s); using single-pass loudnorm", e)
        return None
    finally:
        if script_path:
            try:
                os.remove(script_path)
            except Exception:
                pass
