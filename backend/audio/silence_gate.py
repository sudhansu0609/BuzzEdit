"""Re-impose the Chatterbox source's silences onto a seed-vc conversion (VOICE_DELIVERY_PLAN.md
brief F8): seed-vc leaves a ~-70 dB floor where the source was truly silent, not real silence, so
every run of real quiet in the source is zeroed again in the converted file -- never a separate
time-stretch pass, which is why this is a gate on the conversion, not a second pace change.

10 ms RMS frames at the source's own sample rate; a frame is quiet when it is `QUIET_DROP_DB` under
the `QUIET_PERCENTILE`th-percentile loudness of the whole clip (the Chatterbox take's own speech
level, not a fixed floor). A run of quiet frames shorter than `MIN_RUN_S` is a breath inside speech,
left alone. A qualifying run's time span (`t0`, `t1`, in the source's own seconds) is scaled by
`length_adjust` -- seed-vc's own `--params length_adjust` stretches/compresses the whole file
uniformly, so a source time `t` lands at `t * length_adjust` in the converted file -- and its edges
pulled in by `MARGIN_S` so a word's own attack/decay right at the edge of the pause is never
clipped, except at the very start or end of the source, where there is nothing to protect and the
margin is dropped. `FADE_S` linear fades in/out of each zeroed span, so there is no click.

Ported from the measured prototype (scratchpad `voice/gate.py`, VOICE_DELIVERY_PLAN.md F7).
"""
from __future__ import annotations

from typing import Any

FRAME_S = 0.01
QUIET_PERCENTILE = 90.0
QUIET_DROP_DB = 35.0
MIN_RUN_S = 0.12
MARGIN_S = 0.03
FADE_S = 0.015


def gate(source: Any, source_sr: int, converted: Any, converted_sr: int, length_adjust: float) -> Any:
    """`converted` (seed-vc's output, mono float) with every silent run of `source` (mono float,
    its own sample rate) zeroed over the matching span, scaled by `length_adjust`. Pure numpy, no
    file I/O (`gate_files` below is the WAV-in/WAV-out wrapper); `converted` is never modified in
    place."""
    import numpy as np
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    out = np.array(converted, dtype=np.float32).reshape(-1)
    hop = max(1, int(round(source_sr * FRAME_S)))
    n = source.size // hop
    if n < 3:
        return out                                            # too short to measure a silent run

    frames = source[:n * hop].reshape(n, hop).astype(np.float64)
    db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    threshold = np.percentile(db, QUIET_PERCENTILE) - QUIET_DROP_DB
    quiet = db < threshold

    gain = np.ones(out.size, dtype=np.float32)
    fade = int(round(converted_sr * FADE_S))
    run_start = None
    for i, is_quiet in enumerate(list(quiet) + [False]):       # a trailing False closes a run at n
        if is_quiet and run_start is None:
            run_start = i
        if not is_quiet and run_start is not None:
            t0, t1 = run_start * FRAME_S, i * FRAME_S
            if t1 - t0 >= MIN_RUN_S:
                a = int(round((t0 * length_adjust + (MARGIN_S if run_start else 0.0)) * converted_sr))
                b = int(round((t1 * length_adjust - (MARGIN_S if i < n else 0.0)) * converted_sr))
                a, b = max(0, a), min(b, out.size)
                if b - a > 2 * fade:
                    gain[a:b] = 0.0
                    if fade > 0:
                        gain[a:a + fade] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
                        gain[b - fade:b] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            run_start = None
    return out * gain


def gate_files(source_path: str, converted_path: str, length_adjust: float, out_path: str) -> str:
    """`gate()` over two WAV files (either may have more than one channel; both are mixed to
    mono for the measurement and the output), written to `out_path` as 16-bit PCM at the
    converted file's own sample rate."""
    import soundfile as sf
    source, source_sr = sf.read(source_path, dtype="float32", always_2d=True)
    converted, converted_sr = sf.read(converted_path, dtype="float32", always_2d=True)
    gated = gate(source.mean(axis=1), source_sr, converted.mean(axis=1), converted_sr, length_adjust)
    sf.write(out_path, gated, converted_sr, subtype="PCM_16")
    return out_path
