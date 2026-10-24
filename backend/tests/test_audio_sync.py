"""backend/sync/audio_sync.py on synthetic recordings (MULTICAM_SYNC_PLAN.md §1).

A 30 s master (irregular white-noise bursts over a sine) is cut into two 20 s
"recordings" of the same event. The second is shifted, low-passed (another
microphone) and 10 dB quieter: other_time = ref_time + 3.370.
"""

import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from config import FFMPEG_BIN
from sync.audio_sync import (MIN_CONFIDENCE, SyncResult, decode_mono, find_offset,
                             write_aligned_wav)

OFFSET = 3.370


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-600:]


def _master(path: Path, seed: int, sine_hz: int) -> Path:
    rng = random.Random(seed)
    bursts, t = [], 0.3
    while t < 29.5:
        length = rng.uniform(0.04, 0.22)
        bursts.append((t, t + length))
        t += length + rng.uniform(0.15, 1.1)
    gate = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in bursts)
    graph = (f"[1:a]volume=volume=0:enable='not({gate})'[bursts];"
             f"[0:a][bursts]amix=inputs=2:normalize=0:duration=shortest")
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency={sine_hz}:sample_rate=48000:duration=30",
            "-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude=0.5:sample_rate=48000"
                                 f":duration=30:seed={seed}",
            "-filter_complex", graph, "-c:a", "pcm_s16le", str(path))
    return path


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sync")
    master = _master(folder / "master.wav", seed=5, sine_hz=330)
    ref = folder / "ref.wav"
    other = folder / "other.wav"
    drifted = folder / "drifted.wav"
    unrelated = folder / "unrelated.wav"
    # ref starts OFFSET into the event, other at its start: other_time = ref_time + OFFSET.
    _ffmpeg("-ss", f"{OFFSET}", "-t", "20", "-i", str(master), "-c:a", "pcm_s16le", str(ref))
    _ffmpeg("-t", "20", "-i", str(master), "-af", "lowpass=f=1800,volume=-10dB",
            "-c:a", "pcm_s16le", str(other))
    # The drift copy is 90 s: atempo retimes in WSOLA steps of a few ms, so on 20 s
    # its actual retiming wanders +-25% around 1.0005 (measured 378..584 ppm);
    # over 90 s the steps average out to within a few percent.
    long_take = folder / "long_take.wav"
    parts = [_master(folder / f"part{k}.wav", seed=20 + k, sine_hz=220 + 40 * k) for k in range(3)]
    _ffmpeg(*[arg for part in parts for arg in ("-i", str(part))],
            "-filter_complex", "concat=n=3:v=0:a=1", "-c:a", "pcm_s16le", str(long_take))
    _ffmpeg("-i", str(long_take), "-af", "atempo=1.0005", "-c:a", "pcm_s16le", str(drifted))
    _master(folder / "unrelated_master.wav", seed=99, sine_hz=180)
    _ffmpeg("-t", "20", "-i", str(folder / "unrelated_master.wav"), "-c:a", "pcm_s16le",
            str(unrelated))
    return {"ref": ref, "other": other, "drifted": drifted, "long_take": long_take, "unrelated": unrelated,
            "folder": folder}


def test_decode_mono_window(recordings):
    pcm = decode_mono(str(recordings["ref"]), sr=8000, start_s=2.0, dur_s=1.5)
    assert pcm.dtype == np.float32
    assert abs(pcm.size - 12000) <= 8


def test_offset_survives_eq_and_gain(recordings):
    result = find_offset(str(recordings["ref"]), str(recordings["other"]))
    assert abs(result.offset_s - OFFSET) < 0.005, result
    assert result.confidence > MIN_CONFIDENCE
    assert abs(result.drift_ppm) < 20


def test_offset_sign_is_other_minus_ref(recordings):
    result = find_offset(str(recordings["other"]), str(recordings["ref"]))
    assert abs(result.offset_s + OFFSET) < 0.005, result


def test_drift_of_a_retimed_copy(recordings):
    # atempo 1.0005 plays 0.05% fast: other_time = ref_time / 1.0005, i.e. -500 ppm.
    # The +-15% bound is the brief's; 500 * 0.15 = 75.
    result = find_offset(str(recordings["long_take"]), str(recordings["drifted"]))
    assert result.confidence > MIN_CONFIDENCE
    assert abs(abs(result.drift_ppm) - 500) < 75, result
    assert result.drift_ppm < 0


def test_unrelated_audio_is_unreliable(recordings):
    result = find_offset(str(recordings["ref"]), str(recordings["unrelated"]))
    assert result.confidence < MIN_CONFIDENCE, result
    assert not result.reliable


def test_aligned_wav_has_the_reference_length_and_lines_up(recordings):
    out = recordings["folder"] / "aligned.wav"
    result = find_offset(str(recordings["ref"]), str(recordings["other"]))
    write_aligned_wav(str(recordings["other"]), result, 20.0, str(out))
    pcm = decode_mono(str(out), sr=48000)
    assert abs(pcm.size - 20 * 48000) <= 2
    # Laid on the reference clock it matches the reference at zero lag.
    check = find_offset(str(recordings["ref"]), str(out))
    assert abs(check.offset_s) < 0.003, check


def test_aligned_wav_pads_a_late_recorder(recordings):
    out = recordings["folder"] / "aligned_neg.wav"
    result = SyncResult(offset_s=-1.0, confidence=5.0, drift_ppm=0.0)
    write_aligned_wav(str(recordings["ref"]), result, 10.0, str(out))
    pcm = decode_mono(str(out), sr=8000)
    assert abs(pcm.size - 80000) <= 2
    assert np.abs(pcm[:7900]).max() == 0.0          # a second of silence first
