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
"""

from typing import List, Optional

from timeline.schema import AudioMaster


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def build_voice_chain(master: Optional[AudioMaster]) -> List[str]:
    """Denoise → de-ess → compress on the programme voice."""
    if master is None:
        return []
    chain: List[str] = []
    denoise = _clamp(master.voice_denoise)
    if denoise > 0.0:
        # nr is the reduction in dB; nf the assumed noise floor. Tracking (tn)
        # lets the filter follow a floor that changes across a long recording.
        chain.append(f"afftdn=nr={6 + 18 * denoise:.1f}:nf=-45:tn=1")
    deess = _clamp(master.voice_deess)
    if deess > 0.0:
        chain.append(f"deesser=i={0.15 + 0.6 * deess:.2f}:m=0.5:f=0.5")
    compress = _clamp(master.voice_compress)
    if compress > 0.0:
        # Broadcast-style: a moderate ratio over a knee, quick attack so
        # plosives do not punch through, and enough makeup that the average
        # level rises — which is the point of compressing speech.
        threshold_db = -14.0 - 8.0 * compress
        ratio = 2.0 + 2.5 * compress
        makeup = 1.0 + 1.2 * compress
        chain.append(
            f"acompressor=threshold={threshold_db:.1f}dB:ratio={ratio:.2f}:"
            f"attack=8:release=140:knee=4:makeup={makeup:.2f}")
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


def build_loudness_chain(master: Optional[AudioMaster]) -> List[str]:
    """EBU R128 normalisation to the target, then back to the graph's sample rate.

    loudnorm resamples internally and emits 192 kHz; without the aresample the
    encoder would be handed a rate the rest of the graph never agreed to.
    """
    if master is None or master.loudness_lufs is None:
        return []
    target = max(-40.0, min(-5.0, float(master.loudness_lufs)))
    peak = max(-9.0, min(0.0, float(master.true_peak_db)))
    return [f"loudnorm=I={target:.1f}:TP={peak:.1f}:LRA=11", "aresample=48000"]
