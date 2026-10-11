# Copied from SkinWalker -- B:\youtubeProjects\Buzzcaf_Media\skinWalker\app\segments.py --
# on 2026-10-09 for BuzzEdit's seed-vc CLI (engines/seedvc/cli.py, VOICE_CHANGER_PLAN.md brief S0).
# This is the creator's own region stitcher: the code below the header is the original, unchanged.
# Keep it in step with SkinWalker's copy until S4 retires SkinWalker.

"""Pure-Python timeline splitting and stitching for selective, region-based
audio conversion.

This module has NO dependency on gradio, torch, or any project engine code.
It only requires ``numpy`` (and, optionally, ``librosa`` for high quality
resampling — a numpy linear-interpolation fallback is used when librosa is
not importable).

Manifest schema (input to :func:`parse_manifest`)::

    {
        "duration": 12.5,
        "master_sr": 44100,
        "regions": [
            {"id": "r1", "start": 1.2, "end": 3.4, "kind": "convert",
             "voice": "demon", "preset": "Demon",
             "engine": {...}, "fx": {...}},
            {"id": "r2", "start": 8.0, "end": 9.1, "kind": "cut"}
        ]
    }

Region ``kind`` is either ``"convert"`` or ``"cut"``. Gaps between regions
on the timeline are implicit ``"keep"`` spans (original audio, untouched).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["parse_manifest", "build_spans", "apply_edit"]

# Small epsilon used to avoid spurious zero-length spans caused by floating
# point rounding when comparing timeline positions.
_EPS = 1e-9


def parse_manifest(s: str | dict) -> dict:
    """Parse and normalize an edit manifest.

    Accepts either a JSON string or an already-parsed dict. Returns a dict
    with the shape ``{"duration": float, "master_sr": int, "regions": list}``.

    Normalization performed:
        * ``master_sr`` defaults to 44100 when absent.
        * Regions are sorted by ``start``.
        * Region ``start``/``end`` are clamped to ``[0, duration]``.
        * Regions where ``end <= start`` (after clamping) are dropped.
        * Overlaps are resolved by trimming a later region's ``start`` up to
          the previous (already resolved) region's ``end``. If trimming
          collapses the region (``end <= start``), it is dropped entirely.
    """
    if isinstance(s, str):
        data: dict = json.loads(s)
    elif isinstance(s, dict):
        data = s
    else:
        raise TypeError(f"parse_manifest expects a JSON str or dict, got {type(s)!r}")

    try:
        duration = float(data.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    duration = max(duration, 0.0)

    try:
        master_sr = int(data.get("master_sr", 44100) or 44100)
    except (TypeError, ValueError):
        master_sr = 44100

    raw_regions = data.get("regions") or []

    cleaned: list[dict] = []
    for r in raw_regions:
        if not isinstance(r, dict):
            continue
        try:
            start = float(r.get("start", 0.0))
            end = float(r.get("end", 0.0))
        except (TypeError, ValueError):
            continue

        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        if end <= start:
            continue

        region = dict(r)
        region["start"] = start
        region["end"] = end
        cleaned.append(region)

    cleaned.sort(key=lambda r: r["start"])

    resolved: list[dict] = []
    for region in cleaned:
        if resolved:
            last_end = resolved[-1]["end"]
            if region["start"] < last_end:
                region["start"] = last_end
        if region["end"] <= region["start"]:
            continue
        resolved.append(region)

    return {"duration": duration, "master_sr": master_sr, "regions": resolved}


def build_spans(duration: float, regions: list[dict]) -> list[dict]:
    """Build an ordered, gap-free list of spans covering ``[0, duration]``.

    ``regions`` must already be normalized (sorted, non-overlapping) — as
    produced by :func:`parse_manifest`. Gaps between regions (and before the
    first / after the last region) become ``"keep"`` spans with
    ``region=None``. Each span is a dict::

        {"start": float, "end": float, "kind": "convert"|"cut"|"keep",
         "region": dict | None}
    """
    spans: list[dict] = []
    cursor = 0.0

    for region in regions:
        start = float(region["start"])
        end = float(region["end"])

        if start > cursor + _EPS:
            spans.append({"start": cursor, "end": start, "kind": "keep", "region": None})

        kind = region.get("kind", "convert")
        if kind not in ("convert", "cut"):
            kind = "convert"

        spans.append({"start": start, "end": end, "kind": kind, "region": region})
        cursor = max(cursor, end)

    if duration - cursor > _EPS:
        spans.append({"start": cursor, "end": duration, "kind": "keep", "region": None})

    return spans


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a mono float array from ``sr_in`` to ``sr_out``.

    Uses ``librosa.resample`` when importable, otherwise falls back to a
    numpy linear interpolation resampler. Empty arrays and equal sample
    rates are handled as no-ops.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio.astype(np.float32)
    if sr_in == sr_out or sr_in <= 0 or sr_out <= 0:
        return audio.astype(np.float32)

    try:
        import librosa  # type: ignore

        resampled = librosa.resample(audio.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)
        return np.asarray(resampled, dtype=np.float32)
    except ImportError:
        pass

    n_in = audio.shape[0]
    duration = n_in / float(sr_in)
    n_out = max(1, int(round(duration * sr_out)))
    x_old = np.linspace(0.0, duration, num=n_in, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)


def _equal_power_crossfade(tail: np.ndarray, head: np.ndarray, overlap: int) -> np.ndarray:
    """Mix the last ``overlap`` samples of ``tail`` with the first ``overlap``
    samples of ``head`` using an equal-power (cos/sin) crossfade curve."""
    if overlap <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, np.pi / 2.0, num=overlap, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    return (tail[-overlap:] * fade_out + head[:overlap] * fade_in).astype(np.float32)


def _stitch(chunks: list[np.ndarray], sr: int, xfade_ms: float) -> np.ndarray:
    """Concatenate audio chunks with an equal-power crossfade overlap-add
    between consecutive chunks. The first chunk has no leading fade. If a
    chunk is shorter than the nominal overlap, the overlap is shrunk for
    that seam."""
    chunks = [np.asarray(c, dtype=np.float32) for c in chunks if c is not None and c.size > 0]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].copy()

    nominal_overlap = max(0, int(round(sr * xfade_ms / 1000.0)))
    if nominal_overlap == 0:
        return np.concatenate(chunks).astype(np.float32)

    result = chunks[0].copy()
    for chunk in chunks[1:]:
        overlap = min(nominal_overlap, len(result), len(chunk))
        if overlap <= 0:
            result = np.concatenate([result, chunk])
            continue
        head = result[:-overlap] if overlap < len(result) else result[:0]
        mixed = _equal_power_crossfade(result, chunk, overlap)
        result = np.concatenate([head, mixed, chunk[overlap:]]).astype(np.float32)

    return result


def apply_edit(
    source_audio: np.ndarray,
    sr: int,
    manifest: str | dict,
    convert_fn: Callable[[dict, np.ndarray, int], tuple[np.ndarray, int]],
    master_sr: int = 44100,
    xfade_ms: float = 15.0,
    progress: Optional[Callable[..., Any]] = None,
) -> np.ndarray:
    """Render an edited timeline from ``source_audio`` according to ``manifest``.

    Args:
        source_audio: float32 mono audio at ``sr``.
        sr: sample rate of ``source_audio``.
        manifest: JSON str or dict matching the manifest schema.
        convert_fn: called as ``convert_fn(region, seg, sr) -> (out, out_sr)``
            for each "convert" span. If it raises, the original segment is
            used instead (resampled to ``master_sr``) and rendering continues.
        master_sr: output sample rate.
        xfade_ms: crossfade duration (milliseconds) applied between
            consecutive chunks when stitching the final timeline.
        progress: optional callable invoked as ``progress(fraction, desc=...)``
            after each span is processed. Any exception it raises is
            swallowed (best-effort only).

    Returns:
        float32 mono audio at ``master_sr``, clipped to ``[-1, 1]``.
    """
    parsed = parse_manifest(manifest)
    duration = parsed["duration"]
    if not duration or duration <= 0:
        duration = (len(source_audio) / sr) if sr else 0.0

    spans = build_spans(duration, parsed["regions"])
    n = len(spans)

    chunks: list[np.ndarray] = []
    for i, span in enumerate(spans):
        kind = span["kind"]
        start_idx = int(span["start"] * sr)
        end_idx = int(span["end"] * sr)
        seg = source_audio[start_idx:end_idx]

        if kind == "cut":
            chunk = None
        elif kind == "convert":
            region = span["region"]
            try:
                out, out_sr = convert_fn(region, seg, sr)
                chunk = _resample(np.asarray(out, dtype=np.float32), int(out_sr), master_sr)
            except Exception:
                chunk = _resample(seg, sr, master_sr)
        else:  # "keep"
            chunk = _resample(seg, sr, master_sr)

        if chunk is not None and chunk.size > 0:
            chunks.append(chunk)

        if progress is not None:
            try:
                progress((i + 1) / n if n else 1.0, desc=f"Rendering span {i + 1}/{n} ({kind})")
            except Exception:
                pass

    result = _stitch(chunks, master_sr, xfade_ms)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    # Self-test: no external audio files required.
    sr = 44100
    total_seconds = 5.0
    t = np.linspace(0.0, total_seconds, num=int(sr * total_seconds), endpoint=False, dtype=np.float32)
    sine = (0.5 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    def identity_convert(region: dict, seg: np.ndarray, seg_sr: int) -> tuple[np.ndarray, int]:
        return seg, seg_sr

    manifest_with_cut = {
        "duration": total_seconds,
        "master_sr": sr,
        "regions": [
            {"id": "r1", "start": 1.0, "end": 2.0, "kind": "convert"},
            {"id": "r2", "start": 3.0, "end": 3.5, "kind": "cut"},
        ],
    }

    out1 = apply_edit(sine, sr, manifest_with_cut, identity_convert, master_sr=sr)
    expected_len = (total_seconds - 0.5) * sr
    tolerance = sr * 0.05  # 50ms tolerance for crossfade rounding
    assert abs(len(out1) - expected_len) <= tolerance, (
        f"expected ~{expected_len} samples, got {len(out1)}"
    )

    manifest_keep_only = {"duration": total_seconds, "master_sr": sr, "regions": []}
    out2 = apply_edit(sine, sr, manifest_keep_only, identity_convert, master_sr=sr)
    expected_keep_len = total_seconds * sr
    assert abs(len(out2) - expected_keep_len) <= tolerance, (
        f"expected ~{expected_keep_len} samples, got {len(out2)}"
    )

    print("segments self-test OK")
