"""Pass 2: NVDEC -> eye warp (CUDA graph) -> NVENC, then mux with the source's own timestamps.

Two ordering rules, both learned from corrupted frames:
  * PyNvVideoCodec's threaded decoder recycles its surfaces as soon as the next batch
    is requested, so each batch is copied out and the copies allowed to land first;
  * NVENC reads its input on its own stream, unordered with torch's, so torch's queue
    is drained before every Encode (otherwise it can grab a frame before the warp —
    or the copy into it — has finished, and encode a stale one).
"""
import logging
import os
import subprocess
from fractions import Fraction
from typing import Callable, Optional

import numpy as np
import torch

from .gpu import VideoInfo, frame_tensor, torch_cuda_handles
from .warp import GraphWarp, frame_rows, moves, window_size

logger = logging.getLogger(__name__)

# Constant-QP NVENC. Measured on a noisy 4K phone recording (41 Mbps H.264 source):
#   standard QP22 -> 46.3 dB luma PSNR vs the source, ~23 Mbps
#   high     QP18 -> 49.0 dB, ~39 Mbps (about the source's own size; visually transparent)
#   max      QP14 -> 51.5 dB, ~59 Mbps
# Constant QP rather than VBR "cq": through PyNvVideoCodec the VBR mode's output tracked the
# max-bitrate ceiling instead of the quality target, so its quality was unpredictable.
QUALITY_QP = {"standard": 22, "high": 18, "max": 14}


def encoder_config(quality: str, fps: float, codec: str = "hevc") -> dict:
    return dict(codec=codec, preset="P5", tuning_info="high_quality", rc="constqp",
                constqp=str(QUALITY_QP.get(quality, QUALITY_QP["high"])),
                gop=str(max(1, int(round(fps * 2)))), bf="0", aq="1")


class _CAI:
    def __init__(self, shape, strides, typestr, ptr):
        self.__cuda_array_interface__ = {"shape": shape, "strides": strides, "typestr": typestr,
                                         "data": (ptr, False), "version": 3}


class _Frame:
    """What PyNvVideoCodec's encoder wants: an object whose .cuda() lists the planes.
    16-bit (P010) planes are described exactly as NVIDIA's own sample does — the encoder
    rejects torch's '<u2' typestr and expects '|u2' with byte strides."""

    def __init__(self, t, H, W):
        if t.dtype == torch.uint8:
            self.planes = [t[:H].view(H, W, 1), t[H:].view(H // 2, W // 2, 2)]
        else:
            base = t.data_ptr()
            self.planes = [_CAI((H, W, 1), (W * 2, 2, 1), "|u2", base),
                           _CAI((H // 2, W // 2, 2), (W * 2, 2, 1), "|u2", base + W * H * 2)]

    def cuda(self):
        return self.planes


def render(src: str, dst: str, plan: dict, info: VideoInfo, quality: str = "high", codec: str = "hevc",
           progress: Optional[Callable[[float], None]] = None,
           cancelled: Optional[Callable[[], bool]] = None) -> int:
    import PyNvVideoCodec as nvc
    ctx, _ = torch_cuda_handles()
    W, H = info.width, info.height
    dec = nvc.ThreadedDecoder(src, buffer_size=12, cuda_context=ctx, use_device_memory=True,
                              output_color_type=nvc.OutputColorType.NATIVE)
    dtype = torch.uint16 if info.ten_bit else torch.uint8
    bh, bw = window_size(plan, H, W)
    gw = GraphWarp(H, W, bh, bw, dtype=dtype)
    cfg = encoder_config(quality, info.fps, codec)
    if info.ten_bit:
        cfg["profile"] = "main10"
    enc = nvc.CreateEncoder(W, H, "P010" if info.ten_bit else "NV12", False, cudacontext=ctx, **cfg)
    ring = [torch.empty_like(gw.frame) for _ in range(2)]
    held = [torch.empty_like(gw.frame) for _ in range(8)]

    raw_path = dst + ".video.tmp"
    pts = []
    i = 0
    n_plan = len(plan["corr"])
    total = max(info.frames, n_plan, 1)
    try:
        with open(raw_path, "wb") as raw, torch.no_grad():
            while True:
                if cancelled and cancelled():
                    raise InterruptedError("cancelled")
                batch = dec.get_batch_frames(8)
                if not batch:
                    break
                for j, fr in enumerate(batch):
                    held[j].copy_(frame_tensor(fr, H, W, info.ten_bit))
                torch.cuda.current_stream().synchronize()
                for j, fr in enumerate(batch):
                    gw.frame.copy_(held[j])
                    if moves(plan, i):
                        gw.set_eyes(frame_rows(plan, i, H, W, bh, bw))
                        gw.run()
                    slot = ring[i % len(ring)]
                    slot.copy_(gw.frame)
                    torch.cuda.current_stream().synchronize()
                    pp = nvc.NV_ENC_PIC_PARAMS()
                    pp.inputTimeStamp = i
                    for pkt in enc.Encode(_Frame(slot, H, W), pp):
                        raw.write(bytes(pkt["data"]))
                    pts.append(int(fr.timestamp))
                    i += 1
                if progress:
                    progress(min(i / total, 1.0))
            for pkt in enc.EndEncode():
                raw.write(bytes(pkt["data"]))
        if i != n_plan:
            logger.warning("Render saw %d frames but the plan has %d", i, n_plan)
        mux(src, raw_path, dst, pts, info, codec)
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    return i


def mux(src: str, raw_video: str, dst: str, pts, info: VideoInfo, codec: str = "hevc"):
    """MP4 with the new video stream stamped with the source's own PTS (VFR-safe) and the
    source's audio copied untouched; colour tags, creation time and rotation carried over."""
    import av
    tmp = dst + ".mux.tmp.mp4"
    with av.open(src) as s, av.open(raw_video, format=codec) as v, \
            av.open(tmp, "w", options={"movflags": "+faststart"}) as o:
        sv = s.streams.video[0]
        for key in ("creation_time", "location", "com.apple.quicktime.location.ISO6709"):
            if key in s.metadata:
                o.metadata[key] = s.metadata[key]
        ov = o.add_stream_from_template(v.streams.video[0])
        ov.time_base = sv.time_base
        for attr in ("color_primaries", "color_trc", "colorspace", "color_range"):
            try:
                setattr(ov.codec_context, attr, getattr(sv.codec_context, attr))
            except Exception:
                pass
        auds = {a.index: o.add_stream_from_template(a) for a in s.streams.audio}
        o.start_encoding()   # header first: the muxer may change the stream time base
        tb_src, tb_out = sv.time_base, ov.time_base
        apk = s.demux(*s.streams.audio) if auds else iter(())
        nxt_a = next(apk, None)

        def flush_audio(until):
            nonlocal nxt_a
            while nxt_a is not None and (until is None or nxt_a.pts is None
                                         or nxt_a.pts * float(nxt_a.time_base) <= until):
                if nxt_a.size:
                    nxt_a.stream = auds[nxt_a.stream.index]
                    o.mux(nxt_a)
                nxt_a = next(apk, None)

        step = int(pts[-1] - pts[-2]) if len(pts) > 1 else 1
        for n, p in enumerate(pk for pk in v.demux(v.streams.video[0]) if pk.size):
            cur = int(pts[n]) if n < len(pts) else int(pts[-1]) + (n - len(pts) + 1) * step
            nxt = int(pts[n + 1]) if n + 1 < len(pts) else cur + step
            p.stream = ov
            p.time_base = tb_out
            p.pts = p.dts = int(round(Fraction(cur) * tb_src / tb_out))
            p.duration = max(1, int(round(Fraction(nxt - cur) * tb_src / tb_out)))
            flush_audio(cur * float(tb_src))
            o.mux(p)
        flush_audio(None)
    if info.rotation:
        # PyAV can't write a display matrix; ffmpeg can, without touching the streams
        rot = dst + ".rot.tmp.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation:v:0", str(info.rotation), "-i", tmp,
                        "-map", "0", "-c", "copy", "-movflags", "+faststart", rot], check=True)
        os.replace(rot, tmp)
    os.replace(tmp, dst)
