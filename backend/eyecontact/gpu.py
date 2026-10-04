"""CUDA plumbing shared by both passes: context sharing, NV12/P016 frames, video facts."""
import ctypes
import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

DEV = torch.device("cuda")

# Video-range YCbCr -> RGB [0,1]. BT.709 for HD/4K, BT.601 for SD. Only ever used to feed
# the landmark net, which doesn't care about the last percent of colour accuracy.
_M709 = [[1.164, 0.0, 1.793], [1.164, -0.213, -0.533], [1.164, 2.112, 0.0]]
_M601 = [[1.164, 0.0, 1.596], [1.164, -0.392, -0.813], [1.164, 2.017, 0.0]]


def torch_cuda_handles():
    """(CUcontext, CUstream) torch uses, so NVDEC/NVENC live in the same context.

    Separate contexts per decoder made PyNvVideoCodec fail with
    CUDA_ERROR_INVALID_CONTEXT as soon as two decoders were open."""
    torch.zeros(1, device=DEV)   # makes the primary context current on this thread
    lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    ctx = ctypes.c_void_p()
    rc = lib.cuCtxGetCurrent(ctypes.byref(ctx))
    if rc != 0 or not ctx.value:
        raise RuntimeError(f"cuCtxGetCurrent failed ({rc})")
    return ctx.value, torch.cuda.current_stream().cuda_stream


@dataclass
class VideoInfo:
    width: int
    height: int
    frames: int                # container's estimate; the passes count the real frames
    fps: float
    time_base: Fraction
    rotation: int = 0          # display rotation in degrees, as ffprobe reports it
    ten_bit: bool = False
    colorspace: str = "bt709"
    has_audio: bool = False


def probe(path: str) -> VideoInfo:
    """Stream facts without reading the file: both passes get exact per-frame PTS from
    the decoder anyway, so there is no need to demux a 10 GB recording up front."""
    import av
    with av.open(path) as c:
        v = c.streams.video[0]
        cc = v.codec_context
        fps = float(v.average_rate) if v.average_rate else 30.0
        frames = v.frames or int(round(float(v.duration * v.time_base) * fps)) if v.duration else v.frames
        pix = str(getattr(cc, "pix_fmt", "") or "")
        space = str(getattr(cc, "colorspace", "") or "")
        return VideoInfo(width=cc.width, height=cc.height, frames=int(frames or 0), fps=fps,
                         time_base=v.time_base, rotation=_ffprobe_rotation(path),
                         ten_bit=any(b in pix for b in ("10", "12", "16")),
                         colorspace="bt601" if space in ("5", "6", "bt470bg", "smpte170m") else "bt709",
                         has_audio=len(c.streams.audio) > 0)


def _ffprobe_rotation(path: str) -> int:
    import json
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream_side_data=rotation:stream_tags=rotate", "-of", "json", path],
                             capture_output=True, text=True, timeout=30).stdout
        s = (json.loads(out).get("streams") or [{}])[0]
        for sd in s.get("side_data_list") or []:
            if "rotation" in sd:
                return int(round(float(sd["rotation"])))
        return int(s.get("tags", {}).get("rotate", 0))
    except Exception:
        return 0


class _CAI:
    def __init__(self, ptr, shape, strides, typestr):
        self.__cuda_array_interface__ = {"shape": shape, "strides": strides, "typestr": typestr,
                                         "data": (ptr, False), "version": 3}


def frame_tensor(fr, H: int, W: int, ten_bit: bool) -> torch.Tensor:
    """A decoded frame as one (H*3/2, W) tensor, zero-copy.

    8-bit NV12 comes through DLPack bit-exact. For 10-bit (P016) PyNvVideoCodec's DLPack
    export is inconsistent (uint8 dtype, strides that make torch skip every other pixel),
    so the planes are addressed from their raw pointers with a correct descriptor."""
    if not ten_bit:
        return torch.from_dlpack(fr)
    p0, p1 = int(fr.GetPtrToPlane(0)), int(fr.GetPtrToPlane(1))
    pitch = (p1 - p0) // H
    if pitch == W * 2:
        return torch.as_tensor(_CAI(p0, (H * 3 // 2, W), (W * 2, 2), "<u2"), device=DEV)
    # padded pitch: view each plane and pack them (the caller copies the frame anyway)
    y = torch.as_tensor(_CAI(p0, (H, W), (pitch, 2), "<u2"), device=DEV)
    uv = torch.as_tensor(_CAI(p1, (H // 2, W), (pitch, 2), "<u2"), device=DEV)
    return torch.cat([y, uv], 0)


def planes(frame: torch.Tensor, H: int, W: int):
    """Y (H,W) and interleaved UV (H/2,W/2,2) views of an NV12 / P016 frame tensor."""
    return frame[:H], frame[H:].view(H // 2, W // 2, 2)


_GRIDS = {}


def _base_grid(oh, ow):
    key = (oh, ow)
    if key not in _GRIDS:
        vv, uu = torch.meshgrid(torch.arange(oh, device=DEV, dtype=torch.float32) + 0.5,
                                torch.arange(ow, device=DEV, dtype=torch.float32) + 0.5, indexing="ij")
        _GRIDS[key] = (uu, vv)
    return _GRIDS[key]


def sample_rgb_batch(Y, UV, affines, out_hw, ss=1, colorspace="bt709"):
    """RGB patches (N,3,oh,ow) in [0,1]; patch pixel (u,v) maps to image affines[i] @ [u,v,1]
    (image px, pixel centres at +0.5). ss>1 supersamples and box-filters so big downscales
    don't alias. The crop box is found on the CPU from the affines: no host syncs."""
    H, W = Y.shape
    As = np.asarray(affines, np.float64).reshape(-1, 2, 3).copy()
    n = len(As)
    oh, ow = out_hw
    corners = np.array([[0, 0, 1], [ow, 0, 1], [0, oh, 1], [ow, oh, 1]], np.float64).T
    pts = As @ corners
    x0 = max(int(np.floor(pts[:, 0].min())) - 2, 0); x1 = min(int(np.ceil(pts[:, 0].max())) + 3, W)
    y0 = max(int(np.floor(pts[:, 1].min())) - 2, 0); y1 = min(int(np.ceil(pts[:, 1].max())) + 3, H)
    x0 -= x0 % 2; y0 -= y0 % 2
    x1 = min(x1 + x1 % 2, W); y1 = min(y1 + y1 % 2, H)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return torch.zeros(n, 3, oh, ow, device=DEV)
    As[:, :, :2] /= ss
    uu, vv = _base_grid(oh * ss, ow * ss)
    At = torch.as_tensor(As, device=DEV, dtype=torch.float32)
    x = At[:, 0, 0, None, None] * uu + At[:, 0, 1, None, None] * vv + At[:, 0, 2, None, None]
    y = At[:, 1, 0, None, None] * uu + At[:, 1, 1, None, None] * vv + At[:, 1, 2, None, None]
    scale = 1.0 / 256.0 if Y.dtype != torch.uint8 else 1.0   # P016: 10-bit samples in the high bits
    yc = (Y[y0:y1, x0:x1].float() * scale)[None, None].expand(n, 1, -1, -1)
    uvc = (UV[y0 // 2:y1 // 2, x0 // 2:x1 // 2].permute(2, 0, 1).float() * scale)[None].expand(n, 2, -1, -1)
    grid = torch.stack([(x - x0) / (x1 - x0) * 2 - 1, (y - y0) / (y1 - y0) * 2 - 1], -1)
    ys = F.grid_sample(yc, grid, mode="bilinear", padding_mode="border", align_corners=False)
    uvs = F.grid_sample(uvc, grid, mode="bilinear", padding_mode="border", align_corners=False)
    yuv = torch.cat([ys - 16, uvs - 128], 1)
    M = torch.tensor(_M601 if colorspace == "bt601" else _M709, device=DEV)
    rgb = (torch.einsum("ij,njhw->nihw", M, yuv) / 255).clamp(0, 1)
    return F.avg_pool2d(rgb, ss) if ss > 1 else rgb
