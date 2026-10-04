"""Pass 1: face/iris landmarks for every frame, entirely on the GPU (NVDEC -> crop -> net).

Each frame's face crop comes from the previous frame's landmarks (MediaPipe's own
tracking scheme); YuNet only (re)acquires the face at the start or after it is lost.
Every crop is supersampled and run at four sub-pixel jitters whose answers are
averaged: that cut iris jitter by ~40%, which matters because a correction can't be
steadier than the measurement it is computed from.
"""
import logging
import math
from typing import Callable, Optional

import numpy as np
import torch

from .gpu import DEV, VideoInfo, frame_tensor, planes, sample_rgb_batch, torch_cuda_handles
from .landmarks import GraphNet, ensure_model
from .plan import USED

logger = logging.getLogger(__name__)

# (dx, dy, scale) jitters of the face crop, in 1/64ths of the crop
TTA = [(0, 0, 1.0), (0.35, 0.2, 1.03), (-0.3, 0.3, 0.97), (0.2, -0.35, 1.0)]
SUPERSAMPLE = 3
ROI_SCALE = 1.5          # MediaPipe's face crop: landmark box x 1.5, square
DETECT_WIDTH = 960


def roi_affine(ctr, size, ang, n=256):
    ca, sa = math.cos(ang), math.sin(ang)
    k = size / n
    A = np.array([[k * ca, -k * sa, 0.0], [k * sa, k * ca, 0.0]])
    A[:, 2] = np.asarray(ctr, np.float64) - A[:, :2] @ np.array([n / 2, n / 2])
    return A


def roi_from_landmarks(p):
    """Face crop around full-mesh landmarks p (478,>=2), rotated so the eyes are level."""
    l, r = p[33, :2], p[263, :2]
    ang = math.atan2(r[1] - l[1], r[0] - l[0])
    c, s = math.cos(-ang), math.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    q = p[:, :2] @ R.T
    mn, mx = q.min(0), q.max(0)
    return ((mn + mx) / 2) @ R, float(max(mx - mn) * ROI_SCALE), ang


class Tracker:
    def __init__(self, colorspace="bt709"):
        self.colorspace = colorspace
        self.net = GraphNet(ensure_model(), len(TTA), DEV)
        self.det = None
        self.det_size = None
        self.prev = None
        self.orientation = 0

    # -- (re)acquisition --------------------------------------------------------
    def _detector(self, w, h):
        import cv2
        from presentation.facezoom import ensure_model as ensure_yunet
        if self.det is None or self.det_size != (w, h):
            model = ensure_yunet()
            if model is None:
                raise RuntimeError("Face detection model (YuNet) is unavailable")
            self.det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.6, 0.3, 5)
            self.det_size = (w, h)
        return self.det

    def detect(self, Y, UV):
        """Face crops (centre, size, angle) for the largest face seen in each of the four
        90-degree orientations. YuNet tolerates rotation well enough to "find" a sideways
        face in an unrotated portrait frame, but with nonsense eye points — so every
        orientation is offered and the landmark net's own presence score picks (track())."""
        H, W = Y.shape
        found = []
        for k in range(4):
            sw, sh = (W, H) if k % 2 == 0 else (H, W)
            dw = DETECT_WIDTH
            dh = int(round(dw * sh / sw / 2) * 2)
            s = sw / dw
            # det-image (u,v) -> coded frame (x,y), rotated by k*90 degrees
            A = [np.array([[s, 0, 0], [0, s, 0]]), np.array([[0, -s, W], [s, 0, 0]]),
                 np.array([[-s, 0, W], [0, -s, H]]), np.array([[0, s, 0], [-s, 0, H]])][k]
            img = (sample_rgb_batch(Y, UV, [A], (dh, dw), 1, self.colorspace)[0] * 255).byte()
            bgr = np.ascontiguousarray(img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
            _, faces = self._detector(dw, dh).detect(bgr)
            if faces is None or len(faces) == 0:
                continue
            f = max(faces, key=lambda r: r[2] * r[3])
            ctr = A[:, :2] @ np.array([f[0] + f[2] / 2, f[1] + f[3] / 2]) + A[:, 2]
            # the face is taken as upright in this det image: its x axis, plus YuNet's roll there
            roll = math.atan2(f[7] - f[5], f[6] - f[4])
            ang = math.atan2(A[1, 0], A[0, 0]) + max(-0.6, min(0.6, roll))
            found.append(((ctr, float(max(f[2], f[3]) * s * 1.6), ang), k))
        return found

    # -- landmarks ----------------------------------------------------------------
    def run_roi(self, Y, UV, roi):
        (cx, cy), size, ang = roi
        px = size / 256 * 4
        As = [roi_affine((cx + dx * px, cy + dy * px), size * sc, ang) for dx, dy, sc in TTA]
        rgb = sample_rgb_batch(Y, UV, As, (256, 256), SUPERSAMPLE, self.colorspace)
        out = self.net(rgb)
        lms = out[0].view(len(As), 478, 3).cpu().numpy().astype(np.float64)
        score = float(out[1][0, 0])
        acc = np.zeros((478, 3))
        for A, lm, (_, _, sc) in zip(As, lms, TTA):
            acc[:, :2] += lm[:, :2] @ A[:, :2].T + A[:, 2]
            acc[:, 2] += lm[:, 2] * (size * sc / 256)
        return acc / len(As), score

    def track(self, Y, UV):
        """Full-mesh landmarks (478,3) in frame pixels and a presence logit (>0 = face)."""
        if self.prev is None:
            best = (None, -99.0)
            for roi, k in self.detect(Y, UV):
                pts, sc = self.run_roi(Y, UV, roi)
                if sc > 0:
                    pts, sc = self.run_roi(Y, UV, roi_from_landmarks(pts))   # refine on its own crop
                if sc > best[1]:
                    best = (pts, sc)
                    self.orientation = k
            pts, sc = best
            if pts is None:
                return None, sc
        else:
            pts, sc = self.run_roi(Y, UV, roi_from_landmarks(self.prev))
        if sc < 0:
            self.prev = None
            return None, sc
        self.prev = pts
        return pts, sc


def track_video(src: str, info: VideoInfo, progress: Optional[Callable[[float], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None) -> dict:
    """Landmarks (only the points the correction uses) + presence + PTS for every frame."""
    import PyNvVideoCodec as nvc
    ctx, _ = torch_cuda_handles()
    dec = nvc.ThreadedDecoder(src, buffer_size=12, cuda_context=ctx, use_device_memory=True,
                              output_color_type=nvc.OutputColorType.NATIVE)
    W, H = info.width, info.height
    tracker = Tracker(info.colorspace)
    pts_all, scores, ts = [], [], []
    total = max(info.frames, 1)
    with torch.no_grad():
        while True:
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            batch = dec.get_batch_frames(8)
            if not batch:
                break
            for fr in batch:
                Y, UV = planes(frame_tensor(fr, H, W, info.ten_bit), H, W)
                pts, sc = tracker.track(Y, UV)
                pts_all.append(pts[USED].astype(np.float32) if pts is not None
                               else np.full((len(USED), 3), np.nan, np.float32))
                scores.append(sc)
                ts.append(int(fr.timestamp))
            # every result above was read back to the host, so the decoder may recycle its surfaces
            torch.cuda.current_stream().synchronize()
            if progress:
                progress(min(len(ts) / total, 1.0))
    lost = sum(1 for s in scores if s < 0)
    logger.info("Tracked %d frames (%d without a face)", len(ts), lost)
    return {"pts": np.stack(pts_all) if pts_all else np.zeros((0, len(USED), 3), np.float32),
            "score": np.asarray(scores, np.float32), "ts": np.asarray(ts, np.int64)}
