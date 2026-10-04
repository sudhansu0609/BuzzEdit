"""The eye warp: slide each iris inside its eye opening, on the GPU, in place on an NV12/P016 frame.

Only existing pixels move — nothing is generated — so the eye keeps its own texture,
catchlights and lashes. The displacement is a smooth field in each eye's own frame
(s along the inter-ocular axis, t across it):
  * along s: a rigid core over the iris, easing to exactly zero at both eye corners,
    so the whites on one side compress and the other side stretches;
  * across t: full strength inside the lid opening, easing to zero a little way into
    the lids, so the iris slides under the lid edge the way a turning eye does.
Every pixel outside the eye windows is left bit-identical.

Both eyes (luma bicubic, chroma bilinear) run as one CUDA-graph replay per frame;
geometry reaches the GPU as one small pinned upload, never as per-op syncs.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .gpu import DEV
from .plan import EYES, P

LID_DEG = 3
K = 23   # per-eye parameter vector, see eye_params()


class EyeGeom:
    """One eye in one frame, reduced to floats. Lids are cubic fits t(s) pinned to the corners."""

    def __init__(self, mid, ex, corners, upper, lower, iris, r):
        self.mx, self.my = (float(v) for v in mid)
        e = np.asarray(ex, np.float64)
        e = e / np.linalg.norm(e)
        self.cx_, self.cy_ = float(e[0]), float(e[1])
        cs, ct = self.local(corners)
        o = np.argsort(cs)
        self.s_lc, self.s_rc = float(cs[o[0]]), float(cs[o[1]])
        self.t_lc, self.t_rc = float(ct[o[0]]), float(ct[o[1]])
        si, ti = self.local(np.asarray(iris)[None])
        self.si, self.ti, self.r = float(si[0]), float(ti[0]), float(r)
        self.up = self._fit(upper)
        self.lo = self._fit(lower)

    def local(self, p):
        d = np.atleast_2d(np.asarray(p, np.float64)) - [self.mx, self.my]
        return d @ [self.cx_, self.cy_], d @ [-self.cy_, self.cx_]

    def _fit(self, pts):
        s, t = self.local(pts)
        s = np.concatenate([[self.s_lc, self.s_rc], s])
        t = np.concatenate([[self.t_lc, self.t_rc], t])
        w = np.ones_like(s)
        w[:2] = 4.0
        span = max(self.s_rc - self.s_lc, 1.0)
        return [float(v) for v in np.polyfit(s / span, t, LID_DEG, w=w)], span


def eye_geom(plan, i, k) -> EyeGeom:
    e, row = EYES[k], plan["pts"][i:i + 1]
    corners = np.vstack([P(row, e["outer"])[0, :2], P(row, e["inner"])[0, :2]])
    iris = P(row, e["iris"])[0, :2]
    r = np.mean([np.linalg.norm(P(row, j)[0, :2] - iris) for j in e["ring"]])
    upper = np.array([P(row, j)[0, :2] for j in e["upper"]])
    lower = np.array([P(row, j)[0, :2] for j in e["lower"]])
    return EyeGeom(corners.mean(0), plan["ex"][i], corners, upper, lower, iris, r)


def eye_box(g: EyeGeom, w_eye, H, W):
    """Even-aligned luma box containing every pixel the field can move."""
    ext = (0.04 + 0.14 + 0.18) * w_eye
    ss = np.linspace(g.s_lc - 0.2 * w_eye, g.s_rc + 0.2 * w_eye, 9)
    tu = np.polyval(g.up[0], ss / g.up[1]) - ext
    tl = np.polyval(g.lo[0], ss / g.lo[1]) + ext
    S, T = np.concatenate([ss, ss]), np.concatenate([tu, tl])
    x = g.mx + S * g.cx_ - T * g.cy_
    y = g.my + S * g.cy_ + T * g.cx_
    x0, x1 = int(np.floor(x.min())) - 2, int(np.ceil(x.max())) + 2
    y0, y1 = int(np.floor(y.min())) - 2, int(np.ceil(y.max())) + 2
    x0 = max(x0 - x0 % 2, 0); y0 = max(y0 - y0 % 2, 0)
    return x0, y0, min(x1 + x1 % 2, W), min(y1 + y1 % 2, H)


def window_for(g: EyeGeom, w_eye, H, W, bh, bw):
    """Origin of the fixed-size window around the eye, or None if the eye doesn't fit in it
    (e.g. half out of frame) — that eye is then left alone for the frame."""
    x0, y0, x1, y1 = eye_box(g, w_eye, H, W)
    wx = int(round((x0 + x1) / 2 - bw / 2)); wy = int(round((y0 + y1) / 2 - bh / 2))
    wx -= wx % 2; wy -= wy % 2
    wx = min(max(wx, 0), W - bw); wy = min(max(wy, 0), H - bh)
    if x0 < wx or y0 < wy or x1 > wx + bw or y1 > wy + bh:
        return None
    return wx, wy


def eye_params(g: EyeGeom, c_s: float, w_eye: float, x0: int, y0: int):
    r = g.r
    lo_core = min(g.si - r, g.s_rc - 0.25 * w_eye)
    hi_core = max(g.si + r, g.s_lc + 0.25 * w_eye)
    left_len = max(lo_core - g.s_lc, 0.18 * w_eye)
    right_len = max(g.s_rc - hi_core, 0.18 * w_eye)
    # The whites between iris and corner absorb the shift. The ease's steepest slope is
    # 1.5x its mean, so keeping |c_s| <= half that stretch leaves the mapping at >= 25% of
    # its original spacing there: squeezed, never folded.
    c_s = max(-0.5 * left_len, min(0.5 * right_len, c_s))
    (uc, us), (lc, ls) = g.up, g.lo
    return [g.mx, g.my, g.cx_, g.cy_, lo_core, hi_core, left_len, right_len,
            0.04 * w_eye, 0.14 * w_eye, c_s, float(x0), float(y0), us, *uc, ls, *lc]


def _smoothstep(x):
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def field(p, x, y):
    """Forward displacement (dx, dy) at image points (x, y) for parameter rows p (B,K)."""
    Pk = [p[:, k, None, None] for k in range(K)]
    mx, my, cx, cy, lo_core, hi_core, left_len, right_len, margin, band, c_s = Pk[:11]
    us, uc, ls, lc = Pk[13], Pk[14:18], Pk[18], Pk[19:23]
    dx_, dy_ = x - mx, y - my
    s = dx_ * cx + dy_ * cy
    t = dy_ * cx - dx_ * cy
    one = torch.ones_like(s)
    hs = torch.where(s < lo_core, _smoothstep((s - (lo_core - left_len)) / left_len),
                     torch.where(s > hi_core, _smoothstep(((hi_core + right_len) - s) / right_len), one))
    u = s / us
    up = ((uc[0] * u + uc[1]) * u + uc[2]) * u + uc[3]
    u = s / ls
    lo = ((lc[0] * u + lc[1]) * u + lc[2]) * u + lc[3]
    ht = torch.where(t < up - margin, _smoothstep((t - (up - margin - band)) / band),
                     torch.where(t > lo + margin, _smoothstep(((lo + margin + band) - t) / band), one))
    d = c_s * hs * ht
    return d * cx, d * cy


class GraphWarp:
    """Warps both eyes of the frame in `self.frame` with one CUDA-graph replay.
    Windows have a fixed size (bh, bw) and are placed per frame through the params."""

    PAD = 4

    def __init__(self, H, W, bh, bw, dtype=torch.uint8, chroma_left=True, iters=3):
        self.H, self.W, self.bh, self.bw, self.iters = H, W, bh, bw, iters
        self.off = 0.5 if chroma_left else 1.0
        self.wide = dtype != torch.uint8
        self.vmax = 65535.0 if self.wide else 255.0
        self.frame = torch.zeros(H * 3 // 2, W, dtype=dtype, device=DEV)
        self.params = torch.zeros(2, K, device=DEV)
        self.ring = [torch.zeros(2, K).pin_memory() for _ in range(4)]
        self.events = [None] * 4
        self.slot = 0
        pad = self.PAD

        def grid(h, w, dt):
            return torch.meshgrid(torch.arange(h, device=DEV, dtype=dt), torch.arange(w, device=DEV, dtype=dt),
                                  indexing="ij")

        self.by, self.bx = grid(bh, bw, torch.float32)
        self.sy, self.sx = grid(bh + 2 * pad, bw + 2 * pad, torch.long)
        self.cby, self.cbx = grid(bh // 2, bw // 2, torch.float32)
        self.csy, self.csx = grid(bh // 2 + 2 * pad, bw // 2 + 2 * pad, torch.long)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run()

    def _backward(self, p, px, py):
        """Source position q of each output position p: solve q + d(q) = p by fixed point."""
        qx, qy = px, py
        for _ in range(self.iters):
            dx, dy = field(p, qx, qy)
            qx, qy = px - dx, py - dy
        return qx, qy, (dx.abs() + dy.abs()) > 1e-3

    def _plane(self, flat, pw, ph, p, bx, by, sx, sy, scale, off_x, off_y, mode, stride, chan, nchan):
        pad = self.PAD
        x0 = (p[:, 11] / scale).long()
        y0 = (p[:, 12] / scale).long()
        px = (x0[:, None, None] + bx[None]) * scale + off_x      # sample sites in luma coords
        py = (y0[:, None, None] + by[None]) * scale + off_y
        qx, qy, moved = self._backward(p, px, py)
        gx = (x0[:, None, None] - pad + sx[None]).clamp(0, pw - 1)
        gy = (y0[:, None, None] - pad + sy[None]).clamp(0, ph - 1)
        src = self._read(flat, (gy * stride + gx) * nchan + chan)
        hs_, ws_ = src.shape[1], src.shape[2]
        ix = (qx - off_x) / scale - (x0[:, None, None] - pad)
        iy = (qy - off_y) / scale - (y0[:, None, None] - pad)
        g = torch.stack([(ix + 0.5) / ws_ * 2 - 1, (iy + 0.5) / hs_ * 2 - 1], -1)
        out = F.grid_sample(src[:, None], g, mode=mode, padding_mode="border",
                            align_corners=False)[:, 0].round().clamp(0, self.vmax)
        idx = ((y0[:, None, None] + by[None].long()) * stride + x0[:, None, None] + bx[None].long()) * nchan + chan
        flat[idx] = self._store(torch.where(moved, out, self._read(flat, idx)))

    # CUDA has no indexed gather/scatter for uint16, so 10-bit frames are addressed through an
    # int16 view of the same memory and converted at the edges.
    def _read(self, flat, idx):
        v = flat[idx]
        if v.dtype == torch.int16:
            return (v.to(torch.int32) & 0xFFFF).float()
        return v.float()

    def _store(self, vals):
        if not self.wide:
            return vals.to(torch.uint8)
        v = vals.to(torch.int32)
        return torch.where(v > 32767, v - 65536, v).to(torch.int16)

    def _run(self):
        H, W = self.H, self.W
        buf = self.frame.view(torch.int16) if self.wide else self.frame
        Yf = buf[:H].reshape(-1)
        UVf = buf[H:].reshape(-1)
        for e in range(2):   # sequential, so overlapping windows (tiny faces) still compose
            p = self.params[e:e + 1]
            self._plane(Yf, W, H, p, self.bx, self.by, self.sx, self.sy, 1.0, 0.5, 0.5, "bicubic", W, 0, 1)
            for ch in range(2):
                self._plane(UVf, W // 2, H // 2, p, self.cbx, self.cby, self.csx, self.csy, 2.0,
                            self.off, 1.0, "bilinear", W // 2, ch, 2)

    def set_eyes(self, rows):
        """Two K-length parameter rows (c_s = 0 leaves an eye alone). Async upload from a pinned
        ring; a slot is only reused once its previous copy has landed."""
        i = self.slot = (self.slot + 1) % len(self.ring)
        if self.events[i] is not None:
            self.events[i].synchronize()
        self.ring[i].copy_(torch.tensor(rows, dtype=torch.float32))
        self.params.copy_(self.ring[i], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.events[i] = ev

    def run(self):
        self.graph.replay()


def window_size(plan, H, W):
    """One window size for the whole video: the largest eye box, rounded up to a multiple of 8."""
    bh = bw = 16
    for i in np.flatnonzero(plan["valid"])[::5]:
        for k in "RL":
            x0, y0, x1, y1 = eye_box(eye_geom(plan, i, k), float(plan["w_e"][k][i]), H, W)
            bh, bw = max(bh, y1 - y0 + 8), max(bw, x1 - x0 + 8)
    return min(-(-bh // 8) * 8, H - H % 8), min(-(-bw // 8) * 8, W - W % 8)


def frame_rows(plan, i, H, W, bh, bw):
    """Parameter rows for both eyes of frame i (an eye that can't be windowed gets c_s = 0)."""
    rows = []
    per_eye = plan.get("corr_e") or {}
    for k in "RL":
        we = float(plan["w_e"][k][i])
        corr = per_eye[k][i] if k in per_eye else plan["corr"][i]
        c = float(corr) * we if plan["valid"][i] else 0.0
        g = eye_geom(plan, i, k)
        win = window_for(g, we, H, W, bh, bw) if abs(c) > 1e-3 else None
        if win is None:
            c, win = 0.0, (0, 0)
        rows.append(eye_params(g, c, we, *win))
    return rows
