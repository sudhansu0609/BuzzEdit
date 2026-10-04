"""Turn per-frame landmarks into a per-frame iris correction (numpy only, offline).

The model of a teleprompter read:
  gaze(t) = prompter direction + reading sweep (a sawtooth: creep along a line, snap back)
The correction moves the irises so that
  output(t) = lens direction + (1 - steadiness) * (gaze(t) - prompter direction)
i.e. the prompter-to-lens offset is removed entirely and the reading sweep is damped,
while small natural eye movements survive (a fully locked stare looks dead).

Where the prompter is gets measured from the video (a robust running median of the
gaze). Where the lens is, relative to it, comes from the physical setup: which side
of the lens the prompter sits on, how far, and how far away the speaker sits. An
earlier version guessed it from the eyes' left/right symmetry instead; that read a
30 cm side prompter as 3.5 degrees and the fix was invisible. The symmetry guess is
kept only as prompter_side="auto".

Everything is non-causal: the whole timeline is known before a pixel is touched,
so blinks are bridged from both sides and saccades stay sharp (total-variation
denoising keeps steps where a causal filter would smear them into glides).
Gaze is measured as the mean of both irises along the inter-ocular axis (subject's
right eye -> left eye), in eye widths: the eyes' symmetric resting offsets (and
vergence) cancel in that mean, and "the subject's left" stays the same direction
whether the video is mirrored or rotated.
"""
from dataclasses import dataclass

import numpy as np

# MediaPipe face-mesh indices. "R" is the subject's right eye (image left when upright).
EYES = {
    "R": dict(outer=33, inner=133, upper=[246, 161, 160, 159, 158, 157, 173],
              lower=[7, 163, 144, 145, 153, 154, 155], iris=468, ring=[469, 470, 471, 472]),
    "L": dict(outer=263, inner=362, upper=[466, 388, 387, 386, 385, 384, 398],
              lower=[249, 390, 373, 374, 380, 381, 382], iris=473, ring=[474, 475, 476, 477]),
}
FOREHEAD, CHIN = 10, 152
USED = sorted({e[k] for e in EYES.values() for k in ("outer", "inner", "iris")}
              | {i for e in EYES.values() for k in ("upper", "lower", "ring") for i in e[k]}
              | {FOREHEAD, CHIN})
COL = {idx: j for j, idx in enumerate(USED)}

# How far the iris centre moves per unit sin(gaze angle), in eye widths. Measured with the
# iris as a ruler: an adult iris is ~11.7 mm across and sits ~9.9 mm in front of the eye's
# centre of rotation, and the iris spans ~0.42 of the corner-to-corner eye width, so
# 9.9 / 11.7 * 0.42 ~= 0.35.
IRIS_SWING = 0.35
EYE_WIDTHS_PER_DEGREE = IRIS_SWING * np.pi / 180
# Where the lens is, AS MEASURED: the measured gaze is ~0 when the eyes look straight down
# the camera axis (that is what "auto" assumes), so the prompter-to-lens distance in the
# measurement is how far the reading direction sits from 0. On real footage it is far
# smaller than the physical angle (a 2026-10-02 recording: lens ~4.4 deg from the reading
# direction as measured, while the warp needed ~19 deg -- 10 deg + a -9 deg aim -- to look
# right), so the physical angle + aim must only size the push, never decide which frames
# are reading: judged by it, a look into the lens was "reading" and got pushed past it.
# Clamped to [LENS_SEP_MIN_DEG, the physical angle] so a symmetric reader still works.
LENS_SEP_MIN_DEG = 3.0


@dataclass
class Settings:
    prompter_side: str = "left"   # side of the lens the prompter is on, from the speaker's seat
                                  # ("left" / "right"), or "auto" to guess from eye symmetry
    angle_deg: float = 0.0        # how far (degrees) the prompter is from the lens; when > 0 it
                                  # overrides the distance maths below
    prompter_cm: float = 30.0     # sideways distance from the lens to the middle of the text
    camera_cm: float = 110.0      # distance from the speaker's eyes to the camera
    steadiness: float = 0.7       # 0 = keep the reading sweep, 1 = lock the eyes on the lens
    aim_deg: float = 0.0          # fine nudge of the lens direction, + = toward the speaker's left
    max_shift: float = 0.14       # never move an iris more than this many eye widths (~23 deg)
    away_lo: float = 0.07         # gaze this far from both prompter and lens (eye widths) starts to pass through
    away_hi: float = 0.12         # ...and this far is left completely untouched (a deliberate look away)
    baseline_s: float = 20.0      # window for the prompter direction (seconds), reading frames only
    tv_lambda: float = 0.006      # saccade-preserving denoise strength (eye widths)
    head_depth: float = 0.12      # eyeball centre behind the eye corners (eye widths): head-turn parallax

    def prompter_angle_deg(self) -> float:
        if self.angle_deg > 0:
            return float(self.angle_deg)
        return float(np.degrees(np.arctan2(max(self.prompter_cm, 0.0), max(self.camera_cm, 1.0))))


def P(arr, idx):
    """Column of a compact landmark array for a face-mesh index."""
    return arr[:, COL[idx]]


def fill_gaps(x, valid):
    """Linear interpolation across invalid rows (per column), holding the ends."""
    x = np.array(x, np.float64, copy=True)
    if valid.sum() == 0 or valid.all():
        return x
    idx = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    for c in range(flat.shape[1]):
        flat[~valid, c] = np.interp(idx[~valid], idx[valid], flat[valid, c])
    return flat.reshape(x.shape)


def gauss_smooth(x, sigma):
    """Zero-phase Gaussian smoothing along axis 0 (edge-padded)."""
    if sigma <= 0 or len(x) == 0:
        return np.array(x, np.float64, copy=True)
    r = int(3 * sigma + 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    x = np.asarray(x, np.float64)
    flat = x.reshape(len(x), -1)
    pad = np.pad(flat, [(r, r), (0, 0)], mode="edge")
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(pad[:, c], k, mode="valid")
    return out.reshape(x.shape)


def tv_denoise(y, lam):
    """Exact 1-D total-variation denoising (Condat 2013): kills jitter, keeps steps."""
    y = np.asarray(y, np.float64)
    n = len(y)
    if n == 0:
        return y.copy()
    x = np.empty(n)
    k = k0 = kmin = kplus = 0
    vmin, vmax = y[0] - lam, y[0] + lam
    umin, umax = lam, -lam
    while True:
        if k == n - 1:
            if umin < 0:
                x[k0:kmin + 1] = vmin
                k = k0 = kmin = kmin + 1
                vmin = y[k]; umin = lam; umax = y[k] + lam - vmax
                continue
            if umax > 0:
                x[k0:kplus + 1] = vmax
                k = k0 = kplus = kplus + 1
                vmax = y[k]; umax = -lam; umin = y[k] - lam - vmin
                continue
            x[k0:] = vmin + umin / (k - k0 + 1)
            return x
        k += 1
        umin += y[k] - vmin
        umax += y[k] - vmax
        if umin < -lam:
            x[k0:kmin + 1] = vmin
            k = k0 = kmin = kplus = kmin + 1
            vmin = y[k]; vmax = y[k] + 2 * lam; umin = lam; umax = -lam
        elif umax > lam:
            x[k0:kplus + 1] = vmax
            k = k0 = kmin = kplus = kplus + 1
            vmax = y[k]; vmin = y[k] - 2 * lam; umin = lam; umax = -lam
        else:
            if umin >= lam:
                vmin += (umin - lam) / (k - k0 + 1); umin = lam; kmin = k
            if umax <= -lam:
                vmax += (umax + lam) / (k - k0 + 1); umax = -lam; kplus = k


def rolling_median(x, valid, half, stride=1):
    """Median of the valid samples within +-half, evaluated every `stride` samples and
    linearly interpolated between (the caller smooths it anyway)."""
    n = len(x)
    at = np.unique(np.r_[np.arange(0, n, max(1, stride)), n - 1]) if n else np.arange(0)
    vals = np.full(len(at), np.nan)
    for j, i in enumerate(at):
        a, b = max(0, i - half), min(n, i + half + 1)
        v = x[a:b][valid[a:b]]
        if len(v):
            vals[j] = np.median(v)
    ok = np.isfinite(vals)
    if not ok.any():
        return np.full(n, np.nan)
    return np.interp(np.arange(n), at[ok], vals[ok])


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def make_plan(pts, score, fps, s: Settings = Settings()):
    """pts: (N, len(USED), 3) compact landmarks (NaN where no face), score: (N,) presence logits.

    Returns per-frame arrays: smoothed landmarks the warp anchors to, the inter-ocular axis,
    eye widths, and `corr` — the iris shift in eye widths along that axis."""
    n = len(pts)
    valid = np.isfinite(pts[:, 0, 0]) & (np.asarray(score) > 0)
    if valid.sum() < max(3, n // 50):
        return _empty_plan(pts, valid)
    raw = fill_gaps(pts, valid)
    smooth = gauss_smooth(raw, 0.8)        # light zero-phase smoothing of what the warp anchors to

    ex = P(smooth, 263)[:, :2] - P(smooth, 33)[:, :2]
    ex /= np.linalg.norm(ex, axis=1, keepdims=True)
    ey = np.stack([-ex[:, 1], ex[:, 0]], 1)

    s_e, open_e, w_e = {}, {}, {}
    for k, e in EYES.items():
        a, b = P(raw, e["outer"])[:, :2], P(raw, e["inner"])[:, :2]
        w = np.linalg.norm(b - a, axis=1)
        mid = (a + b) / 2
        s_e[k] = ((P(raw, e["iris"])[:, :2] - mid) * ex).sum(1) / w
        up = np.mean([P(raw, i)[:, :2] for i in e["upper"]], 0)
        lo = np.mean([P(raw, i)[:, :2] for i in e["lower"]], 0)
        open_e[k] = ((lo - up) * ey).sum(1) / w
        w_e[k] = gauss_smooth(np.linalg.norm(P(smooth, e["inner"])[:, :2] - P(smooth, e["outer"])[:, :2], axis=1), 2.0)

    # blinks and squints: the iris landmarks are guesses there
    op = np.minimum(open_e["R"], open_e["L"])
    ref = rolling_median(op, valid, int(fps * 3))
    ref = fill_gaps(ref, np.isfinite(ref))
    eye_ok = valid & (op > 0.6 * ref)
    eye_ok &= np.roll(eye_ok, 1) & np.roll(eye_ok, -1)    # a frame of margin either side

    # head turn: face normal projected on the eye axis (parallax of the eyeball centre)
    right = P(raw, 263) - P(raw, 33)
    down = P(raw, CHIN) - P(raw, FOREHEAD)
    nrm = np.cross(down, right)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if np.nanmedian(nrm[:, 2]) > 0:
        nrm = -nrm
    n_along = gauss_smooth((nrm[:, :2] * ex).sum(1), fps * 0.1)

    gaze = (s_e["R"] + s_e["L"]) / 2 + s.head_depth * n_along
    if eye_ok.sum() < 3:
        return _empty_plan(pts, valid)
    gaze = fill_gaps(gaze, eye_ok)
    gaze_f = tv_denoise(gaze, s.tv_lambda)
    half, stride = int(fps * s.baseline_s), max(1, int(fps / 4))

    def baseline(frames):
        b = rolling_median(gaze_f, frames, half, stride)
        return gauss_smooth(fill_gaps(b, np.isfinite(b)), fps * 0.5)

    nudge = s.aim_deg * EYE_WIDTHS_PER_DEGREE
    # Which frames look into the camera is judged on the gaze held for a moment: a reading
    # sweep crosses toward the lens and back within a line, a look into the lens stays.
    gaze_hold = gauss_smooth(gaze_f, fps * 0.4)
    if s.prompter_side in ("left", "right"):
        # Reading looks toward the prompter; the lens is that angle back the other way. The
        # prompter direction is measured from reading frames only: a frame nearer the lens than
        # the prompter is the speaker looking into the camera, and a long stretch of that (an
        # intro, a sign-off) must not drag the estimate onto the lens — that would read
        # camera-looking as reading and push those eyes the full angle past the camera.
        sign = -1.0 if s.prompter_side == "left" else 1.0
        swing = IRIS_SWING * np.sin(np.radians(s.prompter_angle_deg()))
        to_lens = sign * swing + nudge                     # how far a reading frame is moved
        sep_lo = LENS_SEP_MIN_DEG * EYE_WIDTHS_PER_DEGREE
        sep_hi = max(swing, sep_lo)

        def lens_at(b):                                    # the lens in the measurement
            return b + sign * np.clip(sign * (0.0 - b), sep_lo, sep_hi)

        base = np.full(n, np.median(gaze_f[eye_ok]))
        for _ in range(3):
            lens_m = lens_at(base)
            reading = eye_ok & (np.abs(gaze_hold - base) <= np.abs(gaze_hold - lens_m))
            if reading.sum() < 3:
                break
            base = baseline(reading)
        lens_m = lens_at(base)
        lens = base + to_lens
    else:
        base = baseline(eye_ok)
        lens = np.full(n, nudge)          # "auto": the eyes' symmetric position is the lens
        lens_m = np.zeros(n)
    corr = lens - base - s.steadiness * (gaze_f - base)
    # Only reading gets re-aimed. Looking into the lens already (or past it), glancing
    # elsewhere, big head turns and lost faces are left alone, all with soft edges.
    span = lens_m - base
    span = np.where(np.abs(span) < 1e-6, 1e-6, span)
    toward_lens = (gaze_hold - base) / span                # 0 at the prompter, 1 at the lens
    # Full push only in the prompter's half; none from 60% of the way to the lens. Measured
    # camera looks sat at 0.5-0.75 of the way (a centred stare reads a degree or two short of
    # the lens) and got most of the push from a fade that started at 0.45 -- the overshoot.
    w = 1 - smoothstep((toward_lens - 0.3) / 0.3)
    dev_p = np.abs(gaze_f - base)
    dev_l = np.abs(gaze_f - lens_m)
    w *= 1 - smoothstep((np.minimum(dev_p, dev_l) - s.away_lo) / (s.away_hi - s.away_lo))
    w *= 1 - smoothstep((np.abs(n_along) - 0.30) / 0.15)
    w *= 1 - np.clip(gauss_smooth((~valid).astype(float), 3.0) * 3, 0, 1)
    w = gauss_smooth(w, 1.0)
    corr = np.clip(corr * w, -s.max_shift, s.max_shift)
    corr[~valid] = 0.0
    corr_e, lens_e = per_eye_limits(corr, s_e, eye_ok, w, gaze_f, lens_m, fps, s)

    return dict(pts=smooth, ex=ex, w_e=w_e, corr=corr, corr_e=corr_e, lens_e=lens_e,
                gaze=gaze, gaze_f=gaze_f, base=base,
                weight=w, eye_ok=eye_ok, valid=valid, lens_m=lens_m,
                camera_pct=float(100.0 * np.mean((w < 0.5)[valid])) if valid.any() else 0.0,
                lens_sep_deg=float(np.median(np.abs(lens_m - base)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                offset_deg=float(np.median((base - lens)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                symmetry_offset_deg=float(np.median(base[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                sweep_deg=float(np.std((gaze_f - base)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0)


LENS_LOOK_MIN_S = 2.0      # seconds of real camera looks needed to trust each eye's lens position
LENS_MARGIN = 0.01         # eye widths an eye may land past its own lens position


def per_eye_limits(corr, s_e, eye_ok, weight, gaze_f, lens_m, fps, s: Settings):
    """Per-eye corrections that never push an iris past where THAT eye sits when the
    speaker really looks into the lens.

    `corr` moves both irises by the same amount, sized from the physical prompter angle.
    The two eyes do not sit alike in their openings — on a 2026-10-03 Raat3Baje take the
    viewer-left iris rested at -0.07 eye widths and the viewer-right at +0.09 when looking
    into the lens — so a push that overshoots the lens by the same amount in both eyes
    carries the off-centre one into its corner while the other just passes through the
    middle: a lazy eye. Each eye's own lens position is measured from the frames the plan
    left alone because the speaker was looking at the camera; with too few of those, the
    shared correction stands. Returns ({"R": corr_R, "L": corr_L}, {"R": lens_R, "L": lens_L}).
    """
    cam = eye_ok & (weight < 0.2) & (np.abs(gaze_f - lens_m) < s.away_lo)
    if cam.sum() < fps * LENS_LOOK_MIN_S:
        return {k: corr.copy() for k in EYES}, {}
    out, lens_e = {}, {}
    for k in EYES:
        lens_k = float(np.median(s_e[k][cam]))
        pos = tv_denoise(fill_gaps(s_e[k], eye_ok), s.tv_lambda)
        landed = pos + corr
        landed = np.where(corr < 0, np.maximum(landed, lens_k - LENS_MARGIN),
                          np.minimum(landed, lens_k + LENS_MARGIN))
        # Only ever a smaller push in the same direction: never reversed, never larger.
        out[k] = np.where(corr < 0, np.clip(landed - pos, corr, 0.0), np.clip(landed - pos, 0.0, corr))
        lens_e[k] = lens_k
    return out, lens_e


def _empty_plan(pts, valid):
    n = len(pts)
    z = np.zeros(n)
    return dict(pts=np.nan_to_num(pts.astype(np.float64)), ex=np.tile([1.0, 0.0], (n, 1)),
                w_e={"R": z + 1, "L": z + 1}, corr=z, corr_e={"R": z, "L": z}, lens_e={}, gaze=z, gaze_f=z, base=z, weight=z,
                eye_ok=np.zeros(n, bool), valid=valid, lens_m=z, camera_pct=0.0, lens_sep_deg=0.0, offset_deg=0.0, symmetry_offset_deg=0.0, sweep_deg=0.0)
