"""Turn per-frame landmarks into a per-frame iris correction (numpy only, offline).

The model of a teleprompter read, in HEAD coordinates (v2, 2026-10-24):
  gaze = eyes in the head + the head itself
       = eye_s + IRIS_SWING * sin(yaw)      sideways, along the inter-ocular axis
       = eye_t + IRIS_SWING * sin(pitch)    up/down (+ = down)
  gaze(t) = prompter direction + slow line sweep (creep along a line, snap back)
            + fast natural movement (micro-saccades, fixational jitter)
The correction moves the irises so that
  output(t) = lens + (1 - steadiness) * slow sweep + fast movement
i.e. the prompter-to-lens offset is removed entirely, the line sweep is calmed by
`steadiness`, and the fast movement always passes at full amplitude (a locked stare looks
dead). The irises' target is the lens minus that frame's head part: the eyes compensate a
turned or bowed head, so the gaze stays on the lens while the head moves.

Why the head: on the 674aefab recording (2026-10-24) the eyes alone moved only ~2.4 deg
between reading and the lens -- the head carries the rest (reading yaw -2.8 deg, pitch
+7.4 deg) -- and "camera looks" picked from the eyes alone were frames where the eyes merely
drifted to the middle of the opening while the head stayed turned; the fix was invisible.
The combined gaze is bimodal, its modes ~8 deg apart, matching the physical geometry.

Where the lens is gets measured (mode="auto") from the take's own looks into the camera:
HOLDS (>= HOLD_S of steady gaze) in the mode of combined gaze nearest zero beside the
reading mode, only on frames allowed to calibrate (eye width within WIDTH_TOL of its
rolling median, the whole face inside the frame, the head not dropped below its reading
pitch: a speaker leaning into the camera with half the face out of frame is never a camera
look). The user may confirm looks instead (`lens_marks`). With no looks at all, the
geometry (manual) or the symmetric guess (auto) is the fallback.

Mirroring: "R"/"L" (landmarks 33/263) are assigned by IMAGE position, so in a mirrored video
the subject's left is the image's other side and `prompter_side` points the wrong way (the
674aefab recording is mirrored). The push direction therefore comes from the measured
reading offset; the configured side only breaks a tie, and a disagreement is reported as
side_check="video looks mirrored".

Everything is non-causal: the whole timeline is known before a pixel is touched,
so blinks are bridged from both sides and saccades stay sharp (total-variation
denoising keeps steps where a causal filter would smear them into glides).
"""
from dataclasses import dataclass, field, replace

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
    mode: str = "manual"          # "auto": everything measured from the recording itself -- the frames
                                  # where the speaker really looks into the lens say where the gaze
                                  # (eyes + head) is then, sideways and up/down, and every reading frame
                                  # is moved there (see `measure_lens`). The geometry below is only the
                                  # fallback when a take has no such looks.
    prompter_side: str = "left"   # side of the lens the prompter is on, from the speaker's seat
                                  # ("left" / "right"), or "auto" to guess from eye symmetry. Only a
                                  # tie-break: the push direction is measured (a mirrored video flips
                                  # the sides; see side_check)
    angle_deg: float = 0.0        # how far (degrees) the prompter is from the lens; when > 0 it
                                  # overrides the distance maths below, and in auto mode it overrides
                                  # the measured sideways separation's size (its direction stays measured)
    prompter_cm: float = 30.0     # sideways distance from the lens to the middle of the text
    camera_cm: float = 110.0      # distance from the speaker's eyes to the camera
    steadiness: float = 0.7       # how much of the slow LINE SWEEP is removed (0 = kept, 1 = removed):
                                  # the creep along a line (below SWEEP_HZ) and its snap-back. It no
                                  # longer damps anything else: the fast natural movement (micro-
                                  # saccades, fixational jitter) always passes at full amplitude
    aim_deg: float = 0.0          # fine nudge of the lens direction, + = toward the image's 263 side
                                  # (the speaker's left unless mirrored); added on top of the measured
                                  # lens in auto mode too
    pitch_deg: float = 0.0        # vertical re-aim of the re-aimed (reading) gaze, + = lower (down),
                                  # - = raise (up); same degree scale as angle_deg / aim_deg, added on
                                  # top of the measured lens in auto mode too
    auto_pitch: bool = True       # raise the reading gaze by the head's own downward pitch (the
                                  # prompter sits below the lens: see head_pitch_deg), pitch_deg on top
    max_shift: float = 0.14       # never move an iris more than this many eye widths (~23 deg)
    past_lens_deg: float = 12.0   # how far past its own measured camera-look position an eye may be
                                  # pushed (the measurement under-reads the real angle; see per_eye_limits)
    away_lo: float = 0.07         # gaze this far from both prompter and lens (eye widths) starts to pass through
    away_hi: float = 0.12         # ...and this far is left completely untouched (a deliberate look away)
    baseline_s: float = 20.0      # window for the prompter direction (seconds), reading frames only
    tv_lambda: float = 0.006      # saccade-preserving denoise strength (eye widths)
    head_depth: float = 0.12      # unused since v2 (2026-10-24): the head's yaw is part of the gaze
                                  # (IRIS_SWING * sin(yaw)) instead of a parallax correction
    lens_marks: list = field(default_factory=list)
                                  # seconds where the speaker looks into the lens (confirmed by the
                                  # user): the lens is measured from +-LENS_MARK_S around each mark
                                  # instead of the automatic cluster (auto mode)

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


#: The slow line sweep is everything in the reading residual below this frequency, plus the
#: snap-back saccades (single-frame steps larger than SNAP_MIN): a sawtooth's snap is sharp,
#: so a low-pass alone leaves half of it behind as a twitch at every line.
SWEEP_HZ = 1.2
SNAP_MIN = 0.006
#: A camera look is a HOLD: the (TV-denoised) gaze stays within HOLD_TOL for HOLD_S.
HOLD_S = 0.4
HOLD_TOL = 0.015
#: Frames whose eye width is off its rolling median by more than this (leaning in, turning
#: away, half out of frame) never calibrate anything.
WIDTH_TOL = 0.2
#: Seconds either side of a user's lens mark that the lens is measured from.
LENS_MARK_S = 0.3
#: The reading offset must be at least this far from the symmetric zero (degrees) before its
#: sign is trusted over the configured prompter side.
SIDE_MIN_DEG = 1.5
CAM_BIN = 0.004            # histogram bin (eye widths) for finding the reading and lens modes


def lowpass(x, fps, hz=SWEEP_HZ):
    """Zero-phase Gaussian low-pass with its half-power point at `hz`."""
    return gauss_smooth(x, fps * np.sqrt(np.log(2.0)) / (2 * np.pi * hz))


def sweep(resid, fps, hz=SWEEP_HZ, snap=SNAP_MIN):
    """The slow line sweep in a reading residual: its snap-back steps (kept sharp, so removing
    them leaves no twitch) plus everything else below `hz`. resid - sweep is the fast part."""
    resid = np.asarray(resid, np.float64)
    if len(resid) == 0:
        return resid.copy()
    d = np.diff(resid, prepend=resid[0])
    steps = np.cumsum(np.where(np.abs(d) > snap, d, 0.0))
    return steps + lowpass(resid - steps, fps, hz)


def _runs(mask):
    m = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)       # [start, end)


def _keep_runs(mask, min_len):
    out = np.zeros(len(mask), bool)
    for a, b in zip(*_runs(mask)):
        if b - a >= min_len:
            out[a:b] = True
    return out


def _rolling_range(x, half):
    from numpy.lib.stride_tricks import sliding_window_view
    xp = np.pad(np.asarray(x, np.float64), (half, half), mode="edge")
    v = sliding_window_view(xp, 2 * half + 1)
    return v.max(1) - v.min(1)


def _histogram(x, frames):
    v = x[frames]
    lo, hi = max(float(v.min()), -0.5), min(float(v.max()), 0.5)
    edges = np.arange(lo - CAM_BIN, hi + 2 * CAM_BIN, CAM_BIN)
    h, _ = np.histogram(v, edges)
    return (edges[:-1] + edges[1:]) / 2, gauss_smooth(h.astype(np.float64), 1.0)


def main_mode(x, frames):
    """Centre of the most populated bin of `x` over `frames` (the reading direction)."""
    if frames.sum() == 0:
        return 0.0
    c, hs = _histogram(x, frames)
    return float(c[np.argmax(hs)])


def lens_region(x, frames, toward):
    """(lo, hi) of the secondary mode of `x` on the `toward` side (+1 / -1) of its main mode,
    or None: the contiguous bins nearest the main mode where the histogram is more than twice
    its own mirror image about the main peak. The reading mode is near-symmetric; the looks
    into the lens are what is left over on one side (a weak shoulder, not always a peak)."""
    if frames.sum() < 3 or toward == 0:
        return None
    c, hs = _histogram(x, frames)
    k = int(np.argmax(hs))
    peak = c[k]
    mirror = np.interp(2 * peak - c, c, hs, left=0.0, right=0.0)
    excess = ((c - peak) * toward > 0) & (hs > 2 * mirror + max(1.0, 0.01 * hs[k]))
    step = 1 if toward > 0 else -1
    i = k + step
    while 0 <= i < len(c):
        while 0 <= i < len(c) and not excess[i]:
            i += step
        run = []
        while 0 <= i < len(c) and excess[i]:
            run.append(i)
            i += step
        if not run:
            return None
        # A stray tail bin right beside the reading mode is not a look into the lens.
        lo, hi = min(c[run]) - CAM_BIN / 2, max(c[run]) + CAM_BIN / 2
        sel = frames & (x >= lo) & (x <= hi)
        if sel.any() and abs(float(np.median(x[sel])) - peak) >= LENS_SEP_MIN_DEG * EYE_WIDTHS_PER_DEGREE:
            return lo, hi
    return None

def camera_looks(gaze_hold, gaze_thold, frames, fps):
    """Frames that are held looks into the lens, or None. Sideways first: the secondary mode
    of the combined gaze on the side of the reading mode where zero (looking down the camera
    axis) lies. A speaker whose prompter sits straight below the lens has no sideways mode;
    their looks into the lens lift the head, so the vertical mode above reading is next."""
    for x, toward in ((gaze_hold, -np.sign(main_mode(gaze_hold, frames))),
                      (gaze_thold, -1.0)):
        region = lens_region(x, frames, toward)
        if region is None:
            continue
        cam = _keep_runs(frames & (x >= region[0]) & (x <= region[1]), max(1, int(round(fps * HOLD_S))))
        if cam.sum() >= fps * LENS_LOOK_MIN_S:
            return cam
    return None


def look_candidates(cam, fps, k=3):
    """Up to k camera looks (seconds, the middle of the longest holds) for the user to confirm."""
    if cam is None or not cam.any():
        return []
    starts, ends = _runs(cam)
    best = np.argsort(-(ends - starts), kind="stable")[:k]
    return [round(float((starts[i] + ends[i]) / 2 / fps), 1) for i in sorted(best)]


def make_plan(pts, score, fps, s: Settings = Settings(), lens_e=None, frame_size=None):
    """pts: (N, len(USED), 3) compact landmarks (NaN where no face), score: (N,) presence logits.
    `lens_e`: the whole recording's lens measurement, for a short clip of it. `frame_size`:
    (width, height) of the frames the landmarks are in (a face must be fully inside to
    calibrate anything).

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

    # Each iris along (s_e) and across (t_e, + = down) the inter-ocular axis, in eye widths.
    s_e, t_e, open_e, w_e = {}, {}, {}, {}
    for k, e in EYES.items():
        a, b = P(raw, e["outer"])[:, :2], P(raw, e["inner"])[:, :2]
        w = np.linalg.norm(b - a, axis=1)
        mid = (a + b) / 2
        s_e[k] = ((P(raw, e["iris"])[:, :2] - mid) * ex).sum(1) / w
        t_e[k] = ((P(raw, e["iris"])[:, :2] - mid) * ey).sum(1) / w
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

    # head turn: the face normal along the eye axis (sin of the yaw)
    right = P(raw, 263) - P(raw, 33)
    down = P(raw, CHIN) - P(raw, FOREHEAD)
    nrm = np.cross(down, right)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if np.nanmedian(nrm[:, 2]) > 0:
        nrm = -nrm
    n_along = gauss_smooth((nrm[:, :2] * ex).sum(1), fps * 0.1)
    # head pitch: + = the face points below the lens (chin further from the camera than the
    # forehead). A 2026-10-08 Raat3Baje take read with the head 9 deg down the whole way, and
    # every frame of it looked below the lens however far the irises were pushed sideways.
    pitch = gauss_smooth(np.degrees(np.arcsin(np.clip(nrm[:, 1], -1.0, 1.0))), fps * 0.1)

    if eye_ok.sum() < 3:
        return _empty_plan(pts, valid)
    # The gaze in head coordinates: the eyes in the head plus the head itself.
    head_s = IRIS_SWING * n_along
    head_t = IRIS_SWING * np.sin(np.radians(pitch))
    gaze = fill_gaps((s_e["R"] + s_e["L"]) / 2 + head_s, eye_ok)
    gaze_t = fill_gaps((t_e["R"] + t_e["L"]) / 2 + head_t, eye_ok)
    gaze_f = tv_denoise(gaze, s.tv_lambda)
    gaze_tf = tv_denoise(gaze_t, s.tv_lambda)
    half, stride = int(fps * s.baseline_s), max(1, int(fps / 4))

    def baseline(frames, x=gaze_f):
        b = rolling_median(x, frames, half, stride)
        return gauss_smooth(fill_gaps(b, np.isfinite(b)), fps * 0.5)

    nudge = s.aim_deg * EYE_WIDTHS_PER_DEGREE
    auto = s.mode == "auto"
    # Which frames look into the camera is judged on the gaze held for a moment: a reading
    # sweep crosses toward the lens and back within a line, a look into the lens stays.
    gaze_hold = gauss_smooth(gaze_f, fps * 0.4)
    gaze_thold = gauss_smooth(gaze_tf, fps * 0.4)

    # Frames allowed to calibrate: eye width near its rolling median, the whole face inside
    # the frame, the head not dropped below its reading pitch. On 674aefab the plan used to
    # trust t=131 s, the speaker leaning into the camera with half the face out of frame.
    width = (w_e["R"] + w_e["L"]) / 2
    w_ref = rolling_median(width, valid, int(fps * 30), max(1, int(fps)))
    w_ref = fill_gaps(w_ref, np.isfinite(w_ref))
    with np.errstate(invalid="ignore"):
        inside = np.all((pts[:, :, 0] >= 0) & (pts[:, :, 1] >= 0), axis=1)
        if frame_size:
            inside &= np.all((pts[:, :, 0] < frame_size[0]) & (pts[:, :, 1] < frame_size[1]), axis=1)
    pitch_b0 = rolling_median(pitch, eye_ok, half, stride)
    pitch_b0 = gauss_smooth(fill_gaps(pitch_b0, np.isfinite(pitch_b0)), fps * 0.5)
    look_ok = (eye_ok & (np.abs(width / np.maximum(w_ref, 1e-6) - 1) < WIDTH_TOL) & inside
               & ((pitch - pitch_b0) < HEAD_PITCH_AWAY_DEG))
    hold = _rolling_range(gaze_f, max(1, int(round(fps * HOLD_S / 2)))) < HOLD_TOL
    cam_auto = camera_looks(gaze_hold, gaze_thold, look_ok & hold, fps)

    side_check = "unclear"
    side = "auto" if auto else s.prompter_side
    if side in ("left", "right"):
        # Reading looks toward the prompter; the lens is that angle back the other way. The
        # prompter direction is measured from reading frames only: a frame nearer the lens than
        # the prompter is the speaker looking into the camera, and a long stretch of that (an
        # intro, a sign-off) must not drag the estimate onto the lens — that would read
        # camera-looking as reading and push those eyes the full angle past the camera.
        # The direction is the measured one (a mirrored video swaps the sides).
        side_sign = -1.0 if s.prompter_side == "left" else 1.0
        sign, side_check = _push_sign(float(np.median(gaze_f[eye_ok])), side_sign)
        swing = IRIS_SWING * np.sin(np.radians(s.prompter_angle_deg()))
        to_lens = sign * swing + nudge                     # how far a reading frame is moved
        sep_lo = LENS_SEP_MIN_DEG * EYE_WIDTHS_PER_DEGREE
        sep_hi = max(swing, sep_lo)

        def lens_at(b):                                    # the lens in the measurement
            return b + sign * np.clip(sign * (0.0 - b), sep_lo, sep_hi)

        base = np.full(n, np.median(gaze_f[eye_ok]))
        reading = eye_ok
        for _ in range(3):
            lens_m = lens_at(base)
            reading = eye_ok & (np.abs(gaze_hold - base) <= np.abs(gaze_hold - lens_m))
            if reading.sum() < 3:
                break
            base = baseline(reading)
        lens_m = lens_at(base)
        lens = base + to_lens
    else:
        reading = eye_ok
        base = baseline(eye_ok)
        # "auto" side: the symmetric position (combined gaze 0: eyes centred, face to the
        # camera) is the lens. In the measured mode that is only the fallback.
        lens = np.full(n, nudge)
        lens_m = np.zeros(n)
        if auto:
            sign, side_check = _push_sign(float(np.median(base[eye_ok])),
                                          -1.0 if s.prompter_side == "left" else 1.0)
            if s.angle_deg > 0:
                lens = base + sign * IRIS_SWING * np.sin(np.radians(s.angle_deg)) + nudge
    # The head's resting pitch while reading, and how far each frame has dropped from it.
    pitch_b = rolling_median(pitch, reading if reading.sum() >= 3 else eye_ok, half, stride)
    pitch_b = gauss_smooth(fill_gaps(pitch_b, np.isfinite(pitch_b)), fps * 0.5)
    pitch_dev = np.abs(pitch - pitch_b)
    # A head dropped well below its reading pitch is looking at notes, the desk or the
    # floor, not at the prompter and not at the lens: left alone, and never taken for a
    # camera look. On the 2026-10-08 take the "camera looks" the plan trusted most were an
    # 18 s stretch between takes with the head 26 deg down and no face in the eye window.
    steady = pitch_dev < HEAD_PITCH_AWAY_DEG
    slow = sweep(gaze_f - base, fps)
    corr_raw, w = reaim(base, lens, lens_m, gaze_f, gaze_hold, n_along, pitch_dev, valid, s, slow=slow)
    corr = corr_raw * w

    measured, base_t, corr_t_raw, cam, lens_source = None, np.zeros(n), None, None, "geometry"
    if auto:
        # The real target: where the gaze (eyes + head) is when the speaker looks into the
        # lens -- the user's marks, else the held looks found above. A short clip of a
        # recording gets the whole recording's measurement (`lens_e`).
        known = lens_e if (lens_e and all(k in lens_e for k in MEASURED_KEYS)) else None
        if known is None and s.lens_marks:
            cam = _mark_frames(s.lens_marks, n, fps) & look_ok
            cam = cam if cam.sum() >= 3 else None
        from_marks = cam is not None
        if cam is None and known is None:
            cam = cam_auto
        rest_f = None
        if cam is not None:
            lc, ltc = float(np.median(gaze[cam])), float(np.median(gaze_t[cam]))
            ps, pt = main_mode(gaze_hold, look_ok), main_mode(gaze_thold, look_ok)
            d_read = np.hypot(gaze_hold - ps, gaze_thold - pt)
            d_lens = np.hypot(gaze_hold - lc, gaze_thold - ltc)
            rest_f = look_ok & ~cam & (d_read <= d_lens)
        measured = measure_lens(known, s_e, t_e, cam, rest_f, gaze, gaze_t, n_along, pitch, fps, from_marks)
        if measured:
            lens_source = "marks" if measured.get("from_marks") else "measured"
            lc, ltc = measured["lens_c"], measured["tc_lens"]
            lens_m = np.full(n, lc)
            # Reading frames are judged against the measured lens (sideways and up/down), and
            # the prompter direction comes from them alone.
            # Anchored on the MEASURED reading position: a clip that is mostly camera looks has
            # no reading of its own, and a self-referential split there invents one (a
            # 2026-10-24 preview lowered camera-look frames 1.5 px toward a phantom "reading").
            d_b = np.hypot(gaze_hold - measured["rest_c"], gaze_thold - measured["tc_rest"])
            d_l = np.hypot(gaze_hold - lc, gaze_thold - ltc)
            reading = eye_ok & (d_b <= d_l)
            if reading.sum() >= 3:
                base, base_t = baseline(reading), baseline(reading, gaze_tf)
            else:
                base, base_t = np.full(n, measured["rest_c"]), np.full(n, measured["tc_rest"])
            sign = 1.0 if measured["lens_c"] >= measured["rest_c"] else -1.0
            if abs(measured["lens_c"] - measured["rest_c"]) >= SIDE_MIN_DEG * EYE_WIDTHS_PER_DEGREE:
                side_check = "ok" if sign == (-1.0 if s.prompter_side == "left" else 1.0) else "video looks mirrored"
            if s.angle_deg > 0:
                # The user's angle sizes the sideways push; its direction stays measured.
                lens = base + sign * IRIS_SWING * np.sin(np.radians(s.angle_deg)) + nudge
            else:
                lens = np.full(n, lc + nudge)
            lens_t = ltc + s.pitch_deg * EYE_WIDTHS_PER_DEGREE
            span_s, span_t = lc - base, ltc - base_t
            toward = ((gaze_hold - base) * span_s + (gaze_thold - base_t) * span_t) / np.maximum(
                span_s ** 2 + span_t ** 2, 1e-12)
            sep = float(np.hypot(measured["lens_c"] - measured["rest_c"], measured["tc_lens"] - measured["tc_rest"]))
            slow = sweep(gaze_f - base, fps)
            slow_t = sweep(gaze_tf - base_t, fps)
            corr_raw, w = reaim(base, lens, lens_m, gaze_f, gaze_hold, n_along, pitch_dev, valid, s,
                                gate_toward=sep >= MIN_MEASURED_SEP, slow=slow, toward=toward)
            corr = corr_raw * w
            corr_t_raw = lens_t - base_t - s.steadiness * slow_t
    if cam is None and measured:
        # A clip measured on its whole recording: its camera looks are the frames left alone.
        cam = eye_ok & (w < 0.1)
    if cam is None:
        cam = cam_auto if cam_auto is not None else np.zeros(n, bool)
    fast = (gaze_f - base) - slow             # the natural movement: never damped, never clamped
    max_shift = AUTO_MAX_SHIFT if (auto and measured) else s.max_shift
    corr = np.clip(corr, -max_shift, max_shift)
    corr[~valid] = 0.0
    if auto and measured:
        # Each eye lands on its own measured lens position, not past it; that position moves
        # with the head (the eyes compensate a turned head) and with the user's nudges.
        limits_s = replace(s, past_lens_deg=AUTO_PAST_LENS_DEG)
        shift = (lens - head_s) - (measured["R"] + measured["L"]) / 2
        corr_e, _ = per_eye_limits(corr, s_e, eye_ok & steady, w, gaze_f, lens_m, fps, limits_s, measured,
                                   shift=shift, fast=fast)
        lens_e = dict(measured)
        # Up/down: the measured difference between reading and the lens, eyes + head, same
        # cap as the modelled version (the warp folds past ~11 deg).
        t_cap = IRIS_SWING * np.sin(np.radians(MAX_PITCH_DEG))
        t_raw = np.clip(corr_t_raw, -t_cap, t_cap)
        corr_t = t_raw * np.asarray(w, np.float64)
        corr_t[~valid] = 0.0
        moved = (w > 0.5) & eye_ok
        t_med = float(np.median(t_raw[moved])) if moved.any() else 0.0
        vertical_deg = round(float(np.degrees(np.arcsin(np.clip(t_med / IRIS_SWING, -1.0, 1.0)))), 1)
    else:
        # The per-eye lens positions are measured where the head was then; elsewhere they
        # move with the head.
        cal = eye_ok & steady & look_ok
        head_cam = (lens_e or {}).get("head_s")
        if head_cam is None:
            c = cal & (w < 0.2) & (np.abs(gaze_f - lens_m) < s.away_lo)
            head_cam = float(np.median(head_s[c])) if c.sum() >= fps * LENS_LOOK_MIN_S else None
        shift = (float(head_cam) - head_s) if head_cam is not None else None
        corr_e, lens_e = per_eye_limits(corr, s_e, eye_ok & steady, w, gaze_f, lens_m, fps, s, lens_e,
                                        shift=shift, fast=fast, cal=cal)
        if lens_e and head_cam is not None:
            lens_e = {**lens_e, "head_s": float(head_cam)}
        corr_t, vertical_deg = vertical_shift(s, w, valid, pitch_b if s.auto_pitch else None)

    return dict(pts=smooth, ex=ex, w_e=w_e, corr=corr, corr_e=corr_e, lens_e=lens_e, corr_t=corr_t,
                gaze=gaze, gaze_f=gaze_f, base=base, gaze_t=gaze_t, base_t=base_t, head_s=head_s,
                weight=w, eye_ok=eye_ok, valid=valid, lens_m=lens_m, cam=cam, look_ok=look_ok,
                head_pitch_deg=float(np.median(pitch_b[eye_ok])) if eye_ok.any() else 0.0,
                vertical_deg=vertical_deg,
                mode=s.mode, auto_measured=bool(auto and measured), lens_source=lens_source,
                side_check=side_check, camera_look_candidates=look_candidates(cam_auto, fps),
                measured_shift_deg=(float((measured["lens_c"] - measured["rest_c"]) / EYE_WIDTHS_PER_DEGREE)
                                    if measured else 0.0),
                camera_pct=float(100.0 * np.mean((w < 0.5)[valid])) if valid.any() else 0.0,
                lens_sep_deg=float(np.median(np.abs(lens_m - base)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                offset_deg=float(np.median((base - lens)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                symmetry_offset_deg=float(np.median(base[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                sweep_deg=float(np.std((gaze_f - base)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0)


def _push_sign(reading_offset, side_sign):
    """The push direction (+1 / -1 along the eye axis) and a side check. The lens is at the
    symmetric zero's side of the reading offset; the configured side only breaks a tie."""
    if abs(reading_offset) < SIDE_MIN_DEG * EYE_WIDTHS_PER_DEGREE:
        return side_sign, "unclear"
    sign = -1.0 if reading_offset > 0 else 1.0
    return sign, ("ok" if sign == side_sign else "video looks mirrored")


def _mark_frames(marks, n, fps):
    out = np.zeros(n, bool)
    r = max(1, int(round(LENS_MARK_S * fps)))
    for t in marks:
        i = int(round(float(t) * fps))
        if 0 <= i < n:
            out[max(0, i - r):i + r + 1] = True
    return out


#: The measured mode may move an iris further than the modelled one (the measurement is
#: the truth, not a guess), but never into the corner.
AUTO_MAX_SHIFT = 0.22
#: ...and each eye stops at its own measured lens position (plus a hair, so the shared
#: push is not undone by the per-eye balance).
AUTO_PAST_LENS_DEG = 1.0
MEASURED_KEYS = ("R", "L", "rest_R", "rest_L", "t_lens", "t_rest", "lens", "rest", "head_raise_deg",
                 "lens_c", "rest_c", "tc_lens", "tc_rest", "yaw_lens", "pitch_lens")
#: A head this much above its reading pitch is up at the lens (a prompter below the lens
#: is read with the head bowed; the look into the camera lifts it).
HEAD_UP_DEG = 3.0


#: Below this separation (eye widths) between the measured reading and lens positions,
#: the gaze cannot tell reading from a camera look.
MIN_MEASURED_SEP = 0.01


def reaim(base, lens, lens_m, gaze_f, gaze_hold, n_along, pitch_dev, valid, s: Settings, gate_toward=True,
          slow=None, toward=None):
    """The shared correction (eye widths, before the weight) and its weight: only reading
    gets re-aimed. Looking into the lens already (or past it), glancing elsewhere, big head
    turns and lost faces are left alone, all with soft edges. The caller multiplies.

    `slow`: the reading residual's slow line sweep (see `sweep`); only it is calmed by
    `steadiness`, the fast natural movement is never subtracted. `toward`: how far each held
    gaze is from the prompter (0) to the lens (1), when the caller measures it in 2-D."""
    corr = lens - base - s.steadiness * ((gaze_f - base) if slow is None else slow)
    if gate_toward:
        if toward is None:
            span = lens_m - base
            span = np.where(np.abs(span) < 1e-6, 1e-6, span)
            toward = (gaze_hold - base) / span             # 0 at the prompter, 1 at the lens
        # Full push only in the prompter's half; none from 60% of the way to the lens. Measured
        # camera looks sat at 0.5-0.75 of the way (a centred stare reads a degree or two short of
        # the lens) and got most of the push from a fade that started at 0.45 -- the overshoot.
        w = 1 - smoothstep((toward - 0.3) / 0.3)
    else:
        w = np.ones(len(gaze_f))
    dev_p = np.abs(gaze_f - base)
    dev_l = np.abs(gaze_f - lens_m)
    w *= 1 - smoothstep((np.minimum(dev_p, dev_l) - s.away_lo) / (s.away_hi - s.away_lo))
    w *= 1 - smoothstep((np.abs(n_along) - 0.30) / 0.15)
    w *= 1 - smoothstep((pitch_dev - HEAD_PITCH_AWAY_DEG) / HEAD_PITCH_AWAY_DEG)
    w *= 1 - np.clip(gauss_smooth((~valid).astype(float), 3.0) * 3, 0, 1)
    w = gauss_smooth(w, 1.0)
    return corr, w


def measure_lens(known, s_e, t_e, cam, rest, gaze, gaze_t, yaw_s, pitch, fps, from_marks=False):
    """Where the gaze is when the speaker looks into the lens, measured from the take.

    `cam`: the camera-look frames (held looks found by `camera_looks`, or the user's marks);
    `rest`: the reading frames to compare them with. Returns, in eye widths, {"R", "L": each
    eye's sideways position in its opening at the lens; "rest_R", "rest_L": while reading;
    "lens", "rest": their means; "t_lens", "t_rest": the irises' vertical positions (+ = down);
    "lens_c", "rest_c", "tc_lens", "tc_rest": the combined (eyes + head) gaze, sideways and
    vertical, at the lens and while reading; "yaw_lens", "pitch_lens": the head at the lens
    (deg); "head_raise_deg": how far the head comes up for the lens; "from_marks": 1.0 when the
    looks were the user's}, or None when the take holds fewer than LENS_LOOK_MIN_S seconds of
    camera looks (any amount for marks). `known` (the same dict from the whole recording) is
    used as is for a short clip of it."""
    if known and all(k in known for k in MEASURED_KEYS):
        return {k: float(v) for k, v in known.items() if isinstance(v, (int, float, np.floating))}
    if cam is None or rest is None:
        return None
    if cam.sum() < (3 if from_marks else fps * LENS_LOOK_MIN_S) or rest.sum() < 3:
        return None
    out = {}
    for k in EYES:
        out[k] = float(np.median(s_e[k][cam]))
        out["rest_" + k] = float(np.median(s_e[k][rest]))
    out["lens"] = float(np.mean([out["R"], out["L"]]))
    out["rest"] = float(np.mean([out["rest_R"], out["rest_L"]]))
    out["t_lens"] = float(np.mean([np.median(t_e[k][cam]) for k in EYES]))
    out["t_rest"] = float(np.mean([np.median(t_e[k][rest]) for k in EYES]))
    out["lens_c"], out["rest_c"] = float(np.median(gaze[cam])), float(np.median(gaze[rest]))
    out["tc_lens"], out["tc_rest"] = float(np.median(gaze_t[cam])), float(np.median(gaze_t[rest]))
    out["yaw_lens"] = float(np.degrees(np.arcsin(np.clip(np.median(yaw_s[cam]), -1.0, 1.0))))
    out["pitch_lens"] = float(np.median(pitch[cam]))
    out["head_raise_deg"] = float(np.median(pitch[rest]) - out["pitch_lens"])
    out["from_marks"] = 1.0 if from_marks else 0.0
    return out


MAX_PITCH_DEG = 10.0       # the vertical field folds past ~11 deg (warp clamps it there anyway)
HEAD_PITCH_AWAY_DEG = 6.0  # a head this far below (or above) its reading pitch is looking elsewhere


def vertical_shift(s: Settings, weight, valid, head_pitch=None):
    """Per-frame vertical iris shift in eye widths, + = down (across the inter-ocular axis),
    and the median degrees applied (+ = down) for the report.

    The prompter sits below the lens as often as beside it, and the speaker reads it with the
    head bowed by that angle while the eyes stay neutral in the head; `head_pitch` (degrees,
    + = facing below the lens, per frame, already smoothed) is that bow, and the gaze is
    raised by it so it meets the lens. `pitch_deg` is the manual nudge on top, by the same
    degree scale the sideways aim uses. Both ride on the sideways correction's weight: only
    re-aimed reading frames move, and a real look into the lens (or away) stays where it was.
    The warp folds past ~11 deg, so the total is capped at MAX_PITCH_DEG either way."""
    weight = np.asarray(weight, np.float64)
    total = np.full(len(weight), float(s.pitch_deg))
    if head_pitch is not None:
        total = total - np.asarray(head_pitch, np.float64)
    total = np.clip(total, -MAX_PITCH_DEG, MAX_PITCH_DEG)
    out = IRIS_SWING * np.sin(np.radians(total)) * weight
    out[~valid] = 0.0
    moved = weight > 0.5
    applied = float(np.median(total[moved])) if moved.any() else 0.0
    return out, round(applied, 1)


LENS_LOOK_MIN_S = 2.0      # seconds of real camera looks needed to trust each eye's lens position
LENS_MARGIN = 0.01         # eye widths an eye may land past its own lens position


def per_eye_limits(corr, s_e, eye_ok, weight, gaze_f, lens_m, fps, s: Settings, known=None,
                   shift=None, fast=None, cal=None):
    """Per-eye corrections: the shared push, balanced between the eyes and bounded past each
    eye's own camera-look position.

    `corr` moves both irises by the same amount, sized from the physical prompter angle and
    the aim. The two eyes do not sit alike in their openings — on a 2026-10-03 Raat3Baje take
    the viewer-left iris rested at -0.07 eye widths and the viewer-right at +0.09 when looking
    into the lens — so each eye is pushed by the shared amount plus the extra travel its own
    lens position asks for relative to the other eye's, and never further than
    `past_lens_deg` beyond where it was measured when the speaker looked into the camera.

    Until 2026-10-08 an eye STOPPED at its measured camera-look position. The measurement
    under-reads the real angle (the iris landmarks compress it, and most "camera looks" are
    glances that fall short of the lens), so that cap silently threw away the aim: on
    Raat3Baje ep2 every setting from 10 to 20 deg produced the same ~10 deg push, and the
    creator could not get the eyes onto the lens however far he pushed. The margin lets the
    physical angle and the aim mean what they say, while the balance still keeps one eye from
    running into its corner while the other passes through the middle.

    Each eye's lens position is measured from the frames the plan left alone because the
    speaker was looking at the camera (head steady); with too few of those, the shared
    correction stands. Returns ({"R": corr_R, "L": corr_L}, {"R": lens_R, "L": lens_L}).

    `known` ({"R", "L"}) supplies the lens positions measured over the WHOLE recording, for a
    short clip of it (a preview, a sample): 30 s of reading rarely holds the 2 s of camera looks
    needed.

    `shift` (per frame, eye widths) moves every eye's lens position with the head and the
    user's nudges (the eyes compensate a turned head, so the lens sits elsewhere in the
    opening); `fast` (per frame) is the natural movement, taken out of each eye's position
    before the limit so it rides on top untouched; `cal` restricts which frames may measure
    the lens positions (default `eye_ok`).
    """
    if known and all(k in known for k in EYES):
        lens_e = {k: float(known[k]) for k in EYES}
    else:
        cam = (eye_ok if cal is None else cal) & (weight < 0.2) & (np.abs(gaze_f - lens_m) < s.away_lo)
        if cam.sum() < fps * LENS_LOOK_MIN_S:
            return {k: corr.copy() for k in EYES}, {}
        lens_e = {k: float(np.median(s_e[k][cam])) for k in EYES}
    margin = max(0.0, float(s.past_lens_deg)) * EYE_WIDTHS_PER_DEGREE
    pos_e = {k: tv_denoise(fill_gaps(s_e[k], eye_ok), s.tv_lambda) for k in EYES}
    if fast is not None:
        pos_e = {k: v - np.asarray(fast, np.float64) for k, v in pos_e.items()}
    off = 0.0 if shift is None else np.asarray(shift, np.float64)
    # Each eye's resting place while reading, and the extra travel its own lens asks for.
    rest = {k: float(np.median(pos_e[k][eye_ok])) if eye_ok.any() else 0.0 for k in EYES}
    lens_mean = float(np.mean(list(lens_e.values())))
    rest_mean = float(np.mean(list(rest.values())))
    out = {}
    for k in EYES:
        extra = (lens_e[k] - lens_mean) - (rest[k] - rest_mean)
        corr_k = corr + extra * np.asarray(weight, np.float64)
        landed = pos_e[k] + corr_k
        lens_k = lens_e[k] + off
        landed = np.where(corr_k < 0, np.maximum(landed, lens_k - margin),
                          np.minimum(landed, lens_k + margin))
        # Never reversed, never larger than the balanced push.
        out[k] = np.where(corr_k < 0, np.clip(landed - pos_e[k], corr_k, 0.0),
                          np.clip(landed - pos_e[k], 0.0, corr_k))
    return out, lens_e


def _empty_plan(pts, valid):
    n = len(pts)
    z = np.zeros(n)
    return dict(pts=np.nan_to_num(pts.astype(np.float64)), ex=np.tile([1.0, 0.0], (n, 1)),
                w_e={"R": z + 1, "L": z + 1}, corr=z, corr_e={"R": z, "L": z}, lens_e={}, corr_t=z, gaze=z, gaze_f=z, base=z, weight=z,
                eye_ok=np.zeros(n, bool), valid=valid, lens_m=z, camera_pct=0.0, lens_sep_deg=0.0, offset_deg=0.0, symmetry_offset_deg=0.0, sweep_deg=0.0,
                head_pitch_deg=0.0, vertical_deg=0.0, mode="manual", auto_measured=False,
                gaze_t=z, base_t=z, head_s=z, cam=np.zeros(n, bool), look_ok=np.zeros(n, bool),
                lens_source="geometry", side_check="unclear", camera_look_candidates=[],
                measured_shift_deg=0.0)
