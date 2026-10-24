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
from dataclasses import dataclass, replace

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
                                  # where the speaker really looks into the lens say where each iris
                                  # sits then, sideways and up/down, and every reading frame is moved
                                  # there (see `measure_lens`). The prompter geometry below is only the
                                  # fallback when a take has no such looks.
    prompter_side: str = "left"   # side of the lens the prompter is on, from the speaker's seat
                                  # ("left" / "right"), or "auto" to guess from eye symmetry
    angle_deg: float = 0.0        # how far (degrees) the prompter is from the lens; when > 0 it
                                  # overrides the distance maths below
    prompter_cm: float = 30.0     # sideways distance from the lens to the middle of the text
    camera_cm: float = 110.0      # distance from the speaker's eyes to the camera
    steadiness: float = 0.7       # 0 = keep the reading sweep, 1 = lock the eyes on the lens
    aim_deg: float = 0.0          # fine nudge of the lens direction, + = toward the speaker's left
    pitch_deg: float = 0.0        # vertical re-aim of the re-aimed (reading) gaze, + = lower (down),
                                  # - = raise (up); same degree scale as angle_deg / aim_deg
    auto_pitch: bool = True       # raise the reading gaze by the head's own downward pitch (the
                                  # prompter sits below the lens: see head_pitch_deg), pitch_deg on top
    max_shift: float = 0.14       # never move an iris more than this many eye widths (~23 deg)
    past_lens_deg: float = 12.0   # how far past its own measured camera-look position an eye may be
                                  # pushed (the measurement under-reads the real angle; see per_eye_limits)
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


def make_plan(pts, score, fps, s: Settings = Settings(), lens_e=None):
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
    # head pitch: + = the face points below the lens (chin further from the camera than the
    # forehead). The eyes sit much the same in the head whether the speaker reads or looks
    # up, so where the gaze is VERTICALLY is mostly where the head points: a 2026-10-08
    # Raat3Baje take read with the head 9 deg down the whole way, and every frame of it
    # looked below the lens however far the irises were pushed sideways.
    pitch = gauss_smooth(np.degrees(np.arcsin(np.clip(nrm[:, 1], -1.0, 1.0))), fps * 0.1)

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
    auto = s.mode == "auto"
    # Which frames look into the camera is judged on the gaze held for a moment: a reading
    # sweep crosses toward the lens and back within a line, a look into the lens stays.
    gaze_hold = gauss_smooth(gaze_f, fps * 0.4)
    # Each iris across the inter-ocular axis (+ = down), in eye widths: the vertical
    # counterpart of s_e, for the measured mode.
    t_e = {}
    for k, e in EYES.items():
        a, b = P(raw, e["outer"])[:, :2], P(raw, e["inner"])[:, :2]
        w_k = np.linalg.norm(b - a, axis=1)
        t_e[k] = ((P(raw, e["iris"])[:, :2] - (a + b) / 2) * ey).sum(1) / w_k
    side = "auto" if auto else s.prompter_side
    if side in ("left", "right"):
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
        # "auto" side: the eyes' symmetric position is the lens. In the measured mode the
        # aim nudge is not applied here: the real lens position is measured below.
        lens = np.full(n, 0.0 if auto else nudge)
        lens_m = np.zeros(n)
    # The head's resting pitch while reading, and how far each frame has dropped from it.
    pitch_b = rolling_median(pitch, reading if reading.sum() >= 3 else eye_ok, half, stride)
    pitch_b = gauss_smooth(fill_gaps(pitch_b, np.isfinite(pitch_b)), fps * 0.5)
    pitch_dev = np.abs(pitch - pitch_b)
    # A head dropped well below its reading pitch is looking at notes, the desk or the
    # floor, not at the prompter and not at the lens: left alone, and never taken for a
    # camera look. On the 2026-10-08 take the "camera looks" the plan trusted most were an
    # 18 s stretch between takes with the head 26 deg down and no face in the eye window.
    steady = pitch_dev < HEAD_PITCH_AWAY_DEG
    corr_raw, w = reaim(base, lens, lens_m, gaze_f, gaze_hold, n_along, pitch_dev, valid, s)
    corr = corr_raw * w

    measured = None
    if auto:
        # The real target: where each iris sits when the speaker looks into the lens, from
        # the frames the symmetric guess above left alone as camera looks (head steady).
        # A short clip of a recording gets the whole recording's measurement (`lens_e`).
        measured = measure_lens(lens_e, s_e, t_e, eye_ok & steady, w, gaze_f, lens_m, reading, fps, s,
                                pitch=pitch, pitch_b=pitch_b, eye_open=eye_ok)
        if measured:
            lens = np.full(n, measured["lens"])
            lens_m = lens.copy()
            # Reading frames are judged against the measured lens now, and the prompter
            # direction comes from them alone.
            reading = eye_ok & (np.abs(gaze_hold - base) <= np.abs(gaze_hold - lens_m))
            if reading.sum() >= 3:
                base = baseline(reading)
            # With the eyes sitting (almost) the same for reading and for the lens, "how far
            # toward the lens" says nothing: the head says which is which instead.
            sep = abs(float(measured["lens"] - measured["rest"]))
            corr_raw, w = reaim(base, lens, lens_m, gaze_f, gaze_hold, n_along, pitch_dev, valid, s,
                                gate_toward=sep >= MIN_MEASURED_SEP)
            if measured.get("head_raise_deg", 0.0) > 0:
                # A head up at the lens is a camera look: left alone, with a soft edge.
                w = w * (1 - smoothstep(((pitch_b - pitch) - 0.5 * HEAD_UP_DEG) / HEAD_UP_DEG))
            corr = corr_raw * w
    max_shift = AUTO_MAX_SHIFT if (auto and measured) else s.max_shift
    corr = np.clip(corr, -max_shift, max_shift)
    corr[~valid] = 0.0
    if auto and measured:
        # Each eye lands on its own measured lens position, not past it.
        limits_s = replace(s, past_lens_deg=AUTO_PAST_LENS_DEG)
        corr_e, lens_e = per_eye_limits(corr, s_e, eye_ok & steady, w, gaze_f, lens_m, fps, limits_s, measured)
        lens_e = dict(measured)
        # Up/down: the measured difference between reading and the lens, same cap as the
        # modelled version (the warp folds past ~11 deg).
        t_cap = IRIS_SWING * np.sin(np.radians(MAX_PITCH_DEG))
        # The irises' own vertical difference, plus the head's: a speaker who looks into
        # the lens by lifting the head (eyes neutral in it) reads with the head bowed by
        # `head_raise_deg`, and the eyes have to come up by that much to meet the lens.
        head_raise = float(measured.get("head_raise_deg", 0.0))
        t_delta = float(np.clip((measured["t_lens"] - measured["t_rest"])
                                - IRIS_SWING * np.sin(np.radians(head_raise)), -t_cap, t_cap))
        corr_t = t_delta * np.asarray(w, np.float64)
        corr_t[~valid] = 0.0
        vertical_deg = round(float(np.degrees(np.arcsin(np.clip(t_delta / IRIS_SWING, -1.0, 1.0)))), 1)
    else:
        corr_e, lens_e = per_eye_limits(corr, s_e, eye_ok & steady, w, gaze_f, lens_m, fps, s, lens_e)
        corr_t, vertical_deg = vertical_shift(s, w, valid, pitch_b if s.auto_pitch else None)

    return dict(pts=smooth, ex=ex, w_e=w_e, corr=corr, corr_e=corr_e, lens_e=lens_e, corr_t=corr_t,
                gaze=gaze, gaze_f=gaze_f, base=base,
                weight=w, eye_ok=eye_ok, valid=valid, lens_m=lens_m,
                head_pitch_deg=float(np.median(pitch_b[eye_ok])) if eye_ok.any() else 0.0,
                vertical_deg=vertical_deg,
                mode=s.mode, auto_measured=bool(auto and measured),
                measured_shift_deg=(float((measured["lens"] - measured["rest"]) / EYE_WIDTHS_PER_DEGREE)
                                    if measured else 0.0),
                camera_pct=float(100.0 * np.mean((w < 0.5)[valid])) if valid.any() else 0.0,
                lens_sep_deg=float(np.median(np.abs(lens_m - base)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                offset_deg=float(np.median((base - lens)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                symmetry_offset_deg=float(np.median(base[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0,
                sweep_deg=float(np.std((gaze_f - base)[eye_ok]) / EYE_WIDTHS_PER_DEGREE) if eye_ok.any() else 0.0)


#: The measured mode may move an iris further than the modelled one (the measurement is
#: the truth, not a guess), but never into the corner.
AUTO_MAX_SHIFT = 0.22
#: ...and each eye stops at its own measured lens position (plus a hair, so the shared
#: push is not undone by the per-eye balance).
AUTO_PAST_LENS_DEG = 1.0
MEASURED_KEYS = ("R", "L", "rest_R", "rest_L", "t_lens", "t_rest", "lens", "rest", "head_raise_deg")
#: A head this much above its reading pitch is up at the lens (a prompter below the lens
#: is read with the head bowed; the look into the camera lifts it).
HEAD_UP_DEG = 3.0


#: Below this sideways separation (eye widths) between the measured reading and lens
#: positions, the eyes alone cannot tell reading from a camera look.
MIN_MEASURED_SEP = 0.01


def reaim(base, lens, lens_m, gaze_f, gaze_hold, n_along, pitch_dev, valid, s: Settings, gate_toward=True):
    """The shared correction (eye widths, before the weight) and its weight: only reading
    gets re-aimed. Looking into the lens already (or past it), glancing elsewhere, big head
    turns and lost faces are left alone, all with soft edges. The caller multiplies."""
    corr = lens - base - s.steadiness * (gaze_f - base)
    if gate_toward:
        span = lens_m - base
        span = np.where(np.abs(span) < 1e-6, 1e-6, span)
        toward_lens = (gaze_hold - base) / span            # 0 at the prompter, 1 at the lens
        # Full push only in the prompter's half; none from 60% of the way to the lens. Measured
        # camera looks sat at 0.5-0.75 of the way (a centred stare reads a degree or two short of
        # the lens) and got most of the push from a fade that started at 0.45 -- the overshoot.
        w = 1 - smoothstep((toward_lens - 0.3) / 0.3)
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


def measure_lens(known, s_e, t_e, eye_ok, weight, gaze_f, lens_m, reading, fps, s: Settings,
                 pitch=None, pitch_b=None, eye_open=None):
    """Where the irises sit when the speaker looks into the lens, measured from the take.

    A look into the lens is either the eyes swinging to it with the head steady (the
    symmetric guess's own camera looks) or the head lifting to it from its reading bow
    (`pitch` HEAD_UP_DEG above `pitch_b`, eyes open): a prompter below the lens is read
    with the head bowed, and that speaker's eyes never move at all.

    Returns {"R", "L": each eye's sideways lens position; "rest_R", "rest_L": each eye's
    reading position; "lens", "rest": their means; "t_lens", "t_rest": the vertical
    positions (+ = down); "head_raise_deg": how far the head comes up for the lens} in eye
    widths, or None when the take holds fewer than LENS_LOOK_MIN_S seconds of camera looks.
    `known` (the same dict from the whole recording) is used as is for a short clip of it."""
    if known and all(k in known for k in MEASURED_KEYS):
        return {k: float(known[k]) for k in MEASURED_KEYS}
    cam = eye_ok & (weight < 0.2) & (np.abs(gaze_f - lens_m) < s.away_lo)
    head_up = np.zeros(len(cam), bool)
    if pitch is not None and pitch_b is not None:
        head_up = (eye_open if eye_open is not None else eye_ok) & ((pitch_b - pitch) > HEAD_UP_DEG)
        cam = cam | head_up
    rest_frames = eye_ok & reading & ~cam
    if cam.sum() < fps * LENS_LOOK_MIN_S or rest_frames.sum() < 3:
        return None
    out = {"head_raise_deg": 0.0}
    if head_up.sum() >= fps * LENS_LOOK_MIN_S and pitch is not None:
        out["head_raise_deg"] = float(max(0.0, np.median(pitch_b[rest_frames]) - np.median(pitch[head_up])))
    for k in EYES:
        out[k] = float(np.median(s_e[k][cam]))
        out["rest_" + k] = float(np.median(s_e[k][rest_frames]))
    out["lens"] = float(np.mean([out["R"], out["L"]]))
    out["rest"] = float(np.mean([out["rest_R"], out["rest_L"]]))
    out["t_lens"] = float(np.mean([np.median(t_e[k][cam]) for k in EYES]))
    out["t_rest"] = float(np.mean([np.median(t_e[k][rest_frames]) for k in EYES]))
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


def per_eye_limits(corr, s_e, eye_ok, weight, gaze_f, lens_m, fps, s: Settings, known=None):
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
    """
    if known and all(k in known for k in EYES):
        lens_e = {k: float(known[k]) for k in EYES}
    else:
        cam = eye_ok & (weight < 0.2) & (np.abs(gaze_f - lens_m) < s.away_lo)
        if cam.sum() < fps * LENS_LOOK_MIN_S:
            return {k: corr.copy() for k in EYES}, {}
        lens_e = {k: float(np.median(s_e[k][cam])) for k in EYES}
    margin = max(0.0, float(s.past_lens_deg)) * EYE_WIDTHS_PER_DEGREE
    pos_e = {k: tv_denoise(fill_gaps(s_e[k], eye_ok), s.tv_lambda) for k in EYES}
    # Each eye's resting place while reading, and the extra travel its own lens asks for.
    rest = {k: float(np.median(pos_e[k][eye_ok])) if eye_ok.any() else 0.0 for k in EYES}
    lens_mean = float(np.mean(list(lens_e.values())))
    rest_mean = float(np.mean(list(rest.values())))
    out = {}
    for k in EYES:
        extra = (lens_e[k] - lens_mean) - (rest[k] - rest_mean)
        corr_k = corr + extra * np.asarray(weight, np.float64)
        landed = pos_e[k] + corr_k
        lens_k = lens_e[k]
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
                measured_shift_deg=0.0)
