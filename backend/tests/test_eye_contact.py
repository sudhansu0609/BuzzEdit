"""Tests for the teleprompter eye-contact correction.

The GPU passes need a real recording and an NVIDIA card (backend/tools/eye_contact.py
exercises those); what is tested here is everything that decides *what* happens to
the eyes: the per-frame plan, the shape of the warp field, and the project swap.
"""
import numpy as np
import pytest
import torch

from eyecontact.plan import (COL, EYES, EYE_WIDTHS_PER_DEGREE, IRIS_SWING, USED, Settings, make_plan,
                             tv_denoise)
from eyecontact.warp import K, EyeGeom, eye_params, field

FPS = 24.0
EYE_W = 66.0


def _tv_reference(y, lam, iters=40000):
    """Slow dual projected-gradient solver for the same objective."""
    z = np.zeros(len(y) - 1)
    for _ in range(iters):
        x = y.copy(); x[:-1] += z; x[1:] -= z
        z = np.clip(z + 0.25 * (x[1:] - x[:-1]), -lam, lam)
    x = y.copy(); x[:-1] += z; x[1:] -= z
    return x


def test_tv_denoise_is_exact_and_keeps_steps():
    rng = np.random.default_rng(1)
    clean = np.repeat(rng.normal(0, 0.03, 6), 15)
    y = clean + rng.normal(0, 0.003, len(clean))
    x = tv_denoise(y, 0.006)
    assert np.abs(x - _tv_reference(y, 0.006)).max() < 1e-6
    # a saccade stays a step: the jump lands within one frame of where it was
    jumps = np.flatnonzero(np.abs(np.diff(clean)) > 0.02)
    for j in jumps:
        assert abs(x[j + 1] - x[j]) > 0.5 * abs(clean[j + 1] - clean[j])


def _face(gaze_px, blink=False):
    """Compact landmarks for one frontal face; both irises shifted by gaze_px along x."""
    p = np.zeros((len(USED), 3))

    def put(idx, x, y):
        p[COL[idx]] = (x, y, 0.0)

    for k, (x0, x1) in {"R": (100, 166), "L": (220, 286)}.items():
        e = EYES[k]
        outer, inner = (x0, x1) if k == "R" else (x1, x0)
        put(e["outer"], outer, 100); put(e["inner"], inner, 100)
        open_h = 1.0 if blink else 12.0
        for j, idx in enumerate(e["upper"]):
            s = (j + 1) / (len(e["upper"]) + 1)
            x = outer + (inner - outer) * s
            put(idx, x, 100 - open_h * np.sin(np.pi * s))
        for j, idx in enumerate(e["lower"]):
            s = (j + 1) / (len(e["lower"]) + 1)
            x = outer + (inner - outer) * s
            put(idx, x, 100 + (1.0 if blink else 10.0) * np.sin(np.pi * s))
        cx = (x0 + x1) / 2 + gaze_px + (np.random.default_rng().normal(0, 6) if blink else 0)
        put(e["iris"], cx, 98)
        for j, idx in enumerate(e["ring"]):
            a = j * np.pi / 2
            put(idx, cx + 14 * np.cos(a), 98 + 14 * np.sin(a))
    put(10, 193, 0); put(152, 193, 300)
    return p


def _reading(seconds=24, offset=0.03, sweep=0.015, blink_every=4.0):
    """Gaze offset toward a prompter plus a reading sawtooth, with regular blinks."""
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    saw = sweep * (1 - 2 * ((t % 1.2) / 1.2))       # creep along a line, snap back
    gaze = offset + saw
    blinks = np.zeros(n, bool)
    for b in np.arange(2.0, seconds, blink_every):
        blinks[int(b * FPS):int(b * FPS) + 3] = True
    pts = np.stack([_face(g * EYE_W, bl) for g, bl in zip(gaze, blinks)])
    return pts, gaze, blinks


def test_plan_reaims_by_the_prompter_angle_toward_the_lens():
    """Prompter 30 cm to the speaker's left of a lens 110 cm away: reading looks ~15 degrees
    toward their left, so the irises go ~15 degrees back toward their right (-ex)."""
    pts, gaze, blinks = _reading()
    s = Settings(prompter_side="left", prompter_cm=30, camera_cm=110, steadiness=0.7)
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, s)
    ok = plan["eye_ok"] & ~blinks
    swing = IRIS_SWING * np.sin(np.arctan2(30, 110))
    out = gaze + plan["corr"]
    assert abs(np.median(out[ok]) - (np.median(gaze[ok]) - swing)) < 0.006
    assert abs(plan["offset_deg"] - np.degrees(np.arctan2(30, 110))) < 1.0
    right = make_plan(pts, np.full(len(pts), 10.0), FPS,
                      Settings(prompter_side="right", prompter_cm=30, camera_cm=110))
    assert np.median(right["corr"][ok]) > 0 > np.median(plan["corr"][ok])


def test_looking_into_the_lens_is_left_alone():
    """Mostly reading, but one stretch looking straight into the lens: that stretch must not be
    pushed further, or the eyes would overshoot past the camera."""
    pts, gaze, _ = _reading(blink_every=100)
    s = Settings(prompter_side="left", prompter_cm=30, camera_cm=110)
    swing = IRIS_SWING * np.sin(np.arctan2(30, 110))
    lens_gaze = np.median(gaze) - swing
    pts[300:340] = np.stack([_face(lens_gaze * EYE_W)] * 40)
    corr = make_plan(pts, np.full(len(pts), 10.0), FPS, s)["corr"]
    assert np.abs(corr[308:332]).max() < 0.2 * swing


def test_long_camera_look_is_not_mistaken_for_reading():
    """A 10 s stretch straight into the lens (an intro, say) is longer than any short running
    window: the prompter estimate must still come from the reading around it."""
    pts, gaze, _ = _reading(seconds=40, blink_every=100)
    s = Settings(prompter_side="left", angle_deg=10)
    swing = IRIS_SWING * np.sin(np.radians(10))
    reading_dir = np.median(gaze)
    look = slice(int(12 * FPS), int(22 * FPS))
    gaze = gaze.copy()
    gaze[look] = reading_dir - swing
    pts[look] = np.stack([_face((reading_dir - swing) * EYE_W)] * (look.stop - look.start))
    corr = make_plan(pts, np.full(len(pts), 10.0), FPS, s)["corr"]
    inner = slice(look.start + 12, look.stop - 12)
    assert np.abs(corr[inner]).max() < 0.1 * swing                     # camera look left alone
    assert abs(np.median(corr[: look.start - 24]) + swing) < 0.3 * swing  # reading still re-aimed
    assert abs(np.median(corr[look.stop + 24:]) + swing) < 0.3 * swing


def test_camera_look_is_found_where_the_eyes_measure_it_not_where_the_push_aims():
    """The creator's real setup (2026-10-02): reading measures ~4.5 deg off the lens, but the
    push that looks right is 19 deg (a 10 deg prompter + a -9 deg aim). Judged by the push,
    no frame was ever near the lens, every frame got the full 19 deg, and looks into the
    camera overshot it. A look at the measured lens (gaze ~0) must be left alone, while
    reading still gets the full push."""
    reading_dir = 4.5 * EYE_WIDTHS_PER_DEGREE
    pts, gaze, _ = _reading(seconds=40, offset=reading_dir, sweep=0.006, blink_every=100)
    look = slice(int(12 * FPS), int(16 * FPS))
    gaze = gaze.copy()
    gaze[look] = 0.3 * EYE_WIDTHS_PER_DEGREE
    pts[look] = np.stack([_face(gaze[look][0] * EYE_W)] * (look.stop - look.start))
    s = Settings(prompter_side="left", angle_deg=10, aim_deg=-9)
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, s)
    push = IRIS_SWING * np.sin(np.radians(10)) + 9 * EYE_WIDTHS_PER_DEGREE
    inner = slice(look.start + 12, look.stop - 12)
    assert np.abs(plan["corr"][inner]).max() < 0.1 * push              # the camera look is left alone
    assert abs(np.median(plan["corr"][: look.start - 24]) + push) < 0.3 * push   # reading re-aimed
    assert abs(plan["lens_sep_deg"] - 4.5) < 1.0
    assert plan["camera_pct"] > 5


def test_auto_mode_removes_the_symmetric_offset_and_damps_the_sweep():
    pts, gaze, blinks = _reading()
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(prompter_side="auto", steadiness=0.7))
    ok = plan["eye_ok"] & ~blinks
    out = gaze + plan["corr"]
    assert abs(plan["offset_deg"] - 0.03 / EYE_WIDTHS_PER_DEGREE) < 0.8
    assert abs(np.median(out[ok])) < 0.004                  # centred on the lens
    sweep_in = np.std(gaze[ok] - np.median(gaze[ok]))
    assert np.std(out[ok]) < 0.45 * sweep_in                # sweep damped (~70%)
    assert np.std(out[ok]) > 0.1 * sweep_in                 # ...but not a dead stare
    assert not plan["eye_ok"][blinks].any()


def test_plan_is_continuous_through_blinks():
    pts, gaze, blinks = _reading()
    corr = make_plan(pts, np.full(len(pts), 10.0), FPS)["corr"]
    idx = np.flatnonzero(blinks)
    # garbage iris positions inside a blink must not leak into the correction
    for i in idx:
        assert abs(corr[i] - corr[i - 1]) < 0.01


def test_plan_leaves_lost_faces_and_look_aways_alone():
    pts, gaze, _ = _reading(blink_every=100)
    score = np.full(len(pts), 10.0)
    score[200:240] = -5                                     # face lost
    pts[300:360] = np.stack([_face(0.2 * EYE_W)] * 60)      # a big deliberate look away
    plan = make_plan(pts, score, FPS)
    assert np.all(plan["corr"][205:235] == 0)
    assert np.abs(plan["corr"][315:345]).max() < 0.005


def test_plan_without_a_face_is_a_no_op():
    pts = np.full((50, len(USED), 3), np.nan)
    plan = make_plan(pts, np.full(50, -5.0), FPS)
    assert not plan["corr"].any()


def _eye(c_s=3.0):
    g = EyeGeom(mid=(500, 300), ex=(1, 0), corners=[(467, 300), (533, 300)],
                upper=[(467 + 66 * s, 300 - 12 * np.sin(np.pi * s)) for s in np.linspace(0.1, 0.9, 7)],
                lower=[(467 + 66 * s, 300 + 10 * np.sin(np.pi * s)) for s in np.linspace(0.1, 0.9, 7)],
                iris=(500, 298), r=14)
    return torch.tensor([eye_params(g, c_s, EYE_W, 0, 0)], dtype=torch.float32)


def _d(p, pts):
    x = torch.tensor([[[q[0] for q in pts]]], dtype=torch.float32)
    y = torch.tensor([[[q[1] for q in pts]]], dtype=torch.float32)
    dx, dy = field(p, x, y)
    return dx[0, 0].numpy(), dy[0, 0].numpy()


def test_field_moves_the_iris_and_pins_the_corners():
    p = _eye(3.0)
    assert p.shape == (1, K)
    dx, dy = _d(p, [(500, 298), (490, 300), (510, 296)])
    assert np.allclose(dx, 3.0, atol=1e-4) and np.allclose(dy, 0.0, atol=1e-4)
    dx, _ = _d(p, [(467, 300), (533, 300), (460, 300), (540, 300)])
    assert np.allclose(dx, 0.0, atol=1e-4)                  # corners never move
    dx, _ = _d(p, [(500, 260), (500, 340)])
    assert np.allclose(dx, 0.0, atol=1e-4)                  # brow and cheek untouched


def test_field_never_folds_the_image():
    """x -> x + d(x) must stay monotonic along every row, or pixels would duplicate or tear."""
    p = _eye(5.0)
    xs = np.linspace(440, 560, 1201)
    for row in (290, 296, 300, 304, 309, 313):
        dx, _ = _d(p, [(x, row) for x in xs])
        assert np.all(np.diff(xs + dx) > 0)


def test_oversized_shift_is_clamped_before_it_can_fold():
    p = _eye(40.0)                       # far more than the whites beside the iris can absorb
    assert 0 < float(p[0, 10]) < 40.0
    xs = np.linspace(440, 560, 1201)
    for row in (296, 300, 304):
        dx, _ = _d(p, [(x, row) for x in xs])
        assert np.all(np.diff(xs + dx) > 0)


def test_saved_setup_fills_unset_fields(monkeypatch):
    from routes import eyecontact as ec
    monkeypatch.setattr(ec.app_settings, "get", lambda key, default=None: {"prompter_cm": 25.0, "junk": 1})
    merged = ec._merged(ec.EyeContactSetup(steadiness=0.5))
    assert merged["prompter_cm"] == 25.0 and merged["steadiness"] == 0.5
    assert merged["prompter_side"] == "left" and "junk" not in merged


def test_swap_source_moves_every_reference():
    from routes.eyecontact import swap_source
    old, new = r"B:\Footage\take1.MOV", r"B:\Footage\take1_eyecontact.mp4"
    p = {"source_video": old.lower(),
         "timeline": {"sources": {"s1": {"path": old}, "s2": {"path": r"B:\Footage\broll.mp4"}}}}
    assert swap_source(p, old, new) == 2
    assert p["source_video"] == new
    assert p["timeline"]["sources"]["s1"]["path"] == new
    assert p["timeline"]["sources"]["s2"]["path"] == r"B:\Footage\broll.mp4"
    assert swap_source(p, new, old) == 2                     # and back (revert)
    assert p["timeline"]["sources"]["s1"]["path"] == old


def test_converted_landmark_net_runs():
    from eyecontact.landmarks import SPEC_PATH, TFLiteNet
    if not SPEC_PATH.exists():
        pytest.skip("landmark model not converted on this machine yet")
    net = TFLiteNet(torch.load(SPEC_PATH, weights_only=True)).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 256, 256))
    assert out[0].shape == (1, 478 * 3)
    assert out[1].shape == (1, 1)


def test_each_eye_stops_at_its_own_lens_position():
    """The lazy-eye bug (Raat3Baje, 2026-10-03): both irises got the same push, but the two eyes
    rest differently in their openings (here -0.07 and +0.09 eye widths when looking into the
    lens), so an overshoot carried the off-centre eye into its corner. Each eye now stops where
    it sits in the frames where the speaker really looked into the camera."""
    from eyecontact.plan import LENS_MARGIN, per_eye_limits
    n = 400
    cam = np.zeros(n, bool)
    cam[:100] = True                                   # an intro looking into the lens
    s_e = {"R": np.where(cam, -0.07, -0.044), "L": np.where(cam, 0.086, 0.112)}
    corr = np.where(cam, 0.0, -0.061)                  # the shared push: overshoots both eyes
    weight = np.where(cam, 0.0, 1.0)
    gaze = (s_e["R"] + s_e["L"]) / 2
    corr_e, lens_e = per_eye_limits(corr, s_e, np.ones(n, bool), weight, gaze, gaze.copy(), FPS,
                                    Settings(tv_lambda=0.0))
    assert lens_e == pytest.approx({"R": -0.07, "L": 0.086})
    for k in "RL":
        landed = s_e[k][~cam] + corr_e[k][~cam]
        assert np.all(landed >= lens_e[k] - LENS_MARGIN - 1e-9)        # never past its lens position
        assert np.all(np.abs(corr_e[k]) <= np.abs(corr) + 1e-12)        # only ever a smaller push
    assert np.median(corr_e["R"][~cam]) == pytest.approx(np.median(corr_e["L"][~cam]), abs=0.005)


def test_without_camera_looks_the_shared_push_stands():
    from eyecontact.plan import per_eye_limits
    n = 200
    corr = np.full(n, -0.06)
    s_e = {"R": np.full(n, -0.04), "L": np.full(n, 0.11)}
    corr_e, lens_e = per_eye_limits(corr, s_e, np.ones(n, bool), np.ones(n), np.zeros(n), np.zeros(n),
                                    FPS, Settings())
    assert lens_e == {} and np.array_equal(corr_e["R"], corr) and np.array_equal(corr_e["L"], corr)


def test_a_short_clip_uses_the_whole_recordings_lens_positions():
    """A 30 s preview rarely holds 2 s of camera looks, so on its own it fell back to the shared
    push and showed the old over-pushed eyes; given the whole recording's lens positions, the
    same clip stops each eye at its own."""
    from eyecontact.plan import LENS_MARGIN, per_eye_limits
    n = 200
    corr = np.full(n, -0.061)
    s_e = {"R": np.full(n, -0.044), "L": np.full(n, 0.112)}
    known = {"R": -0.07, "L": 0.086}
    corr_e, lens_e = per_eye_limits(corr, s_e, np.ones(n, bool), np.ones(n), np.zeros(n), np.zeros(n),
                                    FPS, Settings(tv_lambda=0.0), known)
    assert lens_e == known
    for k in "RL":
        assert np.all(s_e[k] + corr_e[k] >= known[k] - LENS_MARGIN - 1e-9)
        assert np.all(np.abs(corr_e[k]) < np.abs(corr))


def _eye_v(c_t, c_s=0.0):
    g = EyeGeom(mid=(500, 300), ex=(1, 0), corners=[(467, 300), (533, 300)],
                upper=[(467 + 66 * s, 300 - 12 * np.sin(np.pi * s)) for s in np.linspace(0.1, 0.9, 7)],
                lower=[(467 + 66 * s, 300 + 10 * np.sin(np.pi * s)) for s in np.linspace(0.1, 0.9, 7)],
                iris=(500, 298), r=14)
    return torch.tensor([eye_params(g, c_s, EYE_W, 0, 0, c_t=c_t)], dtype=torch.float32)


def test_vertical_shift_moves_the_iris_down_and_leaves_the_face_alone():
    p = _eye_v(1.5)
    dx, dy = _d(p, [(500, 298), (495, 300), (505, 296)])
    assert np.allclose(dy, 1.5, atol=1e-4) and np.allclose(dx, 0.0, atol=1e-4)
    dx, dy = _d(p, [(467, 300), (533, 300), (500, 260), (500, 340)])
    assert np.allclose(dy, 0.0, atol=1e-4)                  # corners, brow and cheek untouched
    _, up = _d(_eye_v(-1.5), [(500, 298)])
    assert np.allclose(up, -1.5, atol=1e-4)                 # negative looks higher


def test_vertical_shift_never_folds_and_is_clamped():
    for c_t in (2.0, 40.0):
        p = _eye_v(c_t)
        assert abs(float(p[0, 23])) <= 0.5 * 0.14 * EYE_W + 1e-6
        ys = np.linspace(260, 340, 801)
        for col in (480, 490, 500, 510, 520):
            _, dy = _d(p, [(col, y) for y in ys])
            assert np.all(np.diff(ys + dy) > 0)


def test_pitch_moves_only_the_reaimed_frames():
    from eyecontact.plan import EYE_WIDTHS_PER_DEGREE, vertical_shift
    weight = np.array([0.0, 0.5, 1.0, 1.0])
    valid = np.array([True, True, True, False])
    assert not vertical_shift(Settings(), weight, valid).any()
    v = vertical_shift(Settings(pitch_deg=2.0), weight, valid)
    assert v[0] == 0 and v[3] == 0                            # a camera look and a lost face stay
    assert v[2] == pytest.approx(2.0 * EYE_WIDTHS_PER_DEGREE, rel=1e-3) and v[1] == pytest.approx(v[2] / 2)
    assert vertical_shift(Settings(pitch_deg=-2.0), weight, valid)[2] < 0


def test_plan_carries_the_vertical_shift():
    pts, _gaze, _ = _reading(seconds=8)
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(angle_deg=10.0, pitch_deg=2.0))
    assert plan["corr_t"].shape == plan["corr"].shape and plan["corr_t"].max() > 0
