"""The measured ("auto") eye-contact mode: the frames where the speaker really looks into
the lens say where each iris sits then, sideways and up/down, and every reading frame is
moved there. No prompter geometry, no aim nudge, no under-read model."""
import numpy as np

from eyecontact.plan import (EYES, COL, EYE_WIDTHS_PER_DEGREE, IRIS_SWING, MAX_PITCH_DEG, MEASURED_KEYS,
                             Settings, make_plan)
from tests.test_eye_contact import EYE_W, FPS, _face, _reading


def _raise_iris(p, dy):
    """Move both irises (and their rings) up by dy pixels in a compact landmark row."""
    p = p.copy()
    for e in EYES.values():
        for idx in [e["iris"], *e["ring"]]:
            p[COL[idx], 1] -= dy
    return p


def _take(cam_seconds=6.0, read_seconds=24.0, cam_gaze=-0.02, cam_raise_px=4.0, offset=0.04):
    """A take that opens with a look into the lens (eyes a little left of symmetric and
    raised), then reads from a prompter to the right with the usual sweep."""
    cam = np.stack([_raise_iris(_face(cam_gaze * EYE_W), cam_raise_px) for _ in range(int(cam_seconds * FPS))])
    read, gaze, blinks = _reading(seconds=read_seconds, offset=offset)
    pts = np.concatenate([cam, read])
    is_cam = np.r_[np.ones(len(cam), bool), np.zeros(len(read), bool)]
    gaze_all = np.r_[np.full(len(cam), cam_gaze), gaze]
    blinks_all = np.r_[np.zeros(len(cam), bool), blinks]
    return pts, gaze_all, is_cam, blinks_all


def test_auto_moves_reading_frames_onto_the_measured_lens_position():
    pts, gaze, is_cam, blinks = _take()
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto", steadiness=0.7))
    assert plan["auto_measured"] is True
    m = plan["lens_e"]
    assert all(k in m for k in MEASURED_KEYS)
    assert abs(m["lens"] - (-0.02)) < 0.004 and abs(m["rest"] - 0.04) < 0.006
    reading = plan["eye_ok"] & ~is_cam & ~blinks
    reading[: int(2 * FPS) + len(np.flatnonzero(is_cam))] = False   # clear of the join's fade
    out = gaze + plan["corr_e"]["R"]
    # Reading frames land on the measured lens position (the sweep damped, not frozen).
    assert abs(np.median(out[reading]) - m["lens"]) < 0.006
    assert np.std(out[reading]) < 0.5 * np.std(gaze[reading])
    # The real camera looks are left alone.
    cam_mid = is_cam.copy()
    n_cam = int(is_cam.sum())
    cam_mid[n_cam - int(2.5 * FPS):] = False      # the held-gaze fade spans the join
    assert np.abs(plan["corr"][cam_mid]).max() < 0.004
    # Up/down: the eyes are raised by the measured difference (4 px of a 66 px eye).
    assert abs(m["t_lens"] - m["t_rest"] - (-4.0 / EYE_W)) < 0.006
    assert np.median(plan["corr_t"][reading]) < -0.045
    assert np.abs(plan["corr_t"][cam_mid]).max() < 0.004
    assert abs(plan["measured_shift_deg"]) > 5.0


def test_auto_without_camera_looks_falls_back_to_the_symmetric_guess():
    pts, gaze, blinks = _reading()
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto", steadiness=0.7))
    assert plan["auto_measured"] is False
    ok = plan["eye_ok"] & ~blinks
    assert abs(np.median((gaze + plan["corr"])[ok])) < 0.006       # centred on the symmetric lens


def test_a_short_clip_reuses_the_whole_takes_measurement():
    pts, gaze, is_cam, blinks = _take()
    whole = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto"))
    clip_pts = pts[len(np.flatnonzero(is_cam)) + int(4 * FPS):]       # reading only, no camera look
    clip_gaze = gaze[len(np.flatnonzero(is_cam)) + int(4 * FPS):]
    alone = make_plan(clip_pts, np.full(len(clip_pts), 10.0), FPS, Settings(mode="auto"))
    assert alone["auto_measured"] is False
    with_known = make_plan(clip_pts, np.full(len(clip_pts), 10.0), FPS, Settings(mode="auto"), whole["lens_e"])
    assert with_known["auto_measured"] is True
    ok = with_known["eye_ok"]
    assert abs(np.median((clip_gaze + with_known["corr_e"]["R"])[ok]) - whole["lens_e"]["lens"]) < 0.006


def test_vertical_measurement_is_capped_where_the_warp_folds():
    pts, gaze, is_cam, blinks = _take(cam_raise_px=14.0)
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto"))
    cap = IRIS_SWING * np.sin(np.radians(MAX_PITCH_DEG))
    assert np.abs(plan["corr_t"]).max() <= cap + 1e-9


def _bow_rows(pts, frames, dz):
    """Bow the head on `frames` by moving the chin landmark back by dz (as the real tracker
    sees a head that faces below the lens: chin further from the camera than the forehead)."""
    pts = pts.copy()
    pts[frames, COL[152], 2] += dz
    return pts


def test_a_speaker_who_lifts_the_head_for_the_lens_gets_the_eyes_raised_by_the_bow():
    """The 2026-10-20 ep3 take: the prompter sits below the lens, the whole read is done with
    the head bowed ~7 deg and the eyes neutral in it; a look into the camera lifts the head and
    leaves the irises where they were. The head-up frames are the camera looks, and the eyes
    are raised by the bow to meet the lens."""
    read, gaze, blinks = _reading(seconds=30, offset=0.0)
    n = len(read)
    # bow the head on every frame except an 8 s look into the lens in the middle (head level)
    bowed = np.ones(n, bool)
    bowed[int(11 * FPS):int(19 * FPS)] = False
    pts = _bow_rows(read, bowed, 40.0)
    plan = make_plan(pts, np.full(n, 10.0), FPS, Settings(mode="auto"))
    assert plan["auto_measured"] is True
    assert plan["lens_e"]["head_raise_deg"] > 3.0
    reading = plan["eye_ok"] & bowed & ~blinks
    reading[int(8 * FPS):int(22 * FPS)] = False
    # raised (negative = up) by the bow, on the reading frames only
    assert np.median(plan["corr_t"][reading]) < -0.015
    lens_look = ~bowed
    lens_look[int(11 * FPS):int(13 * FPS)] = False
    lens_look[int(17 * FPS):] = False
    assert np.abs(plan["corr_t"][lens_look]).max() < 0.004


# --- v2 (2026-10-24): gaze in head coordinates, user knobs in auto mode, lively sweep removal ---

def _turn(pts, frames, sin_yaw):
    """Turn the head on `frames` (face normal along the eye axis = sin_yaw) by moving the
    subject's-left eye corner in depth; the eye corners in the image stay where they were."""
    pts = pts.copy()
    dx = 186.0                                 # 263 - 33 in the synthetic face
    pts[frames, COL[263], 2] = sin_yaw * dx / np.sqrt(max(1e-9, 1 - sin_yaw ** 2))
    return pts


def _reading_frames(plan, is_cam, blinks):
    r = plan["eye_ok"] & ~is_cam & ~blinks
    r[: int(2 * FPS) + int(is_cam.sum())] = False
    return r


def test_a_head_turned_toward_the_prompter_gets_the_extra_iris_push():
    """The eyes sit the same, but the head reads turned toward the prompter: the gaze is
    further off the lens by the head's part, and the irises are pushed that much further."""
    pts, gaze, is_cam, blinks = _take()
    flat = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto"))
    turned_pts = _turn(pts, ~is_cam, 0.06)
    turned = make_plan(turned_pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto"))
    assert flat["auto_measured"] and turned["auto_measured"]
    r = _reading_frames(flat, is_cam, blinks)
    extra = np.median(flat["corr"][r]) - np.median(turned["corr"][r])
    assert abs(extra - IRIS_SWING * 0.06) < 0.006
    assert turned["measured_shift_deg"] < flat["measured_shift_deg"] - 2.0      # more negative
    # ...and the per-eye limits move with the head, so each eye really gets it.
    extra_e = np.median(flat["corr_e"]["R"][r]) - np.median(turned["corr_e"]["R"][r])
    assert extra_e > 0.8 * IRIS_SWING * 0.06


def _lean_in(p, scale=1.5):
    """The face leaning into the camera: everything scaled up about the face centre."""
    p = p.copy()
    c = np.array([193.0, 150.0])
    p[:, :2] = c + scale * (p[:, :2] - c)
    return p


def test_outlier_eye_width_frames_never_become_camera_looks():
    """674aefab t=131 s: the speaker leans into the camera; the gaze there sits where a look
    into the lens would. Eye width far off its rolling median: never a camera look."""
    pts, gaze, is_cam, blinks = _take()
    pts[is_cam] = np.stack([_lean_in(p) for p in pts[is_cam]])
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto"))
    assert not plan["cam"][is_cam].any()
    assert plan["auto_measured"] is False
    # and a face partly out of the frame is not one either
    pts2, _, is_cam2, _ = _take()
    pts2[is_cam2, :, 1] -= 120.0                 # forehead above the top edge
    plan2 = make_plan(pts2, np.full(len(pts2), 10.0), FPS, Settings(mode="auto"), frame_size=(400, 400))
    assert not plan2["cam"][is_cam2].any() and plan2["auto_measured"] is False


def test_lens_marks_override_the_automatic_cluster():
    pts, gaze, is_cam, blinks = _take()
    # a second, short look elsewhere at 20 s into the read: the user says THAT is the lens
    look = slice(int((6 + 20) * FPS), int((6 + 21) * FPS))
    pts[look] = np.stack([_face(-0.06 * EYE_W)] * (look.stop - look.start))
    auto = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto"))
    assert auto["lens_source"] == "measured" and abs(auto["lens_e"]["lens_c"] - (-0.02)) < 0.006
    assert auto["camera_look_candidates"] and auto["camera_look_candidates"][0] < 6.0
    marked = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto", lens_marks=[26.5]))
    assert marked["lens_source"] == "marks" and marked["lens_e"]["from_marks"] == 1.0
    assert abs(marked["lens_e"]["lens_c"] - (-0.06)) < 0.006
    r = _reading_frames(auto, is_cam, blinks)
    r[look.start - int(2 * FPS):look.stop + int(2 * FPS)] = False
    assert np.median(marked["corr"][r]) < np.median(auto["corr"][r]) - 0.03


def test_aim_and_pitch_nudges_apply_in_auto_mode():
    pts, gaze, is_cam, blinks = _take()
    base = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto"))
    nudged = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto", aim_deg=-3.0, pitch_deg=2.0))
    r = _reading_frames(base, is_cam, blinks)
    d_s = np.median(nudged["corr"][r]) - np.median(base["corr"][r])
    d_t = np.median(nudged["corr_t"][r]) - np.median(base["corr_t"][r])
    assert abs(d_s - (-3.0 * EYE_WIDTHS_PER_DEGREE)) < 0.004
    assert abs(d_t - 2.0 * EYE_WIDTHS_PER_DEGREE) < 0.004
    # the per-eye limits move with the aim instead of swallowing it
    d_e = np.median(nudged["corr_e"]["R"][r]) - np.median(base["corr_e"]["R"][r])
    assert d_e < -0.8 * 3.0 * EYE_WIDTHS_PER_DEGREE
    # angle_deg sizes the sideways push; its direction stays measured
    sized = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto", angle_deg=12.0))
    assert np.median(sized["corr"][r]) < np.median(base["corr"][r]) - 0.005 < 0


def test_the_fast_movement_survives_while_the_line_sweep_is_removed():
    """steadiness=1: the sawtooth (creep + snap-back) goes, a 3 Hz fixational wobble stays."""
    cam_n, read_n = int(6 * FPS), int(24 * FPS)
    t = np.arange(read_n) / FPS
    saw = 0.015 * (1 - 2 * ((t % 1.2) / 1.2))
    wobble = 0.004 * np.sin(2 * np.pi * 3.0 * t)
    gaze = np.r_[np.full(cam_n, -0.02), 0.04 + saw + wobble]
    pts = np.stack([_raise_iris(_face(g * EYE_W), 4.0 if i < cam_n else 0.0) for i, g in enumerate(gaze)])
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(mode="auto", steadiness=1.0))
    assert plan["auto_measured"]
    r = np.zeros(len(pts), bool)
    r[cam_n + int(3 * FPS):-int(1 * FPS)] = True
    out = (gaze + plan["corr_e"]["R"])[r]
    tt = (np.arange(len(pts)) / FPS)[r]

    def amp(x, f):
        return 2 * abs(np.mean((x - x.mean()) * np.exp(-2j * np.pi * f * tt)))

    assert amp(out, 3.0) > 0.85 * amp(gaze[r], 3.0)                 # the natural movement is kept
    assert amp(out, 1 / 1.2) < 0.15 * amp(gaze[r], 1 / 1.2)         # the line sweep is gone
    assert np.std(out - np.r_[np.zeros(cam_n), wobble][r]) < 0.25 * np.std(saw)


def test_contact_sheet_writes_a_jpeg(tmp_path):
    import cv2
    from eyecontact.sheet import contact_sheet, sheet_path
    pts, gaze, blinks = _reading(seconds=3, blink_every=100)
    plan = make_plan(pts, np.full(len(pts), 10.0), FPS, Settings(prompter_side="left", angle_deg=10))
    paths = []
    for name, shade in (("src.mp4", 90), ("dst.mp4", 120)):
        path = str(tmp_path / name)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (400, 320))
        for i in range(len(pts)):
            img = np.full((320, 400, 3), shade, np.uint8)
            cv2.circle(img, (133 + i % 5, 98), 12, (20, 20, 20), -1)
            vw.write(img)
        vw.release()
        paths.append(path)
    out = sheet_path(paths[1])
    assert out.endswith("dst_eyes.jpg")
    assert contact_sheet(paths[0], paths[1], plan, out) == out
    with open(out, "rb") as f:
        data = f.read()
    assert data[:2] == b"\xff\xd8" and len(data) > 1000
