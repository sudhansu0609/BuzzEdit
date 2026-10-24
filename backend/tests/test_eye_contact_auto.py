"""The measured ("auto") eye-contact mode: the frames where the speaker really looks into
the lens say where each iris sits then, sideways and up/down, and every reading frame is
moved there. No prompter geometry, no aim nudge, no under-read model."""
import numpy as np

from eyecontact.plan import (EYES, COL, IRIS_SWING, MAX_PITCH_DEG, MEASURED_KEYS, Settings, make_plan)
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
