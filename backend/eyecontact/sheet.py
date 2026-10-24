"""A contact sheet for judging a run by eye: enlarged eye crops, before over after.

Six reading frames (spread over the take) and three camera looks, each labelled with its
time, the plan's weight, each eye's sideways shift and the vertical shift in pixels. Green
ticks above and below each crop mark the middle of each eye opening (corner to corner), so
an iris moving toward or past it is easy to see. Decoded on the CPU with OpenCV: it never
touches the GPU pipeline.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from .plan import P

SCALE = 5
COLS = 3
LABEL_H = 26


def sheet_path(dst: str) -> str:
    p = Path(dst)
    return str(p.with_name(f"{p.stem}_eyes.jpg"))


def _spread(idx, k):
    """k indexes spread over idx, clear of both ends."""
    if len(idx) == 0:
        return []
    pos = np.linspace(0, len(idx) - 1, k + 2)[1:-1].round().astype(int)
    return sorted({int(idx[p]) for p in pos})


def pick_frames(plan, reading_n=6, camera_n=3):
    ok = plan["valid"] & plan["eye_ok"]
    reading = np.flatnonzero(ok & (plan["weight"] > 0.95))
    cam = plan.get("cam")
    cam_idx = np.flatnonzero(ok & cam) if cam is not None and np.any(cam) else np.flatnonzero(ok & (plan["weight"] < 0.1))
    return _spread(reading, reading_n), _spread(cam_idx, camera_n)


class _Reader:
    """Frame-indexed reads: forward reads for nearby frames, seeks for far ones."""

    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)    # landmarks are in coded-frame pixels
        self.pos = 0
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self, i):
        if not (self.pos <= i < self.pos + 60):
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i
        while self.pos < i:
            self.cap.grab()
            self.pos += 1
        ok, img = self.cap.read()
        self.pos += 1
        return img if ok else None

    def close(self):
        self.cap.release()


def _crop_box(plan, i, H, W):
    row = plan["pts"][i:i + 1]
    xy = np.array([P(row, k)[0, :2] for k in (33, 133, 362, 263)])
    w_eye = float(np.mean([plan["w_e"]["R"][i], plan["w_e"]["L"][i]]))
    x0, x1 = xy[:, 0].min() - 0.5 * w_eye, xy[:, 0].max() + 0.5 * w_eye
    cy = float(xy[:, 1].mean())
    y0, y1 = cy - 0.5 * w_eye, cy + 0.5 * w_eye
    x0, y0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
    x1, y1 = int(min(W, np.ceil(x1))), int(min(H, np.ceil(y1)))
    centres = [(xy[0, 0] + xy[1, 0]) / 2, (xy[2, 0] + xy[3, 0]) / 2]
    return x0, y0, x1, y1, centres


def contact_sheet(src: str, dst_video: str, plan: dict, out_png: str) -> Optional[str]:
    """Write a JPEG grid of 5x eye crops (before over after) for 6 reading frames and 3
    camera-look frames of `plan`, from `src` and its corrected `dst_video`. Returns the
    path written (whatever extension `out_png` has, the image is a JPEG), or None."""
    import cv2
    reading, camera = pick_frames(plan)
    frames = [(i, "READ") for i in reading] + [(i, "CAM") for i in camera]
    if not frames:
        return None
    before, after = _Reader(src), _Reader(dst_video)
    fps = before.fps or 25.0
    cells = []
    try:
        for i, kind in sorted(frames):
            a, b = before.read(i), after.read(i)
            if a is None or b is None:
                continue
            H, W = a.shape[:2]
            x0, y0, x1, y1, centres = _crop_box(plan, i, H, W)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            cells.append((i, kind, a[y0:y1, x0:x1], b[y0:y1, x0:x1], [c - x0 for c in centres]))
    finally:
        before.close()
        after.close()
    if not cells:
        return None
    cw = int(np.median([c[2].shape[1] for c in cells]) * SCALE)
    ch = int(np.median([c[2].shape[0] for c in cells]) * SCALE)
    corr_e, corr_t = plan.get("corr_e") or {}, plan.get("corr_t")
    tiles = {}
    for i, kind, a, b, centres in cells:
        sx = cw / a.shape[1]
        imgs = []
        for img, tag in ((a, "before"), (b, "after")):
            big = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_CUBIC)
            for cx in centres:
                x = int(round(cx * sx))
                cv2.line(big, (x, 0), (x, 9), (0, 255, 0), 2)
                cv2.line(big, (x, ch - 10), (x, ch - 1), (0, 255, 0), 2)
            cv2.putText(big, tag, (6, ch - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(big, tag, (6, ch - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            imgs.append(big)
        px = {k: float(v[i]) * float(plan["w_e"][k][i]) for k, v in corr_e.items()}
        w_eye = float(np.mean([plan["w_e"]["R"][i], plan["w_e"]["L"][i]]))
        vpx = float(corr_t[i]) * w_eye if corr_t is not None else 0.0
        text = (f"{kind} t={i / fps:.1f}s w={float(plan['weight'][i]):.2f} "
                + " ".join(f"{k}{v:+.1f}" for k, v in px.items()) + f" v{vpx:+.1f}px")
        bar = np.full((LABEL_H, cw, 3), 32, np.uint8)
        cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles[(kind, i)] = np.vstack([bar, imgs[0], np.full((2, cw, 3), 255, np.uint8), imgs[1]])
    th, tw = next(iter(tiles.values())).shape[:2]
    pad = 8
    blank = np.full((th, tw, 3), 16, np.uint8)
    rows = []
    reading_tiles = [tiles[k] for k in sorted(tiles) if k[0] == "READ"]
    cam_tiles = [tiles[k] for k in sorted(tiles) if k[0] == "CAM"]
    for group in (reading_tiles, cam_tiles):
        for r in range(0, len(group), COLS):
            row = group[r:r + COLS] + [blank] * (COLS - len(group[r:r + COLS]))
            gap = np.full((th, pad, 3), 16, np.uint8)
            parts = []
            for t in row:
                parts += [t, gap]
            rows.append(np.hstack(parts[:-1]))
            rows.append(np.full((pad, rows[-1].shape[1], 3), 16, np.uint8))
    grid = np.vstack(rows[:-1])
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_png).write_bytes(buf.tobytes())
    return str(out_png)
