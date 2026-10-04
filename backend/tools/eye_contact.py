"""Fix teleprompter eye contact in a recording (GPU).

    python backend/tools/eye_contact.py take1.mp4
    python backend/tools/eye_contact.py take1.mp4 -o fixed.mp4 --steadiness 0.8 --quality max
    python backend/tools/eye_contact.py take1.mp4 --compare      # also write a zoomed before/after

Writes <name>_eyecontact.mp4 next to the input by default. Landmarks are cached per
source file, so re-running with other settings only repeats the render pass.
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyecontact import Settings, correct_eye_contact, default_output  # noqa: E402

FONT = "C\\:/Windows/Fonts/arialbd.ttf"   # drawtext needs a pinned font on this machine


def write_compare(src: str, fixed: str, out: str) -> None:
    """Side-by-side crop around the face, at full source resolution."""
    import numpy as np
    from eyecontact import analyse
    _, plan = analyse(src)
    pts = plan["pts"][plan["valid"]]
    if not len(pts):
        return
    from eyecontact.plan import P
    cx = float(np.median((P(pts, 33)[:, 0] + P(pts, 263)[:, 0]) / 2))
    cy = float(np.median((P(pts, 33)[:, 1] + P(pts, 263)[:, 1]) / 2))
    iod = float(np.median(np.linalg.norm(P(pts, 263)[:, :2] - P(pts, 33)[:, :2], axis=1)))
    # tight on the face and scaled to full-HD panels: a few-pixel iris shift has to be visible
    w = int(iod * 4.0) // 2 * 2
    h = int(w * 9 / 16) // 2 * 2
    x, y = max(int(cx - w / 2), 0) // 2 * 2, max(int(cy - h * 0.42), 0) // 2 * 2
    label = lambda t: f"scale=1920:1080:flags=lanczos,drawtext=fontfile='{FONT}':text='{t}':x=30:y=24:fontsize=54:fontcolor=white:borderw=4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", fixed, "-filter_complex",
                    f"[0:v]crop={w}:{h}:{x}:{y},{label('ORIGINAL')}[a];"
                    f"[1:v]crop={w}:{h}:{x}:{y},{label('EYE CONTACT')}[b];[a][b]hstack=2[v]",
                    "-map", "[v]", "-map", "0:a?", "-c:v", "hevc_nvenc", "-preset", "p5", "-cq", "18",
                    "-c:a", "copy", out], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--side", choices=["left", "right", "auto"], default=Settings.prompter_side,
                    help="which side of the lens the prompter is on, from your seat (default %(default)s)")
    ap.add_argument("--angle", type=float, default=0.0,
                    help="degrees between the prompter text and the lens, as seen from your seat; "
                         "overrides --prompter-cm/--camera-cm (e.g. 10)")
    ap.add_argument("--prompter-cm", type=float, default=Settings.prompter_cm,
                    help="sideways distance from the lens to the middle of the text (default %(default)s)")
    ap.add_argument("--camera-cm", type=float, default=Settings.camera_cm,
                    help="distance from your eyes to the camera (default %(default)s)")
    ap.add_argument("--steadiness", type=float, default=Settings.steadiness,
                    help="0 keeps the reading sweep, 1 locks the eyes on the lens (default %(default)s)")
    ap.add_argument("--aim", type=float, default=0.0,
                    help="fine nudge of the lens direction in degrees, + = toward your left")
    ap.add_argument("--quality", choices=["standard", "high", "max"], default="high")
    ap.add_argument("--codec", choices=["hevc", "h264"], default="hevc")
    ap.add_argument("--compare", action="store_true", help="also write <output>_compare.mp4")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    out = args.output or default_output(args.input)
    last = [0.0]

    def progress(f):
        if f - last[0] >= 0.05 or f >= 1.0:
            last[0] = f
            print(f"  {f * 100:5.1f}%", flush=True)

    t0 = time.time()
    settings = Settings(prompter_side=args.side, angle_deg=args.angle,
                        prompter_cm=args.prompter_cm, camera_cm=args.camera_cm,
                        steadiness=args.steadiness, aim_deg=args.aim)
    report = correct_eye_contact(args.input, out, settings,
                                 args.quality, args.codec, progress)
    print(json.dumps(report, indent=2))
    if args.compare:
        cmp_path = str(Path(out).with_name(Path(out).stem + "_compare.mp4"))
        write_compare(args.input, out, cmp_path)
        print("compare:", cmp_path)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
