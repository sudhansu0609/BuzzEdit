"""Prove the fast generation graphs on THIS machine's live ComfyUI before a
production uses them, and measure what they cost.

    .venv/Scripts/python.exe backend/tools/clip_speed_check.py [--adopt] [--still <png>]

Runs, on the ComfyUI BuzzEdit started (port 8188), with the GPU otherwise idle:

    1. zimage1_esrgan.json      a 1024x576 still, RealESRGAN x2, no SageAttention
    2. zimage_wan22_i2v_lite16  a 832x480x81 clip from a still, no SageAttention
    3. zimage1_fast.json        the same still through SageAttention
    4. zimage_wan22_i2v_fast16  the same clip through SageAttention

(the SageAttention patch is global once a graph applies it, so the plain
graphs run first). Each pair shares a seed, so the SageAttention output must
look like the plain one: a middle frame is compared, and a blank or badly
different picture fails the check. `--adopt` then points the app settings at
the plain graphs (`workflow_broll_video` / `workflow_broll_image`) on failure,
or leaves the fast ones selected on success. The timings go to
`data/bench/<date>_speed/summary.md`. Exit 0 = the fast graphs are in use.

Why (2026-10-21): the HQ clip graph took 7.4 min a clip and a 22-minute
video 8 hours; the creator's budget is 3.5 hours. The fast graphs drop the
per-frame ESRGAN, run at 832x480 and use SageAttention -- pieces that pass
unit tests but had never run against the live server, which is how the local
editor lost 36 minutes of a production the same morning.
"""
import argparse
import glob
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "tools"))

from bench_generation import BASE, COMFY_DATA, _post, comfy_up, output_files, run  # noqa: E402

ROOT = BACKEND.parent
MAX_DIFF = 0.12          # mean |fast - plain| over a frame, 0..1
MIN_SPREAD = 0.03        # a frame's std below this is a blank picture
CLIP_W, CLIP_H, CLIP_LEN = 832, 480, 81
STILL_W, STILL_H = 1024, 576
PROMPT = ("A narrow village lane at night in rural India, a single oil lamp in a window, "
          "mist drifting between mud walls, a bicycle leaning on a wall, cinematic photograph")


def upload(path: Path) -> str:
    boundary = f"----speedcheck{int(time.time() * 1000)}"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
    tail = (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
            f"\r\n--{boundary}--\r\n").encode("utf-8")
    req = urllib.request.Request(f"{BASE}/upload/image", data=head + path.read_bytes() + tail,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    name = body.get("name") or path.name
    sub = body.get("subfolder") or ""
    return f"{sub}/{name}" if sub else name


def frame_of(path: Path, out: Path, at: float = 2.5) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(at), "-i", str(path), "-frames:v", "1", str(out)],
                   check=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out


def picture_stats(a: Path, b: Path):
    """(mean |a-b| 0..1, std of a, std of b) at a common small size."""
    import numpy as np
    from PIL import Image
    ia = Image.open(a).convert("L").resize((256, 144))
    ib = Image.open(b).convert("L").resize((256, 144))
    na, nb = np.asarray(ia, dtype="float32") / 255.0, np.asarray(ib, dtype="float32") / 255.0
    return float(abs(na - nb).mean()), float(na.std()), float(nb.std())


def build_still(file_name: str, seed: int, prefix: str):
    from presentation import workflows
    resolved = workflows._bind("broll_image", file_name, {})
    return resolved.build(positive=PROMPT, width=STILL_W, height=STILL_H, seed=seed, prefix=prefix)


def build_clip(file_name: str, seed: int, prefix: str, start_image: str):
    from presentation import workflows
    manifest = workflows.load_manifest().get("roles", {}).get("broll_video", {})
    resolved = workflows._bind("broll_video", file_name, manifest)
    negative = resolved.graph["w_neg"]["inputs"]["text"]
    graph = resolved.build(positive="slow push in, mist drifting, the lamp flickering, gentle cinematic camera movement",
                           negative=negative, width=CLIP_W, height=CLIP_H, seed=seed, length=CLIP_LEN, fps=16,
                           prefix=prefix, out_width=1920, out_height=1080)
    graph["bz_start_image"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
    graph["w_i2v"]["inputs"]["start_image"] = ["bz_start_image", 0]
    return graph


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adopt", action="store_true", help="write the outcome into the app settings")
    p.add_argument("--still", default="", help="a PNG to start the clips from (default: the newest project still)")
    p.add_argument("--seed", type=int, default=20261021)
    args = p.parse_args()

    if not comfy_up():
        print("FAIL ComfyUI is not answering on", BASE)
        return 2
    out_dir = ROOT / "data" / "bench" / f"{datetime.now():%Y-%m-%d_%H%M}_speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        _post("/free", {"unload_models": True, "free_memory": True})
    except Exception:  # noqa: BLE001
        pass

    def one(label, graph, timeout):
        print(f"-- {label}", flush=True)
        got = run(graph, timeout=timeout)
        files = output_files(got["outputs"])
        print(f"   {got['seconds']} s, error={got['error']!r}, files={[f.name for f in files]}", flush=True)
        results[label] = {"seconds": got["seconds"], "error": got["error"], "files": [str(f) for f in files]}
        return files if not got["error"] else []

    # 1 + 3: the still, plain then through SageAttention.
    plain_still = one("still_esrgan", build_still("zimage1_esrgan.json", args.seed, "speed/still_plain"), 600)
    start_png = Path(args.still) if args.still else (plain_still[0] if plain_still else None)
    if start_png is None:
        candidates = sorted(glob.glob(str(ROOT / "data" / "projects" / "*" / "assets" / "generated" / "*.png")),
                            key=lambda f: Path(f).stat().st_mtime)
        start_png = Path(candidates[-1]) if candidates else None
    if start_png is None:
        print("FAIL no still to start a clip from")
        return 2
    start_name = upload(start_png)
    plain_clip = one("clip_lite16", build_clip("zimage_wan22_i2v_lite16.json", args.seed, "speed/clip_plain", start_name), 1200)
    fast_still = one("still_fast", build_still("zimage1_fast.json", args.seed, "speed/still_fast"), 600)
    fast_clip = one("clip_fast16", build_clip("zimage_wan22_i2v_fast16.json", args.seed, "speed/clip_fast", start_name), 1200)

    verdict = {"sage_ok": True, "reasons": []}
    if not fast_clip or not fast_still:
        verdict["sage_ok"] = False
        verdict["reasons"].append("the SageAttention graph did not produce a file")
    else:
        try:
            if plain_still and fast_still:
                diff, s_plain, s_fast = picture_stats(plain_still[0], fast_still[0])
                results["still_diff"] = {"mean_abs_diff": round(diff, 4), "std_plain": round(s_plain, 4), "std_fast": round(s_fast, 4)}
                if diff > MAX_DIFF or s_fast < MIN_SPREAD:
                    verdict["sage_ok"] = False
                    verdict["reasons"].append(f"still differs by {diff:.3f} (std {s_fast:.3f})")
            if plain_clip and fast_clip:
                fa = frame_of(next(f for f in plain_clip if f.suffix == ".mp4"), out_dir / "clip_plain.png")
                fb = frame_of(next(f for f in fast_clip if f.suffix == ".mp4"), out_dir / "clip_fast.png")
                diff, s_plain, s_fast = picture_stats(fa, fb)
                results["clip_diff"] = {"mean_abs_diff": round(diff, 4), "std_plain": round(s_plain, 4), "std_fast": round(s_fast, 4)}
                if diff > MAX_DIFF or s_fast < MIN_SPREAD:
                    verdict["sage_ok"] = False
                    verdict["reasons"].append(f"clip frame differs by {diff:.3f} (std {s_fast:.3f})")
        except Exception as e:  # noqa: BLE001
            verdict["sage_ok"] = False
            verdict["reasons"].append(f"could not compare the outputs: {e}")
    # Same picture is necessary, not sufficient: the fast graphs are adopted
    # only when they are faster. On the RTX 5060 Ti (2026-10-22) SageAttention
    # rendered the same clip in 313 s against 232 s plain, and the stills in
    # the same time, and this check adopted it on the picture match alone.
    plain_s = (results.get("clip_lite16") or {}).get("seconds")
    fast_s = (results.get("clip_fast16") or {}).get("seconds")
    if plain_s and fast_s and fast_s > plain_s * 0.95:
        verdict["sage_ok"] = False
        verdict["reasons"].append(f"not faster: clip {fast_s:.0f} s with SageAttention vs {plain_s:.0f} s plain")
    results["verdict"] = verdict

    lines = [f"# Generation speed check {datetime.now():%Y-%m-%d %H:%M}", "",
             "| Job | Seconds | Error |", "|---|---|---|"]
    for label in ("still_esrgan", "still_fast", "clip_lite16", "clip_fast16"):
        r = results.get(label) or {}
        lines.append(f"| {label} | {r.get('seconds', '-')} | {r.get('error') or ''} |")
    lines += ["", f"still diff: {results.get('still_diff')}", f"clip diff: {results.get('clip_diff')}", "",
              f"SageAttention: {'OK' if verdict['sage_ok'] else 'REJECTED -- ' + '; '.join(verdict['reasons'])}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))

    if args.adopt:
        from store.app_settings import AppSettings
        settings = AppSettings()
        if verdict["sage_ok"]:
            settings.set("workflow_broll_image", "zimage1_fast.json")
            settings.set("workflow_broll_video", "")
            print("ADOPTED the SageAttention graphs (hq mode -> zimage_wan22_i2v_fast16.json, stills zimage1_fast.json)")
        else:
            settings.set("workflow_broll_image", "zimage1_esrgan.json")
            settings.set("workflow_broll_video", "zimage_wan22_i2v_lite16.json")
            print("ADOPTED the plain graphs (no SageAttention): zimage_wan22_i2v_lite16.json, zimage1_esrgan.json")
    try:
        _post("/free", {"unload_models": True, "free_memory": True})
    except Exception:  # noqa: BLE001
        pass
    return 0 if verdict["sage_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
