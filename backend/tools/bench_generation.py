"""Benchmark the B-roll generation workflows against each other on real prompts.

Starts its own ComfyUI (the same install and arguments BuzzEdit uses), runs the
current clip workflow and the upgraded one on the same prompts and seeds, does
the same for stills, times every job from submit to finished file, and writes
side-by-side frames plus a report. It changes no workflow BuzzEdit uses.

    python tools/bench_generation.py --out <dir> [--prompts file.json] [--n 5]

The upgraded clip workflow (2026-10-03):
  first frame   Z-Image Turbo, then a 1.5x hi-res pass (crisper source image)
  motion        Wan 2.2 I2V 14B Q5_K_M (was Q4_K_S) + lightx2v 4-step 1022
                LoRAs (was v1), 1024x576 (was 832x480), 81 frames
  upscale       FlashVSR v1.1 tiny, x2 -> 2048x1152 (temporally stable)
  smoothing     RIFE v4.26 x2 -> 32 fps (was 16 fps)
The upgraded still: Z-Image Turbo 1584x896 + a 1.5x hi-res pass (2376x1344).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / "workflows"
COMFY_CODE = os.environ.get("COMFY_CODE", "C:/Users/singh/ComfyUI-Installs/ComfyUI/ComfyUI")
COMFY_DATA = os.environ.get("COMFY_DATA", "C:/Users/singh/Documents/ComfyUI")
COMFY_PYTHON = os.environ.get("COMFY_PYTHON", os.path.join(COMFY_DATA, ".venv", "Scripts", "python.exe"))
PORT = int(os.environ.get("BENCH_COMFY_PORT", "8188"))
BASE = f"http://127.0.0.1:{PORT}"
BUDGET_S = 300.0                      # the creator's limit per clip
NEG = None                            # taken from the current workflow


# ------------------------------------------------------------------ ComfyUI

def _get(path: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post(path: str, body: Dict[str, Any]) -> Any:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def comfy_up() -> bool:
    try:
        _get("/system_stats", 3)
        return True
    except Exception:
        return False


def start_comfy(log: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [COMFY_PYTHON, "main.py", "--base-directory", COMFY_DATA, "--listen", "127.0.0.1", "--port", str(PORT)],
        cwd=COMFY_CODE, stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(240):
        if comfy_up():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited at start; see {log}")
        time.sleep(1)
    raise RuntimeError("ComfyUI did not come up in 4 minutes")


def free_vram() -> None:
    try:
        _post("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass


def run(graph: Dict[str, Any], timeout: float = 1800) -> Dict[str, Any]:
    """Queue a graph and wait for it; returns {seconds, outputs, error}."""
    t0 = time.time()
    prompt_id = _post("/prompt", {"prompt": graph})["prompt_id"]
    while time.time() - t0 < timeout:
        hist = _get(f"/history/{prompt_id}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            err = None
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                err = (msgs[0][1].get("exception_message") if msgs else "error")[:400]
            return {"seconds": round(time.time() - t0, 1), "outputs": entry.get("outputs", {}), "error": err}
        time.sleep(1.0)
    return {"seconds": round(time.time() - t0, 1), "outputs": {}, "error": "timeout"}


def output_files(outputs: Dict[str, Any]) -> List[Path]:
    out = []
    for node in outputs.values():
        for key in ("images", "videos", "gifs", "animated"):
            for item in node.get(key, []) or []:
                if isinstance(item, dict) and item.get("filename"):
                    out.append(Path(COMFY_DATA) / "output" / item.get("subfolder", "") / item["filename"])
    return out


# ------------------------------------------------------------- the graphs

def current_clip(prompt: str, seed: int, w: int = 832, h: int = 480) -> Dict[str, Any]:
    g = json.loads((WORKFLOWS / "zimage_wan22_i2v.json").read_text(encoding="utf-8"))
    g["z_pos"]["inputs"]["text"] = prompt
    g["w_pos"]["inputs"]["text"] = prompt
    for node in ("z_latent", "w_i2v"):
        g[node]["inputs"]["width"], g[node]["inputs"]["height"] = w, h
    g["z_sample"]["inputs"]["seed"] = seed
    g["w_sample_high"]["inputs"]["noise_seed"] = seed
    g["w_sample_low"]["inputs"]["noise_seed"] = seed
    g["w_i2v"]["inputs"]["length"] = 81
    g["v_create"]["inputs"]["fps"] = 16.0
    g["v_save"]["inputs"]["filename_prefix"] = f"bench/current_{seed}"
    return g


def upgraded_clip(prompt: str, seed: int, w: int = 1024, h: int = 576,
                  flashvsr: bool = True, rife: bool = True) -> Dict[str, Any]:
    g = current_clip(prompt, seed, w, h)
    # A sharper first frame: the same Z-Image picture, refined at 1.5x.
    g["z_up"] = {"class_type": "LatentUpscaleBy",
                 "inputs": {"samples": ["z_sample", 0], "upscale_method": "bislerp", "scale_by": 1.5}}
    g["z_sample2"] = {"class_type": "KSampler", "inputs": {
        "model": ["z_shift", 0], "positive": ["z_pos", 0], "negative": ["z_neg", 0],
        "latent_image": ["z_up", 0], "seed": seed, "steps": 4, "cfg": 1, "sampler_name": "res_multistep",
        "scheduler": "simple", "denoise": 0.4}}
    g["z_decode"]["inputs"]["samples"] = ["z_sample2", 0]
    # Better motion model and the newer 4-step distillation.
    g["w_unet_high"]["inputs"]["unet_name"] = "wan2.2_i2v_high_noise_14B_Q5_K_M.gguf"
    g["w_unet_low"]["inputs"]["unet_name"] = "wan2.2_i2v_low_noise_14B_Q5_K_M.gguf"
    g["w_lora_high"]["inputs"]["lora_name"] = "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
    g["w_lora_low"]["inputs"]["lora_name"] = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
    frames, fps = ["w_decode", 0], 16.0
    if flashvsr:
        g["u_vsr"] = {"class_type": "FlashVSRNode", "inputs": {
            "frames": frames, "model": "FlashVSR-v1.1", "mode": "tiny", "scale": 2,
            "tiled_vae": True, "tiled_dit": True, "unload_dit": True, "seed": seed}}
        frames = ["u_vsr", 0]
    if rife:
        g["r_model"] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": "rife_v4.26.safetensors"}}
        g["r_run"] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": ["r_model", 0], "images": frames, "multiplier": 2}}
        frames, fps = ["r_run", 0], 32.0
    g["v_create"]["inputs"]["images"] = frames
    g["v_create"]["inputs"]["fps"] = fps
    g["v_save"]["inputs"]["filename_prefix"] = f"bench/upgraded_{w}x{h}_{seed}"
    return g


def current_still(prompt: str, seed: int, w: int = 1584, h: int = 896) -> Dict[str, Any]:
    g = json.loads((WORKFLOWS / "zimage1.json").read_text(encoding="utf-8"))
    g["70"]["inputs"]["text"] = prompt
    g["66"]["inputs"]["width"], g["66"]["inputs"]["height"] = w, h
    g["69"]["inputs"]["seed"] = seed
    g["9"]["inputs"]["filename_prefix"] = f"bench/still_current_{seed}"
    return g


def upgraded_still(prompt: str, seed: int, w: int = 1584, h: int = 896) -> Dict[str, Any]:
    g = current_still(prompt, seed, w, h)
    g["up"] = {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["69", 0], "upscale_method": "bislerp", "scale_by": 1.5}}
    g["hires"] = {"class_type": "KSampler", "inputs": {
        "model": ["67", 0], "positive": ["70", 0], "negative": ["71", 0], "latent_image": ["up", 0],
        "seed": seed, "steps": 4, "cfg": 1, "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 0.4}}
    g["65"]["inputs"]["samples"] = ["hires", 0]
    g["9"]["inputs"]["filename_prefix"] = f"bench/still_upgraded_{seed}"
    return g


# ---------------------------------------------------------------- frames

def frame_at(video: Path, out: Path, at: float = 2.5) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(at), "-i", str(video), "-frames:v", "1",
                    "-vf", "scale=960:-2:flags=lanczos", str(out)], check=False)


def side_by_side(left: Path, right: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(left), "-i", str(right), "-filter_complex",
                    "[0]scale=960:540[a];[1]scale=960:540[b];[a][b]hstack", str(out)], check=False)


def probe(video: Path) -> str:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    return r.stdout.strip()


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True, help="JSON list of {clip, still} prompts")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))[: args.n]
    report: Dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "budget_s": BUDGET_S, "runs": []}

    def save() -> None:
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    proc: Optional[subprocess.Popen] = None
    if not comfy_up():
        proc = start_comfy(out / "comfyui.log")
    try:
        # Warm-up: the first clip of a cold ComfyUI pays model loading; BuzzEdit
        # runs many in a row, so timed runs start warm.
        warm = run(current_clip(prompts[0]["clip"], 1))
        report["warmup_s"] = warm["seconds"]
        save()
        variants = [("current", lambda p, s: current_clip(p, s)),
                    ("upgraded_1024x576", lambda p, s: upgraded_clip(p, s, 1024, 576))]
        for name, build in variants:
            free_vram()
            run(build(prompts[0]["clip"], 7))   # warm this variant's models too
            for i, p in enumerate(prompts):
                seed = 1000 + i
                r = run(build(p["clip"], seed))
                files = [f for f in output_files(r["outputs"]) if f.suffix == ".mp4"]
                row = {"kind": "clip", "variant": name, "shot": i, "seconds": r["seconds"], "error": r["error"],
                       "file": str(files[0]) if files else None, "probe": probe(files[0]) if files else None}
                if files:
                    frame_at(files[0], out / f"clip{i}_{name}.png")
                report["runs"].append(row)
                save()
                print(json.dumps(row), flush=True)
        # Over budget at 1024x576: try the fallback size.
        up = [r["seconds"] for r in report["runs"] if r["variant"] == "upgraded_1024x576" and not r["error"]]
        if up and sum(up) / len(up) > BUDGET_S - 10:
            free_vram()
            for i, p in enumerate(prompts):
                r = run(upgraded_clip(p["clip"], 1000 + i, 960, 544))
                files = [f for f in output_files(r["outputs"]) if f.suffix == ".mp4"]
                row = {"kind": "clip", "variant": "upgraded_960x544", "shot": i, "seconds": r["seconds"],
                       "error": r["error"], "file": str(files[0]) if files else None}
                if files:
                    frame_at(files[0], out / f"clip{i}_upgraded_960x544.png")
                report["runs"].append(row)
                save()
        for i in range(len(prompts)):
            a, b = out / f"clip{i}_current.png", out / f"clip{i}_upgraded_1024x576.png"
            if a.exists() and b.exists():
                side_by_side(a, b, out / f"compare_clip{i}.png")
        # Stills.
        free_vram()
        for name, build in (("current", current_still), ("upgraded", upgraded_still)):
            run(build(prompts[0]["still"], 7))
            for i, p in enumerate(prompts):
                r = run(build(p["still"], 2000 + i))
                files = [f for f in output_files(r["outputs"]) if f.suffix.lower() in (".png", ".jpg", ".webp")]
                report["runs"].append({"kind": "still", "variant": name, "shot": i, "seconds": r["seconds"],
                                       "error": r["error"], "file": str(files[0]) if files else None})
                save()
        for i in range(len(prompts)):
            stills = {r["variant"]: r["file"] for r in report["runs"]
                      if r["kind"] == "still" and r["shot"] == i and r["file"]}
            if len(stills) == 2:
                side_by_side(Path(stills["current"]), Path(stills["upgraded"]), out / f"compare_still{i}.png")
        # Summary.
        summary = {}
        for r in report["runs"]:
            key = f"{r['kind']}:{r['variant']}"
            s = summary.setdefault(key, {"n": 0, "ok": 0, "total_s": 0.0, "max_s": 0.0, "errors": []})
            s["n"] += 1
            if r["error"]:
                s["errors"].append(r["error"])
            else:
                s["ok"] += 1
                s["total_s"] += r["seconds"]
                s["max_s"] = max(s["max_s"], r["seconds"])
        for s in summary.values():
            s["mean_s"] = round(s["total_s"] / s["ok"], 1) if s["ok"] else None
        report["summary"] = summary
        report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        lines = ["# Generation benchmark", "",
                 f"{report['started']} -> {report['finished']} -- clip budget {BUDGET_S:.0f} s", "",
                 "| What | Runs ok | Mean s | Slowest s | Within 5 min? | Errors |",
                 "|---|---|---|---|---|---|"]
        for key, s in summary.items():
            within = ("yes" if (s["max_s"] or 0) <= BUDGET_S else "NO") if key.startswith("clip:") and s["ok"] else ""
            lines.append(f"| {key} | {s['ok']}/{s['n']} | {s['mean_s']} | {round(s['max_s'], 1)} | {within} | "
                         f"{'; '.join(e[:80] for e in s['errors'][:2])} |")
        lines += ["", "Side by side (left = current, right = upgraded): `compare_clip*.png`, `compare_still*.png`.",
                  "Full clips: the `file` of each run in `report.json` (ComfyUI output/bench/)."]
        (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
    finally:
        free_vram()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(30)
            except Exception:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
