"""Person segmentation matte for the text-fx pass's behind-head and focus effects.

Two backends. torchvision (DeepLabV3-ResNet101, COCO-with-VOC-labels weights —
already cached in the torch hub cache, nothing downloaded here) runs on the GPU
when one is free and cuts a clean, soft edge; GrabCut (OpenCV's classical
foreground/background cut, seeded from a fixed centre-weighted box —
talking-head footage keeps the speaker roughly centred and filling most of the
frame) is the CPU fallback, used when torchvision is not importable or its
inference fails for any reason. `detect_backend` also still probes for
mediapipe and rembg, so a venv that later gains one of those uses it without a
code change, and degrades to "off" — behind-head and matte-based focus simply
do not fire — when nothing at all is available.

Computed once per project for exactly the seconds the text-fx planner actually
asked for (not the whole video), at a reduced fps/resolution and upscaled, then
cached to `data/projects/<id>/matte_<hash>.mp4` — a grayscale clip, white where
the model believes there is a person. A manifest alongside it
(`matte_<hash>.json`) records where each requested window landed inside that
file, since the windows are rendered back-to-back rather than at their original
spacing on the programme.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import FFMPEG_BIN
from utils.proc import NO_WINDOW

logger = logging.getLogger("presentation.matte")

MATTE_FPS = 6.0
MATTE_MAX_WIDTH = 480
GRABCUT_ITERS = 3
# Windows this close together are rendered as one continuous span rather than
# two, so the cache does not pay for the same handful of frames twice.
MERGE_GAP_S = 1.0

# torchvision's COCO-with-VOC-labels segmentation head: channel 15 is "person".
TORCHVISION_PERSON_CLASS = 15
# GPU broker tenant name for the inference — see _run_with_gpu_lease.
TORCHVISION_TENANT = "torchvision_matte"
TORCHVISION_VRAM_MB = 2500.0
# Exponential weight on each new frame's raw probability map, blended against
# the running smoothed value — low enough to damp single-frame flicker,
# high enough that the matte still tracks real movement inside a window.
TORCHVISION_TEMPORAL_ALPHA = 0.55
TORCHVISION_FEATHER_KERNEL = (5, 5)


def detect_backend() -> str:
    """The best person-segmentation backend already importable, without installing
    anything or reaching the network. "off" when nothing at all is available."""
    try:
        import mediapipe  # noqa: F401
        return "mediapipe"
    except Exception:
        pass
    try:
        import torchvision  # noqa: F401
        return "torchvision"
    except Exception:
        pass
    try:
        import rembg  # noqa: F401
        return "rembg"
    except Exception:
        pass
    try:
        import cv2  # noqa: F401
        return "grabcut"
    except Exception:
        return "off"


class MatteResult:
    def __init__(self, path: str, backend: str, windows: List[Dict[str, float]]):
        self.path = path
        self.backend = backend
        self.windows = windows  # [{"start_s","end_s","offset_s"}], sorted by start_s

    def offset_for(self, start_s: float, end_s: float) -> Optional[float]:
        """The matte-file-local second a real [start_s, end_s) window begins at,
        or None when that window was never rendered into this matte file."""
        for w in self.windows:
            if w["start_s"] - 0.05 <= start_s and end_s <= w["end_s"] + 0.05:
                return w["offset_s"] + max(0.0, start_s - w["start_s"])
        return None

    def to_dict(self) -> Dict[str, object]:
        return {"path": self.path, "backend": self.backend, "windows": self.windows}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MatteResult":
        return cls(str(data["path"]), str(data["backend"]), list(data["windows"]))


def _merge_windows(windows: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted((max(0.0, s), max(0.0, e)) for s, e in windows if e > s)
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + MERGE_GAP_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _cache_key(source_video: str, windows: Sequence[Tuple[float, float]],
              backend: str, canvas_w: int, canvas_h: int) -> str:
    try:
        stat = Path(source_video).stat()
        stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        stamp = "0"
    payload = repr((source_video, stamp, tuple(windows), backend, canvas_w, canvas_h,
                    MATTE_FPS, MATTE_MAX_WIDTH))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _seed_rect(width: int, height: int) -> Tuple[int, int, int, int]:
    """A centre-weighted box GrabCut starts from: talking-head footage keeps the
    speaker roughly centred and filling most of the vertical frame."""
    x = int(width * 0.16)
    y = int(height * 0.06)
    w = width - 2 * x
    h = int(height * 0.94) - y
    return x, y, max(1, w), max(1, h)


def _grabcut_mask(frame_bgr, seed_rect: Tuple[int, int, int, int]):
    """A 0/255 mask for one frame, via OpenCV's classical foreground cut."""
    import cv2
    import numpy as np

    mask = np.zeros(frame_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame_bgr, mask, seed_rect, bgd_model, fgd_model,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        logger.warning("grabCut failed on a frame, falling back to the seed box: %s", e)
        out = np.zeros(frame_bgr.shape[:2], np.uint8)
        x, y, w, h = seed_rect
        out[y:y + h, x:x + w] = 255
        return out
    keep = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    out = (keep.astype("uint8")) * 255
    # A light close+blur: grabCut's edge is a jagged pixel boundary, and the
    # matte gets upscaled several times to reach canvas resolution.
    kernel = np.ones((5, 5), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    out = cv2.GaussianBlur(out, (7, 7), 0)
    return out


def _resized_frame(frame):
    """`frame` (BGR) scaled down so its width is at most MATTE_MAX_WIDTH —
    shared by every backend so the cache key's own MATTE_MAX_WIDTH stays true
    regardless of which one actually ran."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(1.0, MATTE_MAX_WIDTH / max(1, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return frame


def _iter_frames(cap, start_s: float, end_s: float):
    """BGR frames for one window, resized to MATTE_MAX_WIDTH, sampled at
    MATTE_FPS — the raw material both backends sample identically."""
    import cv2

    step = 1.0 / MATTE_FPS
    t = start_s
    while t < end_s - 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            yield _resized_frame(frame)
        t += step


def _sample_window(cap, start_s: float, end_s: float) -> List:
    """Grayscale GrabCut mask frames for one window, sampled at MATTE_FPS."""
    frames = []
    last_rect = None
    for frame in _iter_frames(cap, start_s, end_s):
        seed = last_rect or _seed_rect(frame.shape[1], frame.shape[0])
        mask = _grabcut_mask(frame, seed)
        frames.append(mask)
        last_rect = _seed_rect(frame.shape[1], frame.shape[0])
    return frames


def _sample_frames(cap, start_s: float, end_s: float) -> List:
    """Raw resized BGR frames for one window — the torchvision backend
    batches these through the model instead of cutting each one separately
    the way GrabCut does."""
    return list(_iter_frames(cap, start_s, end_s))


# --- torchvision backend -----------------------------------------------------

def _load_torchvision_model():
    """DeepLabV3-ResNet101, COCO-with-VOC-labels weights already in the torch
    hub cache (nothing downloaded here), on the GPU when one is free.
    Returns (model, transform, device, use_fp16). Loaded once per matte
    computation by `_compute_torchvision`, freed again by
    `_free_torchvision_model` when it is done with every window."""
    import torch
    from torchvision.models.segmentation import (DeepLabV3_ResNet101_Weights,
                                                  deeplabv3_resnet101)

    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    model.to(device)
    return model, weights.transforms(), device, device.type == "cuda"


def _free_torchvision_model(model, device) -> None:
    """Drops the model and empties the CUDA cache — ComfyUI needs the card
    right after this pass runs, so the matte does not sit on VRAM it is
    finished with."""
    del model
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()


def _torchvision_masks(model, transform, device, use_fp16: bool, frames_bgr: List) -> List:
    """Soft (0-255 grayscale) person-probability masks for one window's
    frames, batched through the model in a single forward pass, temporally
    smoothed across the sequence and lightly feathered — the neural
    equivalent of GrabCut's own close+blur."""
    import cv2
    import numpy as np
    import torch

    rgb_tensors = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                   for f in frames_bgr]
    batch = torch.stack([transform(t) for t in rgb_tensors]).to(device)

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_fp16 else contextlib.nullcontext())
    with torch.no_grad(), autocast_ctx:
        logits = model(batch)["out"]
        probs = torch.softmax(logits.float(), dim=1)[:, TORCHVISION_PERSON_CLASS]

    masks: List = []
    smoothed = None
    for i, frame in enumerate(frames_bgr):
        h, w = frame.shape[:2]
        prob = probs[i].unsqueeze(0).unsqueeze(0)
        prob = torch.nn.functional.interpolate(prob, size=(h, w), mode="bilinear",
                                               align_corners=False)[0, 0]
        prob = prob.detach().to("cpu", dtype=torch.float32).numpy()
        smoothed = prob if smoothed is None else (
            TORCHVISION_TEMPORAL_ALPHA * prob + (1.0 - TORCHVISION_TEMPORAL_ALPHA) * smoothed)
        mask = np.clip(smoothed * 255.0, 0, 255).astype(np.uint8)
        mask = cv2.GaussianBlur(mask, TORCHVISION_FEATHER_KERNEL, 0)
        masks.append(mask)
    return masks


def _run_with_gpu_lease(tenant: str, vram_mb: float, fn):
    """Runs `fn()` while holding the GPU broker's lease for `tenant` — the
    same acquire-then-release-in-a-finally shape as
    comfyui_bridge/queue_manager.py and asr/faster_whisper_engine.py
    (runtime/gpu_broker.py). `compute_person_matte` is itself synchronous and
    is always run off the event loop thread in production (director.py's
    text-fx stage goes through asyncio.to_thread, precisely so this has a
    thread with no loop of its own to work with), so a private event loop
    here is safe. If something upstream ever calls in from a thread that
    already has a loop running, the lease is skipped rather than risking a
    deadlock or a cross-loop error — the matte still computes, just without
    the broker's coordination for that one run."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        return fn()

    from runtime.gpu_broker import gpu_broker

    async def _leased():
        await gpu_broker.acquire_lease(tenant, required_vram_mb=vram_mb)
        try:
            return fn()
        finally:
            await gpu_broker.release_lease(tenant)

    return asyncio.run(_leased())


def _compute_torchvision(cap, merged: Sequence[Tuple[float, float]]) -> Tuple[List[Dict[str, float]], List]:
    """Every window's masks, via one model load shared across all of them."""

    def _run():
        model, transform, device, use_fp16 = _load_torchvision_model()
        try:
            manifest: List[Dict[str, float]] = []
            all_frames: List = []
            offset = 0.0
            for start_s, end_s in merged:
                frames = _sample_frames(cap, start_s, end_s)
                if not frames:
                    continue
                masks = _torchvision_masks(model, transform, device, use_fp16, frames)
                manifest.append({"start_s": start_s, "end_s": end_s, "offset_s": offset})
                all_frames.extend(masks)
                offset += len(masks) / MATTE_FPS
            return manifest, all_frames
        finally:
            _free_torchvision_model(model, device)

    return _run_with_gpu_lease(TORCHVISION_TENANT, TORCHVISION_VRAM_MB, _run)


def _compute_grabcut(cap, merged: Sequence[Tuple[float, float]]) -> Tuple[List[Dict[str, float]], List]:
    manifest: List[Dict[str, float]] = []
    all_frames: List = []
    offset = 0.0
    for start_s, end_s in merged:
        window_frames = _sample_window(cap, start_s, end_s)
        if not window_frames:
            continue
        manifest.append({"start_s": start_s, "end_s": end_s, "offset_s": offset})
        all_frames.extend(window_frames)
        offset += len(window_frames) / MATTE_FPS
    return manifest, all_frames


def _encode(frames: List, width: int, height: int, path: Path) -> bool:
    """Grayscale frames -> a small mp4, piped straight into ffmpeg."""
    import cv2
    import numpy as np

    if not frames:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
        "-r", str(MATTE_FPS), "-i", "-",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=NO_WINDOW)
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)
        return proc.returncode == 0 and path.exists()
    except Exception as e:
        logger.warning("Matte encode failed: %s", e)
        return False


def compute_person_matte(
    source_video: str,
    windows: Sequence[Tuple[float, float]],
    project_dir: Path,
    canvas_w: int,
    canvas_h: int,
) -> Optional[MatteResult]:
    """A cached person matte covering exactly `windows`, or None when there is
    no backend available or nothing to compute."""
    backend = detect_backend()
    if backend == "off":
        logger.info("No person-segmentation backend available; behind-head/focus "
                    "matte effects are skipped this run")
        return None
    merged = _merge_windows(windows)
    if not merged or not source_video or not Path(source_video).exists():
        return None

    key = _cache_key(source_video, merged, backend, canvas_w, canvas_h)
    project_dir = Path(project_dir)
    video_path = project_dir / f"matte_{key}.mp4"
    manifest_path = project_dir / f"matte_{key}.json"
    if video_path.exists() and manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return MatteResult.from_dict(data)
        except Exception:
            pass  # fall through and recompute a corrupt/partial cache

    if backend not in ("grabcut", "torchvision"):
        # Probed above but not wired up: nothing installed in this environment
        # exercises this path today. Treated the same as "off" rather than
        # guessing at an unimplemented API.
        logger.warning("Matte backend %r detected but not implemented; degrading to off", backend)
        return None

    import cv2

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        logger.warning("Could not open %s for the person matte", source_video)
        return None

    result_backend = backend
    try:
        if backend == "torchvision":
            try:
                manifest, all_frames = _compute_torchvision(cap, merged)
            except Exception as e:
                logger.warning("torchvision matte failed (%s); falling back to GrabCut", e)
                cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                manifest, all_frames = _compute_grabcut(cap, merged)
                result_backend = "grabcut"
        else:
            manifest, all_frames = _compute_grabcut(cap, merged)
    except Exception as e:
        logger.warning("Person matte computation failed (%s backend): %s", backend, e)
        return None
    finally:
        cap.release()

    if not all_frames:
        return None
    width = max(2, int(round(all_frames[0].shape[1] / 2)) * 2)
    height = max(2, int(round(all_frames[0].shape[0] / 2)) * 2)
    if not _encode(all_frames, width, height, video_path):
        return None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    logger.info("Person matte (%s): %d frame(s) across %d window(s) -> %s",
               result_backend, len(all_frames), len(manifest), video_path)
    return MatteResult(str(video_path), result_backend, manifest)
