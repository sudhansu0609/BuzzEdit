"""The torchvision person-matte backend: batching, temporal smoothing,
model lifecycle, GPU-broker leasing, and the GrabCut fallback on error.

Every test below except the one marked "smoke" substitutes a tiny fake model
for the real DeepLabV3-ResNet101 (no real inference) — real `torch` tensor
ops still run underneath it, only the heavyweight backbone is skipped.
"""

import subprocess
from pathlib import Path

import pytest

from presentation import matte as matte_mod


class _FakeModel:
    """A stand-in for DeepLabV3: same `{"out": logits}` contract, cheap."""

    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        import torch
        self.calls += 1
        n, _c, h, w = batch.shape
        out = torch.zeros((n, 21, h, w))
        # "Person" (channel 15) reads strongly in the top half of the frame,
        # weakly in the bottom half — enough contrast to check the mask is
        # not just uniform noise without needing a real network.
        out[:, matte_mod.TORCHVISION_PERSON_CLASS, : h // 2, :] = 8.0
        return {"out": out}


def _fake_backend(monkeypatch, model=None, load_calls=None, free_calls=None):
    """Wires `detect_backend`/`_load_torchvision_model`/`_free_torchvision_model`
    to a fake model, on the CPU, and returns the model used."""
    import torch
    fake_model = model or _FakeModel()
    load_calls = load_calls if load_calls is not None else []
    free_calls = free_calls if free_calls is not None else []

    monkeypatch.setattr(matte_mod, "detect_backend", lambda: "torchvision")

    def fake_load():
        load_calls.append(1)
        return fake_model, (lambda t: t.float()), torch.device("cpu"), False

    def fake_free(model, device):
        free_calls.append(1)

    monkeypatch.setattr(matte_mod, "_load_torchvision_model", fake_load)
    monkeypatch.setattr(matte_mod, "_free_torchvision_model", fake_free)
    return fake_model


def _real_clip(tmp_path) -> Path:
    from config import FFMPEG_BIN
    src = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=12:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0 or not src.exists():
        pytest.skip("ffmpeg unavailable")
    return src


# --- unit: batched masks -----------------------------------------------------

def test_torchvision_masks_batches_and_returns_soft_feathered_masks():
    import numpy as np
    model = _FakeModel()
    frames = [np.zeros((40, 60, 3), dtype=np.uint8) for _ in range(3)]

    masks = matte_mod._torchvision_masks(
        model, lambda t: t.float(), __import__("torch").device("cpu"), False, frames)

    assert model.calls == 1  # one batched forward pass, not one per frame
    assert len(masks) == 3
    for mask in masks:
        assert mask.shape == (40, 60)
        assert mask.dtype == np.uint8
        assert mask.min() >= 0 and mask.max() <= 255
    # The fake model reads "person" strongly in the top half only.
    assert masks[-1][:20, :].mean() > masks[-1][20:, :].mean()


def test_torchvision_masks_smooth_across_frames():
    """A single bright outlier frame is damped by its smoothed neighbours,
    not reproduced verbatim — the temporal smoothing is actually doing
    something, not a no-op alpha=1.0."""
    import numpy as np
    import torch

    class _FlickerModel:
        def __call__(self, batch):
            n, _c, h, w = batch.shape
            out = torch.zeros((n, 21, h, w))
            # Frame 0 dim, frame 1 a bright spike, frame 2 dim again.
            out[0, matte_mod.TORCHVISION_PERSON_CLASS] = 0.1
            out[1, matte_mod.TORCHVISION_PERSON_CLASS] = 8.0
            out[2, matte_mod.TORCHVISION_PERSON_CLASS] = 0.1
            return {"out": out}

    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    masks = matte_mod._torchvision_masks(
        _FlickerModel(), lambda t: t.float(), torch.device("cpu"), False, frames)
    # Un-smoothed, frame 1 would jump to ~max brightness outright; smoothed it
    # lands measurably below that ceiling.
    assert 0 < masks[1].mean() < 250


# --- the full compute_person_matte pipeline, fake model + a real clip -------

def test_torchvision_backend_computes_a_real_matte_with_a_fake_model(tmp_path, monkeypatch):
    src = _real_clip(tmp_path)
    load_calls: list = []
    free_calls: list = []
    _fake_backend(monkeypatch, load_calls=load_calls, free_calls=free_calls)

    result = matte_mod.compute_person_matte(str(src), [(1.0, 3.0)], tmp_path, 320, 240)

    assert result is not None
    assert result.backend == "torchvision"
    assert Path(result.path).exists()
    assert result.windows == [{"start_s": 1.0, "end_s": 3.0, "offset_s": 0.0}]
    # Loaded once for the whole matte (however many windows/frames), then freed.
    assert load_calls == [1]
    assert free_calls == [1]


def test_torchvision_error_falls_back_to_grabcut(tmp_path, monkeypatch):
    src = _real_clip(tmp_path)
    monkeypatch.setattr(matte_mod, "detect_backend", lambda: "torchvision")

    def _boom():
        raise RuntimeError("no CUDA context")

    monkeypatch.setattr(matte_mod, "_load_torchvision_model", _boom)

    result = matte_mod.compute_person_matte(str(src), [(1.0, 2.0)], tmp_path, 320, 240)
    assert result is not None
    assert result.backend == "grabcut"
    assert Path(result.path).exists()


# --- the GPU broker lease ----------------------------------------------------

def test_gpu_lease_acquired_and_released_around_the_work(monkeypatch):
    from runtime.gpu_broker import gpu_broker
    calls: list = []

    async def fake_acquire(tenant, required_vram_mb=0.0):
        calls.append(("acquire", tenant, required_vram_mb))

    async def fake_release(tenant):
        calls.append(("release", tenant))

    monkeypatch.setattr(gpu_broker, "acquire_lease", fake_acquire)
    monkeypatch.setattr(gpu_broker, "release_lease", fake_release)

    result = matte_mod._run_with_gpu_lease("unit_test_tenant", 111.0, lambda: "done")

    assert result == "done"
    assert calls == [("acquire", "unit_test_tenant", 111.0), ("release", "unit_test_tenant")]


def test_gpu_lease_skipped_best_effort_inside_a_running_loop(monkeypatch):
    """compute_person_matte is meant to run off the event loop thread
    (director.py's text-fx stage uses asyncio.to_thread precisely for this);
    if something ever calls in from a thread that already has a loop
    running, the work still happens, just without the broker's lease."""
    import asyncio
    from runtime.gpu_broker import gpu_broker
    calls: list = []

    async def fake_acquire(tenant, required_vram_mb=0.0):
        calls.append("acquire")

    async def fake_release(tenant):
        calls.append("release")

    monkeypatch.setattr(gpu_broker, "acquire_lease", fake_acquire)
    monkeypatch.setattr(gpu_broker, "release_lease", fake_release)

    async def _inside():
        return matte_mod._run_with_gpu_lease("unit_test_tenant", 111.0, lambda: "done")

    result = asyncio.run(_inside())
    assert result == "done"
    assert calls == []


# --- optional real-inference smoke test --------------------------------------

def test_real_torchvision_inference_smoke():
    """Real DeepLabV3-ResNet101 forward pass — skipped wherever CUDA or the
    cached weights are not available, so this never needs the network."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        model, transform, device, use_fp16 = matte_mod._load_torchvision_model()
    except Exception as e:
        pytest.skip(f"torchvision weights unavailable: {e}")
    try:
        import numpy as np
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        masks = matte_mod._torchvision_masks(model, transform, device, use_fp16, frames)
    finally:
        matte_mod._free_torchvision_model(model, device)
    assert len(masks) == 2
    assert masks[0].shape == (64, 64)
    assert masks[0].dtype.name == "uint8"
