"""Generation-time rules measured on the user's RTX 5060 Ti (2026-09-26)."""
import asyncio

from presentation import assets
from runtime.gpu_broker import GPUBroker


def test_clip_length_follows_the_beat_and_stays_4n_plus_1():
    for seconds in (0.5, 2.0, 3.0, 4.0, 9.0):
        frames = assets._video_length_for(seconds, 16)
        assert (frames - 1) % 4 == 0
        assert assets.MIN_VIDEO_FRAMES <= frames <= assets.MAX_VIDEO_FRAMES
    assert assets._video_length_for(3.0, 16) < assets.DEFAULT_VIDEO_LENGTH  # no 5s clip for a 3s beat
    assert assets._video_length_for(3.0, 16) >= 3.0 * 16                   # still covers the beat


def test_images_generate_at_about_1_4_megapixels_keeping_aspect():
    w, h = assets._image_gen_size((1920, 1080))
    assert w * h <= assets.MAX_IMAGE_GEN_PIXELS
    assert abs(w / h - 1920 / 1080) < 0.03
    assert assets._image_gen_size((1024, 576)) == (1024, 576)  # small canvases untouched


def test_comfyui_lease_never_unloads_comfyuis_own_models(monkeypatch):
    broker = GPUBroker()
    freed = []
    monkeypatch.setattr(broker, "get_free_vram_mb", lambda: 1000.0)

    async def fake_release(*_a, **_k):
        freed.append(True)
        return True
    monkeypatch.setattr(broker, "release_comfyui_vram", fake_release)

    asyncio.run(broker.acquire_lease("comfyui", required_vram_mb=8000))
    assert freed == [], "ComfyUI's next job reuses the loaded models"
    asyncio.run(broker.acquire_lease("whisper", required_vram_mb=8000))
    assert freed == [True], "another tenant still gets ComfyUI to let go"
