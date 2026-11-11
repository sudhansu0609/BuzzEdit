"""M6 2.5D parallax (MOTION_GRAPHICS_PLAN.md): a few small, model-free checks.

Real renders (Depth Anything V2 Small + the Parallax Remotion composition) are
verified by inspecting rendered sample frames directly
(Samples/motion_gallery/m6_parallax_*.mp4), not re-run here -- these three
cover the pure selection/contrast logic and the "never touches a still it
should not" fallback path only.
"""

import numpy as np

from presentation import motion_parallax
from presentation.models import Asset, Beat, PresentationSettings


def test_depth_contrast_ok_rejects_a_flat_depth_map():
    flat = np.full((20, 20), 128.0, dtype=np.float32)
    varied = np.random.default_rng(0).normal(loc=3.0, scale=2.0, size=(20, 20)).astype(np.float32)
    assert motion_parallax.depth_contrast_ok(flat) is False
    assert motion_parallax.depth_contrast_ok(varied) is True


def test_select_is_seeded_and_capped_by_share():
    candidates = [Asset(beat_id=f"b{i}", kind="image", path=f"b{i}.png") for i in range(10)]
    chosen_a = motion_parallax._select(candidates, share=0.3, seed=7)
    chosen_b = motion_parallax._select(candidates, share=0.3, seed=7)
    assert len(chosen_a) == 3
    assert {a.beat_id for a in chosen_a} == {a.beat_id for a in chosen_b}  # same seed -> same picks


def test_apply_parallax_leaves_assets_untouched_when_share_is_zero():
    beats = [Beat(id="b0", start_s=0.0, end_s=5.0, kind="broll_image")]
    assets = [Asset(beat_id="b0", kind="image", path="nonexistent.png", source="generated")]
    settings = PresentationSettings(parallax_share=0.0)
    out_assets, stats = motion_parallax.apply_parallax(assets, beats, "/tmp/doesnotmatter",
                                                        1920, 1080, "horror", settings, seed=1)
    assert out_assets == assets
    assert stats["selected"] == 0
